// GPU-side display of camera images (docs/WEB_TWIN.md §18.2): blit a renderer's 8-bit sensor texture (rgba8unorm,
// camera RGB, already tone-mapped) to a WebGPU canvas on the renderer's own device. No CPU readback for display.
//
// One fullscreen triangle per canvas. The image is letterboxed ('contain') on near-black inside the canvas; the rest
// of the frame is that near-black. Magnified images are filtered with a 9-tap Catmull-Rom (bicubic, from bilinear
// taps), minified ones with 4 bilinear taps a quarter of an output pixel apart (a cheap box), so the grid tiles do not
// alias and the scaled-up main view stays sharp. The canvas format is the platform's preferred one (bgra8unorm on
// macOS); values are written as they are (the sensor already encoded them), so there is no second gamma.
import { letterbox } from './views.js';

export const NEAR_BLACK = Object.freeze([12, 13, 12]);       // CSS rgb(12, 13, 12): the frame behind every image

const WGSL = /* wgsl */`
struct U {
  rect : vec4f,     // x, y, w, h of the image in canvas px
  bg   : vec4f,     // near-black, 0..1
  src  : vec4f,     // texture w, h, 1/w, 1/h
  mode : vec4f,     // x: 0 bilinear, 1 bicubic, 2 minify (4 taps); y: output px in source px
};
@group(0) @binding(0) var tex : texture_2d<f32>;
@group(0) @binding(1) var smp : sampler;
@group(0) @binding(2) var<uniform> u : U;

@vertex fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4f {
  var p = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], 0.0, 1.0);
}

fn tap(uv : vec2f) -> vec3f { return textureSampleLevel(tex, smp, uv, 0.0).rgb; }

fn catmullRom(uv : vec2f) -> vec3f {
  let size = u.src.xy;
  let sp = uv * size;
  let t1 = floor(sp - 0.5) + 0.5;
  let f = sp - t1;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let o12 = w2 / w12;
  let p0 = (t1 - 1.0) * u.src.zw;
  let p3 = (t1 + 2.0) * u.src.zw;
  let p12 = (t1 + o12) * u.src.zw;
  var c = vec3f(0.0);
  c += tap(vec2f(p0.x, p0.y)) * w0.x * w0.y;
  c += tap(vec2f(p12.x, p0.y)) * w12.x * w0.y;
  c += tap(vec2f(p3.x, p0.y)) * w3.x * w0.y;
  c += tap(vec2f(p0.x, p12.y)) * w0.x * w12.y;
  c += tap(vec2f(p12.x, p12.y)) * w12.x * w12.y;
  c += tap(vec2f(p3.x, p12.y)) * w3.x * w12.y;
  c += tap(vec2f(p0.x, p3.y)) * w0.x * w3.y;
  c += tap(vec2f(p12.x, p3.y)) * w12.x * w3.y;
  c += tap(vec2f(p3.x, p3.y)) * w3.x * w3.y;
  return c;
}

@fragment fn fs(@builtin(position) fp : vec4f) -> @location(0) vec4f {
  let q = (fp.xy - u.rect.xy) / u.rect.zw;
  if (q.x < 0.0 || q.y < 0.0 || q.x >= 1.0 || q.y >= 1.0) { return vec4f(u.bg.rgb, 1.0); }
  var c : vec3f;
  if (u.mode.x > 1.5) {
    let d = 0.25 * u.mode.y * u.src.zw;          // a quarter of an output pixel, in uv
    c = 0.25 * (tap(q + vec2f(-d.x, -d.y)) + tap(q + vec2f(d.x, -d.y)) + tap(q + vec2f(-d.x, d.y)) + tap(q + vec2f(d.x, d.y)));
  } else if (u.mode.x > 0.5) {
    c = catmullRom(q);
  } else {
    c = tap(q);
  }
  return vec4f(clamp(c, vec3f(0.0), vec3f(1.0)), 1.0);
}
`;

const U_BYTES = 64;

export class Presenter {
  /** device: the renderer's GPUDevice (the textures live there). */
  constructor(device, { background = NEAR_BLACK } = {}) {
    this.device = device;
    this.bg = background.map((x) => x / 255);
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.module = device.createShaderModule({ code: WGSL, label: 'present' });
    this.pipe = device.createRenderPipeline({
      label: 'present', layout: 'auto',
      vertex: { module: this.module, entryPoint: 'vs' },
      fragment: { module: this.module, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    this.slots = new WeakMap();   // canvas -> {ctx, ub, groups: WeakMap(texture -> bind group)}
    this.count = 0;
  }

  _slot(canvas) {
    let s = this.slots.get(canvas);
    if (!s) {
      const ctx = canvas.getContext('webgpu');
      ctx.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });
      s = { ctx, ub: this.device.createBuffer({ size: U_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'present.u' }), groups: new WeakMap() };
      this.slots.set(canvas, s);
    }
    return s;
  }

  /**
   * Draw `texture` (rgba8unorm, TEXTURE_BINDING) letterboxed into `canvas` (its backing size), in one submit.
   * Returns the image rectangle in canvas px ({x, y, w, h}) for overlays, or null when nothing was drawn.
   */
  present(texture, canvas) {
    if (!texture || !canvas || !canvas.width || !canvas.height) return null;
    const s = this._slot(canvas);
    const W = canvas.width, H = canvas.height, tw = texture.width, th = texture.height;
    const rect = letterbox(tw, th, W, H);
    const k = rect.w / tw;                                   // output px per source px
    const mode = k > 1.05 ? 1 : k < 0.95 ? 2 : 0;
    this.device.queue.writeBuffer(s.ub, 0, new Float32Array([rect.x, rect.y, rect.w, rect.h, ...this.bg, 1, tw, th, 1 / tw, 1 / th, mode, 1 / k, 0, 0]));
    let g = s.groups.get(texture);
    if (!g) {
      g = this.device.createBindGroup({ layout: this.pipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: texture.createView() }, { binding: 1, resource: this.sampler }, { binding: 2, resource: { buffer: s.ub } }] });
      s.groups.set(texture, g);
    }
    const enc = this.device.createCommandEncoder({ label: 'present' });
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: s.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: this.bg[0], g: this.bg[1], b: this.bg[2], a: 1 } }] });
    rp.setPipeline(this.pipe); rp.setBindGroup(0, g); rp.draw(3); rp.end();
    this.device.queue.submit([enc.finish()]);
    this.count++;
    return rect;
  }

  /** Fill a canvas with the near-black frame (a view with no image yet). */
  clear(canvas) {
    if (!canvas?.width) return;
    const s = this._slot(canvas);
    const enc = this.device.createCommandEncoder({ label: 'present.clear' });
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: s.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: this.bg[0], g: this.bg[1], b: this.bg[2], a: 1 } }] });
    rp.end();
    this.device.queue.submit([enc.finish()]);
  }
}
