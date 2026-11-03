// Module worker: one camera's simulated MJPG encode, off the UI thread (docs/WEB_TWIN.md §9 step 5).
// in:  {id, name, rgb: ArrayBuffer (packed rgb8, transferred), width, height, qY?, qC?, quality?, channels?}
// out: {id, name, jpeg: ArrayBuffer (transferred), ms}   or   {id, name, error}
import { encodeJPEG } from './jpeg_encoder.js';

self.onmessage = (e) => {
  const { id, name, rgb, width, height, qY, qC, quality, channels } = e.data;
  try {
    const t0 = performance.now();
    const jpeg = encodeJPEG(new Uint8Array(rgb), width, height, { qY, qC, quality, channels });
    const ms = performance.now() - t0;
    self.postMessage({ id, name, jpeg: jpeg.buffer, ms }, [jpeg.buffer]);   // encodeJPEG returns a standalone buffer
  } catch (err) {
    self.postMessage({ id, name, error: String(err?.stack ?? err) });
  }
};
