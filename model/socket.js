// The page's client for the model socket, /ws/twin (docs/WEB_TWIN.md §7). Browser and node (WebSocket is global
// in both; tests inject a fake through `WebSocketImpl`).
//
// Page -> server, per rendered bundle: a text truth message {"type": "truth", "frame_id", ...} FIRST, then the binary
// bundle (model/bundle.js) with the same frame_id. The truth stays on the server; only the bundle reaches the model.
// Latest-only: while the previous frame is still in the socket's send buffer (bufferedAmount > 0) a new frame is
// dropped and counted (droppedPage), never queued, so the model always gets the newest images. Recording and
// scripted runs pass mustDeliver: every frame is sent, and the returned promise resolves once the send buffer has
// drained below DRAIN_BYTES, which is the back-pressure the caller waits on instead of dropping.
//
// Server -> page: JSON text. Each message is dispatched as CustomEvent(msg.type, {detail: msg}) ("status", "score",
// "record_started", "record_stopped", "error", ...), and also as a generic "message" event.

import { packBundle } from './bundle.js';

const DRAIN_BYTES = 1 << 20;     // mustDeliver resolves once less than this is still buffered
const DRAIN_POLL_MS = 4;
const DRAIN_TIMEOUT_MS = 10000;  // a stalled socket does not hang the caller forever

function defaultUrl() {
  const loc = globalThis.location;
  if (!loc || !loc.host) throw new Error('TwinSocket: no url given and no page location to derive it from');
  return `${loc.protocol === 'https:' ? 'wss' : 'ws'}://${loc.host}/ws/twin`;
}

export class TwinSocket extends EventTarget {
  /**
   * @param {{url?: string, hello?: () => object, WebSocketImpl?: any, autoConnect?: boolean,
   *          minBackoffMs?: number, maxBackoffMs?: number}} opts
   *   hello() supplies the `calibration` and `scene_ref` sent on every (re)connect.
   */
  constructor({ url, hello = () => ({}), WebSocketImpl, autoConnect = true, minBackoffMs = 500, maxBackoffMs = 5000 } = {}) {
    super();
    this.url = url ?? defaultUrl();
    this.hello = hello;
    this.WS = WebSocketImpl ?? globalThis.WebSocket;
    this.minBackoffMs = minBackoffMs;
    this.maxBackoffMs = maxBackoffMs;
    this.backoffMs = minBackoffMs;
    this.state = 'offline';
    this.ws = null;
    this.sent = 0;            // bundles handed to the socket
    this.droppedPage = 0;     // bundles dropped here because the previous one had not flushed (latest-only)
    this.notOpen = 0;         // bundles offered while the socket was not open
    this.lastStatus = null;
    this.lastScore = null;
    this._timer = null;
    this._closed = false;
    if (autoConnect) this.connect();
  }

  _setState(s) {
    if (s === this.state) return;
    this.state = s;
    this.dispatchEvent(new CustomEvent('state', { detail: { state: s } }));
  }

  connect() {
    this._closed = false;
    if (this.ws && (this.state === 'open' || this.state === 'connecting')) return;
    clearTimeout(this._timer);
    this._timer = null;
    let ws;
    try {
      ws = new this.WS(this.url);
    } catch (e) {
      this._scheduleReconnect();
      return;
    }
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    this._setState('connecting');
    ws.onopen = () => {
      if (ws !== this.ws) return;
      this.backoffMs = this.minBackoffMs;
      this._setState('open');
      try {
        ws.send(JSON.stringify({ type: 'hello', role: 'page', ...this.hello() }));
      } catch (e) {
        this.dispatchEvent(new CustomEvent('error', { detail: { type: 'error', where: 'hello', message: String(e) } }));
      }
    };
    ws.onmessage = (ev) => this._onMessage(ev.data);
    ws.onclose = () => {
      if (ws !== this.ws) return;
      this.ws = null;
      this._setState('offline');
      this._scheduleReconnect();
    };
    ws.onerror = () => { /* onclose follows and reconnects */ };
  }

  _scheduleReconnect() {
    if (this._closed || this._timer) return;
    const wait = this.backoffMs;
    this.backoffMs = Math.min(this.maxBackoffMs, this.backoffMs * 2);
    this._timer = setTimeout(() => { this._timer = null; this.connect(); }, wait);
  }

  /** Stop and do not reconnect. */
  close() {
    this._closed = true;
    clearTimeout(this._timer);
    this._timer = null;
    const ws = this.ws;
    this.ws = null;
    if (ws) { try { ws.close(); } catch { /* already closed */ } }
    this._setState('offline');
  }

  _onMessage(data) {
    if (typeof data !== 'string') return;           // the server sends the page JSON text only
    let msg;
    try { msg = JSON.parse(data); } catch { return; }
    if (!msg || typeof msg.type !== 'string') return;
    if (msg.type === 'status') this.lastStatus = msg;
    else if (msg.type === 'score') this.lastScore = msg;
    this.dispatchEvent(new CustomEvent(msg.type, { detail: msg }));
    this.dispatchEvent(new CustomEvent('message', { detail: msg }));
  }

  /**
   * Send one frame: the truth text message, then the binary bundle.
   * @param {object} truthMsg   the rest of the truth message (frame_id is taken from header.frame_id)
   * @param {object} header     §7.2 header ({type: "frames", frame_id, source, cameras: [...]})
   * @param {Uint8Array[]} jpegs
   * @param {{mustDeliver?: boolean}} opts
   * @returns {boolean | Promise<boolean>} without mustDeliver: whether the frame was sent (false = dropped);
   *   with mustDeliver: a promise of the same, resolved once the send buffer has drained below 1 MB.
   */
  sendFrame(truthMsg, header, jpegs, { mustDeliver = false } = {}) {
    const ws = this.ws;
    if (!ws || this.state !== 'open') {
      this.notOpen++;
      return mustDeliver ? Promise.resolve(false) : false;
    }
    if (!mustDeliver && ws.bufferedAmount > 0) {
      this.droppedPage++;
      return false;
    }
    const frame_id = header.frame_id;
    if (truthMsg && truthMsg.frame_id !== undefined && truthMsg.frame_id !== frame_id) {
      throw new RangeError(`sendFrame: truth frame_id ${truthMsg.frame_id} != bundle frame_id ${frame_id}`);
    }
    const bundle = packBundle(header, jpegs);
    ws.send(JSON.stringify({ type: 'truth', frame_id, ...truthMsg }));
    ws.send(bundle);
    this.sent++;
    if (!mustDeliver) return true;
    return this._drained(ws).then(() => true);
  }

  _drained(ws) {
    return new Promise((resolve) => {
      const t0 = Date.now();
      const check = () => {
        if (ws !== this.ws || ws.bufferedAmount < DRAIN_BYTES || Date.now() - t0 > DRAIN_TIMEOUT_MS) resolve();
        else setTimeout(check, DRAIN_POLL_MS);
      };
      check();
    });
  }

  /** Send {type, ...body} and resolve with the next `replyType` message (e.g. record_start -> record_started). */
  request(type, body = {}, replyType, timeoutMs = 5000) {
    return new Promise((resolve, reject) => {
      const ws = this.ws;
      if (!ws || this.state !== 'open') { reject(new Error(`request ${type}: socket is ${this.state}`)); return; }
      let timer = null;
      const onReply = (ev) => {
        clearTimeout(timer);
        this.removeEventListener('error', onError);
        resolve(ev.detail);
      };
      const onError = (ev) => {
        if (ev.detail?.request !== type) return;     // only errors the server ties to this request
        clearTimeout(timer);
        this.removeEventListener(replyType, onReply);
        this.removeEventListener('error', onError);
        reject(new Error(ev.detail.message ?? `request ${type} failed`));
      };
      timer = setTimeout(() => {
        this.removeEventListener(replyType, onReply);
        this.removeEventListener('error', onError);
        reject(new Error(`request ${type}: no ${replyType} within ${timeoutMs} ms`));
      }, timeoutMs);
      this.addEventListener(replyType, onReply, { once: true });
      this.addEventListener('error', onError);
      ws.send(JSON.stringify({ type, ...body }));
    });
  }

  stats() {
    return {
      state: this.state, sent: this.sent, droppedPage: this.droppedPage, notOpen: this.notOpen,
      bufferedAmount: this.ws?.bufferedAmount ?? 0, lastStatus: this.lastStatus,
    };
  }
}
