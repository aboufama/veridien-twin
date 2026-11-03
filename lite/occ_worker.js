// Module worker: builds LiteRenderer's occlusion BVHs (occ_build.js) off the main thread and transfers them back.
import { build } from './occ_build.js';

self.onmessage = (e) => {
  const { out, transfer } = build(e.data);
  self.postMessage(out, transfer);
};
