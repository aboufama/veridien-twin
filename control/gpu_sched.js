// The GPU calendar of the live page (docs/WEB_TWIN.md §12, §14, §17): which GPU job goes next, when, and which
// captures are skipped. Pure policy and bookkeeping, no GPU and no DOM: the page passes its clock, its timers and
// its job runners in, and web/twin/tests/control_sched.test.mjs drives this same class against a simulated GPU.
//
// Jobs.
//   camera  one camera of one bundle (setPoses, visible, trace, sensor, readback copy), issued at that camera's own
//           exposure end. A bundle is admitted whole or skipped whole: once admitted every camera renders (§14:
//           partial bundles are never sent). The same camera is never outstanding twice (its AE reads the previous
//           frame's statistics, as UVC AE does).
//   aid     the overview or the scope view: slack only (never ahead of a released or soon-due last camera), or one
//           slot ahead of a bundle's first cameras when it is overdue (aidMaxMs, the 10 fps floor of §17).
//   side    tiny jobs (the renderer's loop-cap flags): slack, or after sideWaitMs.
//   exclusive  judge renders and the like: the queue drains, the job runs alone, then the calendar resumes.
//
// Queue. At most `depth` jobs outstanding and at most `maxQueuedMs` of predicted GPU work: the next job is queued
// before the current one ends, so the GPU never waits on a JavaScript round trip between jobs, and nothing piles
// up. Every submit stays <= 8 ms of GPU time (the renderer's tiling, §12 rule 3).
//
// §12 rule 3. Back-to-back work never runs longer than `runMs` without a `gapMs` idle gap. A new bundle that would
// carry the run past runMs waits: the queue drains and stays idle gapMs (so bundles stay whole). Any natural idle
// of gapMs restarts the run. Nothing at all is issued past hardRunMs.
//
// Admission (live). Of the bundles whose first camera has been released and that are not yet admitted, take the
// oldest one that is at most catchMs late (late = predicted finish minus its finish on an idle GPU; no skip),
// otherwise the newest. Admit it at once when it is at most lateMs late; otherwise as soon as only one job is left
// on the GPU (so its first camera is queued behind it and the GPU is never idle while a capture waits). Skipped
// bundles are counted with the reason that held them when their first camera was released.

export const SCHED_DEFAULTS = Object.freeze({
  depth: 2,           // GPU jobs outstanding at once
  maxQueuedMs: 80,    // predicted GPU ms queued at once (a second job is only queued while this holds)
  gapMs: 50,          // §12 rule 3: the idle gap ...
  runMs: 1800,        // ... taken before back-to-back work would pass this (the rule's bound is 2000 ms)
  hardRunMs: 1950,    // nothing is issued past this, not even the rest of a bundle
  lateMs: 25,         // admit while the GPU is busy when the bundle is predicted at most this late
  conserveMs: 4,      // otherwise admit once one job is left on the GPU (conserveOne), or the queue drains within this
  conserveOne: 0,     // (measured: no throughput gain, +25 ms latency; left as an option for the A/B judges)
  catchMs: 60,        // an older released bundle is preferred (no skip) while at most this late
  aidMinMs: 83,       // aids at most 12 fps ...
  aidMaxMs: 95,       // ... and at least 10 fps (§17): an aid that would pass this before the next chance goes now
  aidMarginMs: 1.5,   // slack an aid must leave before the next camera job
  sideWaitMs: 1000,   // a side job waits at most this long for slack
  stallMs: 1000,      // a job this far past its predicted end is counted as a stall
});

/** Service time (ms) per job key: the 75th percentile of its last 8 samples (robust to a first-use shader compile,
 * quick to follow a real change), falling back to the kind's, then to defaults. q = 0.5: the median, used for
 * predicting when the queue drains (an over-prediction there idles the GPU). */
export class CostModel {
  constructor(init = {}, n = 8) {
    this.m = new Map();
    this.n = n;
    this.init = { camera: 20, aid: 5, side: 0.5, ...init };
  }

  _est(k, q = 0.75) {
    const v = this.m.get(k);
    if (!v || !v.length) return null;
    const s = [...v].sort((a, b) => a - b);
    return s[Math.min(s.length - 1, Math.floor(q * s.length))];
  }

  get(key, kind, q = 0.75) { return this._est(key, q) ?? this._est(`kind:${kind}`, q) ?? this.init[kind] ?? 10; }

  observe(key, kind, ms) {
    if (!Number.isFinite(ms) || ms < 0) return;
    for (const k of [key, `kind:${kind}`]) {
      let v = this.m.get(k);
      if (!v) { v = []; this.m.set(k, v); }
      v.push(ms);
      if (v.length > this.n) v.shift();
    }
  }

  snapshot() { return Object.fromEntries([...this.m.keys()].filter((k) => !k.startsWith('kind:')).map((k) => [k, Math.round(this._est(k) * 100) / 100])); }
}

const COUNTED = (reason) => !String(reason).startsWith('app:') || reason === 'app:inflight';

export class GpuScheduler {
  /**
   * @param {{now: () => number, setTimer: (fn: () => void, ms: number) => (() => void), source?: object,
   *          aid?: (now: number) => ({key, since, run: () => Promise, minMs?, maxMs?, onService?: (ms) => void} | null),
   *          log?: Function}} cfg
   * source (live bundles): {ready() -> true | reason, newest(now) -> id (-1: none released yet),
   *   describe(id) -> {jobs: [{key, cam, release}]} | null, admit(id) -> [{key, cam, release, issue}] | null}.
   *   issue() submits the job's GPU work synchronously and returns a promise that settles when the GPU is done.
   */
  constructor({ now, setTimer, source = null, aid = null, log = () => {}, costInit = {}, ...opts } = {}) {
    this.o = { ...SCHED_DEFAULTS, ...opts };
    this.now = now;
    this.setTimer = setTimer;
    this.source = source;
    this.aid = aid;
    this.log = log;
    this.costs = new CostModel(costInit);
    this.out = [];                // outstanding jobs, in issue (= GPU) order
    this.lastDone = -Infinity;    // completion time of the last finished job
    this.drainAt = -Infinity;     // when the queue last ran dry
    this.runStart = null;         // start of the current back-to-back run (§12)
    this.gapUntil = -Infinity;
    this.gapWanted = false;
    this.cur = null;              // the admitted bundle whose jobs are being issued: {id, jobs, next}
    this.batch = [];              // batch bundles waiting for admission (runScript): {id, jobs, resolve}
    this.sideQ = [];
    this.excl = [];
    this.exclRunning = false;
    this.lastAdmitted = -1;
    this.overloadT = -Infinity;     // the last capture skipped because the GPU was busy
    this.noted = -1;
    this.reasons = [];            // [{from, to, reason}]: why bundles released while not admitted were held
    this.lastReason = null;
    this.timer = null;
    this.timerAt = Infinity;
    this._in = false;
    this._again = false;
    this.stats = {
      issued: 0, admitted: 0, skipped: 0, skippedBy: {}, gaps: 0, gapIdleMs: 0, longestRunMs: 0, aids: 0, sides: 0,
      exclusive: 0, stalls: 0, errors: 0, depthHist: [0, 0, 0, 0, 0], maxDepth: 0, lateAdmits: 0, catchUps: 0,
    };
  }

  // ---------------------------------------------------------------- public
  setSource(src) { this.source = src; this.pump(); }

  /** Forget the live bundle numbering (after a scripted run, or when the clock restarts). */
  resetLive(lastId = -1) { this.lastAdmitted = lastId; this.noted = lastId; this.reasons = []; }

  /** Batch (runScript): admit this bundle as soon as the calendar can take it; resolves when admitted. */
  admitBatch(id, jobs) {
    return new Promise((resolve) => { this.batch.push({ id, jobs, resolve }); this.pump(); });
  }

  /** A small job, run when a slot is free and it fits the slack (or after sideWaitMs). */
  side(key, run) {
    return new Promise((resolve, reject) => { this.sideQ.push({ key, run, resolve, reject, t: this.now() }); this.pump(); });
  }

  /** Run fn with the GPU queue drained and nothing else issued until it settles. */
  exclusive(fn) {
    return new Promise((resolve, reject) => { this.excl.push({ fn, resolve, reject }); this.pump(); });
  }

  /** Predicted GPU ms still queued. */
  queuedMs(now = this.now()) { return Math.max(0, this._queueFree(now) - now); }

  get busy() { return this.out.length > 0 || !!this.cur || this.batch.length > 0 || this.exclRunning; }

  pump() {
    if (this._in) { this._again = true; return; }
    this._in = true;
    try {
      let n = 0;
      do { this._again = false; this._step(); } while (this._again && ++n < 8);
    } finally { this._in = false; }
  }

  // ---------------------------------------------------------------- the step
  _step() {
    const o = this.o, now = this.now();
    if (this.runStart != null && !this.out.length && !this.exclRunning && now - this.drainAt >= o.gapMs) this.runStart = null;
    this._noteReleases(now);
    for (const j of this.out) if (!j.stalled && now - j.tEnd > o.stallMs) { j.stalled = true; this.stats.stalls++; this.log(`GPU job ${j.key} ${Math.round(now - j.tEnd)} ms past its predicted end`); }
    if (this.exclRunning) return;
    if (this.excl.length && !this.cur) {
      if (!this.out.length) this._runExclusive(now);
      return;
    }
    if (now < this.gapUntil) { this._arm(this.gapUntil); return; }
    if (this.runStart != null && now - this.runStart > o.hardRunMs) this.gapWanted = true;
    if (this.gapWanted) {
      if (this.out.length) return;                                    // drain, then idle gapMs
      this.gapWanted = false;
      this.gapUntil = Math.max(now, this.drainAt) + o.gapMs;
      this.stats.gaps++;
      this.stats.gapIdleMs += this.gapUntil - Math.max(now, this.drainAt);
      this._arm(this.gapUntil);
      return;
    }
    for (let guard = 0; guard < 8 && this.out.length < o.depth; guard++) {
      if (!this.cur && !this.excl.length) this._admit(now);
      if (this.gapWanted) { this._again = true; return; }
      const job = this.cur ? this.cur.jobs[this.cur.next] : null;
      const qFree = this._queueFree(now);
      const camFree = job ? !this.out.some((j) => j.cam != null && j.cam === job.cam) : false;
      if (job && job.release <= now + 0.2 && camFree) {
        // an aid (or a side job) that would be overdue before the next chance goes ahead of this camera, unless this
        // is a bundle's last camera (in sustained overload, a last camera that is late anyway yields too); the next
        // chance is the following job when it can be issued at once, else this job's completion
        const overloaded = now - this.overloadT < 1000;
        const yields = (j) => !j.last || (overloaded && Math.max(qFree, now) - j.release > o.lateMs);
        if (yields(job) && !this.excl.length) {
          // the next decision comes when the job at the head of the queue completes (or this one, on an empty queue)
          const after = this.cur.jobs[this.cur.next + 1];
          const head = this.out.length ? this._headEnd(now) - now : this.costs.get(job.key, 'camera', 0.5);
          const horizon = after && yields(after) && after.release <= now + 0.2 ? 0 : Math.max(0.5, head);
          if (horizon > 0 && this._issueAid(now, -Infinity, horizon)) continue;
          if (this._issueSide(now, -Infinity, true)) continue;
        }
        if (this.out.length && qFree - now + this.costs.get(job.key, 'camera') > o.maxQueuedMs) break;
        if (++this.cur.next >= this.cur.jobs.length) this.cur = null;
        this._issue({ ...job, kind: 'camera' }, now);
        continue;
      }
      if (this.excl.length) break;                                   // draining for an exclusive job
      // slack before the next camera job can start: an aid or a side job that fits
      const nextStart = job ? Math.max(job.release, camFree ? -Infinity : qFree) : this._nextFirstRelease(now);
      const slack = nextStart - qFree;
      const horizon = job ? Math.max(0, job.release - now) : Math.max(0, qFree - now - o.conserveMs);
      if (this._issueAid(now, slack, !job || !job.last ? horizon : null)) continue;
      if (this._issueSide(now, slack, false)) continue;
      if (job && job.release > now) this._arm(job.release);
      else if (!job && nextStart > now && Number.isFinite(nextStart)) this._arm(nextStart);
      break;
    }
  }

  // ---------------------------------------------------------------- admission
  _admit(now) {
    const o = this.o;
    if (this.batch.length) {                                          // scripted runs: in order, never skipped
      const b = this.batch[0];
      if (!this._runFits(now, b.jobs)) { this.gapWanted = true; return; }
      this.batch.shift();
      this._take(b.id, b.jobs);
      b.resolve();
      return;
    }
    const src = this.source;
    if (!src) return;
    const ready = src.ready();
    if (ready !== true) return;
    const newest = src.newest(now);
    if (newest < 0 || newest <= this.lastAdmitted) return;
    // catching up (no skip) only pays with headroom; in sustained overload a capture is skipped either way, so the
    // newest is taken (lower latency, the same throughput)
    const overloaded = now - this.overloadT < 1000;
    const oldest = this.lastAdmitted < 0 || overloaded ? newest : Math.max(this.lastAdmitted + 1, newest - 3);
    let pick = -1, desc = null, late = 0;
    for (let id = oldest; id <= newest; id++) {
      const d = src.describe(id);
      if (!d || !d.jobs.length) continue;
      const L = this._lateness(now, d.jobs);
      if (L <= o.catchMs || id === newest) { pick = id; desc = d; late = L; break; }
    }
    if (!desc) { this.lastAdmitted = newest; return; }
    if (!(late <= o.lateMs || (o.conserveOne && this.out.length <= 1) || this._queueFree(now) - now <= o.conserveMs)) {   // held: the GPU is busy
      if (!o.conserveOne) this._arm(this._queueFree(now) - o.conserveMs);
      return;
    }
    if (!this._runFits(now, desc.jobs)) { this.gapWanted = true; return; }
    const jobs = src.admit(pick);
    for (let id = this.lastAdmitted + 1; id < pick; id++) if (this.lastAdmitted >= 0) this._skip(id);
    if (pick < newest) this.stats.catchUps++;
    if (late > o.lateMs) this.stats.lateAdmits++;
    this.lastAdmitted = pick;
    this.reasons = this.reasons.filter((r) => r.to > pick);
    if (jobs && jobs.length) this._take(pick, jobs);
  }

  _take(id, jobs) {
    jobs[jobs.length - 1].last = true;
    this.cur = { id, jobs, next: 0 };
    this.stats.admitted++;
  }

  _skip(id) {
    const r = this.reasons.find((x) => x.from <= id && id <= x.to)?.reason ?? 'gpu';
    if (r === 'gpu') this.overloadT = this.now();
    this.stats.skippedBy[r] = (this.stats.skippedBy[r] ?? 0) + 1;
    if (COUNTED(r)) this.stats.skipped++;
  }

  /** Why a bundle released now would not be admitted at once (for the skip reasons). */
  _blockReason(now) {
    if (this.exclRunning || this.excl.length) return 'exclusive';
    if (now < this.gapUntil || this.gapWanted) return 'gap';
    if (this.cur || this.out.length >= this.o.depth) return 'gpu';        // the root cause when the GPU is busy too
    const r = this.source?.ready?.();
    if (r !== true) return `app:${r ?? 'off'}`;
    return null;
  }

  _noteReleases(now) {
    const src = this.source;
    if (!src) return;
    const newest = src.newest(now);
    if (newest <= this.noted) return;
    // every bundle released since the last look gets the reason that holds now; when nothing holds now, the previous
    // look's page-side reason (a hidden tab's timers run at 1 Hz, so captures pile up unseen until it shows again)
    const now0 = this._blockReason(now), prev = this.lastReason;
    this.lastReason = now0;
    const reason = now0 ?? (prev && prev !== 'gpu' ? prev : null), from = Math.max(this.noted + 1, this.lastAdmitted + 1);
    const last = this.reasons.at(-1);
    if (reason && from <= newest) {
      if (last && last.reason === reason && last.to === from - 1) last.to = newest;
      else this.reasons.push({ from, to: newest, reason });
      if (this.reasons.length > 256) this.reasons.splice(0, this.reasons.length - 256);
    }
    this.noted = newest;
  }

  /** Predicted finish of these jobs after the queue, minus their finish on an idle GPU (ms). */
  _lateness(now, jobs) {
    const f = this._finish(now, jobs);
    let ti = -Infinity;
    for (const j of jobs) ti = Math.max(ti, j.release) + this.costs.get(j.key, 'camera');
    return f - ti;
  }

  _finish(now, jobs) {
    let t = this._queueFree(now);
    for (const j of jobs) t = Math.max(t, j.release) + this.costs.get(j.key, 'camera');
    return t;
  }

  /** Whether these jobs end within the §12 run (or a natural gap comes first). */
  _runFits(now, jobs) {
    if (this.runStart == null) return true;
    const qFree = this._queueFree(now);
    const start = Math.max(qFree, now, jobs[0]?.release ?? -Infinity);
    if (start - qFree >= this.o.gapMs) return true;
    return this._finish(now, jobs) - this.runStart <= this.o.runMs;
  }

  _nextFirstRelease(now) {
    const src = this.source;
    if (!src || src.ready() !== true) return Infinity;
    const n = src.newest(now);
    if (n > this.lastAdmitted) return now;                            // a released bundle is waiting
    const d = src.describe(n + 1);
    return d?.jobs?.length ? d.jobs[0].release : Infinity;
  }

  // ---------------------------------------------------------------- aids, side jobs, exclusive
  /** horizon: ms until the next chance to issue an aid (null: overdue aids not allowed here). */
  _issueAid(now, slack, horizon) {
    const a = this.aid?.(now);
    if (!a) return false;
    const o = this.o, cost = this.costs.get(a.key, 'aid');
    const minMs = a.minMs ?? o.aidMinMs, maxMs = a.maxMs ?? o.aidMaxMs;      // an aid may bring its own cadence
    const fits = a.since >= minMs && cost + o.aidMarginMs <= slack;
    const overdue = horizon != null && a.since >= minMs / 2 && (a.since >= maxMs || a.since + horizon > maxMs);
    if (!fits && !overdue) return false;
    if (this.out.length && this.queuedMs(now) + cost > this.o.maxQueuedMs && !overdue) return false;
    this._issue({ kind: 'aid', key: a.key, cam: null, release: now, issue: a.run, onService: a.onService }, now);
    this.stats.aids++;
    return true;
  }

  /** ahead: allowed ahead of a bundle's first cameras (a side job is a fraction of a ms) once it has waited 50 ms. */
  _issueSide(now, slack, ahead) {
    const s = this.sideQ[0];
    if (!s) return false;
    const cost = this.costs.get(s.key, 'side');
    if (!(cost + this.o.aidMarginMs <= slack || (ahead && now - s.t >= 50) || now - s.t >= this.o.sideWaitMs)) return false;
    this.sideQ.shift();
    let res;
    this._issue({ kind: 'side', key: s.key, cam: null, release: now, issue: () => (res = (async () => s.run())()) }, now);
    res.then(s.resolve, s.reject);
    this.stats.sides++;
    return true;
  }

  _runExclusive(now) {
    const e = this.excl.shift();
    this.exclRunning = true;
    this.stats.exclusive++;
    if (this.runStart == null) this.runStart = now;
    Promise.resolve().then(e.fn).then(e.resolve, e.reject).finally(() => {
      this.exclRunning = false;
      const t = this.now();
      this.drainAt = t; this.lastDone = t;
      this.pump();
    });
  }

  // ---------------------------------------------------------------- the queue
  _issue(job, now) {
    if (!this.out.length && this.runStart == null) this.runStart = now;
    job.cost = this.costs.get(job.key, job.kind, 0.5);
    job.tIssue = now;
    job.tEnd = Math.max(now, this._queueFree(now)) + job.cost;
    this.out.push(job);
    this.stats.issued++;
    const d = this.out.length;
    this.stats.maxDepth = Math.max(this.stats.maxDepth, d);
    this.stats.depthHist[Math.min(4, d)]++;
    let p;
    try { p = Promise.resolve(job.issue()); } catch (e) { p = Promise.reject(e); }
    p.then(() => this._done(job, null), (e) => this._done(job, e));
  }

  _done(job, err) {
    const t = this.now();
    const i = this.out.indexOf(job);
    if (i >= 0) this.out.splice(i, 1);
    const svc = t - Math.max(job.tIssue, this.lastDone);               // its own GPU time plus bubbles, not its wait
    if (!err) { this.costs.observe(job.key, job.kind, svc); job.onService?.(svc); }
    this.lastDone = Math.max(this.lastDone, t);
    if (!this.out.length) {
      this.drainAt = t;
      if (this.runStart != null) this.stats.longestRunMs = Math.max(this.stats.longestRunMs, t - this.runStart);
    }
    if (err) { this.stats.errors++; this.lastError = String(err?.message ?? err); this.log(`GPU job ${job.key} failed: ${this.lastError}`); }
    this.pump();
  }

  /** Predicted end of the job at the head of the queue. */
  _headEnd(now) {
    const j = this.out[0];
    if (!j) return now;
    const e = Math.max(this.lastDone, j.tIssue) + j.cost;
    return e < now ? now + 0.5 : e;
  }

  /** Predicted time the outstanding jobs are all done (now when the queue is empty). */
  _queueFree(now) {
    if (!this.out.length) return now;
    let prev = this.lastDone;
    for (const j of this.out) {
      let e = Math.max(prev, j.tIssue) + j.cost;
      if (e < now) e = now + 0.5;                                     // overran its prediction: still running
      prev = e;
    }
    return prev;
  }

  _arm(t) {
    if (!Number.isFinite(t)) return;
    if (this.timer && this.timerAt <= t && this.timerAt > this.now()) return;
    this.timer?.();
    this.timerAt = t;
    this.timer = this.setTimer(() => { this.timer = null; this.timerAt = Infinity; this.pump(); }, Math.max(0, t - this.now()));
  }
}
