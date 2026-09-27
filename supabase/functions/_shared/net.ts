/**
 * The live half of the ensemble, by hand.
 *
 * ONNX Runtime cannot load in Supabase's edge sandbox — 1.19 ships only
 * threaded wasm (no SharedArrayBuffer there) and 1.18's browser build breaks
 * under Deno's node-compat. The graph is small and every op is standard, so
 * it is reimplemented here against raw float32 weights. Verified against
 * PyTorch on real games by verify_ts_parity.
 */

export const MARGIN_LO = -56, MARGIN_HI = 56, N_MARGIN = 115;
export const TOTAL_LO = 10, TOTAL_HI = 110, N_TOTAL = 103;
export const REM_HI = 110, N_REM = 112;

const centres = (lo: number, hi: number, tail = 6) => {
  const out = new Float64Array(hi - lo + 3);
  out[0] = lo - tail;
  for (let i = lo; i <= hi; i++) out[i - lo + 1] = i;
  out[out.length - 1] = hi + tail;
  return out;
};
export const MARGIN_C = centres(MARGIN_LO, MARGIN_HI);
export const TOTAL_C = centres(TOTAL_LO, TOTAL_HI);
/** remaining points: inner bins ARE their index (0..110), one upper tail */
export const REM_C = (() => {
  const o = new Float64Array(N_REM);
  for (let i = 0; i <= REM_HI; i++) o[i] = i;
  o[N_REM - 1] = REM_HI + 6;
  return o;
})();

const marginBin = (m: number) =>
  m < MARGIN_LO ? 0 : m > MARGIN_HI ? N_MARGIN - 1
    : Math.min(Math.max(m - MARGIN_LO + 1, 1), N_MARGIN - 2);
const totalBin = (t: number) =>
  t < TOTAL_LO ? 0 : t > TOTAL_HI ? N_TOTAL - 1
    : Math.min(Math.max(t - TOTAL_LO + 1, 1), N_TOTAL - 2);
const remBin = (r: number) =>
  r > REM_HI ? N_REM - 1 : Math.min(Math.max(r, 0), REM_HI);

/* ------------------------------------------------------------ primitives */

/** Numerical Recipes erfc, fractional error < 1.2e-7 — well inside the 1e-5
 *  tolerance the parity gate holds the whole pipeline to. */
function erf(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  // the polynomial is in t itself — an earlier transcription shifted it by
  // 0.5 and erf(0) came back 0.489, which the parity gate caught
  const ans = t * Math.exp(
    -z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
      t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
        t * (-0.82215223 + t * 0.17087277)))))))));
  const erfc = x >= 0 ? ans : 2 - ans;
  return 1 - erfc;
}

/** PyTorch nn.GELU default is the exact erf form, not the tanh approximation. */
const gelu = (x: number) => 0.5 * x * (1 + erf(x * Math.SQRT1_2));
const softplus = (x: number) => (x > 20 ? x : Math.log1p(Math.exp(x)));
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** y = W x + b, with W row-major [out, in] exactly as PyTorch stores it. */
function linear(x: Float64Array, W: Float32Array, b: Float32Array,
                nIn: number, nOut: number, out: Float64Array): void {
  for (let o = 0; o < nOut; o++) {
    let s = b[o];
    const base = o * nIn;
    for (let i = 0; i < nIn; i++) s += x[i] * W[base + i];
    out[o] = s;
  }
}

function layerNorm(x: Float64Array, w: Float32Array, b: Float32Array,
                   n: number, eps: number): void {
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) { const d = x[i] - mean; v += d * d; }
  v /= n;                                    // PyTorch uses the biased variance
  const inv = 1 / Math.sqrt(v + eps);
  for (let i = 0; i < n; i++) x[i] = (x[i] - mean) * inv * w[i] + b[i];
}

function softmax(x: Float64Array, n: number): void {
  let mx = -Infinity;
  for (let i = 0; i < n; i++) if (x[i] > mx) mx = x[i];
  let s = 0;
  for (let i = 0; i < n; i++) { x[i] = Math.exp(x[i] - mx); s += x[i]; }
  for (let i = 0; i < n; i++) x[i] /= s;
}

/** Move a PMF by an integer offset and re-bin, mass collapsing into the tails.
 *  Pure score arithmetic — the same identity the Python side encodes. */
function shiftPmf(p: Float64Array, src: Float64Array, offset: number,
                  binFn: (v: number) => number, out: Float64Array): void {
  out.fill(0);
  for (let j = 0; j < src.length; j++) out[binFn(Math.round(src[j] + offset))] += p[j];
}

/* ---------------------------------------------------------------- weights */

type Ref = { off: number; shape: number[] };
type Block = { lin_w: Ref; lin_b: Ref; ln_w: Ref; ln_b: Ref; ln_eps: number };
type MemberMeta = {
  blocks: Block[]; margin_w: Ref; margin_b: Ref; total_w: Ref; total_b: Ref;
  gate: { w_time: number; w_lead: number; bias: number; w_sharp: number; floor: number };
};
export type Meta = {
  model_version: string; n_members: number; n_state: number;
  state_features: string[]; members: MemberMeta[];
};

export class Net {
  private buf: Float32Array;
  readonly meta: Meta;
  /* Scratch, allocated once. Scoring a full game is 16 members x ~200 snaps;
     allocating per member per snap churned ~50k arrays and tripped Supabase's
     WORKER_RESOURCE_LIMIT the first time a real curve was written. */
  private s: {
    pmP: Float64Array; ptP: Float64Array; hIn: Float64Array;
    hA: Float64Array; hB: Float64Array;
    sm: Float64Array; st: Float64Array;
    preM: Float64Array; preT: Float64Array;
    bm: Float64Array; bt: Float64Array;
    outM: Float64Array; outT: Float64Array;
    mSum: Float64Array; tSum: Float64Array;
  };
  constructor(weights: ArrayBuffer, meta: Meta) {
    this.buf = new Float32Array(weights);
    this.meta = meta;
    const hid = meta.members[0].blocks[0].lin_w.shape[0];
    this.s = {
      pmP: new Float64Array(N_MARGIN), ptP: new Float64Array(N_TOTAL),
      hIn: new Float64Array(meta.n_state + 2),
      hA: new Float64Array(hid), hB: new Float64Array(hid),
      sm: new Float64Array(N_MARGIN), st: new Float64Array(N_REM),
      preM: new Float64Array(N_MARGIN), preT: new Float64Array(N_REM),
      bm: new Float64Array(N_MARGIN), bt: new Float64Array(N_REM),
      outM: new Float64Array(N_MARGIN), outT: new Float64Array(N_TOTAL),
      mSum: new Float64Array(N_MARGIN), tSum: new Float64Array(N_TOTAL),
    };
  }
  private slice(r: Ref): Float32Array {
    const n = r.shape.reduce((a, b) => a * b, 1);
    return this.buf.subarray(r.off, r.off + n);
  }

  /**
   * One snap. `pm`/`pt` are that game's per-member pregame logits.
   * Returns the ensemble-averaged final-space PMFs.
   */
  run(state: Float64Array, fr: number, al: number, lead: number, pts: number,
      pm: Float32Array[], pt: Float32Array[]): { margin: Float64Array; total: Float64Array } {
    const M = this.meta.n_members;
    const nS = this.meta.n_state;
    const S = this.s;
    const { pmP, ptP, sm, st, preM, preT, bm, bt, outM, outT, mSum, tSum } = S;
    mSum.fill(0);
    tSum.fill(0);

    for (let k = 0; k < M; k++) {
      const mm = this.meta.members[k];

      // cond: the only pregame signal the state tower sees
      for (let i = 0; i < N_MARGIN; i++) pmP[i] = pm[k][i];
      softmax(pmP, N_MARGIN);
      for (let i = 0; i < N_TOTAL; i++) ptP[i] = pt[k][i];
      softmax(ptP, N_TOTAL);
      let em = 0, et = 0;
      for (let i = 0; i < N_MARGIN; i++) em += pmP[i] * MARGIN_C[i];
      for (let i = 0; i < N_TOTAL; i++) et += ptP[i] * TOTAL_C[i];

      // body: [state | cond] through 3 x (Linear -> LayerNorm -> GELU),
      // ping-ponging between two scratch buffers
      S.hIn.set(state.subarray(0, nS));
      S.hIn[nS] = em / 20;
      S.hIn[nS + 1] = (et - 55) / 20;
      let src: Float64Array = S.hIn;
      let dst: Float64Array = S.hA;
      for (const blk of mm.blocks) {
        const [nOut, nIn] = blk.lin_w.shape;
        linear(src, this.slice(blk.lin_w), this.slice(blk.lin_b), nIn, nOut, dst);
        layerNorm(dst, this.slice(blk.ln_w), this.slice(blk.ln_b), nOut, blk.ln_eps);
        for (let i = 0; i < nOut; i++) dst[i] = gelu(dst[i]);
        src = dst;
        dst = dst === S.hA ? S.hB : S.hA;
      }
      const h = src;
      const [nMo, nMi] = mm.margin_w.shape;
      linear(h, this.slice(mm.margin_w), this.slice(mm.margin_b), nMi, nMo, sm);
      const [nTo, nTi] = mm.total_w.shape;
      linear(h, this.slice(mm.total_w), this.slice(mm.total_b), nTi, nTo, st);

      // gate
      const g = mm.gate;
      const z = softplus(g.w_time) * fr - softplus(g.w_lead) * al + g.bias;
      const ramp = Math.pow(Math.min(Math.max(fr, 0), 1), softplus(g.w_sharp));
      const a = g.floor + (1 - g.floor) * sigmoid(z) * ramp;

      // pregame PMFs -> delta space, blend in log space, back to final space
      const EPS = 1e-9;
      shiftPmf(pmP, MARGIN_C, -lead, marginBin, preM);
      shiftPmf(ptP, TOTAL_C, -pts, remBin, preT);
      for (let i = 0; i < N_MARGIN; i++) bm[i] = a * Math.log(preM[i] + EPS) + (1 - a) * sm[i];
      softmax(bm, N_MARGIN);
      for (let i = 0; i < N_REM; i++) bt[i] = a * Math.log(preT[i] + EPS) + (1 - a) * st[i];
      softmax(bt, N_REM);

      shiftPmf(bm, MARGIN_C, lead, marginBin, outM);
      shiftPmf(bt, REM_C, pts, totalBin, outT);
      // the shipped path softmaxes each member's log-probs before averaging;
      // softmax(log p) renormalises to p, so accumulate the PMFs directly
      let sM = 0, sT = 0;
      for (let i = 0; i < N_MARGIN; i++) sM += outM[i];
      for (let i = 0; i < N_TOTAL; i++) sT += outT[i];
      for (let i = 0; i < N_MARGIN; i++) mSum[i] += outM[i] / sM;
      for (let i = 0; i < N_TOTAL; i++) tSum[i] += outT[i] / sT;
    }
    for (let i = 0; i < N_MARGIN; i++) mSum[i] /= M;
    for (let i = 0; i < N_TOTAL; i++) tSum[i] /= M;
    return { margin: mSum, total: tSum };
  }
}

/** wp / expected margin / expected total — the three numbers the app shows. */
export function readouts(margin: Float64Array, total: Float64Array) {
  let wp = 0, em = 0, et = 0;
  for (let i = 0; i < N_MARGIN; i++) { if (MARGIN_C[i] > 0) wp += margin[i]; em += margin[i] * MARGIN_C[i]; }
  for (let i = 0; i < N_TOTAL; i++) et += total[i] * TOTAL_C[i];
  return { wp, em, et };
}
