/**
 * Grades the hand-written forward pass against PyTorch on real games.
 *
 * Fixtures carry the prepped state matrix, that game's pregame logits and the
 * shipped ensemble's answer for every snap, so this tests the port and nothing
 * else — no ESPN, no database, no network.
 *
 *   deno run --allow-read supabase/functions/_shared/verify_ts_parity.ts
 */
import { Meta, Net, readouts } from "./net.ts";

const ROOT = new URL("../../", import.meta.url);
const TOL_WP = 1e-5, TOL_PTS = 1e-3;

const meta: Meta = JSON.parse(await Deno.readTextFile(new URL("model/weights.json", ROOT)));
const bin = await Deno.readFile(new URL("model/weights.bin", ROOT));
const net = new Net(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), meta);

const index = JSON.parse(await Deno.readTextFile(new URL("fixtures/index.json", ROOT)));
let worstWp = 0, worstEm = 0, worstEt = 0, rows = 0;
let t0 = performance.now();

for (const { game_id } of index) {
  const fx = JSON.parse(await Deno.readTextFile(new URL(`fixtures/${game_id}.expected.json`, ROOT)));
  if (!fx.wp) { console.log(`${game_id}: no expected outputs, skipping`); continue; }

  // feature order must match what the exporter recorded, not just by count
  const mismatch = meta.state_features.findIndex((f, i) => f !== fx.state_features[i]);
  if (mismatch >= 0) throw new Error(`state feature order differs at ${mismatch}`);

  const pm = fx.pregame_margin_logits.map((r: number[]) => Float32Array.from(r));
  const pt = fx.pregame_total_logits.map((r: number[]) => Float32Array.from(r));

  let gw = 0, ge = 0, gt = 0;
  for (let i = 0; i < fx.n_rows; i++) {
    const out = net.run(
      Float64Array.from(fx.state[i]),
      fx.t_left[i] / 3600, Math.abs(fx.home_lead[i]) / 21,
      fx.home_lead[i], fx.total_pts_before[i], pm, pt,
    );
    const r = readouts(out.margin, out.total);
    gw = Math.max(gw, Math.abs(r.wp - fx.wp[i]));
    ge = Math.max(ge, Math.abs(r.em - fx.em[i]));
    gt = Math.max(gt, Math.abs(r.et - fx.et[i]));
    rows++;
  }
  worstWp = Math.max(worstWp, gw); worstEm = Math.max(worstEm, ge); worstEt = Math.max(worstEt, gt);
  console.log(`${game_id}: ${String(fx.n_rows).padStart(4)} snaps | ` +
    `|dwp| ${gw.toExponential(2)}  |dmargin| ${ge.toExponential(2)}  |dtotal| ${gt.toExponential(2)}`);
}

const ms = performance.now() - t0;
console.log(`\n${rows} snaps in ${ms.toFixed(0)} ms  (${(ms / rows).toFixed(2)} ms/snap)`);
const ok = worstWp < TOL_WP && worstEm < TOL_PTS && worstEt < TOL_PTS;
console.log(ok
  ? `TS PARITY OK — worst |dwp| ${worstWp.toExponential(2)}`
  : `TS PARITY FAIL — |dwp| ${worstWp.toExponential(2)} |dm| ${worstEm.toExponential(2)} |dt| ${worstEt.toExponential(2)}`);
if (!ok) Deno.exit(1);
