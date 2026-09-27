/**
 * Grades the ported feature construction against pandas.
 *
 * Replays the exact play rows derive_state saw and compares all 53 columns,
 * per snap, to what Python produced. Offline — no ESPN, no database.
 *
 *   deno run --allow-read supabase/functions/_shared/verify_features.ts
 */
import { deriveState, PlayRow, STATE_FEATURES } from "./features.ts";

const ROOT = new URL("../../", import.meta.url);
const TOL = 1e-9;

const index = JSON.parse(await Deno.readTextFile(new URL("fixtures/index.json", ROOT)));
let worst = 0, worstCol = "", worstGame = 0, rows = 0;

for (const { game_id } of index) {
  const fx = JSON.parse(await Deno.readTextFile(new URL(`fixtures/${game_id}.expected.json`, ROOT)));
  if (!fx.rows) { console.log(`${game_id}: no raw rows, skipping`); continue; }

  const order = fx.state_features as string[];
  const mism = STATE_FEATURES.findIndex((f, i) => f !== order[i]);
  if (mism >= 0) throw new Error(`feature order differs at ${mism}: ${STATE_FEATURES[mism]} vs ${order[mism]}`);

  const got = deriveState(fx.rows as PlayRow[]);
  if (got.length !== fx.n_rows) throw new Error(`${game_id}: ${got.length} rows, expected ${fx.n_rows}`);

  let gw = 0, gc = "";
  for (let i = 0; i < fx.n_rows; i++) {
    for (let j = 0; j < STATE_FEATURES.length; j++) {
      const d = Math.abs(got[i][j] - fx.state[i][j]);
      if (d > gw) { gw = d; gc = STATE_FEATURES[j]; }
    }
    rows++;
  }
  if (gw > worst) { worst = gw; worstCol = gc; worstGame = game_id; }
  const flag = gw > TOL ? "  <-- OFF" : "";
  console.log(`${game_id}: ${String(fx.n_rows).padStart(4)} snaps | max |diff| ${gw.toExponential(2)}` +
    (gw > 0 ? ` on ${gc}` : "") + flag);
}

console.log(`\n${rows} snaps x ${STATE_FEATURES.length} features`);
if (worst > TOL) {
  console.log(`FEATURE PARITY FAIL — ${worst.toExponential(3)} on ${worstCol} (game ${worstGame})`);
  Deno.exit(1);
}
console.log(`FEATURE PARITY OK — worst |diff| ${worst.toExponential(2)}`);
