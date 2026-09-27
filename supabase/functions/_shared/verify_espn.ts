/**
 * Grades the ported ESPN parse against Python on frozen payloads.
 *
 *   deno run --allow-read supabase/functions/_shared/verify_espn.ts
 */
import { LiveState, parsePlayRows } from "./espn.ts";
import { PlayRow } from "./features.ts";

const ROOT = new URL("../../", import.meta.url);
// t_left is not graded here: these fixtures froze the retired running-minimum
// clock cleaning. cleanClock is graded against the Python scorer on real
// feeds in clock.test.ts (fixtures/clock_cases.json).
const KEYS: (keyof PlayRow)[] = [
  "period", "half", "home_lead", "total_pts_before", "hp", "ap",
  "home_has_ball", "down", "distance", "home_ytg", "off_ytg", "home_to",
  "away_to", "recv_2h", "drive_no", "synthetic", "yds", "is_scrim", "is_pass",
  "is_sack", "is_to", "is_pen", "is_poss_chg",
];

async function readGz(url: URL) {
  const raw = await Deno.readFile(url);
  const ds = new DecompressionStream("gzip");
  const stream = new Blob([raw]).stream().pipeThrough(ds);
  return JSON.parse(await new Response(stream).text());
}

const index = JSON.parse(await Deno.readTextFile(new URL("fixtures/index.json", ROOT)));
let bad = 0, checked = 0;

for (const { game_id } of index) {
  const fx = JSON.parse(await Deno.readTextFile(new URL(`fixtures/${game_id}.expected.json`, ROOT)));
  const summary = await readGz(new URL(`fixtures/${game_id}.summary.json.gz`, ROOT));
  const got = parsePlayRows(summary, fx.scoreboard_row as LiveState);
  const want = fx.rows as PlayRow[];

  if (got.length !== want.length) {
    console.log(`${game_id}: ROW COUNT ${got.length} vs ${want.length}`);
    bad++; continue;
  }
  let gw = 0, gk = "", gi = -1;
  for (let i = 0; i < want.length; i++) {
    for (const k of KEYS) {
      const d = Math.abs(Number(got[i][k]) - Number(want[i][k]));
      if (d > gw) { gw = d; gk = String(k); gi = i; }
    }
    checked++;
  }
  const ok = gw < 1e-9;
  if (!ok) bad++;
  console.log(`${game_id}: ${String(want.length).padStart(4)} rows | ` +
    (ok ? "exact" : `max |diff| ${gw.toExponential(2)} on ${gk} at row ${gi}`));
}

console.log(`\n${checked} rows x ${KEYS.length} fields`);
console.log(bad === 0 ? "ESPN PARSE PARITY OK" : `ESPN PARSE PARITY FAIL (${bad} games)`);
if (bad) Deno.exit(1);
