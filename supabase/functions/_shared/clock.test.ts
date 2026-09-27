import { cleanClock, tLeftOf } from "./espn.ts";
import { staleRows, StoredCurve } from "./curve_history.ts";
import { PlayRow } from "./features.ts";

const assert = (ok: boolean, message: string) => { if (!ok) throw new Error(message); };
const row = (period: number, clock: string, t_left = tLeftOf(period, clock)) =>
  ({ period, half: period <= 2 ? 1 : 2, clock, t_left }) as PlayRow;
const inQuarter = (r: PlayRow) => r.period >= 5 ? r.t_left === 0
  : r.t_left >= (4 - r.period) * 900 && r.t_left <= (4 - r.period) * 900 + 900;
const ordered = (rows: PlayRow[]) => rows.every((r, i) => i === 0 || r.t_left <= rows[i - 1].t_left);

// Real stamp sequences frozen from ESPN, with the Python scorer's answers
// (python3 tests: tests/test_live_clock.py pins the same file).
const corpus = JSON.parse(await Deno.readTextFile(new URL("../../fixtures/clock_cases.json", import.meta.url)));

Deno.test("real ESPN glitches clean exactly as the Python scorer cleans them", () => {
  for (const c of corpus) {
    const got = cleanClock(c.rows.map(([p, clk, t]: [number, string, number]) => row(p, clk, t)), c.now_last);
    c.expect.forEach(([p, clk, t]: [number, string, number], i: number) => {
      assert(Math.abs(got[i].t_left - t) < 1e-6 && got[i].period === p && got[i].clock === clk,
        `${c.name} row ${i}: ${JSON.stringify(got[i])} != ${JSON.stringify([p, clk, t])}`);
    });
    assert(ordered(got), `${c.name}: time runs backward`);
    assert(got.every(inQuarter), `${c.name}: a point sits outside its own quarter`);
    assert(got.every((r) => r.half === (r.period <= 2 ? 1 : 2)), `${c.name}: half disagrees with period`);
  }
});

Deno.test("a quarter's first play stamped 0:00 cannot drag the whole quarter", () => {
  // Nebraska @ Michigan State: every later Q2 play piled onto the Q2/Q3 line
  const rows = cleanClock([row(1, "0:05"), row(1, "0:00"), row(2, "0:00"),
    row(2, "15:00"), row(2, "15:00"), row(2, "11:54"), row(2, "10:00"), row(4, "0:00", 0)], true);
  assert(rows[2].t_left <= 2700 && rows[2].t_left >= 2514, `bad stamp misplaced: ${rows[2].t_left}`);
  assert(rows[5].t_left === tLeftOf(2, "11:54") && rows[6].t_left === tLeftOf(2, "10:00"), "good stamps moved");
  assert(rows.every(inQuarter) && ordered(rows), "quarter bounds or order broken");
});

Deno.test("a lone bad latest stamp loses to the scoreboard clock", () => {
  // the 0:00 play is the newest; the scoreboard says 14:52 of the same quarter
  const rows = cleanClock([row(1, "0:05"), row(1, "0:00"), row(2, "0:00"), row(2, "14:52")], true);
  assert(rows[3].t_left === tLeftOf(2, "14:52"), `live point moved: ${rows[3].t_left}`);
  assert(rows[2].t_left >= rows[3].t_left, "bad stamp still ahead of the live point");
});

Deno.test("an impossible period is relabelled and cannot pin the game to 0:00", () => {
  // Houston Christian @ Incarnate Word: an 'End of OT' row inside Q2
  const rows = cleanClock([row(2, "15:00"), row(5, "0:00"), row(2, "15:00"),
    row(2, "14:26"), row(2, "13:45"), row(3, "10:00"), row(4, "0:00", 0)], true);
  assert(rows[1].period === 2 && rows[1].half === 1 && inQuarter(rows[1]), `OT row kept: ${JSON.stringify(rows[1])}`);
  assert(rows[5].t_left === tLeftOf(3, "10:00"), "later plays were dragged to 0:00");
});

Deno.test("a batch-stamped tail never spills into the next quarter", () => {
  const drive = Array.from({ length: 40 }, () => row(2, "1:00"));
  const rows = cleanClock([row(2, "5:00"), ...drive]);
  assert(rows.every(inQuarter), `tail left Q2: ${rows.at(-1)!.t_left}`);
});

Deno.test("a scoreboard trailing the feed sits just after the last play", () => {
  const rows = cleanClock([row(3, "10:00"), row(3, "9:40"), row(3, "9:30"), row(3, "9:50")], true);
  assert(rows[3].t_left === tLeftOf(3, "9:30") - 1, `trailing now: ${rows[3].t_left}`);
});

Deno.test("plays before the live point hold still while the clock runs", () => {
  // 25 s apart from the drive's stamp; squeezed only while "now" is closer
  const drive = [row(2, "8:21"), row(2, "8:21"), row(2, "8:21"), row(2, "8:21")];
  const tick = (clock: string) => cleanClock([...drive.map((r) => ({ ...r })), row(2, clock)], true);
  const [a, b, squeezed] = [tick("6:07"), tick("4:40"), tick("8:00")];
  drive.forEach((_, i) => assert(a[i].t_left === b[i].t_left, `play ${i} re-timed every tick`));
  assert(squeezed.slice(0, 4).every((r) => r.t_left > squeezed[4].t_left), "plays placed after the live point");
});

Deno.test("stored points are re-priced only when their clock or score really moved", () => {
  const rows = cleanClock([row(1, "15:00"), row(1, "14:20"), row(1, "13:10"), row(1, "12:00")]);
  rows.forEach((r) => { r.hp = 0; r.ap = 0; });
  const stored: StoredCurve = {
    t_left: [3600.0001, 3560, 3490 + 40, 3420],   // float4 noise, exact, 40 s off, exact
    period: [1, 1, 1, 1], home_points: [0, 0, 0, 7], away_points: [0, 0, 0, 0],
  };
  const stale = staleRows(stored, rows, 4);
  assert(JSON.stringify(stale) === "[2,3]", `stale rows ${stale}`);
  assert(staleRows(stored, rows, 2).length === 0, "rows at or past the resume point double-counted");
  assert(staleRows(undefined, rows, 4).length === 0, "missing history invented stale rows");
});

Deno.test("a point within tolerance is still re-priced if it would step backward", () => {
  const rows = cleanClock([row(2, "2:00"), row(2, "1:55"), row(2, "1:50"), row(2, "1:45")]);
  rows.forEach((r) => { r.hp = 0; r.ap = 0; });
  const zeros = [0, 0, 0, 0], period = [2, 2, 2, 2];
  // row 1 drifted 8 s low (below fresh row 2), row 2 drifted 7 s high (above fresh row 1)
  const t = rows.map((r) => r.t_left);
  const lowish: StoredCurve = { t_left: [t[0], t[1] - 8, t[2], t[3]], period, home_points: zeros, away_points: zeros };
  const highish: StoredCurve = { t_left: [t[0], t[1], t[2] + 7, t[3]], period, home_points: zeros, away_points: zeros };
  assert(JSON.stringify(staleRows(lowish, rows, 4)) === "[1]", `low drift ${staleRows(lowish, rows, 4)}`);
  assert(JSON.stringify(staleRows(highish, rows, 4)) === "[2]", `high drift ${staleRows(highish, rows, 4)}`);
  // equal stamps survive the real (float4) round-trip without churn
  const tied = cleanClock([row(2, "1:00"), row(2, "1:00"), row(2, "1:00")], true);
  tied.forEach((r) => { r.hp = 0; r.ap = 0; });
  const f4 = tied.map((r) => Math.fround(r.t_left));
  assert(staleRows({ t_left: f4, period: [2, 2, 2], home_points: [0, 0, 0], away_points: [0, 0, 0] }, tied, 3).length === 0,
    "float4 rounding flagged as stale");
});
