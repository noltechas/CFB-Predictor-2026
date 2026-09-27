import { liveStates, scoreboardContext } from "./espn.ts";
import type { LiveState } from "./espn.ts";

function equal(actual: unknown, expected: unknown) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error(`${JSON.stringify(actual)} != ${JSON.stringify(expected)}`);
}
const stamp = "2026-09-06T22:30:00.000Z";
const state: LiveState = { game_id: 1, state: "in", period: 3, clock: "2:45", home_points: 10, away_points: 0,
  home_id: "264", away_id: "265", home_abbr: "WASH", away_abbr: "WSU", possession: "264", possession_text: "WSU 31",
  down: 2, distance: 6, down_distance: "2nd & 6", home_to: 2, away_to: 3, status_name: "STATUS_IN_PROGRESS" };

Deno.test("possession comes from the team ID, never the side of the field", () => {
  equal(scoreboardContext(state, stamp).possession, "home");
  equal(scoreboardContext({ ...state, possession: "265" }, stamp).possession, "away");
  equal(scoreboardContext({ ...state, possession: null, possession_text: "WASH 31" }, stamp).possession, null);
});
Deno.test("display data preserves missing values, zero distance and neutral field position", () => {
  const missing = scoreboardContext({ ...state, possession: null, down: null, distance: null, possession_text: "" }, stamp);
  equal([missing.possession, missing.down, missing.distance, missing.field_position], [null, null, null, null]);
  const goal = scoreboardContext({ ...state, distance: 0, down_distance: "2nd & Goal", possession_text: "50" }, stamp);
  equal([goal.distance, goal.down_distance, goal.field_position], [0, "2nd & Goal", "50"]);
});
Deno.test("FBS/FCS ingestion deduplicates games and never reads provider probability", async () => {
  const situation = { down: 1, distance: 0, shortDownDistanceText: "1st & Goal", possession: "264", possessionText: "WSU 6" };
  Object.defineProperty(situation, "lastPlay", { get() { throw Error("Provider probability must not be read"); } });
  const event = { id: "1", status: { type: { state: "in", name: "STATUS_IN_PROGRESS", shortDetail: "2:45 - 3rd" }, period: 3, displayClock: "2:45" },
    competitions: [{ competitors: [{ homeAway: "home", team: { id: "264", abbreviation: "WASH" }, score: "10" },
      { homeAway: "away", team: { id: "265", abbreviation: "WSU" }, score: "0" }], situation }] };
  const saved = globalThis.fetch, urls: string[] = [];
  globalThis.fetch = ((url: string) => {
    urls.push(url);
    return Promise.resolve({ ok: true, json: () => Promise.resolve({ events: [event] }) } as Response);
  }) as typeof fetch;
  try {
    const rows = await liveStates();
    equal(urls.length, 2); equal(rows.length, 1);
    const result = scoreboardContext(rows[0], stamp);
    equal([result.possession, result.down_distance, result.field_position, result.status_text], ["home", "1st & Goal", "WSU 6", "2:45 - 3rd"]);
  } finally { globalThis.fetch = saved; }
});
