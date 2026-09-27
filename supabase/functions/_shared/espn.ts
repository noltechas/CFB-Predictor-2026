/**
 * ESPN ingestion, ported from live_scorer.py.
 *
 * Only score / clock / down / distance / possession are read. ESPN's own win
 * probability sits in `situation.lastPlay` and is deliberately NOT parsed —
 * the model may never see another model's output.
 */
import { PlayRow } from "./features.ts";

export const SCOREBOARD = (grp: number) =>
  `https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?groups=${grp}&limit=400`;
export const SUMMARY = (gid: number) =>
  `https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event=${gid}`;

export type LiveState = {
  game_id: number; state: string; period: number; clock: string;
  home_points: number; away_points: number;
  home_id: string; away_id: string; home_abbr: string; away_abbr: string;
  down: number | null; distance: number | null;
  possession: string | null; possession_text: string;
  home_to: number | null; away_to: number | null;
  status_name?: string | null; status_text?: string | null;
  down_distance?: string | null;
};

/** Display state is separate from synthNowRow's defaults used by the model.
 * possessionText describes the side of the FIELD, not the team with the ball. */
export function scoreboardContext(r: LiveState, stamp: string) {
  const ball = r.possession == null ? "" : String(r.possession);
  return {
    updated_at: stamp,
    possession: ball && ball === r.home_id ? "home" : ball && ball === r.away_id ? "away" : null,
    down: r.down != null && r.down >= 1 && r.down <= 4 ? r.down : null,
    distance: r.distance != null && r.distance >= 0 ? r.distance : null,
    down_distance: r.down_distance || null,
    field_position: r.possession_text || null,
    status_name: r.status_name || null, status_text: r.status_text || null,
  };
}

const PLAY_COLS = {
  yds: 0, is_scrim: 0, is_pass: 0, is_sack: 0, is_to: 0, is_pen: 0,
  is_poss_chg: 0, h_pqb_play: 0, a_pqb_play: 0, h_prb_play: 0, a_prb_play: 0,
  h_pqb_known: 0, a_pqb_known: 0,
};

/** Python's `x or default`: 0 and "" fall through, not just null. Using `??`
 *  here turned a `distance: 0` into 0 where the scorer sees 10. */
const pyOr = <T>(v: any, d: T): T => (v === 0 || v === "" || v == null ? d : v as T);

export const clockSeconds = (d: string) => {
  const m = /^(\d+):(\d+)$/.exec(d ?? "");
  return m ? +m[1] * 60 + +m[2] : 0;
};
export const tLeftOf = (period: number, clock: string) =>
  period <= 4 ? Math.max(0, 4 - period) * 900 + clockSeconds(clock) : 0;

/** A period's slice of regulation time. Every overtime period is t_left 0. */
const periodRange = (p: number): [number, number] =>
  p >= 5 ? [0, 0] : [Math.max(0, 4 - p) * 900, Math.max(0, 4 - p) * 900 + 900];

/** The regulation quarter holding t; a boundary goes to the one nearest `own`. */
function quarterOf(t: number, own: number): number {
  let best = -1;
  for (const q of [1, 2, 3, 4]) {
    const [lo, hi] = periodRange(q);
    if (lo <= t && t <= hi && (best < 0 || Math.abs(q - own) < Math.abs(best - own))) best = q;
  }
  return best < 0 ? own : best;
}

const fmtClock = (t: number, p: number) => {
  const s = Math.floor(t - periodRange(p)[0]);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

/**
 * Turn ESPN's per-play clock stamps into a usable game-time axis, in place.
 *
 * Play ORDER (sequenceNumber) is trustworthy; stamps are not. Whole drives
 * share their start clock, FCS feeds stamp backward mid-backfill, and single
 * rows arrive wildly low: the first play of a quarter at "0:00", an
 * interception at "Q4 0:00" with 18 plays still to come, even an "End of OT"
 * row sequenced inside the second quarter. A running minimum let any one of
 * those drag every later play down to it, which piled a whole quarter onto
 * one point and pushed the live point quarters ahead of the real clock
 * (Nebraska @ Michigan State, 2026-09-26).
 *
 * So the anchors are the longest non-increasing run of stamps, the stamps the
 * most plays agree with. Ties keep higher stamps, since a bad stamp is
 * usually too low. Every other row is placed between its neighbouring
 * anchors, a batch-stamped drive is spread toward the next distinct stamp,
 * and no row leaves its own quarter unless its period is itself impossible.
 *
 * `nowLast` marks the final row as the scoreboard's observed state. It votes
 * like a play, so a lone bad latest stamp loses to the real clock. It is a
 * floor for the plays before it, not a target, so their positions stay put
 * from one tick to the next.
 */
export function cleanClock(rows: PlayRow[], nowLast = false): PlayRow[] {
  const n = rows.length;
  if (!n) return rows;
  const raw = rows.map((r) => r.t_left);
  const len = new Array<number>(n).fill(1), prev = new Array<number>(n).fill(-1);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) {
      if (raw[j] >= raw[i] && (len[j] + 1 > len[i] || (len[j] + 1 === len[i] && raw[j] > raw[prev[i]]))) {
        len[i] = len[j] + 1; prev[i] = j;
      }
    }
  }
  let end = 0;
  for (let i = 1; i < n; i++) if (len[i] > len[end] || (len[i] === len[end] && raw[i] > raw[end])) end = i;
  const anchor = new Array<boolean>(n).fill(false);
  for (let i = end; i >= 0; i = prev[i]) anchor[i] = true;
  const A: number[] = [];
  for (let i = 0; i < n; i++) if (anchor[i]) A.push(i);
  const heads = A.filter((a, k) => k === 0 || raw[a] < raw[A[k - 1]]);

  // [start, end) of each group, the stamp it starts from, and the next distinct one
  const segs: [number, number, number, number | null][] = [];
  if (A[0] > 0) segs.push([0, A[0], 3600, raw[A[0]]]);
  heads.forEach((a, h) => {
    const b = h + 1 < heads.length ? heads[h + 1] : n;
    segs.push([a, b, raw[a], b < n ? raw[b] : null]);
  });

  const t = raw.slice();
  const nowFree = nowLast && !anchor[n - 1];
  for (const [a, b0, top, nxt] of segs) {
    const b = nxt == null && nowFree ? b0 - 1 : b0;   // a trailing scoreboard row goes last
    const m = b - a;
    if (m <= 0) continue;
    let bottom: number, floor: number;
    if (nxt != null && nowLast && b === n - 1) { bottom = Math.max(top - 25 * m, nxt); floor = nxt; }
    else if (nxt != null) { bottom = nxt; floor = nxt; }
    else if (nowLast && anchor[n - 1]) { bottom = top; floor = top; }
    else { bottom = Math.max(top - 25 * m, 0); floor = 0; }
    const span = Math.max(top - bottom, 0);
    for (let k = 0; k < m; k++) {
      const pos = a + k;
      const v = top - span * k / m;
      const [loP, hiP] = periodRange(rows[pos].period);
      const lo = Math.max(loP, floor), hi = Math.min(hiP, top);
      t[pos] = lo <= hi ? Math.min(Math.max(v, lo), hi) : v;
    }
  }
  if (nowFree) {
    const before = n >= 2 ? t[n - 2] : raw[n - 1];
    const v = Math.max(before - 1, 0);
    const [loP, hiP] = periodRange(rows[n - 1].period);
    const lo = loP, hi = Math.min(hiP, before);
    t[n - 1] = lo <= hi ? Math.min(Math.max(v, lo), hi) : v;
  }
  for (let k = 1; k < n; k++) if (t[k] > t[k - 1]) t[k] = t[k - 1];
  rows.forEach((r, k) => {
    r.t_left = t[k];
    const [loP, hiP] = periodRange(r.period);
    if (!(loP <= t[k] && t[k] <= hiP)) {
      r.period = quarterOf(t[k], r.period);
      r.half = r.period <= 2 ? 1 : 2;
      r.clock = fmtClock(t[k], r.period);
    }
  });
  return rows;
}

/**
 * ESPN's WAF is picky about who is asking. Deno's own `Deno/2.x` UA gets a
 * 403 text/html "Access Denied", and so do browser-shaped UAs — measured, all
 * four rejected. The one that passes is the library UA the Mac scorer has
 * always sent, so send exactly that rather than pretending to be a browser.
 */
const ESPN_HEADERS = {
  Accept: "application/json",
  "User-Agent": "python-requests/2.31.0",
};

async function getJson(url: string, ms: number): Promise<any | null> {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), ms);
  try {
    const r = await fetch(url, { signal: ctl.signal, headers: ESPN_HEADERS });
    return r.ok ? await r.json() : null;
  } catch { return null; } finally { clearTimeout(t); }
}

/** Every in-progress or just-finished game on both scoreboards. */
export async function liveStates(): Promise<LiveState[]> {
  const out: LiveState[] = [];
  const seen = new Set<number>();
  for (const grp of [80, 81]) {
    const d = await getJson(SCOREBOARD(grp), 20_000);
    for (const e of d?.events ?? []) {
      const gid = Number(e?.id);
      if (!Number.isFinite(gid) || seen.has(gid)) continue;   // placeholder events have no id
      seen.add(gid);
      const st = e.status ?? {};
      const state = st.type?.state;
      if (state !== "in" && state !== "post") continue;
      const c = e.competitions?.[0];
      const comp: Record<string, any> = {};
      for (const x of c?.competitors ?? []) comp[x.homeAway] = x;
      if (!comp.home || !comp.away) continue;
      const sit = c.situation ?? {};
      out.push({
        game_id: gid, state, period: Number(pyOr(st.period, 1)),
        clock: st.displayClock || "0:00",
        home_points: Number(comp.home.score ?? 0), away_points: Number(comp.away.score ?? 0),
        home_id: String(comp.home.team?.id ?? ""), away_id: String(comp.away.team?.id ?? ""),
        home_abbr: comp.home.team?.abbreviation ?? "", away_abbr: comp.away.team?.abbreviation ?? "",
        down: sit.down ?? null, distance: sit.distance ?? null,
        possession: sit.possession ?? null, possession_text: sit.possessionText ?? "",
        home_to: sit.homeTimeouts ?? null, away_to: sit.awayTimeouts ?? null,
        status_name: st.type?.name ?? null,
        status_text: st.type?.shortDetail ?? st.type?.detail ?? null,
        down_distance: sit.shortDownDistanceText ?? sit.downDistanceText?.split(/\s+at\s+/i)[0] ?? null,
      });
    }
  }
  return out;
}

/** Rebuild a scoreboard-shaped state from the summary header. The live
 *  scoreboard drops finished games, so replaying one needs this. */
export async function stateFromSummary(gid: number): Promise<LiveState | null> {
  const d = await getJson(SUMMARY(gid), 12_000);
  const c = d?.header?.competitions?.[0];
  if (!c) return null;
  const comp: Record<string, any> = {};
  for (const x of c.competitors ?? []) comp[x.homeAway] = x;
  if (!comp.home || !comp.away) return null;
  const st = c.status ?? {};
  return {
    game_id: gid, state: st.type?.state ?? "post",
    period: Number(pyOr(st.period, 4)), clock: pyOr(st.displayClock, "0:00"),
    home_points: Number(comp.home.score ?? 0), away_points: Number(comp.away.score ?? 0),
    home_id: String(comp.home.team?.id ?? ""), away_id: String(comp.away.team?.id ?? ""),
    home_abbr: comp.home.team?.abbreviation ?? "", away_abbr: comp.away.team?.abbreviation ?? "",
    down: null, distance: null, possession: null, possession_text: "",
    home_to: null, away_to: null,
    status_name: st.type?.name ?? null, status_text: st.type?.shortDetail ?? st.type?.detail ?? null,
  };
}

function teamAliases(d: any, r: LiveState): [Set<string>, Set<string>] {
  const home = new Set([r.home_abbr.toUpperCase()]);
  const away = new Set([r.away_abbr.toUpperCase()]);
  for (const c of d?.header?.competitions?.[0]?.competitors ?? []) {
    const t = c.team ?? {};
    const names = ["abbreviation", "location", "shortDisplayName", "displayName", "nickname"]
      .map((k) => String(t[k] ?? "").toUpperCase()).filter(Boolean);
    for (const nm of names) (c.homeAway === "home" ? home : away).add(nm);
  }
  return [home, away];
}

/** The state RIGHT NOW from the scoreboard poll. */
export function synthNowRow(r: LiveState, driveNo: number, recv2h: number,
                            synthetic: number, hToFb = 3, aToFb = 3): PlayRow {
  const period = Math.min(r.period, 5);
  let hb = 0.5;
  if (r.possession === r.home_id) hb = 1;
  else if (r.possession === r.away_id) hb = 0;
  else if (r.possession_text.startsWith(r.home_abbr + " ")) hb = 1;
  else if (r.possession_text.startsWith(r.away_abbr + " ")) hb = 0;

  let offYtg = 50;
  const m = /^([A-Z&'-]+) (\d+)$/.exec(r.possession_text);
  if (m && (hb === 0 || hb === 1)) {
    const own = m[1] === (hb === 1 ? r.home_abbr : r.away_abbr);
    const yl = +m[2];
    offYtg = own ? 100 - yl : yl;
  }
  const down = r.down && r.down > 0 ? r.down : 1;
  const dist = r.distance && r.distance > 0 ? r.distance : 10;
  let sit = r.down && r.down > 0 ? `${down} & ${dist}` : "";
  if (r.possession_text) sit = sit ? `${sit} at ${r.possession_text}` : r.possession_text;

  return {
    play_index: 0, period, half: period <= 2 ? 1 : 2,
    t_left: tLeftOf(period, r.clock), clock: r.clock,
    home_lead: r.home_points - r.away_points,
    total_pts_before: r.home_points + r.away_points,
    hp: r.home_points, ap: r.away_points,
    home_has_ball: hb, down, distance: dist,
    home_ytg: hb === 1 ? offYtg : hb === 0 ? 100 - offYtg : 50,
    off_ytg: offYtg,
    home_to: r.home_to ?? hToFb, away_to: r.away_to ?? aToFb,
    recv_2h: recv2h, drive_no: driveNo, sit, play_text: "",
    synthetic, ...PLAY_COLS,
  };
}

/**
 * The game's play sequence so far, state strictly BEFORE each play, with
 * RAW clock stamps (see cleanClock).
 * Empty when ESPN has no drives feed for the game (true of every matchup
 * below FCS — scores exist, plays do not).
 */
export async function fetchPlayRows(r: LiveState): Promise<PlayRow[]> {
  const d = await getJson(SUMMARY(r.game_id), 12_000);
  return d ? parsePlayRows(d, r) : [];
}

/** The parse itself, split out so it can be graded on frozen payloads. */
export function parsePlayRows(d: any, r: LiveState): PlayRow[] {
  const [homeNames, awayNames] = teamAliases(d, r);
  const dr = d.drives ?? {};
  const drives: any[] = [...(dr.previous ?? [])];
  const cur = dr.current;
  if (cur && drives.every((x) => x?.id !== cur.id)) drives.push(cur);

  const plays: { seq: number; di: number; off: string; p: any }[] = [];
  drives.forEach((drv, di) => {
    const off = String(drv?.team?.id ?? "");
    for (const p of drv?.plays ?? []) plays.push({ seq: Number(p.sequenceNumber ?? 0), di, off, p });
  });
  if (!plays.length) return [];
  plays.sort((a, b) => a.seq - b.seq);

  const firstOff = plays[0].off;
  // whoever received the opening kick kicks off the second half
  const recv2h = firstOff === r.away_id ? 1 : firstOff === r.home_id ? 0 : 0.5;

  const rows: PlayRow[] = [];
  let hp = 0, ap = 0, hTo = 3, aTo = 3, in2h = false;
  let prevOff: string | null = null;

  for (const { di, off, p } of plays) {
    const st = p.start ?? {};
    const period = Number(pyOr(p.period?.number, 1));
    const clock = pyOr(p.clock?.displayValue, "0:00");
    const tl = String(p.type?.text ?? "").toLowerCase();
    if (period >= 3 && !in2h) { hTo = 3; aTo = 3; in2h = true; }

    const hb = off === r.home_id ? 1 : off === r.away_id ? 0 : 0.5;
    const isKick = tl.includes("kickoff");
    const offYtg = Number(pyOr(st.yardsToEndzone, 50));
    const isPass = tl.includes("pass") || tl.includes("sack") || tl.includes("interception") ? 1 : 0;
    const isRush = tl.includes("rush") ? 1 : 0;
    const down = Number(st.down ?? 0);

    rows.push({
      play_index: rows.length, period, half: period <= 2 ? 1 : 2,
      t_left: tLeftOf(period, clock), clock,
      home_lead: hp - ap, total_pts_before: hp + ap, hp, ap,
      home_has_ball: hb,
      // kickoffs are down=0 — that IS the kick-state flag derive_state reads
      down: isKick ? 0 : (down > 0 ? down : 1),
      distance: Number(pyOr(st.distance, 10)),
      home_ytg: hb === 1 ? offYtg : hb === 0 ? 100 - offYtg : 50,
      off_ytg: offYtg,
      home_to: hTo, away_to: aTo, recv_2h: recv2h, drive_no: di + 1,
      sit: st.downDistanceText ?? "", play_text: String(p.text ?? "").slice(0, 220),
      synthetic: 0, ...PLAY_COLS,
      yds: Number(p.statYardage ?? 0),
      is_scrim: isPass || isRush ? 1 : 0,
      is_pass: isPass,
      is_sack: tl.includes("sack") ? 1 : 0,
      is_to: tl.includes("interception") || tl.includes("fumble") ? 1 : 0,
      is_pen: tl.includes("penalty") ? 1 : 0,
    });

    // training flags the LAST play of the old possession
    if (prevOff !== null && off && off !== prevOff && rows.length >= 2) {
      rows[rows.length - 2].is_poss_chg = 1;
    }
    if (off) prevOff = off;

    const txt = String(p.text ?? "");
    if (tl.includes("timeout") || txt.toLowerCase().startsWith("timeout")) {
      const up = txt.toUpperCase();
      if ([...homeNames].some((n) => n && up.includes(n))) hTo = Math.max(hTo - 1, 0);
      else if ([...awayNames].some((n) => n && up.includes(n))) aTo = Math.max(aTo - 1, 0);
    }
    // scores are monotone; max() shrugs off unpopulated feed zeros
    if (p.homeScore != null) hp = Math.max(hp, Number(p.homeScore));
    if (p.awayScore != null) ap = Math.max(ap, Number(p.awayScore));
  }

  // Clocks are left exactly as ESPN stamped them. prepareCurve cleans them
  // once the scoreboard's current state is appended, because that state
  // votes on which stamps to trust.
  return rows;
}
