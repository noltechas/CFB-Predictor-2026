import { cleanClock, LiveState, synthNowRow } from "./espn.ts";
import { PlayRow } from "./features.ts";

/** Add the observed current/final state after the states before each play.
 * A missing or truncated feed must never masquerade as a complete history.
 * Clocks are cleaned here, with the observed state voting, never earlier. */
export function prepareCurve(rows: PlayRow[], state: LiveState): PlayRow[] {
  if (!rows.length) return [];
  const prows = rows.map(p => ({ ...p }));
  const last = prows[prows.length - 1];
  let observed = false;
  if (state.state === "in" || (state.state === "post" && last.period >= 4)) {
    const now = synthNowRow(state, last.drive_no, prows[0].recv_2h, 0, last.home_to, last.away_to);
    if (state.state === "post") {
      now.t_left = 0; now.clock = "FINAL"; now.sit = "Final";
      // Some final summary headers report Q4 even after multiple overtimes.
      now.period = Math.max(now.period, last.period);
    }
    prows.push(now);
    observed = true;
  }
  cleanClock(prows, observed);
  for (let i = prows.length - 1; i > 0; i--) prows[i].play_text = prows[i - 1].play_text;
  prows[0].play_text = "";
  prows.forEach((p, i) => { p.play_index = i; });
  return prows;
}

export function finalHistoryComplete(state: LiveState, rows: PlayRow[]): boolean {
  const last = rows.at(-1);
  return state.state === "post" && !!last && last.clock === "FINAL"
    && last.t_left === 0 && last.hp === state.home_points && last.ap === state.away_points;
}

export function resumeCurve(previous: { max: number; n: number } | undefined, length: number,
  reset: boolean, live: boolean): { full: boolean; from: number } {
  const full = reset || !previous || previous.n > length || previous.n !== previous.max + 1;
  // Replace the temporary current-state point when the next real play arrives.
  return { full, from: full ? 0 : Math.max(0, previous!.max + (live ? 0 : 1)) };
}

/** What a stored curve holds per point, in play_index order. */
export type StoredCurve = {
  t_left: number[]; period: number[]; home_points: number[]; away_points: number[];
};

/** Seconds a stored point may drift before it is re-priced (~1 px of chart). */
export const RETIME_TOLERANCE = 15;

/**
 * Stored points before `upto` that no longer match the fresh parse. Later
 * plays legitimately move earlier ones: a batch-stamped drive is re-spread
 * once its real end arrives, and a bad stamp is out-voted by the plays after
 * it. Every state feature reads only its own row's clock, so re-pricing just
 * these rows is exact. A point inside the tolerance is still stale if it
 * would sit out of order next to its re-priced neighbours; otherwise small
 * drifts written on different ticks leave the line stepping backward.
 */
export function staleRows(stored: StoredCurve | undefined, rows: PlayRow[], upto: number,
  tol = RETIME_TOLERANCE): number[] {
  if (!stored) return [];
  const out: number[] = [];
  const n = Math.min(upto, rows.length, stored.t_left?.length ?? 0);
  const eps = 0.01;                                  // real (float4) round-trip
  for (let i = 0; i < n; i++) {
    const r = rows[i], t = stored.t_left[i];
    if (!(Math.abs(t - r.t_left) <= tol) || stored.period[i] !== r.period
      || stored.home_points[i] !== r.hp || stored.away_points[i] !== r.ap
      || (i > 0 && t > rows[i - 1].t_left + eps)
      || (i + 1 < rows.length && t < rows[i + 1].t_left - eps)) out.push(i);
  }
  return out;
}
