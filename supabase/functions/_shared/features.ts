/**
 * `derive_state` + `prep_play`, ported for one game at a time.
 *
 * The Python版 groups by game_id; here every call handles a single game's
 * play sequence, so the group-bys collapse to running scans. Graded against
 * the frozen fixtures by verify_features.ts — every number here has to match
 * pandas to 1e-9 or the edge function is not the scorer.
 */

export type PlayRow = {
  play_index: number;
  period: number; half: number;
  t_left: number; clock: string;
  home_lead: number; total_pts_before: number;
  hp: number; ap: number;
  home_has_ball: number;
  down: number; distance: number;
  home_ytg: number; off_ytg: number;
  home_to: number; away_to: number;
  recv_2h: number; drive_no: number;
  sit: string; play_text: string;
  synthetic: number;
  yds: number; is_scrim: number; is_pass: number; is_sack: number;
  is_to: number; is_pen: number; is_poss_chg: number;
  h_pqb_play: number; a_pqb_play: number; h_prb_play: number; a_prb_play: number;
  h_pqb_known: number; a_pqb_known: number;
};

/** Order is load-bearing: it must equal weights.json's state_features. */
export const STATE_FEATURES = [
  "lead_scaled", "lead_over_sqrt_t", "lead_x_elapsed", "frac_elapsed",
  "t_left_scaled", "sqrt_t", "home_has_ball", "home_has_ball_x_late",
  "down_scaled", "log_dist", "home_ytg_scaled", "home_to_scaled",
  "away_to_scaled", "recv_2h", "under_two", "goal_to_go", "lead_over_t",
  "under_one_min", "t_lt_10",
  "ig_h_ypp", "ig_a_ypp", "ig_ypp_diff", "ig_h_pass_share", "ig_a_pass_share",
  "ig_h_expl", "ig_a_expl", "ig_h_to", "ig_a_to", "ig_to_diff", "ig_h_sacked",
  "ig_a_sacked", "ig_h_pen", "ig_a_pen", "ig_h_third", "ig_a_third",
  "ig_plays", "ig_poss", "ig_total_pts", "ig_pace", "ig_net6", "ig_lead_d10",
  "ig_drive_play", "ig_drive_yds", "ig_drive_start", "ig_red_zone",
  "ig_h_pqb_seen", "ig_a_pqb_seen", "ig_h_prb_seen", "ig_a_prb_seen",
  "ig_h_pqb_known", "ig_a_pqb_known", "ig_half_t", "ig_is_kick_state",
] as const;

const EPS = 3.0;

/** Sum over strictly prior rows — pandas' `cumsum() - v`. */
function priorSum(v: Float64Array): Float64Array {
  const out = new Float64Array(v.length);
  let run = 0;
  for (let i = 0; i < v.length; i++) { out[i] = run; run += v[i]; }
  return out;
}

/**
 * The 53-column state matrix for one game's plays, in STATE_FEATURES order.
 * Rows must already be in play_index order.
 */
export function deriveState(rows: PlayRow[]): Float64Array[] {
  const n = rows.length;
  const col = (f: (r: PlayRow) => number) => {
    const a = new Float64Array(n);
    for (let i = 0; i < n; i++) a[i] = f(rows[i]);
    return a;
  };
  const mul = (a: Float64Array, b: Float64Array) => {
    const o = new Float64Array(n);
    for (let i = 0; i < n; i++) o[i] = a[i] * b[i];
    return o;
  };

  // --- _reconstruct
  const ih = col((r) => r.home_has_ball);
  const away = new Float64Array(n);
  for (let i = 0; i < n; i++) away[i] = 1 - ih[i];
  const tLeft = col((r) => r.t_left);
  const halfT = col((r) => (r.half === 1 ? r.t_left - 1800 : r.t_left));
  const underTwo = col((r, ) => 0);
  for (let i = 0; i < n; i++) underTwo[i] = halfT[i] <= 120 ? 1 : 0;
  const distance = col((r) => r.distance);
  const offYtg = col((r) => r.off_ytg);
  const goalToGo = new Float64Array(n);
  for (let i = 0; i < n; i++) goalToGo[i] = distance[i] >= offYtg[i] ? 1 : 0;
  const down = col((r) => r.down);
  const yds = col((r) => r.yds);
  const scrim = col((r) => r.is_scrim);
  const isThird = new Float64Array(n);
  const isThirdConv = new Float64Array(n);
  const isExpl10 = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    isThird[i] = down[i] === 3 ? 1 : 0;
    isThirdConv[i] = down[i] === 3 && yds[i] >= distance[i] ? 1 : 0;
    isExpl10[i] = scrim[i] === 1 && yds[i] >= 10 ? 1 : 0;
  }

  // --- cumulative prior aggregates
  const hScrim = priorSum(mul(scrim, ih)), aScrim = priorSum(mul(scrim, away));
  const sy = mul(yds, scrim);
  const hYds = priorSum(mul(sy, ih)), aYds = priorSum(mul(sy, away));
  const isPass = col((r) => r.is_pass);
  const hPass = priorSum(mul(isPass, ih)), aPass = priorSum(mul(isPass, away));
  const hExpl = priorSum(mul(isExpl10, ih)), aExpl = priorSum(mul(isExpl10, away));
  const isTo = col((r) => r.is_to);
  const hTo = priorSum(mul(isTo, ih)), aTo = priorSum(mul(isTo, away));
  const isSack = col((r) => r.is_sack);
  const hSack = priorSum(mul(isSack, ih)), aSack = priorSum(mul(isSack, away));
  const isPen = col((r) => r.is_pen);
  const hPen = priorSum(mul(isPen, ih)), aPen = priorSum(mul(isPen, away));
  const h3a = priorSum(mul(isThird, ih)), a3a = priorSum(mul(isThird, away));
  const h3c = priorSum(mul(isThirdConv, ih)), a3c = priorSum(mul(isThirdConv, away));
  const poss = priorSum(col((r) => r.is_poss_chg));

  // --- momentum: rolling 6 of signed scrimmage yards, then shifted one back
  const signed = new Float64Array(n);
  for (let i = 0; i < n; i++) signed[i] = (scrim[i] === 1 ? yds[i] : 0) * (ih[i] === 1 ? 1 : -1);
  const roll6 = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = Math.max(0, i - 5); k <= i; k++) s += signed[k];
    roll6[i] = s;
  }
  const lead = col((r) => r.home_lead);

  // --- drive context: cumcount / prior-sum / first, within (game, drive_no)
  const drivePlay = new Float64Array(n);
  const driveYds = new Float64Array(n);
  const driveStart = new Float64Array(n);
  {
    const seen = new Map<number, number>();
    const acc = new Map<number, number>();
    const first = new Map<number, number>();
    for (let i = 0; i < n; i++) {
      const d = rows[i].drive_no;
      const c = seen.get(d) ?? 0;
      drivePlay[i] = c;
      seen.set(d, c + 1);
      const a = acc.get(d) ?? 0;
      driveYds[i] = a;
      acc.set(d, a + (scrim[i] === 1 ? yds[i] : 0));
      if (!first.has(d)) first.set(d, offYtg[i]);
      driveStart[i] = first.get(d)!;
    }
  }

  // --- participation: cummax of "seen", shifted one back
  const seenFlag = (f: (r: PlayRow) => number) => {
    const o = new Float64Array(n);
    let m = 0;
    for (let i = 0; i < n; i++) { o[i] = i === 0 ? 0 : m; m = Math.max(m, f(rows[i])); }
    return o;
  };
  const hQb = seenFlag((r) => r.h_pqb_play ?? 0);
  const aQb = seenFlag((r) => r.a_pqb_play ?? 0);
  const hRb = seenFlag((r) => r.h_prb_play ?? 0);
  const aRb = seenFlag((r) => r.a_prb_play ?? 0);

  const out: Float64Array[] = [];
  for (let i = 0; i < n; i++) {
    const r = rows[i];
    const t = tLeft[i], ld = lead[i], plays = i;           // cumcount within game
    const fracElapsed = (3600 - t) / 3600;
    const v = new Float64Array(STATE_FEATURES.length);
    let k = 0;
    v[k++] = ld / 21;                                             // lead_scaled
    v[k++] = ld / Math.sqrt(Math.max(t, 0) + 1) / 3;              // lead_over_sqrt_t
    v[k++] = (ld / 21) * fracElapsed;                             // lead_x_elapsed
    v[k++] = fracElapsed;
    v[k++] = t / 3600;                                            // t_left_scaled
    v[k++] = Math.sqrt(Math.max(t, 0)) / 60;                      // sqrt_t
    v[k++] = r.home_has_ball;
    v[k++] = r.home_has_ball * (1 - fracElapsed);                 // ..._x_late
    v[k++] = r.down / 4;                                          // down_scaled
    v[k++] = Math.log1p(Math.min(Math.max(r.distance, 0), 40)) / 3; // log_dist
    v[k++] = r.home_ytg / 100;
    v[k++] = r.home_to / 3;
    v[k++] = r.away_to / 3;
    v[k++] = r.recv_2h;
    v[k++] = underTwo[i];
    v[k++] = goalToGo[i];
    v[k++] = Math.min(Math.max(ld / (t / 60 + 0.25), -4), 4) / 4; // lead_over_t
    v[k++] = Math.min(t, t > 1800 ? t - 1800 : t) <= 60 ? 1 : 0;  // under_one_min
    v[k++] = t <= 10 ? 1 : 0;                                     // t_lt_10

    const hYpp = hYds[i] / (hScrim[i] + EPS) / 6;
    const aYpp = aYds[i] / (aScrim[i] + EPS) / 6;
    v[k++] = hYpp;
    v[k++] = aYpp;
    v[k++] = hYpp - aYpp;
    v[k++] = hPass[i] / (hScrim[i] + EPS);
    v[k++] = aPass[i] / (aScrim[i] + EPS);
    v[k++] = hExpl[i] / (hScrim[i] + EPS) * 4;
    v[k++] = aExpl[i] / (aScrim[i] + EPS) * 4;
    v[k++] = hTo[i] / 3;
    v[k++] = aTo[i] / 3;
    v[k++] = (aTo[i] - hTo[i]) / 3;
    v[k++] = hSack[i] / 4;
    v[k++] = aSack[i] / 4;
    v[k++] = hPen[i] / 6;
    v[k++] = aPen[i] / 6;
    v[k++] = h3c[i] / (h3a[i] + EPS);
    v[k++] = a3c[i] / (a3a[i] + EPS);
    v[k++] = plays / 120;                                         // ig_plays
    v[k++] = poss[i] / 24;
    v[k++] = r.total_pts_before / 40;
    v[k++] = ((3600 - t) / Math.max(plays, 1)) / 40;              // ig_pace
    v[k++] = (i === 0 ? 0 : roll6[i - 1]) / 40;                   // ig_net6, shifted
    v[k++] = (ld - (i >= 10 ? lead[i - 10] : ld)) / 14;           // ig_lead_d10
    v[k++] = drivePlay[i] / 12;
    v[k++] = driveYds[i] / 60;
    v[k++] = driveStart[i] / 100;
    v[k++] = offYtg[i] <= 20 ? 1 : 0;                             // ig_red_zone
    v[k++] = hQb[i];
    v[k++] = aQb[i];
    v[k++] = hRb[i];
    v[k++] = aRb[i];
    v[k++] = r.h_pqb_known ?? 0;
    v[k++] = r.a_pqb_known ?? 0;
    v[k++] = halfT[i] / 1800;                                     // ig_half_t
    v[k++] = down[i] === 0 ? 1 : 0;                               // ig_is_kick_state
    for (let j = 0; j < v.length; j++) if (!Number.isFinite(v[j])) v[j] = 0;
    out.push(v);
  }

  // lone synthetic rows have no prior plays, so the pace/plays ratios explode;
  // the scorer substitutes college norms before inference and so must this
  for (let i = 0; i < n; i++) {
    if (rows[i].synthetic !== 1) continue;
    const el = 3600 - rows[i].t_left;
    out[i][STATE_FEATURES.indexOf("ig_plays")] = el / 26 / 120;
    out[i][STATE_FEATURES.indexOf("ig_poss")] = el / 160 / 24;
    out[i][STATE_FEATURES.indexOf("ig_pace")] = el > 0 ? 26 / 40 : 0;
  }
  return out;
}
