"""In-game state, taken to its raw-data limit.

Two products:

  features/play_raw.parquet    one row per play: the raw pre-snap situation
                               plus raw per-play EVENT columns. This is what
                               goes to Supabase — raw facts only.
  derive_state(df)             deterministic window aggregation over play_raw,
                               run at load time. Everything is computed from
                               plays STRICTLY BEFORE the current one, within
                               the same game — the in-game analogue of the
                               point-in-time rule.

What the state gains beyond score/clock/down/distance/timeouts:

  in-game performance   yards/play, pass share, explosiveness, turnovers,
                        sacks, penalties, third-down conversion — both teams,
                        so "trailing but outgaining 7-to-3" is visible
  points so far         the total head previously never saw combined scoring
  momentum              net scrimmage yards over the last 6 plays, lead change
                        over the last 10 plays
  drive context         play # in drive, yards this drive, drive start field
                        position
  pace / possessions    seconds per play so far, possession changes so far
  participation, live   has the season-to-date primary QB / primary rusher
                        appeared in THIS game yet — an injury or benching is
                        visible the moment the backup takes a snap
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.config import FEAT, PBP, PQ  # noqa: E402
from cfbpred.player_features import primary_names_pbp  # noqa: E402

RAW_OUT = FEAT / "play_raw.parquet"

BUILD = """
WITH g AS (
  SELECT game_id, season, week,
         CASE WHEN season_type = 'postseason' THEN week + 20 ELSE week END AS week_c,
         home_team, away_team
  FROM read_parquet('{games}')
  WHERE week IS NOT NULL
),
pr AS (SELECT * FROM primaries),
base AS (
  SELECT p.game_id, p."year" AS season, p.play_seq, p.game_play_number AS gpn,
         p.period AS period_src, p.adj_TimeSecsRem AS t_src,
         CASE WHEN p.pos_team = p.home THEN p.pos_team_score
              ELSE p.def_pos_team_score END AS h_after_src,
         CASE WHEN p.pos_team = p.away THEN p.pos_team_score
              ELSE p.def_pos_team_score END AS a_after_src,
         CASE WHEN p.pos_team = p.home THEN 1 ELSE 0 END AS home_has_ball,
         COALESCE(p.down, 0) AS down,
         COALESCE(p.distance, 10) AS distance,
         CASE WHEN p.pos_team = p.home THEN COALESCE(p.yards_to_goal, 50)
              ELSE 100 - COALESCE(p.yards_to_goal, 50) END AS home_ytg,
         COALESCE(p.yards_to_goal, 50) AS off_ytg,
         p.pos_team_timeouts_rem_before AS pos_to,
         p.def_pos_team_timeouts_rem_before AS def_to,
         CAST(p.pos_team_receives_2H_kickoff AS INT) AS recv_2h,
         CAST(COALESCE(p.Goal_To_Go, false) AS INT) AS goal_to_go,
         -- raw per-play events
         COALESCE(p.yards_gained, 0) AS yds,
         CAST(COALESCE(p.rush, false) OR COALESCE(p."pass", false) AS INT) AS is_scrim,
         CAST(COALESCE(p."pass", false) AS INT) AS is_pass,
         CAST(COALESCE(p.sack, false) AS INT) AS is_sack,
         CAST(COALESCE(p.turnover, false) AS INT) AS is_to,
         CAST(COALESCE(p.penalty_flag, false) AS INT) AS is_pen,
         CAST(COALESCE(p.change_of_pos_team, false) AS INT) AS is_poss_chg,
         COALESCE(p.drive_number, 0) AS drive_no,
         CAST(p.down = 3 AS INT) AS is_third,
         CAST(p.down = 3 AND p.yards_gained >= p.distance AS INT) AS is_third_conv,
         CAST((COALESCE(p.rush, false) OR COALESCE(p."pass", false))
              AND p.yards_gained >= 10 AS INT) AS is_expl10,
         CASE WHEN p.pos_team = p.home THEN 1 ELSE 0 END AS is_home_off,
         -- live participation: is THIS play run by the season-to-date primary?
         CAST(p.pos_team = p.home AND p.passer_player_name = pr_h.primary_qb
              AS INT) AS h_pqb_play,
         CAST(p.pos_team = p.away AND p.passer_player_name = pr_a.primary_qb
              AS INT) AS a_pqb_play,
         CAST(p.pos_team = p.home AND p.rusher_player_name = pr_h.primary_rb
              AS INT) AS h_prb_play,
         CAST(p.pos_team = p.away AND p.rusher_player_name = pr_a.primary_rb
              AS INT) AS a_prb_play,
         CAST(pr_h.primary_qb IS NOT NULL AS INT) AS h_pqb_known,
         CAST(pr_a.primary_qb IS NOT NULL AS INT) AS a_pqb_known
  FROM read_parquet('{pbp}') p
  JOIN g ON g.game_id = p.game_id
  LEFT JOIN pr pr_h ON pr_h.season = p."year" AND pr_h.week_c = g.week_c
                   AND pr_h.team = p.home
  LEFT JOIN pr pr_a ON pr_a.season = p."year" AND pr_a.week_c = g.week_c
                   AND pr_a.team = p.away
  WHERE p.period <= 4
    AND p.adj_TimeSecsRem IS NOT NULL
    AND p.pos_score_diff_start IS NOT NULL
)
SELECT * FROM base
ORDER BY game_id, gpn, play_seq
"""


def _longest_monotone_mask(vals, nonincreasing: bool) -> np.ndarray:
    """Indices of the longest non-decreasing (or non-increasing) subsequence.
    The rows on that chain are the game's self-consistent narrative; everything
    off it is treated as corrupt. Robust where greedy passes are not: a single
    end-of-game row misplaced at play 1 (observed: period-4, 42-39, t=97 rows
    numbered gpn 1-3) poisons a running-min clock check and a cumulative-max
    score repair for the WHOLE game, but it cannot outvote a 200-row chain."""
    import bisect
    x = [-float(v) for v in vals] if nonincreasing else [float(v) for v in vals]
    n = len(x)
    tails: list = []; tails_idx: list = []; prev = [-1] * n
    for i, val in enumerate(x):
        j = bisect.bisect_right(tails, val)
        if j == len(tails):
            tails.append(val); tails_idx.append(i)
        else:
            tails[j] = val; tails_idx[j] = i
        prev[i] = tails_idx[j - 1] if j > 0 else -1
    mask = np.zeros(n, bool)
    i = tails_idx[-1] if tails_idx else -1
    while i >= 0:
        mask[i] = True
        i = prev[i]
    return mask


def _repair_game(d: pd.DataFrame) -> pd.DataFrame:
    """Row-level repair of one game's feed. The source stamps some rows —
    kickoffs, sometimes whole opening drives — with RESET clocks (3600) and
    RESET scores (0-0), and neither play number nor timestamp ordering is
    reliable alone (measured: 976 of 1,657 games in 2025 are score-inconsistent
    under play-number order; the 2025 CFP semifinal froze its first 33 rows on
    three timestamps). Order by play number for narrative, then:

      scores  plausibility-gate (0..120, finite), then keep the longest
              non-decreasing chain of each side's score; carry forward from
              chain rows, lead/total derived from the repair
      clock   keep the longest non-increasing chain; linearly interpolate the
              rest between valid anchors (virtual 3600 before the first row)
      period  re-derived from the repaired clock
    """
    n = len(d)
    h = d["h_after_src"].to_numpy(float)
    a = d["a_after_src"].to_numpy(float)
    # The score columns are sometimes not scores at all: 1,218 rows in 230
    # games carry score-differential-like garbage (observed range -93..141;
    # a CFP first round stamped 6 / -3 on a whole mid-Q1 block). Negative
    # values are monotone-compatible with everything after them, so the chain
    # alone cannot reject them — plausibility must gate candidacy first.
    plaus = (np.isfinite(h) & np.isfinite(a)
             & (h >= 0) & (a >= 0) & (h <= 120) & (a <= 120))
    ok = np.zeros(n, bool)
    if plaus.any():
        sub = (_longest_monotone_mask(h[plaus], False)
               & _longest_monotone_mask(a[plaus], False))
        ok[np.where(plaus)[0][sub]] = True
    h = np.nan_to_num(h)
    a = np.nan_to_num(a)
    for name, v in (("h_after", h), ("a_after", a)):
        out = np.empty(n)
        last = 0.0
        for i in range(n):
            if ok[i]:
                last = v[i]
            out[i] = last
        d[name] = out

    t = d["t_src"].to_numpy(float)
    t = np.where(np.isnan(t), -1.0, t)
    rep = np.empty(n)
    in_range = (t >= 0.0) & (t <= 3600.0)
    valid = np.zeros(n, bool)
    if in_range.any():
        sub = _longest_monotone_mask(t[in_range], nonincreasing=True)
        valid[np.where(in_range)[0][sub]] = True
    idx = np.arange(n)
    if valid.any():
        anchors_x = np.concatenate([[-1.0], idx[valid].astype(float)])
        anchors_y = np.concatenate([[3600.0], t[valid]])
        rep = np.interp(idx, anchors_x, anchors_y)
        rep[valid] = t[valid]
    else:
        rep = np.linspace(3600, 0, n)

    # The feed only carries a real clock on SOME plays and repeats it between:
    # 62% of games have >30% of plays sharing a frozen value (a 2025 bowl put
    # 205 plays on 68 distinct clocks, runs up to 16). Frozen runs stack snaps
    # on one chart x and freeze the model's time features. Spread each equal
    # run linearly to its next distinct reading, treating a reading as true as
    # of the LAST play that carries it, with a virtual 3600 anchor so openings
    # with a lost clock still start at kickoff.
    knots = [i for i in range(n) if i == n - 1 or rep[i + 1] < rep[i]]
    kx = np.concatenate([[-1.0], np.array(knots, float)])
    ky = np.concatenate([[3600.0], rep[knots]])
    rep = np.interp(idx, kx, ky)
    d["t_left"] = rep
    d["period"] = np.clip(1 + np.floor((3600.0 - rep) / 900.0 + 1e-9),
                          1, 4).astype(int)
    d["half"] = np.where(d["period"] <= 2, 1, 2)
    d["half_t_left"] = np.where(d["half"] == 1, rep - 1800.0, rep).clip(0)
    d["under_two"] = (d["half_t_left"] <= 120).astype(int)

    h = d["h_after"].to_numpy(); a = d["a_after"].to_numpy()
    d["home_lead"] = np.concatenate([[0.0], (h - a)[:-1]])
    d["total_pts_before"] = np.concatenate([[0.0], (h + a)[:-1]])
    return d


def build_raw() -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    pr = primary_names_pbp(con)
    con.register("primaries", pr)
    df = con.execute(BUILD.format(games=PQ / "games.parquet",
                                  pbp=PBP / "pbp_*.parquet")).fetchdf()
    # play_seq is the ingest-time chronological key (period, clock DESC, play
    # number). Measured against period+play-number ordering across all 12,510
    # games: identical final-score agreement (99.22%) but 30% fewer impossible
    # adjacent-play lead jumps (277 vs 391), and it is the only ordering that
    # places blocks with scrambled play numbers (a CFP first round numbered a
    # mid-Q1 drive gpn 1-12) at their true position in the game.
    df = df.sort_values(["game_id", "play_seq"],
                        na_position="last").reset_index(drop=True)
    # drop byte-identical replays of the same play number
    df = df.drop_duplicates(
        ["game_id", "gpn", "t_src", "down", "distance", "yds",
         "h_after_src", "a_after_src", "is_scrim", "is_pass"],
        keep="first").reset_index(drop=True)
    df = (df.groupby("game_id", group_keys=False, sort=False)
          .apply(_repair_game))
    df = df.drop(columns=["h_after_src", "a_after_src", "t_src",
                          "period_src", "gpn", "h_after", "a_after"])
    # play order IS the repair order — the repaired clock is monotone under it
    # by construction; re-sorting by clock would scramble ties (timeouts).
    df = df.reset_index(drop=True)
    df["play_index"] = df.groupby("game_id").cumcount()
    for c in ("pos_to", "def_to"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(3)
    df["home_to"] = np.where(df.home_has_ball == 1, df.pos_to, df.def_to)
    df["away_to"] = np.where(df.home_has_ball == 1, df.def_to, df.pos_to)
    df = df.drop(columns=["pos_to", "def_to"])
    df.to_parquet(RAW_OUT, index=False, compression="zstd")
    print(f"play_raw: {len(df):,} rows x {df.shape[1]} cols "
          f"({RAW_OUT.stat().st_size / 1e6:.0f} MB)")
    return df


# ------------------------------------------------------------- derivation

def _shifted_cum(g: pd.core.groupby.DataFrameGroupBy, col: pd.Series):
    return g[col.name].cumsum().shift(1) if False else None


# Columns derivable from the stored ones; reconstructed here so the Supabase
# table can stay lean (they cost ~60 MB of tier budget stored explicitly).
def _reconstruct(df: pd.DataFrame) -> pd.DataFrame:
    if "is_home_off" not in df.columns:
        df["is_home_off"] = df["home_has_ball"]
    if "half_t_left" not in df.columns:
        df["half_t_left"] = np.where(df["half"] == 1,
                                     df["t_left"] - 1800, df["t_left"])
    if "under_two" not in df.columns:
        df["under_two"] = (df["half_t_left"] <= 120).astype(int)
    if "goal_to_go" not in df.columns:
        df["goal_to_go"] = (df["distance"] >= df["off_ytg"]).astype(int)
    if "is_third" not in df.columns:
        df["is_third"] = (df["down"] == 3).astype(int)
    if "is_third_conv" not in df.columns:
        df["is_third_conv"] = ((df["down"] == 3)
                               & (df["yds"] >= df["distance"])).astype(int)
    if "is_expl10" not in df.columns:
        df["is_expl10"] = ((df["is_scrim"] == 1) & (df["yds"] >= 10)).astype(int)
    return df


def derive_state(df: pd.DataFrame) -> pd.DataFrame:
    """Window aggregation over play_raw — strictly prior plays, same game.
    Deterministic, so it runs at load time rather than being stored."""
    df = _reconstruct(df.copy())
    df = df.sort_values(["game_id", "play_index"]).reset_index(drop=True)
    g = df.groupby("game_id", sort=False)

    def cum_prior(mask_col: str, val: pd.Series | None = None) -> np.ndarray:
        """Sum of `val` (or the 0/1 column itself) over prior plays."""
        v = df[mask_col].to_numpy(float) if val is None else val
        s = pd.Series(v).groupby(df["game_id"], sort=False)
        return (s.cumsum() - v).to_numpy()   # exclude the current play

    ihо = df["is_home_off"].to_numpy(float)
    scrim = df["is_scrim"].to_numpy(float)
    yds = df["yds"].to_numpy(float)

    h_scrim = cum_prior("is_scrim", scrim * ihо)
    a_scrim = cum_prior("is_scrim", scrim * (1 - ihо))
    h_yds = cum_prior("yds", yds * scrim * ihо)
    a_yds = cum_prior("yds", yds * scrim * (1 - ihо))
    h_pass = cum_prior("is_pass", df.is_pass.to_numpy(float) * ihо)
    a_pass = cum_prior("is_pass", df.is_pass.to_numpy(float) * (1 - ihо))
    h_expl = cum_prior("is_expl10", df.is_expl10.to_numpy(float) * ihо)
    a_expl = cum_prior("is_expl10", df.is_expl10.to_numpy(float) * (1 - ihо))
    h_to = cum_prior("is_to", df.is_to.to_numpy(float) * ihо)
    a_to = cum_prior("is_to", df.is_to.to_numpy(float) * (1 - ihо))
    h_sacked = cum_prior("is_sack", df.is_sack.to_numpy(float) * ihо)
    a_sacked = cum_prior("is_sack", df.is_sack.to_numpy(float) * (1 - ihо))
    h_pen = cum_prior("is_pen", df.is_pen.to_numpy(float) * ihо)
    a_pen = cum_prior("is_pen", df.is_pen.to_numpy(float) * (1 - ihо))
    h_3a = cum_prior("is_third", df.is_third.to_numpy(float) * ihо)
    a_3a = cum_prior("is_third", df.is_third.to_numpy(float) * (1 - ihо))
    h_3c = cum_prior("is_third_conv", df.is_third_conv.to_numpy(float) * ihо)
    a_3c = cum_prior("is_third_conv", df.is_third_conv.to_numpy(float) * (1 - ihо))
    poss = cum_prior("is_poss_chg")
    plays = g.cumcount().to_numpy(float)

    eps = 3.0
    out = pd.DataFrame(index=df.index)
    out["ig_h_ypp"] = h_yds / (h_scrim + eps) / 6.0
    out["ig_a_ypp"] = a_yds / (a_scrim + eps) / 6.0
    out["ig_ypp_diff"] = out["ig_h_ypp"] - out["ig_a_ypp"]
    out["ig_h_pass_share"] = h_pass / (h_scrim + eps)
    out["ig_a_pass_share"] = a_pass / (a_scrim + eps)
    out["ig_h_expl"] = h_expl / (h_scrim + eps) * 4.0
    out["ig_a_expl"] = a_expl / (a_scrim + eps) * 4.0
    out["ig_h_to"] = h_to / 3.0
    out["ig_a_to"] = a_to / 3.0
    out["ig_to_diff"] = (a_to - h_to) / 3.0
    out["ig_h_sacked"] = h_sacked / 4.0
    out["ig_a_sacked"] = a_sacked / 4.0
    out["ig_h_pen"] = h_pen / 6.0
    out["ig_a_pen"] = a_pen / 6.0
    out["ig_h_third"] = h_3c / (h_3a + eps)
    out["ig_a_third"] = a_3c / (a_3a + eps)
    out["ig_plays"] = plays / 120.0
    out["ig_poss"] = poss / 24.0

    # points so far — the total head's missing signal
    home_after = np.where(ihо == 1,
                          df.get("home_pts_after", pd.Series(np.nan)).to_numpy(float)
                          if "home_pts_after" in df.columns else np.nan, np.nan)
    # reconstruct from lead + scoring: total so far comes from lead & scores —
    # play_raw carries home_lead only, so carry total via cumulative TD/FG is
    # lossy; instead use the raw score columns captured below.
    out["ig_total_pts"] = df["total_pts_before"].to_numpy(float) / 40.0 \
        if "total_pts_before" in df.columns else 0.0

    # pace: seconds burned per play so far
    elapsed = 3600.0 - df["t_left"].to_numpy(float)
    out["ig_pace"] = (elapsed / np.maximum(plays, 1.0)) / 40.0

    # momentum: net scrimmage yards, last 6 plays; lead change, last 10 plays
    signed = np.where(scrim == 1, yds, 0.0) * np.where(ihо == 1, 1.0, -1.0)
    s = pd.Series(signed)
    out["ig_net6"] = (s.groupby(df["game_id"], sort=False)
                      .rolling(6, min_periods=1).sum()
                      .reset_index(drop=True)
                      .groupby(df["game_id"], sort=False).shift(1)
                      .fillna(0).to_numpy()) / 40.0
    lead = df["home_lead"].to_numpy(float)
    lead10 = (pd.Series(lead).groupby(df["game_id"], sort=False)
              .shift(10))
    out["ig_lead_d10"] = (lead - lead10.fillna(pd.Series(lead)).to_numpy()) / 14.0

    # drive context
    gd = df.groupby(["game_id", "drive_no"], sort=False)
    out["ig_drive_play"] = gd.cumcount().to_numpy(float) / 12.0
    dy = pd.Series(np.where(scrim == 1, yds, 0.0))
    out["ig_drive_yds"] = ((dy.groupby([df["game_id"], df["drive_no"]],
                                       sort=False).cumsum() - dy)
                           .to_numpy()) / 60.0
    out["ig_drive_start"] = (gd["off_ytg"].transform("first")
                             .to_numpy(float)) / 100.0
    out["ig_red_zone"] = (df["off_ytg"].to_numpy(float) <= 20).astype(float)

    # live participation: has the primary appeared in THIS game so far?
    for c, flag in [("h_pqb_play", "ig_h_pqb_seen"),
                    ("a_pqb_play", "ig_a_pqb_seen"),
                    ("h_prb_play", "ig_h_prb_seen"),
                    ("a_prb_play", "ig_a_prb_seen")]:
        v = df[c].fillna(0).to_numpy(float)
        cm = pd.Series(v).groupby(df["game_id"], sort=False).cummax()
        out[flag] = (cm.groupby(df["game_id"], sort=False).shift(1)
                     .fillna(0).to_numpy())
    out["ig_h_pqb_known"] = df["h_pqb_known"].fillna(0).to_numpy(float)
    out["ig_a_pqb_known"] = df["a_pqb_known"].fillna(0).to_numpy(float)

    out["ig_half_t"] = df["half_t_left"].to_numpy(float) / 1800.0
    out["ig_is_kick_state"] = (df["down"].to_numpy(float) == 0).astype(float)
    return pd.concat([df, out], axis=1)


IG_FEATURES = [
    "ig_h_ypp", "ig_a_ypp", "ig_ypp_diff", "ig_h_pass_share",
    "ig_a_pass_share", "ig_h_expl", "ig_a_expl", "ig_h_to", "ig_a_to",
    "ig_to_diff", "ig_h_sacked", "ig_a_sacked", "ig_h_pen", "ig_a_pen",
    "ig_h_third", "ig_a_third", "ig_plays", "ig_poss", "ig_total_pts",
    "ig_pace", "ig_net6", "ig_lead_d10", "ig_drive_play", "ig_drive_yds",
    "ig_drive_start", "ig_red_zone", "ig_h_pqb_seen", "ig_a_pqb_seen",
    "ig_h_prb_seen", "ig_a_prb_seen", "ig_h_pqb_known", "ig_a_pqb_known",
    "ig_half_t", "ig_is_kick_state",
]


if __name__ == "__main__":
    df = build_raw()
    d = derive_state(df)
    print(f"derived state: {len(IG_FEATURES)} in-game features")
    smp = d[d.game_id == d.game_id.iloc[100]]
    print("sanity, one game (last 3 plays):")
    print(smp[["play_index", "t_left", "home_lead", "ig_h_ypp", "ig_a_ypp",
               "ig_total_pts", "ig_poss", "ig_h_pqb_seen"]].tail(3).to_string())
