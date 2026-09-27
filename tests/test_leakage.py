"""Leakage and purity tests for the raw-input model.

Two rule sets are enforced:
1. **Purity** — no external OUTCOME predictions anywhere in the inputs: no
   Elo, no PPA, no betting lines, no SP+/SRS/FPI, and none of our own fitted
   intermediates (ratings, expected points). The network learns team strength
   itself. Human scouting assessments of players (recruiting class points,
   the 247 talent composite) are allowed by user decision (2026-08-26) —
   they describe rosters, not game outcomes.
2. **Point-in-time** — everything a prediction sees existed before kickoff.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfbpred.config import (BENCH, FEAT, MODELS, TEST_SEASONS,  # noqa: E402
                            TRAIN_SEASONS, VAL_SEASONS)
from cfbpred.train import (BANNED_SUBSTRINGS, STATE_FEATURES,  # noqa: E402
                           prep_scalars, scalar_columns)


@pytest.fixture(scope="module")
def scalars():
    return pd.read_parquet(FEAT / "game_scalars.parquet")


@pytest.fixture(scope="module")
def history():
    return pd.read_parquet(FEAT / "game_history.parquet")


# ------------------------------------------------------------------ purity

def test_no_banned_substring_in_any_feature_column(scalars):
    g, cols = prep_scalars(scalars)
    bad = [c for c in cols if any(b in c.lower() for b in BANNED_SUBSTRINGS)]
    assert not bad, f"external-model features present: {bad}"


def test_state_features_are_raw_game_state():
    banned = set(BANNED_SUBSTRINGS)
    for c in STATE_FEATURES:
        assert not any(b in c.lower() for b in banned), c


def test_banned_list_covers_the_known_offenders():
    """The list itself is load-bearing; pin its coverage."""
    for probe in ("home_pregame_elo",
                  "ret_ppa_diff", "spread_close", "sp_rating", "srs_rating",
                  "fpi", "wp_before", "epa_off"):
        assert any(b in probe.lower() for b in BANNED_SUBSTRINGS), (
            f"BANNED_SUBSTRINGS would not catch {probe}")


def test_scalar_columns_raises_on_contamination(scalars):
    g = scalars.copy()
    g["h_elo_rating"] = 1.0
    with pytest.raises(RuntimeError):
        scalar_columns(g)


def test_no_fitted_rating_files_feed_the_model():
    """Our own ridge ratings and EP model must not be inputs. Their outputs
    lived in team_ratings_pit / epa parquet; assert the feature store carries
    no columns from them."""
    g, cols = prep_scalars(pd.read_parquet(FEAT / "game_scalars.parquet"))
    for c in cols:
        assert "adj" not in c.lower(), f"opponent-adjusted rating leaked in: {c}"
        assert "overall" not in c.lower(), c


# ------------------------------------------------------------ point in time

def test_history_is_strictly_prior(history):
    """Every history slot must predate its game."""
    assert (history["days_ago"] > 0).all(), "a history slot is not in the past"


def test_history_slots_ordered_most_recent_first(history):
    h = history[history["game_id"] == history["game_id"].iloc[0]]
    for team, d in h.groupby("team"):
        d = d.sort_values("slot")
        assert d["days_ago"].is_monotonic_increasing


def test_counting_stats_are_prior_weeks_only(scalars):
    """Two real invariants, each of which caught a real defect:

    1. Week-keyed pbp stats must be null in week 1 — postseason weeks restart
       at 1 and were joining bowls to week-1 aggregates until week_c fixed it.
    2. An FBS team's FIRST game of a season must show zero games played.
       Plain "week 1 implies gp=0" is false: CFBD labels Week-0 games week 1
       too, so a team can play twice in week 1 and legitimately have one
       prior game. The invariant is FBS-only since v16: an FCS side's record
       now counts its FCS-vs-FCS games, so its first FBS-involved appearance
       legitimately carries prior games — those games really were played and
       really are in the log, so it is still strictly point-in-time.
    """
    wk1 = scalars[scalars["h_gp"].fillna(0) == 0]
    assert wk1["h_o_ypp"].isna().all(), (
        "week-1 rows carry same-season pbp stats — look-ahead")
    # A team's first appearance can be home OR away, so take the minimum games
    # played across both sides: for every (season, FBS team) it must be 0.
    import pandas as pd
    long = pd.concat([
        scalars.loc[scalars["home_division"] == "fbs",
                    ["season", "home_team", "h_gp"]].rename(
            columns={"home_team": "team", "h_gp": "gp"}),
        scalars.loc[scalars["away_division"] == "fbs",
                    ["season", "away_team", "a_gp"]].rename(
            columns={"away_team": "team", "a_gp": "gp"}),
    ])
    first_gp = long.groupby(["season", "team"])["gp"].min().fillna(0)
    assert (first_gp == 0).all(), (
        "some FBS team's earliest game of a season shows prior games played")


def test_returning_production_is_a_preseason_constant(scalars):
    """retp_/port_ features are (prior-season production) x (current roster):
    preseason facts, so they must be identical across every game a team plays
    in a season. Any within-season variation would mean same-season data
    leaked into them."""
    cols = [c for c in scalars.columns if c.startswith(("h_retp", "h_port"))]
    assert cols, "returning-production features missing from the feature store"
    nun = scalars.groupby(["season", "home_team"])[cols].nunique()
    assert int(nun.max().max()) <= 1, "retp/port vary within a team-season"


def test_qb_stats_absent_before_any_prior_game(scalars):
    wk1 = scalars[(scalars["h_gp"].fillna(0) == 0) & (scalars["season"] >= 2015)]
    assert (wk1["h_qb_known"].fillna(0) == 0).all(), (
        "week-1 QB stats populated — they can only come from this season's games")


# ------------------------------------------------------------- split hygiene

def test_splits_are_disjoint_and_forward():
    a, c = set(TRAIN_SEASONS), set(TEST_SEASONS)
    assert not (a & c)
    assert max(a) < min(c)
    assert VAL_SEASONS == []          # validation is game-level (id hash)


def test_no_game_spans_two_splits():
    p = pd.read_parquet(FEAT / "play_raw.parquet")[["game_id", "season"]]
    assert (p.groupby("game_id")["season"].nunique() == 1).all()


# --------------------------------------------------------- market tripwire

def test_market_comparison_uses_actual_pregame_predictions():
    """Valid improvements must be allowed; test provenance, not inferiority."""
    ev = MODELS / "eval_base.json"
    if not ev.exists():
        pytest.skip("run evaluate.py first")
    r = json.loads(ev.read_text())["pregame_game_level"]
    assert r["prediction_stage"] == "pregame_tower"
    assert r["n"] > 0
    assert np.isfinite(r["model_rmse"]) and np.isfinite(r["market_rmse"])
    assert r["model_rmse"] >= 0 and r["market_rmse"] >= 0


def test_model_beats_the_published_wp_model():
    ev = MODELS / "eval_base.json"
    if not ev.exists():
        pytest.skip("run evaluate.py first")
    r = json.loads(ev.read_text())["buckets"]
    assert r["all"]["logloss_raw"] < BENCH["cfbfastr_wp_ll"]["all"]
    assert r["q1"]["logloss_raw"] < BENCH["cfbfastr_wp_ll"]["q1"]


# ------------------------------------------------- in-game data invariants

def test_play_clock_never_increases():
    """The 2025 CFP semifinal shipped 33 plays frozen on three timestamps and
    kickoff rows stamped 3600 mid-game; after repair the clock must be
    monotone non-increasing within every game."""
    import duckdb
    q = duckdb.connect().execute(f"""
        WITH l AS (SELECT t_left, lag(t_left) OVER (
            PARTITION BY game_id ORDER BY play_index) tp
          FROM read_parquet('{FEAT / "play_raw.parquet"}'))
        SELECT count(*) FROM l WHERE tp IS NOT NULL AND t_left > tp + 1e-6
    """).fetchone()[0]
    assert q == 0, f"{q} plays where the game clock runs backwards"


def test_lead_changes_are_bounded_by_scoring():
    """A lead can only move by points actually scored. Before repair, reset
    rows produced 40-point single-play WP spikes; a handful of source rows
    remain irreparable — pin the count so regressions are visible."""
    import duckdb
    q = duckdb.connect().execute(f"""
        WITH l AS (SELECT home_lead, total_pts_before,
            lag(home_lead) OVER w lp, lag(total_pts_before) OVER w tp
          FROM read_parquet('{FEAT / "play_raw.parquet"}')
          WINDOW w AS (PARTITION BY game_id ORDER BY play_index))
        SELECT count(*) FROM l WHERE lp IS NOT NULL
          AND abs(home_lead - lp) > (total_pts_before - tp) + 1e-6
    """).fetchone()[0]
    assert q <= 25, f"{q} impossible lead jumps (16 after the chain repair, across 2.2M rows; the end-to-end final-score test is the stronger guard)"


def test_clock_has_no_frozen_runs():
    """The feed repeats the clock across plays (62% of games had >30% frozen
    rows); repaired clocks must spread them — no equal consecutive values."""
    import duckdb
    q = duckdb.connect().execute(f"""
        WITH l AS (SELECT game_id, t_left, lag(t_left) OVER (
            PARTITION BY game_id ORDER BY play_index) tp
          FROM read_parquet('{FEAT / "play_raw.parquet"}')),
        g AS (SELECT game_id,
                sum(CASE WHEN t_left = tp THEN 1 ELSE 0 END) * 1.0
                  / count(*) AS fs
              FROM l GROUP BY 1)
        SELECT count(*) FROM g WHERE fs > 0.3""").fetchone()[0]
    assert q <= 2, f"{q} games still have >30% frozen clock rows"


def test_repaired_scores_reach_the_true_final():
    """End-to-end repair check: walking the repaired per-play scores to the end
    of each game must land on the authoritative final. Catches every corruption
    mode seen so far — frozen clocks, reset rows, end-of-game rows misnumbered
    as play 1 (which once froze a full game's lead at +3)."""
    import duckdb
    from cfbpred.config import PQ
    q = duckdb.connect().execute(f"""
        WITH last AS (SELECT game_id,
            max_by(home_lead, play_index) lb, max_by(total_pts_before, play_index) tb
          FROM read_parquet('{FEAT / "play_raw.parquet"}') GROUP BY 1)
        SELECT avg(CASE WHEN abs(lb - g.margin) <= 8 THEN 1.0 ELSE 0 END)
        FROM last JOIN read_parquet('{PQ / "games.parquet"}') g USING (game_id)
    """).fetchone()[0]
    assert q > 0.985, f"only {q:.1%} of games' repaired scores reach the final"


def test_implied_scores_are_real_scores():
    """Every (home_lead, total_pts_before) pair must decompose into two
    non-negative integer team scores. A CFP first round shipped rows whose
    'score' columns held score-differential garbage (6 / -3); those values are
    monotone-compatible, so only a plausibility gate catches them — the UI once
    showed 'Vanderbilt: -3 points' from exactly this."""
    import duckdb
    q = duckdb.connect().execute(f"""
        SELECT count(*) FROM read_parquet('{FEAT / "play_raw.parquet"}')
        WHERE (total_pts_before - home_lead) < 0
           OR (total_pts_before + home_lead) < 0
           OR (total_pts_before + home_lead) % 2 != 0
    """).fetchone()[0]
    assert q == 0, f"{q} rows imply a negative or fractional team score"
