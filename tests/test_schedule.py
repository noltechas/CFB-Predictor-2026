"""Gates for the current-season schedule flowing through the feature store.

Scheduled-but-unplayed games are prediction targets only. They must never be
a source of anything: no history slot, no season-to-date stat, no label — and
their feature rows must look exactly like a historical first-game row so the
model is never asked to extrapolate off-distribution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfbpred.config import (CURRENT_SEASON, FEAT, PQ,  # noqa: E402
                            TEST_SEASONS, TRAIN_SEASONS)

SCHEDULE = PQ / "schedule.parquet"


@pytest.fixture(scope="module")
def scalars():
    return pd.read_parquet(FEAT / "game_scalars.parquet")


@pytest.fixture(scope="module")
def sched():
    if not SCHEDULE.exists():
        pytest.skip("no schedule.parquet — preseason refresh not run")
    return pd.read_parquet(SCHEDULE)


def test_current_season_is_never_trained_on():
    assert CURRENT_SEASON not in TRAIN_SEASONS
    assert TEST_SEASONS == [CURRENT_SEASON]


def test_schedule_games_reach_the_feature_store(scalars, sched):
    fbs = sched[(sched["home_division"] == "fbs")
                | (sched["away_division"] == "fbs")]
    have = set(scalars.loc[scalars["season"] == CURRENT_SEASON, "game_id"])
    missing = set(fbs["game_id"]) - have
    assert not missing, f"{len(missing)} scheduled FBS games missing features"


def test_unplayed_games_carry_no_labels(scalars):
    cur = scalars[scalars["season"] == CURRENT_SEASON]
    unplayed = cur[cur["home_points"].isna()]
    assert unplayed["margin"].isna().all()
    assert unplayed["total"].isna().all()


def test_history_slots_are_played_games_only():
    h = pd.read_parquet(FEAT / "game_history.parquet")
    assert h["pf"].notna().all() and h["pa"].notna().all(), (
        "a scheduled (unplayed) game filled a history slot")


def test_unplayed_rows_match_the_first_game_convention(scalars):
    """A team that has played no games yet must carry no season-to-date
    stats — the preseason shape.

    The filter is h_gp == 0, NOT week_c == 1. CFBD's 2026 week 1 spans
    Aug 27 - Sep 7 (two weekends), so a team can play its SECOND week-1 game
    with h_gp == 1 legitimately — Stanford hosts Hawai'i on Aug 29 and Miami
    on Sep 5, both labelled week 1. Asserting week_c == 1 implies h_gp == 0
    failed on 49 real rows once week 1's first weekend completed."""
    cur = scalars[(scalars["season"] == CURRENT_SEASON)
                  & scalars["home_points"].isna()]
    if cur.empty:
        pytest.skip("current season fully graded")
    first = cur[cur["h_gp"].fillna(0) == 0]
    if first.empty:
        pytest.skip("no first-game rows left unplayed")
    assert first["h_o_ypp"].isna().all(), "unplayed rows carry pbp stats"
    assert first["h_qb_known"].isna().all() | (first["h_qb_known"] == 0).all()
