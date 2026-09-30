"""Cross-season talent percentiles, opt-in scalar schema, recency weights and the median readout."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfbpred import train as train_module  # noqa: E402
from cfbpred.ensemble import median_readout  # noqa: E402
from cfbpred.features_raw import TalentPercentiles, schedule_context  # noqa: E402
from cfbpred.model import MARGIN_CENTRES  # noqa: E402
from cfbpred.player_full_history_net import SCALAR_EXTRAS_BY_VARIANT, TALENT_FIX_VARIANTS  # noqa: E402
from cfbpred.player_sequence_train import recency_weights  # noqa: E402


def _talent():
    return pd.DataFrame({"season": [2023, 2023, 2023, 2023, 2024, 2024],
                         "team": ["A", "B", "C", "F", "A", "B"],
                         "talent_composite": [900.0, 500.0, 100.0, 80.0, 950.0, 700.0],
                         "recruit_pts_4c": np.nan})


DIVS = {(2023, "A"): "fbs", (2023, "B"): "fbs", (2023, "C"): "fbs", (2023, "F"): "fcs",
        (2024, "A"): "fbs", (2024, "B"): "fbs", (2024, "C"): "fbs", (2024, "F"): "fcs"}


def test_percentile_is_within_season_among_fbs_only():
    t = TalentPercentiles(_talent(), DIVS)
    # 2023: the FCS row (80) must not enter the FBS ranking
    assert t.value(2023, "C", "fbs") == pytest.approx(1 / 3)
    assert t.value(2023, "A", "fbs") == pytest.approx(1.0)
    # 2024: C has no published composite -> unknown, never imputed
    assert np.isnan(t.value(2024, "C", "fbs"))
    assert t.value(2024, "B", "fbs") == pytest.approx(0.5)


def test_non_fbs_is_zero_in_every_talent_season_and_unknown_before():
    t = TalentPercentiles(_talent(), DIVS)
    assert t.value(2023, "F", "fcs") == 0.0          # had a raw value; still 0.0
    assert t.value(2024, "F", "fcs") == 0.0          # no raw value; same 0.0
    assert np.isnan(t.value(2014, "A", "fbs"))       # before the composite exists


def test_sos_percentile_uses_only_prior_played_games_and_counts_fcs_as_zero():
    t = TalentPercentiles(_talent(), DIVS)
    kick = pd.to_datetime(["2024-09-01", "2024-09-08", "2024-09-15", "2024-09-22"], utc=True)
    log = pd.DataFrame({"game_id": [1, 2, 3, 4], "season": 2024, "week": [1, 2, 3, 4], "kick": kick,
                        "neutral": False, "team": "A", "opp": ["F", "B", "C", "B"],
                        "opp_div": ["fcs", "fbs", "fbs", "fbs"],
                        "pf": [50.0, 30.0, np.nan, np.nan], "pa": [0.0, 20.0, np.nan, np.nan], "was_home": 1.0})
    s = schedule_context(log, _talent(), t).set_index("game_id").sos_tpct_talent
    assert np.isnan(s[1])                            # opener: nothing faced yet
    assert s[2] == pytest.approx(0.0)                # FCS opponent faced -> 0.0
    assert s[3] == pytest.approx(0.25)               # mean(0.0, 0.5)
    assert s[4] == pytest.approx(0.25)               # unplayed game 3 contributes nothing


def test_tpct_columns_are_opt_in_and_drops_remove_columns_and_flags():
    g = pd.DataFrame({"game_id": [1], "season": [2024], "neutral_site": [False], "conference_game": [True],
                      "week_c": [3], "home_division": ["fbs"], "away_division": ["fcs"],
                      "h_talent_composite": [700.0], "a_talent_composite": [np.nan],
                      "h_tpct_talent": [0.6], "a_tpct_talent": [0.0], "h_ppg_for": [30.0], "a_ppg_for": [20.0]})
    previous = train_module.SCALAR_EXTRAS
    try:
        train_module.SCALAR_EXTRAS = ()
        _, cols = train_module.prep_scalars(g)
        assert "h_tpct_talent" not in cols and "h_talent_composite" in cols and "h_has_talent" in cols
        train_module.SCALAR_EXTRAS = SCALAR_EXTRAS_BY_VARIANT["full_context_v4"]
        _, cols = train_module.prep_scalars(g)
        assert {"h_tpct_talent", "a_tpct_talent"} <= set(cols)
        assert not {"h_talent_composite", "a_talent_composite", "h_has_talent", "a_has_talent"} & set(cols)
    finally:
        train_module.SCALAR_EXTRAS = previous


def test_every_talent_fix_variant_shares_the_schema():
    assert {SCALAR_EXTRAS_BY_VARIANT[v] for v in TALENT_FIX_VARIANTS} == {SCALAR_EXTRAS_BY_VARIANT["full_context_v4"]}


def test_recency_weights_halve_per_half_life_and_average_one():
    seasons = np.array([2021, 2022, 2023, 2024, 2025, 2026])
    train = np.arange(5)                              # 2026 is not a training season
    w = recency_weights(seasons, train, 4.0).numpy()
    assert w[train].mean() == pytest.approx(1.0, abs=1e-6)
    assert w[0] / w[4] == pytest.approx(0.5 ** (4 / 4), rel=1e-5)
    assert np.all(np.diff(w[train]) > 0)


def test_median_readout_interpolates_inside_the_crossing_bin():
    n = len(MARGIN_CENTRES)
    zero = int((MARGIN_CENTRES == 0).nonzero()[0])
    p = torch.zeros(2, n)
    p[0, zero] = 1.0                                  # a point mass reads as its own value
    p[1, zero - 3], p[1, zero + 7] = 0.4, 0.6        # crosses 0.5 inside the +7 bin
    m = median_readout(p, MARGIN_CENTRES)
    assert m[0] == pytest.approx(0.0)
    assert m[1] == pytest.approx(7 - 0.5 + (0.5 - 0.4) / 0.6)


def test_transfers_count_only_before_the_teams_opener(tmp_path, monkeypatch):
    import cfbpred.features_raw as fr
    monkeypatch.setattr(fr, "CFBD", tmp_path)
    pd.DataFrame({"season": [2024, 2024, 2024], "origin": ["B", "B", "C"], "destination": ["A", "A", None],
                  "transferDate": ["2024-01-10T00:00:00Z", "2024-09-15T00:00:00Z", "2024-01-05T00:00:00Z"],
                  "stars": [4, 3, None], "rating": [0.9, None, None]}).to_parquet(tmp_path / "portal.parquet")
    pd.DataFrame({"year": [2024], "collegeTeam": ["B"], "overall": [10], "round": [1]}).to_parquet(tmp_path / "draft_picks.parquet")
    g = pd.DataFrame({"season": [2024, 2024], "kick": pd.to_datetime(["2024-08-31", "2024-09-07"], utc=True),
                      "home_team": ["A", "B"], "away_team": ["C", "A"]})
    m = fr.roster_movement_features(g).set_index(["season", "team"])
    assert m.loc[(2024, "A"), "mvmt_in_n"] == 1            # the September transfer came after A's opener
    assert m.loc[(2024, "A"), "mvmt_in_top"] == 1 and m.loc[(2024, "A"), "mvmt_in_rating"] == pytest.approx(0.9)
    assert m.loc[(2024, "B"), "mvmt_out_n"] == 1 and m.loc[(2024, "B"), "mvmt_draft_r1"] == 1
    assert m.loc[(2024, "C"), "mvmt_out_n"] == 1           # an uncommitted entrant still leaves


def test_coach_change_is_dated_to_the_game_it_took_effect(tmp_path, monkeypatch):
    import cfbpred.features_raw as fr
    monkeypatch.setattr(fr, "CFBD", tmp_path)
    seasons_a = [{"year": 2020, "school": "A", "games": 12}, {"year": 2021, "school": "A", "games": 3}]
    seasons_b = [{"year": 2021, "school": "A", "games": 1}]
    pd.DataFrame({"id": [1, 2], "seasons": [seasons_a, seasons_b]}).to_parquet(tmp_path / "coaches.parquet")
    kick = pd.to_datetime(["2021-09-01", "2021-09-08", "2021-09-15", "2021-09-22", "2021-09-29"], utc=True)
    log = pd.DataFrame({"game_id": [1, 2, 3, 4, 5], "season": 2021, "kick": kick, "team": "A"})
    c = fr.coach_change_features(log).set_index("game_id")
    assert c.cchg_changed.tolist() == [0, 0, 0, 1, 1]      # 3 games by the incumbent, then the interim
    assert c.cchg_new_since_last.tolist() == [0, 0, 0, 1, 0]
