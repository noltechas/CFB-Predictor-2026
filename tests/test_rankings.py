"""Ranking math and current-snapshot boundaries, without network or training."""
import sys
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfbpred.rankings import (_current_roster_games, _current_scalar_fields,
                              _preseason_roster_games, _preseason_scalar_fields,
                              _restrict_player_store_to_cutoff, pair_frame,
                              period_start, rating_readout)
from cfbpred.ranking_snapshot import catalog_for, completed_before, result_inputs
import cfbpred.ranking_snapshot as snapshots
from cfbpred.player_features import build_box_features


def test_common_pool_cancels_input_order_bias_and_centers_fbs():
    strength = np.array([20., 10., -4., -30.])
    reference = np.array([True, True, False, False])
    # Even a pair-specific symmetric slot bias must not affect the ratings.
    bias = np.array([[0, 3, 6, 5], [3, 0, 9, 2], [6, 9, 0, 1], [5, 2, 1, 0]])
    matrix = strength[:, None] - strength[None, :] + bias
    ratings = rating_readout(matrix, reference)
    np.testing.assert_allclose(ratings, strength - strength[reference].mean())
    assert ratings[reference].mean() == 0


def test_ratings_do_not_depend_on_iteration_order():
    rng = np.random.default_rng(8)
    matrix = rng.normal(size=(6, 6))
    reference = np.array([True, True, True, False, False, False])
    permutation = np.array([3, 0, 5, 1, 4, 2])
    result = rating_readout(matrix, reference)
    changed = rating_readout(matrix[permutation][:, permutation], reference[permutation])
    np.testing.assert_allclose(changed, result[permutation])


def test_invalid_matchup_matrices_fail_closed():
    for matrix, ref in [(np.ones((2, 3)), np.array([True, False])),
                         (np.full((2, 2), np.nan), np.array([True, False])),
                         (np.zeros((2, 2)), np.array([False, False]))]:
        with pytest.raises(ValueError):
            rating_readout(matrix, ref)


def games():
    return pd.DataFrame([
        {"game_id": 1, "season": 2026, "week": 1, "week_c": 1, "kick": pd.Timestamp("2026-09-01T12:00Z"),
         "home_team": "A", "away_team": "B", "home_division": "fbs", "away_division": "fcs", "home_conference": "X", "away_conference": "Y",
         "home_points": 30., "away_points": 10., "neutral_site": False, "completed": True},
        {"game_id": 2, "season": 2026, "week": 2, "week_c": 2, "kick": pd.Timestamp("2026-09-08T12:00Z"),
         "home_team": "B", "away_team": "A", "home_division": "fcs", "away_division": "fbs", "home_conference": "Y", "away_conference": "X",
         "home_points": 99., "away_points": 0., "neutral_site": True, "completed": True},
        {"game_id": 3, "season": 2026, "week": 1, "week_c": 1, "kick": pd.Timestamp("2026-09-06T12:00Z"),
         "home_team": "A", "away_team": "B", "home_division": "fbs", "away_division": "fcs", "home_conference": "X", "away_conference": "Y",
         "home_points": 7., "away_points": 3., "neutral_site": False, "completed": False},
    ])


def test_future_and_in_progress_games_never_enter_snapshot():
    cutoff = pd.Timestamp("2026-09-06T15:00Z")
    completed = completed_before(games(), cutoff)
    assert completed.game_id.tolist() == [1]
    catalog = catalog_for(games(), 2026, {"A": 1, "B": 2})
    records, hist, opponents = result_inputs(catalog, completed, {"team": {"A": 1, "B": 2}}, 2026, cutoff, {})
    assert records.loc["A", "gp"] == 1
    assert records.loc["A", "wins"] == 1
    assert records.loc["B", "losses"] == 1
    assert records.loc["A", "ppg_for"] == 30
    assert hist[:, :, -1].sum() == 2
    assert opponents[0, 0] == 2
    assert hist[0, 0, 4] == pytest.approx(5.125 / 100)


def test_pair_context_is_equal_in_raw_units_and_flags_follow_teams():
    fields = pd.DataFrame({"gp": [1., 2.], "n_players": [100., np.nan], "bx_pass_ypg": [250., np.nan], "rest": [2., 19.], "travel": [5., 2000.]})
    catalog = pd.DataFrame({"division": ["fbs", "fcs"]})
    frame = pair_frame(fields, catalog, np.array([0, 1]), np.array([1, 0]), 2)
    assert (frame.h_rest == 7).all() and (frame.a_rest == 7).all()
    assert (frame.h_travel == 250).all() and (frame.a_travel == 250).all()
    assert (frame.is_home_site == 0).all()
    assert frame.h_has_roster.tolist() == [1., 0.]
    assert frame.a_has_box.tolist() == [0., 1.]
    assert frame.home_is_fcs.tolist() == [0., 1.]
    assert (frame.h_body_clock_shift == 0).all() and (frame.a_body_clock_shift == 0).all()


def test_pair_rebuilds_opponent_dependent_recent_form():
    fields = pd.DataFrame({
        "o_r5_pass_ypp": [10., 4.], "o_r5_rush_ypp": [6., 3.],
        "o_r5_sack_rate": [.05, .10], "o_r5_expl20_rate": [.20, .05],
        "o_r5_pass_share": [.60, .40],
        "d_r5_pass_ypp": [5., 8.], "d_r5_rush_ypp": [2., 5.],
        "d_r5_sack_rate": [.08, .03], "d_r5_expl20_rate": [.10, .15],
        "d_r5_pass_share": [.50, .50],
        # These values belong to other scheduled opponents and must be ignored.
        "match_r5_pass_ypp_diff": [999., 999.],
    })
    catalog = pd.DataFrame({"division": ["fbs", "fbs"]})
    frame = pair_frame(fields, catalog, np.array([0]), np.array([1]), 2)
    assert frame.h_match_r5_pass_ypp_diff.iloc[0] == 2.
    assert frame.a_match_r5_pass_ypp_diff.iloc[0] == -1.
    assert frame.h_match_r5_pass_style.iloc[0] == pytest.approx(4.8)
    assert frame.a_match_r5_rush_style.iloc[0] == pytest.approx(1.2)


def test_current_scalar_fields_uses_each_teams_selected_side():
    catalog = pd.DataFrame({"team": ["A", "B"]})
    base = {
        "G_gid": np.array([11, 12]),
        "scalar_cols": ["h_power", "a_power", "is_home_site"],
        "G_raw_scalars": np.array([[9., 1., 1.], [2., 7., 1.]], np.float32),
    }
    chosen = pd.DataFrame({"team": ["A", "B"], "game_id": [11, 12],
                           "side": ["home", "away"]})
    fields = _current_scalar_fields(catalog, base, chosen)
    assert fields.power.tolist() == [9., 7.]


def test_preseason_scalar_fields_remove_current_season_state():
    catalog = pd.DataFrame({"team": ["A", "B"]})
    columns = [f"{side}_{name}" for side in ("h", "a") for name in
               ("gp", "o_ypp", "o_r5_games", "talent", "pre_qb_ypa")]
    base = {"G_gid": np.array([11, 12]), "scalar_cols": columns,
            "G_raw_scalars": np.array([
                [2., 9., 2., 90., 8., 8., 1., 8., 10., 5.],
                [3., 7., 3., 70., 7., 1., 4., 1., 60., 6.],
            ], np.float32)}
    chosen = pd.DataFrame({"team": ["A", "B"], "game_id": [11, 12],
                           "side": ["home", "away"]})
    historical = pd.DataFrame({"gp": [0., 0.]}, index=["A", "B"])
    fields = _preseason_scalar_fields(catalog, base, chosen, historical)
    assert fields.gp.tolist() == [0., 0.]
    assert fields.o_r5_games.tolist() == [0., 0.]
    assert fields.o_ypp.isna().all()
    assert fields.talent.tolist() == [90., 60.]
    assert fields.pre_qb_ypa.tolist() == [8., 6.]


def test_preseason_player_cutoff_rewinds_current_and_sequence_history(tmp_path):
    class Store:
        player_refs = np.array([0, 1, 2])
        current_refs = np.array([0, 7, 8])
        history_tips = np.array([0, 2, 3])
        event_kick = np.array([0,
            pd.Timestamp("2025-12-01T12:00Z").value,
            pd.Timestamp("2026-08-28T12:00Z").value,
            pd.Timestamp("2025-11-01T12:00Z").value])

        @staticmethod
        def roster_refs(_):
            return np.array([[[1, 2]]]), np.array([[[True, True]]])

    players, sequences = tmp_path / "players", tmp_path / "sequences"
    players.mkdir(); sequences.mkdir()
    pd.DataFrame({"person_id": ["p1", "p2"]}).to_parquet(
        players / "player_seasons.parquet", index=False)
    pd.DataFrame([
        {"event_id": 1, "person_id": "p1", "kick": "2025-12-01T12:00Z"},
        {"event_id": 2, "person_id": "p1", "kick": "2026-08-28T12:00Z"},
        {"event_id": 3, "person_id": "p2", "kick": "2025-11-01T12:00Z"},
    ]).to_parquet(sequences / "events.parquet", index=False)
    store = Store()
    count = _restrict_player_store_to_cutoff(
        store, np.array([0]), pd.Timestamp("2026-08-27T22:00Z"), players, sequences)
    assert count == 2
    assert store.current_refs.tolist() == [0, 0, 0]
    assert store.history_tips.tolist() == [0, 1, 3]


def test_box_snapshot_includes_last_game_without_requiring_another_game():
    g = games().iloc[:1]
    raw = pd.DataFrame([
        {"game_id": 1, "home_away": "home", "category": "passing", "stat_type": "YDS", "stat": "250", "athlete": "Player"},
        {"game_id": 1, "home_away": "home", "category": "passing", "stat_type": "C/ATT", "stat": "20/30", "athlete": "Player"},
    ])
    targets = pd.DataFrame({"team": ["A", "B"], "season": [2026, 2026], "week_c": [10000, 10000], "game_id": [-1, -2]})
    result = build_box_features(g, raw=raw, targets=targets).query("week_c == 10000").set_index("team")
    assert result.loc["A", "box_gp"] == 1
    assert result.loc["A", "bx_pass_ypg"] == 250
    assert result.loc["A", "bx_comp_pct"] == pytest.approx(20 / 31)
    assert result.loc["B", "box_gp"] == 0
    assert np.isnan(result.loc["B", "bx_pass_ypg"])


def test_period_uses_sunday_central_time_and_handles_year_boundary():
    assert period_start(pd.Timestamp("2026-09-06T04:00Z")) == "2026-08-30"
    assert period_start(pd.Timestamp("2026-09-06T06:00Z")) == "2026-09-06"
    assert period_start(pd.Timestamp("2027-01-01T12:00Z")) == "2026-12-27"


def test_current_roster_source_is_populated_future_row_before_cutoff():
    cat = pd.DataFrame({"team": ["A", "B"], "division": ["fbs", "fcs"]})
    rosters = pd.DataFrame([
        {"game_id": 1, "season": 2026, "team": "A", "side": "home", "players": 0,
         "kick": "2026-09-08T12:00Z", "last_history_kick": pd.NaT},
        {"game_id": 2, "season": 2026, "team": "A", "side": "away", "players": 90,
         "kick": "2026-09-15T12:00Z", "last_history_kick": "2026-09-01T12:00Z"},
        {"game_id": 3, "season": 2026, "team": "B", "side": "home", "players": 80,
         "kick": "2026-09-09T12:00Z", "last_history_kick": "2026-09-01T12:00Z"},
    ])
    chosen = _current_roster_games(cat, rosters, np.array([1, 2, 3]),
                                   pd.Timestamp("2026-09-07T12:00Z"))
    assert chosen.game_id.tolist() == [2, 3]
    assert chosen.players.tolist() == [90, 80]


def test_preseason_roster_source_is_each_teams_first_game():
    cat = pd.DataFrame({"team": ["A", "B"], "division": ["fbs", "fcs"]})
    rosters = pd.DataFrame([
        {"game_id": 1, "season": 2026, "team": "A", "side": "home", "players": 90,
         "kick": "2026-08-27T22:00Z", "last_history_kick": "2025-12-01T12:00Z"},
        {"game_id": 2, "season": 2026, "team": "A", "side": "away", "players": 95,
         "kick": "2026-09-05T12:00Z", "last_history_kick": "2026-08-27T22:00Z"},
        {"game_id": 3, "season": 2026, "team": "B", "side": "home", "players": 80,
         "kick": "2026-09-01T12:00Z", "last_history_kick": "2025-11-20T12:00Z"},
    ])
    chosen = _preseason_roster_games(cat, rosters, np.array([1, 2, 3]),
                                     pd.Timestamp("2026-08-27T22:00Z"))
    assert chosen.game_id.tolist() == [1, 3]
    assert chosen.players.tolist() == [90, 80]


def test_preseason_reconstruction_excludes_current_results_and_dynamic_inputs(tmp_path, monkeypatch):
    f = tmp_path / "game_scalars.parquet"
    f.touch()
    feature_time = pd.Timestamp("2026-09-06T18:00Z").timestamp()
    os.utime(f, (feature_time, feature_time))
    (tmp_path / "team_vocab.json").write_text(json.dumps({"team": {"A": 1, "B": 2}, "conf": {"X": 1, "Y": 2}}))
    current = games()
    prior = current.iloc[:1].copy()
    prior["game_id"], prior["season"], prior["kick"] = 0, 2025, pd.Timestamp("2025-11-01T12:00Z")
    source = pd.concat([prior, current], ignore_index=True)
    scalars = pd.DataFrame({"season": [2026], "home_team": ["A"], "away_team": ["B"],
                            "h_n_players": [100.], "a_n_players": [90.],
                            "h_gp": [9.], "a_gp": [9.], "h_bx_pass_ypg": [999.], "a_bx_pass_ypg": [999.]})
    tables = {
        "game_scalars.parquet": scalars,
        "games.parquet": pd.DataFrame({"id": source.game_id, "completed": source.completed}),
        "teams.parquet": pd.DataFrame({"id": [1, 2], "school": ["A", "B"], "season_pulled": [2026, 2026]}),
        "schedule.parquet": pd.DataFrame({"home_team": ["A"], "home_id": [1], "away_team": ["B"], "away_id": [2]}),
    }
    monkeypatch.setattr(snapshots, "FEAT", tmp_path)
    monkeypatch.setattr(snapshots, "load_games", lambda *a, **kw: source.copy())
    monkeypatch.setattr(pd, "read_parquet", lambda path, **kw: tables[Path(path).name].copy())
    result = snapshots.build_snapshot(2026, pd.Timestamp("2026-09-06T15:00Z"), preseason=True)
    assert result["as_of"] == "2026-09-01T12:00:00+00:00"
    assert result["source_week"] == 0
    assert result["reconstructed"]
    assert (result["catalog"][["wins", "losses", "ties"]] == 0).all().all()
    assert (result["fields"].gp == 0).all() and (result["fields"].box_gp == 0).all()
    assert "bx_pass_ypg" not in result["fields"]
    assert result["history"][:, :, -1].sum() == 2  # one 2025 game per team
    assert (result["history"][:, :, 5] == 0).all()  # no current-season history


def test_replay_records_use_frozen_restored_schedule(tmp_path, monkeypatch):
    (tmp_path / "game_scalars.parquet").touch()
    (tmp_path / "team_vocab.json").write_text(json.dumps({"team": {"A": 1, "B": 2}}))
    source = games().iloc[:1].copy()
    source["season"] = 2001
    source["start_date"] = "2001-12-28T12:00Z"
    source["season_type"] = "postseason"
    source["home_id"], source["away_id"] = 1, 2
    virtual = source.copy()
    virtual["game_id"], virtual["start_date"] = -1, "2002-01-09T12:00Z"
    # Virtual targets are never result history, even if malformed with scores.
    frozen = pd.concat([source, virtual], ignore_index=True)
    tables = {
        "frozen.parquet": frozen,
        "game_scalars.parquet": pd.DataFrame({"season": [2001], "home_team": ["A"], "away_team": ["B"],
                                             "h_n_players": [80.], "a_n_players": [75.]}),
        "games.parquet": pd.DataFrame({"id": [1], "completed": [True]}),
        "teams.parquet": pd.DataFrame({"id": [1, 2], "school": ["A", "B"], "season_pulled": [2026, 2026]}),
        "schedule.parquet": pd.DataFrame({"home_team": ["A"], "home_id": [1], "away_team": ["B"], "away_id": [2]}),
    }
    def wrong_corpus(*args, **kwargs):
        raise AssertionError("Replay read mutable global games")
    monkeypatch.setattr(snapshots, "load_games", wrong_corpus)
    monkeypatch.setattr(pd, "read_parquet", lambda path, **kw: tables[Path(path).name].copy())
    result = snapshots.build_snapshot(2001, pd.Timestamp("2002-01-10T12:00Z"),
                                      feature_dir=tmp_path, schedule_path=tmp_path / "frozen.parquet",
                                      reconstructed=True, results_only=True, allow_short_history=True)
    assert result["fields"].gp.tolist() == [1., 1.]
    assert result["catalog"].wins.tolist() == [1, 0]
    assert result["history"][:, :, -1].sum() == 2
    assert result["source_week"] == 21


@pytest.mark.parametrize("architecture, variant, expected", [
    ("player_full_context_v1", "full_context", "player"),
    ("player_full_context_v1", "full_context_v2", "player"),
    ("architecture_research_v1", "tabm16", "player"),
    ("raw_dnn", "delta", "raw"),
])
def test_generate_dispatches_every_full_context_variant_to_player_model(
        tmp_path, monkeypatch, architecture, variant, expected):
    import cfbpred.rankings as rankings
    active = tmp_path / "player_sequence_production/active.json"
    active.parent.mkdir(parents=True)
    active.write_text(json.dumps({"architecture": architecture, "variant": variant}))
    monkeypatch.setattr(rankings, "MODELS", tmp_path)
    monkeypatch.setattr(rankings, "generate_player", lambda *a, **kw: "player")
    monkeypatch.setattr(rankings, "generate_raw", lambda *a, **kw: "raw")
    assert rankings.generate(2026) == expected
