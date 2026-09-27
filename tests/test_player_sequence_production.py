import json

import pandas as pd
import pytest

import cfbpred.player_sequence_production as production
from cfbpred.pregame_experiments import sha


def test_installed_context_source_cannot_change_silently(tmp_path):
    source = tmp_path / "model/source.json"
    source.parent.mkdir()
    source.write_text('{}')
    plan = source.parent / "plan.json"
    plan.write_text(json.dumps({"source_sha256": sha(source)}))
    bundle = dict(architecture="player_full_context_v1", input_source=str(source), plan_sha256=sha(plan))
    assert production.context_source(tmp_path / "inputs", bundle) == source
    source.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="lineage changed"):
        production.context_source(tmp_path / "inputs", bundle)


def test_checkpoint_vocabulary_mapping_uses_saved_known_identities():
    current = {"A": 1, "B": 2, "New": 3}
    saved = {"B": 1, "A": 2, "Old": 3}
    assert production._mapping(current, saved, [0, 1]).tolist() == [0, 0, 1, 0]


def test_local_promotion_replaces_future_and_preserves_started(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "PQ", tmp_path)
    registry = tmp_path / "registry"
    version = registry / production.VERSION
    version.mkdir(parents=True)
    bundle = {"architecture": "player_sequence_v1", "variant": "context", "members": []}
    (version / "bundle.json").write_text(json.dumps(bundle))
    active = {"version": production.VERSION, "scope": "production_pregame",
              "bundle": str(version / "bundle.json"), "bundle_sha256": sha(version / "bundle.json")}
    (registry / "active.json").write_text(json.dumps(active))
    pd.DataFrame({"game_id": [1, 2], "season": [2026, 2026],
                  "home_win_prob": [.2, .3], "pred_margin": [-7., -3.], "pred_total": [42., 45.],
                  "pred_home_pts": [17, 21], "pred_away_pts": [24, 24], "split": ["test", "test"]}).to_parquet(
                      tmp_path / "game_predictions.parquet", index=False)
    pd.DataFrame({"game_id": [1, 2], "season": [2026, 2026],
                  "margin_pmf": [[.2], [.3]], "total_pmf": [[.4], [.5]]}).to_parquet(
                      tmp_path / "game_pmfs.parquet", index=False)
    pd.DataFrame({"game_id": [1, 2, 3],
                  "start_date": ["2026-09-01T00:00:00Z", "2026-09-12T00:00:00Z",
                                 "2026-09-19T00:00:00Z"]}).to_parquet(tmp_path / "schedule.parquet", index=False)
    forecasts = tmp_path / "future.parquet"
    pd.DataFrame({"game_id": [2, 3], "season": [2026, 2026],
                  "home_win_prob": [.8, .7], "pred_margin": [7., 4.], "pred_total": [50., 48.],
                  "pred_home_pts": [29, 26], "pred_away_pts": [22, 22], "split": ["test", "test"],
                  "margin_pmf": [[.8], [.7]], "total_pmf": [[.9], [.8]]}).to_parquet(forecasts, index=False)
    forecasts.with_suffix(".json").write_text(json.dumps({"cutoff": "2026-09-06T00:00:00+00:00"}))

    report = production.promote_local(forecasts, registry)
    games = pd.read_parquet(tmp_path / "game_predictions.parquet").set_index("game_id")
    pmfs = pd.read_parquet(tmp_path / "game_pmfs.parquet").set_index("game_id")
    assert report["passed"] and games.loc[1, "pred_margin"] == -7
    assert games.loc[2, "pred_margin"] == 7 and games.loc[3, "pred_margin"] == 4
    assert pmfs.loc[1, "margin_pmf"].tolist() == [.2]
    assert pmfs.loc[2, "margin_pmf"].tolist() == [.8]
    assert set(games.index) == set(pmfs.index) == {1, 2, 3}


def test_full_season_promotion_replaces_started_and_future(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "PQ", tmp_path)
    registry = tmp_path / "registry"
    version = registry / production.VERSION
    version.mkdir(parents=True)
    bundle = {"architecture": "player_sequence_v1", "variant": "context", "members": []}
    (version / "bundle.json").write_text(json.dumps(bundle))
    active = {"version": production.VERSION, "scope": "production_pregame",
              "bundle": str(version / "bundle.json"), "bundle_sha256": sha(version / "bundle.json")}
    (registry / "active.json").write_text(json.dumps(active))
    pd.DataFrame({"game_id": [1, 2], "season": [2026, 2026],
                  "home_win_prob": [.2, .3], "pred_margin": [-7., -3.], "pred_total": [42., 45.],
                  "pred_home_pts": [17, 21], "pred_away_pts": [24, 24],
                  "split": ["test", "test"]}).to_parquet(tmp_path / "game_predictions.parquet", index=False)
    pd.DataFrame({"game_id": [1, 2], "season": [2026, 2026],
                  "margin_pmf": [[.2], [.3]], "total_pmf": [[.4], [.5]]}).to_parquet(
                      tmp_path / "game_pmfs.parquet", index=False)
    pd.DataFrame({"game_id": [1, 2],
                  "start_date": ["2026-09-01T00:00:00Z", "2026-09-12T00:00:00Z"]}).to_parquet(
                      tmp_path / "schedule.parquet", index=False)
    forecasts = tmp_path / "season.parquet"
    pd.DataFrame({"game_id": [1, 2], "season": [2026, 2026],
                  "home_win_prob": [.7, .8], "pred_margin": [4., 7.], "pred_total": [48., 50.],
                  "pred_home_pts": [26, 29], "pred_away_pts": [22, 22],
                  "split": ["test", "test"], "margin_pmf": [[.7], [.8]],
                  "total_pmf": [[.8], [.9]]}).to_parquet(forecasts, index=False)
    forecasts.with_suffix(".json").write_text(json.dumps({
        "cutoff": "2026-09-06T00:00:00+00:00", "coverage": "full_season", "version": production.VERSION}))

    # Historical reconstruction publishes PMFs and version tags for old
    # seasons too; subsequent daily/current-season promotions preserve them.
    old = pd.read_parquet(tmp_path / "game_predictions.parquet")
    historic = old.iloc[[0]].assign(game_id=90, season=2025, split="train")
    pd.concat([old, historic]).assign(model_version="prior").to_parquet(tmp_path / "game_predictions.parquet", index=False)
    old_pmf = pd.read_parquet(tmp_path / "game_pmfs.parquet")
    pd.concat([old_pmf, old_pmf.iloc[[0]].assign(game_id=90, season=2025)]).to_parquet(tmp_path / "game_pmfs.parquet", index=False)

    report = production.promote_local(forecasts, registry, include_started=True)
    games = pd.read_parquet(tmp_path / "game_predictions.parquet").set_index("game_id")
    assert report["completed_and_future_current_season_rows_replaced"]
    assert games.loc[1, "pred_margin"] == 4 and games.loc[2, "pred_margin"] == 7
    assert games.loc[90, "model_version"] == "prior" and games.loc[90, "pred_margin"] == -7
    assert games.loc[1, "model_version"] == production.VERSION
    assert set(pd.read_parquet(tmp_path / "game_pmfs.parquet").game_id) == {1, 2, 90}


def test_position_source_must_match_the_checkpoint():
    # stores and checkpoints from before the field exist are the recruiting-only source
    assert production.check_position_source({}, {}) == "scouting"
    roster = {"player_schema": {"position_source": "scouting_then_roster"}}
    assert production.check_position_source(roster, {"position_source": "scouting_then_roster"}) == "scouting_then_roster"
    with pytest.raises(ValueError, match="position source"):
        production.check_position_source({}, {"position_source": "scouting_then_roster"})
    with pytest.raises(ValueError, match="position source"):
        production.check_position_source(roster, {})


def test_daily_refresh_builds_the_store_the_active_checkpoint_needs(tmp_path, monkeypatch):
    import torch
    import cfbpred.refresh_player_sequence_production as refresher
    bundle_dir = tmp_path / "v9"
    bundle_dir.mkdir()
    torch.save({"player_schema": {"position_source": "scouting_then_roster"}}, bundle_dir / "member.pt")
    monkeypatch.setattr(refresher, "active_bundle",
                        lambda path: ({}, bundle_dir / "bundle.json", {"members": ["member.pt"]}))
    assert refresher.active_position_source(tmp_path / "active.json") == "scouting_then_roster"
    torch.save({}, bundle_dir / "member.pt")   # a checkpoint predating the field
    assert refresher.active_position_source(tmp_path / "active.json") == "scouting"
