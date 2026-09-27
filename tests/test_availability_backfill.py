import pandas as pd
import json
import pytest

from cfbpred.availability_backfill import coverage_ledger, validate_fact
from cfbpred.pregame_experiments import sha


def schedule():
    return pd.DataFrame([
        dict(game_id=1, season=2024, kick="2024-09-01T16:00:00Z", conference_game=False,
             home_team="A", away_team="B", home_division="fbs", away_division="fcs",
             home_conference="Old", away_conference="FCS", home_points=7, away_points=0),
        dict(game_id=2, season=2024, kick="2024-09-08T16:00:00Z", conference_game=True,
             home_team="A", away_team="C", home_division="fbs", away_division="fbs",
             home_conference="Old", away_conference="Old", home_points=None, away_points=None),
        dict(game_id=3, season=2024, kick="2024-09-01T16:00:00Z", conference_game=False,
             home_team="D", away_team="E", home_division="fcs", away_division="fcs",
             home_conference="FCS", away_conference="FCS", home_points=0, away_points=7)])


def test_ledger_preserves_missing_rosters_opponents_and_unsearched_games():
    coverage = pd.DataFrame([dict(game_id=1, team="A", roster_players=100, reports_known=2, depth_known=0)])
    facts = pd.DataFrame([dict(game_id=1, team="A", kind="availability", source_url="https://a.example/report",
                               known_at="2024-09-01T14:00:00Z")])
    frame = coverage_ledger(schedule(), coverage, facts, "2024-09-02T00:00:00Z")
    assert len(frame) == 4
    assert set(frame.game_id) == {1, 2}
    assert frame.completed.sum() == 2
    home = frame[(frame.game_id == 1) & (frame.team == "A")].iloc[0]
    opponent = frame[(frame.game_id == 1) & (frame.team == "B")].iloc[0]
    assert home.unknown_roster_statuses == 98
    assert home.search_status == "facts_verified_partial"
    assert not opponent.roster_in_model
    assert opponent.search_status == "not_yet_searched"
    assert not frame.all_roster_availability_verified.any()


def test_ingestion_rejects_revisions_future_captures_and_wrong_teams(tmp_path):
    raw = tmp_path / "source.json"
    raw.write_text('{"link":"https://official.example/report", "date_gmt":"2024-09-01T12:00:00", "modified_gmt":"2024-09-01T14:00:00"}')
    row = dict(game_id=1, season=2024, team="A", name="John Smith", kind="availability", value="out",
               source_path=str(raw), source_url="https://official.example/report", source_sha256=sha(raw),
               known_at="2024-09-01T14:00:00Z", effective_at="2024-09-01T14:00:00Z",
               valid_until="2024-09-02T00:00:00Z", published_at="2024-09-01T12:00:00Z",
               modified_at="2024-09-01T14:00:00Z")
    games = schedule().set_index("game_id")
    assert validate_fact(row, games) is None
    assert validate_fact({**row, "source_quote": "John Smith is out"}, games) == "quoted_text_not_in_archived_source"
    assert validate_fact({**row, "modified_at": "2024-09-01T18:00:00Z"}, games) == "publisher_timestamp_after_claimed_knowledge"
    assert validate_fact({**row, "team": "C"}, games) == "team_not_in_game"
    assert validate_fact({**row, "known_at": "2024-09-01T16:00:00Z"}, games) == "fact_not_available_at_kickoff"
    assert validate_fact({**row, "published_at": "2024-09-01T11:00:00Z"}, games) == "publisher_timestamps_not_in_archived_source"
    without_version = {k: v for k, v in row.items() if k != "modified_at"}
    assert validate_fact(without_version, games) == "original_publication_version_unverified"
    assert validate_fact({**without_version, "observed_at": "2026-01-01T00:00:00Z"}, games) == "original_publication_version_unverified"
    observed = {**without_version, "observed_at": "2024-09-01T13:00:00Z"}
    assert validate_fact(observed, games) == "collection_timestamp_unverified"
    raw.with_suffix(".json.source.json").write_text(json.dumps(
        dict(observed_at=observed["observed_at"], sha256=sha(raw))))
    assert validate_fact(observed, games) is None
    raw.write_text("changed report")
    assert validate_fact(row, games) == "source_hash_mismatch"


def test_pdf_creation_is_not_publisher_evidence(tmp_path):
    raw = tmp_path / "report.pdf"
    raw.write_bytes(b"%PDF-file metadata does not establish publication")
    row = dict(game_id=1, season=2024, team="A", name="John Smith", kind="availability", value="out",
               source_path=str(raw), source_url="https://official.example/report", source_sha256=sha(raw),
               known_at="2024-09-01T14:00:00Z", effective_at="2024-09-01T14:00:00Z",
               valid_until="2024-09-02T00:00:00Z", published_at="2024-09-01T12:00:00Z",
               modified_at="2024-09-01T14:00:00Z")
    assert validate_fact(row, schedule().set_index("game_id")) == "pdf_metadata_is_not_publication_evidence"


def test_search_attempts_do_not_imply_healthy_or_exhaustive_search():
    coverage = pd.DataFrame(columns=["game_id", "team", "roster_players", "reports_known", "depth_known"])
    facts = pd.DataFrame(columns=["game_id", "kind"])
    attempts = pd.DataFrame([
        dict(game_id=1, team="A", searched_at="2024-09-02T00:00:00Z", status="source_found_unverified"),
        dict(game_id=1, team="A", searched_at="2024-10-02T00:00:00Z", status="facts_verified_partial")])
    result = coverage_ledger(schedule(), coverage, facts, "2024-09-03T00:00:00Z", attempts)
    row = result[(result.game_id == 1) & (result.team == "A")].iloc[0]
    assert row.search_status == "source_found_unverified"
    assert row.search_attempts == 1
    assert row.reports_known == 0


def test_default_refresh_uses_expanded_registry_and_rejects_tampering(tmp_path, monkeypatch):
    from cfbpred.refresh_player_context import default_reviewed
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    reviewed = tmp_path / "expanded.json"
    reviewed.write_text("[]")
    (tmp_path / "data/player_context_reviewed_latest.json").write_text(json.dumps(
        dict(reviewed=str(reviewed), sha256=sha(reviewed))))
    assert default_reviewed() == reviewed
    reviewed.write_text('[{"changed":true}]')
    with pytest.raises(ValueError, match="changed after validation"):
        default_reviewed()
