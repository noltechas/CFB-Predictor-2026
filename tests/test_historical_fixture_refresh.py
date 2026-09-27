import hashlib
import json

import pandas as pd
import pytest

from cfbpred.historical_fixture_refresh import _database_fixture_identity, _enrich_provider_ids, install, install_stat_aliases


def _row(gid=2, date="2020-09-05T18:00:00Z"):
    return {"game_id": gid, "season": 2020, "week": 1, "season_type": "regular",
            "start_date": date, "neutral_site": False, "conference_game": True,
            "venue_id": None, "home_team": "New Team", "home_conference": "X",
            "home_division": "fbs", "home_points": 24, "away_team": "Old Team",
            "away_conference": "Y", "away_division": "fcs", "away_points": 7,
            "margin": 17, "total": 31}


def _setup(tmp_path, rows):
    canonical = tmp_path / "games.parquet"
    pd.DataFrame([dict(_row(1), home_team="Existing", away_team="Visitor")]).to_parquet(canonical, index=False)
    source = tmp_path / "source.parquet"
    pd.DataFrame(rows).to_parquet(source, index=False)
    pointer = tmp_path / "pointer.json"
    pointer.write_text(json.dumps({"schedule": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}))
    return pointer, canonical


def test_missing_pointer_is_noop(tmp_path):
    canonical = tmp_path / "games.parquet"
    pd.DataFrame([_row(1)]).to_parquet(canonical, index=False)
    result = install(pointer=tmp_path / "missing.json", canonical=canonical)
    assert result["status"] == "no_reviewed_pointer"
    assert len(pd.read_parquet(canonical)) == 1


def test_atomic_install_and_backup(tmp_path):
    pointer, canonical = _setup(tmp_path, [_row()])
    result = install(pointer=pointer, canonical=canonical, backup_dir=tmp_path / "backups")
    assert result["added"] == 1
    assert len(pd.read_parquet(canonical)) == 2
    assert list((tmp_path / "backups").glob("*.bak"))


def test_full_source_is_idempotent(tmp_path, monkeypatch):
    pointer, canonical = _setup(tmp_path, [_row()])
    assert install(pointer=pointer, canonical=canonical)["added"] == 1
    result = install(pointer=pointer, canonical=canonical)
    assert result["status"] == "already_current"
    assert len(pd.read_parquet(canonical)) == 2


def test_db_retry_after_local_install(tmp_path, monkeypatch):
    pointer, canonical = _setup(tmp_path, [_row()])
    install(pointer=pointer, canonical=canonical)
    calls = []
    monkeypatch.setattr("cfbpred.historical_fixture_refresh.publish_db",
                        lambda rows: calls.append(len(rows)) or {"rows": len(rows)})
    result = install(pointer=pointer, canonical=canonical, publish=True)
    assert result["status"] == "db_retry"
    assert calls == [1]


def test_sha_is_required(tmp_path):
    pointer, canonical = _setup(tmp_path, [_row()])
    doc = json.loads(pointer.read_text()); del doc["sha256"]; pointer.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="SHA-256"):
        install(pointer=pointer, canonical=canonical)


def test_provider_ids_are_enriched_without_overwriting_existing(tmp_path):
    rows = pd.DataFrame([{"game_id": 2, "home_id": None, "away_id": 77}])
    raw = tmp_path / "raw.parquet"
    pd.DataFrame([{"id": 2, "homeId": 55, "awayId": 66}]).to_parquet(raw, index=False)
    got = _enrich_provider_ids(rows, raw)
    assert int(got.iloc[0].home_id) == 55
    assert int(got.iloc[0].away_id) == 77


def test_stat_alias_hook_is_atomic_and_idempotent(tmp_path):
    aliases = tmp_path / "aliases.json"
    aliases.write_text(json.dumps({"10": 20}))
    stats = tmp_path / "stats.parquet"
    frame = pd.DataFrame([{"game_id": 10, "home_away": "home", "athlete_id": "a",
                           "category": "passing", "stat_type": "YDS", "stat": 1,
                           "stat_value": "1"}])
    frame.to_parquet(stats, index=False)
    digest = hashlib.sha256(aliases.read_bytes()).hexdigest()
    pointer = {"stat_aliases": str(aliases), "stat_aliases_sha256": digest}
    result = install_stat_aliases(pointer, stats_path=stats, backup_dir=tmp_path / "backups")
    assert result["remapped"] == 1
    assert int(pd.read_parquet(stats).iloc[0].game_id) == 20
    result2 = install_stat_aliases(pointer, stats_path=stats, backup_dir=tmp_path / "backups")
    assert result2["status"] == "already_current"
    assert len(list((tmp_path / "backups").glob("*.bak"))) == 1


def test_duplicate_normalized_fixture_rejected(tmp_path):
    pointer, canonical = _setup(tmp_path, [_row(2), _row(3)])
    with pytest.raises(ValueError, match="duplicate normalized fixtures"):
        install(pointer=pointer, canonical=canonical)


def test_existing_normalized_fixture_rejected(tmp_path):
    pointer, canonical = _setup(tmp_path, [{**_row(), "game_id": 9, "home_team": "Existing", "away_team": "Visitor"}])
    with pytest.raises(ValueError, match="conflicts"):
        install(pointer=pointer, canonical=canonical)


def test_database_readback_accepts_reviewed_school_alias_but_not_changed_fixture():
    aliases = {"Appalachian State": "App State"}
    row = [1, 2008, "2008-08-30T15:00Z", "LSU", "Appalachian State", 41, 13, 99, 2026]
    expected = _database_fixture_identity(*row, aliases)
    row[4] = "App State"
    assert _database_fixture_identity(*row, aliases) == expected
    for index, changed in [(5, 40), (8, 999), (2, "2008-08-30T16:00Z"), (4, "Other School")]:
        different = row.copy(); different[index] = changed
        assert _database_fixture_identity(*different, aliases) != expected
