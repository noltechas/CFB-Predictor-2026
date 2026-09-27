import json

import pytest

from scripts.publish_full_context_release import _assessment_gate, _documents, _runtime_source


def _doc(path, week, version="player-full-context-tabm16-v4"):
    path.write_text(json.dumps({
        "ranking_week": week,
        "model_version": version,
        "diagnostics": {"strict_global_player_cutoff": True},
    }))


def test_documents_are_discovered_without_hardcoded_week_count(tmp_path):
    _doc(tmp_path / "2026-week-7.json", 7)
    _doc(tmp_path / "2026-week-0.json", 0)
    assert [d["ranking_week"] for d in _documents(tmp_path, 2026, "player-full-context-tabm16-v4")] == [0, 7]


def test_assessment_gate_requires_explicit_acceptance(tmp_path):
    with pytest.raises(ValueError, match="Missing accepted"):
        _assessment_gate(tmp_path)
    p = tmp_path / "assessment_manifest.json"
    p.write_text(json.dumps({"accepted": False}))
    with pytest.raises(ValueError, match="not accepted"):
        _assessment_gate(tmp_path, p)
    p.write_text(json.dumps({"accepted": True, "candidate": "v4"}))
    assert _assessment_gate(tmp_path, p)["candidate"] == "v4"


def test_runtime_source_uses_release_context_and_symlinked_stores(tmp_path):
    source_root = tmp_path / "source"
    for name in ("inputs", "players", "sequences"):
        (source_root / name).mkdir(parents=True)
    source = source_root / "source.json"
    source.write_text(json.dumps({
        "inputs": str(source_root / "inputs"),
        "players": str(source_root / "players"),
        "sequences": str(source_root / "sequences"),
        "full_history": str(source_root / "full_history"),
        "timeline": str(source_root / "timeline"),
    }))
    release = tmp_path / "release"; release.mkdir()
    active = {"input_source": str(source), "bundle": str(tmp_path / "bundle.json")}
    (tmp_path / "bundle.json").write_text(json.dumps({"input_source": str(source)}))
    runtime = _runtime_source(release, active)
    assert (runtime / "inputs").is_symlink()
    provenance = json.loads((runtime / "provenance.json").read_text())
    assert provenance["context_source_sha256"]
    assert json.loads((runtime / "context/source.json").read_text())["source_run"] == str(runtime.resolve())
