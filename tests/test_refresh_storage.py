import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cfbpred.refresh_storage import (
    InsufficientStorageError,
    after_successful_refresh,
    before_refresh,
)


def _run(root: Path, name: str, managed: bool = False) -> Path:
    run = root / name
    run.mkdir()
    (run / "predictions.parquet").write_bytes(b"keep")
    (run / "reports").mkdir()
    (run / "reports" / "report.json").write_text("{}")
    (run / "sequences").mkdir()
    (run / "sequences" / "large.bin").write_bytes(b"derived")
    if managed:
        after_successful_refresh(run)
    return run


def test_budget_does_not_touch_preexisting_runs(tmp_path):
    root = tmp_path / "refreshes"; root.mkdir()
    old = _run(root, "old")
    with patch("cfbpred.refresh_storage.shutil.disk_usage", return_value=(0, 0, 0)):
        with pytest.raises(InsufficientStorageError):
            before_refresh(root, project_root=tmp_path, required_free_bytes=10)
    assert (old / "sequences" / "large.bin").exists()


def test_prunes_only_unpinned_managed_store_and_records_metadata(tmp_path):
    root = tmp_path / "refreshes"; root.mkdir()
    current = _run(root, "20260903")
    predecessor = _run(root, "20260902")
    candidate = _run(root, "20260901", managed=True)
    # Make the live pointer pin current explicitly.
    (root / "latest.json").write_text(json.dumps({"run_directory": str(current)}))

    def usage(*_):
        return (0, 0, 10 if (candidate / "sequences").exists() else 20)

    with patch("cfbpred.refresh_storage.shutil.disk_usage", side_effect=usage):
        result = before_refresh(root, project_root=tmp_path, required_free_bytes=15)
    assert str(candidate) in result.pruned_runs
    assert not (candidate / "sequences").exists()
    assert (candidate / "predictions.parquet").exists()
    assert json.loads((candidate / "retention.json").read_text())["pruned"] == ["sequences"]
    assert (current / "sequences").exists()
    assert (predecessor / "sequences").exists()


def test_symlink_target_is_never_deleted(tmp_path):
    root = tmp_path / "refreshes"; root.mkdir()
    target = _run(root, "20260902-target", managed=True)
    candidate = _run(root, "20260901-candidate", managed=True)
    _run(root, "20260903-anchor")
    (candidate / "full_history").unlink(missing_ok=True)
    (candidate / "full_history").symlink_to(target / "sequences", target_is_directory=True)
    with patch("cfbpred.refresh_storage.shutil.disk_usage", side_effect=lambda *_: (0, 0, 10 if (candidate / "sequences").exists() else 20)):
        before_refresh(root, project_root=tmp_path, required_free_bytes=15)
    assert (target / "sequences" / "large.bin").exists()


def test_pinned_model_source_prevents_pruning(tmp_path):
    root = tmp_path / "refreshes"; root.mkdir()
    pinned = _run(root, "20260901-pinned", managed=True)
    _run(root, "20260902-other", managed=True)
    _run(root, "20260831-candidate", managed=True)
    _run(root, "20260903-anchor")
    model = tmp_path / "models" / "experiments" / "bundle"; model.mkdir(parents=True)
    (model / "source.json").write_text(json.dumps({"run_directory": str(pinned)}))
    with patch("cfbpred.refresh_storage.shutil.disk_usage", side_effect=lambda *_: (0, 0, 10 if (root / "20260831-candidate" / "sequences").exists() else 20)):
        before_refresh(root, project_root=tmp_path, required_free_bytes=15)
    assert (pinned / "sequences").exists()


def test_active_runtime_pointer_prevents_pruning(tmp_path):
    root = tmp_path / "refreshes"; root.mkdir()
    pinned = _run(root, "20260901-pinned", managed=True)
    _run(root, "20260902-other", managed=True)
    candidate = _run(root, "20260831-candidate", managed=True)
    _run(root, "20260903-anchor")
    active = tmp_path / "models" / "player_sequence_production"
    active.mkdir(parents=True)
    (active / "active.json").write_text(json.dumps({"source_candidate": str(pinned)}))
    with patch("cfbpred.refresh_storage.shutil.disk_usage",
               side_effect=lambda *_: (0, 0, 10 if (candidate / "sequences").exists() else 20)):
        before_refresh(root, project_root=tmp_path, required_free_bytes=15)
    assert (pinned / "sequences").exists()


def test_self_referential_report_does_not_pin_unmanaged_candidate(tmp_path):
    root = tmp_path / "refreshes"; root.mkdir()
    _run(root, "20260903-anchor")
    _run(root, "20260902-predecessor")
    candidate = _run(root, "20260901-candidate", managed=True)
    (candidate / "refresh_report.json").write_text(json.dumps({"run_directory": str(candidate)}))
    with patch("cfbpred.refresh_storage.shutil.disk_usage", side_effect=lambda *_: (0, 0, 10 if (candidate / "sequences").exists() else 20)):
        before_refresh(root, project_root=tmp_path, required_free_bytes=15)
    assert not (candidate / "sequences").exists()


def test_cross_root_cleanup_keeps_two_live_runs_per_root(tmp_path):
    daily = tmp_path / "refreshes"
    light = tmp_path / "availability_refreshes"
    daily.mkdir(); light.mkdir()
    daily_old = _run(daily, "20260901", managed=True)
    _run(daily, "20260902")
    _run(daily, "20260903")
    light_old = _run(light, "20260901", managed=True)
    _run(light, "20260902")
    _run(light, "20260903")

    def usage(*_):
        pending = daily_old / "sequences" / "large.bin"
        pending_light = light_old / "sequences" / "large.bin"
        return (0, 0, 10 if pending.exists() or pending_light.exists() else 20)

    with patch("cfbpred.refresh_storage.shutil.disk_usage", side_effect=usage):
        result = before_refresh(daily, project_root=tmp_path, required_free_bytes=15,
                                retention_roots=(light,))
    assert str(daily_old) in result.pruned_runs
    assert str(light_old) in result.pruned_runs
    for root in (daily, light):
        assert (root / "20260903" / "sequences" / "large.bin").exists()
        assert (root / "20260902" / "sequences" / "large.bin").exists()


def test_historical_availability_sources_only_pin_live_two_runs(tmp_path):
    root = tmp_path / "refreshes"; root.mkdir()
    candidate = _run(root, "20260901", managed=True)
    _run(root, "20260902", managed=True)
    _run(root, "20260903", managed=True)
    availability = tmp_path / "models" / "player_sequence_production" / "availability_refreshes"
    for stamp in ("20260901", "20260902", "20260903"):
        context = availability / stamp / "context"
        context.mkdir(parents=True)
        # The old source would pin the historical candidate indefinitely. The
        # active-two policy intentionally does not scan this old source.
        target = candidate if stamp == "20260901" else root / stamp / "inputs"
        (context / "source.json").write_text(json.dumps({"inputs": str(target)}))

    with patch("cfbpred.refresh_storage.shutil.disk_usage",
               side_effect=lambda *_: (0, 0, 10 if (candidate / "sequences").exists() else 20)):
        before_refresh(root, project_root=tmp_path, required_free_bytes=15)
    assert not (candidate / "sequences").exists()


def test_obsolete_light_symlink_does_not_pin_old_daily_store(tmp_path):
    from cfbpred.refresh_storage import _referenced_runs
    daily=tmp_path/'daily'; light=tmp_path/'light';daily.mkdir();light.mkdir()
    d=[_run(daily,f'2026090{i}',managed=True) for i in range(1,5)]
    l=[_run(light,f'2026090{i}',managed=True) for i in range(1,4)]
    (l[0]/'inputs').symlink_to(d[0]/'sequences',target_is_directory=True)
    (l[1]/'inputs').symlink_to(d[1]/'sequences',target_is_directory=True)
    pins=_referenced_runs(tmp_path,d+l)
    assert d[0].resolve() not in pins
    assert d[1].resolve() in pins  # retained predecessor dependency
