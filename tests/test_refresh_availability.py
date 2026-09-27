import json
from pathlib import Path
import pytest

import cfbpred.refresh_availability as mod
from cfbpred.refresh_availability import _safe_retention, _link, RefreshInputsStale, preflight_inputs
import pandas as pd


def test_link_and_retention_never_delete_target(tmp_path):
    target = tmp_path / "immutable"
    target.mkdir(); (target / "keep.txt").write_text("keep")
    root = tmp_path / "runs"; root.mkdir()
    old = root / "old"; old.mkdir(); (old / "availability_refresh.json").write_text("{}")
    _link(old / "players", target)
    latest = root / "latest.json"
    latest.write_text(json.dumps({"run_directory": str(old)}))
    _safe_retention(root, keep=0)
    assert target.exists() and (target / "keep.txt").exists()
    assert old.exists()  # latest is protected


def test_no_publish_is_read_only_and_uses_current_source(tmp_path, monkeypatch):
    target = tmp_path / "source_store"
    for name in ("inputs", "players", "sequences"):
        (target / name).mkdir(parents=True, exist_ok=True)
    (target / "full_history").mkdir()
    output = tmp_path / "availability"
    calls = []

    monkeypatch.setattr(mod, "_active_source", lambda active: {
        "inputs": str(target / "inputs"), "players": str(target / "players"),
        "sequences": str(target / "sequences"), "full_history": str(target / "full_history")})
    monkeypatch.setattr(mod, "preflight_inputs", lambda paths: {"passed": True})
    monkeypatch.setattr(mod, "default_reviewed", lambda: tmp_path / "reviewed.json")
    (tmp_path / "reviewed.json").write_text("[]")
    monkeypatch.setattr(mod, "refresh_context", lambda *a, **k: calls.append("context"))
    monkeypatch.setattr(mod, "score_snapshot", lambda *a, **k: calls.append("score") or Path(a[3]).write_text("forecast"))
    monkeypatch.setattr(mod, "audit_snapshot", lambda *a, **k: {"coverage": "future", "passed": True})
    monkeypatch.setattr(mod, "promote_local", lambda *a, **k: calls.append("promote"))
    monkeypatch.setattr(mod, "sha", lambda p: "digest")
    before = (mod.SEQUENCE_REFRESHES / "latest.json").read_bytes() if (mod.SEQUENCE_REFRESHES / "latest.json").exists() else None
    run = mod.refresh(output_root=output, publish=False, active=tmp_path / "active.json")
    assert run.exists() and (run / "inputs").is_symlink()
    assert calls == ["context", "score"]
    assert not (output / "latest.json").exists()
    after = (mod.SEQUENCE_REFRESHES / "latest.json").read_bytes() if (mod.SEQUENCE_REFRESHES / "latest.json").exists() else None
    assert before == after


def test_preflight_ignores_order_and_tbd_but_rejects_kickoff_and_results(tmp_path):
    base = pd.DataFrame([{"game_id": 1, "kick": "2026-09-01T16:00:00Z", "home_team": "A",
                          "away_team": "B", "home_points": None, "away_points": None,
                          "start_time_tbd": True}])
    live = base.sample(frac=1, random_state=4).copy()
    live["start_time_tbd"] = False
    bp = tmp_path / "inputs"; bp.mkdir(); base.to_parquet(bp / "snapshot_schedule.parquet", index=False)
    lp = tmp_path / "schedule.parquet"; live.to_parquet(lp, index=False)
    assert preflight_inputs({"inputs": str(bp)}, lp)["passed"]
    changed = live.copy(); changed.loc[0, "kick"] = "2026-09-01T17:00:00Z"
    changed.to_parquet(lp, index=False)
    with pytest.raises(RefreshInputsStale, match="kickoff_changed"):
        preflight_inputs({"inputs": str(bp)}, lp)
    changed = base.copy(); changed.loc[0, "home_points"] = 21
    changed.to_parquet(lp, index=False)
    with pytest.raises(RefreshInputsStale, match="result_changed"):
        preflight_inputs({"inputs": str(bp)}, lp)


def test_preflight_compares_only_current_division_one_schedule(tmp_path):
    import pandas as pd
    base=tmp_path/'inputs';base.mkdir()
    rows=[dict(game_id=i,season=y,kick=f'{y}-09-01T18:00:00Z',home_team='A',away_team='B',home_points=10,away_points=7,home_division=d,away_division=d) for i,y,d in [(1,2025,'fbs'),(2,2026,'fbs'),(3,2026,'ii')]]
    pd.DataFrame(rows).to_parquet(base/'snapshot_schedule.parquet')
    live=pd.DataFrame(rows[1:]);live.loc[live.game_id.eq(3),'home_points']=50
    p=tmp_path/'live.parquet';live.to_parquet(p)
    assert preflight_inputs({'inputs':str(base)},p)['game_ids']==1
    live.loc[live.game_id.eq(2),'home_points']=20;live.to_parquet(p)
    with pytest.raises(RefreshInputsStale):preflight_inputs({'inputs':str(base)},p)
