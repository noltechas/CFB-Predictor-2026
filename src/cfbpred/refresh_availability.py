"""Lightweight no-training player-context refresh and future publication.

The refresh copies no large stores: each immutable input/player/sequence store is
referenced by a symlink in a dated run, while the dated context timeline is new.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from cfbpred.pregame_experiments import save_json, sha
from cfbpred.config import MODELS, ROOT as PROJECT_ROOT
from cfbpred.player_sequence_production import ACTIVE, active_bundle, audit_snapshot, promote_local, score_snapshot
from cfbpred.refresh_player_context import default_reviewed, refresh as refresh_context

SEQUENCE_REFRESHES = MODELS / "player_sequence_production/refreshes"
ROOT = SEQUENCE_REFRESHES.parent / "availability_refreshes"
LOCK = Path("/tmp/cfbpred_publish.lock")


class RefreshInputsStale(RuntimeError):
    """The immutable daily player stores no longer match the live schedule."""

    exit_code = 42


def _link(dst: Path, target: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def _active_source(active: Path) -> dict:
    _, _, bundle = active_bundle(active)
    source = Path(bundle["input_source"])
    latest = SEQUENCE_REFRESHES / "latest.json"
    if latest.exists():
        run = Path(json.loads(latest.read_text()).get("run_directory", ""))
        for candidate in (run / "context" / "source.json", run / "context_source.json"):
            if candidate.exists():
                source = candidate
                break
    paths = json.loads(source.read_text())
    if not paths.get("full_history"):
        raise ValueError(f"Context source lacks full_history: {source}")
    return paths


def preflight_inputs(paths: dict, current_schedule: Path | None = None) -> dict:
    """Fail before collection when the daily snapshot is behind live results.

    Schedule row order and TBD flags are deliberately ignored. Game identity,
    kickoff, participants, and newly/changed completed results are material.
    """
    import pandas as pd

    baseline_path = Path(paths["inputs"]) / "snapshot_schedule.parquet"
    current_path = current_schedule or (PROJECT_ROOT / "data/parquet/schedule.parquet")
    if not baseline_path.exists() or not current_path.exists():
        raise RefreshInputsStale(f"missing schedule snapshot: baseline={baseline_path} current={current_path}")
    base = pd.read_parquet(baseline_path).set_index("game_id")
    live = pd.read_parquet(current_path).set_index("game_id")
    # The pinned schedule contains all training seasons; the live table holds
    # only the current season. Lower-division-only fixtures are outside the
    # FBS/FCS prediction and team-history population.
    if "season" in base and "season" in live:
        base = base[base.season.isin(live.season.unique())]
    if all(c in base and c in live for c in ("home_division", "away_division")):
        def relevant(frame):
            return frame.home_division.isin(["fbs", "fcs"]) | frame.away_division.isin(["fbs", "fcs"])
        scope = set(base.index[relevant(base)]) | set(live.index[relevant(live)])
        base, live = base.loc[base.index.isin(scope)], live.loc[live.index.isin(scope)]
    def utc(value):
        out = pd.Timestamp(value)
        return out.tz_localize("UTC") if out.tzinfo is None else out.tz_convert("UTC")
    missing = sorted(set(live.index) - set(base.index))
    removed = sorted(set(base.index) - set(live.index))
    material = []
    common = base.index.intersection(live.index)
    for gid in common:
        b, c = base.loc[gid], live.loc[gid]
        bk = utc(b.get("kick", b.get("start_date")))
        ck = utc(c.get("kick", c.get("start_date")))
        if bk != ck:
            material.append(f"kickoff_changed:{gid}:{bk.isoformat()}->{ck.isoformat()}")
        for side in ("home", "away"):
            if str(b.get(f"{side}_team")) != str(c.get(f"{side}_team")):
                material.append(f"participant_changed:{gid}:{side}")
            bp, cp = b.get(f"{side}_points"), c.get(f"{side}_points")
            if pd.notna(cp) and (pd.isna(bp) or float(bp) != float(cp)):
                material.append(f"result_changed:{gid}:{side}")
    if missing or removed or material:
        pieces = []
        if missing: pieces.append(f"new_game_ids={missing[:10]}")
        if removed: pieces.append(f"removed_game_ids={removed[:10]}")
        pieces.extend(material[:10])
        raise RefreshInputsStale("live schedule/results differ from active snapshot: " + "; ".join(pieces))
    return {"passed": True, "baseline": str(baseline_path.resolve()), "current": str(current_path.resolve()), "game_ids": len(common)}


def _safe_retention(root: Path, keep: int = 3) -> None:
    # Keep forecasts and evidence permanently. Only explicitly managed derived
    # arrays may be pruned, with model/training/source dependencies protected.
    from cfbpred.refresh_storage import prune_for_budget
    prune_for_budget(
        root,
        project_root=PROJECT_ROOT,
        required_free_bytes=512 * 1024**2,
        retention_roots=(root.parent / "refreshes",),
    )


def refresh(*, output_root: Path = ROOT, active: Path = ACTIVE, reviewed: Path | None = None,
            publish: bool = True, retain: int = 3) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    from cfbpred.refresh_storage import before_light_refresh, after_successful_refresh
    paths = _active_source(active)
    preflight = preflight_inputs(paths)
    before_light_refresh(
        output_root,
        project_root=PROJECT_ROOT,
        retention_roots=(output_root.parent / "refreshes",),
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = output_root / stamp
    run.mkdir(parents=True, exist_ok=False)
    for name in ("inputs", "players", "sequences"):
        _link(run / name, Path(paths[name]))
    source_store = Path(paths.get("source_run", "")) / "source"
    if source_store.is_dir():
        _link(run / "source", source_store)
    source = run / "context_source.json"
    save_json(source, {"source_run": str(run.resolve()), **{k: str((run / k).resolve()) for k in ("inputs", "players", "sequences")}, "full_history": str(Path(paths["full_history"]).resolve())})
    context = run / "context"
    refresh_context(source, reviewed or default_reviewed(), context, train=False, year=2026)
    forecasts = run / "future_predictions.parquet"
    score_snapshot(run / "inputs", run / "players", run / "sequences", forecasts, active)
    audit = audit_snapshot(run / "inputs", run / "players", run / "sequences", forecasts, active)
    if audit.get("coverage") != "future":
        raise ValueError("availability refresh must audit future predictions only")
    promotion = None
    report = {"passed": True, "module": "refresh_availability", "run_directory": str(run.resolve()),
              "forecast_path": str(forecasts.resolve()), "forecast_sha256": sha(forecasts),
              "audit": audit, "preflight": preflight, "local_promotion": promotion, "published": False}
    if publish:
        promotion = promote_local(forecasts, active.parent)
        old_latest = SEQUENCE_REFRESHES / "latest.json"
        old_bytes = old_latest.read_bytes() if old_latest.exists() else None
        pointer = {"passed": True, "module": "refresh_availability", "run_directory": str(run.resolve()),
                   "forecast_path": str(forecasts.resolve()), "forecast_sha256": sha(forecasts),
                   "production_audit": audit, "local_promotion": promotion}
        pending = SEQUENCE_REFRESHES / f"latest.{stamp}.pending.json"
        save_json(pending, pointer); pending.replace(old_latest)
        try:
            subprocess.run(["python3", "src/cfbpred/publish_future_predictions.py"], check=True)
            subprocess.run(["python3", "src/cfbpred/verify_player_sequence_publish.py"], check=True)
            subprocess.run(["python3", "src/cfbpred/rankings.py", "--season", "2026", "--publish"], check=True)
            report["published"] = True
        except Exception:
            if old_bytes is not None:
                old_latest.write_bytes(old_bytes)
            else:
                old_latest.unlink(missing_ok=True)
            raise
    report["local_promotion"] = promotion
    save_json(run / "availability_refresh.json", report)
    if publish:
        pending = output_root / f"latest.{stamp}.pending.json"
        save_json(pending, report)
        pending.replace(output_root / "latest.json")
    if publish:
        after_successful_refresh(run)
        _safe_retention(output_root, retain)
    return run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, default=ROOT)
    p.add_argument("--reviewed", type=Path)
    p.add_argument("--no-publish", action="store_true")
    args = p.parse_args()
    try:
        refresh(output_root=args.output_root, reviewed=args.reviewed, publish=not args.no_publish)
    except RefreshInputsStale as exc:
        print(f"STALE_INPUTS: {exc}", file=__import__("sys").stderr)
        raise SystemExit(RefreshInputsStale.exit_code)


if __name__ == "__main__":
    main()
