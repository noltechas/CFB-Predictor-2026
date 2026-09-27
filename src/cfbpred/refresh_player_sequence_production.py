"""Rebuild current player inputs, score the active model, and promote locally."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import numpy as np
import pandas as pd

import torch

from cfbpred.config import FEAT, MODELS, ROOT as PROJECT_ROOT
from cfbpred.features_raw import load_games
from cfbpred.player_sequence_inputs import build as build_sequences
from cfbpred.player_sequence_production import ACTIVE, active_bundle, audit_snapshot, promote_local, score_snapshot
from cfbpred.prepare_player_data import prepare
from cfbpred.pregame_experiments import save_json, sha
from cfbpred.recent_form import build as build_form


OUTPUT = MODELS / "player_sequence_production/refreshes"


def active_position_source(active_path: Path = ACTIVE) -> str:
    """Build the player store on the position source the ACTIVE checkpoint was fitted on.

    The roster encoder pools by position group, so a store built with a different
    source is refused at scoring time. Reading it from the deployed checkpoint keeps
    the daily rebuild correct across promotions and rollbacks without a second switch.
    """
    _, bundle_path, bundle = active_bundle(active_path)
    checkpoint = torch.load(bundle_path.parent / bundle["members"][0], map_location="cpu", weights_only=False)
    return checkpoint.get("player_schema", {}).get("position_source", "scouting")


def refresh(feature_dir: Path = FEAT, output_root: Path = OUTPUT, active_path: Path = ACTIVE,
            promote: bool = False) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    from cfbpred.refresh_storage import before_daily_refresh, after_successful_refresh
    # Daily and lightweight availability refreshes share the same disk. Let
    # the retention pass reclaim an old sibling run while preserving each
    # root's current and predecessor runs.
    before_daily_refresh(
        output_root,
        project_root=PROJECT_ROOT,
        retention_roots=(output_root.parent / "availability_refreshes",),
    )
    # The isolated input store is ~650 MB plus intermediate arrays. Fail before
    # writing partial snapshots when the daily job cannot safely fit the rebuild.
    if shutil.disk_usage(output_root).free < 2 * 1024**3:
        raise RuntimeError("Player refresh needs at least 2 GiB of free disk space. No new snapshot was created.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = output_root / stamp
    source, inputs, players, sequences = (run / name for name in ("source", "inputs", "players", "sequences"))
    source.mkdir(parents=True, exist_ok=False)
    for name in ("game_scalars.parquet", "game_history.parquet", "team_vocab.json"):
        shutil.copy2(feature_dir / name, source / name)
    with duckdb.connect() as connection:
        schedule = load_games(connection, fbs_only=False)
    schedule.to_parquet(source / "snapshot_schedule.parquet", index=False, compression="zstd")
    pd.DataFrame({"game_id": pd.read_parquet(source / "game_scalars.parquet", columns=["game_id"]).game_id,
                  "spread_close": np.nan}).to_parquet(source / "market.parquet", index=False)
    build_form(source)
    save_json(source / "source_manifest.json", {p.name: sha(p) for p in source.iterdir() if p.is_file()})
    prepare(source, inputs, players, rebuild_players=True,
            position_source=active_position_source(active_path))
    build_sequences(players, inputs, sequences)
    active, _, bundle = active_bundle(active_path)
    if bundle["architecture"] == "player_full_context_v1":
        from cfbpred.refresh_player_context import default_reviewed, refresh as refresh_context
        context_source = run / "context_source.json"
        save_json(context_source, {"source_run": str(run.resolve()), **{
            k: str(v.resolve()) for k, v in (("inputs", inputs), ("players", players), ("sequences", sequences))}})
        refresh_context(context_source, default_reviewed(), run / "context", train=False)
    forecasts = run / "future_predictions.parquet"
    score_snapshot(inputs, players, sequences, forecasts, active_path)
    audit = audit_snapshot(inputs, players, sequences, forecasts, active_path)
    promotion = promote_local(forecasts, active_path.parent) if promote else None
    report = {
        "passed": True, "run_directory": str(run.resolve()), "forecast_path": str(forecasts.resolve()),
        "forecast_sha256": sha(forecasts), "feature_source": str(feature_dir.resolve()),
        "feature_source_sha256": sha(source / "game_scalars.parquet"),
        "production_audit": audit, "local_promotion": promotion,
    }
    save_json(run / "refresh_report.json", report)
    pending = output_root / f"latest.{stamp}.pending.json"
    save_json(pending, report)
    pending.replace(output_root / "latest.json")
    if promote and output_root.resolve() == OUTPUT.resolve():
        after_successful_refresh(run)
    print(json.dumps(report, indent=2))
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=FEAT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--active", type=Path, default=ACTIVE)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    refresh(args.features, args.output_root, args.active, args.promote)
