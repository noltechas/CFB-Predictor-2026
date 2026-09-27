"""Independent audit for the frozen 2026 player-sequence candidate.

This module deliberately lives outside the candidate's training-code hash set.
It lets us validate an immutable trained bundle without changing the source that
created it.  Saved parquet list columns are explicitly restored as float32,
matching the dtype used when the distributions were produced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cfbpred.ensemble import readouts
from cfbpred.model import MARGIN_CENTRES, TOTAL_CENTRES
from cfbpred.player_sequence_experiment import INPUTS, PLAYERS, SEQUENCES, ROOT, load
from cfbpred.pregame_experiments import save_json, sha


DEFAULT = ROOT / "models/experiments/player_sequences_20260907_v2/current_2026_context_v2"


def audit(directory: Path = DEFAULT, forecasts: Path | None = None) -> dict:
    forecasts = directory / "future_predictions.parquet" if forecasts is None else forecasts
    plan_path = directory / "plan.json"
    bundle_path = directory / "bundle.json"
    meta_path = forecasts.with_suffix(".json")
    plan = json.loads(plan_path.read_text())
    bundle = json.loads(bundle_path.read_text())
    meta = json.loads(meta_path.read_text())

    if sha(plan_path) != bundle["plan_sha256"]:
        raise ValueError("Bundle plan hash differs")
    for name, digest in plan["code"].items():
        if sha(Path(__file__).parent / name) != digest:
            raise ValueError(f"Frozen candidate code changed: {name}")
    for name, digest in plan["inputs"].items():
        if sha(INPUTS / name) != digest:
            raise ValueError(f"Frozen candidate input changed: {name}")
    if sha(PLAYERS / "manifest.json") != plan["player_manifest_sha256"]:
        raise ValueError("Player manifest changed")
    if sha(SEQUENCES / "manifest.json") != plan["sequence_manifest_sha256"]:
        raise ValueError("Player-sequence manifest changed")
    if sha(bundle_path) != meta["bundle_sha256"] or sha(forecasts) != meta["forecast_sha256"]:
        raise ValueError("Forecast lineage differs")

    frame = pd.read_parquet(forecasts)
    base = load()
    schedule = pd.read_parquet(INPUTS / "snapshot_schedule.parquet").set_index("game_id")
    cutoff = pd.Timestamp(meta["cutoff"])
    if cutoff.tzinfo is None:
        raise ValueError("Forecast cutoff is timezone-naive")
    kick = pd.to_datetime(schedule.loc[base["G_gid"], "kick"], utc=True)
    expected = base["G_gid"][(base["G_season"] == 2026) & (kick > cutoff).to_numpy()]
    if frame.game_id.duplicated().any() or set(frame.game_id) != set(expected):
        raise ValueError("Future forecast coverage differs")
    if not frame["snapshot_at"].eq(cutoff).all():
        raise ValueError("Forecast rows do not share the recorded cutoff")
    if not frame["model"].eq("player_sequence_context_v1").all() or not frame["status"].eq("retained_candidate").all():
        raise ValueError("Forecast model identity differs")
    scheduled = schedule.loc[frame.game_id]
    if not frame.home_team.reset_index(drop=True).equals(scheduled.home_team.reset_index(drop=True)):
        raise ValueError("Forecast home teams differ from schedule")
    if not frame.away_team.reset_index(drop=True).equals(scheduled.away_team.reset_index(drop=True)):
        raise ValueError("Forecast away teams differ from schedule")
    if not pd.to_datetime(frame.kick, utc=True).reset_index(drop=True).equals(pd.to_datetime(scheduled.kick, utc=True).reset_index(drop=True)):
        raise ValueError("Forecast kickoffs differ from schedule")

    pm = torch.as_tensor(np.stack(frame.margin_pmf), dtype=torch.float32)
    pt = torch.as_tensor(np.stack(frame.total_pmf), dtype=torch.float32)
    if pm.shape != (len(frame), len(MARGIN_CENTRES)) or pt.shape != (len(frame), len(TOTAL_CENTRES)):
        raise ValueError("Saved probability dimensions differ")
    max_mass_error = 0.0
    for values in (pm, pt):
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError("Saved probabilities are invalid")
        max_mass_error = max(max_mass_error, float((values.sum(-1) - 1).abs().max()))
        if max_mass_error > 1e-5:
            raise ValueError("Saved probability mass differs from one")
    derived = readouts(pm, pt)
    max_readout_error = 0.0
    for col, key in (("home_win_prob", "wp"), ("pred_margin", "em"), ("pred_total", "et")):
        error = float(np.max(np.abs(frame[col].to_numpy() - derived[key])))
        max_readout_error = max(max_readout_error, error)
        if error > 1e-5:
            raise ValueError(f"Saved {col} differs from its PMF")

    roster = pd.read_parquet(PLAYERS / "game_rosters.parquet").set_index(["game_id", "side"])
    target_index = pd.MultiIndex.from_product([expected, ["home", "away"]], names=["game_id", "side"])
    if not target_index.isin(roster.index).all():
        raise ValueError("A future team-game roster snapshot is absent")
    target = roster.loc[target_index].reset_index()
    target["division"] = [schedule.loc[row.game_id, row.side + "_division"] for row in target.itertuples()]
    fbs = target.division.eq("fbs")
    if target.loc[fbs, "players"].eq(0).any():
        raise ValueError("A future FBS roster is empty")
    observed = target.last_history_kick.notna()
    if (pd.to_datetime(target.loc[observed, "last_history_kick"], utc=True).to_numpy()
            >= pd.to_datetime(target.loc[observed, "kick"], utc=True).to_numpy()).any():
        raise ValueError("A player history observation is not strictly pregame")

    members = []
    for name in bundle["members"]:
        path = directory / name
        if sha(path) != bundle["member_sha256"][name]:
            raise ValueError(f"Candidate member changed: {name}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        seasons = schedule.loc[checkpoint["train_game_ids"], "season"]
        if seasons.ge(2026).any() or int(seasons.max()) != 2025:
            raise ValueError("A 2026 outcome entered candidate fitting")
        if checkpoint["variant"] != "context" or not checkpoint["pregame_only"]:
            raise ValueError("Candidate checkpoint semantics differ")
        members.append({
            "member": name,
            "seed": int(checkpoint["seed"]),
            "epochs": int(checkpoint["epochs"]),
            "training_games": int(len(seasons)),
            "latest_training_season": int(seasons.max()),
        })

    report = {
        "passed": True,
        "auditor": "player_sequence_current_audit_v1",
        "future_games": int(len(frame)),
        "future_team_sides": int(len(target)),
        "future_fbs_sides": int(fbs.sum()),
        "missing_future_fbs_rosters": 0,
        "strict_player_history_before_kickoff": True,
        "no_2026_outcomes_used": True,
        "all_expected_future_games_scored": True,
        "frozen_code_and_inputs_match": True,
        "probability_distributions_valid": True,
        "max_probability_mass_error": max_mass_error,
        "max_pmf_readout_error": max_readout_error,
        "members": members,
        "production_modified": False,
    }
    save_json(directory / "forecast_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--forecasts", type=Path)
    args = parser.parse_args()
    audit(args.directory, args.forecasts)
