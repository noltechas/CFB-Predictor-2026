"""Refit the development-selected TabM player architecture for 2026.

The model-selection decision is frozen in the architecture research report.
This module selects only the number of training epochs on 2025, refits each
seed through 2025, and scores future 2026 games without using 2026 outcomes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cfbpred.architecture_research_experiment import OUTPUT as RESEARCH, fold_data
from cfbpred.architecture_research_net import make
from cfbpred.ensemble import readouts
from cfbpred.model import MARGIN_CENTRES, TOTAL_CENTRES, scores_from
from cfbpred.player_sequence_experiment import INPUTS, PLAYERS, SEQUENCES, ROOT, load
from cfbpred.player_sequence_fast_inputs import FastSequenceStore
from cfbpred.player_sequence_train import fit, predict
from cfbpred.pregame_experiments import save_json, sha


OUTPUT = RESEARCH / "current_2026_tabm16"
ARCHITECTURE = "architecture_research_v1"
VARIANT = "tabm16"
MODEL_VERSION = "player-sequence-tabm16-v2"
CODE = [
    "architecture_research_net.py", "architecture_research_forecast.py",
    "player_sequence_inputs.py", "player_sequence_fast_inputs.py",
    "player_sequence_net.py", "player_sequence_train.py", "player_sequence_experiment.py",
    "player_inputs.py", "player_net.py", "train.py", "raw_net.py", "consistent_net.py",
    "model.py",
]


def initialize(threads: int) -> None:
    global BASE
    torch.set_num_threads(threads)
    BASE = load()
    BASE["player_store"].__class__ = FastSequenceStore


def train_member(seed: int, directory: Path, forecast_year: int = 2026,
                 micro_batch_size: int = 256) -> str:
    base = BASE
    train_mask = base["G_labeled"] & (base["G_season"] < forecast_year - 1)
    validation_mask = (base["G_labeled"] & (base["G_season"] == forecast_year - 1)
                       & base["G_fbs_inv"])
    refit_mask = base["G_labeled"] & (base["G_season"] < forecast_year)
    fitted, refitted = fold_data(base, train_mask), fold_data(base, refit_mask)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make(fitted, VARIANT)
    fit(model, fitted, torch.device("cpu"), np.where(train_mask)[0],
        np.where(validation_mask)[0], micro_batch_size=micro_batch_size,
        log_path=directory / f"s{seed}_selection.json")
    epochs = model.pre.selected_epochs
    del model

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make(refitted, VARIANT)
    fit(model, refitted, torch.device("cpu"), np.where(refit_mask)[0],
        np.where(validation_mask)[0], fixed_epochs=epochs,
        micro_batch_size=micro_batch_size,
        log_path=directory / f"s{seed}_refit.json")
    checkpoint = {
        "architecture": ARCHITECTURE, "variant": VARIANT, "model_version": MODEL_VERSION,
        "pregame_only": True,
        "state_dict": {key: value.detach().cpu() for key, value in model.pre.state_dict().items()},
        "mu": refitted["mu"], "sd": refitted["sd"],
        "scalar_cols": base["scalar_cols"], "vocab": base["vocab"],
        "known_sequence_teams": refitted["player_store"].known_teams,
        "known_teams": np.unique(np.concatenate([
            refitted[field].numpy().ravel() for field in ("G_hidx", "G_aidx", "G_opp")])),
        "known_confs": np.unique(np.concatenate([
            refitted[field].numpy().ravel() for field in ("G_hconf", "G_aconf")])),
        "forecast_year": forecast_year, "seed": seed, "epochs": epochs,
        "train_game_ids": base["G_gid"][refit_mask],
        "train_max_season": int(base["G_season"][refit_mask].max()),
        "player_schema": base["player_store"].manifest,
        "sequence_schema": base["player_store"].sequence_manifest,
    }
    path = directory / f"s{seed}.pt"
    torch.save(checkpoint, path)
    print(f"TabM production candidate seed={seed} epochs={epochs}", flush=True)
    return path.name


def train(directory: Path = OUTPUT, seeds: int = 3, workers: int = 3,
          threads: int = 2, micro_batch_size: int = 256) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    selection = json.loads((RESEARCH / "selection.json").read_text())
    if selection["selected"] != VARIANT:
        raise ValueError("TabM was not the frozen development selection")
    plan = {
        "architecture": ARCHITECTURE, "variant": VARIANT, "model_version": MODEL_VERSION,
        "forecast_year": 2026, "seeds": list(range(seeds)), "workers": workers,
        "threads_per_worker": threads, "micro_batch": micro_batch_size,
        "selection_year": 2025, "refit_max_year": 2025, "2026_labels_used": False,
        "selection_sha256": sha(RESEARCH / "selection.json"),
        "assessment_report_sha256": sha(RESEARCH / "report.json"),
        "inputs": {p.name: sha(p) for p in INPUTS.iterdir() if p.is_file()},
        "player_manifest_sha256": sha(PLAYERS / "manifest.json"),
        "sequence_manifest_sha256": sha(SEQUENCES / "manifest.json"),
        "code": {name: sha(Path(__file__).parent / name) for name in CODE},
    }
    plan_path = directory / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Production candidate plan changed; use a new directory")
    save_json(plan_path, plan)
    with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize, initargs=(threads,)) as pool:
        jobs = {
            pool.submit(train_member, seed, directory, 2026, micro_batch_size): seed
            for seed in range(seeds) if not (directory / f"s{seed}.pt").exists()
        }
        for job in as_completed(jobs):
            job.result()
    members = [f"s{seed}.pt" for seed in range(seeds)]
    if not all((directory / name).exists() for name in members):
        raise ValueError("Incomplete TabM production candidate")
    save_json(directory / "bundle.json", {
        "architecture": ARCHITECTURE, "variant": VARIANT, "model_version": MODEL_VERSION,
        "pregame_only": True, "forecast_year": 2026, "members": members,
        "member_sha256": {name: sha(directory / name) for name in members},
        "plan_sha256": sha(plan_path),
    })


def score(directory: Path = OUTPUT, output: Path | None = None, cutoff=None) -> pd.DataFrame:
    output = directory / "future_predictions.parquet" if output is None else output
    bundle = json.loads((directory / "bundle.json").read_text())
    if sha(directory / "plan.json") != bundle["plan_sha256"]:
        raise ValueError("Candidate plan changed")
    base = load()
    base["player_store"].__class__ = FastSequenceStore
    refit_mask = base["G_labeled"] & (base["G_season"] < bundle["forecast_year"])
    data = fold_data(base, refit_mask)
    schedule = pd.read_parquet(INPUTS / "snapshot_schedule.parquet").set_index("game_id")
    cutoff = pd.Timestamp.now(tz="UTC") if cutoff is None else pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("Cutoff needs timezone")
    kick = pd.to_datetime(schedule.loc[base["G_gid"], "kick"], utc=True)
    indices = np.where((base["G_season"] == 2026) & (kick > cutoff).to_numpy())[0]
    sums = [None, None]
    for name in bundle["members"]:
        path = directory / name
        if sha(path) != bundle["member_sha256"][name]:
            raise ValueError("Candidate checkpoint changed")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["train_max_season"] >= 2026 or checkpoint["variant"] != VARIANT:
            raise ValueError("Invalid candidate checkpoint")
        if not np.array_equal(checkpoint["mu"], data["mu"]) or not np.array_equal(checkpoint["sd"], data["sd"]):
            raise ValueError("Candidate normalization differs")
        model = make(data, VARIANT)
        model.pre.load_state_dict(checkpoint["state_dict"])
        pm, pt = predict(model, data, indices, torch.device("cpu"))
        sums = [pm if sums[0] is None else sums[0] + pm,
                pt if sums[1] is None else sums[1] + pt]
        del model
    pm, pt = (value / len(bundle["members"]) for value in sums)
    for values in (pm, pt):
        if (not torch.isfinite(values).all() or (values < 0).any()
                or not torch.allclose(values.sum(-1), torch.ones(len(indices)), atol=1e-5)):
            raise ValueError("Invalid candidate probability distribution")
    derived = readouts(pm, pt)
    home, away = scores_from(torch.tensor(derived["mm"]), torch.tensor(derived["mt"]))
    frame = pd.DataFrame({
        "game_id": base["G_gid"][indices], "season": base["G_season"][indices],
        "home_win_prob": derived["wp"], "pred_margin": derived["em"],
        "pred_total": derived["et"], "pred_home_pts": home.numpy().astype(int),
        "pred_away_pts": away.numpy().astype(int), "split": "test",
        "margin_pmf": pm.numpy().tolist(), "total_pmf": pt.numpy().tolist(),
        "snapshot_at": cutoff, "model": MODEL_VERSION,
    }).merge(schedule[["home_team", "away_team", "kick"]], left_on="game_id", right_index=True,
             validate="one_to_one")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False, compression="zstd")
    save_json(output.with_suffix(".json"), {
        "version": MODEL_VERSION, "future_games": len(frame), "cutoff": cutoff.isoformat(),
        "members": len(bundle["members"]), "bundle_sha256": sha(directory / "bundle.json"),
        "forecast_sha256": sha(output), "all_future_targets_scored": True,
        "no_2026_outcomes_used": True,
    })
    print(f"Scored {len(frame):,} future games with TabM candidate", flush=True)
    return frame


def audit(directory: Path = OUTPUT, forecasts: Path | None = None) -> dict:
    forecasts = directory / "future_predictions.parquet" if forecasts is None else forecasts
    plan = json.loads((directory / "plan.json").read_text())
    bundle = json.loads((directory / "bundle.json").read_text())
    for name, digest in plan["code"].items():
        if sha(Path(__file__).parent / name) != digest:
            raise ValueError(f"Candidate code changed: {name}")
    for name, digest in plan["inputs"].items():
        if sha(INPUTS / name) != digest:
            raise ValueError(f"Candidate input changed: {name}")
    meta = json.loads(forecasts.with_suffix(".json").read_text())
    frame = pd.read_parquet(forecasts)
    if sha(directory / "bundle.json") != meta["bundle_sha256"] or sha(forecasts) != meta["forecast_sha256"]:
        raise ValueError("Candidate forecast lineage differs")
    base = load()
    schedule = pd.read_parquet(INPUTS / "snapshot_schedule.parquet").set_index("game_id")
    cutoff = pd.Timestamp(meta["cutoff"])
    kick = pd.to_datetime(schedule.loc[base["G_gid"], "kick"], utc=True)
    expected = base["G_gid"][(base["G_season"] == 2026) & (kick > cutoff).to_numpy()]
    if frame.game_id.duplicated().any() or set(frame.game_id) != set(expected):
        raise ValueError("Future candidate coverage differs")
    pm = torch.tensor(np.stack(frame.margin_pmf))
    pt = torch.tensor(np.stack(frame.total_pmf))
    if pm.shape != (len(frame), len(MARGIN_CENTRES)) or pt.shape != (len(frame), len(TOTAL_CENTRES)):
        raise ValueError("Candidate PMF dimensions differ")
    max_mass = max(float((pm.sum(-1) - 1).abs().max()), float((pt.sum(-1) - 1).abs().max()))
    derived = readouts(pm, pt)
    max_readout = max(float(np.max(np.abs(frame[column].to_numpy() - derived[key])))
                      for column, key in (("home_win_prob", "wp"), ("pred_margin", "em"),
                                          ("pred_total", "et")))
    if max_mass > 1e-5 or max_readout > 1e-5 or not torch.isfinite(pm).all() or not torch.isfinite(pt).all():
        raise ValueError("Candidate PMFs are invalid")
    roster = pd.read_parquet(PLAYERS / "game_rosters.parquet").set_index(["game_id", "side"])
    target_index = pd.MultiIndex.from_product([expected, ["home", "away"]], names=["game_id", "side"])
    if not target_index.isin(roster.index).all():
        raise ValueError("Future candidate roster reference missing")
    target = roster.loc[target_index].reset_index()
    target["division"] = [schedule.loc[row.game_id, row.side + "_division"] for row in target.itertuples()]
    fbs = target.division.eq("fbs")
    if target.loc[fbs, "players"].eq(0).any():
        raise ValueError("Future FBS candidate roster is empty")
    observed = target.last_history_kick.notna()
    if (pd.to_datetime(target.loc[observed, "last_history_kick"], utc=True).to_numpy()
            >= pd.to_datetime(target.loc[observed, "kick"], utc=True).to_numpy()).any():
        raise ValueError("Future player history is not strictly pregame")
    members = []
    for name in bundle["members"]:
        checkpoint = torch.load(directory / name, map_location="cpu", weights_only=False)
        seasons = schedule.loc[checkpoint["train_game_ids"], "season"]
        if seasons.ge(2026).any() or int(seasons.max()) != 2025:
            raise ValueError("2026 outcome entered candidate fitting")
        members.append({"member": name, "seed": int(checkpoint["seed"]),
                        "epochs": int(checkpoint["epochs"]), "training_games": len(seasons),
                        "latest_training_season": int(seasons.max())})
    report = {
        "passed": True, "architecture": ARCHITECTURE, "variant": VARIANT,
        "version": MODEL_VERSION, "future_games": len(frame), "future_team_sides": len(target),
        "future_fbs_sides": int(fbs.sum()), "missing_future_fbs_rosters": 0,
        "strict_player_history_before_kickoff": True, "no_2026_outcomes_used": True,
        "all_expected_future_games_scored": True, "probability_distributions_valid": True,
        "max_probability_mass_error": max_mass, "max_pmf_readout_error": max_readout,
        "members": members, "production_ready": True,
    }
    save_json(directory / "forecast_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "score", "audit"])
    parser.add_argument("--directory", type=Path, default=OUTPUT)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, default=256)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cutoff")
    args = parser.parse_args()
    if args.action == "train":
        train(args.directory, args.seeds, args.workers, args.threads, args.micro_batch)
    elif args.action == "score":
        score(args.directory, args.output, args.cutoff)
    else:
        audit(args.directory, args.output)
