"""Frozen chronological search for the next player-model architecture.

The search uses only 2020-2022 to screen and select.  The selected architecture
is saved before the reused 2023-2025 assessment is read.  The exact production
context forecasts are reused as the control.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from cfbpred.architecture_experiment import fold_masks
from cfbpred.architecture_research_net import VARIANTS, make
from cfbpred.consistent_net import fit_consistent_inputs
from cfbpred.ensemble import readouts
from cfbpred.model import count_params, discrete_crps
from cfbpred.player_sequence_experiment import INPUTS, PLAYERS, SEQUENCES, load
from cfbpred.player_sequence_fast_inputs import FastSequenceStore
from cfbpred.player_sequence_train import fit, predict
from cfbpred.pregame_experiments import paired, save_json, sha, summarize


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "models/experiments/architecture_research_20260907_v1"
CONTROL = ROOT / "models/experiments/player_sequences_20260907_v2/runs/context"
CODE = [
    "architecture_research_net.py", "architecture_research_experiment.py",
    "player_sequence_net.py", "player_sequence_train.py", "player_sequence_inputs.py",
    "player_sequence_fast_inputs.py", "player_net.py", "player_inputs.py",
    "consistent_net.py", "raw_net.py", "model.py", "train.py",
    "architecture_experiment.py", "pregame_experiments.py", "ensemble.py",
]


def prepare(output: Path, workers: int, threads: int, micro_batch: int):
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "question": "Best architecture for raw game, team-history, and complete individual-player inputs",
        "control": "Frozen production player-sequence context architecture and cached predictions",
        "candidates": VARIANTS,
        "screen": {"forecast_years": [2022], "seeds": [0, 1],
                   "advance": "Three lowest margin-MAE candidates with winner logloss <= control + 0.010"},
        "development": {"forecast_years": [2020, 2021, 2022], "seeds": [0, 1],
                        "selection": "Lowest pooled margin MAE among control and advanced candidates, subject to winner logloss <= control + 0.005, Brier <= control + 0.002, and total MAE <= control + 0.10"},
        "assessment": {"forecast_years": [2023, 2024, 2025], "seeds": [0, 1, 2],
                       "rule": "Read and report only after the development selection is persisted; reused historical benchmark"},
        "training": {"selection": "Train before Y-1; select epochs on Y-1 winner logloss; refit through Y-1; forecast Y",
                     "max_epochs": 200, "patience": 20, "effective_batch": 2048,
                     "micro_batch": micro_batch, "fcs_extra_sampling": .25,
                     "tabm": {"members": 16, "blocks": 2, "width": 192, "dropout": .10,
                              "lr": .002, "weight_decay": .0003,
                              "loss": "Mean of member losses; probability average at evaluation"}},
        "constraints": {"market_or_external_model_inputs": False, "2026_labels_used": False,
                        "every_roster_member_retained": True, "strictly_pregame": True,
                        "team_swap_consistency": True, "training_only_normalization": True},
        "research_basis": [
            "https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html",
            "https://proceedings.iclr.cc/paper_files/paper/2025/hash/571799482291411607c54984153190b0-Abstract-Conference.html",
            "https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html",
            "https://proceedings.mlr.press/v97/lee19d.html",
            "https://papers.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html",
        ],
        "assessment_previously_examined": True,
        "production_modified": False,
        "execution": {"device": "cpu", "workers": workers, "threads_per_worker": threads},
        "inputs": {p.name: sha(p) for p in INPUTS.iterdir() if p.is_file()},
        "player_manifest_sha256": sha(PLAYERS / "manifest.json"),
        "sequence_manifest_sha256": sha(SEQUENCES / "manifest.json"),
        "control_predictions_sha256": {
            p.name: sha(p) for p in CONTROL.glob("*_predictions.parquet")},
        "code": {name: sha(Path(__file__).parent / name) for name in CODE},
    }
    path = output / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Frozen architecture research changed; use a new output directory")
    save_json(path, plan)
    snapshot = output / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in CODE:
        shutil.copy2(Path(__file__).parent / name, snapshot / name)


def control_frame(years):
    frames = []
    for year in years:
        seeds = 2 if year <= 2022 else 3
        frames.append(pd.read_parquet(CONTROL / f"{year}_{seeds}seeds_predictions.parquet"))
    return pd.concat(frames, ignore_index=True)


def fold_data(base, mask):
    data = fit_consistent_inputs(base, mask)
    data["player_store"] = base["player_store"].for_fold(np.where(mask)[0], True, False)
    return data


def run_variant(base, variant, years, seeds, output, micro_batch):
    directory = output / "runs" / variant
    directory.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(INPUTS / "game_scalars.parquet")[[
        "game_id", "season", "week_c", "margin", "total"]].merge(
            pd.read_parquet(INPUTS / "market.parquet"), on="game_id", how="left").set_index("game_id")
    frames = []
    for year in years:
        tr, va, te, refit = fold_masks(base, year, None)
        fitted, refitted = fold_data(base, tr), fold_data(base, refit)
        idx = np.where(te)[0]
        margins, totals, epochs = [], [], []
        for seed in seeds:
            cache = directory / f"{year}_s{seed}.npz"
            if cache.exists():
                saved = np.load(cache)
                if not np.array_equal(saved["game_id"], base["G_gid"][idx]):
                    raise ValueError("Cached architecture target order differs")
                pm, pt, epoch = (torch.tensor(saved["pm"]), torch.tensor(saved["pt"]),
                                 int(saved["epochs"]))
            else:
                started = time.monotonic()
                print(f"START {variant} year={year} seed={seed}", flush=True)
                save_json(output / f"status_{variant}.json",
                          {"variant": variant, "year": year, "seed": seed, "stage": "select_epochs"})
                torch.manual_seed(seed); np.random.seed(seed)
                model = make(fitted, variant)
                parameters = count_params(model.pre)
                fit(model, fitted, torch.device("cpu"), np.where(tr)[0], np.where(va)[0],
                    micro_batch_size=micro_batch,
                    log_path=directory / f"{year}_s{seed}_selection.json")
                epoch = model.pre.selected_epochs
                del model
                torch.manual_seed(seed); np.random.seed(seed)
                model = make(refitted, variant)
                save_json(output / f"status_{variant}.json",
                          {"variant": variant, "year": year, "seed": seed,
                           "stage": "refit", "epochs": epoch})
                fit(model, refitted, torch.device("cpu"), np.where(refit)[0], np.where(va)[0],
                    fixed_epochs=epoch, micro_batch_size=micro_batch,
                    log_path=directory / f"{year}_s{seed}_refit.json")
                pm, pt = predict(model, refitted, idx, torch.device("cpu"))
                checkpoint = {
                    "architecture": "architecture_research_v1", "variant": variant,
                    "pregame_only": True, "state_dict": {k: v.detach().cpu() for k, v in model.pre.state_dict().items()},
                    "mu": refitted["mu"], "sd": refitted["sd"],
                    "scalar_cols": base["scalar_cols"], "vocab": base["vocab"],
                    "known_sequence_teams": refitted["player_store"].known_teams,
                    "forecast_year": year, "seed": seed, "epochs": epoch,
                    "train_game_ids": base["G_gid"][refit],
                    "train_max_season": int(base["G_season"][refit].max()),
                    "player_schema": base["player_store"].manifest,
                    "sequence_schema": base["player_store"].sequence_manifest,
                    "parameters": parameters, "seconds": time.monotonic() - started,
                }
                torch.save(checkpoint, directory / f"{year}_s{seed}.pt")
                np.savez_compressed(cache, pm=pm.numpy(), pt=pt.numpy(),
                                    game_id=base["G_gid"][idx], epochs=epoch)
                print(f"MEMBER DONE {variant} year={year} seed={seed} epochs={epoch} "
                      f"parameters={parameters} seconds={checkpoint['seconds']:.1f}", flush=True)
                del model
            margins.append(pm); totals.append(pt); epochs.append(epoch)
        pm, pt = torch.stack(margins).mean(0), torch.stack(totals).mean(0)
        frame = pd.DataFrame(readouts(pm, pt), index=base["G_gid"][idx]).join(meta)
        frame["crps"] = discrete_crps(pm.clamp(min=1e-30).log(), base["G_margin"][idx],
                                      reduction="none").numpy()
        frame["fbs_fbs"] = base["G_fbs_fbs"][idx]
        frame.index.name = "game_id"
        frame = frame.reset_index()
        frame.to_parquet(directory / f"{year}_{len(seeds)}seeds_predictions.parquet", index=False)
        metrics = summarize(frame)
        save_json(directory / f"{year}_{len(seeds)}seeds_report.json",
                  {"epochs": epochs, "metrics": metrics})
        print(f"YEAR DONE {variant} {year}: MAE={metrics['margin_mae']:.5f} "
              f"LL={metrics['wp_logloss']:.5f}", flush=True)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def comparison(control, candidates):
    return {
        "metrics": {"control": summarize(control),
                    **{name: summarize(frame) for name, frame in candidates.items()}},
        "paired_vs_control": {name: paired(control, frame) for name, frame in candidates.items()},
        "by_year": {"control": {str(y): summarize(control[control.season.eq(y)])
                                  for y in sorted(control.season.unique())},
                    **{name: {str(y): summarize(frame[frame.season.eq(y)])
                              for y in sorted(frame.season.unique())}
                       for name, frame in candidates.items()}},
    }


def initialize_worker(threads):
    global BASE
    torch.set_num_threads(threads)
    BASE = load()
    BASE["player_store"].__class__ = FastSequenceStore


def worker(variant, years, seeds, output, micro_batch):
    return run_variant(BASE, variant, years, seeds, output, micro_batch)


def parallel_stage(pool, names, years, seeds, output, micro_batch, label):
    results = {}
    jobs = {pool.submit(worker, name, years, seeds, output, micro_batch): name for name in names}
    for job in as_completed(jobs):
        name = jobs[job]
        results[name] = job.result()
        save_json(output / f"{label}_progress.json",
                  {n: summarize(f) for n, f in results.items()})
    return results


def run(output=OUTPUT, workers=3, threads=2, micro_batch=256):
    prepare(output, workers, threads, micro_batch)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                             initializer=initialize_worker, initargs=(threads,)) as pool:
        save_json(output / "status.json", {"stage": "screen", "members": len(VARIANTS) * 2})
        screen_control = control_frame([2022])
        screen = parallel_stage(pool, list(VARIANTS), [2022], [0, 1], output, micro_batch, "screen")
        screen_report = comparison(screen_control, screen)
        eligible = [name for name, frame in screen.items()
                    if summarize(frame)["wp_logloss"] <= summarize(screen_control)["wp_logloss"] + .010]
        advanced = sorted(eligible, key=lambda name: summarize(screen[name])["margin_mae"])[:3]
        save_json(output / "advancement.json",
                  {"advanced": advanced, "screen": screen_report, "saved_before_full_development": True})

        save_json(output / "status.json", {"stage": "development", "advanced": advanced})
        development = parallel_stage(pool, advanced, [2020, 2021, 2022], [0, 1],
                                     output, micro_batch, "development")
        dev_control = control_frame([2020, 2021, 2022])
        dev_report = comparison(dev_control, development)
        cm = summarize(dev_control)
        admissible = [name for name, frame in development.items()
                      if summarize(frame)["wp_logloss"] <= cm["wp_logloss"] + .005
                      and summarize(frame)["brier"] <= cm["brier"] + .002
                      and summarize(frame)["total_mae"] <= cm["total_mae"] + .10
                      and summarize(frame)["margin_mae"] < cm["margin_mae"]]
        selected = (min(admissible, key=lambda name: summarize(development[name])["margin_mae"])
                    if admissible else "control")
        selection = {"selected": selected, "admissible": admissible,
                     "development": dev_report, "saved_before_assessment": True}
        save_json(output / "selection.json", selection)
        report = {"screen": screen_report, **selection, "production_modified": False,
                  "2026_labels_used": False, "assessment_is_reused_historical_benchmark": True}
        save_json(output / "report.json", report)

        assessment_control = control_frame([2023, 2024, 2025])
        if selected == "control":
            assessment = {}
        else:
            save_json(output / "status.json", {"stage": "assessment", "selected": selected})
            assessment = parallel_stage(pool, [selected], [2023, 2024, 2025], [0, 1, 2],
                                        output, micro_batch, "assessment")
        report["assessment"] = comparison(assessment_control, assessment)
        if selected != "control":
            report["assessment"]["early_weeks_through_3"] = {
                "control": summarize(assessment_control[assessment_control.week_c.le(3)]),
                selected: summarize(assessment[selected][assessment[selected].week_c.le(3)]),
            }
        save_json(output / "report.json", report)
    save_json(output / "status.json", {"stage": "complete", "selected": selected,
                                       "production_modified": False})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, default=256)
    args = parser.parse_args()
    run(args.output, args.workers, args.threads, args.micro_batch)
