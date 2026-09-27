"""Frozen comparison of order-consistent networks and recent training windows.

Choose on rolling 2020-2022 forecasts before reading 2023-2025 assessments.
The latter are reused historical benchmarks. 2026 outcomes never fit or select
models. The archived margin-form experiment supplies identical inputs/control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.backtest_pregame import fit_inputs, partitions, predict
from cfbpred.config import MODELS
from cfbpred.consistent_net import ConsistentPregameTower, fit_consistent_inputs
from cfbpred.ensemble import readouts
from cfbpred.model import discrete_crps
from cfbpred.pregame_experiments import (ASSESSMENT, DEVELOPMENT, SEEDS, choose,
                                        paired, save_json, sha, summarize)
from cfbpred.train import build_tensors, make_model, pretrain, s2d_indices

REFERENCE = MODELS / "experiments" / "margin_form_20260906"
OUTPUT = MODELS / "experiments" / "architecture_20260906"
VARIANTS = {
    "baseline": {"consistent": False, "window": None},
    "consistent": {"consistent": True, "window": None},
    "recent8": {"consistent": False, "window": 8},
    "consistent_recent8": {"consistent": True, "window": 8},
}
CODE = ("architecture_experiment.py", "consistent_net.py", "train.py",
        "raw_net.py", "model.py", "backtest_pregame.py", "pregame_experiments.py")


def prepare(output):
    source = REFERENCE / "inputs"
    old_plan = json.loads((REFERENCE / "plan.json").read_text())
    # A cached control is valid only if its actual implementation is unchanged.
    for name, digest in old_plan["code_sha256"].items():
        if sha(Path(__file__).parent / name) != digest:
            raise ValueError(f"Archived control code changed: {name}")
    for name, digest in old_plan["input_sha256"].items():
        if sha(source / name) != digest:
            raise ValueError(f"Archived input changed: {name}")
    control_dir = REFERENCE / "runs" / "baseline__none"
    control_hash = {p.name: sha(p) for p in sorted(control_dir.glob("*_predictions.parquet"))}
    plan = {
        "variants": VARIANTS, "development": DEVELOPMENT, "assessment": ASSESSMENT,
        "seeds": SEEDS, "2026_labels_used": False,
        "selection": "Lowest development margin MAE with gain >=0.03, WP LL <=control+0.005, Brier <=control+0.002; freeze before assessment",
        "promotion": "Assessment paired MAE CI wholly below zero, MAE gain >=0.10, probability guardrails, and total MAE <=control+0.10; otherwise retain production",
        "window_semantics": "Last N seasons before validation, then last N before forecast for refit; all-prior histories retained",
        "source": str(source), "input_sha256": old_plan["input_sha256"],
        "control_sha256": control_hash,
        "code_sha256": {name: sha(Path(__file__).parent / name) for name in CODE},
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Frozen architecture experiment changed; use a new output directory")
    save_json(path, plan)
    snapshot = output / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in CODE:
        shutil.copy2(Path(__file__).parent / name, snapshot / name)
    return source


def fold_masks(base, year, window):
    tr, va, te = partitions(base, year, "chronological")
    refit = base["G_labeled"] & (base["G_season"] < year)
    if window is not None:
        tr &= base["G_season"] >= year - 1 - window
        refit &= base["G_season"] >= year - window
    assert base["G_season"][refit].max() < year
    return tr, va, te, refit


def challenger(d, consistent):
    model = make_model(d, s2d_gate=True, delta=True)
    if consistent:
        model.pre = ConsistentPregameTower(
            len(d["vocab"]["team"]) + 1, len(d["vocab"]["conf"]) + 1,
            d["scalar_cols"], use_s2d_gate=True,
            s2d_idx_h=s2d_indices(d["scalar_cols"], "h"),
            s2d_idx_a=s2d_indices(d["scalar_cols"], "a"))
    return model


def run_variant(base, source, output, variant, years, device):
    if variant == "baseline":
        return pd.concat([pd.read_parquet(REFERENCE / "runs" / "baseline__none" /
                                         f"{year}_predictions.parquet") for year in years], ignore_index=True)
    settings = VARIANTS[variant]
    directory = output / "runs" / variant
    directory.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(source / "game_scalars.parquet")[["game_id", "season", "week_c", "margin", "total", "neutral_site"]]
    meta = meta.merge(pd.read_parquet(source / "market.parquet"), on="game_id", how="left").set_index("game_id")
    outputs = []
    for year in years:
        tr, va, te, refit = fold_masks(base, year, settings["window"])
        fit = fit_consistent_inputs if settings["consistent"] else fit_inputs
        d_fit, d_refit = fit(base, tr), fit(base, refit)
        idx = np.where(te)[0]
        margins, totals, epochs = [], [], []
        for seed in SEEDS:
            path = directory / f"{year}_s{seed}.npz"
            if path.exists():
                with np.load(path) as saved:
                    if not np.array_equal(saved["game_id"], base["G_gid"][idx]):
                        raise ValueError("Cached prediction order mismatch")
                    pm, pt, epoch = torch.tensor(saved["pm"]), torch.tensor(saved["pt"]), int(saved["epochs"])
            else:
                print(f"{variant} forecast={year} seed={seed} train={tr.sum()} refit={refit.sum()}", flush=True)
                torch.manual_seed(seed); np.random.seed(seed)
                model = challenger(d_fit, settings["consistent"]).to(device)
                pretrain(model, d_fit, device, train_indices=np.where(tr)[0], val_indices=np.where(va)[0])
                epoch = model.pre.selected_epochs
                torch.manual_seed(seed); np.random.seed(seed)
                model = challenger(d_refit, settings["consistent"]).to(device)
                pretrain(model, d_refit, device, train_indices=np.where(refit)[0],
                         val_indices=np.where(va)[0], fixed_epochs=epoch)
                pm, pt = predict(model, d_refit, idx, device)
                torch.save({"state_dict": {k:v.cpu() for k,v in model.pre.state_dict().items()},
                            "mu": d_refit["mu"], "sd": d_refit["sd"], "scalar_cols": base["scalar_cols"],
                            "variant": variant, "forecast_year": year, "seed": seed, "epochs": epoch,
                            "train_game_ids": base["G_gid"][refit]}, directory / f"{year}_s{seed}.pt")
                np.savez_compressed(path, pm=pm.numpy(), pt=pt.numpy(), epochs=epoch, game_id=base["G_gid"][idx])
                del model
                if device.type == "mps":
                    torch.mps.empty_cache()
            margins.append(pm); totals.append(pt); epochs.append(epoch)
        pm, pt = torch.stack(margins).mean(0), torch.stack(totals).mean(0)
        frame = pd.DataFrame(readouts(pm, pt), index=base["G_gid"][idx]).join(meta)
        frame["crps"] = discrete_crps(pm.clamp(min=1e-30).log(), base["G_margin"][idx], reduction="none").numpy()
        frame["fbs_fbs"] = base["G_fbs_fbs"][idx]
        frame["variant"] = variant
        frame.index.name = "game_id"
        frame = frame.reset_index()
        frame.to_parquet(directory / f"{year}_predictions.parquet", index=False)
        result = summarize(frame)
        save_json(directory / f"{year}_report.json", {"epochs": epochs, "metrics": result})
        print(f"Completed {variant} {year}: MAE={result['margin_mae']:.4f} LL={result['wp_logloss']:.5f}", flush=True)
        outputs.append(frame)
    return pd.concat(outputs, ignore_index=True)


def run(output):
    source = prepare(output)
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base = build_tensors(data_dir=source, include_plays=False)
    dev = {name: run_variant(base, source, output, name, DEVELOPMENT, device) for name in VARIANTS}
    scores = {name: summarize(frame) for name, frame in dev.items()}
    selected = choose(scores, "baseline")
    choice = {"development": scores, "selected": selected,
              "development_pairing": paired(dev["baseline"], dev[selected])}
    choice_path = output / "choice.json"
    if choice_path.exists() and json.loads(choice_path.read_text()) != choice:
        raise ValueError("Development choice changed after it was frozen")
    save_json(choice_path, choice)
    print("Development choice: " + json.dumps(choice, indent=2), flush=True)
    assessment = {name: run_variant(base, source, output, name, ASSESSMENT, device)
                  for name in dict.fromkeys(["baseline", selected])}
    scored = {name: summarize(frame) for name, frame in assessment.items()}
    comparison = paired(assessment["baseline"], assessment[selected])
    c, n = scored["baseline"], scored[selected]
    qualifies = (selected != "baseline" and comparison["week_cluster_bootstrap_95ci"][1] < 0
                 and n["margin_mae"] <= c["margin_mae"] - .10
                 and n["wp_logloss"] <= c["wp_logloss"] + .005 and n["brier"] <= c["brier"] + .002
                 and n["total_mae"] <= c["total_mae"] + .10)
    result = {**choice, "assessment": scored, "assessment_pairing": comparison,
              "assessment_by_year": {name: {str(y): summarize(f[f.season==y]) for y in ASSESSMENT}
                                     for name, f in assessment.items()},
              "qualifies_for_confirmation": bool(qualifies), "production_modified": False}
    save_json(output / "report.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output)
