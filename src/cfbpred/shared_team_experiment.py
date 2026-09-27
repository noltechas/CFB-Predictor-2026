"""Evaluate a shared team encoder; retain reusable per-seed checkpoints."""
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
from cfbpred.architecture_experiment import REFERENCE, fold_masks
from cfbpred.backtest_pregame import fit_inputs, predict
from cfbpred.config import MODELS
from cfbpred.consistent_net import fit_consistent_inputs
from cfbpred.ensemble import readouts
from cfbpred.model import discrete_crps
from cfbpred.pregame_experiments import (ASSESSMENT, DEVELOPMENT, choose, paired, save_json, sha, summarize)
from cfbpred.shared_team_net import SharedTeamPregameTower
from cfbpred.train import build_tensors, make_model, pretrain, s2d_indices

OUTPUT = MODELS / "experiments" / "shared_team_20260906"
CODE = ("shared_team_experiment.py", "shared_team_net.py", "architecture_experiment.py",
        "consistent_net.py", "train.py", "raw_net.py", "model.py", "backtest_pregame.py", "pregame_experiments.py")


def make_shared(d):
    model = make_model(d, s2d_gate=True, delta=True)
    model.pre = SharedTeamPregameTower(
        len(d["vocab"]["team"]) + 1, len(d["vocab"]["conf"]) + 1, d["scalar_cols"],
        use_s2d_gate=True, s2d_idx_h=s2d_indices(d["scalar_cols"], "h"),
        s2d_idx_a=s2d_indices(d["scalar_cols"], "a"))
    return model


def run_variant(base, source, output, name, years, seeds, device, *, model_factory=None):
    directory = output / "runs" / name
    directory.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(source / "game_scalars.parquet")
    meta = rows[["game_id", "season", "week_c", "margin", "total", "neutral_site"]].merge(
        pd.read_parquet(source / "market.parquet"), on="game_id", how="left").set_index("game_id")
    outputs = []
    for year in years:
        tr, va, te, refit = fold_masks(base, year, None)
        fit = fit_consistent_inputs if name == "shared" else fit_inputs
        d_fit, d_refit = fit(base, tr), fit(base, refit)
        idx = np.where(te)[0]
        margins, totals, epochs = [], [], []
        for seed in seeds:
            cache = directory / f"{year}_s{seed}.npz"
            original_cache = REFERENCE / "runs" / "baseline__none" / cache.name
            if name == "baseline" and seed < 3:
                with np.load(original_cache) as saved:
                    pm, pt, epoch = torch.tensor(saved["pm"]), torch.tensor(saved["pt"]), int(saved["epochs"])
            elif cache.exists():
                with np.load(cache) as saved:
                    if not np.array_equal(saved["game_id"], base["G_gid"][idx]):
                        raise ValueError("Cached game order changed")
                    pm, pt, epoch = torch.tensor(saved["pm"]), torch.tensor(saved["pt"]), int(saved["epochs"])
            else:
                def make(d):
                    if model_factory is not None:
                        return model_factory(d)
                    return make_shared(d) if name == "shared" else make_model(d, s2d_gate=True, delta=True)
                print(f"{name} forecast={year} seed={seed}", flush=True)
                torch.manual_seed(seed); np.random.seed(seed)
                model = make(d_fit).to(device)
                pretrain(model, d_fit, device, train_indices=np.where(tr)[0], val_indices=np.where(va)[0])
                epoch = model.pre.selected_epochs
                torch.manual_seed(seed); np.random.seed(seed)
                model = make(d_refit).to(device)
                pretrain(model, d_refit, device, train_indices=np.where(refit)[0],
                         val_indices=np.where(va)[0], fixed_epochs=epoch)
                pm, pt = predict(model, d_refit, idx, device)
                torch.save({"state_dict": {k:v.cpu() for k,v in model.pre.state_dict().items()},
                            "mu":d_refit["mu"], "sd":d_refit["sd"], "scalar_cols":base["scalar_cols"],
                            "variant":name,"forecast_year":year,"seed":seed,"epochs":epoch,
                            "train_game_ids":base["G_gid"][refit]}, directory / f"{year}_s{seed}.pt")
                np.savez_compressed(cache, pm=pm.numpy(), pt=pt.numpy(), epochs=epoch, game_id=base["G_gid"][idx])
                del model
                if device.type == "mps":
                    torch.mps.empty_cache()
            margins.append(pm); totals.append(pt); epochs.append(epoch)
        pm, pt = torch.stack(margins).mean(0), torch.stack(totals).mean(0)
        frame = pd.DataFrame(readouts(pm, pt), index=base["G_gid"][idx]).join(meta)
        frame["crps"] = discrete_crps(pm.clamp(min=1e-30).log(), base["G_margin"][idx], reduction="none").numpy()
        frame["fbs_fbs"] = base["G_fbs_fbs"][idx]
        frame["variant"] = name
        frame.index.name = "game_id"
        frame = frame.reset_index()
        frame.to_parquet(directory / f"{year}_{len(seeds)}seeds_predictions.parquet", index=False)
        result = summarize(frame)
        save_json(directory / f"{year}_{len(seeds)}seeds_report.json", {"epochs":epochs,"metrics":result})
        print(f"Completed {name} {year} seeds={len(seeds)}: MAE={result['margin_mae']:.4f} LL={result['wp_logloss']:.5f}", flush=True)
        outputs.append(frame)
    return pd.concat(outputs, ignore_index=True)


def prepare(output):
    old = json.loads((REFERENCE / "plan.json").read_text())
    source = REFERENCE / "inputs"
    for name, digest in old["code_sha256"].items():
        if sha(Path(__file__).parent / name) != digest:
            raise ValueError(f"Control code changed: {name}")
    for name, digest in old["input_sha256"].items():
        if sha(source / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    original_caches = REFERENCE / "runs" / "baseline__none"
    plan = {
        "candidate": "Shared 128->64 team encoder, order-consistent matchup decoder, all raw inputs; zero empty history",
        "motivation": "Follow-up after consistency-only later-year gain was too small; no shared-encoder later-year scores yet inspected",
        "development": DEVELOPMENT, "assessment": ASSESSMENT, "seeds": [0,1,2],
        "selection": "Development MAE gain >=0.03 with WP LL <=control+0.005 and Brier <=control+0.002",
        "confirmation": "If selected and three-seed assessment gain >=0.10 with WP LL <=control+0.005, Brier <=control+0.002, total MAE <=control+0.10: extend BOTH models to 16 seeds",
        "promotion": "16-seed MAE gain >=0.10, paired week bootstrap 95CI wholly below zero, same probability and total guardrails; no automatic publication",
        "2026_labels_used":False, "input_sha256":old["input_sha256"],
        "control_sha256":{p.name:sha(p) for p in sorted(original_caches.glob("*.npz"))},
        "code_sha256":{n:sha(Path(__file__).parent/n) for n in CODE}}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Experiment changed; use a new output directory")
    save_json(path, plan)
    snapshot = output / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in CODE:
        shutil.copy2(Path(__file__).parent / name, snapshot / name)
    return source


def comparison(frames):
    scores = {name:summarize(f) for name,f in frames.items()}
    pair = paired(frames["baseline"], frames["shared"])
    c,n = scores["baseline"],scores["shared"]
    strong = (n["margin_mae"] <= c["margin_mae"]-.10 and n["wp_logloss"] <= c["wp_logloss"]+.005
              and n["brier"] <= c["brier"]+.002 and n["total_mae"] <= c["total_mae"]+.10)
    return {"scores":scores,"pairing":pair,"passes_point_estimate_gate":bool(strong),
            "passes_promotion_gate":bool(strong and pair["week_cluster_bootstrap_95ci"][1]<0),
            "by_year":{n:{str(y):summarize(f[f.season==y]) for y in sorted(f.season.unique())} for n,f in frames.items()}}


def run(output):
    source = prepare(output)
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base = build_tensors(data_dir=source, include_plays=False)
    dev = {n:run_variant(base,source,output,n,DEVELOPMENT,range(3),device) for n in ["baseline","shared"]}
    scores = {n:summarize(f) for n,f in dev.items()}
    selected = choose(scores,"baseline")
    choice = {"development":comparison(dev),"selected":selected}
    path = output / "choice.json"
    if path.exists() and json.loads(path.read_text()) != choice:
        raise ValueError("Frozen development choice changed")
    save_json(path,choice)
    print("Development choice: " + json.dumps(choice,indent=2),flush=True)
    report = {**choice,"production_modified":False}
    save_json(output / "report.json",report)
    if selected == "shared":
        assess = {n:run_variant(base,source,output,n,ASSESSMENT,range(3),device) for n in ["baseline","shared"]}
        report["assessment_3seed"] = comparison(assess)
        save_json(output / "report.json",report)
        if report["assessment_3seed"]["passes_point_estimate_gate"]:
            print("Extending both frozen configurations to 16 seeds",flush=True)
            assess = {n:run_variant(base,source,output,n,ASSESSMENT,range(16),device) for n in ["baseline","shared"]}
            report["confirmation_16seed"] = comparison(assess)
            save_json(output / "report.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args = parser.parse_args()
    run(args.output)
