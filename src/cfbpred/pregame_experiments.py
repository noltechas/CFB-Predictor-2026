"""Sequential margin-objective and recent-form experiments with frozen inputs.

Settings are selected on rolling 2020-2022 development forecasts. The selected
settings are then evaluated on 2023-2025, which are historical benchmarks, not
new untouched tests. No 2026 outcome is used to fit or choose anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.backtest_pregame import fit_inputs, metrics, partitions, predict
from cfbpred.config import FEAT, MODELS, PBP
from cfbpred.ensemble import readouts
from cfbpred.model import discrete_crps
from cfbpred.train import BANNED_SUBSTRINGS, build_tensors, make_model, pretrain

DEVELOPMENT = [2020, 2021, 2022]
ASSESSMENT = [2023, 2024, 2025]
SEEDS = [0, 1, 2]
OBJECTIVES = {
    "baseline": {"sel_mae_w": 0.0, "margin_crps_w": 0.0},
    "margin_selection": {"sel_mae_w": 0.05, "margin_crps_w": 0.0},
    "crps_010": {"sel_mae_w": 0.05, "margin_crps_w": 0.10},
    "crps_030": {"sel_mae_w": 0.05, "margin_crps_w": 0.30},
}
FORM_VARIANTS = ["none", "recent3", "recent5", "recent3_matchup", "recent5_matchup"]
DEFAULT_OUTPUT = MODELS / "experiments" / "margin_form_20260906"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2))


def frozen_inputs(output):
    output.mkdir(parents=True, exist_ok=True)
    source = output / "inputs"
    source.mkdir(exist_ok=True)
    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        for name in ("game_scalars.parquet", "game_history.parquet", "team_vocab.json"):
            shutil.copy2(FEAT / name, source / name)
        with duckdb.connect() as con:
            # Identical archived benchmark convention to backtest_pregame.py.
            lines = con.execute(f"SELECT game_id, first(spread_close) AS spread_close FROM read_parquet('{PBP / 'pbp_*.parquet'}') GROUP BY game_id").fetchdf()
        lines.to_parquet(source / "market.parquet", index=False)
        save_json(manifest_path, {p.name: sha(p) for p in source.iterdir() if p.is_file()})
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest.items():
        if sha(source / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    plan = {
        "development_years": DEVELOPMENT, "assessment_years": ASSESSMENT,
        "seeds": SEEDS, "objectives": OBJECTIVES, "form_variants": FORM_VARIANTS,
        "selection": "Lowest pooled development FBS-FBS margin MAE, at least 0.03 points better than control, subject to WP logloss <= control+0.005 and Brier <= control+0.002",
        "assessment": "Frozen choice; paired week-cluster bootstrap; no choice based on 2023-2025 scores",
        "2026_labels_used": False,
        "code_sha256": {name: sha(Path(__file__).parent / name) for name in
                        ("pregame_experiments.py", "train.py", "model.py", "raw_net.py", "backtest_pregame.py")},
        "input_sha256": manifest,
    }
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Experiment plan/code changed; use a new output directory")
    save_json(plan_path, plan)
    return source


def add_form(base, source, variant):
    if variant == "none":
        return base
    features = pd.read_parquet(source / "recent_form.parquet").set_index("game_id")
    if features.index.duplicated().any() or not set(base["G_gid"]).issubset(features.index):
        raise ValueError("Every target needs exactly one recent-form row")
    window = 3 if variant.startswith("recent3") else 5
    cols = sorted(c for c in features if f"_r{window}_" in c and
                  ("_match_" not in c or variant.endswith("_matchup")))
    if not cols or any(any(b in c.lower() for b in BANNED_SUBSTRINGS) for c in cols):
        raise ValueError("Invalid recent-form feature contract")
    out = base.copy()
    out["G_raw_scalars"] = np.concatenate([base["G_raw_scalars"], features.loc[base["G_gid"], cols].to_numpy(np.float32)], axis=1)
    out["scalar_cols"] = base["scalar_cols"] + cols
    return out


def summarize(frame):
    f = frame[frame.fbs_fbs]
    result = metrics(f)
    result["crps"] = float(f.crps.mean())
    result["ece_10"] = float(sum(abs((g.margin > 0).mean() - g.wp.mean()) * len(g) / len(f)
        for _, g in f.groupby(pd.cut(f.wp, np.linspace(0, 1, 11), include_lowest=True), observed=True)))
    return result


def paired(control, candidate):
    c = control[control.fbs_fbs].set_index("game_id")
    n = candidate[candidate.fbs_fbs].set_index("game_id").loc[c.index]
    diff = abs(n.em-n.margin) - abs(c.em-c.margin)
    clusters = [g.to_numpy() for _, g in diff.groupby([c.season, c.week_c])]
    rng = np.random.default_rng(0)
    boot = [np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))]).mean() for _ in range(2000)]
    return {"n": len(diff), "margin_mae_change": float(diff.mean()),
            "week_cluster_bootstrap_95ci": np.percentile(boot, [2.5,97.5]).tolist()}


def choose(results, control):
    ref = results[control]
    eligible = [name for name, r in results.items() if name != control and
                r["margin_mae"] <= ref["margin_mae"] - 0.03 and
                r["wp_logloss"] <= ref["wp_logloss"] + 0.005 and
                r["brier"] <= ref["brier"] + 0.002]
    return min(eligible, key=lambda name: results[name]["margin_mae"]) if eligible else control


def run_variant(base, source, output, objective, form, years, device):
    variant = f"{objective}__{form}"
    d0 = add_form(base, source, form)
    run_dir = output / "runs" / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = {"objective": OBJECTIVES[objective], "form": form,
                "form_sha256": sha(source / "recent_form.parquet") if form != "none" else None}
    settings_path = run_dir / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Run settings or form features changed")
    save_json(settings_path, settings)
    meta = pd.read_parquet(source / "game_scalars.parquet")[["game_id", "season", "week_c", "margin", "total"]]
    meta = meta.merge(pd.read_parquet(source / "market.parquet"), on="game_id", how="left").set_index("game_id")
    outputs = []
    for year in years:
        tr, va, te = partitions(d0, year, "chronological")
        idx = np.where(te)[0]
        fitted = fit_inputs(d0, tr)
        refit = fit_inputs(d0, d0["G_labeled"] & (d0["G_season"] < year))
        pm_all, pt_all, epochs = [], [], []
        for seed in SEEDS:
            cache = run_dir / f"{year}_s{seed}.npz"
            if cache.exists():
                with np.load(cache) as saved:
                    pm, pt = torch.tensor(saved["pm"]), torch.tensor(saved["pt"])
                    selected = int(saved["epochs"])
            else:
                print(f"{variant} forecast={year} seed={seed} columns={len(d0['scalar_cols'])}", flush=True)
                torch.manual_seed(seed); np.random.seed(seed)
                model = make_model(fitted, s2d_gate=True, delta=True).to(device)
                pretrain(model, fitted, device, train_indices=np.where(tr)[0],
                         val_indices=np.where(va)[0], **OBJECTIVES[objective])
                selected = model.pre.selected_epochs
                torch.manual_seed(seed); np.random.seed(seed)
                model = make_model(refit, s2d_gate=True, delta=True).to(device)
                pretrain(model, refit, device, train_indices=np.where(d0["G_labeled"] & (d0["G_season"] < year))[0],
                         val_indices=np.where(va)[0], fixed_epochs=selected, **OBJECTIVES[objective])
                pm, pt = predict(model, refit, idx, device)
                np.savez_compressed(cache, pm=pm.numpy(), pt=pt.numpy(), epochs=selected)
                del model
                if device.type == "mps":
                    torch.mps.empty_cache()
            pm_all.append(pm); pt_all.append(pt); epochs.append(selected)
        pm = torch.stack(pm_all).mean(0)
        pt = torch.stack(pt_all).mean(0)
        frame = pd.DataFrame(readouts(pm, pt), index=d0["G_gid"][idx]).join(meta)
        frame["crps"] = discrete_crps(pm.clamp(min=1e-30).log(), d0["G_margin"][idx], reduction="none").numpy()
        frame["fbs_fbs"] = d0["G_fbs_fbs"][idx]
        frame["variant"] = variant
        frame.index.name = "game_id"
        frame = frame.reset_index()
        frame.to_parquet(run_dir / f"{year}_predictions.parquet", index=False)
        result = summarize(frame)
        save_json(run_dir / f"{year}_report.json", {"epochs": epochs, "metrics": result})
        print(f"Completed {variant} {year}: MAE={result['margin_mae']:.4f} LL={result['wp_logloss']:.5f}", flush=True)
        outputs.append(frame)
    return pd.concat(outputs, ignore_index=True)


def stage(output, kind):
    source = frozen_inputs(output)
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base = build_tensors(data_dir=source, include_plays=False)
    if kind == "objective":
        options = {name: (name, "none") for name in OBJECTIVES}
        control = "baseline"
    else:
        previous = json.loads((output / "objective_report.json").read_text())
        objective = previous["selected_on_development"]
        options = {form: (objective, form) for form in FORM_VARIANTS}
        control = "none"
    dev_frames = {name: run_variant(base, source, output, *args, DEVELOPMENT, device)
                  for name, args in options.items()}
    dev_scores = {name: summarize(frame) for name, frame in dev_frames.items()}
    selected = choose(dev_scores, control)
    # Record the choice before assessing later years.
    result = {"stage": kind, "development": dev_scores, "selected_on_development": selected,
              "control": control, "choice_made_before_assessment": True,
              "selection_pairing": paired(dev_frames[control], dev_frames[selected])}
    save_json(output / f"{kind}_report.json", result)
    assess = {name: run_variant(base, source, output, *options[name], ASSESSMENT, device)
              for name in dict.fromkeys([control, selected])}
    result["assessment"] = {name: summarize(frame) for name, frame in assess.items()}
    result["assessment_by_year"] = {name: {str(year): summarize(frame[frame.season==year]) for year in ASSESSMENT}
                                      for name, frame in assess.items()}
    result["assessment_pairing"] = paired(assess[control], assess[selected])
    save_json(output / f"{kind}_report.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["objective", "form"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    stage(args.output, args.stage)
