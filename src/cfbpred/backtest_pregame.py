"""Compare hashed-game selection with chronological selection on future seasons.

Both methods forecast the SAME unseen season and have the same information
cutoff. Chronological selection reserves the previous season, selects an
epoch count, then refits through that season. Nothing is fitted on test data.
Only pregame towers are needed to test the Vegas/spread question.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.config import FEAT, MODELS, PBP, VAL_MOD, VAL_REM
from cfbpred.ensemble import readouts
from cfbpred.train import build_tensors, game_batch, make_model, pretrain


def partitions(d, year, strategy):
    labeled = d["G_labeled"]
    s, gid = d["G_season"], d["G_gid"]
    if strategy == "chronological":
        tr = labeled & (s < year - 1)
        va = labeled & (s == year - 1) & d["G_fbs_inv"]
    else:
        tr = labeled & (s < year) & (gid % VAL_MOD != VAL_REM)
        va = labeled & (s < year) & (s >= 2014) & (gid % VAL_MOD == VAL_REM) & d["G_fbs_inv"]
    te = labeled & (s == year) & d["G_fbs_inv"]
    assert not (tr & va).any() and not ((tr | va) & te).any()
    assert s[tr | va].max() < s[te].min()
    if strategy == "chronological":
        assert s[tr].max() < s[va].min()
    return tr, va, te


def fit_inputs(base, mask):
    d = base.copy()
    raw = base["G_raw_scalars"]
    mu, sd = np.nanmean(raw[mask], axis=0), np.nanstd(raw[mask], axis=0)
    mu = np.nan_to_num(mu)
    sd = np.nan_to_num(sd, nan=1.0)
    sd[sd < 1e-6] = 1
    d["mu"], d["sd"] = mu, sd
    d["G_scalars"] = torch.tensor(np.nan_to_num((raw - mu) / sd, nan=0, posinf=0, neginf=0))
    # Future-only teams/conferences never receive random untrained identities.
    for fields in (("G_hidx", "G_aidx", "G_opp"), ("G_hconf", "G_aconf")):
        known = np.unique(np.concatenate([base[f].numpy()[mask].ravel() for f in fields]))
        for f in fields:
            values = base[f].numpy()
            d[f] = torch.tensor(np.where(np.isin(values, known), values, 0), dtype=torch.long)
    return d


def predict(model, d, idx, device):
    model.eval()
    margins, totals = [], []
    batch_size = 128 if "player_store" in d else 4096
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            j = torch.tensor(idx[start:start+batch_size])
            _, pm, pt = model.pregame(game_batch(d, j, device))
            margins.append(pm.softmax(-1).cpu())
            totals.append(pt.softmax(-1).cpu())
    return torch.cat(margins), torch.cat(totals)


def metrics(frame):
    y = (frame.margin > 0).to_numpy()
    p = frame.wp.clip(1e-6, 1-1e-6).to_numpy()
    out = {"n": len(frame), "margin_mae": float(abs(frame.em-frame.margin).mean()),
           "margin_rmse": float(np.sqrt(np.mean((frame.em-frame.margin)**2))),
           "median_margin_mae": float(abs(frame.mm-frame.margin).mean()),
           "total_mae": float(abs(frame.et-frame.total).mean()),
           "total_rmse": float(np.sqrt(np.mean((frame.et-frame.total)**2))),
           "wp_logloss": float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
           "brier": float(np.mean((p-y)**2))}
    market = frame.dropna(subset=["spread_close"])
    out["market_matched"] = {"n": len(market),
        "model_mae": float(abs(market.em-market.margin).mean()),
        "market_mae": float(abs(-market.spread_close-market.margin).mean()),
        "model_rmse": float(np.sqrt(np.mean((market.em-market.margin)**2))),
        "market_rmse": float(np.sqrt(np.mean((-market.spread_close-market.margin)**2)))}
    return out


def run(years, seeds, output, max_epochs=200):
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(4)
    base = build_tensors(data_dir=FEAT, include_plays=False)
    meta = pd.read_parquet(FEAT / "game_scalars.parquet")[["game_id", "season", "week_c", "margin", "total"]]
    with duckdb.connect() as con:
        lines = con.execute(f"SELECT game_id, first(spread_close) AS spread_close FROM read_parquet('{PBP / 'pbp_*.parquet'}') GROUP BY game_id").fetchdf()
    meta = meta.merge(lines, on="game_id", how="left").set_index("game_id")
    report = {"years": years, "seeds": seeds, "device": str(device),
              "feature_sha256": hashlib.sha256((FEAT / "game_scalars.parquet").read_bytes()).hexdigest(),
              "protocol": "Future-season test; chronological epoch selection followed by all-prior-season refit; pregame only; no market inputs",
              "folds": {}}
    previous = output / "report.json"
    if previous.exists() and json.loads(previous.read_text())["feature_sha256"] != report["feature_sha256"]:
        raise ValueError("Feature data changed; use a new experiment output directory")
    outputs = []
    for year in years:
        for strategy in ("hash", "chronological"):
            tr, va, te = partitions(base, year, strategy)
            fit = fit_inputs(base, tr)
            idx = np.where(te)[0]
            ms, ts, epochs = [], [], []
            for seed in seeds:
                dest = output / f"{strategy}_{year}_s{seed}.npz"
                if dest.exists():
                    saved = np.load(dest)
                    ms.append(torch.tensor(saved["pm"])); ts.append(torch.tensor(saved["pt"]))
                    epochs.append(int(saved["epochs"]))
                    continue
                torch.manual_seed(seed); np.random.seed(seed)
                model = make_model(fit, s2d_gate=True, delta=True).to(device)
                print(f"\n{strategy} test={year} seed={seed} train={tr.sum()} validation={va.sum()} test={te.sum()}", flush=True)
                pretrain(model, fit, device, max_epochs=max_epochs,
                         train_indices=np.where(tr)[0], val_indices=np.where(va)[0])
                selected = model.pre.selected_epochs
                d = fit
                if strategy == "chronological":
                    refit = base["G_labeled"] & (base["G_season"] < year)
                    d = fit_inputs(base, refit)
                    torch.manual_seed(seed); np.random.seed(seed)
                    model = make_model(d, s2d_gate=True, delta=True).to(device)
                    pretrain(model, d, device, train_indices=np.where(refit)[0],
                             val_indices=np.where(va)[0], fixed_epochs=selected)
                pm, pt = predict(model, d, idx, device)
                np.savez_compressed(dest, pm=pm.numpy(), pt=pt.numpy(), epochs=selected)
                ms.append(pm); ts.append(pt); epochs.append(selected)
                print(f"Finished {strategy} {year} seed={seed}, selected epochs={selected}", flush=True)
                del model
                if device.type == "mps": torch.mps.empty_cache()
            r = readouts(torch.stack(ms).mean(0), torch.stack(ts).mean(0))
            frame = pd.DataFrame(r, index=base["G_gid"][idx]).join(meta)
            frame.index.name = "game_id"
            frame["fbs_fbs"] = base["G_fbs_fbs"][idx]
            frame["strategy"] = strategy
            frame = frame.reset_index()
            outputs.append(frame)
            report["folds"][f"{strategy}_{year}"] = {"epochs": epochs,
                "fbs_involved": metrics(frame), "fbs_fbs": metrics(frame[frame.fbs_fbs])}
            print(json.dumps(report["folds"][f"{strategy}_{year}"], indent=2), flush=True)
            (output / "report.json").write_text(json.dumps(report, indent=2))
    all_rows = pd.concat(outputs, ignore_index=True)
    all_rows.to_parquet(output / "predictions.parquet", index=False)
    report["pooled_fbs_fbs"] = {s: metrics(g[g.fbs_fbs]) for s, g in all_rows.groupby("strategy")}
    ff = all_rows[all_rows.fbs_fbs].copy()
    ff["ae"] = abs(ff.em-ff.margin)
    paired = ff.pivot(index=["game_id", "season", "week_c"], columns="strategy", values="ae")
    diff = (paired.chronological-paired.hash).rename("difference").reset_index()
    groups = [g.difference.to_numpy() for _, g in diff.groupby(["season", "week_c"])]
    rng = np.random.default_rng(0)
    boot = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(2000)]
    report["paired_margin_mae_chronological_minus_hash"] = {
        "difference": float(diff.difference.mean()),
        "week_cluster_bootstrap_95ci": np.percentile(boot, [2.5,97.5]).tolist()}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["pooled_fbs_fbs"], indent=2))
    print(report["paired_margin_mae_chronological_minus_hash"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--output", type=Path, default=MODELS / "experiments" / "chronological_20260906")
    args = parser.parse_args()
    run(args.years, args.seeds, args.output, args.max_epochs)
