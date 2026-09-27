"""Diagnose expanded context inputs with the retained checkpoint, no fitting.

The output is a supplemental input-only diagnostic.  It reuses the baseline
checkpoint's normalization and vocabularies and evaluates the exact 2025
benchmark cohort, so it cannot be mistaken for a newly trained candidate.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cfbpred.backtest_pregame import metrics
from cfbpred.ensemble import readouts
from cfbpred.model import scores_from
from cfbpred.player_sequence_production import _checkpoint_inputs, _make_model, load_snapshot
from cfbpred.player_sequence_train import predict
from cfbpred.pregame_experiments import paired, save_json, sha


def fixed_cohort(frame: pd.DataFrame, season: int = 2025) -> pd.DataFrame:
    required = {"game_id", "season", "week_c", "margin", "total", "fbs_fbs", "spread_close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"benchmark missing cohort columns: {sorted(missing)}")
    out = frame[(frame.season == season) & frame.fbs_fbs & frame.spread_close.notna()].copy()
    if out.game_id.duplicated().any() or out.empty:
        raise ValueError("benchmark cohort must be non-empty and unique")
    return out.sort_values("game_id").reset_index(drop=True)


def _checkpoint_with_old_schema(checkpoint_path: Path, source: Path) -> dict:
    """Enrich the old checkpoint exactly as the staging adapter does."""
    import torch
    pinned = json.loads(Path(source).read_text())
    paths = {k: Path(v) for k, v in pinned.items()}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint = copy.deepcopy(checkpoint)
    checkpoint["sequence_schema"] = json.loads((paths["sequences"] / "manifest.json").read_text())
    checkpoint["context_cols"] = json.loads((paths["timeline"] / "manifest.json").read_text())["context_cols"]
    checkpoint["full_history_schema"] = json.loads((paths["full_history"] / "manifest.json").read_text())["event_cols"]
    return checkpoint


def _score(base: dict, checkpoint: dict, players: Path, ids: np.ndarray,
           batch_size: int = 256) -> pd.DataFrame:
    d = _checkpoint_inputs(base, checkpoint, players)
    model = _make_model(d, checkpoint["architecture"], checkpoint["variant"])
    model.pre.load_state_dict(checkpoint["state_dict"])
    with torch.no_grad():
        margin, total = predict(model, d, ids, torch.device("cpu"), batch_size=batch_size)
    derived = readouts(margin, total)
    home, away = scores_from(torch.tensor(derived["mm"]), torch.tensor(derived["mt"]))
    return pd.DataFrame({"game_id": base["G_gid"][ids].astype("int64"),
                         "wp": derived["wp"], "em": derived["em"], "et": derived["et"],
                         "mm": derived["mm"], "mt": derived["mt"],
                         "pred_home_pts": home.numpy(), "pred_away_pts": away.numpy()})


def diagnose(baseline_source: Path, expanded_source: Path, checkpoint: Path,
             benchmark_predictions: Path, output: Path, batch_size: int = 256) -> dict:
    torch.set_num_threads(2)
    baseline_source, expanded_source, checkpoint = map(Path, (baseline_source, expanded_source, checkpoint))
    benchmark = fixed_cohort(pd.read_parquet(benchmark_predictions))
    bpinned = json.loads(baseline_source.read_text())
    epinned = json.loads(expanded_source.read_text())
    bbase = load_snapshot(Path(bpinned["inputs"]), Path(bpinned["players"]),
                          Path(bpinned["sequences"]), baseline_source)
    ebase = load_snapshot(Path(epinned["inputs"]), Path(epinned["players"]),
                          Path(epinned["sequences"]), expanded_source)
    ids = benchmark.game_id.to_numpy(np.int64)
    bindex = {int(g): i for i, g in enumerate(np.asarray(bbase["G_gid"]))}
    eindex = {int(g): i for i, g in enumerate(np.asarray(ebase["G_gid"]))}
    if not set(ids).issubset(bindex) or not set(ids).issubset(eindex):
        raise ValueError("expanded or baseline source is missing fixed benchmark IDs")
    bidx = np.asarray([bindex[int(g)] for g in ids], dtype=np.int64)
    eidx = np.asarray([eindex[int(g)] for g in ids], dtype=np.int64)
    checkpoint_obj = _checkpoint_with_old_schema(checkpoint, baseline_source)
    if checkpoint_obj['train_max_season'] != 2024 or checkpoint_obj['forecast_year'] != 2025:
        raise ValueError('Diagnostic requires the fit-through-2024 assessment checkpoint')
    for pinned in (bpinned, epinned):
        labels = pd.read_parquet(Path(pinned['inputs']) / 'game_scalars.parquet').set_index('game_id').loc[ids]
        for name in ('season', 'week_c', 'margin', 'total'):
            if not np.array_equal(labels[name].to_numpy(), benchmark[name].to_numpy(), equal_nan=True):
                raise ValueError('Fixed diagnostic labels differ: ' + name)
    # _checkpoint_inputs applies the saved mu/sd and saved vocab to each store.
    baseline_pred = _score(bbase, checkpoint_obj, Path(bpinned["players"]), bidx, batch_size)
    expanded_pred = _score(ebase, checkpoint_obj, Path(epinned["players"]), eidx, batch_size)
    baseline_error = max(float(np.max(np.abs(baseline_pred[name].to_numpy() - benchmark[name].to_numpy())))
                         for name in ('wp', 'em', 'et', 'mm', 'mt'))
    # Float32 reductions can differ by a few ULPs across batch layouts;
    # discrete median readouts must still agree exactly.
    same = all(np.allclose(baseline_pred[name], benchmark[name], rtol=1e-6, atol=2e-6)
               for name in ('wp','em','et'))
    same = same and all(np.array_equal(baseline_pred[name], benchmark[name]) for name in ('mm','mt'))
    if not same:
        raise ValueError('Retained checkpoint does not reproduce the published benchmark')
    for pred in (baseline_pred, expanded_pred):
        if not np.isfinite(pred[['wp','em','et','mm','mt']]).all().all() or not pred.wp.between(0,1).all():
            raise ValueError('Invalid diagnostic forecast')
    for pred in (baseline_pred, expanded_pred):
        pred["season"] = benchmark.season.to_numpy()
        pred["week_c"] = benchmark.week_c.to_numpy()
        pred["margin"] = benchmark.margin.to_numpy()
        pred["total"] = benchmark.total.to_numpy()
        pred["fbs_fbs"] = True
        pred["spread_close"] = benchmark.spread_close.to_numpy()
    report = {
        "assessment": "fixed_weights_input_expansion",
        "fit": False, "labels_after_2024_used_for_fitting": False,
        "fixed_weights": True, "target_season": 2025, "target_games": len(benchmark),
        "benchmark_reproduced": True, "max_baseline_readout_error": baseline_error,
        "target_labels_exact": True, "reused_2025_benchmark": True,
        "baseline": metrics(baseline_pred), "expanded": metrics(expanded_pred),
        "paired": paired(baseline_pred, expanded_pred),
        "source_hashes": {"baseline": sha(baseline_source), "expanded": sha(expanded_source)},
        "checkpoint_sha256": sha(checkpoint), "benchmark_sha256": sha(benchmark_predictions),
        "production_modified": False,
    }
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    pd.concat([baseline_pred.assign(input_variant="baseline"), expanded_pred.assign(input_variant="expanded")], ignore_index=True).to_parquet(output / "predictions.parquet", index=False)
    save_json(output / "comparison.json", report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-source", type=Path, required=True)
    p.add_argument("--expanded-source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--benchmark-predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=256)
    a = p.parse_args()
    print(json.dumps(diagnose(a.baseline_source, a.expanded_source, a.checkpoint,
                              a.benchmark_predictions, a.output, a.batch_size), indent=2))


if __name__ == "__main__":
    main()
