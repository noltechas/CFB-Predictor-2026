"""Fair fixed-cohort comparison for a supplemental complete-history fit.

This is deliberately separate from the ordinary experiment acceptance checks:
the candidate may have a changed historical store, but architecture, code,
training settings, and the 2025 target cohort must remain fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cfbpred.backtest_pregame import metrics
from cfbpred.pregame_experiments import paired, save_json, sha


def _read(run: Path) -> tuple[dict, pd.DataFrame]:
    run = Path(run)
    plan = json.loads((run / "plan.json").read_text())
    frame = pd.read_parquet(run / "predictions.parquet")
    required = {"game_id", "season", "week_c", "margin", "total", "fbs_fbs",
                "wp", "em", "et", "mm", "spread_close"}
    missing = required.difference(frame.columns)
    if missing or frame.game_id.duplicated().any():
        raise ValueError(f"invalid prediction frame: missing={sorted(missing)} duplicates={frame.game_id.duplicated().sum()}")
    return plan, frame


def _fixed_metrics(frame: pd.DataFrame) -> dict:
    out = metrics(frame)
    if "crps" in frame:
        out["crps"] = float(frame.crps.mean())
    bins = pd.cut(frame.wp, np.linspace(0, 1, 11), include_lowest=True)
    out["ece_10"] = float(sum(abs((g.margin > 0).mean() - g.wp.mean()) * len(g) / len(frame)
                              for _, g in frame.groupby(bins, observed=True)))
    return out


def compare(benchmark: Path, candidate: Path, market: Path, output: Path,
            expected_variant: str = "full_context", expected_year: int = 2025,
            expected_seed: int = 0, expected_epochs: int = 14,
            expected_refit_through: int = 2024, expected_micro_batch: int = 64,
            expected_epoch_selection_year: int = 2024) -> dict:
    bp, baseline = _read(benchmark)
    cp, expanded = _read(candidate)
    # Architecture/model code/training settings remain fixed; source and
    # history hashes are the intentional supplemental change.
    for key in ("variant", "year", "seed", "fixed_epochs", "refit_through_year",
                "micro_batch", "epoch_selection_year", "labels_2026_used"):
        if bp.get(key) != cp.get(key):
            raise ValueError(f"comparison setting changed: {key}")
    if (bp.get("variant"), bp.get("year"), bp.get("seed"), bp.get("fixed_epochs"),
            bp.get("refit_through_year"), bp.get("labels_2026_used")) != (expected_variant, expected_year, expected_seed,
                                                                            expected_epochs, expected_refit_through, False):
        raise ValueError("benchmark does not match requested 2025 14-epoch fit contract")
    if bp.get("micro_batch") != expected_micro_batch or bp.get("epoch_selection_year") != expected_epoch_selection_year:
        raise ValueError("benchmark micro-batch or epoch-selection year differs")
    if bp.get("code") != cp.get("code"):
        raise ValueError("architecture/model code hashes differ")
    if bp.get("epoch_source") != cp.get("epoch_source"):
        raise ValueError("epoch-selection provenance differs")

    market_frame = pd.read_parquet(market)
    if market_frame.game_id.duplicated().any() or "spread_close" not in market_frame:
        raise ValueError("market cohort is not unique or lacks spread_close")
    market_ids = set(market_frame.loc[market_frame.spread_close.notna(), "game_id"])
    base_target = baseline[(baseline.season == expected_year) & baseline.fbs_fbs &
                           baseline.spread_close.notna()].copy()
    target_ids = set(base_target.game_id)
    if not target_ids.issubset(market_ids):
        raise ValueError("market file is missing benchmark target IDs")
    cand_target = expanded[expanded.game_id.isin(target_ids)].copy()
    if set(cand_target.game_id) != target_ids or len(cand_target) != len(target_ids):
        raise ValueError("candidate target IDs differ from fixed benchmark cohort")
    base_target = base_target.sort_values("game_id").reset_index(drop=True)
    cand_target = cand_target.sort_values("game_id").reset_index(drop=True)
    market_values = market_frame.set_index("game_id").loc[base_target.game_id, "spread_close"].to_numpy()
    if not np.array_equal(base_target.spread_close.to_numpy(), market_values, equal_nan=True):
        raise ValueError("benchmark market values differ from supplied fixed market cohort")
    if not np.array_equal(base_target.game_id, cand_target.game_id):
        raise ValueError("candidate target ordering differs")
    for col in ("margin", "total", "season", "week_c", "fbs_fbs"):
        if not np.array_equal(base_target[col].to_numpy(), cand_target[col].to_numpy(), equal_nan=True):
            raise ValueError(f"candidate target labels differ: {col}")
    for frame_name, frame in (("benchmark", base_target), ("candidate", cand_target)):
        for col in ("wp", "em", "et", "mm"):
            if not np.isfinite(frame[col].to_numpy(dtype=float)).all():
                raise ValueError(f"{frame_name} contains non-finite predictions: {col}")
        if not ((frame.wp >= 0).all() and (frame.wp <= 1).all()):
            raise ValueError(f"{frame_name} win probabilities outside [0,1]")
    if not np.array_equal(base_target.spread_close.to_numpy(), cand_target.spread_close.to_numpy(), equal_nan=True):
        raise ValueError("market values differ between candidates")

    # paired() supplies the established deterministic week-cluster bootstrap;
    # mark this fixed cohort as the only eligible population.
    base_target = base_target.copy(); cand_target = cand_target.copy()
    base_target["fbs_fbs"] = True; cand_target["fbs_fbs"] = True
    paired_report = paired(base_target, cand_target)
    provenance = {
        "benchmark_plan_sha256": sha(Path(benchmark) / "plan.json"),
        "candidate_plan_sha256": sha(Path(candidate) / "plan.json"),
        "benchmark_source_sha256": bp.get("source_sha256"),
        "candidate_source_sha256": cp.get("source_sha256"),
        "benchmark_stores": bp.get("stores"), "candidate_stores": cp.get("stores"),
        "history_changed": bp.get("stores") != cp.get("stores") or bp.get("source_sha256") != cp.get("source_sha256"),
    }
    report = {
        "assessment": "supplemental_complete_history_fixed_2025_market_cohort",
        "target_season": expected_year, "target_games": len(target_ids),
        "market_target_games": len(target_ids), "market_path": str(Path(market).resolve()),
        "benchmark_run": str(Path(benchmark).resolve()), "candidate_run": str(Path(candidate).resolve()),
        "benchmark": _fixed_metrics(base_target), "candidate": _fixed_metrics(cand_target),
        "paired": paired_report, "provenance": provenance,
        "reused_2025_benchmark": True, "fresh_untouched_test": False,
        "labels_after_2024_used_for_fitting": False,
        "historical_reconstructed_pages_separate": True,
        "primary_acceptance": {"margin_mae_decrease": bool(
            _fixed_metrics(cand_target)["margin_mae"] < _fixed_metrics(base_target)["margin_mae"]),
            "no_acceptance_semantics_changed": True}, "production_modified": False,
    }
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    save_json(output / "comparison.json", report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--market", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(compare(a.benchmark, a.candidate, a.market, a.output), indent=2))


if __name__ == "__main__":
    main()
