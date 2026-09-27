"""Compare expanded public evidence with the retained full-context candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cfbpred.pregame_experiments import paired, save_json, sha, summarize


def read_run(path):
    if (path / "INVALIDATED.json").exists():
        raise ValueError("Cannot compare an invalidated run")
    plan = json.loads((path / "plan.json").read_text())
    bundle = json.loads((path / "bundle.json").read_text())
    for field, name in [("plan_sha256", "plan.json"), ("predictions_sha256", "predictions.parquet"),
                        ("member_sha256", "member.pt")]:
        if bundle[field] != sha(path / name):
            raise ValueError("Saved run changed: " + name)
    for name, digest in plan["code"].items():
        if sha(path / "code_snapshot" / name) != digest:
            raise ValueError("Code snapshot changed")
    frame = pd.read_parquet(path / "predictions.parquet")
    if frame.game_id.duplicated().any():
        raise ValueError("Duplicate forecast")
    return plan, frame


def compare(retained, expanded, output):
    before, baseline = read_run(retained)
    after, candidate = read_run(expanded)
    for key in ["variant", "year", "seed", "inputs", "code", "epoch_source", "fixed_epochs"]:
        if before[key] != after[key]:
            raise ValueError("Comparison changes more than evidence: " + key)
    for key in ["players", "sequences", "full_history"]:
        if before["stores"][key] != after["stores"][key]:
            raise ValueError("Pinned base store changed: " + key)
    if set(baseline.game_id) != set(candidate.game_id):
        raise ValueError("Comparison game coverage differs")
    report = dict(retained_run=str(retained.resolve()), expanded_run=str(expanded.resolve()),
                  retained=summarize(baseline), expanded=summarize(candidate),
                  paired=paired(baseline, candidate),
                  same_model_code_and_training_settings=True,
                  changed_input="Expanded timestamp-checked public availability; baseline facts retained",
                  reused_2025_benchmark=True, outer_seeds=1,
                  all_fbs_historical_availability_verified=False, production_modified=False)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "comparison.json", report)
    lines = ["| Candidate | Market-matched margin MAE | FBS/FBS margin MAE | Winner log loss |",
             "|---|---:|---:|---:|"]
    for label, key in [("Retained full context", "retained"), ("Expanded public availability", "expanded")]:
        metrics = report[key]
        lines.append(f"| {label} | {metrics['market_matched']['model_mae']:.6f} | {metrics['margin_mae']:.6f} | {metrics['wp_logloss']:.6f} |")
    lo, hi = report["paired"]["week_cluster_bootstrap_95ci"]
    lines += ["", f"Paired FBS/FBS margin MAE change: {report['paired']['margin_mae_change']:+.6f}; week-cluster 95% interval [{lo:+.6f}, {hi:+.6f}].",
              "", "Same architecture, code, pinned base inputs, seed and 14-epoch training budget; only the availability evidence changed. This is a reused historical benchmark. The retained model and production activation are unchanged."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["retained", "expanded", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    compare(args.retained, args.expanded, args.output)
