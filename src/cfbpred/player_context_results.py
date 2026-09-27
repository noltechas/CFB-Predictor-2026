"""Paired comparison of the clean full-history/context runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cfbpred.pregame_experiments import sha,save_json,summarize,paired


def compare(root):
    names={"control":"control_2025_s0","full_history":"full_history_2025_s0",
           "full_context":"full_context_checked_2025_s0"}
    frames={};reports={};lineage={}
    for name,folder in names.items():
        path=root/"runs"/folder
        if (path/"INVALIDATED.json").exists():raise ValueError("Invalidated comparison run")
        plan=json.loads((path/"plan.json").read_text());bundle=json.loads((path/"bundle.json").read_text())
        if sha(path/"plan.json")!=bundle["plan_sha256"]:raise ValueError("Comparison plan changed")
        if sha(path/"predictions.parquet")!=bundle["predictions_sha256"]:raise ValueError("Forecasts changed")
        for code,digest in plan["code"].items():
            if sha(path/"code_snapshot"/code)!=digest:raise ValueError("Code snapshot changed")
        frame=pd.read_parquet(path/"predictions.parquet")
        if frame.game_id.duplicated().any():raise ValueError("Duplicate assessment forecast")
        frames[name]=frame;reports[name]=summarize(frame)
        lineage[name]=dict(plan_sha256=sha(path/"plan.json"),predictions_sha256=sha(path/"predictions.parquet"))
    control=frames["control"]
    for frame in frames.values():
        if set(frame.game_id)!=set(control.game_id):raise ValueError("Assessment coverage differs")
    changes={name:paired(control,frame) for name,frame in frames.items() if name!="control"}
    report=dict(forecast_year=2025,outer_seeds=1,internal_tabm_members=16,
        comparison_is_reused_historical_benchmark=True,metrics=reports,paired=changes,lineage=lineage,
        same_14_epoch_budget=True,production_modified=False,
        conclusions=["All resolved collected prior games are accessible without an eight-game cap",
                     "One-year/one-outer-seed results do not establish a durable improvement",
                     "Historical injury coverage is sparse; role/context ablation combines changes",
                     "No production promotion is justified by this exploratory comparison alone"],
        invalidated_runs=["full_context_2025_s0","full_context_2026_s0"],
        invalidation_reason="Four timestamped news articles had revisions at or after kickoff; 56 facts excluded and context candidates refit from scratch")
    save_json(root/"comparison.json",report)
    lines=["| Candidate | FBS/FBS margin MAE | Market-matched MAE | Winner log loss | Total MAE |",
           "|---|---:|---:|---:|---:|"]
    for name,m in reports.items():
        lines.append(f"| {name} | {m['margin_mae']:.4f} | {m['market_matched']['model_mae']:.4f} | {m['wp_logloss']:.4f} | {m['total_mae']:.4f} |")
    lines.append(f"\nMarket margin MAE on the same {reports['control']['market_matched']['n']} games: {reports['control']['market_matched']['market_mae']:.4f}.")
    for name,r in changes.items():
        lo,hi=r['week_cluster_bootstrap_95ci']
        lines.append(f"\n{name}: margin MAE change {r['margin_mae_change']:+.4f}; week-cluster bootstrap 95% interval [{lo:+.4f}, {hi:+.4f}].")
    (root/"comparison.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args();compare(a.root)
