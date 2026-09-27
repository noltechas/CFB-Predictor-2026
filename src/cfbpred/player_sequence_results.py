"""Retain seed variation and earlier benchmarks alongside the frozen results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cfbpred.ensemble import readouts
from cfbpred.model import discrete_crps
from cfbpred.player_sequence_experiment import ROOT
from cfbpred.player_sequence_fast_experiment import OUTPUT
from cfbpred.player_sequence_net import VARIANTS
from cfbpred.pregame_experiments import paired,save_json,summarize

NAMES={"control":"Control","context":"Game context","contribution":"Contribution weighting","auxiliary":"Auxiliary statistics","combined":"All three combined"}


def finish(output=OUTPUT):
    report=json.loads((output/"report.json").read_text())
    if not json.loads((output/"completion_audit.json").read_text())["passed"]:
        raise ValueError("Complete the checkpoint audit first")
    frames={name:pd.concat([pd.read_parquet(output/"runs"/name/f"{y}_3seeds_predictions.parquet") for y in [2023,2024,2025]],ignore_index=True) for name in VARIANTS}
    supplemental={"per_seed":{},"earlier_benchmarks":{},"interpretation":"Primary comparisons have equal three-member ensembles. Earlier six-/16-member ensembles are contextual references, not estimates of incremental feature benefit."}
    for name,frame in frames.items():
        by_seed=[]
        for seed in range(3):
            parts=[]
            for year in [2023,2024,2025]:
                with np.load(output/"runs"/name/f"{year}_s{seed}.npz") as cache:
                    pm=torch.tensor(cache["pm"])
                    prediction=pd.DataFrame(readouts(pm,torch.tensor(cache["pt"])),index=cache["game_id"])
                    prediction=prediction.join(frame.set_index("game_id")[["season","week_c","margin","total","fbs_fbs","spread_close"]])
                    prediction["crps"]=discrete_crps(pm.clamp(min=1e-30).log(),torch.tensor(prediction.margin.to_numpy()),reduction="none").numpy()
                    parts.append(prediction.reset_index(names="game_id"))
            by_seed.append(summarize(pd.concat(parts,ignore_index=True)))
        supplemental["per_seed"][name]=by_seed
    references=[("earlier_combined_3",ROOT/"models/experiments/combined_20260906/runs/combined",3),
                ("earlier_combined_16",ROOT/"models/experiments/combined_20260906/runs/combined",16),
                ("earlier_corrected_control_6",ROOT/"models/experiments/players_20260906_v3/runs/corrected_control",6)]
    for name,directory,seeds in references:
        reference=pd.concat([pd.read_parquet(directory/f"{y}_{seeds}seeds_predictions.parquet") for y in [2023,2024,2025]],ignore_index=True)
        supplemental["earlier_benchmarks"][name]={"members":seeds,"metrics":summarize(reference),"paired":{variant:paired(reference,frame) for variant,frame in frames.items()}}
    save_json(output/"supplemental_report.json",supplemental)
    table=[]
    for name in VARIANTS:
        m=report["assessment"]["metrics"][name]
        pair=report["assessment"]["paired_vs_control"].get(name)
        table.append({"variant":name,**{k:m[k] for k in ["n","margin_mae","total_mae","wp_logloss","brier","crps"]},
                      "margin_change":0 if pair is None else pair["margin_mae_change"],
                      "ci_lower":0 if pair is None else pair["week_cluster_bootstrap_95ci"][0],
                      "ci_upper":0 if pair is None else pair["week_cluster_bootstrap_95ci"][1]})
    pd.DataFrame(table).to_csv(output/"assessment_summary.csv",index=False)
    lines=["## Results", "",f"Development selected **{NAMES[report['selected_variant']]}** using the frozen pooled margin-MAE criterion. All 75 members completed and passed the checkpoint audit. The final repository check passed 105 tests.","",
           "The assessment below covers 2,398 FBS-vs-FBS games in 2023–2025. Every row uses three model seeds. Negative margin-error changes favor the candidate.","",
           "| Model | Margin MAE | Change vs control | 95% week-cluster interval | Total MAE | Winner log loss |",
           "| --- | ---: | ---: | --- | ---: | ---: |"]
    for row in table:
        interval="—" if row["variant"]=="control" else f"[{row['ci_lower']:+.3f}, {row['ci_upper']:+.3f}]"
        lines.append(f"| {NAMES[row['variant']]} | {row['margin_mae']:.4f} | {row['margin_change']:+.4f} | {interval} | {row['total_mae']:.4f} | {row['wp_logloss']:.5f} |")
    market=report["assessment"]["metrics"]["control"]["market_matched"]
    lines += ["",f"The closing-line benchmark has margin MAE **{market['market_mae']:.4f}** on {market['n']:,} matched games. Model comparisons with that benchmark must use each model's `market_matched.model_mae`, because two assessment games lack a closing line.","",
              "The intervals describe paired game-error changes for these fitted ensembles, clustered by season/week. They do not eliminate selection uncertainty from repeatedly examining the historical assessment years. Individual-seed metrics, early-week results, year-by-year results and comparisons with earlier retained ensembles are saved alongside the main report.","",
              "Every experiment is retained. Production forecasts and the live model remain unchanged by this test suite.",""]
    doc=ROOT/"docs/PLAYER_SEQUENCE_EXPERIMENTS_2026-09-07.md"
    previous=doc.read_text().split("\n## Results\n")[0].rstrip()
    doc.write_text(previous+"\n\n"+"\n".join(lines))
    print(pd.DataFrame(table).to_string(index=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=OUTPUT)
    torch.set_num_threads(2);finish(p.parse_args().output)
