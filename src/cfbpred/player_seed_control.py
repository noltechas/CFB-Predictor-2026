"""Match the ensemble member count when evaluating player/control mixtures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cfbpred.player_experiment import OUTPUT,load_inputs,run_variant
from cfbpred.pregame_experiments import paired,save_json,sha,summarize


def prepare(output=OUTPUT):
    plan={"purpose":"Separate player-data contribution from the benefit of simply averaging more independent members",
          "development_control_seeds":[0,1,2,3],"assessment_control_seeds":[0,1,2,3,4,5],
          "development_years":[2020,2021,2022],"assessment_years":[2023,2024,2025],
          "selection":"No new model or weight selection; compare the already development-selected mixture with equal-member scalar control",
          "main_plan_sha256":sha(output/"plan.json"),"blend_selection_sha256":sha(output/"blend_selection.json"),
          "code_sha256":sha(__file__)}
    path=output/"seed_control_plan.json"
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError("Frozen seed-count control changed")
    save_json(path,plan);return plan


def train(output=OUTPUT):
    plan=prepare(output);base=load_inputs(output/"inputs")
    torch.set_num_threads(2)
    for phase in ["development","assessment"]:
        run_variant(base,output/"inputs","corrected_control",plan[phase+"_years"],
                    plan[phase+"_control_seeds"],torch.device("cpu"),output)


def evaluate(output=OUTPUT):
    plan=prepare(output)
    control=pd.concat([pd.read_parquet(output/"runs/corrected_control"/f"{y}_6seeds_predictions.parquet")
                       for y in plan["assessment_years"]],ignore_index=True)
    blend=pd.read_parquet(output/"blend_assessment_predictions.parquet")
    report={"control_members":6,"blend_player_members":3,"blend_control_members":3,
            "control":summarize(control),"blend":summarize(blend),"paired_blend_vs_control":paired(control,blend),
            "early_weeks_1_3":{"control":summarize(control[control.week_c.le(3)]),"blend":summarize(blend[blend.week_c.le(3)])},
            "by_year":{str(y):{"control":summarize(control[control.season.eq(y)]),"blend":summarize(blend[blend.season.eq(y)])}
                       for y in plan["assessment_years"]},
            "additional_control_seeds_backend":"cpu; original assessment control seeds use mps",
            "production_modified":False}
    save_json(output/"seed_control_report.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("action",choices=["prepare","train","evaluate"])
    ap.add_argument("--output",type=Path,default=OUTPUT)
    a=ap.parse_args();globals()[a.action](a.output)
