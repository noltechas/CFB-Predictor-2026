"""Predeclared development-only blending of corrected control/player PMFs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cfbpred.ensemble import readouts
from cfbpred.model import discrete_crps
from cfbpred.player_experiment import OUTPUT
from cfbpred.pregame_experiments import paired, save_json, sha, summarize


def prepare(output=OUTPUT):
    plan={"player_variant":"Selected by the main experiment's 2020-2022 development margin MAE",
          "player_weights":[0,.25,.5,.75,1],"development_years":[2020,2021,2022],"development_seeds":[0,1],
          "assessment_years":[2023,2024,2025],"assessment_seeds":[0,1,2],
          "selection":"Lowest pooled development margin MAE, tie broken by lower player weight; freeze before assessment read",
          "combination":"Convex mixture of control/player margin PMFs and total PMFs with the same weight",
          "code_sha256":sha(__file__),"main_plan_sha256":sha(output/"plan.json"),
          "purpose":"Test whether a complementary player model adds value in combination, including small gains"}
    path=output/"blend_plan.json"
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError("Frozen blend plan changed")
    save_json(path,plan)
    return plan


def frame(output,variant,years,seeds,weight):
    result=[]
    for year in years:
        arrays={}
        for n in ["corrected_control",variant]:
            members=[np.load(output/"runs"/n/f"{year}_s{s}.npz") for s in seeds]
            ids=members[0]["game_id"]
            if any(not np.array_equal(m["game_id"],ids) for m in members):raise ValueError("Member game sets differ")
            arrays[n]={key:np.mean([m[key] for m in members],axis=0) for key in ["pm","pt"]}
            arrays[n]["ids"]=ids
        if not np.array_equal(arrays[variant]["ids"],arrays["corrected_control"]["ids"]):raise ValueError("Control/player game sets differ")
        pm,pt=[torch.tensor((1-weight)*arrays["corrected_control"][key]+weight*arrays[variant][key]) for key in ["pm","pt"]]
        f=pd.read_parquet(output/"runs"/variant/f"{year}_{len(seeds)}seeds_predictions.parquet").set_index("game_id").loc[ids].reset_index()
        for key,value in readouts(pm,pt).items():f[key]=value
        f["crps"]=discrete_crps(pm.clamp(min=1e-30).log(),torch.tensor(f.margin.to_numpy()),reduction="none").numpy()
        result.append(f)
    return pd.concat(result,ignore_index=True)


def evaluate(output=OUTPUT):
    plan=prepare(output);main=json.loads((output/"report.json").read_text());variant=main["selected_pooling"]
    dev={w:frame(output,variant,plan["development_years"],plan["development_seeds"],w) for w in plan["player_weights"]}
    weight=min(plan["player_weights"],key=lambda w:summarize(dev[w])["margin_mae"])
    selection={"player_variant":variant,"player_weight":weight,"control_weight":1-weight,
               "development":{str(w):summarize(f) for w,f in dev.items()}}
    path=output/"blend_selection.json"
    if path.exists() and json.loads(path.read_text())!=selection:raise ValueError("Frozen blend selection changed")
    save_json(path,selection)
    if "assessment" not in main:
        print(json.dumps(selection,indent=2));return
    assessment={w:frame(output,variant,plan["assessment_years"],plan["assessment_seeds"],w) for w in set([0,weight,1])}
    candidate=assessment[weight]
    report={**selection,"assessment":summarize(candidate),"paired_vs_corrected_control":paired(assessment[0],candidate),
            "early_weeks_1_3":summarize(candidate[candidate.week_c.le(3)]),
            "by_year":{str(y):summarize(candidate[candidate.season.eq(y)]) for y in plan["assessment_years"]},
            "production_modified":False}
    candidate.to_parquet(output/"blend_assessment_predictions.parquet",index=False)
    save_json(output/"blend_report.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("action",choices=["prepare","evaluate"])
    ap.add_argument("--output",type=Path,default=OUTPUT)
    a=ap.parse_args();prepare(a.output) if a.action=="prepare" else evaluate(a.output)
