"""Predeclared discrete-mixture challenger, then equal-size confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cfbpred.architecture_experiment import REFERENCE
from cfbpred.config import MODELS, MARGIN_LO, MARGIN_HI, TOTAL_LO, TOTAL_HI
from cfbpred.mixture_head import DiscreteMixtureHead
from cfbpred.pregame_experiments import ASSESSMENT, DEVELOPMENT, choose, paired, save_json, sha, summarize
from cfbpred.shared_team_experiment import run_variant
from cfbpred.train import build_tensors, make_model

OUTPUT = MODELS / "experiments" / "mixture_20260906"
CODE = ("mixture_experiment.py","mixture_head.py","shared_team_experiment.py", "shared_team_net.py",
        "architecture_experiment.py","consistent_net.py","train.py","raw_net.py","model.py",
        "backtest_pregame.py","pregame_experiments.py")


def make_mixture(d):
    model = make_model(d,s2d_gate=True,delta=True)
    model.pre.margin = DiscreteMixtureHead(160,MARGIN_LO,MARGIN_HI)
    model.pre.total = DiscreteMixtureHead(160,TOTAL_LO,TOTAL_HI,centre=55.)
    return model


def prepare(output):
    source = REFERENCE / "inputs"
    old = json.loads((REFERENCE / "plan.json").read_text())
    for name,digest in old["code_sha256"].items():
        if sha(Path(__file__).parent/name)!=digest:
            raise ValueError(f"Control code changed: {name}")
    for name,digest in old["input_sha256"].items():
        if sha(source/name)!=digest:
            raise ValueError(f"Frozen input changed: {name}")
    original = REFERENCE / "runs" / "baseline__none"
    plan = {
        "candidate":"Original raw encoder; three-component integrated logistic PMF heads with global learned bin biases for margin and total",
        "development":DEVELOPMENT,"assessment":ASSESSMENT,"seeds":[0,1,2],
        "motivation":"After consistency-only gain did not replicate and shared encoder failed development, test distribution-head regularization",
        "selection":"Development margin MAE gain >=0.03, WP LL <=control+0.005, Brier <=control+0.002",
        "confirmation":"If selected, three-seed assessment gain >=0.10 with WP LL <=control+0.005, Brier <=control+0.002, total MAE <=control+0.10 advances both models to 16 seeds",
        "promotion":"16-seed margin gain >=0.10, paired week-cluster bootstrap 95CI wholly below zero, same probability/total guardrails; requires operational validation",
        "2026_labels_used":False,"input_sha256":old["input_sha256"],
        "control_sha256":{p.name:sha(p) for p in sorted(original.glob("*.npz"))},
        "code_sha256":{n:sha(Path(__file__).parent/n) for n in CODE}}
    output.mkdir(parents=True,exist_ok=True)
    path = output / "plan.json"
    if path.exists() and json.loads(path.read_text())!=plan:
        raise ValueError("Frozen mixture experiment changed; use a new output directory")
    save_json(path,plan)
    snapshot = output / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for n in CODE:
        shutil.copy2(Path(__file__).parent/n,snapshot/n)
    return source


def comparison(frames):
    scores = {n:summarize(f) for n,f in frames.items()}
    pair = paired(frames["baseline"],frames["mixture"])
    c,n = scores["baseline"],scores["mixture"]
    strong = (n["margin_mae"]<=c["margin_mae"]-.10 and n["wp_logloss"]<=c["wp_logloss"]+.005
              and n["brier"]<=c["brier"]+.002 and n["total_mae"]<=c["total_mae"]+.10)
    return {"scores":scores,"pairing":pair,"passes_point_estimate_gate":bool(strong),
            "passes_promotion_gate":bool(strong and pair["week_cluster_bootstrap_95ci"][1]<0),
            "by_year":{n:{str(y):summarize(f[f.season==y]) for y in sorted(f.season.unique())} for n,f in frames.items()}}


def run(output):
    source = prepare(output)
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base = build_tensors(data_dir=source,include_plays=False)
    def forecasts(years,seeds):
        return {n:run_variant(base,source,output,n,years,seeds,device,
                    model_factory=make_mixture if n=="mixture" else None) for n in ["baseline","mixture"]}
    dev = forecasts(DEVELOPMENT,range(3))
    selected = choose({n:summarize(f) for n,f in dev.items()},"baseline")
    choice = {"development":comparison(dev),"selected":selected}
    path = output / "choice.json"
    if path.exists() and json.loads(path.read_text())!=choice:
        raise ValueError("Frozen development choice changed")
    save_json(path,choice)
    print("Development choice: "+json.dumps(choice,indent=2),flush=True)
    report = {**choice,"production_modified":False}
    save_json(output / "report.json",report)
    if selected=="mixture":
        report["assessment_3seed"] = comparison(forecasts(ASSESSMENT,range(3)))
        save_json(output / "report.json",report)
        if report["assessment_3seed"]["passes_point_estimate_gate"]:
            print("Extending both configurations to 16 seeds",flush=True)
            report["confirmation_16seed"] = comparison(forecasts(ASSESSMENT,range(16)))
            save_json(output / "report.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args = parser.parse_args()
    run(args.output)
