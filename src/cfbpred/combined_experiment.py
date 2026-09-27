"""Test cumulative changes, without requiring a minimum-sized MAE gain.

User-requested follow-up to isolated experiments: evaluate the compatible
feature/architecture bundle, then the bundle with the previously tested loss.
Both settings are fixed before results, both later-year results are reported.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cfbpred.architecture_experiment import REFERENCE, challenger, fold_masks
from cfbpred.backtest_pregame import fit_inputs, predict
from cfbpred.config import MODELS
from cfbpred.consistent_net import fit_consistent_inputs
from cfbpred.ensemble import readouts
from cfbpred.model import discrete_crps
from cfbpred.pregame_experiments import ASSESSMENT, DEVELOPMENT, add_form, paired, save_json, sha, summarize
from cfbpred.qb_roster_prior import attach
from cfbpred.train import build_tensors, make_model, pretrain

OUTPUT = MODELS / "experiments" / "combined_20260906"
VARIANTS = {"baseline": {"sel_mae_w":0.,"margin_crps_w":0.},
            "combined": {"sel_mae_w":0.,"margin_crps_w":0.},
            "combined_margin": {"sel_mae_w":.05,"margin_crps_w":.30}}
CODE = ("combined_experiment.py","architecture_experiment.py","consistent_net.py",
        "qb_roster_prior.py","train.py","raw_net.py","model.py","backtest_pregame.py",
        "pregame_experiments.py","ensemble.py")


def prepare(output=OUTPUT):
    old = json.loads((REFERENCE / "plan.json").read_text())
    source = REFERENCE / "inputs"
    for name,digest in old["code_sha256"].items():
        if sha(Path(__file__).parent/name)!=digest:
            raise ValueError(f"Control code changed: {name}")
    for name,digest in old["input_sha256"].items():
        if sha(source/name)!=digest:
            raise ValueError(f"Frozen input changed: {name}")
    qb = MODELS / "experiments" / "qb_prior_20260906" / "inputs" / "game_qb_priors.parquet"
    prior = json.loads((qb.parent / "coverage.json").read_text())
    if sha(qb)!=prior["game_features_sha256"]:
        raise ValueError("Frozen QB prior changed")
    output.mkdir(parents=True,exist_ok=True)
    plan = {
        "variants":VARIANTS,
        "bundle":"Chronological selection/refit; original active inputs + five-game raw form/matchups + preseason roster passing priors; order-consistent network",
        "omitted":"Alternative three-game windows, shared encoder, mixture head, smaller training window: alternatives or measured regressions, not additive gains",
        "development":DEVELOPMENT,"assessment":ASSESSMENT,"initial_seeds":[0,1,2],
        "comparison":"Report BOTH bundles in both periods; no significance or minimum-MAE hurdle for initial assessment",
        "confirmation":"Any later-year positive pooled margin gain with improvement in >=2 of 3 seasons, winner logloss <=control+0.002, Brier <=control+0.001, total MAE <=control+0.10: extend it AND control to 16 seeds",
        "promotion":"No minimum effect size: report magnitude, paired uncertainty and secondary metrics at 16 seeds; retain uncertain candidates for later study without claiming proven improvement",
        "2026_outcomes_used":False,"historical_assessment_previously_examined":True,
        "input_sha256":old["input_sha256"],"form_sha256":sha(source/"recent_form.parquet"),"qb_prior_sha256":sha(qb),
        "code_sha256":{n:sha(Path(__file__).parent/n) for n in CODE},
        "control_sha256":{p.name:sha(p) for p in sorted((REFERENCE/"runs/baseline__none").glob("*.npz"))}}
    path = output/"plan.json"
    if path.exists() and json.loads(path.read_text())!=plan:
        raise ValueError("Frozen combined experiment changed; use a new output directory")
    save_json(path,plan)
    snap=output/"code_snapshot";snap.mkdir(exist_ok=True)
    for n in CODE:
        shutil.copy2(Path(__file__).parent/n,snap/n)
    return source,qb


def run_variant(base,source,name,years,seeds,device,output=OUTPUT):
    directory=output/"runs"/name;directory.mkdir(parents=True,exist_ok=True)
    meta=pd.read_parquet(source/"game_scalars.parquet")[["game_id","season","week_c","margin","total"]].merge(
        pd.read_parquet(source/"market.parquet"),on="game_id",how="left").set_index("game_id")
    rows=[]
    for year in years:
        tr,va,te,refit=fold_masks(base,year,None)
        fit=fit_inputs if name=="baseline" else fit_consistent_inputs
        fitted,refitted=fit(base,tr),fit(base,refit)
        idx=np.where(te)[0]
        margins,totals,epochs=[],[],[]
        for seed in seeds:
            cache=directory/f"{year}_s{seed}.npz"
            original=REFERENCE/"runs/baseline__none"/cache.name
            if (name=="baseline" and seed<3) or cache.exists():
                with np.load(original if name=="baseline" and seed<3 else cache) as saved:
                    if "game_id" in saved and not np.array_equal(saved["game_id"],base["G_gid"][idx]):
                        raise ValueError("Cached prediction order changed")
                    pm,pt,epoch=torch.tensor(saved["pm"]),torch.tensor(saved["pt"]),int(saved["epochs"])
            else:
                make=(lambda d:make_model(d,s2d_gate=True,delta=True)) if name=="baseline" else (lambda d:challenger(d,True))
                print(f"{name} forecast={year} seed={seed} columns={len(base['scalar_cols'])}",flush=True)
                torch.manual_seed(seed);np.random.seed(seed)
                model=make(fitted).to(device)
                pretrain(model,fitted,device,train_indices=np.where(tr)[0],val_indices=np.where(va)[0],**VARIANTS[name])
                epoch=model.pre.selected_epochs
                torch.manual_seed(seed);np.random.seed(seed)
                model=make(refitted).to(device)
                pretrain(model,refitted,device,train_indices=np.where(refit)[0],val_indices=np.where(va)[0],fixed_epochs=epoch,**VARIANTS[name])
                pm,pt=predict(model,refitted,idx,device)
                torch.save({"state_dict":{k:v.cpu() for k,v in model.pre.state_dict().items()},
                            "mu":refitted["mu"],"sd":refitted["sd"],"scalar_cols":base["scalar_cols"],
                            "variant":name,"forecast_year":year,"seed":seed,"epochs":epoch,
                            "train_game_ids":base["G_gid"][refit]},directory/f"{year}_s{seed}.pt")
                np.savez_compressed(cache,pm=pm.numpy(),pt=pt.numpy(),epochs=epoch,game_id=base["G_gid"][idx])
                del model
                if device.type=="mps":torch.mps.empty_cache()
            margins.append(pm);totals.append(pt);epochs.append(epoch)
        pm,pt=torch.stack(margins).mean(0),torch.stack(totals).mean(0)
        frame=pd.DataFrame(readouts(pm,pt),index=base["G_gid"][idx]).join(meta)
        frame["crps"]=discrete_crps(pm.clamp(min=1e-30).log(),base["G_margin"][idx],reduction="none").numpy()
        frame["fbs_fbs"]=base["G_fbs_fbs"][idx];frame["variant"]=name;frame.index.name="game_id"
        frame=frame.reset_index()
        frame.to_parquet(directory/f"{year}_{len(seeds)}seeds_predictions.parquet",index=False)
        scores=summarize(frame)
        save_json(directory/f"{year}_{len(seeds)}seeds_report.json",{"epochs":epochs,"metrics":scores})
        print(f"Completed {name} {year}, {len(seeds)} seeds: MAE={scores['margin_mae']:.4f}, LL={scores['wp_logloss']:.5f}",flush=True)
        rows.append(frame)
    return pd.concat(rows,ignore_index=True)


def worth_confirming(control,candidate):
    """Small positive gains count; repeatability and other outputs also count."""
    c,n=summarize(control),summarize(candidate)
    wins=sum(summarize(candidate[candidate.season==y])["margin_mae"] < summarize(control[control.season==y])["margin_mae"]
             for y in sorted(control.season.unique()))
    return bool(n["margin_mae"]<c["margin_mae"] and wins>=2 and n["wp_logloss"]<=c["wp_logloss"]+.002
                and n["brier"]<=c["brier"]+.001 and n["total_mae"]<=c["total_mae"]+.10)


def compare(frames):
    return {"scores":{n:summarize(f) for n,f in frames.items()},
            "pairings":{n:paired(frames["baseline"],f) for n,f in frames.items() if n!="baseline"},
            "by_year":{n:{str(y):summarize(f[f.season==y]) for y in sorted(f.season.unique())} for n,f in frames.items()}}


def run():
    source,qb=prepare()
    torch.set_num_threads(4)
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base=build_tensors(data_dir=source,include_plays=False)
    bundle=attach(add_form(base,source,"recent5_matchup"),pd.read_parquet(qb))
    assert len(bundle["scalar_cols"])==324
    def forecasts(years,seeds,names):
        return {n:run_variant(base if n=="baseline" else bundle,source,n,years,seeds,device) for n in names}
    dev=forecasts(DEVELOPMENT,range(3),VARIANTS)
    report={"development":compare(dev),"production_modified":False}
    save_json(OUTPUT/"report.json",report)
    assessment=forecasts(ASSESSMENT,range(3),VARIANTS)
    report["assessment_3seed"]=compare(assessment)
    eligible=[n for n in VARIANTS if n!="baseline" and worth_confirming(assessment["baseline"],assessment[n])]
    report["advanced_to_confirmation"]=eligible
    save_json(OUTPUT/"report.json",report)
    if eligible:
        report["confirmation_16seed"]=compare(forecasts(ASSESSMENT,range(16),["baseline"]+eligible))
        save_json(OUTPUT/"report.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    run()
