"""Frozen rolling-year evaluation and deployable pregame player checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cfbpred.architecture_experiment import REFERENCE, challenger, fold_masks
from cfbpred.backtest_pregame import predict
from cfbpred.config import MODELS
from cfbpred.consistent_net import fit_consistent_inputs
from cfbpred.ensemble import readouts
from cfbpred.model import discrete_crps
from cfbpred.player_inputs import OUTPUT as PLAYERS, PlayerStore
from cfbpred.player_net import make_player_model
from cfbpred.pregame_experiments import add_form, paired, save_json, sha, summarize
from cfbpred.qb_roster_prior import attach
from cfbpred.train import build_tensors, pretrain

OUTPUT = MODELS/"experiments/players_20260906_v3"
CODE = ["player_identity.py","player_inputs.py","player_scalar_corrections.py","player_net.py",
        "player_experiment.py","train.py","raw_net.py","consistent_net.py","backtest_pregame.py",
        "model.py","qb_roster_prior.py","pregame_experiments.py","ensemble.py", "player_catalog.py",
        "game_feature_snapshots.py", "player_features.py"]


def prepare(output=OUTPUT, player_path=PLAYERS):
    inputs = output/"inputs"
    for name in ["game_history.parquet","team_vocab.json","market.parquet","recent_form.parquet","snapshot_schedule.parquet"]:
        path = inputs/name
        if path.exists() and sha(path)!=sha(REFERENCE/"inputs"/name):
            raise ValueError(f"Frozen source changed: {name}")
        if not path.exists():shutil.copy2(REFERENCE/"inputs"/name,path)
    if not (inputs/"game_scalars.parquet").exists() or not (inputs/"game_qb_priors.parquet").exists():
        raise ValueError("Build corrected scalar inputs first")
    plan = {"development": [2020,2021,2022], "development_seeds": [0,1],
            "assessment": [2023,2024,2025], "assessment_seeds": [0,1,2],
            "variants": ["corrected_control","meanmax","attention"],
            "selection": "Choose player pooling by lowest development margin MAE; report both, plus corrected scalar-only control. Assess the selected player variant and control on all later years regardless of development gain.",
            "training": "Chronological previous-year early stopping on winner logloss; refit all prior years for selected epoch count; original loss; effective batch2048 with microbatch512; no 2026 outcomes fitted or selected",
            "assessment_is_reused_historical_benchmark": True,
            "deployment": "Retain full player checkpoints and score future games in a separate forecast file. Do not silently replace an established ensemble on an uncertain/regressing result.",
            "inputs": {p.name:sha(p) for p in inputs.iterdir() if p.is_file()},
            "player_manifest_sha256":sha(player_path/"manifest.json"),
            "code":{n:sha(Path(__file__).parent/n) for n in CODE}}
    path = output/"plan.json"
    if path.exists() and json.loads(path.read_text())!=plan:
        raise ValueError("Frozen player experiment changed; use a new experiment directory")
    save_json(path,plan)
    snap=output/"code_snapshot";snap.mkdir(exist_ok=True)
    for name in CODE:shutil.copy2(Path(__file__).parent/name,snap/name)
    return inputs


def load_inputs(inputs, player_path=PLAYERS):
    provenance=json.loads((inputs/"player_data_provenance.json").read_text())
    checks={"player_manifest_sha256":player_path/"manifest.json",
            "schedule_sha256":inputs/"snapshot_schedule.parquet",
            "corrected_scalars_sha256":inputs/"game_scalars.parquet",
            "corrected_qb_sha256":inputs/"game_qb_priors.parquet"}
    if any(sha(path)!=provenance[key] for key,path in checks.items()):
        raise ValueError("Player and scalar inputs are from different refreshes")
    base = build_tensors(data_dir=inputs,include_plays=False)
    base = attach(add_form(base,inputs,"recent5_matchup"),pd.read_parquet(inputs/"game_qb_priors.parquet"))
    base["player_store"] = PlayerStore(player_path,base["G_gid"])
    return base


def make(d,variant):
    if variant=="corrected_control":
        model=challenger(d,True);model.pre.hist.empty_is_zero=True
        return model
    return make_player_model(d,pooling=variant)


def run_variant(base, inputs, variant, years, seeds, device, output=OUTPUT):
    directory = output/"runs"/variant;directory.mkdir(parents=True,exist_ok=True)
    d = base.copy()
    if variant=="corrected_control":d.pop("player_store")
    meta = pd.read_parquet(inputs/"game_scalars.parquet")[["game_id","season","week_c","margin","total"]].merge(
        pd.read_parquet(inputs/"market.parquet"),on="game_id",how="left").set_index("game_id")
    frames = []
    for year in years:
        tr,va,te,refit=fold_masks(d,year,None)
        fitted,refitted=fit_consistent_inputs(d,tr),fit_consistent_inputs(d,refit)
        idx=np.where(te)[0];margins=[];totals=[];epochs=[]
        for seed in seeds:
            cache=directory/f"{year}_s{seed}.npz"
            if cache.exists():
                saved=np.load(cache)
                if not np.array_equal(saved["game_id"],d["G_gid"][idx]):raise ValueError("Prediction order changed")
                pm,pt,epoch=torch.tensor(saved["pm"]),torch.tensor(saved["pt"]),int(saved["epochs"])
            else:
                print(f"START {variant} forecast={year} seed={seed}",flush=True)
                torch.manual_seed(seed);np.random.seed(seed)
                model=make(fitted,variant).to(device)
                pretrain(model,fitted,device,train_indices=np.where(tr)[0],val_indices=np.where(va)[0],
                         micro_batch_size=512)
                epoch=model.pre.selected_epochs
                torch.manual_seed(seed);np.random.seed(seed)
                model=make(refitted,variant).to(device)
                pretrain(model,refitted,device,train_indices=np.where(refit)[0],val_indices=np.where(va)[0],
                         fixed_epochs=epoch,micro_batch_size=512)
                pm,pt=predict(model,refitted,idx,device)
                ck={"architecture":"players_v1" if variant!="corrected_control" else "consistent",
                    "pregame_only":True,"pooling":variant,"state_dict":{k:v.detach().cpu() for k,v in model.pre.state_dict().items()},
                    "mu":refitted["mu"],"sd":refitted["sd"],"scalar_cols":d["scalar_cols"],
                    "vocab":d["vocab"],"forecast_year":year,"seed":seed,"epochs":epoch,
                    "train_game_ids":d["G_gid"][refit],"train_max_season":int(d["G_season"][refit].max()),
                    "player_schema":base["player_store"].manifest,
                    "known_teams":np.unique(np.concatenate([refitted[f].numpy().ravel() for f in ["G_hidx","G_aidx","G_opp"]])),
                    "known_confs":np.unique(np.concatenate([refitted[f].numpy().ravel() for f in ["G_hconf","G_aconf"]]))}
                torch.save(ck,directory/f"{year}_s{seed}.pt")
                np.savez_compressed(cache,pm=pm.numpy(),pt=pt.numpy(),game_id=d["G_gid"][idx],epochs=epoch)
                del model
                if device.type=="mps":torch.mps.empty_cache()
            margins.append(pm);totals.append(pt);epochs.append(epoch)
        pm,pt=torch.stack(margins).mean(0),torch.stack(totals).mean(0)
        frame=pd.DataFrame(readouts(pm,pt),index=d["G_gid"][idx]).join(meta)
        frame["crps"]=discrete_crps(pm.clamp(min=1e-30).log(),d["G_margin"][idx],reduction="none").numpy()
        frame["fbs_fbs"]=d["G_fbs_fbs"][idx];frame.index.name="game_id";frame=frame.reset_index()
        frame.to_parquet(directory/f"{year}_{len(seeds)}seeds_predictions.parquet",index=False)
        save_json(directory/f"{year}_{len(seeds)}seeds_report.json",{"epochs":epochs,"metrics":summarize(frame)})
        print(f"DONE {variant} {year}: {summarize(frame)}",flush=True)
        frames.append(frame)
    return pd.concat(frames,ignore_index=True)


def compare(frames):
    return {"metrics":{n:summarize(f) for n,f in frames.items()},
            "paired_vs_corrected_control":{n:paired(frames["corrected_control"],f) for n,f in frames.items() if n!="corrected_control"},
            "early_weeks_1_3":{n:summarize(f[f.week_c.le(3)]) for n,f in frames.items()},
            "by_year":{n:{str(y):summarize(f[f.season.eq(y)]) for y in sorted(f.season.unique())} for n,f in frames.items()}}


def run(output=OUTPUT):
    inputs=prepare(output);base=load_inputs(inputs)
    torch.set_num_threads(4);dev=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    development={n:run_variant(base,inputs,n,[2020,2021,2022],range(2),dev,output) for n in ["corrected_control","meanmax","attention"]}
    selected=min(["meanmax","attention"],key=lambda n:summarize(development[n])["margin_mae"])
    report={"development":compare(development),"selected_pooling":selected,"production_modified":False}
    save_json(output/"report.json",report)
    assessment={n:run_variant(base,inputs,n,[2023,2024,2025],range(3),dev,output) for n in ["corrected_control",selected]}
    report["assessment"]=compare(assessment)
    old=MODELS/"experiments/combined_20260906/runs"
    for n in ["baseline","combined"]:
        historical=pd.concat([pd.read_parquet(old/n/f"{y}_3seeds_predictions.parquet") for y in [2023,2024,2025]],ignore_index=True)
        report[f"vs_original_{n}"]={"metrics":summarize(historical),"paired":paired(historical,assessment[selected])}
    save_json(output/"report.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,default=OUTPUT)
    args=ap.parse_args();run(args.output)
