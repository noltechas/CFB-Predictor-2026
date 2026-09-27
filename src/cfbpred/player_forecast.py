"""Train and score the player-aware PRE-GAME model, with checkpoint contracts.

Output is a separate forecast artifact until the historical comparison earns
promotion. Started forecasts and the existing live model are never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cfbpred.consistent_net import fit_consistent_inputs
from cfbpred.ensemble import readouts
from cfbpred.player_experiment import load_inputs, make, OUTPUT
from cfbpred.player_inputs import OUTPUT as PLAYERS
from cfbpred.pregame_experiments import save_json, sha
from cfbpred.raw_net import RawTwoTower
from cfbpred.train import game_batch, pretrain


SCHEMA_KEYS = ("version","static_cols","history_cols","positions","history_windows")


def validate_contract(checkpoint,d):
    if checkpoint.get("architecture") not in {"players_v1","corrected_control_v1"} or not checkpoint.get("pregame_only"):
        raise ValueError("Not a supported player pregame checkpoint")
    if checkpoint["scalar_cols"]!=d["scalar_cols"]:
        raise ValueError("Checkpoint scalar schema differs")
    if any(checkpoint["player_schema"][k]!=d["player_store"].manifest[k] for k in SCHEMA_KEYS):
        raise ValueError("Checkpoint individual-player schema differs")
    if checkpoint["train_max_season"]>=checkpoint["forecast_year"]:
        raise ValueError("Checkpoint trained on its forecast season")
    mu,sd=np.asarray(checkpoint["mu"]),np.asarray(checkpoint["sd"])
    if mu.shape!=(len(d["scalar_cols"]),) or sd.shape!=mu.shape:
        raise ValueError("Checkpoint normalization shape differs from scalar schema")
    if not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd<=0).any():
        raise ValueError("Checkpoint normalization must be finite with positive scales")


class PregameMember(nn.Module):
    def __init__(self,pre):
        super().__init__();self.pre=pre
    def pregame(self,batch):
        return RawTwoTower.pregame(self,batch)
    def forward(self,*args,**kwargs):
        raise RuntimeError("Player checkpoint is pregame-only; live conditioning has not been retrained")


def checkpoint_inputs(d,ck):
    validate_contract(ck,d)
    out=d.copy()
    # Use the saved normalization and identity vocabulary, including when the
    # refreshed store contains new teams. New identities map to unknown zero.
    out["G_scalars"]=torch.tensor(np.nan_to_num((d["G_raw_scalars"]-ck["mu"])/ck["sd"],nan=0,posinf=0,neginf=0),dtype=torch.float32)
    for key,fields,known_key in [("team",["G_hidx","G_aidx","G_opp"],"known_teams"),
                                 ("conf",["G_hconf","G_aconf"],"known_confs")]:
        current=d["vocab"][key];saved=ck["vocab"][key]
        mapping=np.zeros(max(current.values(),default=0)+1,dtype=np.int64)
        known=set(ck[known_key])
        for name,index in current.items():
            old=saved.get(name,0);mapping[index]=old if old in known else 0
        for field in fields:out[field]=torch.tensor(mapping[d[field].numpy()],dtype=torch.long)
    out["vocab"]=ck["vocab"]
    return out


def fit_current(inputs,players,directory,pooling,seeds=3,forecast_year=2026,device="auto",threads=4):
    directory.mkdir(parents=True,exist_ok=True)
    base=load_inputs(inputs,players)
    architecture="corrected_control_v1" if pooling=="corrected_control" else "players_v1"
    tr=base["G_labeled"]&(base["G_season"]<forecast_year-1)
    va=base["G_labeled"]&(base["G_season"]==forecast_year-1)&base["G_fbs_inv"]
    refit=base["G_labeled"]&(base["G_season"]<forecast_year)
    fitted,refitted=fit_consistent_inputs(base,tr),fit_consistent_inputs(base,refit)
    if pooling=="corrected_control":
        fitted.pop("player_store");refitted.pop("player_store")
    dev=torch.device("mps" if device=="auto" and torch.backends.mps.is_available() else "cpu" if device=="auto" else device)
    plan={"forecast_year":forecast_year,"seeds":list(range(seeds)),"pooling":pooling,"device":str(dev),"threads":threads,
          "train_max_season":forecast_year-1,"inputs":{p.name:sha(p) for p in inputs.iterdir() if p.is_file()},
          "player_manifest_sha256":sha(players/"manifest.json"),"training_code_sha256":sha(__file__)}
    path=directory/"plan.json"
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError("Current model training plan changed")
    save_json(path,plan)
    torch.set_num_threads(threads)
    members=[]
    for seed in range(seeds):
        path=directory/f"s{seed}.pt";members.append(path.name)
        if path.exists():continue
        print(f"Current player model seed={seed}, training through {forecast_year-1}",flush=True)
        torch.manual_seed(seed);np.random.seed(seed)
        model=make(fitted,pooling).to(dev)
        pretrain(model,fitted,dev,train_indices=np.where(tr)[0],val_indices=np.where(va)[0],micro_batch_size=512)
        epochs=model.pre.selected_epochs
        torch.manual_seed(seed);np.random.seed(seed)
        model=make(refitted,pooling).to(dev)
        pretrain(model,refitted,dev,train_indices=np.where(refit)[0],val_indices=np.where(va)[0],fixed_epochs=epochs,micro_batch_size=512)
        ck={"architecture":architecture,"pregame_only":True,"pooling":pooling,
            "state_dict":{k:v.detach().cpu() for k,v in model.pre.state_dict().items()},
            "mu":refitted["mu"],"sd":refitted["sd"],"scalar_cols":base["scalar_cols"],"vocab":base["vocab"],
            "forecast_year":forecast_year,"train_max_season":int(base["G_season"][refit].max()),
            "train_game_ids":base["G_gid"][refit],"seed":seed,"epochs":epochs,"player_schema":base["player_store"].manifest,
            "known_teams":np.unique(np.concatenate([refitted[f].numpy()[refit].ravel() for f in ["G_hidx","G_aidx","G_opp"]])),
            "known_confs":np.unique(np.concatenate([refitted[f].numpy()[refit].ravel() for f in ["G_hconf","G_aconf"]]))}
        torch.save(ck,path);del model
        if dev.type=="mps":torch.mps.empty_cache()
    save_json(directory/"bundle.json",{"architecture":architecture,"pregame_only":True,"members":members,
                                      "forecast_year":forecast_year,"member_sha256":{n:sha(directory/n) for n in members}})


def score(inputs,players,bundle_path,output,now=None):
    base=load_inputs(inputs,players);bundle=json.loads(bundle_path.read_text())
    if bundle.get("architecture") not in {"players_v1","corrected_control_v1"} or not bundle.get("pregame_only") or not bundle.get("members"):
        raise ValueError("Not a complete player pregame bundle")
    schedule=pd.read_parquet(inputs/"snapshot_schedule.parquet").set_index("game_id")
    now=pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:raise ValueError("Forecast cutoff must include a timezone")
    kickoff=pd.to_datetime(schedule.loc[base["G_gid"],"kick"],utc=True)
    idx=np.where((base["G_season"]==bundle["forecast_year"])&(kickoff>now).to_numpy())[0]
    if not len(idx):raise ValueError("No future games at this cutoff")
    dev=torch.device("mps" if torch.backends.mps.is_available() else "cpu");torch.set_num_threads(4)
    sum_m=sum_t=None
    for name in bundle["members"]:
        path=bundle_path.parent/name
        if sha(path)!=bundle["member_sha256"][name]:raise ValueError("Checkpoint changed after bundle creation")
        ck=torch.load(path,map_location="cpu",weights_only=False)
        if ck["forecast_year"]!=bundle["forecast_year"]:
            raise ValueError("Checkpoint and bundle forecast seasons differ")
        if ck["architecture"]!=bundle["architecture"]:
            raise ValueError("Checkpoint and bundle architectures differ")
        d=checkpoint_inputs(base,ck)
        network=make(d,ck["pooling"])
        network.pre.load_state_dict(ck["state_dict"])
        member=PregameMember(network.pre).to(dev).eval()
        pms=[];pts=[]
        with torch.no_grad():
            for start in range(0,len(idx),128):
                _,pm,pt=member.pregame(game_batch(d,torch.tensor(idx[start:start+128]),dev))
                pms.append(pm.softmax(-1).cpu());pts.append(pt.softmax(-1).cpu())
        pm,pt=torch.cat(pms),torch.cat(pts)
        sum_m=pm if sum_m is None else sum_m+pm;sum_t=pt if sum_t is None else sum_t+pt
        del member,network
        if dev.type=="mps":torch.mps.empty_cache()
    pm,pt=sum_m/len(bundle["members"]),sum_t/len(bundle["members"])
    for probabilities in [pm,pt]:
        if not torch.isfinite(probabilities).all() or (probabilities<0).any() or not torch.allclose(probabilities.sum(-1),torch.ones(len(idx)),atol=1e-5):
            raise ValueError("Player forecast contains an invalid probability distribution")
    r=readouts(pm,pt)
    out=pd.DataFrame({"game_id":base["G_gid"][idx],"season":base["G_season"][idx],"home_win_prob":r["wp"],
                      "pred_margin":r["em"],"pred_total":r["et"],"margin_pmf":pm.numpy().tolist(),
                      "total_pmf":pt.numpy().tolist(),"snapshot_at":now,"model":bundle["architecture"], "status":"candidate"})
    out=out.merge(schedule[["home_team","away_team","kick"]],left_on="game_id",right_index=True,validate="one_to_one")
    output.parent.mkdir(parents=True,exist_ok=True);out.to_parquet(output,index=False)
    save_json(output.with_suffix(".json"),{"future_games":len(out),"cutoff":now.isoformat(),"members":len(bundle["members"]),
              "bundle_sha256":sha(bundle_path),"player_manifest_sha256":sha(players/"manifest.json"),
              "empty_roster_sides":int(base["player_store"].counts[idx].reshape(-1).tolist().count(0)),
              "all_target_games_have_explicit_player_inputs":True,"production_modified":False})
    print(f"Scored all {len(out):,} future games -> {output}",flush=True)
    return out


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("action",choices=["train","score"])
    ap.add_argument("--inputs",type=Path,default=OUTPUT/"inputs");ap.add_argument("--players",type=Path,default=PLAYERS)
    ap.add_argument("--directory",type=Path,default=OUTPUT/"current_2026");ap.add_argument("--pooling",choices=["meanmax","attention","corrected_control"])
    ap.add_argument("--seeds",type=int,default=3);ap.add_argument("--output",type=Path,default=OUTPUT/"future_predictions.parquet")
    ap.add_argument("--device",choices=["auto","cpu","mps"],default="auto");ap.add_argument("--threads",type=int,default=4)
    a=ap.parse_args()
    if a.action=="train":
        selected=a.pooling or json.loads((a.inputs.parent/"report.json").read_text())["selected_pooling"]
        fit_current(a.inputs,a.players,a.directory,selected,a.seeds,device=a.device,threads=a.threads)
    else:score(a.inputs,a.players,a.directory/"bundle.json",a.output)
