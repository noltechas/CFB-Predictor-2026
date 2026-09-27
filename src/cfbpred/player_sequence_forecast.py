"""Train and score the retained game-context player model for 2026."""
from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cfbpred.ensemble import readouts
from cfbpred.player_sequence_experiment import INPUTS,PLAYERS,SEQUENCES,ROOT,fold_data,load
from cfbpred.player_sequence_fast_inputs import FastSequenceStore
from cfbpred.player_sequence_net import make
from cfbpred.player_sequence_train import fit,predict
from cfbpred.pregame_experiments import save_json,sha

OUTPUT=ROOT/"models/experiments/player_sequences_20260907_v2/current_2026_context_v2"
CODE=["player_sequence_inputs.py","player_sequence_fast_inputs.py","player_sequence_net.py",
      "player_sequence_train.py","player_sequence_experiment.py","player_sequence_forecast.py",
      "player_inputs.py","player_net.py","train.py","raw_net.py","consistent_net.py"]


def initialize(threads):
    global BASE
    torch.set_num_threads(threads);BASE=load();BASE["player_store"].__class__=FastSequenceStore


def train_member(seed,directory,forecast_year=2026,micro_batch_size=512):
    base=BASE
    tr=base["G_labeled"]&(base["G_season"]<forecast_year-1)
    va=base["G_labeled"]&(base["G_season"]==forecast_year-1)&base["G_fbs_inv"]
    refit=base["G_labeled"]&(base["G_season"]<forecast_year)
    fitted,refitted=fold_data(base,tr,"context"),fold_data(base,refit,"context")
    torch.manual_seed(seed);np.random.seed(seed);model=make(fitted,"context")
    fit(model,fitted,torch.device("cpu"),np.where(tr)[0],np.where(va)[0],micro_batch_size=micro_batch_size,
        log_path=directory/f"s{seed}_selection.json")
    epochs=model.pre.selected_epochs;del model
    torch.manual_seed(seed);np.random.seed(seed);model=make(refitted,"context")
    fit(model,refitted,torch.device("cpu"),np.where(refit)[0],np.where(va)[0],fixed_epochs=epochs,
        micro_batch_size=micro_batch_size,log_path=directory/f"s{seed}_refit.json")
    ck={"architecture":"player_sequence_v1","variant":"context","pregame_only":True,
        "state_dict":{k:v.detach().cpu() for k,v in model.pre.state_dict().items()},
        "mu":refitted["mu"],"sd":refitted["sd"],"scalar_cols":base["scalar_cols"],"vocab":base["vocab"],
        "known_sequence_teams":refitted["player_store"].known_teams,
        "known_teams":np.unique(np.concatenate([refitted[f].numpy().ravel() for f in ["G_hidx","G_aidx","G_opp"]])),
        "known_confs":np.unique(np.concatenate([refitted[f].numpy().ravel() for f in ["G_hconf","G_aconf"]])),
        "forecast_year":forecast_year,"seed":seed,"epochs":epochs,"train_game_ids":base["G_gid"][refit],
        "train_max_season":int(base["G_season"][refit].max()),"player_schema":base["player_store"].manifest,
        "sequence_schema":base["player_store"].sequence_manifest}
    path=directory/f"s{seed}.pt";torch.save(ck,path)
    print(f"Current context member seed={seed} epochs={epochs}",flush=True)
    return path.name


def train(directory=OUTPUT,seeds=3,workers=3,threads=2,micro_batch_size=512):
    directory.mkdir(parents=True,exist_ok=True)
    plan={"variant":"context","forecast_year":2026,"seeds":list(range(seeds)),"workers":workers,
          "threads_per_worker":threads,"micro_batch":micro_batch_size,"selection_year":2025,
          "refit_max_year":2025,"2026_labels_used":False,
          "inputs":{p.name:sha(p) for p in INPUTS.iterdir() if p.is_file()},
          "player_manifest_sha256":sha(PLAYERS/"manifest.json"),"sequence_manifest_sha256":sha(SEQUENCES/"manifest.json"),
          "code":{name:sha(Path(__file__).parent/name) for name in CODE}}
    if (directory/"plan.json").exists() and json.loads((directory/"plan.json").read_text())!=plan:
        raise ValueError("Current candidate plan changed; use a new directory")
    save_json(directory/"plan.json",plan)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize,initargs=(threads,)) as pool:
        jobs={pool.submit(train_member,seed,directory,2026,micro_batch_size):seed for seed in range(seeds) if not (directory/f"s{seed}.pt").exists()}
        for job in as_completed(jobs):job.result()
    members=[f"s{seed}.pt" for seed in range(seeds)]
    if not all((directory/name).exists() for name in members):raise ValueError("Incomplete current candidate")
    save_json(directory/"bundle.json",{"architecture":"player_sequence_v1","variant":"context","pregame_only":True,
              "forecast_year":2026,"members":members,"member_sha256":{name:sha(directory/name) for name in members},
              "plan_sha256":sha(directory/"plan.json")})


def score(directory=OUTPUT,output=None,cutoff=None):
    output=directory/"future_predictions.parquet" if output is None else output
    plan=json.loads((directory/"plan.json").read_text());bundle=json.loads((directory/"bundle.json").read_text())
    if sha(directory/"plan.json")!=bundle["plan_sha256"]:raise ValueError("Bundle plan changed")
    base=load();base["player_store"].__class__=FastSequenceStore
    refit=base["G_labeled"]&(base["G_season"]<bundle["forecast_year"])
    d=fold_data(base,refit,"context")
    schedule=pd.read_parquet(INPUTS/"snapshot_schedule.parquet").set_index("game_id")
    cutoff=pd.Timestamp.now(tz="UTC") if cutoff is None else pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:raise ValueError("Cutoff needs timezone")
    kick=pd.to_datetime(schedule.loc[base["G_gid"],"kick"],utc=True)
    idx=np.where((base["G_season"]==2026)&(kick>cutoff).to_numpy())[0]
    sums=[None,None]
    for name in bundle["members"]:
        path=directory/name
        if sha(path)!=bundle["member_sha256"][name]:raise ValueError("Current checkpoint changed")
        ck=torch.load(path,map_location="cpu",weights_only=False)
        if ck["train_max_season"]>=2026 or not ck["pregame_only"] or ck["variant"]!="context":raise ValueError("Invalid current checkpoint")
        if not np.array_equal(ck["mu"],d["mu"]) or not np.array_equal(ck["sd"],d["sd"]):raise ValueError("Current normalization differs")
        if not np.array_equal(ck["known_sequence_teams"],d["player_store"].known_teams):raise ValueError("History vocabulary differs")
        model=make(d,"context");model.pre.load_state_dict(ck["state_dict"])
        pm,pt=predict(model,d,idx,torch.device("cpu"));del model
        sums=[pm if sums[0] is None else sums[0]+pm,pt if sums[1] is None else sums[1]+pt]
    pm,pt=(x/len(bundle["members"]) for x in sums)
    for probabilities in [pm,pt]:
        if not torch.isfinite(probabilities).all() or (probabilities<0).any() or not torch.allclose(probabilities.sum(-1),torch.ones(len(idx)),atol=1e-5):
            raise ValueError("Invalid current forecast distribution")
    r=readouts(pm,pt)
    frame=pd.DataFrame({"game_id":base["G_gid"][idx],"season":base["G_season"][idx],"home_win_prob":r["wp"],
                        "pred_margin":r["em"],"pred_total":r["et"],"margin_pmf":pm.numpy().tolist(),
                        "total_pmf":pt.numpy().tolist(),"snapshot_at":cutoff,"model":"player_sequence_context_v1","status":"retained_candidate"})
    frame=frame.merge(schedule[["home_team","away_team","kick"]],left_on="game_id",right_index=True,validate="one_to_one")
    output.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(output,index=False)
    save_json(output.with_suffix(".json"),{"future_games":len(frame),"cutoff":cutoff.isoformat(),"members":len(bundle["members"]),
              "bundle_sha256":sha(directory/"bundle.json"),"forecast_sha256":sha(output),
              "all_future_targets_scored":True,"production_modified":False})
    print(f"Scored {len(frame):,} future games with retained context candidate",flush=True)


def audit(directory=OUTPUT,forecasts=None):
    forecasts=directory/"future_predictions.parquet" if forecasts is None else forecasts
    plan=json.loads((directory/"plan.json").read_text());bundle=json.loads((directory/"bundle.json").read_text())
    for name,digest in plan["code"].items():
        if sha(Path(__file__).parent/name)!=digest:raise ValueError(f"Current candidate code changed: {name}")
    for name,digest in plan["inputs"].items():
        if sha(INPUTS/name)!=digest:raise ValueError(f"Current candidate input changed: {name}")
    meta=json.loads(forecasts.with_suffix(".json").read_text());frame=pd.read_parquet(forecasts)
    if sha(directory/"bundle.json")!=meta["bundle_sha256"] or sha(forecasts)!=meta["forecast_sha256"]:
        raise ValueError("Current forecast lineage differs")
    base=load();schedule=pd.read_parquet(INPUTS/"snapshot_schedule.parquet").set_index("game_id")
    cutoff=pd.Timestamp(meta["cutoff"]);kick=pd.to_datetime(schedule.loc[base["G_gid"],"kick"],utc=True)
    expected=base["G_gid"][(base["G_season"]==2026)&(kick>cutoff).to_numpy()]
    if frame.game_id.duplicated().any() or set(frame.game_id)!=set(expected):raise ValueError("Future forecast coverage differs")
    pm=torch.tensor(np.stack(frame.margin_pmf));pt=torch.tensor(np.stack(frame.total_pmf))
    for values in [pm,pt]:
        if not torch.isfinite(values).all() or (values<0).any() or not torch.allclose(values.sum(-1),torch.ones(len(values)),atol=1e-5):
            raise ValueError("Invalid saved current probabilities")
    values=readouts(pm,pt)
    for col,key in [("home_win_prob","wp"),("pred_margin","em"),("pred_total","et")]:
        if not np.allclose(frame[col],values[key],atol=1e-5):raise ValueError("Current readout differs from PMF")
    roster=pd.read_parquet(PLAYERS/"game_rosters.parquet").set_index(["game_id","side"])
    target=roster.loc[pd.MultiIndex.from_product([expected,["home","away"]])].reset_index()
    target["division"]=[schedule.loc[row.game_id,row.side+"_division"] for row in target.itertuples()]
    if target.loc[target.division.eq("fbs"),"players"].eq(0).any():raise ValueError("Future FBS roster missing")
    if (target.last_history_kick.notna()&(pd.to_datetime(target.last_history_kick,utc=True)>=pd.to_datetime(target.kick,utc=True))).any():
        raise ValueError("Future player snapshot is not pregame")
    members=[]
    for name in bundle["members"]:
        path=directory/name
        if sha(path)!=bundle["member_sha256"][name]:raise ValueError("Current member changed")
        ck=torch.load(path,map_location="cpu",weights_only=False)
        seasons=schedule.loc[ck["train_game_ids"],"season"]
        if seasons.ge(2026).any() or int(seasons.max())!=2025:raise ValueError("2026 outcome entered current member")
        members.append({"member":name,"seed":ck["seed"],"epochs":ck["epochs"],"training_games":len(seasons),"latest_training_season":int(seasons.max())})
    report={"passed":True,"future_games":len(frame),"future_fbs_sides":int(target.division.eq("fbs").sum()),
            "missing_future_fbs_rosters":0,"strict_player_history_before_kickoff":True,"no_2026_outcomes_used":True,
            "probability_distributions_valid":True,"members":members,"production_modified":False}
    save_json(directory/"forecast_audit.json",report);print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["train","score","audit"]);p.add_argument("--directory",type=Path,default=OUTPUT)
    p.add_argument("--seeds",type=int,default=3);p.add_argument("--workers",type=int,default=3);p.add_argument("--threads",type=int,default=2)
    p.add_argument("--micro-batch",type=int,default=512);p.add_argument("--output",type=Path);p.add_argument("--cutoff")
    a=p.parse_args()
    if a.action=="train":train(a.directory,a.seeds,a.workers,a.threads,a.micro_batch)
    elif a.action=="score":score(a.directory,a.output,a.cutoff)
    else:audit(a.directory,a.output)
