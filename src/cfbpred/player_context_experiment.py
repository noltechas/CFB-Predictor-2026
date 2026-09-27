"""Reproducible full-history/context comparison and saved current candidate.

All variants use identical pinned raw inputs and chronological epoch selection.
The public context candidate is specified before assessment; no 2026 labels fit
or select it. This is a reused benchmark, not a new untouched test set.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from cfbpred.architecture_experiment import fold_masks
from cfbpred.architecture_research_net import make as make_control
from cfbpred.consistent_net import fit_consistent_inputs
from cfbpred.ensemble import readouts
from cfbpred.model import discrete_crps
from cfbpred.player_experiment import load_inputs
from cfbpred.player_full_history import FullCareerStore
from cfbpred.player_full_history_net import CONTEXT_VARIANTS, SCALAR_EXTRAS_BY_VARIANT, make
from cfbpred import train as train_module
from cfbpred.player_sequence_fast_inputs import FastSequenceStore
from cfbpred.player_sequence_train import fit,predict
from cfbpred.player_timeline import TimelineStore
from cfbpred.pregame_experiments import save_json,sha,summarize,paired


def load(source,variant):
    paths=json.loads(source.read_text());p={k:Path(v) for k,v in paths.items()}
    train_module.SCALAR_EXTRAS=SCALAR_EXTRAS_BY_VARIANT.get(variant,())
    d=load_inputs(p["inputs"],p["players"])
    if variant=="control":store=FastSequenceStore(p["players"],p["sequences"],d["G_gid"])
    elif variant in CONTEXT_VARIANTS:
        store=TimelineStore(p["players"],p["sequences"],p["full_history"],p["timeline"],d["G_gid"])
    else:store=FullCareerStore(p["players"],p["sequences"],p["full_history"],d["G_gid"])
    d["player_store"]=store
    return d,p


def fold(base,mask,variant):
    d=fit_consistent_inputs(base,mask)
    d["player_store"]=base["player_store"].for_fold(np.flatnonzero(mask),True,variant in ({"full_history_roles"}|CONTEXT_VARIANTS))
    return d


def network(d,variant):return make_control(d,"tabm16") if variant=="control" else make(d,variant)


def run(source,output,variant,year=2025,seed=0,threads=2,micro_batch=64,max_epochs=60,patience=12,epoch_source=None,fixed_epochs=None):
    torch.set_num_threads(threads)
    output.mkdir(parents=True,exist_ok=True)
    base,paths=load(source,variant)
    code_names=["player_full_history.py","player_full_history_net.py","player_timeline.py",
                "player_context_experiment.py","player_sequence_train.py","player_sequence_fast_inputs.py",
                "architecture_research_net.py","consistent_net.py","player_inputs.py","player_net.py",
                "player_sequence_net.py","player_sequence_inputs.py","train.py","raw_net.py","model.py"]
    plan=dict(variant=variant,year=year,seed=seed,max_epochs=max_epochs,patience=patience,
              micro_batch=micro_batch,source_sha256=sha(source),
              inputs={p.name:sha(p) for p in paths["inputs"].iterdir() if p.is_file()},
              stores={key:sha(paths[key]/"manifest.json") for key in ["players","sequences","full_history","timeline"] if key in paths},
              code={name:sha(Path(__file__).parent/name) for name in code_names},
              epoch_selection_year=year-1,refit_through_year=year-1,
              labels_2026_used=False,production_modified=False,
              assessment_is_reused_historical_benchmark=True)
    if epoch_source is not None:
        choice=json.loads(epoch_source.read_text())
        independent=json.loads((epoch_source.parent/"plan.json").read_text())
        if choice["variant"]!="control" or independent["epoch_selection_year"]>=year:
            raise ValueError("Fixed budget must come from an earlier clean control validation")
        if (epoch_source.parent/"INVALIDATED.json").exists():raise ValueError("Invalidated epoch source")
        plan["fixed_epochs"]=int(choice["epochs"])
        if plan["fixed_epochs"]<=0:raise ValueError("Invalid fixed epoch count")
        plan["epoch_source"]={"path":str(epoch_source.resolve()),"sha256":sha(epoch_source)}
        plan["epoch_selection_year"]=independent["epoch_selection_year"]
    elif fixed_epochs is not None:
        # Rolling backtests reuse one training budget in every fold. It was chosen once,
        # on 2024 validation; every arm shares it, so it cannot favour an arm.
        plan["fixed_epochs"]=int(fixed_epochs)
        plan["epoch_source"]={"fixed":int(fixed_epochs),"reason":"shared backtest budget from the 2025 clean control"}
    plan["scalar_extras"]=list(SCALAR_EXTRAS_BY_VARIANT.get(variant,()))
    plan_path=output/"plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError("Frozen experiment changed")
    save_json(plan_path,plan)
    snapshot=output/"code_snapshot";snapshot.mkdir(exist_ok=True)
    for name,digest in plan["code"].items():
        target=snapshot/name
        if target.exists() and sha(target)!=digest:raise ValueError("Frozen code snapshot differs")
        if not target.exists():shutil.copy2(Path(__file__).parent/name,target)
    tr,va,te,refit=fold_masks(base,year,None)
    dfit,drefit=fold(base,tr,variant),fold(base,refit,variant)
    checkpoint=output/"member.pt"
    started=time.monotonic()
    if not checkpoint.exists():
        print(f"START {variant} forecast={year} seed={seed}",flush=True)
        if epoch_source is None and fixed_epochs is None:
            torch.manual_seed(seed);np.random.seed(seed);model=network(dfit,variant)
            fit(model,dfit,torch.device("cpu"),np.flatnonzero(tr),np.flatnonzero(va),
                max_epochs=max_epochs,patience=patience,micro_batch_size=micro_batch,log_path=output/"selection.json")
            epochs=model.pre.selected_epochs;del model
        else:
            epochs=plan["fixed_epochs"]
            save_json(output/"epoch_selection.json",dict(epochs=epochs,source=plan["epoch_source"],
                       reason="Same training budget independently selected by the clean control; no revised article used for selection"))
        torch.manual_seed(seed);np.random.seed(seed);model=network(drefit,variant)
        fit(model,drefit,torch.device("cpu"),np.flatnonzero(refit),np.flatnonzero(va),fixed_epochs=epochs,
            micro_batch_size=micro_batch,log_path=output/"refit.json")
        ck=dict(architecture="player_full_context_v1",variant=variant,pregame_only=True,
                state_dict={k:v.detach().cpu() for k,v in model.pre.state_dict().items()},
                mu=drefit["mu"],sd=drefit["sd"],vocab=base["vocab"],scalar_cols=base["scalar_cols"],
                known_sequence_teams=drefit["player_store"].known_teams,
                known_teams=np.unique(np.concatenate([drefit[k].numpy().ravel() for k in ["G_hidx","G_aidx","G_opp"]])),
                known_confs=np.unique(np.concatenate([drefit[k].numpy().ravel() for k in ["G_hconf","G_aconf"]])),
                forecast_year=year,seed=seed,epochs=epochs,train_game_ids=base["G_gid"][refit],
                train_max_season=int(base["G_season"][refit].max()),player_schema=base["player_store"].manifest,
                scalar_extras=list(SCALAR_EXTRAS_BY_VARIANT.get(variant,())),
                seconds=time.monotonic()-started,plan_sha256=sha(plan_path))
        torch.save(ck,checkpoint);del model
    # Reload the saved bundle for scoring, to test the actual forecast path.
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if ck["plan_sha256"]!=sha(plan_path) or ck["train_max_season"]>=year:raise ValueError("Invalid checkpoint lineage")
    for k in ["mu","sd"]:
        if not np.array_equal(ck[k],drefit[k]):raise ValueError("Normalization differs")
    if not np.array_equal(ck["known_sequence_teams"],drefit["player_store"].known_teams):raise ValueError("Player vocabulary differs")
    model=network(drefit,variant);model.pre.load_state_dict(ck["state_dict"])
    schedule=pd.read_parquet(paths["inputs"]/"snapshot_schedule.parquet").set_index("game_id")
    now=pd.Timestamp.now(tz="UTC")
    target=te if year<2026 else (base["G_season"]==year)&pd.to_datetime(schedule.loc[base["G_gid"],"kick"],utc=True).gt(now).to_numpy()
    idx=np.flatnonzero(target)
    pm,pt=predict(model,drefit,idx,torch.device("cpu"))
    for p in [pm,pt]:
        if not torch.isfinite(p).all() or (p<0).any() or not torch.allclose(p.sum(-1),torch.ones(len(idx)),atol=1e-5):
            raise ValueError("Invalid forecast probabilities")
    meta=pd.read_parquet(paths["inputs"]/"game_scalars.parquet")[["game_id","season","week_c","margin","total"]].set_index("game_id")
    frame=pd.DataFrame(readouts(pm,pt),index=base["G_gid"][idx]).join(meta)
    frame["fbs_fbs"]=base["G_fbs_fbs"][idx]
    frame["crps"]=discrete_crps(pm.clamp(min=1e-30).log(),base["G_margin"][idx],reduction="none").numpy() if year<2026 else np.nan
    frame=frame.join(schedule[["home_team","away_team","kick"]]);frame.index.name="game_id"
    frame=frame.reset_index()
    # Lines are evaluation-only and never passed into tensors/network.
    market=Path(__file__).resolve().parents[2]/"models/experiments/players_20260906_v3/inputs/market.parquet"
    if year<2026 and market.exists():frame=frame.merge(pd.read_parquet(market),on="game_id",how="left",validate="one_to_one")
    frame.to_parquet(output/"predictions.parquet",index=False)
    np.savez_compressed(output/"distributions.npz",pm=pm.numpy(),pt=pt.numpy(),game_id=base["G_gid"][idx])
    report=dict(variant=variant,year=year,seed=seed,epochs=ck["epochs"],games=len(frame),
                seconds=ck["seconds"],scored_from_reloaded_checkpoint=True,
                all_future_targets_scored=year==2026,forecast_cutoff=now.isoformat(),
                metrics=summarize(frame) if year<2026 else None,production_modified=False)
    save_json(output/"report.json",report)
    save_json(output/"bundle.json",dict(architecture="player_full_context_v1",variant=variant,
                source=str(source.resolve()),member="member.pt",member_sha256=sha(checkpoint),
                plan_sha256=sha(plan_path),predictions_sha256=sha(output/"predictions.parquet"),
                forecast_year=year,pregame_only=True))
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--variant",choices=["control","full_history","full_history_roles",*sorted(CONTEXT_VARIANTS)],required=True)
    p.add_argument("--epoch-source",type=Path)
    for name,default in [("year",2025),("seed",0),("threads",2),("micro-batch",64),("max-epochs",60),("patience",12)]:
        p.add_argument("--"+name,type=int,default=default)
    p.add_argument("--fixed-epochs",type=int)
    a=p.parse_args();run(a.source,a.output,a.variant,a.year,a.seed,a.threads,a.micro_batch,a.max_epochs,a.patience,a.epoch_source,a.fixed_epochs)
