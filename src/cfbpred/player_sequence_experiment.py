"""Frozen separate-and-combined tests of richer raw player modeling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing

import numpy as np
import pandas as pd
import torch

from cfbpred.architecture_experiment import fold_masks
from cfbpred.consistent_net import fit_consistent_inputs
from cfbpred.ensemble import readouts
from cfbpred.model import discrete_crps
from cfbpred.player_experiment import load_inputs
from cfbpred.player_sequence_inputs import SequenceStore
from cfbpred.player_sequence_net import VARIANTS,make
from cfbpred.player_sequence_train import fit,predict
from cfbpred.pregame_experiments import paired,save_json,sha,summarize

ROOT=Path(__file__).resolve().parents[2]
INPUTS=ROOT/"models/experiments/players_20260906_v3/inputs"
PLAYERS=ROOT/"data/parquet/player_inputs_v3"
SEQUENCES=ROOT/"data/parquet/player_sequences_v1"
OUTPUT=ROOT/"models/experiments/player_sequences_20260907_v1"
CODE=["player_sequence_inputs.py","player_sequence_net.py","player_sequence_train.py","player_sequence_experiment.py",
      "player_inputs.py","player_net.py","player_experiment.py","train.py","raw_net.py","consistent_net.py",
      "architecture_experiment.py","backtest_pregame.py","pregame_experiments.py","ensemble.py","model.py","qb_roster_prior.py"]


def load():
    base=load_inputs(INPUTS,PLAYERS)
    base["player_store"]=SequenceStore(PLAYERS,SEQUENCES,base["G_gid"])
    manifest=base["player_store"].sequence_manifest
    if manifest["schedule_sha256"]!=sha(INPUTS/"snapshot_schedule.parquet") or manifest["vocab_sha256"]!=sha(INPUTS/"team_vocab.json"):
        raise ValueError("Sequence/game inputs are from different snapshots")
    return base


def prepare(output,device,micro_batch_size,workers=3,threads=2):
    output.mkdir(parents=True,exist_ok=True)
    plan={"variants":{n:dict(context=c,contribution=r,auxiliary=a) for n,(c,r,a) in VARIANTS.items()},
          "development_years":[2020,2021,2022],"development_seeds":[0,1],
          "assessment_years":[2023,2024,2025],"assessment_seeds":[0,1,2],
          "selection":"Lowest pooled development FBS-vs-FBS margin MAE; save before assessment. Assess ALL five variants regardless of development result; report probability/total metrics and uncertainty without a minimum gain cutoff.",
          "training":{"max_epochs":200,"patience":20,"lr":.002,"weight_decay":.03,"effective_batch":2048,
                      "micro_batch":micro_batch_size,"auxiliary_weight":1.,"auxiliary_loss":"Masked smooth L1 on signed log1p(raw stat)/5; equal observed stat weight per game; beta=.1",
                      "selection":"Previous-year FBS-involved winner logloss; then refit all prior seasons for selected epochs","fcs_extra_sampling":.25},
          "assessment_is_reused_historical_benchmark":True,"2026_labels_used":False,
          "sequence":"Last eight observed player box games with program, opponent, venue, prior scores, age, same-season/program and recency rank; existing full-career summaries retained",
          "contribution":"Learned position-specific weighting from last observation, recent three and previous three reported raw counts and missingness; supplements mean/max roster pools. Does not claim full snap or injury coverage.",
          "control":"Same raw team/player inputs, architecture family, trainer and seed counts; no contextual game encoder, contribution weighting or auxiliary loss. Rerun under this protocol.",
          "production_modified":False,"device":str(device),"workers":workers,"threads_per_worker":threads,
          "inputs":{p.name:sha(p) for p in INPUTS.iterdir() if p.is_file()},
          "player_manifest_sha256":sha(PLAYERS/"manifest.json"),"sequence_manifest_sha256":sha(SEQUENCES/"manifest.json"),
          "code":{n:sha(Path(__file__).parent/n) for n in CODE}}
    path=output/"plan.json"
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError("Frozen experiment changed; use a new output directory")
    save_json(path,plan);snapshot=output/"code_snapshot";snapshot.mkdir(exist_ok=True)
    for name in CODE:shutil.copy2(Path(__file__).parent/name,snapshot/name)


def fold_data(base,mask,variant):
    d=fit_consistent_inputs(base,mask)
    context,role,_=VARIANTS[variant]
    d["player_store"]=base["player_store"].for_fold(np.where(mask)[0],context,role)
    return d


def run_variant(base,variant,years,seeds,device,output,micro_batch_size):
    directory=output/"runs"/variant;directory.mkdir(parents=True,exist_ok=True)
    meta=pd.read_parquet(INPUTS/"game_scalars.parquet")[["game_id","season","week_c","margin","total"]].merge(
        pd.read_parquet(INPUTS/"market.parquet"),on="game_id",how="left").set_index("game_id")
    frames=[]
    for year in years:
        tr,va,te,refit=fold_masks(base,year,None)
        fitted=fold_data(base,tr,variant);refitted=fold_data(base,refit,variant)
        idx=np.where(te)[0];margins=[];totals=[];epochs=[]
        for seed in seeds:
            cache=directory/f"{year}_s{seed}.npz"
            if cache.exists():
                saved=np.load(cache)
                if not np.array_equal(saved["game_id"],base["G_gid"][idx]):raise ValueError("Cached target order differs")
                pm,pt,epoch=torch.tensor(saved["pm"]),torch.tensor(saved["pt"]),int(saved["epochs"])
            else:
                start=time.monotonic();print(f"START {variant} year={year} seed={seed}",flush=True)
                save_json(output/f"status_{variant}.json",{"variant":variant,"year":year,"seed":seed,"stage":"select_epochs"})
                torch.manual_seed(seed);np.random.seed(seed);model=make(fitted,variant).to(device)
                fit(model,fitted,device,np.where(tr)[0],np.where(va)[0],micro_batch_size=micro_batch_size,
                    log_path=directory/f"{year}_s{seed}_selection.json")
                epoch=model.pre.selected_epochs
                del model
                if device.type=="mps":torch.mps.empty_cache()
                torch.manual_seed(seed);np.random.seed(seed);model=make(refitted,variant).to(device)
                save_json(output/f"status_{variant}.json",{"variant":variant,"year":year,"seed":seed,"stage":"refit","epochs":epoch})
                fit(model,refitted,device,np.where(refit)[0],np.where(va)[0],fixed_epochs=epoch,micro_batch_size=micro_batch_size,
                    log_path=directory/f"{year}_s{seed}_refit.json")
                pm,pt=predict(model,refitted,idx,device)
                ck={"architecture":"player_sequence_v1","pregame_only":True,"variant":variant,
                    "state_dict":{k:v.detach().cpu() for k,v in model.pre.state_dict().items()},
                    "mu":refitted["mu"],"sd":refitted["sd"],"scalar_cols":base["scalar_cols"],"vocab":base["vocab"],
                    "known_sequence_teams":refitted["player_store"].known_teams,
                    "known_teams":np.unique(np.concatenate([refitted[f].numpy().ravel() for f in ["G_hidx","G_aidx","G_opp"]])),
                    "known_confs":np.unique(np.concatenate([refitted[f].numpy().ravel() for f in ["G_hconf","G_aconf"]])),
                    "forecast_year":year,"seed":seed,"epochs":epoch,"train_game_ids":base["G_gid"][refit],
                    "train_max_season":int(base["G_season"][refit].max()),"player_schema":base["player_store"].manifest,
                    "sequence_schema":base["player_store"].sequence_manifest,"seconds":time.monotonic()-start}
                torch.save(ck,directory/f"{year}_s{seed}.pt")
                np.savez_compressed(cache,pm=pm.numpy(),pt=pt.numpy(),game_id=base["G_gid"][idx],epochs=epoch)
                print(f"MEMBER DONE {variant} {year} seed={seed} epochs={epoch} seconds={ck['seconds']:.1f}",flush=True)
                del model
                if device.type=="mps":torch.mps.empty_cache()
            margins.append(pm);totals.append(pt);epochs.append(epoch)
        pm,pt=torch.stack(margins).mean(0),torch.stack(totals).mean(0)
        frame=pd.DataFrame(readouts(pm,pt),index=base["G_gid"][idx]).join(meta)
        frame["crps"]=discrete_crps(pm.clamp(min=1e-30).log(),base["G_margin"][idx],reduction="none").numpy()
        frame["fbs_fbs"]=base["G_fbs_fbs"][idx];frame.index.name="game_id";frame=frame.reset_index()
        frame.to_parquet(directory/f"{year}_{len(seeds)}seeds_predictions.parquet",index=False)
        metrics=summarize(frame);save_json(directory/f"{year}_{len(seeds)}seeds_report.json",{"epochs":epochs,"metrics":metrics})
        print(f"YEAR DONE {variant} {year}: MAE={metrics['margin_mae']:.5f} LL={metrics['wp_logloss']:.5f}",flush=True)
        frames.append(frame)
    return pd.concat(frames,ignore_index=True)


def compare(frames):
    return {"metrics":{n:summarize(f) for n,f in frames.items()},
            "paired_vs_control":{n:paired(frames["control"],f) for n,f in frames.items() if n!="control"},
            "early_weeks_through_3":{n:summarize(f[f.week_c.le(3)]) for n,f in frames.items()},
            "by_year":{n:{str(y):summarize(f[f.season.eq(y)]) for y in sorted(f.season.unique())} for n,f in frames.items()}}


def initialize_worker(threads):
    global WORKER_BASE
    torch.set_num_threads(threads)
    WORKER_BASE=load()


def worker_run(name,years,seeds,device,output,micro_batch_size):
    return run_variant(WORKER_BASE,name,years,seeds,torch.device(device),output,micro_batch_size)


def run_stage(pool,years,seeds,device,output,micro_batch_size,stage):
    results={}
    jobs={pool.submit(worker_run,name,years,seeds,str(device),output,micro_batch_size):name
          for name in ["combined","context","auxiliary","contribution","control"]}
    for job in as_completed(jobs):
        name=jobs[job];results[name]=job.result()
        save_json(output/f"{stage}_progress.json",{n:summarize(f) for n,f in results.items()})
        save_json(output/f"status_{name}.json",{"variant":name,"stage":stage+"_complete"})
    return {name:results[name] for name in VARIANTS}


def run(output,device,micro_batch_size,workers=3,threads=2):
    if device.type!="cpu" and workers!=1:raise ValueError("GPU training must use one worker")
    prepare(output,device,micro_batch_size,workers,threads)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker,initargs=(threads,)) as pool:
        save_json(output/"status.json",{"stage":"development","members":30})
        dev=run_stage(pool,[2020,2021,2022],list(range(2)),device,output,micro_batch_size,"development")
        selected=min(dev,key=lambda n:summarize(dev[n])["margin_mae"])
        selection={"selected_variant":selected,"development":compare(dev)}
        save_json(output/"selection.json",selection)
        report={**selection,"production_modified":False,"2026_labels_used":False,"assessment_is_reused_historical_benchmark":True}
        save_json(output/"report.json",report)
        save_json(output/"status.json",{"stage":"assessment","members":45})
        assessment=run_stage(pool,[2023,2024,2025],list(range(3)),device,output,micro_batch_size,"assessment")
        report["assessment"]=compare(assessment)
    prior=ROOT/"models/experiments/players_20260906_v3/runs"
    for name in ["corrected_control","meanmax"]:
        reference=pd.concat([pd.read_parquet(prior/name/f"{y}_3seeds_predictions.parquet") for y in [2023,2024,2025]])
        report["vs_previous_"+name]={"metrics":summarize(reference),"paired":{n:paired(reference,f) for n,f in assessment.items()}}
    save_json(output/"report.json",report);save_json(output/"status.json",{"stage":"complete","members":75,"production_modified":False})
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=OUTPUT);p.add_argument("--device",default="cpu")
    p.add_argument("--threads",type=int,default=2);p.add_argument("--micro-batch",type=int,default=512)
    p.add_argument("--workers",type=int,default=3)
    a=p.parse_args();torch.set_num_threads(a.threads)
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu") if a.device=="auto" else torch.device(a.device)
    run(a.output,device,a.micro_batch,a.workers,a.threads)
