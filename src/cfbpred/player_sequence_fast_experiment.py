"""Same frozen statistical experiment, using audited equivalent input batching."""
from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import pandas as pd
import torch

from cfbpred import player_sequence_experiment as experiment
from cfbpred.player_sequence_fast_inputs import FastSequenceStore
from cfbpred.player_sequence_net import VARIANTS
from cfbpred.pregame_experiments import paired,save_json,summarize

OUTPUT=experiment.ROOT/"models/experiments/player_sequences_20260907_v2"


def initialize(threads):
    global BASE
    torch.set_num_threads(threads);BASE=experiment.load()
    BASE["player_store"].__class__=FastSequenceStore


def work(name,years,seeds,output,micro_batch_size):
    return experiment.run_variant(BASE,name,years,seeds,torch.device("cpu"),output,micro_batch_size)


def stage(pool,years,seeds,output,micro_batch_size,label):
    results={}
    jobs={pool.submit(work,n,years,seeds,output,micro_batch_size):n for n in ["combined","context","auxiliary","contribution","control"]}
    for job in as_completed(jobs):
        name=jobs[job];results[name]=job.result()
        save_json(output/f"{label}_progress.json",{n:summarize(f) for n,f in results.items()})
        save_json(output/f"status_{name}.json",{"variant":name,"stage":label+"_complete"})
    return {name:results[name] for name in VARIANTS}


def run(output,threads=2,workers=3,micro_batch_size=512):
    experiment.CODE=experiment.CODE+["player_sequence_fast_inputs.py","player_sequence_fast_experiment.py"]
    experiment.prepare(output,torch.device("cpu"),micro_batch_size,workers,threads)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),initializer=initialize,initargs=(threads,)) as pool:
        save_json(output/"status.json",{"stage":"development","members":30})
        development=stage(pool,[2020,2021,2022],[0,1],output,micro_batch_size,"development")
        selected=min(development,key=lambda n:summarize(development[n])["margin_mae"])
        selection={"selected_variant":selected,"development":experiment.compare(development)}
        save_json(output/"selection.json",selection)
        report={**selection,"production_modified":False,"2026_labels_used":False,"assessment_is_reused_historical_benchmark":True}
        save_json(output/"report.json",report)
        save_json(output/"status.json",{"stage":"assessment","members":45})
        assessment=stage(pool,[2023,2024,2025],[0,1,2],output,micro_batch_size,"assessment")
        report["assessment"]=experiment.compare(assessment)
    prior=experiment.ROOT/"models/experiments/players_20260906_v3/runs"
    for name in ["corrected_control","meanmax"]:
        reference=pd.concat([pd.read_parquet(prior/name/f"{y}_3seeds_predictions.parquet") for y in [2023,2024,2025]])
        report["vs_previous_"+name]={"metrics":summarize(reference),"paired":{n:paired(reference,f) for n,f in assessment.items()}}
    save_json(output/"report.json",report);save_json(output/"status.json",{"stage":"complete","members":75,"production_modified":False})
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=OUTPUT);p.add_argument("--threads",type=int,default=2)
    p.add_argument("--workers",type=int,default=3);p.add_argument("--micro-batch",type=int,default=512)
    a=p.parse_args();torch.set_num_threads(a.threads);run(a.output,a.threads,a.workers,a.micro_batch)
