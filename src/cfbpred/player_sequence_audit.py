"""Verify all frozen experiment members and reproduce a saved prediction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cfbpred.player_sequence_experiment import INPUTS,PLAYERS,SEQUENCES,load,fold_data
from cfbpred.player_sequence_fast_experiment import OUTPUT
from cfbpred.player_sequence_net import VARIANTS,make
from cfbpred.player_sequence_train import predict
from cfbpred.pregame_experiments import save_json,sha


def audit(output=OUTPUT):
    plan=json.loads((output/"plan.json").read_text())
    for name,digest in plan["code"].items():
        if sha(Path(__file__).parent/name)!=digest:raise ValueError(f"Frozen code changed: {name}")
    for name,digest in plan["inputs"].items():
        if sha(INPUTS/name)!=digest:raise ValueError(f"Frozen inputs changed: {name}")
    if sha(PLAYERS/"manifest.json")!=plan["player_manifest_sha256"] or sha(SEQUENCES/"manifest.json")!=plan["sequence_manifest_sha256"]:
        raise ValueError("Frozen player stores changed")
    count=0;last=None;max_mass_error=0.;selected=json.loads((output/"selection.json").read_text())["selected_variant"]
    source=pd.read_parquet(INPUTS/"game_scalars.parquet").set_index("game_id")
    rows=[]
    for name in VARIANTS:
        for years,seeds in [([2020,2021,2022],2),([2023,2024,2025],3)]:
            for year in years:
                for seed in range(seeds):
                    path=output/"runs"/name/f"{year}_s{seed}.pt"
                    ck=torch.load(path,map_location="cpu",weights_only=False)
                    if ck["train_max_season"]>=year or ck["train_max_season"]>=2026 or not ck["pregame_only"]:
                        raise ValueError("Checkpoint training cutoff failed")
                    train=source.loc[ck["train_game_ids"]]
                    if train.margin.isna().any() or train.season.ge(year).any() or int(train.season.max())!=ck["train_max_season"]:
                        raise ValueError("Actual training IDs violate the recorded cutoff")
                    with np.load(path.with_suffix(".npz")) as p:
                        if np.intersect1d(p["game_id"],ck["train_game_ids"]).size:raise ValueError("Forecast targets entered training")
                        expected=source.index[source.season.eq(year)&source.margin.notna()&(source.home_division.eq("fbs")|source.away_division.eq("fbs"))]
                        if not np.array_equal(np.sort(p["game_id"]),np.sort(expected)):raise ValueError("Missing or extra historical forecast targets")
                        for field in ["pm","pt"]:
                            values=p[field]
                            if not np.isfinite(values).all() or (values<0).any():raise ValueError("Invalid forecast probability")
                            error=float(abs(values.sum(-1)-1).max());max_mass_error=max(max_mass_error,error)
                            if error>1e-5:raise ValueError("Forecast distribution not normalized")
                    rows.append({"variant":name,"year":year,"seed":seed,"epochs":ck["epochs"],"seconds":ck["seconds"],
                                 "state_tensor_elements":sum(v.numel() for v in ck["state_dict"].values()),
                                 "checkpoint_sha256":sha(path)})
                    count+=1
                    if name==selected and year==2025 and seed==0:last=ck
    if count!=75 or last is None:raise ValueError("Incomplete experiment")
    torch.set_num_threads(2);base=load();mask=base["G_labeled"]&(base["G_season"]<2025)
    d=fold_data(base,mask,selected)
    if not np.array_equal(last["mu"],d["mu"]) or not np.array_equal(last["sd"],d["sd"]):raise ValueError("Checkpoint normalization differs")
    model=make(d,selected);model.pre.load_state_dict(last["state_dict"])
    cache=np.load(output/"runs"/selected/"2025_s0.npz")
    index=pd.Series(np.arange(len(base["G_gid"])),index=base["G_gid"])
    indices=index.loc[cache["game_id"][:128]].to_numpy()
    pm,pt=predict(model,d,indices,torch.device("cpu"))
    parity=max(float(abs(pm.numpy()-cache["pm"][:128]).max()),float(abs(pt.numpy()-cache["pt"][:128]).max()))
    if parity>1e-6:raise ValueError("Checkpoint reload does not reproduce saved predictions")
    report={"passed":True,"members":count,"max_probability_mass_error":max_mass_error,
            "checkpoint_roundtrip_max_probability_difference":parity,"roundtrip_variant":selected,
            "no_2026_fitting_or_selection":True,"all_forecast_games_excluded_from_member_training":True,
            "frozen_code_and_input_hashes_match":True,"members_detail":rows}
    save_json(output/"completion_audit.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="members_detail"},indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=OUTPUT)
    audit(p.parse_args().output)
