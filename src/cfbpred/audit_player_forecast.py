"""Verify completeness, timing, checkpoint lineage and actual player input use."""
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
from cfbpred.player_experiment import OUTPUT, load_inputs, make
from cfbpred.player_forecast import checkpoint_inputs
from cfbpred.player_inputs import OUTPUT as PLAYERS
from cfbpred.pregame_experiments import save_json, sha
from cfbpred.train import game_batch


def audit(inputs,players,bundle_path,forecasts,output):
    d=load_inputs(inputs,players)
    bundle=json.loads(bundle_path.read_text())
    meta=json.loads(forecasts.with_suffix(".json").read_text())
    if sha(bundle_path)!=meta["bundle_sha256"] or sha(players/"manifest.json")!=meta["player_manifest_sha256"]:
        raise ValueError("Forecast lineage differs from the supplied model/data")
    frame=pd.read_parquet(forecasts)
    schedule=pd.read_parquet(inputs/"snapshot_schedule.parquet").set_index("game_id")
    cutoff=pd.Timestamp(meta["cutoff"])
    kicks=pd.to_datetime(schedule.loc[d["G_gid"],"kick"],utc=True)
    future=(d["G_season"]==bundle["forecast_year"])&(kicks>cutoff).to_numpy()
    ids=d["G_gid"][future]
    if frame.game_id.duplicated().any() or set(ids)!=set(frame.game_id):
        raise ValueError("Forecast artifact omits or duplicates future targets")
    pm=torch.tensor(np.stack(frame.margin_pmf),dtype=torch.float32)
    pt=torch.tensor(np.stack(frame.total_pmf),dtype=torch.float32)
    for p in [pm,pt]:
        if not torch.isfinite(p).all() or (p<0).any() or not torch.allclose(p.sum(-1),torch.ones(len(frame)),atol=1e-5):
            raise ValueError("Invalid forecast distribution")
    readings=readouts(pm,pt)
    for saved,key in [("home_win_prob","wp"),("pred_margin","em"),("pred_total","et")]:
        if not np.allclose(frame[saved],readings[key],atol=1e-5):
            raise ValueError("Forecast summaries disagree with the distributions")
    rosters=pd.read_parquet(players/"game_rosters.parquet")
    dated=rosters.last_history_kick.notna()
    if (rosters.loc[dated,"last_history_kick"]>=rosters.loc[dated,"kick"]).any():
        raise ValueError("Player history includes target/future outcomes")
    future_rosters=rosters[rosters.game_id.isin(ids)].copy()
    future_rosters["division"]=[schedule.loc[r.game_id,r.side+"_division"] for r in future_rosters.itertuples()]
    missing=future_rosters[future_rosters.players.eq(0)]
    if missing.division.eq("fbs").any():
        raise ValueError("At least one future FBS team lacks its player roster")
    fbs_teams=set(future_rosters.loc[future_rosters.division.eq("fbs"),"team"])
    unresolved=pd.read_parquet(players/"quarantined_history.parquet")
    unresolved_fbs=unresolved[unresolved.season.eq(bundle["forecast_year"])&unresolved.team.isin(fbs_teams)&
                              pd.to_numeric(unresolved.athlete_id,errors="coerce").gt(0)]
    if len(unresolved_fbs):
        raise ValueError("Current FBS player box observations still have unresolved identities")

    checks=[]
    for name in bundle["members"]:
        path=bundle_path.parent/name
        if sha(path)!=bundle["member_sha256"][name]:raise ValueError("Checkpoint hash changed")
        ck=torch.load(path,map_location="cpu",weights_only=False)
        if ck["forecast_year"]!=bundle["forecast_year"]:raise ValueError("Mixed forecast seasons")
        checkpoint_inputs(d,ck)
        seasons=schedule.loc[ck["train_game_ids"],"season"]
        if (seasons>=bundle["forecast_year"]).any():raise ValueError("Forecast-season labels entered training")
        checks.append({"member":name,"training_games":len(seasons),"latest_training_season":int(seasons.max())})

    # This is a wiring diagnostic, not a claim of predictive benefit. The
    # historical control comparison is what measures predictive benefit.
    torch.set_num_threads(4)
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    fitted=checkpoint_inputs(d,ck)
    model=make(fitted,ck["pooling"]).to(device).eval()
    model.pre.load_state_dict(ck["state_dict"])
    idx=np.where(future)[0][:128]
    batch=game_batch(fitted,torch.tensor(idx),device)
    sensitivity={}
    with torch.no_grad():
        _,m,t=model.pregame(batch)
        actual=readouts(m.softmax(-1).cpu(),t.softmax(-1).cpu())
        for group in (["scouting","prior_college_history","current_season_history"] if ck["architecture"]=="players_v1" else []):
            changed=batch.copy()
            if group=="scouting":
                changed["player_scout"]=batch["player_scout"].clone()
                changed["player_scout"][...,:27]=0
            else:
                changed["player_history"]=batch["player_history"].clone()
                window=slice(0,3) if group=="prior_college_history" else slice(3,4)
                changed["player_history"][...,window,:]=0
            _,am,at=model.pregame(changed)
            r=readouts(am.softmax(-1).cpu(),at.softmax(-1).cpu())
            sensitivity[group]={"mean_absolute_margin_change":float(np.abs(actual["em"]-r["em"]).mean()),
                                "mean_absolute_total_change":float(np.abs(actual["et"]-r["et"]).mean())}
    report={"passed":True,"future_games":len(frame),"cutoff":cutoff.isoformat(),
            "future_fbs_sides":int(future_rosters.division.eq("fbs").sum()),
            "missing_future_fbs_rosters":int(missing.division.eq("fbs").sum()),
            "missing_other_division_rosters":int(missing.division.ne("fbs").sum()),
            "unresolved_current_fbs_positive_id_box_rows":len(unresolved_fbs),
            "historical_player_timing_verified":True,"members":checks,
            "input_removal_diagnostic":{"games":len(idx),"seed":ck["seed"],"effects":sensitivity,
                                        "applicable":ck["architecture"]=="players_v1",
                                        "interpretation":"Confirms data reaches player-model predictions; does not estimate accuracy improvement. Scalar-only control has no direct player encoder."},
            "forecast_sha256":sha(forecasts),"production_modified":False}
    save_json(output,report)
    print(json.dumps(report,indent=2))
    return report


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--inputs",type=Path,default=OUTPUT/"inputs")
    ap.add_argument("--players",type=Path,default=PLAYERS)
    ap.add_argument("--bundle",type=Path,default=OUTPUT/"current_2026/bundle.json")
    ap.add_argument("--forecasts",type=Path,default=OUTPUT/"future_predictions.parquet")
    ap.add_argument("--output",type=Path,default=OUTPUT/"forecast_audit.json")
    a=ap.parse_args();audit(a.inputs,a.players,a.bundle,a.forecasts,a.output)
