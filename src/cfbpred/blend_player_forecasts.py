"""Apply the development-selected control/player mixture to all future games."""
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
from cfbpred.player_experiment import OUTPUT
from cfbpred.player_forecast import score
from cfbpred.player_inputs import OUTPUT as PLAYERS
from cfbpred.pregame_experiments import save_json,sha


def blend(inputs,players,selection_path,player_forecasts,control_bundle,output):
    selection=json.loads(selection_path.read_text());weight=float(selection["player_weight"])
    if not 0<=weight<=1:raise ValueError("Mixture weight must be a probability")
    meta=json.loads(player_forecasts.with_suffix(".json").read_text())
    if meta["player_manifest_sha256"]!=sha(players/"manifest.json"):
        raise ValueError("Player forecast uses a different player snapshot")
    player=pd.read_parquet(player_forecasts).set_index("game_id")
    control_path=output.with_name(output.stem+"_control.parquet")
    control=score(inputs,players,control_bundle,control_path,now=meta["cutoff"]).set_index("game_id")
    if player.index.has_duplicates or set(player.index)!=set(control.index):
        raise ValueError("Control/player forecast targets differ")
    control=control.loc[player.index]
    for field in ["season","home_team","away_team","kick"]:
        if not player[field].equals(control[field]):raise ValueError("Control/player forecast schedules differ")
    pm,pt=[torch.tensor(weight*np.stack(player[key])+(1-weight)*np.stack(control[key]),dtype=torch.float32)
           for key in ["margin_pmf","total_pmf"]]
    for p in [pm,pt]:
        if not torch.isfinite(p).all() or (p<0).any() or not torch.allclose(p.sum(-1),torch.ones(len(player)),atol=1e-5):
            raise ValueError("Invalid blended distribution")
    r=readouts(pm,pt);out=player.copy()
    out["home_win_prob"]=r["wp"];out["pred_margin"]=r["em"];out["pred_total"]=r["et"]
    out["margin_pmf"]=pm.numpy().tolist();out["total_pmf"]=pt.numpy().tolist()
    out["model"]="players_blend_v1";out["player_weight"]=weight
    out.reset_index().to_parquet(output,index=False)
    save_json(output.with_suffix(".json"),{"future_games":len(out),"cutoff":meta["cutoff"],
              "player_weight":weight,"control_weight":1-weight,"selection_sha256":sha(selection_path),
              "player_forecast_sha256":sha(player_forecasts),"control_forecast_sha256":sha(control_path),
              "player_manifest_sha256":sha(players/"manifest.json"),"production_modified":False})
    print(f"Scored {len(out):,} future games at {weight:.0%} player weight -> {output}")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--inputs",type=Path,default=OUTPUT/"inputs")
    ap.add_argument("--players",type=Path,default=PLAYERS)
    ap.add_argument("--selection",type=Path,default=OUTPUT/"blend_selection.json")
    ap.add_argument("--player-forecasts",type=Path,default=OUTPUT/"future_predictions.parquet")
    ap.add_argument("--control-bundle",type=Path,default=OUTPUT/"current_control_2026/bundle.json")
    ap.add_argument("--output",type=Path,default=OUTPUT/"blended_future_predictions.parquet")
    a=ap.parse_args();blend(a.inputs,a.players,a.selection,a.player_forecasts,a.control_bundle,a.output)
