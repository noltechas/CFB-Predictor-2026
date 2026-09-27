"""Inspect learned contribution weights for every player in saved forecasts.

Weights are normalized within each position group, not snap-share estimates or
medical availability probabilities. Report status is separately attributed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cfbpred.player_context_experiment import load,fold,network
from cfbpred.player_inputs import POSITIONS
from cfbpred.player_timeline import CONTEXT_COLS
from cfbpred.pregame_experiments import sha,save_json
from cfbpred.train import game_batch


def export(directory,output):
    torch.set_num_threads(2)
    if (directory/"INVALIDATED.json").exists():raise ValueError("Candidate was invalidated; use the revision-checked run")
    bundle=json.loads((directory/"bundle.json").read_text())
    if bundle["variant"]!="full_context":raise ValueError("Role export requires a context candidate")
    ckpath=directory/bundle["member"]
    if sha(ckpath)!=bundle["member_sha256"]:raise ValueError("Checkpoint changed")
    source=Path(bundle["source"]);plan=json.loads((directory/"plan.json").read_text())
    if sha(directory/"plan.json")!=bundle["plan_sha256"]:raise ValueError("Bundle plan changed")
    if sha(source)!=plan["source_sha256"]:raise ValueError("Source changed")
    for name,digest in plan["code"].items():
        if sha(Path(__file__).parent/name)!=digest:raise ValueError(f"Candidate code changed: {name}")
    base,paths=load(source,"full_context")
    for name,digest in plan["inputs"].items():
        if sha(paths["inputs"]/name)!=digest:raise ValueError(f"Input changed: {name}")
    for key,digest in plan["stores"].items():
        if sha(paths[key]/"manifest.json")!=digest:raise ValueError(f"Store changed: {key}")
    ck=torch.load(ckpath,map_location="cpu",weights_only=False)
    mask=base["G_labeled"]&(base["G_season"]<ck["forecast_year"])
    d=fold(base,mask,"full_context")
    if any(not np.array_equal(ck[k],d[k]) for k in ["mu","sd"]):raise ValueError("Normalization changed")
    model=network(d,"full_context");model.pre.load_state_dict(ck["state_dict"]);model.eval()
    catalog=pd.read_parquet(paths["players"]/"player_seasons.parquet")
    forecasts=pd.read_parquet(directory/"predictions.parquet")
    indices=pd.Series(np.arange(len(base["G_gid"])),index=base["G_gid"]).loc[forecasts.game_id].to_numpy()
    captured=[]
    hook=model.pre.encoder.role.register_forward_hook(lambda module,args,out:captured.append(out.detach()))
    frames=[];store=d["player_store"]
    positions={v:k for k,v in POSITIONS.items()}
    with torch.no_grad():
        for start in range(0,len(indices),64):
            ids=indices[start:start+64];batch=game_batch(d,torch.tensor(ids),torch.device("cpu"))
            captured.clear();model.pregame(batch)
            if len(captured)!=1:raise ValueError("Unexpected role invocation")
            slots=batch["player_slots"];present=batch["player_mask"]
            scores=torch.full(present.shape,-1e4)
            scores.reshape(-1)[slots]=captured[0].squeeze(-1)
            active=present&~batch["player_confirmed_out"]
            weight=torch.zeros_like(scores)
            for pos in range(len(POSITIONS)):
                valid=active&batch["player_position"].eq(pos)
                w=scores.masked_fill(~valid,-1e4).softmax(-1)*valid
                w=w/w.sum(-1,keepdim=True).clamp(min=1e-8)
                weight+=w
                if not torch.allclose(w.sum(-1),valid.any(-1).float(),atol=1e-6):raise ValueError("Role normalization failed")
            refs,valid=store.roster_refs(ids);packed=refs[valid];player_refs=store.player_refs[packed]
            frame=catalog.iloc[player_refs-1][["person_id","athlete_id","firstName","lastName","team"]].reset_index(drop=True)
            frame["game_id"]=np.broadcast_to(base["G_gid"][ids,None,None],valid.shape)[valid]
            frame["side"]=np.broadcast_to(np.array(["home","away"])[None,:,None],valid.shape)[valid]
            frame["player_ref"]=player_refs
            frame["position"]=pd.Series(batch["player_position"].numpy()[valid]).map(positions)
            frame["membership_active"]=present.numpy()[valid]
            frame["expected_contribution_weight"]=weight.numpy()[valid]
            ctx=batch["player_context"].numpy();status=np.full(len(frame),"unknown",dtype=object)
            for name in ["out","doubtful","questionable","probable","available"]:
                status[ctx[:,CONTEXT_COLS.index(name)]>0]=name
            frame["availability"]=status
            frame["reported_depth_rank"]=np.where(ctx[:,6]>0,1/np.maximum(ctx[:,7],1e-8),np.nan)
            frame["dated_position"]=ctx[:,8]>0;frame["dated_membership"]=ctx[:,9]>0
            frame["verified_transfer_destination"]=ctx[:,11]>0
            frame["identity_quarantined"]=ctx[:,13]>0
            frame["prior_career_games"]=batch["career_counts"].numpy()
            frames.append(frame)
    hook.remove();result=pd.concat(frames,ignore_index=True)
    if result.duplicated(["game_id","side","player_ref"]).any():raise ValueError("Duplicate exported roster slot")
    if result.loc[~result.membership_active|result.availability.eq("out"),"expected_contribution_weight"].ne(0).any():
        raise ValueError("Excluded member or confirmed-out player received active-role weight")
    output.parent.mkdir(parents=True,exist_ok=True);result.to_parquet(output,index=False)
    save_json(output.with_suffix(".json"),dict(player_game_rows=len(result),games=int(result.game_id.nunique()),
        checkpoint_sha256=sha(ckpath),forecast_sha256=sha(directory/"predictions.parquet"),output_sha256=sha(output),
        weight_definition="Learned contribution weight within position group; not calibrated snaps or probability of playing",
        availability_unknown_is_not_healthy=True,all_forecast_rosters_exported=True))
    print(f"Exported {len(result):,} player-game roles for {result.game_id.nunique():,} games",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--directory",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();export(a.directory,a.output)
