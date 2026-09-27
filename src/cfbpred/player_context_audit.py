"""Coverage and timing audit for immutable player context snapshots."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cfbpred.pregame_experiments import save_json,sha


def audit(source,output):
    paths={k:Path(v) for k,v in json.loads(source.read_text()).items()}
    players,timeline=paths["players"],paths["timeline"]
    manifest=json.loads((timeline/"manifest.json").read_text())
    for name,digest in manifest["artifacts"].items():
        if sha(timeline/name)!=digest:raise ValueError("Timeline changed")
    games=pd.read_parquet(players/"game_rosters.parquet")
    schedule=pd.read_parquet(paths["inputs"]/"snapshot_schedule.parquet").set_index("game_id")
    refs=np.load(players/"player_refs.npy",mmap_mode="r")
    basepos=np.load(players/"position.npy",mmap_mode="r")
    context=np.load(timeline/"context.npy",mmap_mode="r")
    dated=np.load(timeline/"position.npy",mmap_mode="r")
    departed=np.load(timeline/"departed.npy",mmap_mode="r")
    now=pd.Timestamp.now(tz="UTC")
    summaries=[];future_refs={};future_positions={}
    for game in games.itertuples():
        ix=np.arange(game.offset,game.offset+game.players)
        active=~departed[ix];newpos=np.where(dated[ix]>=0,dated[ix],basepos[refs[ix]])
        division=schedule.loc[game.game_id,game.side+"_division"]
        summaries.append(dict(game_id=game.game_id,season=game.season,team=game.team,side=game.side,
             kick=game.kick,division=division,roster_players=game.players,active_members=int(active.sum()),
             unknown_position_before=int((basepos[refs[ix]]==0).sum()),
             unknown_position_after=int(((newpos==0)&active).sum()),
             reports_known=int(context[ix,0].sum()),depth_known=int(context[ix,6].sum()),
             known_out=int(context[ix,1].sum()),membership_excluded=int(departed[ix].sum())))
        if game.season==2026 and pd.Timestamp(game.kick)>now and division=="fbs":
            for i,ref in enumerate(refs[ix]):
                future_refs[int(ref)]=int(basepos[ref])
                if active[i]:future_positions[int(ref)]=int(newpos[i])
    frame=pd.DataFrame(summaries);output.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(output/"game_coverage.parquet",index=False)
    facts=pd.read_parquet(timeline/"facts.parquet")
    reported=facts[facts.kind.eq("availability")]
    # Verify all eligible explicit injury/depth facts resolve before kickoff.
    scoped=facts[facts.game_id.notna()].copy()
    scoped["kick"]=scoped.game_id.map(schedule.kick)
    if (scoped.known_at>=scoped.kick).any():raise ValueError("Post-kickoff scoped fact in candidate")
    season=frame.groupby("season").agg(games=("game_id","nunique"),team_games=("game_id","size"),
               roster_slots=("roster_players","sum"),reports_known=("reports_known","sum"),depth_known=("depth_known","sum"))
    report=dict(source_sha256=sha(source),all_target_games=int(frame.game_id.nunique()),
                all_team_game_rosters=len(frame),future_fbs_roster_players_before=len(future_refs),
                future_fbs_active_members=len(future_positions),
                future_fbs_unknown_position_before=sum(p==0 for p in future_refs.values()),
                future_fbs_unknown_position_after=sum(p==0 for p in future_positions.values()),
                explicit_reported_games=int(reported.game_id.nunique()),
                explicit_reported_teams=sorted(reported.team.unique().tolist()),
                reports_by_season=reported.groupby("season").game_id.nunique().to_dict(),
                history=json.loads((paths["full_history"]/"coverage.json").read_text()),
                context=json.loads((timeline/"coverage.json").read_text()),
                season_coverage=season.reset_index().to_dict("records"),
                limits=["Unknown report does not imply available; unlisted players are not automatically healthy",
                        "Learned within-position contribution weights are not calibrated snaps or participation probabilities",
                        "Portal date is not destination announcement date",
                        "Public historical availability remains sparse and uneven across teams and seasons",
                        "Game-level box history begins in 2004; no complete high-school/JUCO snap histories",
                        "Existing coach features remain limited; comprehensive dated staff/scheme histories are not supplied",
                        "Annual roster membership without timestamped contrary evidence is still a historical limitation"])
    save_json(output/"coverage.json",report);print(json.dumps({k:v for k,v in report.items() if k not in {"history","context","season_coverage"}},indent=2),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();audit(a.source,a.output)
