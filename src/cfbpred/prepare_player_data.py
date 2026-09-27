"""Refresh a complete, isolated player-model feature store from raw inputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cfbpred.player_inputs import build, OUTPUT as PLAYERS, POSITION_SOURCES
from cfbpred.player_scalar_corrections import corrected_scalars
from cfbpred.pregame_experiments import save_json, sha


def prepare(base,output,players=PLAYERS,rebuild_players=True,position_source="scouting"):
    if base.resolve()==output.resolve():
        raise ValueError("Use an isolated output directory; source inputs must be preserved")
    if (output/"player_data_provenance.json").exists():
        raise ValueError("A completed input snapshot already exists; use a new output directory")
    if rebuild_players and (players/"manifest.json").exists():
        raise ValueError("A completed player snapshot already exists; use a new --players directory or --reuse-players")
    output.mkdir(parents=True,exist_ok=True)
    if rebuild_players:
        build(base/"snapshot_schedule.parquet",base/"game_scalars.parquet",players,position_source)
    corrected,qb=corrected_scalars(pd.read_parquet(base/"game_scalars.parquet"),
                                 pd.read_parquet(base/"snapshot_schedule.parquet"),players)
    corrected.to_parquet(output/"game_scalars.parquet",index=False)
    qb.to_parquet(output/"game_qb_priors.parquet",index=False)
    for name in ["game_history.parquet","team_vocab.json","market.parquet","recent_form.parquet","snapshot_schedule.parquet"]:
        shutil.copy2(base/name,output/name)
    save_json(output/"player_data_provenance.json",{
        "player_manifest_sha256":sha(players/"manifest.json"),
        "source_scalars_sha256":sha(base/"game_scalars.parquet"),
        "schedule_sha256":sha(base/"snapshot_schedule.parquet"),
        "corrected_scalars_sha256":sha(output/"game_scalars.parquet"),
        "corrected_qb_sha256":sha(output/"game_qb_priors.parquet")})
    print(f"Prepared {len(corrected):,} targets at {output}",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--base",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--players",type=Path,default=PLAYERS)
    ap.add_argument("--reuse-players",action="store_true")
    ap.add_argument("--position-source",choices=POSITION_SOURCES,default="scouting")
    a=ap.parse_args();prepare(a.base,a.output,a.players,not a.reuse_players,a.position_source)
