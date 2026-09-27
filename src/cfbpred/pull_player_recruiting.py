"""Stage individual scouting records for exact player/transfer identity joins."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cfbpred.cfbd_client import CFBDClient
from cfbpred.config import CFBD, RAW
from cfbpred.pregame_experiments import save_json, sha

FIELDS=("id","athleteId","recruitType","year","ranking","name","committedTo",
        "position","height","weight","stars","rating")


def run():
    directory=RAW/"recruiting_players";directory.mkdir(parents=True,exist_ok=True)
    c=CFBDClient()
    rows=[];manifest=[]
    for year in range(2000,2027):
        for kind in ["HighSchool","JUCO","PrepSchool"]:
            path=directory/f"{year}_{kind}.json"
            if not path.exists():
                response=c.get("/recruiting/players",{"year":year,"classification":kind})
                if not isinstance(response,list):raise ValueError("Expected a list of recruits")
                selected=[{k:r.get(k) for k in FIELDS} for r in response]
                save_json(path,selected)
            records=json.loads(path.read_text())
            if any(int(r["year"])!=year or r["recruitType"]!=kind for r in records):
                raise ValueError(f"API recruiting filter mismatch: {year} {kind}")
            rows.extend({k:r.get(k) for k in FIELDS} for r in records)
            manifest.append({"year":year,"classification":kind,"rows":len(records),"file":path.name,
                             "sha256":sha(path),"retrieved_at":pd.Timestamp(path.stat().st_mtime,unit="s",tz="UTC").isoformat()})
            print(f"{year} {kind}: {len(records)}",flush=True)
    frame=pd.DataFrame(rows)
    if frame.duplicated(["id","year","recruitType"]).any():
        raise ValueError("Duplicate recruiting records")
    frame.to_parquet(CFBD/"recruiting_players.parquet",index=False)
    save_json(directory/"manifest.json",{"sources":manifest,"rows":len(frame),
        "output_sha256":sha(CFBD/"recruiting_players.parquet"),
        "scope":"Individual human scouting records only; no game prediction inputs",
        "timing_limit":"Class year is known; historical rating revisions and exact pre-kickoff publication timestamps are not supplied"})
    print(f"Staged {len(frame):,} individual recruiting records",flush=True)


if __name__=="__main__":run()
