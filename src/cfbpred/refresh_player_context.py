"""Collect public evidence, build a dated snapshot, and train/score a candidate.

Use a source.json pointing at an already completed player/scalar refresh.
This command never changes production activation or publishes predictions.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import pandas as pd

from cfbpred.player_context_audit import audit
from cfbpred.player_context_experiment import run
from cfbpred.player_full_history import build as build_history
from cfbpred.player_timeline import build as build_timeline
from cfbpred.public_player_reports import sec_current
from cfbpred.pull_player_context import pull,OUTPUT as RAW_OUTPUT
from cfbpred.pregame_experiments import save_json,sha


def default_reviewed():
    """Follow the reviewed public-data registry without rewriting old runs."""
    pointer = Path("data/player_context_reviewed_latest.json")
    if not pointer.exists():
        return Path("data/player_context_reviewed_revision_checked.json")
    info = json.loads(pointer.read_text())
    reviewed = Path(info["reviewed"])
    if sha(reviewed) != info["sha256"]:
        raise ValueError("Latest reviewed registry changed after validation")
    return reviewed


def refresh(source,reviewed,output,raw=None,train=True,year=2026):
    output.mkdir(parents=True,exist_ok=False)
    paths=json.loads(source.read_text())
    # Capture reviewed evidence before doing network work; subsequent edits
    # to the working registry cannot alter this snapshot's lineage.
    reviewed_bytes=reviewed.read_bytes()
    (output/"reviewed_input.json").write_bytes(reviewed_bytes)
    raw=raw or pull(RAW_OUTPUT,start_year=2021,end_year=year)
    schedule=pd.read_parquet(Path(paths["inputs"])/"snapshot_schedule.parquet")
    facts=json.loads(reviewed_bytes);issues=[]
    from cfbpred.conference_availability import collect, load_facts
    collect()
    extra,bad=load_facts(schedule)
    facts.extend(extra);issues.extend(bad)
    # Programs outside the official conference feeds (Sun Belt, CUSA, Mountain
    # West, Pac-12, independents) and games the feeds missed: Covers captures,
    # only for team-games no official source already reports.
    from cfbpred.sunbelt_availability import collect as sunbelt_collect, load_facts as sunbelt_facts
    try:
        sunbelt_collect()
    except Exception as exc:
        issues.append(dict(reason="sunbelt_capture_failed",error=repr(exc)))
    covered={(int(f["game_id"]),f["team"]) for f in facts
             if f.get("kind")=="availability" and f.get("game_id") is not None}
    extra,bad=sunbelt_facts(schedule)
    facts.extend(f for f in extra if (int(f["game_id"]),f["team"]) not in covered);issues.extend(bad)
    from cfbpred.covers_availability import collect as covers_collect, load_facts as covers_facts
    try:
        covers_collect()
    except Exception as exc:  # a failed capture must not block the refresh
        issues.append(dict(reason="covers_capture_failed",error=repr(exc)))
    covered={(int(f["game_id"]),f["team"]) for f in facts
             if f.get("kind")=="availability" and f.get("game_id") is not None}
    catalog=pd.read_parquet(Path(paths["players"])/"player_seasons.parquet")
    extra,bad=covers_facts(schedule,catalog,covered)
    facts.extend(extra);issues.extend(bad)
    for p in sorted(raw.glob("*_report_public.json")):
        extra,bad=sec_current(p,schedule);facts.extend(extra)
        issues.extend({**row,"archive":str(p)} for row in bad)
    save_json(output/"reviewed_and_current.json",facts)
    save_json(output/"report_source_issues.json",issues)
    if "full_history" not in paths:
        paths["full_history"]=str((output/"full_history").resolve())
        build_history(Path(paths["players"]),Path(paths["sequences"]),Path(paths["inputs"]),Path(paths["full_history"]))
    else:
        history=json.loads((Path(paths["full_history"])/"manifest.json").read_text())
        if history["base_sequence_sha256"]!=sha(Path(paths["sequences"])/"manifest.json"):
            raise ValueError("Full history belongs to a different source refresh")
    frozen_source=output/"source.json";save_json(frozen_source,paths)
    build_timeline(frozen_source,raw,output/"reviewed_and_current.json",output/"timeline")
    paths["timeline"]=str((output/"timeline").resolve());save_json(frozen_source,paths)
    audit(frozen_source,output/"audit")
    if train:run(frozen_source,output/"candidate","full_context",year=year)
    save_json(output/"status.json",dict(complete=True,trained=train,production_modified=False,
                                        source_sha256=sha(frozen_source)))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--reviewed",type=Path)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--raw",type=Path)
    p.add_argument("--no-train",action="store_true");p.add_argument("--year",type=int,default=2026)
    a=p.parse_args();refresh(a.source,a.reviewed or default_reviewed(),a.output,a.raw,not a.no_train,a.year)
