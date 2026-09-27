"""Untruncated, ragged individual histories and all collected raw box fields.

The existing immutable eight-game store remains the control. This extension
references the same events and adds fields without modifying that store.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re

import duckdb
import numpy as np
import pandas as pd
import torch

from cfbpred.config import CFBD
from cfbpred.player_inputs import SPECS, transform_history
from cfbpred.player_sequence_inputs import SequenceStore, EVENT_COLS
from cfbpred.player_sequence_fast_inputs import FastSequenceStore
from cfbpred.pregame_experiments import save_json, sha

# Remaining observed raw fields. Ratios here are ordinary reported arithmetic,
# never an external prediction or player rating. Composite TOT is preserved
# separately because the rare provider variant is not assumed to mean FG.
EXTRA_SPECS = [
    ("defensive", "QB HUR", "qb_hurries", 0), ("defensive", "SOLO", "solo_tackles", 0),
    ("defensive", "TD", "defensive_td", 0), ("fumbles", "FUM", "fumbles", 0),
    ("fumbles", "REC", "fumble_recoveries", 0), ("interceptions", "TD", "interception_td", 0),
    ("interceptions", "YDS", "interception_yards", 0),
    ("kickReturns", "AVG", "kick_return_average", 0), ("kickReturns", "LONG", "kick_return_long", 0),
    ("kickReturns", "TD", "kick_return_td", 0), ("kicking", "LONG", "kick_long", 0),
    ("kicking", "PCT", "kick_percent", 0), ("kicking", "PTS", "kick_points", 0),
    ("kicking", "TOT", "kick_tot_first", 1), ("kicking", "TOT", "kick_tot_second", 2),
    ("passing", "AVG", "passing_average", 0),
    ("puntReturns", "AVG", "punt_return_average", 0), ("puntReturns", "LONG", "punt_return_long", 0),
    ("puntReturns", "TD", "punt_return_td", 0), ("punting", "AVG", "punt_average", 0),
    ("punting", "In 20", "punt_inside20", 0), ("punting", "LONG", "punt_long", 0),
    ("punting", "TB", "punt_touchbacks", 0), ("receiving", "AVG", "receiving_average", 0),
    ("receiving", "LONG", "receiving_long", 0), ("rushing", "AVG", "rushing_average", 0),
    ("rushing", "LONG", "rushing_long", 0),
]
EXTRA_COLS = [s[2] for s in EXTRA_SPECS]


def career_starts(events):
    """First event id for each contiguous canonical person's full career."""
    if not np.array_equal(events.event_id, np.arange(1, len(events)+1)):
        raise ValueError("Event IDs must be dense and start at one")
    starts = events.groupby("person_id", sort=False).event_id.transform("min").to_numpy(np.int32)
    sizes = events.groupby("person_id", sort=False).event_id.transform("size").to_numpy()
    ends = events.groupby("person_id", sort=False).event_id.transform("max").to_numpy()
    if not np.array_equal(ends-starts+1, sizes):
        raise ValueError("A person's events must be contiguous")
    return np.r_[np.int32(0), starts]


def ragged_careers(tips, starts):
    """All earlier event IDs, newest first, without a roster/history cap."""
    tips = np.asarray(tips, np.int64)
    counts = np.where(tips > 0, tips-starts[tips]+1, 0)
    owner = np.repeat(np.arange(len(tips)), counts)
    offset = np.repeat(np.cumsum(counts)-counts, counts)
    rank = np.arange(len(owner))-offset
    return tips[owner]-rank, owner, rank, counts


def read_extra_boxes(paths):
    expressions = []
    for cat, typ, name, part in EXTRA_SPECS:
        value = f"split_part(stat_value, '/', {part})" if part else "stat_value"
        expressions.append(f"max(case when category='{cat}' and stat_type='{typ}' then try_cast({value} as double) end) as {name}")
    with duckdb.connect() as con:
        con.execute("SET threads=2")
        present = con.execute("select distinct category,stat_type from read_parquet(?)", [[str(p) for p in paths]]).fetchall()
        supported = {(c,t) for c,t,_,_ in SPECS+EXTRA_SPECS}
        unexplained = [(c,t) for c,t in present if (c,t) not in supported]
        if unexplained:
            raise ValueError(f"New raw fields need an explicit parser or exclusion: {unexplained}")
        sql = """with unique_stats as (
            select game_id,home_away,cast(athlete_id as varchar) athlete_id,category,stat_type,
            case when count(distinct stat)=1 then max(stat) else null end as stat_value
            from read_parquet(?) where athlete_id is not null group by all)
            select game_id,home_away,athlete_id,"""+",".join(expressions)+" from unique_stats group by all"
        return con.execute(sql, [[str(p) for p in paths]]).fetchdf()


def build(players, sequences, inputs, output, raw_paths=None):
    if (output/"manifest.json").exists():
        raise ValueError("Completed full-career stores are immutable")
    output.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(sequences/"events.parquet")
    raw_paths = raw_paths or [CFBD/"player_game_stats.parquet", CFBD/"player_game_stats_fcs.parquet"]
    extra = read_extra_boxes(raw_paths)
    schedule = pd.read_parquet(inputs/"snapshot_schedule.parquet")
    from cfbpred.pull_espn_roster import team_alias_map
    aliases = team_alias_map()
    sides = pd.concat([schedule[["game_id","season",side+"_team"]].rename(columns={side+"_team":"team"}).assign(home_away=side)
                       for side in ["home","away"]], ignore_index=True)
    sides["team"] = sides.team.replace(aliases)
    extra = extra.merge(sides,on=["game_id","home_away"],validate="many_to_one")
    from cfbpred.player_identity import resolve_history
    extra = resolve_history(extra,pd.read_parquet(players/"identity_crosswalk.parquet"))
    extra = extra[extra.history_usable]
    grouped = extra.groupby(["game_id","team","person_id"])[EXTRA_COLS]
    wide = grouped.first(); wide[grouped.nunique().gt(1)] = np.nan
    wide = events.merge(wide.reset_index(),on=["game_id","team","person_id"],how="left",validate="one_to_one")
    values = wide[EXTRA_COLS].to_numpy(np.float32)
    known = np.isfinite(values)
    encoded = np.concatenate([transform_history(np.nan_to_num(values)),known.astype(np.float32)],-1)
    np.save(output/"extra_events.npy",np.vstack([np.zeros((1,encoded.shape[1]),np.float32),encoded]))
    starts = career_starts(events); np.save(output/"career_starts.npy",starts)
    length = events.groupby("person_id").size()
    report = dict(events=len(events),people=len(length),max_observed_career_games=int(length.max()),
                  people_with_more_than_eight_games=int(length.gt(8).sum()),
                  no_history_truncation=True,extra_raw_fields=EXTRA_COLS,
                  extra_observations={c:int(known[:,i].sum()) for i,c in enumerate(EXTRA_COLS)},
                  raw_stat_types_accounted_for=len({(c,t) for c,t,_,_ in SPECS+EXTRA_SPECS}),
                  limits=["Collected box history, not complete snaps or high-school game history",
                          "Unsupported identities remain quarantined; missing observations are explicit"])
    save_json(output/"coverage.json",report)
    save_json(output/"manifest.json",dict(version=1,base_sequence_sha256=sha(sequences/"manifest.json"),
              event_cols=EVENT_COLS+EXTRA_COLS+[c+"_known" for c in EXTRA_COLS],
              source_sha256={str(p):sha(p) for p in raw_paths},
              artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(report,indent=2),flush=True)


class FullCareerStore(FastSequenceStore):
    def __init__(self, players, sequences, full_path, game_ids):
        super().__init__(players,sequences,game_ids)
        self.full_path=Path(full_path)
        self.full_manifest=json.loads((self.full_path/"manifest.json").read_text())
        if self.full_manifest["base_sequence_sha256"]!=sha(self.sequence_path/"manifest.json"):
            raise ValueError("Full history/sequence stores differ")
        for name,digest in self.full_manifest["artifacts"].items():
            if sha(self.full_path/name)!=digest: raise ValueError(f"Full history changed: {name}")
        self.extra_events=np.load(self.full_path/"extra_events.npy",mmap_mode="r")
        self.career_start=np.load(self.full_path/"career_starts.npy")
        self.include_full_history=True
        self.include_timeline=False

    def for_fold(self,train_indices,context=True,role=False):
        other=copy.copy(self);other.include_context=context;other.include_role=role
        # Include every program/opponent in every prior career observation
        # accessible to training targets, not just those in their last 8.
        offsets=self.offsets[train_indices].ravel();counts=self.counts[train_indices].ravel()
        refs=np.concatenate([np.arange(o,o+c) for o,c in zip(offsets,counts)])
        tips=self.history_tips[refs];tips=tips[tips>0]
        latest=np.zeros(len(self.events),np.int32)
        np.maximum.at(latest,self.career_start[tips],tips)
        end=latest[latest>0]
        ids,_,_,_=ragged_careers(end,self.career_start)
        other.known_teams=np.unique(np.r_[0,self.event_team[ids],self.event_opponent[ids]])
        return other

    def batch(self,indices,device):
        # Reuse audited static/career aggregates. Suppress construction of the
        # bounded context branch, then supply the complete ragged history.
        basic=copy.copy(self);basic.include_context=False
        result=FastSequenceStore.batch(basic,indices,device)
        refs,valid=self.roster_refs(indices);packed=refs[valid]
        p=self.player_refs[packed];tips=self.history_tips[packed]
        if not hasattr(self,"roster_team"):
            vocab=json.loads((self.sequence_path/"team_vocab.json").read_text())["team"]
            cat=pd.read_parquet(self.path/"player_seasons.parquet",columns=["team"])
            self.roster_team=np.r_[0,cat.team.map(vocab).fillna(0).to_numpy(np.int64)]
        kick=np.broadcast_to(self.kicks[indices][...,None],valid.shape)[valid]
        season=np.broadcast_to(self.seasons[indices][...,None],valid.shape)[valid]
        ids,owner,rank,counts=ragged_careers(tips,self.career_start)
        days=(kick[owner]-self.event_kick[ids])/86400e9
        if len(days) and (days<=0).any(): raise ValueError("Non-prior player observation")
        unique,inverse=np.unique(ids,return_inverse=True)
        if self.known_teams is None: raise ValueError("Training-known vocabulary is required")
        team=np.where(np.isin(self.event_team[unique],self.known_teams),self.event_team[unique],0)
        opponent=np.where(np.isin(self.event_opponent[unique],self.known_teams),self.event_opponent[unique],0)
        temporal=np.column_stack([np.log1p(days)/7,self.event_season[ids]==season[owner],
                    (self.event_team[ids]==self.roster_team[p[owner]])&(self.roster_team[p[owner]]>0),
                    np.log1p(rank)/5]).astype(np.float32)
        result.update(career_values=torch.tensor(np.concatenate([self.events[unique],self.extra_events[unique]],-1),device=device),
                      career_team=torch.tensor(team,device=device),career_opponent=torch.tensor(opponent,device=device),
                      career_refs=torch.tensor(inverse,device=device),career_owner=torch.tensor(owner,device=device),
                      career_temporal=torch.tensor(temporal,device=device),
                      career_is_latest=torch.tensor(rank==0,device=device),
                      career_counts=torch.tensor(counts,device=device))
        if self.include_timeline:
            result=self.timeline_batch(result,indices,packed,valid,device)
        return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ["players","sequences","inputs","output"]:parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();build(args.players,args.sequences,args.inputs,args.output)
