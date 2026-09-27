"""Point-in-time public player facts. Unknown is never synonymous with healthy.

Annual rosters supply candidate members. Timestamped evidence overrides their
position/membership per game; portal entry dates never date a destination.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch

from cfbpred.player_full_history import FullCareerStore
from cfbpred.player_inputs import POSITIONS, POSITION_ALIASES
from cfbpred.pregame_experiments import save_json, sha

CONTEXT_COLS = ["report_known", "out", "doubtful", "questionable", "probable", "available",
                "depth_known", "depth_rank_inverse", "position_dated", "membership_dated",
                "departed", "transfer_destination_verified", "fact_age_days_log", "identity_quarantined"]
STATUSES = {"out", "doubtful", "questionable", "probable", "available"}


def name_key(value):
    value = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", str(value).lower())
    return re.sub(r"[^a-z]", "", value)


def eligible(facts, kick, game_id, season):
    """Strict publication cutoff; inclusive effective start; exclusive expiry."""
    return (facts.known_at.lt(kick) & facts.effective_at.le(kick) & facts.valid_until.gt(kick)
            & facts.season.eq(season) & (facts.game_id.isna() | facts.game_id.eq(game_id)))


def resolve_names(facts, catalog):
    """Name is only a selector within one program/season, never a person merge."""
    c = catalog.copy()
    c["name_key"] = (c.firstName.fillna("")+" "+c.lastName.fillna("")).map(name_key)
    candidates = c.groupby(["season", "team", "name_key"]).roster_row.agg(list)
    resolved, rejected = [], []
    for row in facts:
        matches = candidates.get((row["season"], row["team"], name_key(row["name"])), [])
        if row.get("athlete_id"):
            matches = c[c.roster_row.isin(matches) & c.athlete_id.astype(str).eq(str(row["athlete_id"]))].roster_row.tolist()
        if len(matches) != 1:
            rejected.append({**row, "reason": "ambiguous_or_missing_program_season_name"})
        else:
            resolved.append({**row, "player_ref": int(matches[0])+1})
    return resolved, rejected


def normalize(facts):
    f = pd.DataFrame(facts)
    for col in ["known_at", "effective_at", "valid_until"]:
        f[col] = pd.to_datetime(f[col], utc=True, format="ISO8601")
        if f[col].isna().any(): raise ValueError(f"Missing fact timing: {col}")
    if (f.valid_until <= f.effective_at).any(): raise ValueError("Empty fact interval")
    if not f.kind.isin(["position", "membership", "availability", "depth", "identity"]).all():
        raise ValueError("Unknown fact kind")
    if not f.loc[f.kind.eq("availability"), "value"].isin(STATUSES).all():
        raise ValueError("Unknown availability status")
    if f.source_url.isna().any() or f.source_sha256.isna().any(): raise ValueError("Unattributed fact")
    f["game_id"] = pd.to_numeric(f.get("game_id", pd.Series(index=f.index,dtype=float)), errors="raise")
    # Availability/depth describes the named game, never the entire season.
    if f.loc[f.kind.isin(["availability", "depth"]), "game_id"].isna().any():
        raise ValueError("Availability and depth require a specific game")
    return f.sort_values(["known_at", "effective_at"]).reset_index(drop=True)


def compile_arrays(facts, games, player_refs, output):
    """Write sparse facts once into arrays aligned with every game/roster slot."""
    n = len(player_refs)
    values = np.zeros((n, len(CONTEXT_COLS)), np.float32)
    position = np.full(n, -1, np.int8)
    departed = np.zeros(n, bool)
    quarantined = np.zeros(n, bool)
    games = games.copy(); games["kick"] = pd.to_datetime(games.kick, utc=True)
    kicks = np.zeros(n, np.int64); gids = np.zeros(n, np.int64); seasons = np.zeros(n,np.int16)
    for game in games.itertuples():
        sl=slice(game.offset,game.offset+game.players)
        kicks[sl]=game.kick.value;gids[sl]=game.game_id;seasons[sl]=game.season
    order=np.argsort(player_refs,kind="stable"); sorted_refs=player_refs[order]
    # Reject tied contradictory facts rather than arbitrarily taking the last.
    keys=["player_ref","kind","known_at","game_id"]
    conflicts=facts.groupby(keys,dropna=False).value.transform("nunique").gt(1)
    for row in facts[~conflicts].sort_values("known_at",kind="stable").itertuples():
        left=np.searchsorted(sorted_refs,row.player_ref,side="left")
        right=np.searchsorted(sorted_refs,row.player_ref,side="right")
        ids=order[left:right]
        ids=ids[(kicks[ids]>row.known_at.value)&(kicks[ids]>=row.effective_at.value)
                &(kicks[ids]<row.valid_until.value)&(seasons[ids]==row.season)
                &(True if pd.isna(row.game_id) else gids[ids]==row.game_id)]
        if not len(ids): continue
        kind,value=row.kind,row.value
        if kind=="position":
            pos=POSITIONS.get(POSITION_ALIASES.get(value,value),0)
            if pos:position[ids]=pos;values[ids,8]=1
        elif kind=="availability":
            values[ids,0]=1;values[ids,1:6]=0;values[ids,CONTEXT_COLS.index(value)]=1
        elif kind=="depth": values[ids,6:8]=[1,1/max(1,float(value))]
        elif kind=="membership":
            values[ids,9]=1;departed[ids]=value=="departed";values[ids,10]=departed[ids]
            if value=="transfer_arrived":values[ids,11]=1
            elif value=="departed":values[ids,11]=0
        elif kind=="identity":
            quarantined[ids]=value=="quarantine_history";values[ids,13]=quarantined[ids]
        values[ids,12]=np.log1p((kicks[ids]-row.known_at.value)/86400e9)/7
    for name, a in [("context",values),("position",position),("departed",departed),("quarantined",quarantined)]:
        np.save(output/(name+".npy"),a)
    return dict(game_roster_slots=n, games=int(games.game_id.nunique()),
                slots_with_report=int(values[:,0].sum()), slots_with_depth=int(values[:,6].sum()),
                slots_with_dated_position=int(values[:,8].sum()),
                slots_with_dated_membership=int(values[:,9].sum()),
                departed_slots=int(departed.sum()), identity_quarantined_slots=int(quarantined.sum()),
                missing_reports_are_unknown=True, all_game_roster_slots_represented=True)


def build(source, raw, reviewed, output):
    if (output/"manifest.json").exists(): raise ValueError("Completed timelines are immutable")
    output.mkdir(parents=True, exist_ok=True)
    paths = json.loads(source.read_text()); players = Path(paths["players"])
    catalog = pd.read_parquet(players/"player_seasons.parquet")
    facts, rejected, portal_rows = [], [], []
    from cfbpred.pull_espn_roster import team_alias_map
    aliases = team_alias_map()
    current = next(p for p in raw.glob("roster_*.json") if re.fullmatch(r"roster_\d{4}",p.stem))
    year = int(current.stem.split("_")[-1])
    meta = json.loads(current.with_suffix(".json.source.json").read_text())
    lookup = catalog.set_index(["season","team","athlete_id"])
    common = dict(known_at=meta["observed_at"], effective_at=meta["observed_at"],
                  valid_until=f"{year+1}-03-01T00:00:00Z", source_url=meta["url"],
                  source_sha256=meta["sha256"], game_id=None, season=year)
    for row in json.loads(current.read_text()):
        key = (year, aliases.get(row["team"],row["team"]), str(row["id"]))
        if key not in lookup.index:
            rejected.append(dict(kind="roster",reason="not_in_pinned_catalog",season=year,team=key[1],athlete_id=key[2])); continue
        cat = lookup.loc[key]
        if isinstance(cat,pd.DataFrame): raise ValueError("Ambiguous catalog key")
        if cat.identity_uncertain: continue
        base = {**common,"team":key[1],"player_ref":int(cat.roster_row)+1}
        facts += [{**base,"kind":"position","value":str(row.get("position") or "UNK")},
                  {**base,"kind":"membership","value":"present"}]
    # Match transfer names only if both endpoints corroborate the SAME resolved
    # person; archive all other rows. No ID merge from a portal name.
    c = catalog.copy(); c["name_key"]=(c.firstName.fillna("")+" "+c.lastName.fillna("")).map(name_key)
    groups = {}
    for row in c[["season","team","name_key","person_id","roster_row","identity_uncertain"]].itertuples(index=False):
        groups.setdefault((int(row.season),row.team,row.name_key),[]).append(
            (row.person_id,int(row.roster_row),bool(row.identity_uncertain)))
    for p in sorted(raw.glob("portal_*.json")):
        if not re.fullmatch(r"portal_\d{4}",p.stem): continue
        metadata=json.loads(p.with_suffix(".json.source.json").read_text())
        for row in json.loads(p.read_text()):
            y=int(row["season"]); name=name_key(row["firstName"]+" "+row["lastName"])
            origin=aliases.get(row["origin"],row["origin"]); destination=aliases.get(row["destination"],row["destination"])
            old=groups.get((y-1,origin,name)); new=groups.get((y,destination,name))
            verified=(old is not None and new is not None and len(old)==len(new)==1
                      and old[0][0]==new[0][0] and not old[0][2] and not new[0][2])
            portal_rows.append({**row,"destination_known_at":metadata["observed_at"],
                                "endpoint_identity_verified":verified,
                                "reason":"both_roster_endpoints_same_person" if verified else "unverified_endpoint_identity",
                                "source_url":metadata["url"],"source_sha256":metadata["sha256"]})
            if not verified: continue
            base=dict(season=y,team=destination,known_at=metadata["observed_at"],
                      effective_at=metadata["observed_at"],valid_until=f"{max(year,y)+1}-03-01T00:00:00Z",
                      source_url=metadata["url"],source_sha256=metadata["sha256"],game_id=None,
                      player_ref=new[0][1]+1)
            facts.append({**base,"kind":"membership","value":"transfer_arrived"})
    reviewed_rows=json.loads(reviewed.read_text())
    digests = {}  # one hash per source file, not per row (hundreds of thousands)
    for row in reviewed_rows:
        path = row["source_path"]
        if path not in digests: digests[path] = sha(Path(path))
        if digests[path]!=row["source_sha256"]: raise ValueError("Reviewed source changed")
    linked, bad=resolve_names(reviewed_rows,catalog);facts+=linked;rejected+=bad
    f=normalize(facts);f.to_parquet(output/"facts.parquet",index=False)
    pd.DataFrame(portal_rows).to_parquet(output/"transfers.parquet",index=False)
    save_json(output/"quarantine.json",rejected)
    report=compile_arrays(f,pd.read_parquet(players/"game_rosters.parquet"),np.load(players/"player_refs.npy"),output)
    report.update(facts=len(f),reviewed_facts=len(linked),transfers=len(portal_rows),
                  transfers_identity_verified=sum(r["endpoint_identity_verified"] for r in portal_rows),
                  unresolved_rows=len(rejected))
    save_json(output/"coverage.json",report)
    save_json(output/"manifest.json",dict(version=1,context_cols=CONTEXT_COLS,
                player_manifest_sha256=sha(players/"manifest.json"),
                reviewed_sha256=sha(reviewed),artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(report,indent=2),flush=True)


class TimelineStore(FullCareerStore):
    def __init__(self,players,sequences,full_path,timeline_path,game_ids):
        super().__init__(players,sequences,full_path,game_ids)
        self.timeline_path=Path(timeline_path)
        self.timeline_manifest=json.loads((self.timeline_path/"manifest.json").read_text())
        if self.timeline_manifest["player_manifest_sha256"]!=sha(self.path/"manifest.json"):
            raise ValueError("Timeline roster store differs")
        for name,digest in self.timeline_manifest["artifacts"].items():
            if sha(self.timeline_path/name)!=digest: raise ValueError("Timeline artifact changed")
        self.context_cols=self.timeline_manifest["context_cols"]
        for name in ["context","position","departed","quarantined"]:
            setattr(self,"timeline_"+name,np.load(self.timeline_path/(name+".npy"),mmap_mode="r"))
        self.include_timeline=True

    def timeline_batch(self,result,indices,packed,valid,device):
        result["player_context"]=torch.tensor(self.timeline_context[packed],device=device)
        pos=result["player_position"].clone()
        replacements=torch.tensor(self.timeline_position[packed],device=device,dtype=torch.long)
        slots=result["player_slots"];flat=pos.reshape(-1)
        flat[slots]=torch.where(replacements>=0,replacements,flat[slots])
        result["player_position"]=pos
        departed=torch.zeros_like(result["player_mask"])
        departed.reshape(-1)[slots]=torch.tensor(self.timeline_departed[packed],device=device)
        result["player_mask"] &= ~departed
        out=torch.zeros_like(departed)
        out.reshape(-1)[slots]=result["player_context"][:,CONTEXT_COLS.index("out")].bool()
        result["player_confirmed_out"]=out
        quarantine=torch.tensor(self.timeline_quarantined[packed],device=device)
        result["player_history"][quarantine]=0
        result["player_scout"][quarantine]=0
        if "player_role" in result:
            result["player_role"].reshape(-1,result["player_role"].shape[-1])[slots[quarantine]]=0
        # Disputed IDs retain separate team slots; do not borrow their careers.
        result["career_counts"][quarantine]=0
        return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ["source","raw","reviewed","output"]:p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();build(a.source,a.raw,a.reviewed,a.output)
