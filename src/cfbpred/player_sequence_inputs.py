"""Immutable raw player game histories and separately accessed training labels.

Extends, rather than replaces, the audited v3 player store. Every roster
member remains present; the last eight observed individual games supplement
the existing full-career summaries. Missing boxes never imply nonparticipation.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cfbpred.player_inputs import HISTORY_COLS, STAT_COLS, PlayerStore, transform_history
from cfbpred.pregame_experiments import save_json, sha
from cfbpred.pull_espn_roster import team_alias_map

WINDOW = 8
ROLE_STATS = ["att", "rush_att", "rec", "tackles", "fga", "punts", "kr", "pr"]
EVENT_COLS = HISTORY_COLS + ["was_home", "opponent_fbs", "neutral_site", "points_for", "points_against"]
ROLE_COLS = [f"{window}_{stat}_{field}" for window in ["last", "recent3", "previous3"]
             for field in ["value", "known_fraction"] for stat in ROLE_STATS] + ["days_since_box", "same_program", "same_season"]


def person_sequences(boxes, refs, window=WINDOW):
    """Strict as-of join across seasons/programs; ties excluded at kickoff."""
    boxes = boxes.sort_values(["person_id", "kick", "game_id", "team"]).reset_index(drop=True).copy()
    boxes["event_id"] = np.arange(1, len(boxes)+1)
    sequences = np.zeros((len(boxes)+1, window), np.int32)
    tips = np.zeros(len(refs), np.int32)
    ref_groups = refs.groupby("person_id", sort=False).indices
    ref_times = pd.to_datetime(refs.kick, utc=True).astype("int64").to_numpy()
    for person, indices in boxes.groupby("person_id", sort=False).indices.items():
        ids = boxes.event_id.to_numpy()[indices]
        rows = np.arange(len(ids))[:, None] - np.arange(window)[None, :]
        sequences[ids] = np.where(rows >= 0, ids[np.maximum(rows, 0)], 0)
        r = ref_groups.get(person)
        if r is not None:
            times = pd.to_datetime(boxes.kick.iloc[indices], utc=True).astype("int64").to_numpy()
            before = np.searchsorted(times, ref_times[r], side="left")-1
            tips[r] = np.where(before >= 0, ids[np.maximum(before, 0)], 0)
    return boxes, sequences, tips


def build(base_path, inputs, output):
    output.mkdir(parents=True, exist_ok=True)
    if (output/"manifest.json").exists():
        raise ValueError("Completed sequence store is immutable; use a new directory")
    catalog = pd.read_parquet(base_path/"player_seasons.parquet")
    rosters = pd.read_parquet(base_path/"game_rosters.parquet").sort_values("offset")
    player_refs = np.load(base_path/"player_refs.npy")
    repeated = rosters.loc[rosters.index.repeat(rosters.players)].reset_index(drop=True)
    if len(repeated) != len(player_refs):
        raise ValueError("Roster/reference count mismatch")
    refs = repeated[["game_id", "season", "team", "kick"]].copy()
    refs["person_id"] = catalog.person_id.to_numpy()[player_refs-1]
    boxes, sequences, tips = person_sequences(pd.read_parquet(base_path/"clean_player_boxes.parquet"), refs)
    schedule = pd.read_parquet(inputs/"snapshot_schedule.parquet").set_index("game_id")
    aliases=team_alias_map()
    for side in ["home","away"]:schedule[side+"_team"]=schedule[side+"_team"].replace(aliases)
    game = schedule.loc[boxes.game_id].reset_index()
    home = boxes.team.to_numpy() == game.home_team.to_numpy()
    away = boxes.team.to_numpy() == game.away_team.to_numpy()
    if not (home | away).all():
        raise ValueError("Player program missing from source schedule")
    opponent = np.where(home, game.away_team, game.home_team)
    original_vocab = json.loads((inputs/"team_vocab.json").read_text())["team"]
    vocab={}
    for name,index in original_vocab.items():
        canonical=aliases.get(name,name)
        vocab[canonical]=original_vocab.get(canonical,min(index,vocab.get(canonical,index)))
    team = boxes.team.map(vocab).fillna(0).to_numpy(np.int64)
    opp = pd.Series(opponent).map(vocab).fillna(0).to_numpy(np.int64)
    values = np.column_stack([transform_history(boxes[HISTORY_COLS].to_numpy()), home,
                              np.where(home, game.away_division, game.home_division)=="fbs",
                              game.neutral_site.fillna(False).to_numpy(bool),
                              np.where(home, game.home_points, game.away_points)/50.,
                              np.where(home, game.away_points, game.home_points)/50.]).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite completed-game input")
    labels = refs.reset_index(names="ref_id").merge(
        boxes[["game_id", "team", "person_id", "event_id"]],
        on=["game_id", "team", "person_id"], how="left", validate="many_to_one").sort_values("ref_id")
    event_time = np.r_[0, pd.to_datetime(boxes.kick, utc=True).astype("int64").to_numpy()]
    time = pd.to_datetime(refs.kick, utc=True).astype("int64").to_numpy()
    if np.any((tips>0) & (event_time[tips]>=time)):
        raise ValueError("Target/future event entered player history")
    arrays = {"events":np.vstack([np.zeros((1, len(EVENT_COLS)), np.float32), values]),
              "event_team":np.r_[0,team], "event_opponent":np.r_[0,opp],
              "event_kick":event_time, "event_season":np.r_[0,boxes.season.to_numpy(np.int16)],
              "sequence_refs":sequences, "history_tips":tips,
              "label_refs":labels.event_id.fillna(0).to_numpy(np.int32)}
    # Lossless compression avoids duplicating raw labels already present in
    # event columns and reduces storage for sparse observations and padding.
    np.savez_compressed(output/"arrays.npz",**arrays)
    save_json(output/"team_vocab.json",{"team":vocab})
    boxes[["event_id","game_id","person_id","team","season","kick"]].assign(opponent=opponent).to_parquet(output/"events.parquet",index=False)
    future = refs.season.eq(2026).to_numpy()
    old_program = (tips>0) & (np.concatenate([[""],boxes.team.to_numpy()])[tips] != refs.team.to_numpy())
    report = {"events":len(boxes), "roster_references":len(refs), "references_with_history":int((tips>0).sum()),
              "2026_references_with_history":int(((tips>0)&future).sum()),
              "2026_last_observation_at_other_program":int((old_program&future).sum()),
              "strict_history_before_kickoff":True, "training_labels_separate":True,
              "limitations":["Last eight observed box games supplement full-career aggregate windows; this is not complete play-by-play or snap history",
                             "Recent contribution measures reported passing attempts, carries, catches, tackles and special-teams counts; missing rows do not establish zero usage",
                             "No invented historical injuries, starters, targets, blocking grades or full snap participation",
                             "Annual roster/recruit publication-time limitations are inherited from the base store"]}
    save_json(output/"coverage_report.json",report)
    save_json(output/"manifest.json", {"version":1,"window":WINDOW,"event_cols":EVENT_COLS,"role_cols":ROLE_COLS,
              "target_cols":STAT_COLS,"base_manifest_sha256":sha(base_path/"manifest.json"),
              "schedule_sha256":sha(inputs/"snapshot_schedule.parquet"),"vocab_sha256":sha(inputs/"team_vocab.json"),
              "artifacts":{p.name:sha(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps(report,indent=2),flush=True)


class SequenceStore(PlayerStore):
    def __init__(self, base_path, sequence_path, game_ids):
        super().__init__(base_path, game_ids)
        self.sequence_path = Path(sequence_path)
        self.sequence_manifest = json.loads((self.sequence_path/"manifest.json").read_text())
        if self.sequence_manifest["base_manifest_sha256"] != sha(self.path/"manifest.json"):
            raise ValueError("Sequence/base stores differ")
        for name,digest in self.sequence_manifest["artifacts"].items():
            if sha(self.sequence_path/name)!=digest:
                raise ValueError(f"Sequence artifact changed: {name}")
        with np.load(self.sequence_path/"arrays.npz") as arrays:
            for name in arrays.files:setattr(self,name,arrays[name])
        self.targets=self.events[:,:len(STAT_COLS)]
        self.target_mask=self.events[:,len(STAT_COLS):2*len(STAT_COLS)]>0
        order = pd.MultiIndex.from_product([game_ids,["home","away"]])
        g = pd.read_parquet(self.path/"game_rosters.parquet").set_index(["game_id","side"]).loc[order]
        self.kicks = pd.to_datetime(g.kick,utc=True).astype("int64").to_numpy().reshape(-1,2)
        self.seasons = g.season.to_numpy().reshape(-1,2)
        self.game_ids = np.asarray(game_ids)
        self.known_teams = None
        self.include_context = self.include_role = True

    def for_fold(self, train_indices, context, role):
        other = copy.copy(self)
        other.include_context,other.include_role = context,role
        if not context:
            other.known_teams=np.array([0],np.int64)
            return other
        # Only programs actually observed in the training history get trained
        # identities. No validation/future-only random embeddings.
        offsets = self.offsets[train_indices].ravel(); counts = self.counts[train_indices].ravel()
        refs = np.concatenate([np.arange(o,o+c) for o,c in zip(offsets,counts)])
        events = np.unique(self.sequence_refs[self.history_tips[refs]])
        other.known_teams = np.unique(np.r_[self.event_team[events],self.event_opponent[events],0])
        return other

    def roster_refs(self, indices):
        indices = np.asarray(indices)
        n = int(max(1,self.counts[indices].max()))
        valid = np.arange(n)[None,None,:] < self.counts[indices][...,None]
        refs = np.where(valid,self.offsets[indices][...,None]+np.arange(n),0)
        return refs,valid

    def labels(self, indices, device):
        indices = np.asarray(indices)
        if (self.seasons[indices]>=2026).any():
            raise ValueError("2026 player outcomes cannot enter fitting or selection")
        refs,valid = self.roster_refs(indices)
        ids = np.where(valid,self.label_refs[refs],0)
        return (torch.tensor(self.targets[ids],device=device),
                torch.tensor(self.target_mask[ids]&valid[...,None],device=device))

    def batch(self, indices, device):
        refs,valid = self.roster_refs(indices)
        packed_refs=refs[valid]
        p=self.player_refs[packed_refs];c=self.current_refs[packed_refs]
        history=np.concatenate([self.prior_history[p],self.current_history[c][:,None,:]],axis=1)
        position=np.zeros(valid.shape,np.int64);position[valid]=self.position[p]
        result={"player_scout":torch.tensor(self.scouting[p],device=device),
                "player_history":torch.tensor(history,device=device),
                "player_position":torch.tensor(position,device=device),"player_mask":torch.tensor(valid,device=device),
                "player_slots":torch.tensor(np.flatnonzero(valid),device=device)}
        if not (self.include_context or self.include_role):return result
        tips = np.where(valid,self.history_tips[refs],0)
        seq = self.sequence_refs[tips]
        event_valid = seq>0
        days = np.where(event_valid,(self.kicks[indices][...,None,None]-self.event_kick[seq])/86400e9,0)
        if (days[event_valid]<=0).any():
            raise ValueError("Player event is not strictly before target kickoff")
        same_year = self.event_season[seq] == self.seasons[indices][...,None,None]
        # Same-program flag uses names via the catalog's current team mapping,
        # which is also the game side's team. Vocabulary indices are raw IDs.
        if not hasattr(self,"roster_team"):
            vocab_path = self.sequence_path/"team_vocab.json"
            # Construct once from current raw person-season rows, using the
            # persisted vocabulary copied by the experiment preparation.
            vocab = json.loads(vocab_path.read_text())["team"]
            cat = pd.read_parquet(self.path/"player_seasons.parquet",columns=["team"])
            self.roster_team = np.r_[0,cat.team.map(vocab).fillna(0).to_numpy(np.int64)]
        current_team = self.roster_team[np.where(valid,self.player_refs[refs],0)]
        same_team = (self.event_team[seq] == current_team[...,None]) & event_valid & (current_team[...,None]>0)
        if self.include_context:
            # Encode only players with observed history and only unique box
            # rows in this batch, rather than millions of padded game slots.
            slots = np.flatnonzero(tips.ravel()>0)
            selected = seq.reshape(-1,WINDOW)[slots]
            unique,inverse = np.unique(selected,return_inverse=True)
            team,opp = self.event_team[unique],self.event_opponent[unique]
            if self.known_teams is None:
                raise ValueError("Training-known history vocabulary required")
            team = np.where(np.isin(team,self.known_teams),team,0)
            opp = np.where(np.isin(opp,self.known_teams),opp,0)
            temporal = np.stack([np.log1p(days)/7,same_year,same_team,
                                 np.broadcast_to(np.arange(WINDOW)/WINDOW,seq.shape)],-1).astype(np.float32)
            result.update(sequence_values=torch.tensor(self.events[unique],device=device),
                          sequence_team=torch.tensor(team,device=device),sequence_opponent=torch.tensor(opp,device=device),
                          sequence_refs=torch.tensor(inverse.reshape(-1,WINDOW),device=device),
                          sequence_mask=torch.tensor(selected>0,device=device),
                          sequence_temporal=torch.tensor(temporal.reshape(-1,WINDOW,4)[slots],device=device),
                          sequence_slots=torch.tensor(slots,device=device))
        if self.include_role:
            stat_indices = [STAT_COLS.index(s) for s in ROLE_STATS]
            raw = self.events[seq[...,None],stat_indices]
            observed = self.target_mask[seq[...,None],stat_indices]
            parts=[]
            for window in [slice(0,1),slice(0,3),slice(3,6)]:
                values,mask = raw[...,window,:],observed[...,window,:]
                count = mask.sum(-2)
                parts.extend([(values*mask).sum(-2)/np.maximum(count,1),count/(window.stop-window.start)])
            parts.extend([np.log1p(days[...,0,None])/7,same_team[...,0,None],same_year[...,0,None]])
            role=np.concatenate(parts,-1).astype(np.float32)*valid[...,None]
            result["player_role"] = torch.tensor(role,device=device)
        return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--base",type=Path,required=True);p.add_argument("--inputs",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();build(a.base,a.inputs,a.output)
