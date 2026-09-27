"""Numerically identical sequence batches without recomputing padded histories."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import torch

from cfbpred.player_inputs import STAT_COLS
from cfbpred.player_sequence_inputs import SequenceStore,ROLE_STATS,WINDOW


class FastSequenceStore(SequenceStore):
    def batch(self,indices,device):
        refs,valid=self.roster_refs(indices);packed_refs=refs[valid]
        p=self.player_refs[packed_refs];c=self.current_refs[packed_refs]
        history=np.concatenate([self.prior_history[p],self.current_history[c][:,None,:]],axis=1)
        position=np.zeros(valid.shape,np.int64);position[valid]=self.position[p]
        slots=np.flatnonzero(valid)
        result={"player_scout":torch.tensor(self.scouting[p],device=device),"player_history":torch.tensor(history,device=device),
                "player_position":torch.tensor(position,device=device),"player_mask":torch.tensor(valid,device=device),
                "player_slots":torch.tensor(slots,device=device)}
        if not (self.include_context or self.include_role):return result
        tips=self.history_tips[packed_refs]
        if not hasattr(self,"roster_team"):
            vocab=json.loads((self.sequence_path/"team_vocab.json").read_text())["team"]
            cat=pd.read_parquet(self.path/"player_seasons.parquet",columns=["team"])
            self.roster_team=np.r_[0,cat.team.map(vocab).fillna(0).to_numpy(np.int64)]
        team=self.roster_team[p]
        kick=np.broadcast_to(self.kicks[indices][...,None],valid.shape)[valid]
        year=np.broadcast_to(self.seasons[indices][...,None],valid.shape)[valid]
        if self.include_context:
            has=tips>0;seq=self.sequence_refs[tips[has]];present=seq>0
            days=np.where(present,(kick[has,None]-self.event_kick[seq])/86400e9,0)
            if (days[present]<=0).any():raise ValueError("Player event is not strictly before target kickoff")
            same_year=self.event_season[seq]==year[has,None]
            same_team=(self.event_team[seq]==team[has,None])&present&(team[has,None]>0)
            unique,inverse=np.unique(seq,return_inverse=True)
            event_team,event_opp=self.event_team[unique],self.event_opponent[unique]
            if self.known_teams is None:raise ValueError("Training-known history vocabulary required")
            event_team=np.where(np.isin(event_team,self.known_teams),event_team,0)
            event_opp=np.where(np.isin(event_opp,self.known_teams),event_opp,0)
            temporal=np.stack([np.log1p(days)/7,same_year,same_team,np.broadcast_to(np.arange(WINDOW)/WINDOW,seq.shape)],-1).astype(np.float32)
            result.update(sequence_values=torch.tensor(self.events[unique],device=device),
                          sequence_team=torch.tensor(event_team,device=device),sequence_opponent=torch.tensor(event_opp,device=device),
                          sequence_refs=torch.tensor(inverse.reshape(-1,WINDOW),device=device),sequence_mask=torch.tensor(present,device=device),
                          sequence_temporal=torch.tensor(temporal,device=device),sequence_slots=torch.tensor(slots[has],device=device))
        if self.include_role:
            # A tip's raw last/recent/previous statistics are shared by all
            # its pregame uses. Compute once per distinct tip in this batch.
            unique,inverse=np.unique(tips,return_inverse=True)
            seq=self.sequence_refs[unique];stat_indices=[STAT_COLS.index(s) for s in ROLE_STATS]
            raw=self.events[seq[...,None],stat_indices];observed=self.target_mask[seq[...,None],stat_indices]
            parts=[]
            for window in [slice(0,1),slice(0,3),slice(3,6)]:
                values,mask=raw[:,window,:],observed[:,window,:];count=mask.sum(-2)
                parts.extend([(values*mask).sum(-2)/np.maximum(count,1),count/(window.stop-window.start)])
            shared=np.concatenate(parts,-1)[inverse]
            days=np.where(tips>0,(kick-self.event_kick[tips])/86400e9,0)
            if (days[tips>0]<=0).any():raise ValueError("Player event is not strictly before target kickoff")
            same_team=(self.event_team[tips]==team)&(tips>0)&(team>0)
            same_year=self.event_season[tips]==year
            packed=np.column_stack([shared,np.log1p(days)/7,same_team,same_year]).astype(np.float32)
            role=np.zeros((*valid.shape,packed.shape[-1]),np.float32);role[valid]=packed
            result["player_role"]=torch.tensor(role,device=device)
        return result
