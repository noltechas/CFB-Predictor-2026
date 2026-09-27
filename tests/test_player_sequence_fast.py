import copy
import numpy as np
import torch

from cfbpred.player_inputs import HISTORY_COLS,STATIC_COLS,STAT_COLS
from cfbpred.player_sequence_inputs import SequenceStore,EVENT_COLS,WINDOW
from cfbpred.player_sequence_fast_inputs import FastSequenceStore


def test_fast_batches_are_identical_with_padding_missing_histories_and_transfers():
    rng=np.random.RandomState(2);s=SequenceStore.__new__(SequenceStore)
    s.offsets=np.array([[0,3],[4,4],[6,9]]);s.counts=np.array([[3,1],[0,2],[3,1]])
    s.player_refs=np.array([1,2,3,4,1,2,1,2,3,4]);s.current_refs=np.array([1,0,2,1,1,2,2,0,2,1])
    s.scouting=rng.normal(size=(5,len(STATIC_COLS))).astype(np.float32)
    s.prior_history=rng.normal(size=(5,3,len(HISTORY_COLS))).astype(np.float32)
    s.current_history=rng.normal(size=(3,len(HISTORY_COLS))).astype(np.float32)
    s.position=np.array([0,1,2,3,4]);s.events=rng.normal(size=(3,len(EVENT_COLS))).astype(np.float32);s.events[0]=0
    s.history_tips=np.array([1,0,2,1,1,2,2,0,2,1]);s.target_mask=rng.randint(0,2,size=(3,len(STAT_COLS))).astype(bool);s.target_mask[0]=False
    s.sequence_refs=np.zeros((3,WINDOW),np.int32);s.sequence_refs[1,0]=1;s.sequence_refs[2,:2]=[2,1]
    s.event_kick=np.array([0,1,2],np.int64)*86400*10**9;s.kicks=np.array([[3,3],[4,4],[5,5]],np.int64)*86400*10**9
    s.event_season=np.array([0,2024,2025]);s.seasons=np.array([[2025,2025],[2025,2025],[2026,2026]])
    s.event_team=np.array([0,1,2]);s.event_opponent=np.array([0,2,1]);s.roster_team=np.array([0,2,1,2,1]);s.known_teams=np.array([0,1,2])
    for context,role in [(False,False),(True,False),(False,True),(True,True)]:
        s.include_context,s.include_role=context,role;fast=copy.copy(s);fast.__class__=FastSequenceStore
        for indices in [[0,1,2],[1]]:
            a,b=s.batch(indices,"cpu"),fast.batch(indices,"cpu")
            assert a.keys()==b.keys()
            for key in a:assert torch.equal(a[key],b[key]),key
