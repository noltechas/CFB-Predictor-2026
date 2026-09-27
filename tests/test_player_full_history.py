import numpy as np
import pandas as pd
import pytest
import torch

from cfbpred.player_full_history import career_starts,ragged_careers,read_extra_boxes,EXTRA_COLS
from cfbpred.player_full_history_net import CareerEncoder


def test_full_career_keeps_old_events_across_transfers_and_handles_empty():
    events=pd.DataFrame(dict(event_id=np.arange(1,71),person_id=['one']*67+['two']*3))
    starts=career_starts(events)
    ids,owner,rank,counts=ragged_careers(np.array([0,66,70]),starts)
    assert counts.tolist()==[0,66,3]
    assert ids[owner==1].tolist()==list(range(66,0,-1))
    assert ids[owner==2].tolist()==[70,69,68]
    assert rank[owner==1][-1]==65
    assert 67 not in ids  # future event for that person is not included


def test_interleaved_person_events_are_rejected():
    events=pd.DataFrame(dict(event_id=[1,2,3],person_id=['a','b','a']))
    with pytest.raises(ValueError,match='contiguous'):career_starts(events)


def test_extra_stats_preserve_conflicts_missingness_and_all_observed_fields(tmp_path):
    rows=[dict(game_id=1,home_away='home',athlete_id='a',category=c,stat_type=t,stat=v)
          for c,t,v in [('defensive','QB HUR','2'),('defensive','QB HUR','2'),
                        ('rushing','LONG','20'),('rushing','LONG','30'),('kicking','TOT','1/2')]]
    p=tmp_path/'raw.parquet';pd.DataFrame(rows).to_parquet(p)
    d=read_extra_boxes([p]).iloc[0]
    assert d.qb_hurries==2 and np.isnan(d.rushing_long)
    assert d.kick_tot_first==1 and d.kick_tot_second==2
    assert np.isnan(d.punt_inside20)


def test_new_stat_requires_explicit_review_instead_of_silent_omission(tmp_path):
    p=tmp_path/'raw.parquet'
    pd.DataFrame([dict(game_id=1,home_away='home',athlete_id='a',category='passing',stat_type='QBR',stat='90')]).to_parquet(p)
    with pytest.raises(ValueError,match='explicit parser'):read_extra_boxes([p])


def career_batch(values,owner):
    counts=torch.bincount(owner,minlength=3)
    latest=torch.zeros(len(owner),dtype=torch.bool)
    for x in owner.unique():latest[torch.where(owner==x)[0][0]]=True
    return dict(career_values=values,career_team=torch.ones(len(values),dtype=torch.long),
                career_opponent=torch.full((len(values),),2,dtype=torch.long),
                career_refs=torch.arange(len(values)),career_owner=owner,
                career_temporal=torch.zeros(len(values),4),career_is_latest=latest,career_counts=counts)


def test_oldest_game_has_gradient_and_ragged_attention_matches_individual_encoding():
    torch.manual_seed(4);model=CareerEncoder(4,10).eval()
    values=torch.randn(73,10,requires_grad=True);owner=torch.tensor([0]*70+[2]*3)
    batch=career_batch(values,owner);out=model(batch)
    assert torch.equal(out[1],torch.zeros(12))
    separate=career_batch(values[:70],torch.zeros(70,dtype=torch.long))
    assert torch.allclose(out[0],model(separate)[0],atol=1e-6)
    out[0].sum().backward()
    assert values.grad[69].abs().sum()>0  # the 70th-most-recent game is used
    assert values.grad[70:].abs().sum()==0
    changed=values.detach().clone();changed[69]+=10
    assert not torch.allclose(out[0],model(career_batch(changed,owner))[0])
