from types import SimpleNamespace
import copy

import numpy as np
import pytest
import torch

from cfbpred.player_forecast import PregameMember, checkpoint_inputs, validate_contract
from cfbpred.player_net import PlayerPregameTower


def schema():
    return dict(version=1,static_cols=["rating","known"],history_cols=["yards","observed"],
                positions={"unknown":0,"QB":1},history_windows=["last","two_ago","older","current"])


def test_checkpoint_scaler_vocabulary_and_player_schema_are_enforced():
    d={"scalar_cols":["h_value","a_value"],"G_raw_scalars":np.array([[10.,20.]],np.float32),
       "vocab":{"team":{"New":1,"Old":2},"conf":{"C":1}},
       "player_store":SimpleNamespace(manifest=schema()),
       "G_hidx":torch.tensor([1]),"G_aidx":torch.tensor([2]),"G_opp":torch.tensor([[[1,2]]]),
       "G_hconf":torch.tensor([1]),"G_aconf":torch.tensor([1])}
    ck={"architecture":"players_v1","pregame_only":True,"scalar_cols":d["scalar_cols"],
        "player_schema":schema(),"train_max_season":2025,"forecast_year":2026,
        "mu":np.array([2.,4.]),"sd":np.array([2.,4.]),"vocab":{"team":{"Old":1},"conf":{"C":1}},
        "known_teams":[0,1],"known_confs":[0,1]}
    out=checkpoint_inputs(d,ck)
    assert out["G_scalars"].dtype==torch.float32
    assert torch.equal(out["G_scalars"],torch.tensor([[4.,4.]],dtype=out["G_scalars"].dtype))
    assert out["G_hidx"].item()==0 and out["G_aidx"].item()==1
    bad=copy.deepcopy(ck);bad["player_schema"]["static_cols"].reverse()
    with pytest.raises(ValueError,match="individual-player schema"):validate_contract(bad,d)
    bad=copy.deepcopy(ck);bad["train_max_season"]=2026
    with pytest.raises(ValueError,match="forecast season"):validate_contract(bad,d)
    bad=copy.deepcopy(ck);bad["sd"][0]=0
    with pytest.raises(ValueError,match="positive scales"):validate_contract(bad,d)
    bad=copy.deepcopy(ck);bad["mu"]=np.zeros(3)
    with pytest.raises(ValueError,match="normalization shape"):validate_contract(bad,d)


def test_full_player_tower_order_consistency_and_live_guard():
    torch.manual_seed(3)
    pre=PlayerPregameTower(3,2,["is_home_site","h_value","a_value"],schema()).eval()
    batch=dict(scalars=torch.tensor([[1.,.2,-.3]]),h_idx=torch.tensor([1]),a_idx=torch.tensor([2]),
               h_conf=torch.tensor([1]),a_conf=torch.tensor([1]),
               h_hist=torch.zeros(1,14,7),a_hist=torch.zeros(1,14,7),
               h_opp=torch.zeros(1,14,dtype=torch.long),a_opp=torch.zeros(1,14,dtype=torch.long),
               player_scout=torch.randn(1,2,5,2),player_history=torch.randn(1,2,5,4,2),
               player_position=torch.randint(0,2,(1,2,5)),player_mask=torch.ones(1,2,5,dtype=torch.bool))
    _,m,t=pre.forward_batch(batch)
    swap=batch.copy();swap["scalars"]=torch.tensor([[-1.,-.3,.2]])
    for h,a in [("h_idx","a_idx"),("h_conf","a_conf"),("h_hist","a_hist"),("h_opp","a_opp")]:
        swap[h],swap[a]=batch[a],batch[h]
    for key in ["player_scout","player_history","player_position","player_mask"]:
        swap[key]=batch[key].flip(1)
    _,rm,rt=pre.forward_batch(swap)
    assert torch.allclose(m,rm.flip(-1),atol=1e-6)
    assert torch.allclose(t,rt,atol=1e-6)
    assert torch.isfinite(m).all()
    empty=batch.copy();empty.pop("player_history")
    with pytest.raises(ValueError,match="requires player_history"):pre.forward_batch(empty)
    with pytest.raises(RuntimeError,match="pregame-only"):PregameMember(pre)(batch)
