import copy
import numpy as np
import pandas as pd
import pytest
import torch

from cfbpred.player_inputs import HISTORY_COLS,STATIC_COLS,POSITIONS,STAT_COLS
from cfbpred.player_sequence_inputs import build,person_sequences,SequenceStore,WINDOW,ROLE_COLS,EVENT_COLS
from cfbpred.player_sequence_net import RichPlayerEncoder,RichPlayerTower,masked_stat_loss


def test_cross_program_sequences_exclude_target_and_simultaneous_games():
    kicks=pd.to_datetime(["2024-01-01","2025-01-01","2025-01-01","2025-02-01"],utc=True)
    boxes=pd.DataFrame(dict(person_id=["transfer"]*4,kick=kicks,game_id=[1,2,3,4],team=["Old","New","New","New"]))
    refs=pd.DataFrame(dict(person_id=["transfer"]*5+["unseen"],kick=[kicks[0],kicks[1],kicks[1]+pd.Timedelta(days=1),kicks[3],kicks[3]+pd.Timedelta(days=7),kicks[3]]))
    b,seq,tips=person_sequences(boxes,refs)
    assert tips.tolist()==[0,1,3,3,4,0]
    assert seq[3,:4].tolist()==[3,2,1,0]
    assert b.team.iloc[seq[tips[3],2]-1]=="Old"
    assert (b.set_index("event_id").kick.loc[tips[1:5]].to_numpy()<refs.kick.iloc[1:5].to_numpy()).all()


def test_compressed_builder_preserves_values_masks_and_transfer_context(tmp_path):
    base=tmp_path/"base";inputs=tmp_path/"inputs";output=tmp_path/"out"
    base.mkdir();inputs.mkdir();(base/"manifest.json").write_text('{}')
    pd.DataFrame(dict(person_id=["p"],team=["New"])).to_parquet(base/"player_seasons.parquet")
    kicks=pd.to_datetime(["2024-09-01","2025-09-01","2026-09-01"],utc=True)
    pd.DataFrame(dict(game_id=[1,2,3],season=[2024,2025,2026],side=["home"]*3,team=["Old","New","New"],
                      kick=kicks,offset=[0,1,2],players=[1]*3)).to_parquet(base/"game_rosters.parquet")
    np.save(base/"player_refs.npy",np.array([1,1,1]))
    rows=[]
    for i in range(2):
        row=dict(game_id=i+1,season=2024+i,team=["Old","New"][i],person_id="p",kick=kicks[i])
        row.update({c:0. for c in HISTORY_COLS});row.update(att=10.,att_observations=1.,box_games=1.,fbs_box_games=1.)
        rows.append(row)
    pd.DataFrame(rows).to_parquet(base/"clean_player_boxes.parquet")
    pd.DataFrame(dict(game_id=[1,2,3],home_team=["Old","New","New"],away_team=["Opponent"]*3,
                     home_division=["fbs"]*3,away_division=["fbs"]*3,neutral_site=[False]*3,
                     home_points=[21.,28.,np.nan],away_points=[14.,7.,np.nan])).to_parquet(inputs/"snapshot_schedule.parquet")
    (inputs/"team_vocab.json").write_text('{"team":{"Old":1,"New":2,"Opponent":3}}')
    build(base,inputs,output)
    with np.load(output/"arrays.npz") as a:
        assert a["history_tips"].tolist()==[0,1,2]
        assert a["event_team"].tolist()==[0,1,2] and a["event_opponent"].tolist()==[0,3,3]
        assert a["label_refs"].tolist()==[1,2,0]
        assert a["events"][1,STAT_COLS.index("att")]==pytest.approx(np.log1p(10)/5)
        assert a["events"][1,len(STAT_COLS)+STAT_COLS.index("att")]>0
        assert a["events"][1,len(STAT_COLS)+STAT_COLS.index("rec")]==0
    with pytest.raises(ValueError,match="immutable"):build(base,inputs,output)


def test_future_player_labels_rejected_and_padding_never_observed():
    store=SequenceStore.__new__(SequenceStore)
    store.seasons=np.array([[2025,2025],[2026,2026]])
    store.offsets=np.array([[0,2],[3,4]]);store.counts=np.array([[2,1],[1,1]])
    store.label_refs=np.array([1,0,2,0,0]);store.targets=np.arange(81,dtype=np.float32).reshape(3,27)
    store.target_mask=np.ones((3,27),bool);store.target_mask[0]=False
    values,mask=store.labels([0],"cpu")
    assert mask[0,0,0].all() and not mask[0,0,1].any() and not mask[0,1,1].any()
    with pytest.raises(ValueError,match="2026"):store.labels([1],"cpu")


def test_auxiliary_missing_labels_have_zero_gradient():
    pred=torch.ones(2,2,5,27,requires_grad=True)
    target=torch.zeros_like(pred);mask=torch.zeros_like(pred,dtype=torch.bool)
    mask[0,0,0,0]=True
    loss=masked_stat_loss(pred,target,mask);loss.backward()
    assert pred.grad[0,0,0,0]>0 and torch.count_nonzero(pred.grad)==1
    assert masked_stat_loss(pred,target,torch.zeros_like(mask)).item()==0


def example(players=5):
    shape=(1,2,players);slots=2*players
    return dict(player_scout=torch.randn(*shape,len(STATIC_COLS)),player_history=torch.randn(*shape,4,len(HISTORY_COLS)),
                player_position=torch.randint(0,len(POSITIONS),shape),player_mask=torch.ones(shape,dtype=torch.bool),
                player_role=torch.randn(*shape,len(ROLE_COLS)),sequence_values=torch.randn(slots,len(EVENT_COLS)),
                sequence_team=torch.ones(slots,dtype=torch.long),sequence_opponent=torch.full((slots,),2,dtype=torch.long),
                sequence_refs=torch.arange(slots)[:,None].expand(-1,WINDOW),sequence_mask=torch.ones(slots,WINDOW,dtype=torch.bool),
                sequence_temporal=torch.randn(slots,WINDOW,4),sequence_slots=torch.arange(slots))


def test_context_and_contribution_preserve_roster_permutation_and_empty_padding():
    torch.manual_seed(8)
    schema=dict(static_cols=STATIC_COLS,history_cols=HISTORY_COLS,positions=POSITIONS)
    encoder=RichPlayerEncoder(schema,4,True,True).eval();b=example(200)
    original,_=encoder.forward_batch(b)
    perm=torch.randperm(200);c=copy.copy(b)
    for key in ["player_scout","player_history","player_position","player_mask","player_role"]:c[key]=b[key][:,:,perm]
    order=torch.arange(400).reshape(1,2,200)[:,:,perm].flatten()
    for key in ["sequence_refs","sequence_mask","sequence_temporal"]:c[key]=b[key][order]
    shuffled,_=encoder.forward_batch(c)
    assert torch.allclose(original,shuffled,atol=1e-6)
    c["player_mask"]=torch.zeros_like(c["player_mask"])
    empty,_=encoder.forward_batch(c);assert torch.equal(empty,torch.zeros_like(empty))
    b["player_scout"].requires_grad_();encoder.forward_batch(b)[0].sum().backward()
    assert b["player_scout"].grad[:,:,-1].abs().sum()>0
    assert encoder.sequence.event[0].weight.grad.abs().sum()>0
    assert encoder.role[0].weight.grad.abs().sum()>0


def test_packed_rosters_match_padded_encoding_without_dropping_players():
    torch.manual_seed(1)
    schema=dict(static_cols=STATIC_COLS,history_cols=HISTORY_COLS,positions=POSITIONS)
    encoder=RichPlayerEncoder(schema,4,True,True).eval();b=example(20)
    b["player_mask"][:,:,3:6]=False
    padded,_=encoder.forward_batch(b)
    packed=b.copy();valid=b["player_mask"]
    for key in ["player_scout","player_history"]:packed[key]=b[key][valid]
    packed["player_slots"]=torch.flatnonzero(valid.reshape(-1)) if hasattr(torch,"flatnonzero") else torch.nonzero(valid.reshape(-1)).flatten()
    result,_=encoder.forward_batch(packed)
    assert torch.allclose(padded,result,atol=1e-6)


def test_auxiliary_head_uses_inputs_only_and_is_off_during_inference():
    torch.manual_seed(9)
    schema=dict(static_cols=STATIC_COLS,history_cols=HISTORY_COLS,positions=POSITIONS)
    pre=RichPlayerTower(4,2,["is_home_site","h_value","a_value"],schema,True,True,True)
    b=example();b.update(scalars=torch.tensor([[1.,.2,-.3]]),h_idx=torch.tensor([1]),a_idx=torch.tensor([2]),
             h_conf=torch.tensor([1]),a_conf=torch.tensor([1]),h_hist=torch.zeros(1,14,7),a_hist=torch.zeros(1,14,7),
             h_opp=torch.zeros(1,14,dtype=torch.long),a_opp=torch.zeros(1,14,dtype=torch.long),
             h_gp01=torch.tensor([.1]),a_gp01=torch.tensor([.2]))
    pre.train();pre.forward_batch(b)
    assert pre.aux_predictions.shape==(1,2,5,len(STAT_COLS))
    loss=masked_stat_loss(pre.aux_predictions,torch.ones_like(pre.aux_predictions),torch.ones_like(pre.aux_predictions,dtype=torch.bool))
    loss.backward();assert pre.encoder.player[0].weight.grad.abs().sum()>0
    pre.eval();_,m,t=pre.forward_batch(b);assert pre.aux_predictions is None
    swap=b.copy();swap["scalars"]=torch.tensor([[-1.,-.3,.2]])
    for h,a in [("h_idx","a_idx"),("h_conf","a_conf"),("h_hist","a_hist"),("h_opp","a_opp"),("h_gp01","a_gp01")]:swap[h],swap[a]=b[a],b[h]
    for key in ["player_scout","player_history","player_position","player_mask","player_role"]:swap[key]=b[key].flip(1)
    order=torch.arange(10).reshape(1,2,5).flip(1).flatten()
    for key in ["sequence_refs","sequence_mask","sequence_temporal"]:swap[key]=b[key][order]
    _,rm,rt=pre.forward_batch(swap)
    assert torch.allclose(m,rm.flip(-1),atol=1e-6) and torch.allclose(t,rt,atol=1e-6)


def test_readmitted_game_columns_have_exact_swap_rules():
    from cfbpred.consistent_net import ANTISYMMETRIC, swap_schema
    cols = ["h_x", "a_x", "is_home_site", "h2h_last_margin", "h2h_days_since", "h2h_meetings_5y",
            "kick_local_hour", "venue_grass", "venue_capacity", "h_body_clock_shift", "a_body_clock_shift"]
    perm, sign = swap_schema(cols)
    assert [cols[i] for i in perm][:2] == ["a_x", "h_x"]
    for col in ("h2h_days_since", "h2h_meetings_5y", "kick_local_hour", "venue_grass", "venue_capacity"):
        k = cols.index(col); assert perm[k] == k and sign[k] == 1.0
    for col in ANTISYMMETRIC:
        k = cols.index(col); assert perm[k] == k and sign[k] == -1.0
    k = cols.index("h_body_clock_shift"); assert cols[perm[k]] == "a_body_clock_shift"


def test_family_gates_start_equal_to_the_shared_gate():
    import torch
    from cfbpred.raw_net import RawPregameTower
    cols = ["h_o_ypp", "h_d_ypp", "h_bx_pass", "h_qb_ypa", "h_winpct", "a_o_ypp", "a_d_ypp", "a_bx_pass", "a_qb_ypa", "a_winpct"]
    t = RawPregameTower(5, 3, len(cols), s2d_idx_h=list(range(5)), s2d_idx_a=list(range(5, 10)), use_s2d_gate=True)
    x = torch.randn(4, len(cols)); gp = torch.rand(4)
    args = (torch.ones(4, dtype=torch.long), torch.ones(4, dtype=torch.long), torch.zeros(4, dtype=torch.long),
            torch.zeros(4, dtype=torch.long), torch.zeros(4, 14, 7), torch.zeros(4, 14, dtype=torch.long),
            torch.zeros(4, 14, 7), torch.zeros(4, 14, dtype=torch.long), gp, gp.flip(0))
    t.eval()
    with torch.no_grad():
        before = t(x, *args)[1]
        t.enable_family_gates(cols)
        after = t(x, *args)[1]
    assert torch.allclose(before, after)
    assert t.gate_family_h.tolist() == [0, 1, 2, 3, 4]
