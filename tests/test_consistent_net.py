import numpy as np
import pytest
import torch

from cfbpred.consistent_net import ConsistentPregameTower, fit_consistent_inputs, swap_schema
from cfbpred.architecture_experiment import fold_masks
from cfbpred.model import MARGIN_CENTRES
from cfbpred.shared_team_net import SharedTeamPregameTower


COLS = ["a_gp", "h_gp", "home_is_fcs", "is_home_site", "opp_is_fcs", "week_scaled"]


def fixture(tower=ConsistentPregameTower):
    torch.manual_seed(123)
    net = tower(12, 4, COLS, hidden=16, use_s2d_gate=True,
                                 s2d_idx_h=[1], s2d_idx_a=[0]).eval()
    b = dict(scalars=torch.randn(3, len(COLS)), h_idx=torch.tensor([1,2,3]),
             a_idx=torch.tensor([4,5,6]), h_conf=torch.tensor([1,2,3]),
             a_conf=torch.tensor([3,1,2]), h_hist=torch.rand(3,14,7),
             a_hist=torch.rand(3,14,7), h_opp=torch.randint(12,(3,14)),
             a_opp=torch.randint(12,(3,14)), gp_h=torch.tensor([0.,.5,1.]),
             gp_a=torch.tensor([.2,.4,.8]))
    b["h_hist"][...,-1] = 1
    b["a_hist"][...,-1] = 1
    b["scalars"][:,3] = torch.tensor([0.,1.,-1.])
    return net, b


def test_reversal_is_involution_and_preserves_physical_venue():
    net, b = fixture()
    swapped = net.swap_scalars(b["scalars"])
    torch.testing.assert_close(net.swap_scalars(swapped), b["scalars"])
    torch.testing.assert_close(swapped[:,3], torch.tensor([0.,-1.,1.]))
    torch.testing.assert_close(swapped[:,2], b["scalars"][:,4])


@pytest.mark.parametrize("tower", [ConsistentPregameTower, SharedTeamPregameTower])
def test_margin_reverses_and_total_stays_same_for_neutral_and_hosted_games(tower):
    net, b = fixture(tower)
    swapped = {**b, "scalars":net.swap_scalars(b["scalars"])}
    for h,a in [("h_idx","a_idx"),("h_conf","a_conf"),("h_hist","a_hist"),
                ("h_opp","a_opp"),("gp_h","gp_a")]:
        swapped[h], swapped[a] = b[a], b[h]
    with torch.no_grad():
        _, m, t = net(**b)
        _, ms, ts = net(**swapped)
    torch.testing.assert_close(ms.softmax(-1), m.softmax(-1).flip(-1), atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(ts.softmax(-1), t.softmax(-1), atol=1e-7, rtol=1e-6)
    torch.testing.assert_close((ms.softmax(-1)*MARGIN_CENTRES).sum(-1),
                               -(m.softmax(-1)*MARGIN_CENTRES).sum(-1), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("tower", [ConsistentPregameTower, SharedTeamPregameTower])
def test_consistent_logits_have_finite_nonzero_training_gradients(tower):
    net, b = fixture(tower)
    net.train()
    _, m, t = net(**b)
    loss = -m.log_softmax(-1)[:,60].mean() - t.log_softmax(-1)[:,40].mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    assert net.margin.weight.grad.abs().sum() > 0


def test_shared_encoder_covers_every_scalar_once_and_empty_history_is_finite():
    net,b = fixture(SharedTeamPregameTower)
    cols = torch.cat([net.home_columns,net.away_columns,net.context_columns])
    assert sorted(cols.tolist()) == list(range(len(COLS)))
    b["h_hist"].zero_()
    b["a_hist"].zero_()
    with torch.no_grad():
        outputs = net(**b)
    assert all(torch.isfinite(x).all() for x in outputs)


def test_pooled_normalization_uses_only_fit_rows_and_keeps_missing_zero():
    raw = np.array([[2,4,0,1,1,1], [6,np.nan,1,0,0,2], [999,999,0,1,0,3]], np.float32)
    base = dict(scalar_cols=COLS,G_raw_scalars=raw,
                G_hidx=torch.tensor([1,2,9]),G_aidx=torch.tensor([2,3,8]),
                G_opp=torch.tensor([[1,2],[2,3],[8,9]]),
                G_hconf=torch.tensor([1,2,3]),G_aconf=torch.tensor([2,1,4]))
    d = fit_consistent_inputs(base, np.array([True,True,False]))
    assert d["mu"][0] == d["mu"][1] == 4
    assert d["sd"][0] == d["sd"][1] == pytest.approx(np.std([2,4,6]))
    assert d["G_scalars"][1,1] == 0
    torch.testing.assert_close(d["G_scalars"][:,3], torch.tensor([1.,0.,1.]))
    assert d["G_hidx"][-1] == d["G_aconf"][-1] == 0


def test_unknown_scalar_semantics_and_unpaired_fields_fail_closed():
    with pytest.raises(ValueError, match="No team-swap semantics"):
        swap_schema(COLS + ["new_context"])
    with pytest.raises(ValueError, match="Missing swap partner"):
        swap_schema(COLS + ["h_new_stat"])


def test_window_split_is_chronological_and_never_includes_2026():
    seasons = np.arange(2001,2027)
    d = dict(G_season=seasons,G_gid=seasons,G_labeled=np.ones(len(seasons),bool),
             G_fbs_inv=np.ones(len(seasons),bool))
    tr,va,te,refit = fold_masks(d,2025,8)
    assert seasons[tr].tolist() == list(range(2016,2024))
    assert seasons[va].tolist() == [2024]
    assert seasons[te].tolist() == [2025]
    assert seasons[refit].tolist() == list(range(2017,2025))
