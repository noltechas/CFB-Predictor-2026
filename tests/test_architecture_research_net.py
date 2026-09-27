import torch

from cfbpred.architecture_research_net import (BatchEnsembleLinear,
                                                IndependentEnsembleLinear,
                                                ResearchCore, ResidualBody)
from cfbpred.model import soft_ce


def scalar_cols():
    return ["h_x", "a_x", "is_home_site", "is_conf_game", "is_postseason", "week_scaled",
            "home_is_fcs", "opp_is_fcs"]


def inputs(batch=4):
    return (torch.randn(batch, len(scalar_cols())),
            torch.randint(0, 8, (batch,)), torch.randint(0, 8, (batch,)),
            torch.randint(0, 3, (batch,)), torch.randint(0, 3, (batch,)),
            torch.zeros(batch, 14, 7), torch.zeros(batch, 14, dtype=torch.long),
            torch.zeros(batch, 14, 7), torch.zeros(batch, 14, dtype=torch.long),
            torch.zeros(batch), torch.zeros(batch))


def test_batchensemble_and_independent_heads_keep_member_axis():
    x = torch.randn(5, 7)
    h = BatchEnsembleLinear(7, 11, 3, first=True)(x)
    assert h.shape == (5, 3, 11)
    assert IndependentEnsembleLinear(11, 4, 3)(h).shape == (5, 3, 4)


def test_soft_ce_trains_ensemble_members_independently():
    logits = torch.randn(5, 3, 7, requires_grad=True)
    targets = torch.softmax(torch.randn(5, 7), -1)
    loss = soft_ce(logits, targets)
    expected = -(targets[:, None] * logits.log_softmax(-1)).sum(-1).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None


def test_residual_body_shape_and_gradient():
    x = torch.randn(6, 20, requires_grad=True)
    y = ResidualBody(20, 16)(x)
    assert y.shape == (6, 16)
    y.sum().backward()
    assert x.grad is not None


def test_research_cores_preserve_team_swap_semantics():
    for architecture in ["residual", "tabm"]:
        model = ResearchCore(8, 3, scalar_cols(), architecture=architecture,
                             history_attention=True)
        model.eval()
        args = inputs()
        _, margin, total = model(*args)
        scalars, hi, ai, hc, ac, hh, ho, ah, ao, gh, ga = args
        swapped = model.swap_scalars(scalars)
        _, reverse_margin, reverse_total = model(
            swapped, ai, hi, ac, hc, ah, ao, hh, ho, ga, gh)
        assert margin.shape == (4, 115)
        assert total.shape == (4, 103)
        assert torch.allclose(margin, reverse_margin.flip(-1), atol=1e-6)
        assert torch.allclose(total, reverse_total, atol=1e-6)


def test_tabm_exposes_members_only_during_training():
    model = ResearchCore(8, 3, scalar_cols(), architecture="tabm")
    model.train()
    _, margin, total = model(*inputs())
    assert margin.shape == (4, 16, 115)
    assert total.shape == (4, 16, 103)
