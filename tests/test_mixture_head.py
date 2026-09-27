import torch

from cfbpred.mixture_head import DiscreteMixtureHead


def test_integrated_logistic_bins_match_analytic_cdf_with_tail_mass():
    head = DiscreteMixtureHead(4,-2,2,components=1)
    with torch.no_grad():
        head.parameters_head.weight.zero_()
        head.offsets.zero_()
    logits = head(torch.zeros(2,4))
    scale = 2+8*torch.nn.functional.softplus(torch.tensor(0.))
    cdf = (head.edges/scale).sigmoid()
    expected = torch.cat([cdf[:1],cdf[1:]-cdf[:-1],1-cdf[-1:]])
    torch.testing.assert_close(logits.exp(),expected.expand(2,-1),atol=1e-7,rtol=1e-5)
    torch.testing.assert_close(logits.softmax(-1).sum(-1),torch.ones(2))


def test_extreme_misses_keep_finite_tail_gradients():
    head = DiscreteMixtureHead(4,-56,56)
    with torch.no_grad():
        head.parameters_head.bias[:3].fill_(50)
    logits = head(torch.ones(2,4))
    assert torch.isfinite(logits).all()
    loss = -logits.log_softmax(-1)[:,0].mean()
    loss.backward()
    grad = head.parameters_head.bias.grad[:3]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0


def test_shape_parameters_and_learned_key_number_bias_both_receive_gradients():
    head = DiscreteMixtureHead(4,-56,56)
    logits = head(torch.ones(3,4))
    assert logits.shape == (3,115)
    loss = -logits.log_softmax(-1)[:,64].mean()  # +7 points
    loss.backward()
    assert head.bin_bias.grad[64]<0
    assert torch.isfinite(head.parameters_head.weight.grad).all()
    assert head.parameters_head.weight.grad.abs().sum()>0
