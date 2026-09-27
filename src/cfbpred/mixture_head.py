"""Raw-DNN distribution head: smooth conditional shape plus learned bin bias.

The network predicts mixture weights, locations and scales. Integrating each
logistic component over integer-score bins gives an ordinary discrete PMF,
including open-ended tail bins. Shared bin biases learn common final margins
directly from training outcomes; no market values or separate fitted model.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DiscreteMixtureHead(nn.Module):
    def __init__(self, hidden, lo, hi, centre=0.0, components=3):
        super().__init__()
        self.components = components
        self.centre = centre
        self.parameters_head = nn.Linear(hidden, components * 3)
        nn.init.normal_(self.parameters_head.weight, std=.01)
        nn.init.zeros_(self.parameters_head.bias)
        self.register_buffer("offsets", torch.linspace(-12.,12.,components))
        self.register_buffer("edges", torch.arange(lo-.5, hi+1., dtype=torch.float32))
        self.bin_bias = nn.Parameter(torch.zeros(hi-lo+3))

    def forward(self, hidden):
        raw = self.parameters_head(hidden).reshape(-1,3,self.components)
        location = self.centre + 20 * raw[:,0] + self.offsets
        scale = 2.0 + 8.0 * F.softplus(raw[:,1])
        weights = raw[:,2].log_softmax(-1)
        z = (self.edges[None,None,:] - location[:,:,None]) / scale[:,:,None]
        # Stable log differences: sigmoid(b)-sigmoid(a) equals
        # sigmoid(b)*sigmoid(-a)*(1-exp(a-b)). Avoid subtracting two CDFs
        # rounded to 1, which would destroy tail gradients.
        inner = F.logsigmoid(z[:,:,1:]) + F.logsigmoid(-z[:,:,:-1]) + torch.log(-torch.expm1(z[:,:,:-1]-z[:,:,1:]))
        log_bins = torch.cat([F.logsigmoid(z[:,:,:1]), inner, F.logsigmoid(-z[:,:,-1:])], dim=-1)
        return torch.logsumexp(log_bins + weights[:,:,None], dim=1) + self.bin_bias
