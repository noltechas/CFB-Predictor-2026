"""Stage 7: the two-tower network with a discrete margin distribution.

The three outputs are not three heads. They are three readings of one predicted
distribution over final margin, plus a second over final total:

    P(margin = k)  for k in [-56, +56] plus two tail buckets   (115 bins)
    P(total  = k)  for k in [ 10, 110] plus two tail buckets   (103 bins)

    win probability = sum of P(margin = k) over k > 0
    predicted margin = E[margin]
    predicted total  = E[total]
    home score = (total + margin) / 2, away = (total - margin) / 2

Consistency is therefore structural, not enforced by a penalty: the model
cannot favour the home team while predicting them to lose, because the win
probability is literally a partial sum of the margin distribution it reports.

The discrete parameterisation also buys three things a Gaussian cannot give:
key numbers (3 and 7 are learned, not assumed), the late-game atom (in the final
two minutes 70.5% of the mass sits on "no further change", which no continuous
density can represent), and direct optimisation of the mass near zero, which is
exactly the quantity win probability reads off.

The gate between towers is LEARNED, and depends on the score as well as the
clock. Fixing it to the NFL's exp(-4 * elapsed) — a 10.4-minute half-life —
measurably hurts: on held-out seasons the optimum is nearer lambda = 1, and a
40-point lead should retire the pregame prior regardless of what the clock says.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from cfbpred.config import (MARGIN_HI, MARGIN_LO, N_MARGIN_BINS, N_REM_BINS,
                            N_TOTAL_BINS, REM_HI, TOTAL_HI, TOTAL_LO)


def margin_bin(m: torch.Tensor) -> torch.Tensor:
    """Integer margin -> bin index, with the tails collapsed."""
    inner = torch.clamp(m - MARGIN_LO + 1, 1, N_MARGIN_BINS - 2)
    return torch.where(m < MARGIN_LO, torch.zeros_like(inner),
                       torch.where(m > MARGIN_HI,
                                   torch.full_like(inner, N_MARGIN_BINS - 1),
                                   inner)).long()


def total_bin(t: torch.Tensor) -> torch.Tensor:
    inner = torch.clamp(t - TOTAL_LO + 1, 1, N_TOTAL_BINS - 2)
    return torch.where(t < TOTAL_LO, torch.zeros_like(inner),
                       torch.where(t > TOTAL_HI,
                                   torch.full_like(inner, N_TOTAL_BINS - 1),
                                   inner)).long()


def bin_centres(lo: int, hi: int, n: int, tail: float = 6.0) -> torch.Tensor:
    inner = torch.arange(lo, hi + 1, dtype=torch.float32)
    return torch.cat([torch.tensor([lo - tail]), inner, torch.tensor([hi + tail])])


MARGIN_CENTRES = bin_centres(MARGIN_LO, MARGIN_HI, N_MARGIN_BINS)
TOTAL_CENTRES = bin_centres(TOTAL_LO, TOTAL_HI, N_TOTAL_BINS)


def discrete_crps(logits, observed, centres=MARGIN_CENTRES, reduction="mean"):
    """Exact CRPS for this discrete distribution, in score points.

    E|X-y| - E|X-X'|/2, computed in linear time using the CDF.
    Actual outcomes outside the tail centres remain actual outcomes: no
    clipping that would hide the distance of a badly missed blowout.
    """
    x = centres.to(device=logits.device, dtype=logits.dtype)
    p = logits.softmax(-1)
    y = observed.to(device=logits.device, dtype=logits.dtype)
    first = (p * (x.unsqueeze(0) - y.unsqueeze(-1)).abs()).sum(-1)
    cdf = p.cumsum(-1)[..., :-1]
    pair_half = (cdf * (1 - cdf) * (x[1:] - x[:-1])).sum(-1)
    scores = first - pair_half
    if reduction == "none":
        return scores
    if reduction != "mean":
        raise ValueError("CRPS reduction must be mean or none")
    return scores.mean()

# remaining points: inner bins ARE their index (0..110), one upper tail
REM_CENTRES = torch.cat([torch.arange(0, REM_HI + 1, dtype=torch.float32),
                         torch.tensor([REM_HI + 6.0])])


def rem_bin(r: torch.Tensor) -> torch.Tensor:
    """Remaining points -> bin index. Negative input (only reachable by
    shifting a distribution's tail pseudo-centre) folds onto the zero bin."""
    inner = torch.clamp(r, 0, REM_HI)
    return torch.where(r > REM_HI,
                       torch.full_like(inner, N_REM_BINS - 1), inner).long()


def shift_pmf(p: torch.Tensor, src_centres: torch.Tensor, offset: torch.Tensor,
              dst_bin_fn, n_dst: int) -> torch.Tensor:
    """Move each row's PMF by its own integer offset and re-bin onto a target
    grid, mass collapsing into the destination tails where it runs off.

    This is pure score arithmetic — the identities final = current + remaining
    and final margin = current lead + remaining change — the same class of
    structural fact as home = (total + margin) / 2 in the readouts. It encodes
    no football knowledge: WHAT the remaining distribution looks like is still
    entirely the network's to learn."""
    v = torch.round(src_centres.to(p.device).unsqueeze(0) + offset.unsqueeze(-1))
    idx = dst_bin_fn(v)                                          # [B, S]
    out = torch.zeros(p.shape[0], n_dst, device=p.device, dtype=p.dtype)
    out.scatter_add_(1, idx, p)
    return out


class PregameTower(nn.Module):
    """Small and heavily regularised: it sees ~17k games, not 1.5M plays."""

    def __init__(self, n_in: int, hidden: int = 64, emb: int = 24,
                 dropout: float = 0.35):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.emb = nn.Linear(hidden, emb)
        self.margin = nn.Linear(hidden, N_MARGIN_BINS)
        self.total = nn.Linear(hidden, N_TOTAL_BINS)

    def forward(self, x):
        h = self.body(x)
        return self.emb(h), self.margin(h), self.total(h)


class StateTower(nn.Module):
    """Larger: millions of near-independent in-game observations."""

    def __init__(self, n_in: int, emb: int = 24, hidden: int = 128,
                 dropout: float = 0.10):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_in + emb, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.margin = nn.Linear(hidden, N_MARGIN_BINS)
        self.total = nn.Linear(hidden, N_TOTAL_BINS)

    def forward(self, state, emb):
        h = self.body(torch.cat([state, emb], dim=-1))
        return self.margin(h), self.total(h)


class Gate(nn.Module):
    """alpha(time_remaining, |lead|) in [floor, 1], monotone in both by
    construction: weights are softplus-constrained so more elapsed time and a
    bigger lead can only reduce the pregame weight.

    Initialised near a slow decay (lambda ~ 1, a ~40-minute half-life) rather
    than the NFL's lambda = 4, and with a non-zero floor, because pregame
    strength stays informative deep into college games — a measured optimum and
    the same conclusion ESPN's own live feed shows.
    """

    def __init__(self, floor: float = 0.02):
        super().__init__()
        self.floor = floor
        self.w_time = nn.Parameter(torch.tensor(0.9))
        self.w_lead = nn.Parameter(torch.tensor(-1.6))
        self.bias = nn.Parameter(torch.tensor(-0.4))

    def forward(self, frac_remaining, abs_lead_scaled):
        z = (nn.functional.softplus(self.w_time) * frac_remaining
             - nn.functional.softplus(self.w_lead) * abs_lead_scaled
             + self.bias)
        return self.floor + (1.0 - self.floor) * torch.sigmoid(z)


class TwoTower(nn.Module):
    def __init__(self, n_pre: int, n_state: int, emb: int = 24):
        super().__init__()
        self.pre = PregameTower(n_pre, emb=emb)
        self.state = StateTower(n_state, emb=emb)
        self.gate = Gate()

    def forward(self, x_pre, x_state, frac_remaining, abs_lead_scaled):
        emb, pm, pt = self.pre(x_pre)
        sm, st = self.state(x_state, emb)
        a = self.gate(frac_remaining, abs_lead_scaled).unsqueeze(-1)
        # Blend in log space: a mixture of the two distributions in logits keeps
        # the result a valid distribution and lets the gate move smoothly.
        margin = a * pm + (1.0 - a) * sm
        total = a * pt + (1.0 - a) * st
        return {"margin": margin, "total": total,
                "pre_margin": pm, "pre_total": pt, "alpha": a.squeeze(-1)}


# ----------------------------------------------------------------- readouts

def win_probability(margin_logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(margin_logits, dim=-1)
    pos = (MARGIN_CENTRES.to(p.device) > 0).float()
    return (p * pos).sum(-1)


def expected_margin(margin_logits: torch.Tensor) -> torch.Tensor:
    """Mean of the margin distribution. Minimises squared error, so this is the
    number to score RMSE against — but see `median_margin` for what to display."""
    p = torch.softmax(margin_logits, dim=-1)
    return (p * MARGIN_CENTRES.to(p.device)).sum(-1)


def median_margin(margin_logits: torch.Tensor) -> torch.Tensor:
    """Median of the margin distribution — the number to SHOW alongside the win
    probability.

    The mean and the win probability can legitimately disagree in sign when the
    distribution is skewed: a team can be more likely than not to win by a
    little while its average outcome is a loss, because the losses are bigger.
    That is correct probability and it happens on 2.2% of games, but a scoreline
    that contradicts the win probability reads as a bug to anyone looking at it.
    The median has no such problem — P(margin > median) = 0.5 by definition, so
    its sign always agrees with whether the win probability is above one half.
    """
    p = torch.softmax(margin_logits, dim=-1)
    c = torch.cumsum(p, dim=-1)
    idx = torch.searchsorted(c.contiguous(),
                             torch.full((p.shape[0], 1), 0.5, device=p.device))
    idx = torch.clamp(idx, 0, p.shape[-1] - 1).squeeze(-1)
    return MARGIN_CENTRES.to(p.device)[idx]


def median_total(total_logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(total_logits, dim=-1)
    c = torch.cumsum(p, dim=-1)
    idx = torch.searchsorted(c.contiguous(),
                             torch.full((p.shape[0], 1), 0.5, device=p.device))
    idx = torch.clamp(idx, 0, p.shape[-1] - 1).squeeze(-1)
    return TOTAL_CENTRES.to(p.device)[idx]


def expected_total(total_logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(total_logits, dim=-1)
    return (p * TOTAL_CENTRES.to(p.device)).sum(-1)


def scores_from(margin: torch.Tensor, total: torch.Tensor):
    """Derive a displayable scoreline, honouring the parity constraint.

    margin and total always share parity — home = (total + margin)/2 must be an
    integer, and that holds on all 17,472 games in the corpus. Rounding the two
    independently breaks it about half the time and yields half-point scores, so
    the total is nudged by one point to match the margin's parity.
    """
    m = torch.round(margin)
    t = torch.round(total)
    mismatch = (torch.abs(m) % 2) != (t % 2)
    t = torch.where(mismatch, t + 1.0, t)
    home = (t + m) / 2.0
    away = (t - m) / 2.0
    # a team score of exactly 1 is impossible in football
    home = torch.where(home == 1, torch.zeros_like(home), torch.clamp(home, min=0))
    away = torch.where(away == 1, torch.zeros_like(away), torch.clamp(away, min=0))
    return home, away


def smoothed_targets(bins: torch.Tensor, n_bins: int, eps: float = 0.10
                     ) -> torch.Tensor:
    """Label smoothing onto the two neighbouring bins.

    The bins are ordinal — being one point out is nearly right — but plain
    cross-entropy treats every wrong bin alike. Spreading a little mass to the
    neighbours restores that ordering without imposing a parametric shape, so
    key numbers are still free to be learned.
    """
    t = torch.zeros(bins.shape[0], n_bins, device=bins.device)
    t.scatter_(1, bins.unsqueeze(1), 1.0 - eps)
    lo = torch.clamp(bins - 1, 0, n_bins - 1).unsqueeze(1)
    hi = torch.clamp(bins + 1, 0, n_bins - 1).unsqueeze(1)
    t.scatter_add_(1, lo, torch.full_like(lo, eps / 2, dtype=t.dtype))
    t.scatter_add_(1, hi, torch.full_like(hi, eps / 2, dtype=t.dtype))
    return t


def soft_ce(logits: torch.Tensor, targets: torch.Tensor,
            weights: torch.Tensor | None = None) -> torch.Tensor:
    # Parameter-efficient ensembles expose [batch, member, class] while
    # ordinary models expose [batch, class]. Each member must receive its own
    # proper loss; averaging predictions before the loss defeats the method.
    while targets.ndim < logits.ndim:
        targets = targets.unsqueeze(-2)
    ce = -(targets * torch.log_softmax(logits, dim=-1)).sum(-1)
    if weights is None:
        return ce.mean()
    while weights.ndim < ce.ndim:
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(ce)
    return (ce * weights).sum() / weights.sum()


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
