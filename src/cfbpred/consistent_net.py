"""Experimental pregame tower with exact team-order consistency at inference.

Swapping the input teams reverses the margin PMF and leaves the total PMF
unchanged. The signed venue input follows the physical home team, so reversing
the labels does not turn a road game into a home game. All active raw inputs
are retained. This module is opt-in; production checkpoints are unaffected.
"""
from __future__ import annotations

import numpy as np
import torch

from cfbpred.backtest_pregame import fit_inputs
from cfbpred.raw_net import RawPregameTower


# Columns whose sign follows the physical home team: reversing the labels negates them.
# They are standardised about zero (never their mean) so the negation stays exact.
ANTISYMMETRIC = {"is_home_site", "h2h_last_margin"}


def swap_schema(cols):
    """Fail closed if a new feature has no declared reversal semantics."""
    if len(set(cols)) != len(cols):
        raise ValueError("Duplicate scalar columns")
    common = {"is_conf_game", "is_postseason", "week_scaled",
              # re-admitted game-level columns that do not depend on which side is "home"
              "h2h_days_since", "h2h_meetings_5y", "kick_local_hour", "venue_grass", "venue_capacity"}
    other = {"home_is_fcs": "opp_is_fcs", "opp_is_fcs": "home_is_fcs"}
    index, sign = [], []
    for col in cols:
        if col.startswith(("h_", "a_")):
            partner = ("a_" if col.startswith("h_") else "h_") + col[2:]
        elif col in other:
            partner = other[col]
        elif col in common or col in ANTISYMMETRIC:
            partner = col
        else:
            raise ValueError(f"No team-swap semantics for {col}")
        if partner not in cols:
            raise ValueError(f"Missing swap partner for {col}: {partner}")
        index.append(cols.index(partner))
        sign.append(-1.0 if col in ANTISYMMETRIC else 1.0)
    return np.asarray(index), np.asarray(sign, dtype=np.float32)


def fit_consistent_inputs(base, mask):
    """Training-only pooled normalization for each pair of side features.

Using the same mean/std on both sides makes reversal valid even when a value
is missing (standardized zero). Venue is deliberately uncentered: +1 means
the first team is physically at home, 0 neutral, -1 the second team at home.
"""
    d = fit_inputs(base, mask)
    cols = base["scalar_cols"]
    permutation, _ = swap_schema(cols)
    raw = base["G_raw_scalars"]
    mu, sd = d["mu"].copy(), d["sd"].copy()
    for i, j in enumerate(permutation):
        if j <= i:
            continue
        values = raw[mask][:, [i, j]].ravel()
        values = values[np.isfinite(values)]
        mean = float(values.mean()) if len(values) else 0.0
        std = float(values.std()) if len(values) else 1.0
        mu[[i, j]] = mean
        sd[[i, j]] = std if std >= 1e-6 else 1.0
    venue = cols.index("is_home_site")
    mu[venue], sd[venue] = 0.0, 1.0
    for col in ANTISYMMETRIC - {"is_home_site"}:
        if col in cols:
            k = cols.index(col); values = raw[mask][:, k]; values = values[np.isfinite(values)]
            mu[k], sd[k] = 0.0, (float(np.sqrt(np.mean(values ** 2))) if len(values) else 1.0) or 1.0
    d["mu"], d["sd"] = mu, sd
    d["G_scalars"] = torch.tensor(np.nan_to_num((raw - mu) / sd, nan=0, posinf=0, neginf=0))
    return d


class ConsistentPregameTower(RawPregameTower):
    """Shared network evaluates both orientations, with tied output logits."""

    def __init__(self, n_teams, n_confs, scalar_cols, **kwargs):
        super().__init__(n_teams, n_confs, len(scalar_cols), **kwargs)
        permutation, sign = swap_schema(scalar_cols)
        self.register_buffer("swap_index", torch.tensor(permutation, dtype=torch.long))
        self.register_buffer("swap_sign", torch.tensor(sign))

    def swap_scalars(self, scalars):
        return scalars[:, self.swap_index] * self.swap_sign

    def forward(self, scalars, h_idx, a_idx, h_conf, a_conf,
                h_hist, h_opp, a_hist, a_opp, gp_h=None, gp_a=None):
        n = len(scalars)
        pair = lambda h, a: torch.cat([h, a], dim=0)
        emb, margin, total = super().forward(
            pair(scalars, self.swap_scalars(scalars)),
            pair(h_idx, a_idx), pair(a_idx, h_idx),
            pair(h_conf, a_conf), pair(a_conf, h_conf),
            pair(h_hist, a_hist), pair(h_opp, a_opp),
            pair(a_hist, h_hist), pair(a_opp, h_opp),
            None if gp_h is None else pair(gp_h, gp_a),
            None if gp_a is None else pair(gp_a, gp_h))
        return ((emb[:n] + emb[n:]) * 0.5,
                (margin[:n] + margin[n:].flip(-1)) * 0.5,
                (total[:n] + total[n:]) * 0.5)
