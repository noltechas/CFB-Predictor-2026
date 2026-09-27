"""Experimental shared team encoder followed by a consistent matchup decoder."""
from __future__ import annotations

import torch
from torch import nn

from cfbpred.consistent_net import ConsistentPregameTower
from cfbpred.raw_net import CONF_EMB, HIST_D, TEAM_EMB


class SharedTeamPregameTower(ConsistentPregameTower):
    """The same weights extract each team's evidence before learning matchups.

This shares the scalar encoder as well as identities/history across home and
away appearances. The matchup decoder still sees both teams simultaneously;
it is not a separately fitted rating or an additive strength formula.
"""

    def __init__(self, n_teams, n_confs, scalar_cols, hidden=160, **kwargs):
        super().__init__(n_teams, n_confs, scalar_cols, hidden=hidden, **kwargs)
        home = [i for i,c in enumerate(scalar_cols) if c.startswith("h_") or c == "home_is_fcs"]
        away = [int(self.swap_index[i]) for i in home]
        context = [i for i in range(len(scalar_cols)) if i not in home + away]
        self.register_buffer("home_columns", torch.tensor(home, dtype=torch.long))
        self.register_buffer("away_columns", torch.tensor(away, dtype=torch.long))
        self.register_buffer("context_columns", torch.tensor(context, dtype=torch.long))
        side_width = TEAM_EMB + CONF_EMB + 2 * HIST_D + len(home)
        self.team_body = nn.Sequential(
            nn.Linear(side_width, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(.30),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU())
        self.body = nn.Sequential(
            nn.Linear(128 + len(context), hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(.30),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(.30))

    def forward(self, scalars, h_idx, a_idx, h_conf, a_conf,
                h_hist, h_opp, a_hist, a_opp, gp_h=None, gp_a=None):
        n = len(scalars)
        pair = lambda h,a: torch.cat([h,a], dim=0)
        if self.use_s2d_gate and gp_h is not None:
            mult = scalars.new_ones(scalars.shape)
            mult[:, self.s2d_idx_h] = torch.sigmoid(self.gate_b + self.gate_w * gp_h).unsqueeze(-1)
            mult[:, self.s2d_idx_a] = torch.sigmoid(self.gate_b + self.gate_w * gp_a).unsqueeze(-1)
            scalars = scalars * mult
        histories = pair(h_hist, a_hist)
        encoded_history = self.hist(histories, pair(h_opp, a_opp))
        # No historical observations means no evidence, including the max
        # pooling block. The original encoder's all-masked sentinel is -10000.
        encoded_history = encoded_history * (histories[..., -1].sum(-1) > 0).unsqueeze(-1)
        z = self.team_body(torch.cat([
            self.emb_drop(self.team_emb(pair(h_idx, a_idx))),
            self.conf_emb(pair(h_conf, a_conf)), encoded_history,
            pair(scalars[:, self.home_columns], scalars[:, self.away_columns])], dim=-1))
        context = scalars[:, self.context_columns]
        reverse_context = context * self.swap_sign[self.context_columns]
        forward = torch.cat([z[:n], z[n:], context], dim=-1)
        reverse = torch.cat([z[n:], z[:n], reverse_context], dim=-1)
        h = self.body(pair(forward, reverse))
        emb, margin, total = self.emb(h), self.margin(h), self.total(h)
        return ((emb[:n] + emb[n:]) * .5,
                (margin[:n] + margin[n:].flip(-1)) * .5,
                (total[:n] + total[n:]) * .5)
