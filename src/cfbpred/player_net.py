"""End-to-end individual scouting/history -> permutation-invariant roster.

No player ID embeddings, fixed player ratings, starter assumptions, or top-N
cuts. All observed roster members contribute. Position is a dated recruiting
role; the network also observes raw college production across all positions.
"""
from __future__ import annotations

import torch
from torch import nn

from cfbpred.consistent_net import ConsistentPregameTower


class PlayerSetEncoder(nn.Module):
    def __init__(self, n_scout, n_history, n_positions=10, pooling="meanmax"):
        super().__init__()
        if pooling not in {"meanmax", "attention"}:
            raise ValueError("Unknown player pooling")
        self.pooling = pooling
        self.n_positions = n_positions
        self.position = nn.Embedding(n_positions, 4)
        self.history = nn.Sequential(nn.Linear(n_history, 12), nn.GELU())
        self.player = nn.Sequential(nn.Linear(n_scout+4*12+4, 32), nn.GELU(),
                                    nn.Dropout(.15), nn.Linear(32, 12), nn.GELU())
        if pooling == "attention":
            self.attention = nn.Linear(12, 1)
        self.group = nn.Sequential(nn.Linear(25, 8), nn.GELU())
        self.team = nn.Sequential(nn.Linear(n_positions*8, 24), nn.LayerNorm(24), nn.GELU())

    def forward(self, scout, history, position, mask):
        # [B,2,P,*]. Linear layers act independently on players; pooling is
        # invariant to roster order and explicitly zero for empty groups.
        h = self.history(history).flatten(-2)
        x = self.player(torch.cat([scout, h, self.position(position)], dim=-1))
        groups = []
        for group in range(self.n_positions):
            valid = mask & position.eq(group)
            count = valid.sum(-1, keepdim=True)
            mean = (x*valid[..., None]).sum(-2)/count.clamp(min=1)
            if self.pooling == "meanmax":
                second = x.masked_fill(~valid[..., None], -1e4).amax(-2)
                second = torch.where(count.gt(0), second, 0.)
            else:
                logits = self.attention(x).squeeze(-1).masked_fill(~valid, -1e4)
                weights = logits.softmax(-1)*valid
                weights = weights/weights.sum(-1, keepdim=True).clamp(min=1e-8)
                second = (x*weights[..., None]).sum(-2)
            features = torch.cat([mean, second, torch.log1p(count.float())/4], -1)
            groups.append(self.group(features)*count.gt(0))
        return self.team(torch.cat(groups, -1))*mask.any(-1, keepdim=True)


class PlayerPregameTower(nn.Module):
    """Add learned roster context to every original game feature."""
    uses_players = True

    def __init__(self, n_teams, n_confs, scalar_cols, player_schema,
                 s2d_idx_h=None, s2d_idx_a=None, hidden=160, pooling="meanmax"):
        super().__init__()
        self.encoder = PlayerSetEncoder(len(player_schema["static_cols"]),
                                        len(player_schema["history_cols"]),
                                        len(player_schema["positions"]), pooling)
        cols = scalar_cols + [f"{side}_player_latent_{i}" for side in ["h", "a"] for i in range(24)]
        self.core = ConsistentPregameTower(n_teams, n_confs, cols, hidden=hidden,
                                          s2d_idx_h=s2d_idx_h, s2d_idx_a=s2d_idx_a,
                                          use_s2d_gate=True)
        self.core.hist.empty_is_zero = True

    def forward_batch(self, batch):
        for name in ["player_scout", "player_history", "player_position", "player_mask"]:
            if name not in batch:
                raise ValueError(f"Player model requires {name}; refusing incomplete inference")
        roster = self.encoder(batch["player_scout"], batch["player_history"],
                              batch["player_position"], batch["player_mask"])
        scalars = torch.cat([batch["scalars"], roster[:, 0], roster[:, 1]], -1)
        return self.core(scalars, batch["h_idx"], batch["a_idx"], batch["h_conf"], batch["a_conf"],
                         batch["h_hist"], batch["h_opp"], batch["a_hist"], batch["a_opp"],
                         batch.get("h_gp01"), batch.get("a_gp01"))


def make_player_model(d, pooling="meanmax"):
    from cfbpred.train import make_model, s2d_indices
    if "player_store" not in d:
        raise ValueError("Attach PlayerStore before creating a player model")
    model = make_model(d, s2d_gate=True, delta=True)
    model.pre = PlayerPregameTower(len(d["vocab"]["team"])+1, len(d["vocab"]["conf"])+1,
                                  d["scalar_cols"], d["player_store"].manifest,
                                  s2d_indices(d["scalar_cols"], "h"), s2d_indices(d["scalar_cols"], "a"),
                                  pooling=pooling)
    return model
