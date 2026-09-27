"""Research-backed pregame architectures over the audited player inputs.

The production context encoder remains the control.  The opt-in alternatives
change only how its raw, chronological information is combined:

* a residual MLP game backbone;
* TabM-style parameter-efficient ensembling;
* attention over prior-game sets and opposing position-unit sets.

All variants retain exact team-order consistency and every roster member.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from cfbpred.config import N_MARGIN_BINS, N_TOTAL_BINS
from cfbpred.consistent_net import ConsistentPregameTower
from cfbpred.player_sequence_net import RichPlayerEncoder, safe_lookup
from cfbpred.raw_net import HIST_D, HIST_FIELDS, TEAM_EMB


VARIANTS = {
    "residual": dict(core="residual", history_attention=False, unit_cross=False),
    "tabm16": dict(core="tabm", history_attention=False, unit_cross=False),
    "history_attention": dict(core="control", history_attention=True, unit_cross=False),
    "unit_cross": dict(core="control", history_attention=False, unit_cross=True),
    "full_hybrid": dict(core="tabm", history_attention=True, unit_cross=True),
}


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float = .20):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.layers = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width * 2, width), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.layers(self.norm(x))


class ResidualBody(nn.Module):
    def __init__(self, n_in: int, width: int = 160):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(n_in, width), nn.LayerNorm(width), nn.GELU())
        self.blocks = nn.Sequential(ResidualBlock(width), ResidualBlock(width), ResidualBlock(width))
        self.output = nn.Sequential(nn.LayerNorm(width), nn.GELU())

    def forward(self, x):
        return self.output(self.blocks(self.input(x)))


class BatchEnsembleLinear(nn.Module):
    """The rank-one weight adapters used by BatchEnsemble and TabM."""
    def __init__(self, n_in: int, n_out: int, k: int, first: bool):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.empty(n_out, n_in))
        self.r = nn.Parameter(torch.empty(k, n_in))
        self.s = nn.Parameter(torch.empty(k, n_out))
        self.bias = nn.Parameter(torch.empty(k, n_out))
        bound = n_in ** -.5
        nn.init.uniform_(self.weight, -bound, bound)
        if first:
            nn.init.normal_(self.r)
        else:
            nn.init.ones_(self.r)
        nn.init.ones_(self.s)
        initial_bias = torch.empty(n_out)
        nn.init.uniform_(initial_bias, -bound, bound)
        with torch.no_grad():
            self.bias.copy_(initial_bias)

    def forward(self, x):
        if x.ndim == 2:
            x = x[:, None, :].expand(-1, self.k, -1)
        return ((x * self.r) @ self.weight.T) * self.s + self.bias


class IndependentEnsembleLinear(nn.Module):
    """Non-shared heads keep the TabM members meaningfully different."""
    def __init__(self, n_in: int, n_out: int, k: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, n_out, n_in))
        self.bias = nn.Parameter(torch.empty(k, n_out))
        bound = n_in ** -.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return torch.einsum("bki,koi->bko", x, self.weight) + self.bias


class TabMBody(nn.Module):
    """Two-block TabM backbone with conservative TabM initialization."""
    def __init__(self, n_in: int, width: int = 192, k: int = 16, dropout: float = .10):
        super().__init__()
        self.first = BatchEnsembleLinear(n_in, width, k, first=True)
        self.second = BatchEnsembleLinear(width, width, k, first=False)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.activation(self.first(x)))
        return self.dropout(self.activation(self.second(x)))


class MeanEnsembleEmbedding(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)

    def forward(self, x):
        return self.linear(x.mean(-2))


class AttentiveHistoryEncoder(nn.Module):
    """Set Transformer style self-attention plus pooling over 14 prior games."""
    def __init__(self, team_emb: nn.Embedding, emb_drop: nn.Dropout):
        super().__init__()
        self.team_emb = team_emb
        self.emb_drop = emb_drop
        self.slot = nn.Sequential(
            nn.Linear(HIST_FIELDS - 1 + TEAM_EMB, 48), nn.GELU(),
            nn.Linear(48, HIST_D), nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            HIST_D, nhead=4, dim_feedforward=64, dropout=.10,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.interactions = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.seed = nn.Parameter(torch.empty(1, 1, HIST_D))
        nn.init.normal_(self.seed, std=.05)
        self.pool = nn.MultiheadAttention(HIST_D, 4, dropout=.10, batch_first=True)
        self.output = nn.Sequential(nn.Linear(3 * HIST_D, 2 * HIST_D), nn.GELU())

    def forward(self, raw, opp_idx):
        mask = raw[..., -1].bool()
        x = self.slot(torch.cat([
            raw[..., :-1], self.emb_drop(self.team_emb(opp_idx))], -1))
        safe = mask.clone()
        empty = ~safe.any(-1)
        safe[empty, 0] = True
        x = x * mask[..., None]
        contextual = self.interactions(x, src_key_padding_mask=~safe)
        pooled = self.pool(self.seed.expand(len(x), -1, -1), contextual, contextual,
                           key_padding_mask=~safe, need_weights=False)[0].squeeze(1)
        count = mask.sum(1, keepdim=True)
        mean = (x * mask[..., None]).sum(1) / count.clamp(min=1)
        maximum = x.masked_fill(~mask[..., None], -1e4).amax(1)
        maximum = torch.where(count.gt(0), maximum, 0.)
        pooled = torch.where(count.gt(0), pooled, 0.)
        return self.output(torch.cat([mean, maximum, pooled], -1))


class UnitCrossEncoder(RichPlayerEncoder):
    """Contextualize each position unit against all opposing position units."""
    def __init__(self, schema, n_teams):
        super().__init__(schema, n_teams, context=True, contribution=False)
        self.unit_projection = nn.Linear(8, 16)
        self.unit_position = nn.Embedding(self.n_positions, 16)
        self.cross_attention = nn.MultiheadAttention(16, 4, dropout=.10, batch_first=True)
        self.cross_norm = nn.LayerNorm(16)
        self.cross_ff = nn.Sequential(
            nn.LayerNorm(16), nn.Linear(16, 32), nn.GELU(), nn.Dropout(.10), nn.Linear(32, 16))
        self.cross_team = nn.Sequential(
            nn.Linear(self.n_positions * 48, 64), nn.GELU(), nn.Dropout(.15), nn.Linear(64, 24))
        self.cross_scale = nn.Parameter(torch.tensor(-2.0))

    def forward_batch(self, batch):
        scout, history, position, mask = (batch[k] for k in
            ["player_scout", "player_history", "player_position", "player_mask"])
        h = self.history(history).flatten(-2)
        slots = batch["player_slots"]
        packed_position = (torch.cat([position.reshape(-1)[slots[i:i+8192]]
                                      for i in range(0, len(slots), 8192)])
                           if len(slots) else position.new_empty(0))
        packed = self.player(torch.cat([scout, h, safe_lookup(self.position, packed_position)], -1))
        x = packed.new_zeros((mask.numel(), 12)).index_copy(0, slots, packed).reshape(*mask.shape, 12)
        x = x + self.sequence(batch, mask.shape)
        groups, present = [], []
        for group in range(self.n_positions):
            valid = mask & position.eq(group)
            count = valid.sum(-1, keepdim=True)
            mean = (x * valid[..., None]).sum(-2) / count.clamp(min=1)
            maximum = x.masked_fill(~valid[..., None], -1e4).amax(-2)
            maximum = torch.where(count.gt(0), maximum, 0.)
            groups.append(self.group(torch.cat(
                [mean, maximum, torch.log1p(count.float()) / 4], -1)) * count.gt(0))
            present.append(count.squeeze(-1).gt(0))
        groups = torch.stack(groups, -2)                 # [B, 2, position, 8]
        present = torch.stack(present, -1)                # [B, 2, position]
        base = self.team(groups.flatten(-2)) * mask.any(-1, keepdim=True)
        pos = self.unit_position.weight.view(1, 1, self.n_positions, 16)
        tokens = (self.unit_projection(groups) + pos) * present[..., None]
        own = tokens.reshape(-1, self.n_positions, 16)
        opposing = tokens.flip(1).reshape(-1, self.n_positions, 16)
        own_mask = present.reshape(-1, self.n_positions)
        opp_mask = present.flip(1).reshape(-1, self.n_positions)
        safe_opp = opp_mask.clone()
        safe_opp[~safe_opp.any(-1), 0] = True
        ctx = self.cross_attention(own, opposing, opposing,
                                   key_padding_mask=~safe_opp, need_weights=False)[0]
        ctx = self.cross_norm(own + ctx)
        ctx = (ctx + self.cross_ff(ctx)) * own_mask[..., None]
        interaction = torch.cat([own, ctx, own * ctx], -1).flatten(-2)
        delta = self.cross_team(interaction).reshape(*base.shape)
        roster = base + torch.sigmoid(self.cross_scale) * delta
        return roster * mask.any(-1, keepdim=True), x


class ResearchCore(ConsistentPregameTower):
    def __init__(self, n_teams, n_confs, scalar_cols, architecture="control",
                 history_attention=False, **kwargs):
        hidden = 192 if architecture == "tabm" else 160
        super().__init__(n_teams, n_confs, scalar_cols, hidden=hidden, **kwargs)
        if history_attention:
            self.hist = AttentiveHistoryEncoder(self.team_emb, self.emb_drop)
        n_in = self.body[0].in_features
        if architecture == "residual":
            self.body = ResidualBody(n_in, hidden)
        elif architecture == "tabm":
            self.body = TabMBody(n_in, hidden, k=16)
            self.emb = MeanEnsembleEmbedding(hidden, 32)
            self.margin = IndependentEnsembleLinear(hidden, N_MARGIN_BINS, 16)
            self.total = IndependentEnsembleLinear(hidden, N_TOTAL_BINS, 16)
        elif architecture != "control":
            raise ValueError(f"Unknown research core: {architecture}")
        self.architecture = architecture

    @staticmethod
    def _collapse(logits):
        return torch.logsumexp(torch.log_softmax(logits, -1), -2) - math.log(logits.shape[-2])

    def forward(self, *args, **kwargs):
        emb, margin, total = super().forward(*args, **kwargs)
        # TabM trains each member against the target independently. Evaluation
        # averages probabilities only after team-order symmetrization.
        if self.architecture == "tabm" and not self.training:
            margin, total = self._collapse(margin), self._collapse(total)
        return emb, margin, total


# Per-side availability inputs: raw counts about who is reported unavailable, in the
# same spirit as returning production. Usage comes from each player's last three box
# scores (stored as log counts, recovered here), availability from the dated context.
ABSENCE_STATS = ("att", "rush_att", "rec", "tackles")
ABSENCE_NAMES = ("report_known", *(f"out_share_{s}" for s in ABSENCE_STATS),
                 "unsure_share_att", "players_out_log")


def absence_features(batch, context_cols):
    """[B, 2, len(ABSENCE_NAMES)] share of each side's recent usage held by players
    reported out. A side with no report reads report_known=0 and zero shares, so the
    network can tell "unreported" from "reported and healthy"."""
    from cfbpred.player_sequence_inputs import ROLE_COLS
    mask = batch["player_mask"]
    B, S, P = mask.shape
    context = batch.get("player_context")
    grid = mask.new_zeros((B * S * P, len(context_cols)), dtype=torch.float32)
    if context is not None and len(batch["player_slots"]):
        grid = grid.index_copy(0, batch["player_slots"], context.float())
    grid = grid.reshape(B, S, P, len(context_cols))
    flag = lambda name: grid[..., context_cols.index(name)].gt(0) & mask
    out = flag("out") | batch.get("player_confirmed_out", torch.zeros_like(mask))
    unsure = (flag("doubtful") | flag("questionable")) & ~out
    known = flag("report_known").any(-1).float()
    role = batch["player_role"]
    usage = torch.stack([torch.expm1(5 * role[..., ROLE_COLS.index(f"recent3_{s}_value")].clamp(min=0))
                         for s in ABSENCE_STATS], -1) * mask[..., None]
    total = usage.sum(-2).clamp(min=1e-6)
    out_share = (usage * out[..., None]).sum(-2) / total
    unsure_share = (usage[..., 0] * unsure).sum(-1) / total[..., 0]
    count = torch.log1p(out.sum(-1).float()) / 3
    return torch.cat([known[..., None], out_share, unsure_share[..., None], count[..., None]], -1)


class ResearchPlayerTower(nn.Module):
    uses_players = True
    auxiliary = False
    # Weight of the roster-only margin loss. Zero (the default) leaves the tower
    # exactly as deployed; the head then does not exist.
    roster_head_weight = 0.0
    roster_margin_logits = None

    def __init__(self, n_teams, n_confs, scalar_cols, schema, variant,
                 s2d_idx_h=None, s2d_idx_a=None, absence=False):
        super().__init__()
        settings = VARIANTS[variant]
        self.encoder = (UnitCrossEncoder(schema, n_teams) if settings["unit_cross"] else
                        RichPlayerEncoder(schema, n_teams, context=True, contribution=False))
        # Off by default: the core's input width is unchanged, so every deployed
        # checkpoint loads and scores exactly as before.
        self.absence = bool(absence)
        cols = scalar_cols + [f"{side}_player_latent_{i}"
                              for side in ["h", "a"] for i in range(24)]
        if self.absence:
            cols = cols + [f"{side}_absence_{name}" for side in ["h", "a"] for name in ABSENCE_NAMES]
        self.core = ResearchCore(
            n_teams, n_confs, cols, architecture=settings["core"],
            history_attention=settings["history_attention"],
            s2d_idx_h=s2d_idx_h, s2d_idx_a=s2d_idx_a, use_s2d_gate=True)
        self.core.hist.empty_is_zero = True
        # The official TabM package default differs from the older production
        # optimizer; the trainer reads these only for TabM variants.
        self.learning_rate = .002
        self.weight_decay = .0003 if settings["core"] == "tabm" else .03

    def attach_roster_head(self, weight: float, hidden: int = 64):
        """Auxiliary head that must predict the margin from the two roster summaries
        ALONE. Trained end to end only through the outcome loss, the roster branch
        collapses toward a constant because team identity, record and history already
        explain strength; this loss makes the summary carry strength in its own right,
        so removing a starter from the pooled unit has somewhere to go."""
        if weight <= 0:
            raise ValueError("Roster head weight must be positive")
        self.roster_head_weight = float(weight)
        self.roster_head = nn.Sequential(nn.Linear(48, hidden), nn.GELU(), nn.Linear(hidden, N_MARGIN_BINS))

    def forward_batch(self, batch):
        roster, _ = self.encoder.forward_batch(batch)
        if self.roster_head_weight > 0:
            # home-then-away, and away-then-home with the margin axis reversed: the
            # same team-order consistency the core enforces.
            forward = self.roster_head(torch.cat([roster[:, 0], roster[:, 1]], -1))
            reverse = self.roster_head(torch.cat([roster[:, 1], roster[:, 0]], -1)).flip(-1)
            self.roster_margin_logits = 0.5 * (forward + reverse)
        parts = [batch["scalars"], roster[:, 0], roster[:, 1]]
        if self.absence:
            from cfbpred.player_timeline import CONTEXT_COLS
            absent = absence_features(batch, CONTEXT_COLS)
            self.absence_inputs = absent.detach()
            parts += [absent[:, 0], absent[:, 1]]
        scalars = torch.cat(parts, -1)
        return self.core(
            scalars, batch["h_idx"], batch["a_idx"], batch["h_conf"], batch["a_conf"],
            batch["h_hist"], batch["h_opp"], batch["a_hist"], batch["a_opp"],
            batch.get("h_gp01"), batch.get("a_gp01"))


def make(d, variant, absence=False):
    from cfbpred.train import make_model, s2d_indices
    if variant not in VARIANTS:
        raise ValueError(f"Unknown architecture variant: {variant}")
    model = make_model(d, s2d_gate=True, delta=True)
    model.pre = ResearchPlayerTower(
        len(d["vocab"]["team"]) + 1, len(d["vocab"]["conf"]) + 1,
        d["scalar_cols"], d["player_store"].manifest, variant,
        s2d_indices(d["scalar_cols"], "h"), s2d_indices(d["scalar_cols"], "a"), absence=absence)
    return model
