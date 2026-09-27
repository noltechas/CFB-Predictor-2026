"""The pure-DNN two-tower network over raw inputs.

No rating algorithm feeds this. Team strength is learned three ways at once,
all inside the network:

  1. **Team embeddings** — a vector per program, fitted during training.
  2. **A history encoder** — each side's last 14 actual results, each slot
     carrying the raw score, venue, recency AND the opponent's embedding, so
     "beat whom, by how much" is opponent-adjusted by the network itself.
  3. **Raw counting stats** — season-to-date yards per play, turnover rates,
     third-down conversion, roster composition, coach record. Counts, not fits.

Output is unchanged from the approved design: a discrete PMF over integer
margins plus one over totals; win probability is a partial sum of the margin
distribution, so the three outputs cannot disagree.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from cfbpred.config import N_MARGIN_BINS, N_REM_BINS, N_TOTAL_BINS
from cfbpred.model import (MARGIN_CENTRES, REM_CENTRES, TOTAL_CENTRES,
                           margin_bin, rem_bin, shift_pmf, total_bin)

TEAM_EMB = 24
CONF_EMB = 8
HIST_FIELDS = 7          # pf, pa, was_home, opp_fbs, days_ago, same_season, mask
# (an 8th opp_talent slot field was measured 2026-08-27: null, reverted;
#  the column remains in game_history for future retests)
HIST_D = 32
# Season-to-date stat families that may each learn their own early-season trust.
S2D_FAMILIES = (("o_",), ("d_",), ("bx_",), ("qb_",), ("ppg_", "winpct"))


class HistoryEncoder(nn.Module):
    """Masked set-encoder over one side's prior games."""

    def __init__(self, team_emb: nn.Embedding, emb_drop: nn.Dropout):
        super().__init__()
        self.team_emb = team_emb
        self.emb_drop = emb_drop
        self.slot = nn.Sequential(
            nn.Linear(HIST_FIELDS - 1 + TEAM_EMB, 48), nn.GELU(),
            nn.Linear(48, HIST_D), nn.GELU(),
        )

    def forward(self, raw, opp_idx):
        # raw: [B, N, HIST_FIELDS] with mask last; opp_idx: [B, N]
        mask = raw[..., -1:]                                  # [B, N, 1]
        x = torch.cat([raw[..., :-1], self.emb_drop(self.team_emb(opp_idx))], dim=-1)
        h = self.slot(x) * mask
        n = mask.sum(1).clamp(min=1.0)
        mean = h.sum(1) / n
        mx = (h + (mask - 1.0) * 1e4).max(1).values
        if getattr(self, "empty_is_zero", False):
            mx = torch.where(mask.sum(1).gt(0), mx, 0.)
        return torch.cat([mean, mx], dim=-1)                  # [B, 2*HIST_D]


class RawPregameTower(nn.Module):
    def __init__(self, n_teams: int, n_confs: int, n_scalar: int,
                 hidden: int = 160, emb_out: int = 32, dropout: float = 0.30,
                 s2d_idx_h=None, s2d_idx_a=None, use_s2d_gate: bool = False):
        super().__init__()
        # Early-season gate: season-to-date stat columns (o_/d_/bx_/qb_/ppg/
        # winpct) are multiplied by sigmoid(b + w * games_played/12), one gate
        # per side, two learned parameters total. Inputs are standardised, so
        # gating shrinks a 1-3-game sample toward the league mean rather than
        # toward zero — the network no longer has to learn from scratch that a
        # September yards-per-play is noise and a November one is signal.
        # Initialised to a mild prior (0.54 at zero games, 0.95 by eight);
        # applied only when use_s2d_gate is set, but the parameters and index
        # buffers always exist so state_dicts stay interchangeable.
        self.use_s2d_gate = use_s2d_gate
        self.register_buffer("s2d_idx_h", torch.as_tensor(
            s2d_idx_h if s2d_idx_h is not None else [], dtype=torch.long))
        self.register_buffer("s2d_idx_a", torch.as_tensor(
            s2d_idx_a if s2d_idx_a is not None else [], dtype=torch.long))
        self.gate_b = nn.Parameter(torch.tensor(0.15))
        self.gate_w = nn.Parameter(torch.tensor(4.2))
        self.team_emb = nn.Embedding(n_teams, TEAM_EMB)
        self.conf_emb = nn.Embedding(n_confs, CONF_EMB)
        nn.init.normal_(self.team_emb.weight, std=0.05)
        nn.init.normal_(self.conf_emb.weight, std=0.05)
        # Identity dropout. Without it the embedding table becomes a lookup key
        # for memorising training-game outcomes: the first run of this
        # architecture hit validation log loss 0.63 at epoch 1, degrading every
        # epoch after, with the gate learning to trust the memorised tower.
        # Zeroing whole identity vectors 30% of the time forces the tower to
        # rely on history and counting stats, which generalise.
        self.emb_drop = nn.Dropout1d(0.30)
        # The scalar block is 230+ standardised, partly collinear features.
        # Without input dropout, widening it from 204 to 231 columns made the
        # early-stopped tower overfit FASTER and cost 0.45 of pregame RMSE —
        # more information, worse model. Randomly hiding 10% of inputs forces
        # redundancy instead of reliance.
        self.scalar_drop = nn.Dropout(0.0)   # tested at 0.10: worse
        self.hist = HistoryEncoder(self.team_emb, self.emb_drop)
        n_in = (2 * TEAM_EMB + 2 * CONF_EMB + 2 * (2 * HIST_D) + n_scalar)
        self.body = nn.Sequential(
            nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.emb = nn.Linear(hidden, emb_out)
        self.margin = nn.Linear(hidden, N_MARGIN_BINS)
        self.total = nn.Linear(hidden, N_TOTAL_BINS)

    def enable_family_gates(self, scalar_cols):
        """One learned (bias, slope) per stat family instead of one for all 78 gated
        columns: yards per play may earn trust after three games while turnover or
        explosive-play rates stay shrunk longer. Initialised to the shared gate, so
        training starts from exactly the single-gate model."""
        def family(i, side):
            key = scalar_cols[int(i)][len(side) + 1:]
            for f, prefixes in enumerate(S2D_FAMILIES):
                if key.startswith(prefixes):
                    return f
            raise ValueError(f"No season-to-date family for {scalar_cols[int(i)]}")
        self.register_buffer("gate_family_h", torch.tensor([family(i, "h") for i in self.s2d_idx_h], dtype=torch.long))
        self.register_buffer("gate_family_a", torch.tensor([family(i, "a") for i in self.s2d_idx_a], dtype=torch.long))
        n = len(S2D_FAMILIES)
        self.gate_fb = nn.Parameter(torch.full((n,), float(self.gate_b.detach())))
        self.gate_fw = nn.Parameter(torch.full((n,), float(self.gate_w.detach())))

    def forward(self, scalars, h_idx, a_idx, h_conf, a_conf,
                h_hist, h_opp, a_hist, a_opp, gp_h=None, gp_a=None):
        if self.use_s2d_gate and gp_h is not None:
            if getattr(self, "gate_fb", None) is not None:
                gh = torch.sigmoid(self.gate_fb[self.gate_family_h] + self.gate_fw[self.gate_family_h] * gp_h[:, None])
                ga = torch.sigmoid(self.gate_fb[self.gate_family_a] + self.gate_fw[self.gate_family_a] * gp_a[:, None])
            else:
                gh = torch.sigmoid(self.gate_b + self.gate_w * gp_h).unsqueeze(-1)
                ga = torch.sigmoid(self.gate_b + self.gate_w * gp_a).unsqueeze(-1)
            mult = scalars.new_ones(scalars.shape)
            mult[:, self.s2d_idx_h] = gh
            mult[:, self.s2d_idx_a] = ga
            scalars = scalars * mult
        z = torch.cat([
            self.emb_drop(self.team_emb(h_idx)), self.emb_drop(self.team_emb(a_idx)),
            self.conf_emb(h_conf), self.conf_emb(a_conf),
            self.hist(h_hist, h_opp), self.hist(a_hist, a_opp),
            self.scalar_drop(scalars)], dim=-1)
        h = self.body(z)
        return self.emb(h), self.margin(h), self.total(h)


class StateTower(nn.Module):
    def __init__(self, n_state: int, emb_in: int = 2, hidden: int = 128,
                 dropout: float = 0.20, n_total_out: int = N_TOTAL_BINS):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_state + emb_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.margin = nn.Linear(hidden, N_MARGIN_BINS)
        self.total = nn.Linear(hidden, n_total_out)

    def forward(self, state, emb):
        h = self.body(torch.cat([state, emb], dim=-1))
        return self.margin(h), self.total(h)


class Gate(nn.Module):
    """alpha(time, |lead|) in [floor, 1], monotone by construction. Initialised
    to a slow decay (the measured CFB optimum, not the NFL's), then learned."""

    def __init__(self, floor: float = 0.02):
        super().__init__()
        self.floor = floor
        self.w_time = nn.Parameter(torch.tensor(0.9))
        self.w_lead = nn.Parameter(torch.tensor(-1.6))
        self.bias = nn.Parameter(torch.tensor(-0.4))
        # exponent of the fr ramp; forces alpha -> floor as the clock hits
        # zero. Measured need: the learned gate kept alpha=0.48 at t=6s,
        # diluting a near-certain in-game verdict with the pregame prior
        # (3-point leads at the final snap: model 73%, reality 96%).
        self.w_sharp = nn.Parameter(torch.tensor(0.0))

    def forward(self, frac_remaining, abs_lead_scaled):
        z = (nn.functional.softplus(self.w_time) * frac_remaining
             - nn.functional.softplus(self.w_lead) * abs_lead_scaled
             + self.bias)
        ramp = frac_remaining.clamp(0.0, 1.0).pow(
            nn.functional.softplus(self.w_sharp))
        return self.floor + (1.0 - self.floor) * torch.sigmoid(z) * ramp


class RawTwoTower(nn.Module):
    """delta_heads=False: both towers predict FINAL margin/total directly
    (the v20-and-earlier design).

    delta_heads=True: the state tower predicts the distribution of what is
    LEFT — margin change from now to final, and remaining points. The frozen
    pregame tower still speaks in final space (pregame, remaining IS final);
    its PMF is shifted into delta space by the live score before the gate
    blends, and the blend is shifted back to final space on the way out, so
    every consumer sees the same final-space grids as before.

    Why (measured on the v20 ensemble, held-out final-30s snaps): predicting
    finals from scratch forced the tower to relearn "the score so far is a
    floor" at every score level — 40% of its total mass sat below the points
    already on the board, and it put 4.7% on "no further scoring" against an
    empirical 78.5%. In delta space that atom is ONE shared bin (0), pooled
    across every game ending ever seen, and sub-floor totals are not a
    pattern to learn but bins that do not exist."""

    def __init__(self, n_teams: int, n_confs: int, n_scalar: int, n_state: int,
                 state_hidden: int = 128, s2d_idx_h=None, s2d_idx_a=None,
                 use_s2d_gate: bool = False, pre_hidden: int = 160,
                 delta_heads: bool = False):
        super().__init__()
        self.delta_heads = delta_heads
        self.pre = RawPregameTower(n_teams, n_confs, n_scalar,
                                   hidden=pre_hidden,
                                   s2d_idx_h=s2d_idx_h, s2d_idx_a=s2d_idx_a,
                                   use_s2d_gate=use_s2d_gate)
        self.state = StateTower(
            n_state, hidden=state_hidden,
            n_total_out=N_REM_BINS if delta_heads else N_TOTAL_BINS)
        self.gate = Gate()

    def pregame(self, batch):
        if getattr(self.pre, "uses_players", False):
            return self.pre.forward_batch(batch)
        return self.pre(batch["scalars"], batch["h_idx"], batch["a_idx"],
                        batch["h_conf"], batch["a_conf"],
                        batch["h_hist"], batch["h_opp"],
                        batch["a_hist"], batch["a_opp"],
                        batch.get("h_gp01"), batch.get("a_gp01"))

    def forward(self, batch, state, frac_remaining, abs_lead_scaled,
                lead=None, pts=None):
        emb, pm, pt = self.pregame(batch)
        # The state tower gets exactly two numbers from pregame: the expected
        # margin and total. Passing the full embedding handed it a near-unique
        # per-game fingerprint, and with 1M plays per epoch it memorised
        # training outcomes through it (validation degraded monotonically).
        # Two coarse, noised scalars carry the "how good are these teams"
        # signal and nothing else.
        with torch.no_grad():
            em = (torch.softmax(pm, -1) * MARGIN_CENTRES.to(pm.device)).sum(-1)
            et = (torch.softmax(pt, -1) * TOTAL_CENTRES.to(pt.device)).sum(-1)
        cond = torch.stack([em / 20.0, (et - 55.0) / 20.0], dim=-1)
        if self.training:
            cond = cond + 0.05 * torch.randn_like(cond)
        sm, st = self.state(state, cond)
        a = self.gate(frac_remaining, abs_lead_scaled).unsqueeze(-1)
        if not self.delta_heads:
            return {"margin": a * pm + (1 - a) * sm,
                    "total": a * pt + (1 - a) * st,
                    "pre_margin": pm, "pre_total": pt, "alpha": a.squeeze(-1)}
        if lead is None or pts is None:
            raise ValueError("delta_heads model needs lead and pts")
        eps = 1e-9
        # pregame PMFs -> delta space (identity at kickoff: lead = pts = 0)
        pre_m = torch.log(shift_pmf(torch.softmax(pm, -1), MARGIN_CENTRES,
                                    -lead, margin_bin, N_MARGIN_BINS) + eps)
        pre_t = torch.log(shift_pmf(torch.softmax(pt, -1), TOTAL_CENTRES,
                                    -pts, rem_bin, N_REM_BINS) + eps)
        bm = a * pre_m + (1 - a) * sm            # blend in delta space,
        bt = a * pre_t + (1 - a) * st            # log-space as before
        # back to final space: log-probs are valid logits downstream
        out_m = torch.log(shift_pmf(torch.softmax(bm, -1), MARGIN_CENTRES,
                                    lead, margin_bin, N_MARGIN_BINS) + eps)
        out_t = torch.log(shift_pmf(torch.softmax(bt, -1), REM_CENTRES,
                                    pts, total_bin, N_TOTAL_BINS) + eps)
        return {"margin": out_m, "total": out_t,
                "pre_margin": pm, "pre_total": pt, "alpha": a.squeeze(-1)}
