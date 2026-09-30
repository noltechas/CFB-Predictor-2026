"""Why the pregame model produced a given number, input by input.

Two different questions need two different counterfactuals, so this module answers
both and never mixes them.

**Team evidence** (record, results, recruiting, site, coaching, stats) is measured
against an AVERAGE FBS TEAM: every scalar at its FBS mean, the mean FBS program and
conference embedding, no prior results, structural flags off. Those inputs interact
strongly, so credit is split by sampled Shapley over structured permutations, and the
contributions sum exactly to the prediction minus the all-reference baseline.

**Players** are measured against a LEAGUE-AVERAGE PLAYER AT THE SAME POSITION, with the
roster slot and the player's own usage left in place. Deleting a roster slot instead
would change position-group counts and pooling statistics, which reads as a huge player
effect when it is really an effect of the slot existing. Under the honest counterfactual
players barely interact, so each one is an exact leave-one-out, not a sampled estimate.

Nothing here changes a prediction; it only re-scores the active checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from cfbpred.config import CURRENT_SEASON, MODELS
from cfbpred.model import MARGIN_CENTRES, TOTAL_CENTRES
from cfbpred.model_card import group_of
from cfbpred.player_inputs import POSITIONS
from cfbpred.player_sequence_inputs import ROLE_COLS
from cfbpred.player_sequence_net import safe_lookup
from cfbpred.player_sequence_production import (ACTIVE, _checkpoint_inputs, _make_model,
                                                active_bundle, bundle_scalar_extras, context_source, load_snapshot)
from cfbpred.pregame_experiments import save_json
from cfbpred.train import game_batch

CHUNK = 256
FLAGS_OFF = ("is_home_site", "home_is_fcs", "opp_is_fcs", "is_conf_game", "is_postseason")
MUTABLE = ("scalars", "h_idx", "a_idx", "h_conf", "a_conf", "h_hist", "a_hist", "h_gp01", "a_gp01")
QB_COLS = {"qb_att_pg", "qb_ypa", "qb_int_rate", "qb_sack_rate", "qb_td_rate", "qb_changed",
           "qb_known", "av_qb_played_last", "cx_qb_gms", "cx_qb_vol", "QB_n", "QB_wt", "QB_ht", "retp_qb1"}

# Human labels. Prefix rules cover the families; the dict overrides the ones worth naming.
EXACT = {
    "gp": "Games played", "winpct": "Win rate", "ppg_for": "Points scored per game",
    "ppg_against": "Points allowed per game", "rest": "Days of rest", "travel": "Travel distance (miles)",
    "alt_delta": "Altitude change (ft)", "week_scaled": "Week of season", "is_home_site": "Home field",
    "is_conf_game": "Conference game", "is_postseason": "Postseason", "home_is_fcs": "Home team is FCS",
    "opp_is_fcs": "Opponent is FCS", "n_players": "Roster size", "returning_share": "Roster continuity",
    "upper_share": "Upperclassman share", "talent_composite": "Talent composite (247)",
    "recruit_pts_4c": "Recruiting points, last 4 classes", "recruit_pts_latest": "Recruiting points, latest class",
    "recruit_rank_best": "Best recruiting class rank", "coach_tenure": "Coach tenure at school",
    "coach_first_year": "First-year coach", "coach_career_games": "Coach career games",
    "coach_career_winpct": "Coach career win rate", "sos_talent": "Opponents faced: mean talent",
    "tpct_talent": "Talent rank among FBS (247, percentile)", "tpct_known": "Talent rank published",
    "sos_tpct_talent": "Opponents faced: mean talent rank",
    "mvmt_in_n": "Transfers in", "mvmt_in_stars": "Transfers in: total stars", "mvmt_in_top": "Transfers in: 4-5 stars",
    "mvmt_in_rating": "Transfers in: total 247 rating", "mvmt_out_n": "Transfers out",
    "mvmt_out_stars": "Transfers out: total stars", "mvmt_out_top": "Transfers out: 4-5 stars",
    "mvmt_out_rating": "Transfers out: total 247 rating", "mvmt_draft_n": "Players drafted last spring",
    "mvmt_draft_r1": "First-round picks last spring", "mvmt_draft_r13": "Round 1-3 picks last spring",
    "mvmt_portal_known": "Transfer portal data available", "cchg_changed": "Head coach changed this season",
    "cchg_new_since_last": "New head coach since last game",
    "sos_recruit": "Opponents faced: mean recruiting", "sos_fbs_share": "Share of opponents that were FBS",
    "qb_att_pg": "QB attempts per game", "qb_ypa": "QB yards per attempt", "qb_int_rate": "QB interception rate",
    "qb_sack_rate": "QB sack rate", "qb_td_rate": "QB touchdown rate", "qb_changed": "Starting QB changed",
    "qb_known": "QB identified", "retp_qb1": "QB1 returned from last season",
    "cx_qb_gms": "QB career games", "cx_qb_vol": "QB career volume",
    "av_qb_played_last": "QB played in the previous game",
}
PREFIX = [
    ("o_", "Offense: ", ""), ("d_", "Defense: ", " allowed"), ("bx_", "Box score: ", ""),
    ("retp_", "Returning share of ", ""), ("port_", "Transfers in: ", ""), ("av_", "Availability: ", ""),
    ("cx_", "Career: ", ""), ("sos_", "Opponents faced: ", ""), ("match_r5_", "Recent form: ", ""),
    ("pre_qb_", "Last season's QB: ", ""), ("has_", "Data present: ", ""), ("coach_", "Coach: ", ""),
]
# Stat shorthand as it appears inside a column name.
TERMS = {
    "ypp": "yards per play", "ypc": "yards per carry", "ypa": "yards per attempt",
    "expl20": "explosive plays (20+ yards)", "expl": "explosive plays", "to": "turnover",
    "int": "interception", "td": "touchdown", "fum": "fumble", "sack": "sack",
    "pen": "penalty", "third": "third down", "fourth": "fourth down", "rz": "red zone",
    "pass": "passing", "rush": "rushing", "recv": "receiving", "rec": "receiving",
    "def": "defensive", "tkl": "tackles", "tfl": "tackles for loss", "pd": "passes defended",
    "hur": "hurries", "fg": "field goal", "xp": "extra point", "punt": "punting",
    "kr": "kick return", "pr": "punt return", "gp": "games played", "pg": "per game",
    "rate": "rate", "share": "share", "pct": "percentage", "att": "attempts",
    "cmp": "completions", "yds": "yards", "top": "top-player", "conv": "conversion",
    "style": "tendency", "diff": "difference vs opponent", "n": "count",
    "wt": "mean weight", "ht": "mean height", "vol": "volume", "gms": "games",
}


def _words(body: str) -> str:
    return " ".join(TERMS.get(part, part) for part in body.split("_"))


# Columns that are genuinely yes/no. Everything else is a measurement, even when it
# happens to be exactly 0 or 1 — an undefeated team's win rate is 1.000, not "yes".
FLAG_KEYS = {"is_home_site", "home_is_fcs", "opp_is_fcs", "is_conf_game", "is_postseason",
             "qb_changed", "qb_known", "coach_first_year", "retp_qb1", "av_qb_played_last",
             "pre_qb_passer_known", "pre_qb_roster_known", "tpct_known", "mvmt_portal_known",
             "cchg_changed", "cchg_new_since_last"}


def value_format(col: str) -> str:
    key = strip(col)
    return "flag" if key in FLAG_KEYS or key.startswith("has_") else "number"


def strip(col: str) -> str:
    return col[2:] if col[:2] in ("h_", "a_") else col


def side_of(col: str) -> str:
    return {"h": "home", "a": "away"}.get(col[0] if col[:2] in ("h_", "a_") else "", "shared")


def label_for(col: str) -> str:
    key = strip(col)
    if key in EXACT:
        return EXACT[key]
    if "_" in key and key.split("_")[0] in POSITIONS:
        unit, rest = key.split("_", 1)
        return f"{unit} unit: {_words(rest)}"
    for pre, head, tail in PREFIX:
        if key.startswith(pre):
            return f"{head}{_words(key[len(pre):])}{tail}".strip()
    return _words(key)


def block_of(col: str) -> str:
    key = strip(col)
    if key.startswith("match_r5"):
        return "Recent form"
    if key.startswith("pre_qb") or key in QB_COLS:
        return "Quarterback"
    return {"Play-by-play rates, offense": "Offense rates", "Play-by-play rates, defense": "Defense rates",
            "Box-score lines, every position": "Box score", "Recruiting & talent": "Recruiting & talent",
            "Schedule context": "Schedule faced", "Known-data flags": "Data coverage",
            "Star availability": "Continuity", "Returning production": "Continuity",
            "Career experience": "Roster experience", "Roster composition": "Roster make-up",
            }.get(group_of(col), group_of(col))


def _roster_forward(self, batch):
    """FullRosterEncoder.forward_batch with a per-slot identity replacement."""
    scout, history, position, mask = (batch[k] for k in
                                      ("player_scout", "player_history", "player_position", "player_mask"))
    slots = batch["player_slots"]
    pos = position.reshape(-1)[slots]
    hist = self.history(history).flatten(-2)
    packed = self.player(torch.cat([scout, hist, safe_lookup(self.position, pos)], -1)) + self.career(batch)
    context = batch.get("player_context")
    if self.context_dim:
        packed = packed + self.context(context)
    replace = batch.get("player_replace")
    if replace is not None:
        flat = replace.reshape(-1)[slots]
        packed = torch.where(flat[:, None], self.group_average[pos], packed)
    self.captured = (packed.detach(), pos.detach())
    x = packed.new_zeros((mask.numel(), 12)).index_copy(0, slots, packed).reshape(*mask.shape, 12)
    logits = None
    if self.roles:
        role = batch["player_role"].reshape(-1, len(ROLE_COLS))[slots]
        fields = [packed, role] + ([context] if self.context_dim else [])
        scores = self.role(torch.cat(fields, -1)).squeeze(-1)
        logits = scores.new_full((mask.numel(),), -1e4).index_copy(0, slots, scores).reshape(mask.shape)
    available = mask & ~batch.get("player_confirmed_out", torch.zeros_like(mask))
    groups = []
    self.unit_share = torch.zeros_like(mask, dtype=x.dtype)
    for group in range(self.n_positions):
        valid = mask & position.eq(group)
        pooled = (valid & available) if getattr(self, "pool_available", False) else valid
        count = pooled.sum(-1, keepdim=True)
        mean = (x * pooled[..., None]).sum(-2) / count.clamp(min=1)
        maximum = x.masked_fill(~pooled[..., None], -1e4).amax(-2)
        maximum = torch.where(count.gt(0), maximum, 0.)
        summary = self.group(torch.cat([mean, maximum, torch.log1p(count.float()) / 4], -1))
        if logits is not None:
            active = valid & available
            weights = logits.masked_fill(~active, -1e4).softmax(-1) * active
            weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-8)
            self.unit_share = self.unit_share + weights.detach()
            summary = summary + self.weighted_group((x * weights[..., None]).sum(-2))
        groups.append(summary * count.gt(0))
    return self.team(torch.cat(groups, -1)) * mask.any(-1, keepdim=True), x


class Context:
    """Loaded model, stores and the two reference states."""


class _Ensemble:
    """Seed ensemble behind the single-model interface the attribution uses: pregame()
    returns the log of the probability mass averaged over members, which is exactly how
    production scores a multi-seed bundle."""

    def __init__(self, members):
        self.members = members

    @property
    def pre(self):
        return self.members[0].pre

    def pregame(self, batch):
        pm = pt = None
        for m in self.members:
            _, a, b = m.pregame(batch)
            a, b = a.softmax(-1), b.softmax(-1)
            pm = a if pm is None else pm + a
            pt = b if pt is None else pt + b
        n = len(self.members)
        return None, torch.log((pm / n).clamp(min=1e-12)), torch.log((pt / n).clamp(min=1e-12))

    @property
    def unit_share(self):
        return torch.stack([m.pre.encoder.unit_share for m in self.members]).mean(0)


def load(active_path: Path = ACTIVE, reference_games: int = 200, threads: int = 6) -> Context:
    torch.set_num_threads(threads)
    c = Context()
    c.active, bundle_path, bundle = active_bundle(active_path)
    runtime = bundle_path.parent / "runtime"
    c.inputs, c.players, c.sequences = runtime / "inputs", runtime / "players", runtime / "sequences"
    c.base = load_snapshot(c.inputs, c.players, c.sequences, context_source(c.inputs, bundle), bundle_scalar_extras(bundle))
    members = []
    for name in bundle["members"]:
        checkpoint = torch.load(bundle_path.parent / name, map_location="cpu", weights_only=False)
        d = _checkpoint_inputs(c.base, checkpoint, c.players)
        if members and not (np.array_equal(d["mu"], c.d["mu"]) and np.array_equal(d["sd"], c.d["sd"])):
            raise ValueError("Ensemble members were fitted on different input scalings")
        c.d = d
        m = _make_model(d, checkpoint["architecture"], checkpoint["variant"])
        m.pre.load_state_dict(checkpoint["state_dict"]); m.eval(); members.append(m)
    c.model = _Ensemble(members)
    c.cols = c.d["scalar_cols"]
    c.mu, c.sd = np.asarray(c.d["mu"]), np.asarray(c.d["sd"])
    c.schedule = pd.read_parquet(c.inputs / "snapshot_schedule.parquet").set_index("game_id")
    c.inv_team = {v: k for k, v in c.d["vocab"]["team"].items()}
    c.inv_conf = {v: k for k, v in c.d["vocab"]["conf"].items()}
    c.inv_pos = {v: k for k, v in POSITIONS.items()}
    c.catalog = pd.read_parquet(c.d["player_store"].path / "player_seasons.parquet",
                                columns=["firstName", "lastName", "position"])

    labelled = (np.asarray(c.base["G_fbs_fbs"]) & np.asarray(c.base["G_labeled"])
                & (np.asarray(c.base["G_season"]) <= CURRENT_SEASON - 1))
    raw = np.asarray(c.base["G_raw_scalars"])
    c.reference = {}
    fill = np.zeros(len(c.cols), np.float32)
    for i, col in enumerate(c.cols):
        if col in FLAGS_OFF:
            value = 0.0
        else:
            pair = ([c.cols.index("h_" + strip(col)), c.cols.index("a_" + strip(col))]
                    if col[:2] in ("h_", "a_") else [i])
            values = raw[labelled][:, pair].ravel()
            values = values[np.isfinite(values)]
            value = float(values.mean()) if len(values) else float(c.mu[i])
        c.reference[col] = value
        fill[i] = (value - c.mu[i]) / c.sd[i]
    c.fill = torch.tensor(fill)
    c.reference_gp01 = float(np.r_[c.d["G_gp01_h"].numpy()[labelled], c.d["G_gp01_a"].numpy()[labelled]].mean())

    teams = np.unique(np.r_[c.d["G_hidx"].numpy()[labelled], c.d["G_aidx"].numpy()[labelled]])
    confs = np.unique(np.r_[c.d["G_hconf"].numpy()[labelled], c.d["G_aconf"].numpy()[labelled]])

    def extend(embedding, rows):
        wider = torch.nn.Embedding(embedding.num_embeddings + 1, embedding.embedding_dim)
        with torch.no_grad():
            wider.weight[:-1] = embedding.weight
            wider.weight[-1] = embedding.weight[torch.as_tensor(rows)].mean(0)
        return wider

    for m in c.model.members:   # each seed gets its own mean-FBS program and conference row
        core = m.pre.core
        core.team_emb = extend(core.team_emb, teams)
        core.hist.team_emb = core.team_emb
        core.conf_emb = extend(core.conf_emb, confs)
    c.team_reference = c.model.pre.core.team_emb.num_embeddings - 1
    c.conf_reference = c.model.pre.core.conf_emb.num_embeddings - 1

    c.absence = bool(getattr(c.model.pre, "absence", False))
    if c.absence:
        # Wrap (never edit) the tower's availability inputs so a counterfactual can
        # reset one side to "no report on file", the state most training games are in.
        import cfbpred.architecture_research_net as arn
        if not getattr(arn.absence_features, "explain_wrapped", False):
            original = arn.absence_features

            def wrapped(batch, context_cols, _original=original):
                values = _original(batch, context_cols)
                zero = batch.get("absence_zero")
                return values if zero is None else torch.where(zero[..., None], torch.zeros_like(values), values)
            wrapped.explain_wrapped = True
            arn.absence_features = wrapped
    pool = np.where(np.asarray(c.base["G_fbs_fbs"]) & (np.asarray(c.base["G_season"]) == CURRENT_SEASON))[0]
    sample = np.random.default_rng(0).choice(pool, size=min(reference_games, len(pool)), replace=False)
    for m in c.model.members:   # a league-average player in THIS seed's embedding space
        encoder = m.pre.encoder
        encoder.group_average = torch.zeros(len(POSITIONS), 12)
        encoder.forward_batch = types.MethodType(_roster_forward, encoder)
        total = torch.zeros(len(POSITIONS), 12)
        count = torch.zeros(len(POSITIONS))
        with torch.no_grad():
            for start in range(0, len(sample), 64):
                batch = game_batch(c.d, torch.tensor(sample[start:start + 64]), torch.device("cpu"))
                m.pregame(batch)
                packed, pos = encoder.captured
                total.index_add_(0, pos, packed)
                count.index_add_(0, pos, torch.ones(len(pos)))
        encoder.group_average = total / count.clamp(min=1)[:, None]
    c.reference_players = {c.inv_pos[i]: int(n) for i, n in enumerate(count)}
    c.members = len(c.model.members)
    return c


def _readouts(logits_margin, logits_total, rows):
    p, q = logits_margin.softmax(-1)[:rows], logits_total.softmax(-1)[:rows]
    return np.stack([(p * MARGIN_CENTRES).sum(-1).numpy(),
                     (p * (MARGIN_CENTRES > 0)).sum(-1).numpy(),
                     (q * TOTAL_CENTRES).sum(-1).numpy()], -1)


def _features(c: Context, position: int):
    """Every team-level input of one game, with its value and its average-FBS reference."""
    clean = game_batch(c.d, torch.full((CHUNK,), int(position)), torch.device("cpu"))
    raw = np.asarray(c.base["G_raw_scalars"][position])
    game_id = int(c.base["G_gid"][position])
    home, away = c.schedule.loc[game_id, "home_team"], c.schedule.loc[game_id, "away_team"]
    feats = []
    for i, col in enumerate(c.cols):
        feats.append(dict(kind="scalar", index=i, key=strip(col), block=block_of(col), side=side_of(col),
                          label=label_for(col), column=col, format=value_format(col),
                          value=None if not np.isfinite(raw[i]) else float(raw[i]),
                          reference=c.reference[col]))
    feats.append(dict(kind="pair", keys=("h_idx", "a_idx"), reference_index=c.team_reference, key="team_embedding",
                      block="Program identity", side="both", label="Who these programs are", format="text",
                      value=f"{c.inv_team.get(int(clean['h_idx'][0]), 'unknown')} vs "
                            f"{c.inv_team.get(int(clean['a_idx'][0]), 'unknown')}",
                      reference="an average FBS program"))
    feats.append(dict(kind="pair", keys=("h_conf", "a_conf"), reference_index=c.conf_reference, key="conference",
                      block="Conference", side="both", label="Conference", format="text",
                      value=f"{c.inv_conf.get(int(clean['h_conf'][0]), 'unknown')} vs "
                            f"{c.inv_conf.get(int(clean['a_conf'][0]), 'unknown')}",
                      reference="an average FBS conference"))
    if getattr(c, "absence", False):
        from cfbpred.architecture_research_net import ABSENCE_NAMES, absence_features
        from cfbpred.player_timeline import CONTEXT_COLS
        with torch.no_grad():   # the clean batch is CHUNK identical copies of this game
            f = absence_features(clean, CONTEXT_COLS)[0].numpy()
        ix = {n: i for i, n in enumerate(ABSENCE_NAMES)}
        for side_index, side in ((0, "home"), (1, "away")):
            v = f[side_index]
            if v[ix["report_known"]] <= 0:
                text = "no report on file"
            else:
                n_out = int(round(np.expm1(v[ix["players_out_log"]] * 3)))
                shares = [f"{v[ix['out_share_' + s]] * 100:.0f}% of {label}" for s, label in
                          (("att", "passing"), ("rush_att", "carries"), ("rec", "receptions"), ("tackles", "tackles"))
                          if v[ix["out_share_" + s]] >= 0.005]
                text = f"{n_out} out" + (": " + ", ".join(shares) if shares else ", none with recent usage")
                if v[ix["unsure_share_att"]] >= 0.005:
                    text += f"; {v[ix['unsure_share_att']] * 100:.0f}% of passing doubtful or questionable"
            feats.append(dict(kind="absence", side_index=side_index, key="availability_report", block="Availability",
                              side=side, label="Players reported out", format="text", value=text,
                              reference="no report on file"))
    for side_index, side, key, opponent in ((0, "home", "h_hist", "h_opp"), (1, "away", "a_hist", "a_opp")):
        slots = clean[key][0]
        opponents = clean[opponent][0]
        for k in range(slots.shape[0]):
            if slots[k, -1] <= 0:
                continue
            scored, allowed = float(slots[k, 0]) * 30, float(slots[k, 1]) * 30
            at_home, opponent_fbs = slots[k, 2] > 0.5, slots[k, 3] > 0.5
            days, same_season = float(slots[k, 4]) * 100, slots[k, 5] > 0.5
            result = "W" if scored > allowed else "L" if scored < allowed else "T"
            when = "Most recent game" if k == 0 else f"{k + 1} games ago"
            feats.append(dict(kind="history", key="prior_result", history_key=key, slot=k, block="Recent results",
                              side=side, format="text",
                              label=f"{when}{'' if same_season else ' (last season)'}",
                              value=f"{result} {scored:.0f}-{allowed:.0f} "
                                    f"{'vs' if at_home else 'at'} {c.inv_team.get(int(opponents[k]), 'unknown')}"
                                    f"{'' if opponent_fbs else ' (FCS)'}",
                              reference="no game on record", days_ago=round(days)))
    return feats, clean, home, away, game_id


def _team_attribution(c: Context, position: int, feats, clean, permutation_pairs: int, seed: int):
    n = len(feats)
    scalars = [k for k, f in enumerate(feats) if f["kind"] == "scalar"]
    columns = torch.tensor([feats[k]["index"] for k in scalars])
    model = c.model

    def evaluate(absent):
        rows = absent.shape[0]
        flags = torch.as_tensor(absent)
        batch = dict(clean)
        for key in MUTABLE:
            batch[key] = clean[key].clone()
        mask = torch.zeros(rows, len(c.cols), dtype=torch.bool)
        mask[:, columns] = flags[:, scalars]
        if getattr(c, "absence", False):
            zero = torch.zeros(CHUNK, 2, dtype=torch.bool)
            for k, f in enumerate(feats):
                if f["kind"] == "absence":
                    zero[:rows, f["side_index"]] |= flags[:, k]
            batch["absence_zero"] = zero
        batch["scalars"][:rows] = torch.where(mask, c.fill[None, :], batch["scalars"][:rows])
        for k, f in enumerate(feats):
            if f["kind"] == "pair":
                for key in f["keys"]:
                    batch[key][:rows] = torch.where(flags[:, k], f["reference_index"], batch[key][:rows])
            elif f["kind"] == "history":
                batch[f["history_key"]][:rows, f["slot"], -1] *= (~flags[:, k]).float()
            elif f["kind"] == "scalar" and f["key"] == "gp":
                gate = "h_gp01" if f["side"] == "home" else "a_gp01"
                batch[gate][:rows] = torch.where(flags[:, k], torch.tensor(c.reference_gp01), batch[gate][:rows])
        with torch.no_grad():
            _, margin, total = model.pregame(batch)
        return _readouts(margin, total, rows)

    def walk(order):
        absent = np.ones((n + 1, n), bool)
        for step, k in enumerate(order):
            absent[step + 1:, k] = False
        out = np.zeros((n + 1, 3))
        for start in range(0, n + 1, CHUNK):
            block = absent[start:start + CHUNK]
            padded = (np.concatenate([block, np.zeros((CHUNK - len(block), n), bool)])
                      if len(block) < CHUNK else block)
            out[start:start + CHUNK] = evaluate(padded)[:len(block)]
        return out

    rng = np.random.default_rng(seed)
    blocks = sorted({f["block"] for f in feats})
    members = {b: [k for k, f in enumerate(feats) if f["block"] == b] for b in blocks}
    contributions = np.zeros((2 * permutation_pairs, n, 3))
    full = empty = None
    for pair in range(permutation_pairs):
        order = np.concatenate([rng.permutation(members[b]) for b in rng.permutation(blocks)])
        for offset, direction in enumerate((order, order[::-1])):
            values = walk(direction)
            contributions[2 * pair + offset, direction] = values[1:] - values[:-1]
            if full is None:
                full, empty = values[-1], values[0]
    return contributions.mean(0), contributions.std(0, ddof=1) / math.sqrt(contributions.shape[0]), full, empty


def _player_attribution(c: Context, position: int, clean, home, away):
    """Exact leave-one-out: swap one player for a league-average player at his position."""
    store = c.d["player_store"]
    refs, _ = store.roster_refs(np.array([position]))
    mask = clean["player_mask"][0]
    groups = clean["player_position"][0]
    context = clean.get("player_context")
    slots = clean["player_slots"].numpy()
    per_slot = {int(s): context[r].numpy() for r, s in enumerate(slots) if s < 2 * mask.shape[1]} if context is not None else {}
    from cfbpred.player_timeline import CONTEXT_COLS
    with torch.no_grad():
        _, margin, total = c.model.pregame(game_batch(c.d, torch.tensor([position]), torch.device("cpu")))
    reference = _readouts(margin, total, 1)[0]
    share = c.model.unit_share[0].numpy()   # [2, P] learned share of each unit, seed-averaged
    rows = []
    for side_index, side, team in ((0, "home", home), (1, "away", away)):
        present = [p for p in range(mask.shape[1]) if mask[side_index, p]]
        for start in range(0, len(present), 64):
            chunk = present[start:start + 64]
            batch = game_batch(c.d, torch.tensor([position] * len(chunk)), torch.device("cpu"))
            replace = torch.zeros_like(batch["player_mask"])
            for row, p in enumerate(chunk):
                replace[row, side_index, p] = True
            batch["player_replace"] = replace
            with torch.no_grad():
                _, m, t = c.model.pregame(batch)
            out = _readouts(m, t, len(chunk))
            for row, p in enumerate(chunk):
                reference_row = int(store.player_refs[refs[0, side_index, p]])
                person = c.catalog.iloc[reference_row - 1]
                cell = per_slot.get(side_index * mask.shape[1] + p)
                status = "unknown"
                if cell is not None:
                    for name in ("out", "doubtful", "questionable", "probable", "available"):
                        if cell[CONTEXT_COLS.index(name)] > 0:
                            status = name
                rows.append(dict(side=side, team=team, name=f"{person.firstName} {person.lastName}".strip(),
                                 position=c.inv_pos[int(groups[side_index, p])],
                                 listed_position=(person.position if isinstance(person.position, str) else None),
                                 availability=status,
                                 unit_share=float(share[side_index, p]),
                                 win_prob_pp=float((reference[1] - out[row][1]) * 100),
                                 margin_pts=float(reference[0] - out[row][0])))
    frame = pd.DataFrame(rows)
    # Depth order: the encoder's learned share of the unit, which tracks the starter,
    # not the size of the swap effect, which is noise-level for this model.
    for (side, group), part in frame.groupby(["side", "position"]):
        order = part.unit_share.sort_values(ascending=False).index
        frame.loc[order, "rank_in_group"] = np.arange(1, len(order) + 1)
    return frame


def explain(c: Context, game_id: int, permutation_pairs: int = 48, seed: int = 0) -> dict:
    position = int(np.where(c.base["G_gid"] == game_id)[0][0])
    started = time.monotonic()
    feats, clean, home, away, game_id = _features(c, position)
    phi, se, full, empty = _team_attribution(c, position, feats, clean, permutation_pairs, seed)
    players = _player_attribution(c, position, clean, home, away)

    table = pd.DataFrame(feats)
    if "slot" not in table:
        table["slot"] = np.nan
    table["win_prob_pp"] = phi[:, 1] * 100
    table["margin_pts"] = phi[:, 0]
    table["total_pts"] = phi[:, 2]
    table["standard_error_pp"] = se[:, 1] * 100
    blocks = []
    order = table.groupby("block").win_prob_pp.sum().abs().sort_values(ascending=False).index
    for name in order:
        part = table[table.block == name]
        part = part.reindex(part.win_prob_pp.abs().sort_values(ascending=False).index)
        blocks.append(dict(
            block=name, win_prob_pp=round(float(part.win_prob_pp.sum()), 4),
            margin_pts=round(float(part.margin_pts.sum()), 4), inputs=[
                dict(label=row.label, side=row.side, key=row.key, format=row.format,
                     order=(int(row.slot) if row.kind == "history" else None),
                     value=(None if row.value is None or (isinstance(row.value, float) and not np.isfinite(row.value))
                            else (round(row.value, 4) if isinstance(row.value, float) else row.value)),
                     reference=(round(row.reference, 4) if isinstance(row.reference, float) else row.reference),
                     win_prob_pp=round(float(row.win_prob_pp), 4),
                     margin_pts=round(float(row.margin_pts), 4),
                     standard_error_pp=round(float(row.standard_error_pp), 4))
                for row in part.itertuples()]))

    roster = []
    for row in players.sort_values(["side", "position", "rank_in_group"]).itertuples():
        roster.append(dict(side=row.side, team=row.team, name=row.name, position=row.position,
                           listed_position=row.listed_position, availability=row.availability,
                           unit_share=round(row.unit_share, 4),
                           win_prob_pp=round(row.win_prob_pp, 4), margin_pts=round(row.margin_pts, 4),
                           rank_in_group=int(row.rank_in_group)))
    return dict(
        game_id=game_id, season=int(c.base["G_season"][position]), home_team=home, away_team=away,
        model_version=c.active["version"], generated_at=pd.Timestamp.now(tz="UTC").isoformat(),
        home_win_prob=round(float(full[1]), 6), pred_margin=round(float(full[0]), 4),
        pred_total=round(float(full[2]), 4),
        baseline=dict(home_win_prob=round(float(empty[1]), 6), pred_margin=round(float(empty[0]), 4),
                      description="every team-level input at its FBS average: each stat at its FBS mean, the "
                                  "mean FBS program and conference, no prior results, neutral site. Both rosters "
                                  "are left in place, so this baseline still carries their roster difference."),
        method=dict(team="sampled Shapley over structured permutations against an average FBS team; "
                         "contributions sum to the prediction minus the baseline",
                    players="exact leave-one-out; each player swapped for a league-average player at the "
                            "same position, keeping the roster slot and his usage. unit_share is the "
                            "encoder's learned share of the position unit, which orders the depth chart",
                    permutations=2 * permutation_pairs, seeds=getattr(c, "members", 1),
                    median_standard_error_pp=round(float(np.median(se[:, 1]) * 100), 4)),
        blocks=blocks, players=roster,
        roster_total_win_prob_pp=round(float(players.win_prob_pp.sum()), 4),
        seconds=round(time.monotonic() - started, 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=int, action="append", help="explain these game ids")
    parser.add_argument("--upcoming-days", type=int, default=8, help="explain FBS games kicking off within N days")
    parser.add_argument("--permutation-pairs", type=int, default=48)
    parser.add_argument("--output", type=Path, default=MODELS / "player_sequence_production/explanations")
    parser.add_argument("--active", type=Path, default=ACTIVE)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    c = load(args.active)
    if args.game:
        targets = list(args.game)
    else:
        now = pd.Timestamp.now(tz="UTC")
        kick = pd.to_datetime(c.schedule.kick, utc=True)
        upcoming = c.schedule[(kick > now) & (kick < now + pd.Timedelta(days=args.upcoming_days))
                              & (c.schedule.home_division.eq("fbs") | c.schedule.away_division.eq("fbs"))]
        known = set(int(g) for g in c.base["G_gid"])
        targets = [int(g) for g in upcoming.index if int(g) in known]
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"model {c.active['version']}; explaining {len(targets)} games", flush=True)
    written = []
    for i, game_id in enumerate(targets):
        payload = explain(c, game_id, args.permutation_pairs)
        save_json(args.output / f"{game_id}.json", payload)
        written.append(payload)
        print(f"  {i+1}/{len(targets)} {payload['away_team']} at {payload['home_team']}: "
              f"P(home)={payload['home_win_prob']:.3f} in {payload['seconds']}s", flush=True)
    save_json(args.output / "manifest.json", dict(
        model_version=c.active["version"], games=len(written),
        generated_at=pd.Timestamp.now(tz="UTC").isoformat(),
        reference_players_per_position=c.reference_players,
        game_ids=[p["game_id"] for p in written]))
    if args.publish:
        from cfbpred.publish_explanations import publish
        publish(args.output)


if __name__ == "__main__":
    main()
