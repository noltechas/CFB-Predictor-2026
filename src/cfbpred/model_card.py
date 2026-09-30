"""Export the model card the dashboard's Model page renders.

Everything on that page is read from web/model_card.json, generated here from
the REAL artifacts — checkpoints, feature store, eval files — never typed by
hand. Run after training (and before/with publish) so the page always
describes the ensemble actually serving predictions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.config import (MODELS, N_MARGIN_BINS, ROOT,  # noqa: E402
                            TEST_SEASONS, TRAIN_SEASONS)
from cfbpred.ensemble import SEEDS, load_members  # noqa: E402
from cfbpred.model import MARGIN_CENTRES, TOTAL_CENTRES  # noqa: E402
from cfbpred.train import (S2D_PREFIXES, STATE_FEATURES, build_tensors,  # noqa: E402
                           game_batch)

WEB = ROOT / "web"

# feature-group taxonomy: (label, side-stripped prefixes or exact names)
GROUPS = [
    ("Play-by-play rates, offense", ("o_",),
     "season-to-date yards/play, explosive/turnover/penalty rates, third down"),
    ("Play-by-play rates, defense", ("d_",),
     "the same rates allowed, from the defense's side of every snap"),
    ("Box-score lines, every position", ("bx_", "box_gp"),
     "official per-game stat lines 2004-2025: passing, rushing, receiving, "
     "defense, kicking, punting, returns, concentration shares"),
    ("Season record", ("ppg_", "winpct", "gp"),
     "points for/against per game, win rate, games played"),
    ("Current-season QB", ("qb_",),
     "this season's primary passer: volume, efficiency, changed-starter flag"),
    ("Career experience", ("cx_",),
     "career games and volume of the current primary QB, back, target, kicker"),
    ("Star availability", ("av_",),
     "did last season's key producers appear in the previous game, and how "
     "much of the season's production showed up there"),
    ("Schedule context", ("sos_",),
     "mean preseason talent and recruiting of opponents faced so far, share "
     "that were FBS — deflates stat lines padded against soft schedules"),
    ("Returning production", ("retp_", "port_"),
     "share of last season's yards and tackles still on the roster (exact "
     "athlete-id joins), whether QB1 returned, and log-yards imported from "
     "other programs via the portal"),
    ("Roster composition", ("QB_", "OL_", "SKILL_", "DB_", "DL_", "LB_",
                            "n_players", "returning_share", "upper_share"),
     "position-group headcounts, heights, weights, headcount continuity"),
    ("Recruiting & talent", ("recruit_", "talent_", "tpct_"),
     "recruiting class points (4-class window) and the 247 talent composite "
     "(v4+: its within-season rank among FBS rosters, one scale for every season) — "
     "human assessments of players, allowed by rule; no outcome predictions"),
    ("Coaching", ("coach_",),
     "tenure, first-year flag, career games and win rate — raw counts"),
    ("Game context", ("rest", "travel", "alt_delta", "b_lat", "b_lon",
                      "b_elev", "is_", "week_scaled", "opp_is_fcs",
                      "home_is_fcs", "elevation", "latitude", "longitude"),
     "rest days, travel distance, altitude change, site, week, FCS flags"),
    ("Known-data flags", ("has_",),
     "explicit is-this-block-populated flags — missingness is a real state"),
]


def group_of(col: str) -> str:
    base = col[2:] if col[:2] in ("h_", "a_") else col
    for label, pres, _ in GROUPS:
        if base.startswith(pres):
            return label
    return "Game context"


def gate_curves(members) -> dict:
    """The two learned gates, sampled from the real parameters, seed-averaged."""
    # in-game gate alpha(time, |lead|), sampled on each member's device
    fr = torch.linspace(0, 1, 61)
    alpha = {}
    for lead in (0, 7, 14, 21):
        vals = []
        for m in members:
            dv = next(m.gate.parameters()).device
            vals.append(m.gate(fr.to(dv),
                               torch.full_like(fr, lead / 21.0).to(dv))
                        .detach().cpu())
        alpha[str(lead)] = torch.stack(vals).mean(0).tolist()
    # season-to-date gate g(games played)
    gp = torch.linspace(0, 1, 13)
    s2, bs, ws = [], [], []
    for m in members:
        b = float(m.pre.gate_b.detach().cpu())
        w = float(m.pre.gate_w.detach().cpu())
        bs.append(b); ws.append(w)
        s2.append(torch.sigmoid(b + w * gp))
    return {"frac_remaining": fr.tolist(), "alpha_by_lead": alpha,
            "gp": [round(x * 12) for x in gp.tolist()],
            "s2d_gate": torch.stack(s2).mean(0).tolist(),
            "s2d_params": {"b": float(np.mean(bs)), "w": float(np.mean(ws))}}


def layer_table(model) -> list[dict]:
    comp = {"pre.team_emb": "Team embeddings", "pre.conf_emb": "Conference embeddings",
            "pre.hist": "History encoder", "pre.body": "Pregame body",
            "pre.emb": "Pregame embedding head", "pre.margin": "Margin PMF head",
            "pre.total": "Total PMF head", "pre.gate": "Season-to-date gate",
            "state": "In-game state tower", "gate": "Pregame/live mixing gate"}
    rows = {}
    for name, p in model.named_parameters():
        key = next((v for k, v in comp.items() if name.startswith(k)), None)
        if key is None:
            key = ("Season-to-date gate" if name.startswith("pre.gate_")
                   else "Pregame body")
        r = rows.setdefault(key, {"component": key, "n_params": 0, "tensors": 0})
        r["n_params"] += p.numel()
        r["tensors"] += 1
    order = ["Team embeddings", "Conference embeddings", "History encoder",
             "Pregame body", "Pregame embedding head", "Margin PMF head",
             "Total PMF head", "Season-to-date gate", "In-game state tower",
             "Pregame/live mixing gate"]
    return [rows[k] for k in order if k in rows]


def main() -> None:
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    d = build_tensors(dev)
    members = load_members(d, dev)
    cols = d["scalar_cols"]

    # ---- input inventory, counted from the live feature store ----
    from collections import Counter
    counts = Counter(group_of(c) for c in cols)
    inventory = [{"group": label, "n": counts.get(label, 0), "desc": desc,
                  "s2d_gated": any(p.startswith(S2D_PREFIXES) or p == "winpct"
                                   for p in pres)}
                 for label, pres, desc in GROUPS if counts.get(label, 0)]
    # each history slot feeds its raw fields + the opponent's 24-d embedding
    from cfbpred.raw_net import HIST_FIELDS, TEAM_EMB
    n_hist_dims = 2 * d["vocab"]["n_hist"] * (HIST_FIELDS - 1 + TEAM_EMB)
    vocab = d["vocab"]

    # ---- example game: the most recent held-out game with a big audience ----
    season, gid_want = max(TEST_SEASONS), None
    te = np.where((d["G_season"] == season) & d["G_fbs_fbs"])[0]
    # deterministic pick: the game whose two teams had the most prior games
    gid_want = int(te[0]) if len(te) else 0
    for i in te:
        if int(d["G_gid"][i]) == 401752677:      # Texas at Ohio State, wk 1
            gid_want = int(i)
            break
    j = torch.tensor([gid_want])
    pms, pts_ = [], []
    with torch.no_grad():
        for m in members:
            _, pm, pt = m.pregame(game_batch(d, j, dev))
            pms.append(torch.softmax(pm, -1))
            pts_.append(torch.softmax(pt, -1))
    pm = torch.stack(pms).mean(0)[0].cpu()
    pt = torch.stack(pts_).mean(0)[0].cpu()
    wp = float(pm[(MARGIN_CENTRES > 0)].sum())
    example = {
        "game_id": int(d["G_gid"][gid_want]),
        "season": int(d["G_season"][gid_want]),
        "margin_centres": MARGIN_CENTRES.tolist(),
        "margin_pmf": [round(float(x), 6) for x in pm],
        "wp": wp,
        "em": float((pm * MARGIN_CENTRES).sum()),
        "et": float((pt * TOTAL_CENTRES).sum()),
    }

    # ---- metrics ----
    met = json.loads((MODELS / "metrics_raw.json").read_text())
    ev = {}
    if (MODELS / "eval_base.json").exists():
        ev = json.loads((MODELS / "eval_base.json").read_text())

    card = {
        "version": None,   # publish stamps model_runs; the page shows run info
        "n_params": sum(p.numel() for p in members[0].parameters()),
        "pre_hidden": members[0].pre.body[0].out_features,
        "n_scalars": len(cols),
        "n_hist_dims": n_hist_dims,
        "n_hist_games": vocab["n_hist"],
        "n_state": len(STATE_FEATURES),
        "n_teams": len(vocab["team"]),
        "n_confs": len(vocab["conf"]),
        "n_margin_bins": N_MARGIN_BINS,
        "seeds": len(SEEDS),
        "train_span": [2001, max(TRAIN_SEASONS)],
        "test_seasons": TEST_SEASONS,
        "inventory": inventory,
        "layers": layer_table(members[0]),
        "gates": gate_curves(members),
        "example": example,
        "metrics": met,
        "eval": {"pregame": ev.get("pregame_game_level"),
                 "buckets": ev.get("buckets")},
    }
    (WEB / "model_card.json").write_text(json.dumps(card))
    print(f"model card -> {WEB / 'model_card.json'} "
          f"({(WEB / 'model_card.json').stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
