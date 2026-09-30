"""Train the pure-DNN model on raw inputs only.

Feature policy, enforced by tests: no external OUTCOME predictions (no Elo,
no PPA, no betting lines, no SP+/FPI) and no hand-fitted intermediate models
of ours either (no ridge ratings, no expected points). Human scouting
assessments of players — recruiting class points and the 247 talent composite
— are allowed by user decision (2026-08-26): they describe rosters, not game
outcomes. Team strength is otherwise learned from embeddings, raw history and
raw counting stats.

Production trains through 2025 and reserves hashed games for checkpoint
selection; 2026 is unseen. For chronological model selection and independent
future-season tests, use backtest_pregame.py. Plays never cross game splits.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.config import (FEAT, MODELS, N_MARGIN_BINS, N_TOTAL_BINS,  # noqa: E402
                            TEST_SEASONS, TRAIN_SEASONS, VAL_MOD,
                            VAL_REM, VAL_SEASONS)
from cfbpred.model import (discrete_crps, expected_margin, expected_total, margin_bin,  # noqa: E402
                           smoothed_targets, soft_ce, total_bin,
                           win_probability)
from cfbpred.raw_net import RawTwoTower  # noqa: E402
from cfbpred.source import load as load_table  # noqa: E402
from cfbpred.features_play import IG_FEATURES, derive_state  # noqa: E402

# Substrings that must never appear in a feature column. The tests scan for
# them; keep this list in sync with the project's no-external-models rule.
# Outcome predictions stay banned. "talent"/"recruit" were removed from this
# list by user decision (2026-08-26): recruiting stars and the 247 composite
# are human scouting assessments of players, not forecasts of games — the one
# prior a production-history model lacks for players who haven't played yet.
BANNED_SUBSTRINGS = ("elo", "ppa", "spread", "sp_",
                     "srs", "fpi", "wp_", "epa", "line")

# Tested and excluded from the MODEL (kept in the data tables — they are raw,
# correctly point-in-time facts). Measured on held-out seasons with a 3-seed
# ensemble to control run variance: adding these 27 columns cost ~0.003 log
# loss and ~0.2 margin RMSE. At ~17k games the pregame tower is saturated —
# more collinear width now buys overfit, not signal. Revisit if the corpus
# grows or the tower gains capacity-matched regularisation.
EXCLUDED_SCALARS = (
    "wx_", "venue_grass", "venue_capacity", "kick_local_hour",
    "body_clock_shift", "h2h_", "qb_career_", "_ret_share",
    "new_upper_share", "wx_indoor_venue",
    # cross-season talent percentiles (features_raw.TalentPercentiles): opt-in, so
    # checkpoints fitted on the raw 247 composite keep exactly their inputs
    "tpct",
    # offseason roster movement (portal, draft) and in-season coach changes: opt-in
    "mvmt_", "cchg_",
)

ID_COLS = {"game_id", "season", "week", "home_team", "away_team",
           "home_conference", "away_conference", "home_division",
           "away_division", "home_points", "away_points", "margin", "total",
           "neutral_site", "conference_game", "h_idx", "a_idx", "week_c",
           "h_conf", "a_conf"}

BASE_STATE = [
    "lead_scaled", "lead_over_sqrt_t", "lead_x_elapsed", "frac_elapsed",
    "t_left_scaled", "sqrt_t", "home_has_ball", "home_has_ball_x_late",
    "down_scaled", "log_dist", "home_ytg_scaled",
    "home_to_scaled", "away_to_scaled", "recv_2h", "under_two", "goal_to_go",
    # endgame determinism: lead_over_t saturates at +-4 as the clock dies —
    # ANY lead at t~0 is decisive, which lead/sqrt(t) alone encodes too
    # weakly (final-snap 3-point leads: reality 96%, model was 73%)
    "lead_over_t", "under_one_min", "t_lt_10",
]
STATE_FEATURES = BASE_STATE + IG_FEATURES
N_BASE_STATE = len(BASE_STATE)


# An experiment or a checkpoint may re-admit named EXCLUDED_SCALARS entries. The
# purity check below still runs, so a banned column can never be re-admitted.
# An entry "-<column>" instead DROPS that exact column (flags included), so a
# variant can replace an input without changing what older checkpoints read.
SCALAR_EXTRAS: tuple = ()


def scalar_drops() -> set:
    return {x[1:] for x in SCALAR_EXTRAS if x.startswith("-")}


def scalar_columns(g: pd.DataFrame) -> list[str]:
    excluded = tuple(x for x in EXCLUDED_SCALARS if x not in SCALAR_EXTRAS)
    drops = scalar_drops()
    cols = [c for c in g.columns if c not in ID_COLS
            and not c.endswith("_team")
            and not any(x in c for x in excluded)
            and c not in drops]
    bad = [c for c in cols
           if any(b in c.lower() for b in BANNED_SUBSTRINGS)]
    if bad:
        raise RuntimeError(f"banned feature columns present: {bad}")
    return sorted(cols)


def prep_scalars(g: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    g = g.copy()
    g["is_home_site"] = (~g["neutral_site"].fillna(False).astype(bool)).astype(float)
    g["is_conf_game"] = g["conference_game"].fillna(False).astype(bool).astype(float)
    g["week_scaled"] = g["week_c"].astype(float).clip(upper=22) / 15.0
    g["is_postseason"] = (g["week_c"] > 20).astype(float)
    g["opp_is_fcs"] = (g["away_division"] != "fbs").astype(float)
    g["home_is_fcs"] = (g["home_division"] != "fbs").astype(float)
    cols = scalar_columns(g)
    # Missingness is a real state (no pbp before 2014, thin rosters early):
    # each block gets an explicit known-flag rather than a silent zero.
    for probe, flag in [("h_o_ypp", "h_has_pbp"), ("a_o_ypp", "a_has_pbp"),
                        ("h_n_players", "h_has_roster"),
                        ("a_n_players", "a_has_roster"),
                        ("h_coach_tenure", "h_has_coach"),
                        ("a_coach_tenure", "a_has_coach"),
                        ("h_bx_pass_ypg", "h_has_box"),
                        ("a_bx_pass_ypg", "a_has_box"),
                        ("h_talent_composite", "h_has_talent"),
                        ("a_talent_composite", "a_has_talent"),
                        ("h_recruit_pts_4c", "h_has_recruit"),
                        ("a_recruit_pts_4c", "a_has_recruit"),
                        ("h_retp_pass", "h_has_retp"),
                        ("a_retp_pass", "a_has_retp")]:
        g[flag] = g[probe].notna().astype(float) if probe in g.columns else 0.0
        if flag not in cols:
            cols.append(flag)
    cols = [c for c in cols if c not in scalar_drops()]
    for c in cols:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    return g, sorted(cols)


def prep_play(p: pd.DataFrame) -> pd.DataFrame:
    p = derive_state(p)
    t = p["t_left"].to_numpy(float)
    lead = p["home_lead"].to_numpy(float)
    p["frac_elapsed"] = (3600.0 - t) / 3600.0
    p["t_left_scaled"] = t / 3600.0
    p["sqrt_t"] = np.sqrt(np.maximum(t, 0)) / 60.0
    p["lead_scaled"] = lead / 21.0
    # the brief's central interaction, in its original deterministic form
    p["lead_over_sqrt_t"] = lead / np.sqrt(np.maximum(t, 0) + 1.0) / 3.0
    p["lead_x_elapsed"] = p["lead_scaled"] * p["frac_elapsed"]
    p["home_has_ball_x_late"] = p["home_has_ball"] * (1.0 - p["frac_elapsed"])
    p["down_scaled"] = p["down"].astype(float) / 4.0
    p["log_dist"] = np.log1p(np.clip(p["distance"].astype(float), 0, 40)) / 3.0
    p["home_ytg_scaled"] = p["home_ytg"].astype(float) / 100.0
    p["home_to_scaled"] = p["home_to"].astype(float) / 3.0
    p["away_to_scaled"] = p["away_to"].astype(float) / 3.0
    p["lead_over_t"] = np.clip(lead / (t / 60.0 + 0.25), -4.0, 4.0) / 4.0
    p["under_one_min"] = (np.minimum(t, np.where(t > 1800, t - 1800, t))
                          <= 60).astype(float)
    p["t_lt_10"] = (t <= 10).astype(float)
    for c in STATE_FEATURES:
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0)
    return p


def build_history_tensors(g_order: pd.DataFrame, n_hist: int, history=None):
    """[G, 2, n_hist, 7] raw slots + [G, 2, n_hist] opponent ids, home side 0."""
    h = load_table("game_history") if history is None else history
    gid_pos = {gid: i for i, gid in enumerate(g_order["game_id"])}
    side_of = {}
    for _, r in g_order.iterrows():
        side_of[(r["game_id"], r["home_team"])] = 0
        side_of[(r["game_id"], r["away_team"])] = 1
    G = len(g_order)
    raw = np.zeros((G, 2, n_hist, 7), np.float32)
    opp = np.zeros((G, 2, n_hist), np.int64)
    gi = h["game_id"].map(gid_pos)
    keep = gi.notna().to_numpy()
    h = h.loc[keep]
    gi = gi[keep].astype(int).to_numpy()
    si = np.array([side_of.get((g_, t_), -1)
                   for g_, t_ in zip(h["game_id"], h["team"])])
    ok = si >= 0
    h, gi, si = h.loc[ok], gi[ok], si[ok]
    sl = h["slot"].to_numpy()
    raw[gi, si, sl, 0] = h["pf"].to_numpy(np.float32) / 30.0
    raw[gi, si, sl, 1] = h["pa"].to_numpy(np.float32) / 30.0
    raw[gi, si, sl, 2] = h["was_home"].to_numpy(np.float32)
    raw[gi, si, sl, 3] = h["opp_fbs"].to_numpy(np.float32)
    raw[gi, si, sl, 4] = h["days_ago"].to_numpy(np.float32) / 100.0
    raw[gi, si, sl, 5] = h["same_season"].to_numpy(np.float32)
    raw[gi, si, sl, 6] = 1.0                      # mask
    opp[gi, si, sl] = h["opp_idx"].to_numpy(np.int64)
    return torch.tensor(raw), torch.tensor(opp)


def build_tensors(dev=None, *, data_dir=None, include_plays=True):
    read_table = (lambda name: pd.read_parquet(Path(data_dir) / f"{name}.parquet")) if data_dir else load_table
    vocab = json.loads(((Path(data_dir) if data_dir else FEAT) / "team_vocab.json").read_text())
    g_raw = read_table("game_scalars")
    g, cols = prep_scalars(g_raw)
    g = g.reset_index(drop=True)

    # standardise on the pretraining seasons only
    tr_g = ((g["season"] <= max(TRAIN_SEASONS))
            & ((g.game_id % VAL_MOD) != VAL_REM) & g.margin.notna())
    X_raw = g[cols].to_numpy(np.float32)
    X = X_raw.copy()
    mu = np.nanmean(X[tr_g.to_numpy()], axis=0)
    sd = np.nanstd(X[tr_g.to_numpy()], axis=0)
    sd[sd < 1e-6] = 1.0
    X = (X - mu) / sd
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    h_raw, h_opp = build_history_tensors(g, vocab["n_hist"], read_table("game_history"))

    p = (prep_play(read_table("play_raw")) if include_plays else
         pd.DataFrame(columns=list(dict.fromkeys(STATE_FEATURES +
            ["game_id", "season", "t_left", "home_lead", "total_pts_before"]))))
    gid_pos = {gid: i for i, gid in enumerate(g["game_id"])}
    p = p[p["game_id"].isin(gid_pos)].reset_index(drop=True)
    gidx = p["game_id"].map(gid_pos).to_numpy(dtype=np.int64)

    T = lambda a, d=torch.float32: torch.tensor(np.asarray(a), dtype=d)  # noqa: E731
    # raw games-played per side, scaled to [0,1] — drives the season-to-date
    # gate; NaN (no record row) means zero prior games
    gp_h = np.clip(pd.to_numeric(g.get("h_gp"), errors="coerce")
                   .fillna(0.0).to_numpy(np.float32) / 12.0, 0.0, 1.0)
    gp_a = np.clip(pd.to_numeric(g.get("a_gp"), errors="coerce")
                   .fillna(0.0).to_numpy(np.float32) / 12.0, 0.0, 1.0)
    return {
        "vocab": vocab, "scalar_cols": cols, "mu": mu, "sd": sd,
        "G_raw_scalars": X_raw, "G_labeled": g.margin.notna().to_numpy(),
        # game-level
        "G_scalars": T(X), "G_hist": h_raw, "G_opp": h_opp,
        "G_gp01_h": T(gp_h), "G_gp01_a": T(gp_a),
        "G_hidx": T(g["h_idx"], torch.long), "G_aidx": T(g["a_idx"], torch.long),
        "G_hconf": T(g["h_conf"], torch.long), "G_aconf": T(g["a_conf"], torch.long),
        # unplayed (scheduled) games carry null labels; 0 here is a
        # placeholder that never reaches a loss — the train/val masks are
        # season-bounded and the current season is excluded from both
        "G_margin": T(g["margin"].fillna(0), torch.long),
        "G_total": T(g["total"].fillna(0), torch.long),
        "G_season": g["season"].to_numpy(), "G_gid": g["game_id"].to_numpy(),
        "G_fbs_fbs": ((g["home_division"] == "fbs")
                      & (g["away_division"] == "fbs")).to_numpy(),
        # FBS-involved rows are the product; anything else in the store is
        # an experimental extra target (CFB_FCS_TARGETS) — never validation
        "G_fbs_inv": ((g["home_division"] == "fbs")
                      | (g["away_division"] == "fbs")).to_numpy(),
        # play-level
        "P_state": T(p[STATE_FEATURES].to_numpy(np.float32)),
        "P_gidx": T(gidx, torch.long),
        "P_season": p["season"].to_numpy(), "P_gid": p["game_id"].to_numpy(),
        "P_t_left": p["t_left"].to_numpy(np.float32),
        "P_lead": p["home_lead"].to_numpy(np.float32),
        "P_pts": p["total_pts_before"].to_numpy(np.float32),
        "P_margin": T(g["margin"].to_numpy()[gidx], torch.long),
        "P_total": T(g["total"].to_numpy()[gidx], torch.long),
        # FBS-vs-FBS play mask. The play-level LOSS trains only on these:
        # FCS-matchup plays are lopsided and easy, and letting them into the
        # objective diluted competitive-game WP by ~0.006 log loss (measured
        # on an identical evaluation population). Scoring still covers all.
        "P_ff": ((g["home_division"] == "fbs")
                 & (g["away_division"] == "fbs")).to_numpy()[gidx],
    }


def game_batch(d, j, dev):
    batch = {"scalars": d["G_scalars"][j].to(dev),
            "h_idx": d["G_hidx"][j].to(dev), "a_idx": d["G_aidx"][j].to(dev),
            "h_conf": d["G_hconf"][j].to(dev), "a_conf": d["G_aconf"][j].to(dev),
            "h_hist": d["G_hist"][j, 0].to(dev), "h_opp": d["G_opp"][j, 0].to(dev),
            "a_hist": d["G_hist"][j, 1].to(dev), "a_opp": d["G_opp"][j, 1].to(dev),
            "h_gp01": d["G_gp01_h"][j].to(dev), "a_gp01": d["G_gp01_a"][j].to(dev)}
    if "player_store" in d:
        batch.update(d["player_store"].batch(j.cpu().numpy(), dev))
    return batch


# Season-to-date blocks (per side): things that accumulate within the season
# and are noise at 1-3 games. Priors (talent, recruiting, roster, coach,
# returning production, history slots) and the has_* flags stay ungated.
S2D_PREFIXES = ("o_", "d_", "bx_", "qb_", "ppg_", "winpct")


def s2d_indices(cols: list[str], side: str) -> list[int]:
    pre = side + "_"
    return [i for i, c in enumerate(cols)
            if c.startswith(pre) and c[len(pre):].startswith(S2D_PREFIXES)]


def make_model(d, state_hidden: int = 128, s2d_gate: bool = False,
               pre_hidden: int = 160, delta: bool = False) -> RawTwoTower:
    return RawTwoTower(len(d["vocab"]["team"]) + 1, len(d["vocab"]["conf"]) + 1,
                       d["G_scalars"].shape[1], len(STATE_FEATURES),
                       state_hidden=state_hidden, pre_hidden=pre_hidden,
                       s2d_idx_h=s2d_indices(d["scalar_cols"], "h"),
                       s2d_idx_a=s2d_indices(d["scalar_cols"], "a"),
                       use_s2d_gate=s2d_gate, delta_heads=delta)


def pretrain(model, d, dev, max_epochs=200, lr=2e-3, patience=20,
             margin_aux=0.0, sel_mae_w=0.0, train_indices=None,
             val_indices=None, fixed_epochs=None, margin_crps_w=0.0,
             micro_batch_size=2048):
    """Early-stopped on held-out games. A fixed epoch count memorises: with
    team embeddings as inputs, 300 epochs on 16k games turned the pregame tower
    into an outcome lookup table (train fit superb, validation ruined).

    Anti-timidity (v16): WP log loss alone never pays for calling a game -21
    when the truth is -35 — once WP saturates near 1, the margin head can be
    arbitrarily timid at zero cost, and blowouts were under-called by 6.5 pts.
    margin_aux adds an L1 term on E[margin]; sel_mae_w folds validation margin
    MAE into the early-stopping score so selection stops rewarding timidity.
    margin_crps_w adds a proper, distance-sensitive distribution loss.
    All three controls default to zero for production compatibility."""
    if min(margin_aux, sel_mae_w, margin_crps_w) < 0:
        raise ValueError("Loss and selection weights must be nonnegative")
    vg = (np.asarray(d["G_gid"]) % VAL_MOD) == VAL_REM
    fi = np.asarray(d["G_fbs_inv"])
    tr_all = (np.where((d["G_season"] <= max(TRAIN_SEASONS)) & ~vg & d["G_labeled"])[0]
              if train_indices is None else np.asarray(train_indices))
    # FCS-vs-FCS extras (experimental store) train at a sampled fraction so
    # 3x sparse-feature rows can't swamp the FBS patterns; validation and
    # early stopping stay FBS-involved-only so scores compare across runs
    tr_fbs, tr_fcs = tr_all[fi[tr_all]], tr_all[~fi[tr_all]]
    FCS_SAMPLE = 0.25
    va = np.where((d["G_season"] <= max(TRAIN_SEASONS))
                  & (d["G_season"] >= min(TRAIN_SEASONS)) & vg & fi)[0]
    if val_indices is not None:
        va = np.asarray(val_indices)
    if not len(tr_all) or (fixed_epochs is None and not len(va)):
        raise ValueError("Training and validation partitions must not be empty")
    tm = smoothed_targets(margin_bin(d["G_margin"]), N_MARGIN_BINS)
    tt = smoothed_targets(total_bin(d["G_total"]), N_TOTAL_BINS)
    y_va = (d["G_margin"].numpy()[va] > 0).astype(np.float32)
    m_va = d["G_margin"].numpy()[va].astype(np.float32)
    opt = torch.optim.AdamW(model.pre.parameters(), lr=lr, weight_decay=3e-2)
    best, best_state, since = 1e9, None, 0
    best_epoch = 0
    for ep in range(fixed_epochs if fixed_epochs is not None else max_epochs):
        sub = (np.random.choice(tr_fcs, int(len(tr_fcs) * FCS_SAMPLE),
                                replace=False)
               if len(tr_fcs) else np.empty(0, np.int64))
        perm = np.random.permutation(np.concatenate([tr_fbs, sub]))
        for i in range(0, len(perm), 2048):
            full = perm[i:i + 2048]
            opt.zero_grad()
            for start in range(0, len(full), micro_batch_size):
                j = torch.tensor(full[start:start+micro_batch_size])
                _, pm, pt = model.pregame(game_batch(d, j, dev))
                loss = soft_ce(pm, tm[j].to(dev)) + 0.5 * soft_ce(pt, tt[j].to(dev))
                if margin_aux:
                    m_true = d["G_margin"][j].to(dev).float().clamp(-56.0, 56.0)
                    loss = loss + margin_aux * (expected_margin(pm)-m_true).abs().mean()
                if margin_crps_w:
                    loss = loss + margin_crps_w * discrete_crps(pm, d["G_margin"][j].to(dev))
                (loss*(len(j)/len(full))).backward()
            opt.step()
        if fixed_epochs is not None:
            continue
        model.eval()
        with torch.no_grad():
            eval_bs = min(micro_batch_size, 128) if "player_store" in d else len(va)
            pm = torch.cat([model.pregame(game_batch(d, torch.tensor(va[start:start+eval_bs]), dev))[1]
                            for start in range(0, len(va), eval_bs)])
            wp = win_probability(pm).cpu().numpy().clip(1e-6, 1 - 1e-6)
            em = expected_margin(pm).cpu().numpy()
        model.train()
        vll = float(-np.mean(y_va * np.log(wp) + (1 - y_va) * np.log(1 - wp)))
        mae = float(np.mean(np.abs(em - m_va)))
        score = vll + sel_mae_w * mae
        if score < best - 1e-4:
            best, since = score, 0
            best_epoch = ep + 1
            best_state = {k: t.detach().cpu().clone()
                          for k, t in model.pre.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
        if ep % 10 == 0:
            print(f"    pretrain ep {ep:3d}  heldout WP LL {vll:.5f}"
                  f"  MAE {mae:.2f}  (best score {best:.5f})")
    if fixed_epochs is not None:
        model.pre.selected_epochs = fixed_epochs
        return None
    model.pre.load_state_dict(best_state)
    model.pre.selected_epochs = best_epoch
    model.pre.selection_score = best
    return best


# MPS silently corrupts gather-heavy forward passes somewhere above 16k rows:
# at bs=100_000, 27% of play WPs differed from CPU (max 0.32); at bs=16_384 the
# outputs are bit-identical to CPU. Never raise this on Apple Silicon.
EVAL_BS = 16_384


def eval_plays(model, d, mask, dev, bs=EVAL_BS):
    model.eval()
    outs = {"wp": [], "em": [], "et": [], "alpha": []}
    idx = np.where(mask)[0]
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = torch.tensor(idx[i:i + bs])
            gj = d["P_gidx"][j]
            fr = (torch.tensor(d["P_t_left"][j.numpy()]) / 3600.0).to(dev)
            al = (torch.tensor(np.abs(d["P_lead"][j.numpy()])) / 21.0).to(dev)
            ld = torch.tensor(d["P_lead"][j.numpy()]).to(dev)
            pz = torch.tensor(d["P_pts"][j.numpy()]).to(dev)
            o = model(game_batch(d, gj, dev), d["P_state"][j].to(dev), fr, al,
                      ld, pz)
            outs["wp"].append(win_probability(o["margin"]).cpu().numpy())
            outs["em"].append(expected_margin(o["margin"]).cpu().numpy())
            outs["et"].append(expected_total(o["total"]).cpu().numpy())
            outs["alpha"].append(o["alpha"].cpu().numpy())
    model.train()
    return {k: np.concatenate(v) for k, v in outs.items()}


def metrics(wp, em, et, margin, total, t_left):
    y = (margin > 0).astype(float)
    p = np.clip(wp, 1e-6, 1 - 1e-6)
    def ll(pp, yy):
        return float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)))
    def ece(pp, yy, nb=20):
        i = np.clip((pp * nb).astype(int), 0, nb - 1)
        return float(sum((i == k).sum() / len(pp)
                         * abs(pp[i == k].mean() - yy[i == k].mean())
                         for k in range(nb) if (i == k).sum()))
    out = {"n": int(len(wp)), "logloss": ll(p, y), "ece": ece(p, y),
           "acc": float(np.mean((p > 0.5) == y)),
           "margin_rmse": float(np.sqrt(np.mean((em - margin) ** 2))),
           "total_rmse": float(np.sqrt(np.mean((et - total) ** 2)))}
    for lo, hi, lab in [(2700, 3601, "q1"), (1800, 2700, "q2"),
                        (900, 1800, "q3"), (0, 900, "q4")]:
        m = (t_left >= lo) & (t_left < hi)
        if m.sum() > 100:
            out[f"logloss_{lab}"] = ll(p[m], y[m])
            out[f"ece_{lab}"] = ece(p[m], y[m])
    return out


def train(epochs=24, seed=0, tag="raw", state_hidden=128, patience=None,
          lr_scale=1.0, s2d_gate=False, margin_aux=0.0, sel_mae_w=0.0,
          pre_hidden=160, delta=False):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    d = build_tensors(dev)
    model = make_model(d, state_hidden, s2d_gate, pre_hidden,
                       delta=delta).to(dev)
    n_par = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"[{tag}] {n_par:,} parameters | {d['G_scalars'].shape[1]} scalar cols "
          f"| input width ~{2 * d['vocab']['n_hist'] * (6 + 24) + d['G_scalars'].shape[1] + 64 + 15}")

    t0 = time.time()
    pretrain(model, d, dev, margin_aux=margin_aux, sel_mae_w=sel_mae_w)
    print(f"[{tag}] pregame pretrained on "
          f"{(d['G_season'] <= max(TRAIN_SEASONS)).sum():,} games "
          f"2001-{max(TRAIN_SEASONS)} ({time.time() - t0:.0f}s)")

    vp = (np.asarray(d["P_gid"]) % VAL_MOD) == VAL_REM
    tr = np.isin(d["P_season"], TRAIN_SEASONS) & d["P_ff"] & ~vp
    va = np.isin(d["P_season"], TRAIN_SEASONS) & d["P_ff"] & vp
    idx_tr = np.where(tr)[0]
    tm = smoothed_targets(margin_bin(d["P_margin"]), N_MARGIN_BINS)
    tt = smoothed_targets(total_bin(d["P_total"]), N_TOTAL_BINS)

    # The pregame tower is FROZEN after early-stopped pretraining. Left
    # trainable, the joint phase re-memorises training-game outcomes through
    # the embeddings (validation degraded monotonically from epoch 1) and the
    # gate learns to trust the memorised tower. Frozen, the joint phase can
    # only fit the state tower and the gate, neither of which can memorise
    # outcomes — state features carry no team identity.
    model.pre.eval()
    for p_ in model.pre.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.AdamW([
        {"params": model.state.parameters(), "lr": 3e-3 * lr_scale,
         "weight_decay": 1e-4},
        {"params": model.gate.parameters(), "lr": 6e-3 * lr_scale,
         "weight_decay": 0.0},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state, since = 1e9, None, 0
    for ep in range(epochs):
        perm = np.random.permutation(idx_tr)
        for i in range(0, len(perm), 8192):
            j = torch.tensor(perm[i:i + 8192])
            gj = d["P_gidx"][j]
            fr = (torch.tensor(d["P_t_left"][j.numpy()]) / 3600.0).to(dev)
            al = (torch.tensor(np.abs(d["P_lead"][j.numpy()])) / 21.0).to(dev)
            xs = d["P_state"][j].to(dev)
            # The in-game cumulative block fingerprints games almost uniquely,
            # and the state tower will memorise training outcomes through it
            # (validation degraded steeply once the block was added). Noise +
            # feature dropout break the fingerprint — but ONLY on the ig_*
            # block. The first attempt corrupted ALL state features, including
            # lead and clock, which is precisely what win probability needs
            # unblurred: it cost 0.005 log loss by itself. Train-time only.
            ig = torch.zeros_like(xs)
            ig[:, N_BASE_STATE:] = 1.0
            keep = torch.where(
                ig.bool(), (torch.rand_like(xs) > 0.10).float(),
                torch.ones_like(xs))
            xs = xs * keep + 0.03 * torch.randn_like(xs) * ig
            ld = torch.tensor(d["P_lead"][j.numpy()]).to(dev)
            pz = torch.tensor(d["P_pts"][j.numpy()]).to(dev)
            o = model(game_batch(d, gj, dev), xs, fr, al, ld, pz)
            # Generic time emphasis, no football logic: endgame rows are
            # 1-2% of plays, so under a plain mean the optimizer never pays
            # for timidity there (measured: 3-point final-snap leads sat at
            # 83% vs 95% reality even with 2 extra seasons and 24 epochs —
            # a 0.0005 mean-loss issue, below selection noise). The weight
            # references only the clock; WHICH late states are safe is left
            # entirely to the network to deduce from possession, timeouts
            # and lead.
            w = 1.0 + 9.0 * torch.exp(-fr * 3600.0 / 60.0)
            # The total term carries the same clock weight in the delta
            # variant: unweighted, late total sharpness is a ~1%-of-rows
            # phenomenon the optimizer never pays for (measured on v20:
            # inside the final 30s the model kept a 10-point total SD
            # against a realized 7, with 4.7% on "no more scoring" vs an
            # empirical 78.5%).
            loss = (soft_ce(o["margin"], tm[j].to(dev), w)
                    + 0.5 * soft_ce(o["total"], tt[j].to(dev),
                                    w if delta else None))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.pre.eval()
        v = eval_plays(model, d, va, dev)
        model.pre.eval()
        m = metrics(v["wp"], v["em"], v["et"],
                    d["P_margin"].numpy()[va], d["P_total"].numpy()[va],
                    d["P_t_left"][va])
        # selection mirrors the training emphasis (same clock-only weight)
        yv = (d["P_margin"].numpy()[va] > 0).astype(float)
        pv = np.clip(v["wp"], 1e-6, 1 - 1e-6)
        wv = 1.0 + 9.0 * np.exp(-d["P_t_left"][va] / 60.0)
        ll_row = -(yv * np.log(pv) + (1 - yv) * np.log(1 - pv))
        m["logloss_w"] = float((ll_row * wv).sum() / wv.sum())
        end = d["P_t_left"][va] < 60
        m["logloss_end"] = float(ll_row[end].mean())
        print(f"[{tag}] epoch {ep + 1:2d}  val LL {m['logloss']:.5f}  "
              f"wLL {m['logloss_w']:.5f}  Q4 {m['logloss_q4']:.5f}  "
              f"end<60s {m['logloss_end']:.5f}  "
              f"mRMSE {m['margin_rmse']:.2f}", flush=True)
        if m["logloss_w"] < best:
            best = m["logloss_w"]; since = 0
            best_state = {k: t.detach().cpu().clone()
                          for k, t in model.state_dict().items()}
        else:
            since += 1
            if patience is not None and since >= patience:
                print(f"[{tag}] early stop at epoch {ep + 1} "
                      f"(no val gain for {patience})", flush=True)
                break
    model.load_state_dict(best_state)

    res = {}
    for name, mask in [("val", va), ("test",
                       np.isin(d["P_season"], TEST_SEASONS) & d["P_ff"])]:
        if not mask.sum():          # test season not yet played
            print(f"[{tag}] no {name} plays yet ({TEST_SEASONS}); skipping")
            continue
        o = eval_plays(model, d, mask, dev)
        res[name] = metrics(o["wp"], o["em"], o["et"],
                            d["P_margin"].numpy()[mask],
                            d["P_total"].numpy()[mask], d["P_t_left"][mask])
        res[name]["alpha_kickoff"] = float(o["alpha"][d["P_t_left"][mask] > 3400].mean())
        res[name]["alpha_q4"] = float(o["alpha"][d["P_t_left"][mask] < 900].mean())
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "scalar_cols": d["scalar_cols"], "mu": d["mu"], "sd": d["sd"],
                "state_cols": STATE_FEATURES, "n_params": n_par,
                "s2d_gate": s2d_gate, "pre_hidden": pre_hidden,
                "delta": delta,
                "metrics": res}, MODELS / f"two_tower_{tag}.pt")
    (MODELS / f"metrics_{tag}.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    r = train()
    key = "test" if "test" in r else "val"
    print(f"\n=== {key.upper()} ({TEST_SEASONS if key == 'test' else 'game-level holdout'}), raw inputs only ===")
    for k, v in r[key].items():
        print(f"  {k:24s} {v:.5f}" if isinstance(v, float) else f"  {k:24s} {v}")
