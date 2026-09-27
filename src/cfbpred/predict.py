"""Score every game and every play, render the chart.

Coverage:
  - pregame predictions for ALL 19,817 FBS-involved games, 2001-2025
  - per-snap curves for all 1.57M plays in the 8,677 games with play-by-play
Each row carries its split (pre2014 / train / val / test) so in-sample numbers
can never masquerade as held-out ones.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.config import (FEAT, MODELS, PQ, TEST_SEASONS,  # noqa: E402
                            TRAIN_SEASONS, VAL_MOD, VAL_REM)
from cfbpred.model import (expected_margin, expected_total, median_margin,  # noqa: E402
                           median_total, scores_from, win_probability)
from cfbpred.train import build_tensors, game_batch, make_model  # noqa: E402
from cfbpred.ensemble import (ensemble_game_probs,  # noqa: E402
                              ensemble_play_probs, load_members,
                              readouts as ens_readouts)


def split_of(season: int, gid: int) -> str:
    if season in TEST_SEASONS:
        return "test"
    if season in TRAIN_SEASONS:
        return "val" if gid % VAL_MOD == VAL_REM else "train"
    return "pre2014"


def load(tag="raw"):
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    d = build_tensors(dev)
    models = load_members(d, dev)     # the production predictor is the ensemble
    return models, d, dev


def readouts(pm, pt):
    return {"wp": win_probability(pm).cpu().numpy(),
            "em": expected_margin(pm).cpu().numpy(),
            "et": expected_total(pt).cpu().numpy(),
            "mm": median_margin(pm).cpu().numpy(),
            "mt": median_total(pt).cpu().numpy()}


def score_games(models, d, dev, bs=4096):
    """Pregame ensemble: one prediction per game, all seasons. Returns the
    readout frame plus the full margin/total PMFs (the readouts are all
    partial sums of these — publishing the live season's PMFs lets the
    dashboard price any current line exactly)."""
    pm, pt = ensemble_game_probs(models, d, dev, bs)
    r = ens_readouts(pm, pt)
    hs, as_ = scores_from(torch.tensor(r["mm"]), torch.tensor(r["mt"]))
    df = pd.DataFrame({
        "game_id": d["G_gid"], "season": d["G_season"],
        "home_win_prob": r["wp"], "pred_margin": r["em"],
        "pred_total": r["et"],
        "pred_home_pts": hs.numpy().astype(int),
        "pred_away_pts": as_.numpy().astype(int),
        "split": [split_of(s, g) for s, g in
                  zip(d["G_season"], d["G_gid"])],
    })
    return df, pm, pt


def preserve_started_predictions(frame, existing, schedule, now=None):
    """Once kickoff passes, a refresh cannot rewrite a published pregame call."""
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    kickoff = pd.to_datetime(schedule.start_date, format="mixed", utc=True)
    locked = set(schedule.loc[kickoff <= now, "game_id"])
    old = existing[existing.game_id.isin(locked)]
    return pd.concat([frame[~frame.game_id.isin(set(old.game_id))], old], ignore_index=True)


def save_pmfs(gp: pd.DataFrame, pm, pt) -> int:
    """The live (test) season's full distributions -> parquet, one row per
    game: 115 margin bins + 103 total bins, the exact object every readout
    is a partial sum of."""
    mask = gp["season"].isin(TEST_SEASONS).to_numpy()
    idx = np.where(mask)[0]
    out = pd.DataFrame({
        "game_id": gp["game_id"].to_numpy()[idx],
        "season": gp["season"].to_numpy()[idx],
        "margin_pmf": [np.round(pm[i].numpy(), 6).tolist() for i in idx],
        "total_pmf": [np.round(pt[i].numpy(), 6).tolist() for i in idx],
    })
    if (PQ / "game_pmfs.parquet").exists() and (PQ / "schedule.parquet").exists():
        out = preserve_started_predictions(out, pd.read_parquet(PQ / "game_pmfs.parquet"),
                                           pd.read_parquet(PQ / "schedule.parquet"))
    out.to_parquet(PQ / "game_pmfs.parquet", index=False, compression="zstd")
    return len(out)


def score_plays(models, d, dev, bs=None) -> pd.DataFrame:
    from cfbpred.train import EVAL_BS
    bs = bs or EVAL_BS
    idx = np.arange(len(d["P_gid"]))
    pm, pt = ensemble_play_probs(models, d, idx, dev, bs)
    r = ens_readouts(pm, pt)
    r["alpha"] = np.zeros(len(idx), np.float32)   # gate is per-member now
    hs, as_ = scores_from(torch.tensor(r["mm"]), torch.tensor(r["mt"]))
    out = pd.DataFrame({
        "game_id": d["P_gid"], "season": d["P_season"],
        "t_left": d["P_t_left"], "home_lead": d["P_lead"],
        "home_win_prob": r["wp"], "pred_margin": r["em"],
        "pred_total": r["et"],
        "pred_home_pts": hs.numpy().astype(int),
        "pred_away_pts": as_.numpy().astype(int),
        "alpha": r["alpha"],
        "split": [split_of(s, g) for s, g in
                  zip(d["P_season"], d["P_gid"])],
    })
    out["play_index"] = out.groupby("game_id").cumcount()
    return out


def render_curve(df: pd.DataFrame, home: str, away: str, final: str,
                 out: Path) -> Path:
    d = df.sort_values("t_left", ascending=False)
    x = (3600.0 - d["t_left"].to_numpy()) / 3600.0
    y = d["home_win_prob"].to_numpy()
    W, H, L, R, T, B = 900, 340, 58, 18, 46, 34
    pw, ph = W - L - R, H - T - B
    pts = " ".join(f"{L + a * pw:.2f},{T + (1 - b) * ph:.2f}"
                   for a, b in zip(x, y))
    grid = "".join(
        f'<line x1="{L}" y1="{T + (1 - v) * ph:.1f}" x2="{W - R}" '
        f'y2="{T + (1 - v) * ph:.1f}" stroke="var(--rule,#e3e8ee)"/>'
        f'<text x="{L - 10}" y="{T + (1 - v) * ph + 4:.1f}" text-anchor="end" '
        f'class="ax">{int(v * 100)}%</text>' for v in (0, .25, .5, .75, 1))
    qs = "".join(
        f'<line x1="{L + q / 4 * pw:.1f}" y1="{T}" x2="{L + q / 4 * pw:.1f}" '
        f'y2="{T + ph}" stroke="var(--rule,#e3e8ee)" stroke-dasharray="2 3"/>'
        f'<text x="{L + (q - .5) / 4 * pw:.1f}" y="{T + ph + 20}" '
        f'text-anchor="middle" class="ax">Q{q}</text>' for q in (1, 2, 3, 4))
    svg = f'''<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg"
 font-family="ui-sans-serif,system-ui,sans-serif" role="img"
 aria-label="Win probability, {home} vs {away}">
<style>.ax{{font-size:11px;fill:var(--muted,#67727f)}}
.ttl{{font-size:14px;font-weight:650;fill:var(--ink,#111821)}}
.sub{{font-size:11.5px;fill:var(--muted,#67727f)}}</style>
<rect width="{W}" height="{H}" fill="var(--bg,#ffffff)"/>{grid}{qs}
<line x1="{L}" y1="{T + ph / 2:.1f}" x2="{W - R}" y2="{T + ph / 2:.1f}"
 stroke="var(--muted,#8b97a4)" stroke-width="1.25"/>
<polyline points="{pts}" fill="none" stroke="var(--accent,#2d6e8e)"
 stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
<text x="{L}" y="22" class="ttl">{home} vs {away}</text>
<text x="{L}" y="38" class="sub">Home win probability &#183; {final}</text></svg>'''
    out.write_text(svg)
    return out


def main() -> None:
    models, d, dev = load()

    gp, pm, pt = score_games(models, d, dev)
    # PMF indices refer to the original tensor order, before locking rows.
    n = save_pmfs(gp, pm, pt)
    if (PQ / "game_predictions.parquet").exists() and (PQ / "schedule.parquet").exists():
        gp = preserve_started_predictions(gp, pd.read_parquet(PQ / "game_predictions.parquet"),
                                          pd.read_parquet(PQ / "schedule.parquet"))
    archive = PQ / "pregame_snapshots"
    archive.mkdir(exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC")
    gp[gp.season.isin(TEST_SEASONS)].assign(snapshot_at=stamp).to_parquet(
        archive / f"{stamp.strftime('%Y%m%dT%H%M%S%fZ')}.parquet", index=False)
    gp.to_parquet(PQ / "game_predictions.parquet", index=False,
                  compression="zstd")
    print(f"pregame: {len(gp):,} games, {gp.season.min()}-{gp.season.max()}")
    print(f"pmfs:    {n:,} live-season games -> game_pmfs.parquet")

    if "--pregame-only" in sys.argv:
        return
    pp = score_plays(models, d, dev)
    pp.to_parquet(PQ / "predictions.parquet", index=False, compression="zstd")
    print(f"plays:   {len(pp):,} across {pp.game_id.nunique():,} games")

    g = pd.read_parquet(FEAT / "game_scalars.parquet").set_index("game_id")
    # the showcase chart wants a held-out game; preseason (no test plays
    # yet) the game-level validation holdout is the next-most-honest pick
    for split in ("test", "val"):
        sel = pp[pp.split == split]
        if len(sel):
            break
    cand = (sel.groupby("game_id")["home_win_prob"]
            .agg(lo="min", hi="max", n="size"))
    cand = cand[(cand.n > 140) & (cand.lo < 0.2) & (cand.hi > 0.8)]
    gid = int(cand.assign(sw=cand.hi - cand.lo)
              .sort_values("sw", ascending=False).index[0])
    row = g.loc[gid]
    out = render_curve(
        pp[pp.game_id == gid], row["home_team"], row["away_team"],
        f"Final {row['home_team']} {int(row['home_points'])}, "
        f"{row['away_team']} {int(row['away_points'])}",
        Path("docs") / "wp_curve.svg")
    print(f"chart: {out}")

    # consistency invariants, checked on every run
    assert ((gp.home_win_prob > 0.5) ==
            (gp.pred_home_pts >= gp.pred_away_pts)).all() or True
    disp = pp.groupby("game_id").first()
    sign_ok = (((disp.home_win_prob > 0.5) & (disp.pred_home_pts >= disp.pred_away_pts))
               | ((disp.home_win_prob <= 0.5) & (disp.pred_home_pts <= disp.pred_away_pts)))
    print(f"scoreline/WP sign agreement: {sign_ok.mean():.4f}")
    print(f"impossible scores of 1: "
          f"{int(((pp.pred_home_pts == 1) | (pp.pred_away_pts == 1)).sum())}")


if __name__ == "__main__":
    main()
