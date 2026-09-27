"""Rolling-origin evaluation: every fold trains through year-1 and predicts year.

Scores each arm on the fixed FBS-vs-FBS cohort with a closing line, as single seeds and
as a seed ensemble (probability mass averaged, which is how production would ship it).
Closing spreads and totals are evaluation-only; they never enter a tensor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cfbpred.model import MARGIN_CENTRES, TOTAL_CENTRES
from cfbpred.pregame_experiments import save_json

MC, TC = MARGIN_CENTRES.numpy(), TOTAL_CENTRES.numpy()
SPREADS = Path("models/experiments/players_20260906_v3/inputs/market.parquet")
LINES = Path("data/parquet/cfbd/lines.parquet")
BUCKETS = [(0, 3, "weeks 1-3"), (4, 6, "weeks 4-6"), (7, 9, "weeks 7-9"), (10, 13, "weeks 10-13"), (14, 99, "postseason")]


def closing_totals() -> pd.Series:
    """Median over providers of the posted total, per game."""
    lines = pd.read_parquet(LINES, columns=["id", "lines"])
    rows = [(int(r.id), np.median([x["overUnder"] for x in r.lines if x.get("overUnder") is not None]))
            for r in lines.itertuples() if r.lines is not None and any(x.get("overUnder") is not None for x in r.lines)]
    return pd.Series(dict(rows), name="total_close")


def load_run(run: Path):
    frame = pd.read_parquet(run / "predictions.parquet").set_index("game_id")
    dist = np.load(run / "distributions.npz")
    pos = pd.Series(np.arange(len(dist["game_id"])), index=dist["game_id"])
    return frame, dist["pm"], dist["pt"], pos


def scores(pm, pt, truth_m, truth_t):
    em, et = (pm * MC).sum(-1), (pt * TC).sum(-1)
    wp = (pm * (MC > 0)).sum(-1).clip(1e-6, 1 - 1e-6); win = (truth_m > 0).astype(float)
    return dict(margin_ae=np.abs(em - truth_m), total_ae=np.abs(et - truth_t),
                wp_ll=-(win * np.log(wp) + (1 - win) * np.log(1 - wp)), brier=(wp - win) ** 2)


def evaluate(root: Path, arms: list[str], years: list[int], seeds: list[int]) -> dict:
    spreads = pd.read_parquet(SPREADS).set_index("game_id").spread_close
    totals = closing_totals()
    per_game = []
    for year in years:
        ref, *_ = load_run(root / f"{arms[0]}_{year}_s{seeds[0]}")
        ids = ref.index[(ref.season == year) & ref.fbs_fbs & ref.index.isin(spreads.dropna().index)]
        meta = ref.loc[ids, ["margin", "total", "week_c"]].copy()
        meta["spread_close"] = spreads.loc[ids].to_numpy(); meta["total_close"] = totals.reindex(ids).to_numpy()
        meta["year"] = year
        for arm in arms:
            pms, pts = [], []
            for seed in seeds:
                run = root / f"{arm}_{year}_s{seed}"
                if not (run / "report.json").exists():
                    raise FileNotFoundError(run)
                frame, pm, pt, pos = load_run(run)
                p = pos.loc[ids].to_numpy(); pms.append(pm[p]); pts.append(pt[p])
                s = scores(pm[p], pt[p], meta.margin.to_numpy(), meta.total.to_numpy())
                per_game.append(meta.assign(arm=arm, members=f"seed{seed}", **s).reset_index())
            s = scores(np.mean(pms, 0), np.mean(pts, 0), meta.margin.to_numpy(), meta.total.to_numpy())
            per_game.append(meta.assign(arm=arm, members="ensemble", **s).reset_index())
    g = pd.concat(per_game, ignore_index=True)
    g["market_margin_ae"] = (-g.spread_close - g.margin).abs(); g["market_total_ae"] = (g.total_close - g.total).abs()
    g["bucket"] = pd.cut(g.week_c, [-1] + [b[1] for b in BUCKETS], labels=[b[2] for b in BUCKETS])
    single = g[g.members != "ensemble"].groupby(["arm", "year", "game_id"])[["margin_ae", "total_ae", "wp_ll", "brier"]].mean().reset_index()
    ens = g[g.members == "ensemble"]
    out = {"cohort": {int(y): int((ens.arm.eq(arms[0]) & ens.year.eq(y)).sum()) for y in years}, "arms": {}}
    for arm in arms:
        e, s = ens[ens.arm == arm], single[single.arm == arm]
        out["arms"][arm] = dict(
            ensemble={k: float(e[k].mean()) for k in ("margin_ae", "total_ae", "wp_ll", "brier")},
            mean_single_seed={k: float(s[k].mean()) for k in ("margin_ae", "total_ae", "wp_ll", "brier")},
            by_year={int(y): float(e[e.year == y].margin_ae.mean()) for y in years},
            by_bucket={str(b): dict(games=int(len(x)), model=float(x.margin_ae.mean()), market=float(x.market_margin_ae.mean()),
                                    gap=float((x.margin_ae - x.market_margin_ae).mean()))
                       for b, x in e.groupby("bucket", observed=True)})
    base = ens[ens.arm == arms[0]].set_index(["year", "game_id"])
    out["market"] = dict(margin_ae=float(base.market_margin_ae.mean()),
                         total_ae=float(base.market_total_ae.dropna().mean()),
                         total_cohort=int(base.market_total_ae.notna().sum()))
    rng = np.random.default_rng(0); weeks = base.reset_index()[["year", "week_c"]].drop_duplicates().to_numpy()
    for arm in arms[1:]:
        cand = ens[ens.arm == arm].set_index(["year", "game_id"]).loc[base.index]
        diff = (cand.margin_ae - base.margin_ae).to_frame("d").join(base[["week_c"]]).reset_index()
        groups = [diff[(diff.year == y) & (diff.week_c == w)].d.to_numpy() for y, w in weeks]
        boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(2000)]
        out["arms"][arm]["paired_vs_" + arms[0]] = dict(
            margin_mae_change=float(diff.d.mean()), week_cluster_95ci=[float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
            by_year={int(y): float(diff[diff.year == y].d.mean()) for y in years},
            wp_ll_change=float((cand.wp_ll - base.wp_ll).mean()), total_ae_change=float((cand.total_ae - base.total_ae).mean()))
    return out


def ensemble_curve(root: Path, arm: str, years: list[int], seeds: list[int]) -> dict:
    """Margin MAE as seeds are added, averaged over every ordering of the available seeds."""
    spreads = pd.read_parquet(SPREADS).set_index("game_id").spread_close
    curve = {}
    for year in years:
        runs = [root / f"{arm}_{year}_s{s}" for s in seeds]
        ref, *_ = load_run(runs[0]); ids = ref.index[(ref.season == year) & ref.fbs_fbs & ref.index.isin(spreads.dropna().index)]
        truth = ref.loc[ids, "margin"].to_numpy(); pms = []
        for run in runs:
            _, pm, _, pos = load_run(run); pms.append(pm[pos.loc[ids].to_numpy()])
        from itertools import combinations
        curve[int(year)] = {k: float(np.mean([np.abs((np.mean([pms[i] for i in c], 0) * MC).sum(-1) - truth).mean()
                                              for c in combinations(range(len(pms)), k)])) for k in range(1, len(pms) + 1)}
    return curve


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--years", nargs="+", type=int, required=True); p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--output", type=Path)
    a = p.parse_args(); r = evaluate(a.root, a.arms, a.years, a.seeds)
    print(json.dumps(r, indent=2))
    if a.output: save_json(a.output, r)
