"""Evaluation harness for the raw model.

The closing spread appears here in exactly one role: as the external BENCHMARK
the pregame predictions are judged against, at game level, one row per game.
It is never an input anywhere.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.config import (BENCH, MODELS, PBP, TEST_SEASONS,  # noqa: E402
                            TRAIN_SEASONS, VAL_MOD, VAL_REM)
from cfbpred.train import (STATE_FEATURES, build_tensors, eval_plays,  # noqa: E402
                           make_model)

QUARTERS = [(2700, 3601, "q1"), (1800, 2700, "q2"),
            (900, 1800, "q3"), (0, 900, "q4")]


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p, y, nb=20):
    i = np.clip((p * nb).astype(int), 0, nb - 1)
    return float(sum((i == k).sum() / len(p)
                     * abs(p[i == k].mean() - y[i == k].mean())
                     for k in range(nb) if (i == k).sum()))


def cluster_bootstrap_ci(p, y, groups, n=200, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by_g = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n):
        sel = np.concatenate([by_g[g] for g in
                              rng.choice(uniq, size=len(uniq), replace=True)])
        vals.append(logloss(p[sel], y[sel]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def maybe_calibrate(ov, yv, tv):
    """Isotonic per quarter, kept only if it helps a held-out half of the
    validation season (split by game). Post-hoc calibration was rejected on the
    previous model too — the discrete head tends to arrive calibrated."""
    vg = np.unique(ov["gid"])
    rng = np.random.default_rng(0); rng.shuffle(vg)
    half = set(vg[: len(vg) // 2].tolist())
    in_a = np.isin(ov["gid"], list(half))
    cals = {}
    for lo, hi, lab in QUARTERS:
        m = (tv >= lo) & (tv < hi) & in_a
        if m.sum() < 500:
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=.001, y_max=.999)
        iso.fit(ov["wp"][m], yv[m])
        cals[lab] = iso
    out = ov["wp"].copy()
    for lo, hi, lab in QUARTERS:
        if lab in cals:
            m = (tv >= lo) & (tv < hi)
            out[m] = cals[lab].predict(ov["wp"][m])
    raw_l = logloss(ov["wp"][~in_a], yv[~in_a])
    cal_l = logloss(out[~in_a], yv[~in_a])
    helps = cal_l < raw_l - 1e-4
    print(f"isotonic calibration, held-out half of validation: "
          f"{raw_l:.5f} -> {cal_l:.5f}  "
          f"({'KEPT' if helps else 'REJECTED — head already calibrated'})")
    return cals if helps else {}


def report(tag="raw"):
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    d = build_tensors(dev)
    if tag == "ensemble":
        # judge what production actually serves: the seed-averaged PMFs
        from cfbpred.ensemble import (SEEDS, ensemble_play_probs,
                                      ensemble_game_probs, load_members,
                                      readouts as ens_read)
        members = load_members(d, dev)
        tag = f"ensemble-{len(SEEDS)}seed"

        class _Ens:
            pass
        model = _Ens()

        def _eval(mask):
            idx = np.where(mask)[0]
            pm, pt = ensemble_play_probs(members, d, idx, dev)
            r = ens_read(pm, pt)
            return {"wp": r["wp"], "em": r["em"], "et": r["et"]}
        model.eval_override = _eval
    else:
        from cfbpred.ensemble import ensemble_game_probs, load_members, readouts as ens_read
        members = load_members(d, dev, tags=[tag])
        model = members[0]

    # Benchmarks are defined on FBS-vs-FBS: cfbfastR's published numbers and
    # every prior result here were measured there. Including FCS matchups
    # (lopsided, easy) flatters the aggregate by ~0.04 log loss.
    va = (np.isin(d["P_season"], TRAIN_SEASONS) & d["P_ff"]
          & ((np.asarray(d["P_gid"]) % VAL_MOD) == VAL_REM))
    te = np.isin(d["P_season"], TEST_SEASONS) & d["P_ff"]
    graded_on = f"held-out test {TEST_SEASONS}"
    if not te.sum():
        # No test-season play data: judge on the game-level
        # validation holdout instead — no gradients ever saw these games,
        # though early stopping did, so read it as slightly generous. The
        # isotonic step is skipped: it would fit and judge the same games.
        te = va
        graded_on = "game-level validation holdout (test-season play data unavailable)"
    runner = getattr(model, "eval_override",
                     lambda mask: eval_plays(model, d, mask, dev))
    ov = runner(va); ov["gid"] = d["P_gid"][va]
    ot = runner(te) if te is not va else ov
    yv = (d["P_margin"].numpy()[va] > 0).astype(float)
    yt = (d["P_margin"].numpy()[te] > 0).astype(float)
    tv, tt = d["P_t_left"][va], d["P_t_left"][te]

    cals = {} if te is va else maybe_calibrate(ov, yv, tv)
    wp = ot["wp"].copy()
    for lo, hi, lab in QUARTERS:
        if lab in cals:
            m = (tt >= lo) & (tt < hi)
            wp[m] = cals[lab].predict(ot["wp"][m])

    print(f"=== {tag}: {graded_on}, raw inputs only ===")
    print(f"{'bucket':>8s} {'n':>8s} {'logloss':>9s} {'ECE':>7s} {'cfbfastR':>9s}")
    rows = {}
    for lo, hi, lab in [(0, 3601, "all")] + QUARTERS:
        m = (tt >= lo) & (tt < hi)
        ref = BENCH["cfbfastr_wp_ll"][lab if lab != "all" else "all"]
        rows[lab] = {"n": int(m.sum()), "logloss_raw": logloss(wp[m], yt[m]),
                     "ece_raw": ece(wp[m], yt[m]), "cfbfastr": ref}
        print(f"{lab:>8s} {m.sum():8d} {rows[lab]['logloss_raw']:9.5f} "
              f"{rows[lab]['ece_raw']:7.4f} {ref:9.5f}")
    lo_ci, hi_ci = cluster_bootstrap_ci(wp, yt, d["P_gid"][te])
    print(f"95% CI, cluster-bootstrapped by game: [{lo_ci:.5f}, {hi_ci:.5f}]")

    # ---- market benchmark, GAME level, one row per game ----
    pre_m, pre_t = ensemble_game_probs(members, d, dev)
    pre = ens_read(pre_m, pre_t)
    game_mask = d["G_fbs_fbs"] & d["G_labeled"]
    if te is va:
        game_mask &= (np.isin(d["G_season"], TRAIN_SEASONS)
                      & ((d["G_gid"] % VAL_MOD) == VAL_REM))
    else:
        game_mask &= np.isin(d["G_season"], TEST_SEASONS)
    con = duckdb.connect()
    sp = con.execute(f"""SELECT DISTINCT game_id, spread_close
        FROM read_parquet('{PBP / "pbp_*.parquet"}')
        WHERE spread_close IS NOT NULL""").fetchdf().drop_duplicates("game_id")
    g = (pd.DataFrame({"game_id": d["G_gid"][game_mask], "em": pre["em"][game_mask],
                       "margin": d["G_margin"].numpy()[game_mask]})
         .merge(sp, on="game_id", how="inner"))
    mk = -g["spread_close"].to_numpy(float)
    rm = float(np.sqrt(np.mean((g.em - g.margin) ** 2)))
    rk = float(np.sqrt(np.mean((mk - g.margin) ** 2)))
    gap = rm - rk
    verdict = "Compare paired errors on unseen seasons; a lower error is not itself evidence of leakage"
    print(f"\npregame, one row per game (n={len(g):,}):")
    print(f"  model  RMSE {rm:.3f}   market RMSE {rk:.3f}   gap {gap:+.3f}")
    print(f"  -> {verdict}")

    out = {"tag": tag, "graded_on": graded_on,
           "buckets": rows, "logloss_ci": [lo_ci, hi_ci],
           "pregame_game_level": {"prediction_stage": "pregame_tower",
                                  "n": int(len(g)), "model_rmse": rm,
                                  "market_rmse": rk, "gap": gap}}
    (MODELS / "eval_base.json").write_text(json.dumps(out, indent=2,
                                                      default=float))
    return out


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "raw")
