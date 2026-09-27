"""The production predictor: a 3-seed ensemble.

Single training runs of this pipeline vary by up to ±0.01 log loss with the
seed — the early-stopped pregame tower is fragile to initialisation. Averaging
the predicted DISTRIBUTIONS (softmax then mean, never logits) across three
seeds removes most of that variance and reliably lands at or below the best
single seed. Every consistency property survives averaging, because a mixture
of margin PMFs is still a margin PMF: win probability remains a partial sum of
the same distribution the scoreline is read from.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.config import MODELS  # noqa: E402
from cfbpred.model import MARGIN_CENTRES, TOTAL_CENTRES  # noqa: E402
from cfbpred.train import EVAL_BS, game_batch, make_model  # noqa: E402

SEEDS = tuple(range(16))


def align_vocabulary(d, saved):
    """Keep embedding identities fixed when the provider adds/removes teams."""
    for kind, fields in (("team", ("G_hidx", "G_aidx", "G_opp")),
                         ("conf", ("G_hconf", "G_aconf"))):
        current = d["vocab"][kind]
        mapping = torch.zeros(max(current.values(), default=0) + 1, dtype=torch.long)
        for name, index in current.items():
            mapping[index] = saved[kind].get(name, 0)
        for field in fields:
            d[field] = mapping.to(d[field].device)[d[field]]
    d["vocab"] = saved


def load_members(d, dev, tags=None):
    models = []
    preprocessing = None
    for name in (tags or [f"raw_s{s}" for s in SEEDS]):
        ck = torch.load(MODELS / f"two_tower_{name}.pt", weights_only=False, map_location="cpu")
        # The original raw checkpoints predate embedded vocabulary metadata.
        # Their retained pre-refresh contract supplies the exact fitted IDs.
        saved_vocab = ck.get("vocab")
        if saved_vocab is None and name.startswith("raw_s"):
            saved_vocab = json.loads((MODELS / "raw_dnn_vocab.json").read_text())
        if saved_vocab is not None:
            if (len(saved_vocab["team"]) + 1 != ck["state_dict"]["pre.team_emb.weight"].shape[0]
                    or len(saved_vocab["conf"]) + 1 != ck["state_dict"]["pre.conf_emb.weight"].shape[0]):
                raise ValueError("Saved vocabulary does not match the frozen embeddings")
            if d["vocab"] != saved_vocab:
                align_vocabulary(d, saved_vocab)
        if ck["scalar_cols"] != d["scalar_cols"]:
            raise ValueError("Feature columns differ from the checkpoint; retrain or restore its feature contract")
        if preprocessing is None:
            preprocessing = (ck["mu"], ck["sd"])
            # Forecasts use the scaler fitted during training, even after a
            # feature-store refresh. Never silently refit it at inference.
            x = (d["G_raw_scalars"] - ck["mu"]) / ck["sd"]
            d["G_scalars"] = torch.tensor(np.nan_to_num(x, nan=0, posinf=0, neginf=0))
            d["mu"], d["sd"] = preprocessing
        elif not all(np.array_equal(a, b, equal_nan=True) for a, b in zip(preprocessing, (ck["mu"], ck["sd"]))):
            raise ValueError("Ensemble members require different preprocessing; cannot share a feature tensor")
        m = make_model(d, s2d_gate=ck.get("s2d_gate", False),
                       pre_hidden=ck.get("pre_hidden", 160),
                       delta=ck.get("delta", False)).to(dev)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models.append(m)
    return models


def ensemble_play_probs(models, d, idx, dev, bs=EVAL_BS):
    """Averaged margin/total PMFs for the given play indices."""
    pm_sum = pt_sum = None
    with torch.no_grad():
        for m in models:
            pms, pts = [], []
            for i in range(0, len(idx), bs):
                j = torch.tensor(idx[i:i + bs])
                gj = d["P_gidx"][j]
                fr = (torch.tensor(d["P_t_left"][j.numpy()]) / 3600.0).to(dev)
                al = (torch.tensor(np.abs(d["P_lead"][j.numpy()])) / 21.0).to(dev)
                ld = torch.tensor(d["P_lead"][j.numpy()]).to(dev)
                pz = torch.tensor(d["P_pts"][j.numpy()]).to(dev)
                o = m(game_batch(d, gj, dev), d["P_state"][j].to(dev), fr, al,
                      ld, pz)
                pms.append(torch.softmax(o["margin"], -1).cpu())
                pts.append(torch.softmax(o["total"], -1).cpu())
            pm, pt = torch.cat(pms), torch.cat(pts)
            pm_sum = pm if pm_sum is None else pm_sum + pm
            pt_sum = pt if pt_sum is None else pt_sum + pt
    return pm_sum / len(models), pt_sum / len(models)


def ensemble_game_probs(models, d, dev, bs=4096):
    """Averaged pregame PMFs for every game."""
    G = len(d["G_gid"])
    pm_sum = pt_sum = None
    with torch.no_grad():
        for m in models:
            pms, pts = [], []
            for i in range(0, G, bs):
                j = torch.arange(i, min(i + bs, G))
                _, pml, ptl = m.pregame(game_batch(d, j, dev))
                pms.append(torch.softmax(pml, -1).cpu())
                pts.append(torch.softmax(ptl, -1).cpu())
            pm, pt = torch.cat(pms), torch.cat(pts)
            pm_sum = pm if pm_sum is None else pm_sum + pm
            pt_sum = pt if pt_sum is None else pt_sum + pt
    return pm_sum / len(models), pt_sum / len(models)


def readouts(pm: torch.Tensor, pt: torch.Tensor) -> dict:
    pos = (MARGIN_CENTRES > 0).float()
    cm = torch.cumsum(pm, -1)
    ct = torch.cumsum(pt, -1)
    half = torch.full((pm.shape[0], 1), 0.5)
    mi = torch.clamp(torch.searchsorted(cm.contiguous(), half), 0,
                     pm.shape[-1] - 1).squeeze(-1)
    ti = torch.clamp(torch.searchsorted(ct.contiguous(), half), 0,
                     pt.shape[-1] - 1).squeeze(-1)
    return {"wp": (pm * pos).sum(-1).numpy(),
            "em": (pm * MARGIN_CENTRES).sum(-1).numpy(),
            "et": (pt * TOTAL_CENTRES).sum(-1).numpy(),
            "mm": MARGIN_CENTRES[mi].numpy(),
            "mt": TOTAL_CENTRES[ti].numpy()}


if __name__ == "__main__":
    import json
    from cfbpred.config import TEST_SEASONS, TRAIN_SEASONS, VAL_MOD, VAL_REM
    from cfbpred.train import build_tensors, metrics
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    d = build_tensors(dev)
    models = load_members(d, dev)
    te = np.isin(d["P_season"], TEST_SEASONS) & d["P_ff"]
    graded_on = f"test {TEST_SEASONS}"
    if not te.sum():
        # the test season hasn't produced plays yet (e.g. 2026 preseason):
        # grade on the game-level validation holdout — never in a gradient,
        # though it did steer early stopping, so it reads slightly generous
        te = (np.isin(d["P_season"], TRAIN_SEASONS) & d["P_ff"]
              & ((np.asarray(d["P_gid"]) % VAL_MOD) == VAL_REM))
        graded_on = "game-level validation holdout (test season unplayed)"
    idx = np.where(te)[0]
    pm, pt = ensemble_play_probs(models, d, idx, dev)
    r = readouts(pm, pt)
    m = metrics(r["wp"], r["em"], r["et"], d["P_margin"].numpy()[te],
                d["P_total"].numpy()[te], d["P_t_left"][te])
    print(f"{len(SEEDS)}-seed ensemble, held-out FBS-vs-FBS ({graded_on}):")
    for k in ("logloss", "logloss_q1", "logloss_q2", "logloss_q3",
              "logloss_q4", "ece", "margin_rmse", "total_rmse"):
        print(f"  {k:14s} {m[k]:.5f}")
    # the production predictor IS the ensemble — its held-out metrics are what
    # publish.py stamps onto the model_runs row
    (MODELS / "metrics_raw.json").write_text(
        json.dumps({"test": m, "graded_on": graded_on,
                    "ensemble_seeds": len(SEEDS)}, indent=2))
    print(f"wrote {MODELS / 'metrics_raw.json'}")
