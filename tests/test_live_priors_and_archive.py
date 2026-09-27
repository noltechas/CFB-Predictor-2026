import json

import numpy as np
import pandas as pd
import torch

import cfbpred.publish_live_priors as priors
from cfbpred.snapshot_sources import canonical


def test_live_prior_rows_reproduce_the_published_pmf_for_every_state_member(tmp_path, monkeypatch):
    pm = np.random.default_rng(0).dirichlet(np.ones(115)); pt = np.random.default_rng(1).dirichlet(np.ones(103))
    pd.DataFrame([dict(game_id=7, season=2026, margin_pmf=pm.tolist(), total_pmf=pt.tolist()),
                  dict(game_id=8, season=2026, margin_pmf=pm.tolist(), total_pmf=pt.tolist())]).to_parquet(tmp_path / "game_pmfs.parquet")
    pd.DataFrame([dict(game_id=7, model_version="player-full-context-tabm16-v9")]).to_parquet(tmp_path / "game_predictions.parquet")
    monkeypatch.setattr(priors, "PQ", tmp_path)
    rows = priors.rows_for({7}, 2026, members=16)
    assert [r[0] for r in rows] == [7]                      # only games in the window
    game_id, season, version, members, margin, total = rows[0]
    assert version == "prior:player-full-context-tabm16-v9" and members == 16
    m = torch.tensor(margin).reshape(16, 115)
    # the edge function softmaxes each member's slice: every member must read the published PMF
    assert torch.allclose(m.softmax(-1), torch.tensor(pm, dtype=torch.float32).expand(16, -1), atol=1e-6)
    assert torch.allclose(torch.tensor(total).reshape(16, 103).softmax(-1)[0], torch.tensor(pt, dtype=torch.float32), atol=1e-6)


def test_archive_hash_ignores_key_order_but_not_content():
    a = canonical([{"team": "Oregon", "players": 114}]); b = canonical([{"players": 114, "team": "Oregon"}])
    assert a == b and canonical([{"team": "Oregon", "players": 115}]) != a
    assert json.loads(a) == [{"players": 114, "team": "Oregon"}]
