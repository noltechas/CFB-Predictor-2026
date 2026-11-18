"""Off-laptop retraining: the CI state follows the active/staged release (and nothing the
daily set owns), and the pre-registered guard accepts or rejects on the stated slack."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cfbpred import ci_retrain, ci_state
from cfbpred.model import MARGIN_CENTRES


def test_static_state_follows_the_active_release_but_not_daily_refreshes(tmp_path, monkeypatch):
    monkeypatch.setattr(ci_state, "ROOT", tmp_path)
    reg = tmp_path / "models/psp"
    (reg / "v11").mkdir(parents=True)
    (reg / "refreshes/20261118T000000Z").mkdir(parents=True)
    (tmp_path / "models/exp/stores/20261118T010101Z").mkdir(parents=True)
    (reg / "active.json").write_text(json.dumps({"bundle": f"{reg}/v11/bundle.json",
                                                 "note": f"{reg}/refreshes/20261118T000000Z/x"}))
    (reg / "v11/source.json").write_text(json.dumps({"inputs": f"{tmp_path}/models/exp/stores/20261118T010101Z/inputs"}))
    spec = {"roots": ["models/psp/active.json",
                      {"runs_from": ["models/psp/active.json"], "under": ["models/psp", "models/exp/stores"],
                       "match": ["v1*", "20[0-9][0-9]*"], "follow": ["source.json"]}]}
    assert ci_state.resolve_roots(spec) == ["models/exp/stores/20261118T010101Z", "models/psp/active.json",
                                            "models/psp/v11"]


def _frame(ids, margins, wps):
    centres = np.asarray(MARGIN_CENTRES, dtype=float)
    pmfs = []
    for m in margins:                                # all mass on the bin nearest the margin
        p = np.zeros(len(centres)); p[int(np.abs(centres - m).argmin())] = 1.0; pmfs.append(p)
    return pd.DataFrame({"game_id": ids, "margin_pmf": pmfs, "home_win_prob": wps})


def test_guard_accepts_within_slack_and_rejects_beyond(tmp_path, monkeypatch):
    monkeypatch.setattr(ci_retrain, "ROOT", tmp_path)
    monkeypatch.setattr(ci_retrain, "GUARD", dict(ci_retrain.GUARD, min_games=3))
    (tmp_path / "data/parquet").mkdir(parents=True)
    ids = [1, 2, 3, 4]
    pd.DataFrame({"game_id": ids, "season": ci_retrain.CURRENT_SEASON, "home_points": [30, 20, 10, 24],
                  "away_points": [20, 27, 14, 21], "home_division": ["fbs", "fbs", "fcs", "fbs"],
                  "away_division": ["fbs", "fcs", "fbs", "fbs"]}).to_parquet(tmp_path / "data/parquet/schedule.parquet")
    active = _frame(ids, [10, -7, -4, 3], [0.8, 0.3, 0.4, 0.6])
    same = ci_retrain.guard(active, active)
    assert same["accepted"] and same["games"] == 4
    worse = ci_retrain.guard(_frame(ids, [0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]), active)
    assert not worse["accepted"] and worse["candidate"]["margin_mae"] > worse["active"]["margin_mae"]
