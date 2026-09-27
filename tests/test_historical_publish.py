import json

import pandas as pd
import pytest

from cfbpred.historical_publish import historical_cohort, validate_archive
from cfbpred.pregame_experiments import sha


def _archive(tmp_path):
    pred = pd.DataFrame({"game_id": [1, 2], "season": [2025, 2026], "split": ["train", "test"],
                         "model_version": ["v", "v"], "home_win_prob": [.4, .6],
                         "pred_margin": [1., 2.], "pred_total": [40., 41.],
                         "pred_home_pts": [21, 22], "pred_away_pts": [20, 19]})
    pmf = pd.DataFrame({"game_id": [1, 2], "season": [2025, 2026],
                        "margin_pmf": [[1.], [1.]], "total_pmf": [[1.], [1.]]})
    pred.to_parquet(tmp_path / "game_predictions.parquet")
    pmf.to_parquet(tmp_path / "game_pmfs.parquet")
    (tmp_path / "metadata.json").write_text(json.dumps({"prediction_sha256": sha(tmp_path / "game_predictions.parquet"),
        "pmf_sha256": sha(tmp_path / "game_pmfs.parquet"), "game_count": 2, "model_version": "v"}))


def test_historical_cohort_preserves_current_season_rows():
    p = pd.DataFrame({"game_id": [1, 2], "season": [2025, 2026]})
    f = pd.DataFrame({"game_id": [1, 2], "season": [2025, 2026]})
    hp, hf = historical_cohort(p, f)
    assert hp.game_id.tolist() == hf.game_id.tolist() == [1]


def test_validate_archive_rejects_tampered_values(tmp_path):
    _archive(tmp_path)
    pd.DataFrame({"game_id": [1, 2], "season": [2025, 2026], "split": ["train", "test"],
                  "model_version": ["v", "v"], "home_win_prob": [.9, .6],
                  "pred_margin": [1., 2.], "pred_total": [40., 41.],
                  "pred_home_pts": [21, 22], "pred_away_pts": [20, 19]}).to_parquet(tmp_path / "game_predictions.parquet")
    with pytest.raises(ValueError, match="hash"):
        validate_archive(tmp_path)
