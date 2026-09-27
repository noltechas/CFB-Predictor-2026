import json

import pandas as pd
import pytest

from cfbpred.complete_history_comparison import compare
from cfbpred.pregame_experiments import sha


def _run(root, source, margin_shift=0.0, epochs=14):
    root.mkdir()
    plan = {"variant": "full_context", "year": 2025, "seed": 0,
            "fixed_epochs": epochs, "refit_through_year": 2024,
            "micro_batch": 64, "epoch_selection_year": 2024, "labels_2026_used": False,
            "code": {"net.py": "same"}, "epoch_source": {"x": 1},
            "source_sha256": source, "stores": {"timeline": source}}
    (root / "plan.json").write_text(json.dumps(plan))
    frame = pd.DataFrame({"game_id": [1, 2, 3, 4], "season": [2025] * 4,
        "week_c": [1, 1, 2, 2], "margin": [3, -2, 7, -4], "total": [40] * 4,
        "fbs_fbs": [True] * 4, "wp": [.7, .4, .8, .3], "em": [4 + margin_shift, -1, 5, -2],
        "et": [40] * 4, "mm": [3] * 4, "spread_close": [-3, 2, -6, 4], "crps": [1.] * 4})
    frame.to_parquet(root / "predictions.parquet")


def test_fixed_market_cohort_and_history_change_are_recorded(tmp_path):
    b, c = tmp_path / "b", tmp_path / "c"
    _run(b, "old"); _run(c, "new", margin_shift=-1)
    market = tmp_path / "market.parquet"
    pd.DataFrame({"game_id": [1, 2, 3, 4], "spread_close": [-3, 2, -6, 4]}).to_parquet(market)
    report = compare(b, c, market, tmp_path / "out")
    assert report["target_games"] == 4
    assert report["provenance"]["history_changed"]
    assert report["primary_acceptance"]["margin_mae_decrease"]


def test_settings_change_is_rejected(tmp_path):
    b, c = tmp_path / "b", tmp_path / "c"
    _run(b, "old"); _run(c, "new", epochs=15)
    market = tmp_path / "market.parquet"
    pd.DataFrame({"game_id": [1, 2, 3, 4], "spread_close": [-3, 2, -6, 4]}).to_parquet(market)
    with pytest.raises(ValueError, match="setting changed"):
        compare(b, c, market, tmp_path / "out")


def test_target_labels_must_match(tmp_path):
    b, c = tmp_path / "b", tmp_path / "c"
    _run(b, "old"); _run(c, "new")
    frame = pd.read_parquet(c / "predictions.parquet"); frame.loc[0, "margin"] = 99; frame.to_parquet(c / "predictions.parquet")
    market = tmp_path / "market.parquet"
    pd.DataFrame({"game_id": [1, 2, 3, 4], "spread_close": [-3, 2, -6, 4]}).to_parquet(market)
    with pytest.raises(ValueError, match="labels differ"):
        compare(b, c, market, tmp_path / "out")
