import pandas as pd
import pytest

from cfbpred.fixed_weight_context_diagnostic import fixed_cohort


def test_fixed_cohort_is_exact_2025_fbs_market_population():
    frame = pd.DataFrame({
        "game_id": [3, 1, 2], "season": [2025, 2025, 2024],
        "week_c": [2, 1, 1], "margin": [4, -2, 1], "total": [40, 44, 31],
        "fbs_fbs": [True, True, True], "spread_close": [-3.5, None, -1.0],
    })
    out = fixed_cohort(frame)
    assert out.game_id.tolist() == [3]


def test_fixed_cohort_rejects_duplicate_ids():
    frame = pd.DataFrame({"game_id": [1, 1], "season": [2025, 2025], "week_c": [1, 1],
                          "margin": [1, 1], "total": [40, 40], "fbs_fbs": [True, True],
                          "spread_close": [-2, -2]})
    with pytest.raises(ValueError, match="unique"):
        fixed_cohort(frame)
