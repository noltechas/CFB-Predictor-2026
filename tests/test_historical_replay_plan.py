import pandas as pd

from cfbpred.historical_replay_plan import enumerate_cutoffs


def test_cutoffs_have_preseason_and_completed_week_targets():
    schedule = pd.DataFrame({
        "game_id": [1, 2, 3], "season": [2001, 2001, 2001],
        "week": [1, 1, 2],
        "kick": ["2001-08-30T18:00:00Z", "2001-09-01T18:00:00Z", "2001-09-08T18:00:00Z"]})
    rows = enumerate_cutoffs(schedule, 2001, 2001)
    assert [r["week"] for r in rows] == [0, 1, 2]
    assert rows[0]["target_game_count"] == 3
    assert rows[1]["completed_before_cutoff"] == 2
    assert rows[1]["target_game_ids"] == [3]
    assert rows[0]["unsupported_reason"] == "no_prior_data"


def test_reconstruction_is_explicitly_not_walk_forward():
    schedule = pd.DataFrame({"game_id": [1], "season": [2026], "week": [1],
                             "kick": ["2026-08-30T18:00:00Z"]})
    row = enumerate_cutoffs(schedule, 2026, 2026)[0]
    assert row["mode"] == "latest_model_reconstruction"
    assert row["historical_accuracy_claim"] is False
    assert row["walk_forward_state"] == "unsupported_current_adapter"
