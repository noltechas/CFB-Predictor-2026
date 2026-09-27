import pandas as pd
from cfbpred.historical_prediction_inputs import remap_indices


def test_reused_player_vocabulary_preserves_unknown_and_team_identity():
    current = {'A': 1, 'B': 2}
    saved = {'B': 7, 'rare_virtual_only': 8, 'A': 9}
    frame = pd.DataFrame({'h_idx': [1, 2, 0], 'a_idx': [2, 0, 1], 'label': [1, 2, 3]})
    got = remap_indices(frame, ['h_idx', 'a_idx'], current, saved)
    assert got.h_idx.tolist() == [9, 7, 0]
    assert got.a_idx.tolist() == [7, 0, 9]
    assert got.label.equals(frame.label)
    assert frame.h_idx.tolist() == [1, 2, 0]
