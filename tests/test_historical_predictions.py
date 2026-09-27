import json

import numpy as np
import pytest

from cfbpred.historical_predictions import positive_game_indices, reconstruct


def test_positive_game_indices_excludes_virtual_and_missing_ids():
    assert positive_game_indices([63770, -1, 0, np.nan, 63771]).tolist() == [0, 4]


def test_reconstruction_rejects_duplicate_positive_ids_before_output(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    staged = tmp_path / "staged.json"
    source.write_text(json.dumps({"inputs": str(tmp_path), "players": str(tmp_path),
                                  "sequences": str(tmp_path)}))
    staged.write_text(json.dumps({"version": "v"}))
    monkeypatch.setattr("cfbpred.historical_predictions.active_bundle",
                        lambda _: ({"version": "v"}, tmp_path / "bundle.json",
                                   {"model_version": "v", "members": ["m"],
                                    "member_sha256": {"m": "h"}}))
    monkeypatch.setattr("cfbpred.historical_predictions.load_snapshot",
                        lambda *args: {"G_gid": np.array([1, -2, 1]),
                                       "G_season": np.array([2001, 2001, 2002])})
    with pytest.raises(ValueError, match="duplicate positive"):
        reconstruct(source, staged, tmp_path / "out")
    assert not (tmp_path / "out").exists()
