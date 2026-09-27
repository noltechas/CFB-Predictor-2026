import json

import pandas as pd
import pytest

from cfbpred.game_id_reconciliation import load_aliases, reconcile_file, reconcile_frame


def _rows(values):
    return pd.DataFrame(values, columns=["game_id", "home_away", "athlete_id", "category", "stat_type", "stat", "athlete"])


def test_remap_deduplicates_exact_rows_and_quarantines_conflicts():
    frame = _rows([
        (10, "home", "p", "rushing", "YDS", "5", "P"),
        (20, "home", "p", "rushing", "YDS", "5", "P"),  # exact after remap
        (10, "away", "q", "passing", "YDS", "8", "Q"),
        (20, "away", "q", "passing", "YDS", "9", "Q"),  # conflicting stat
    ])
    result = reconcile_frame(frame, {20: 10})
    assert len(result.frame) == 1
    assert result.frame.iloc[0].game_id == 10
    assert len(result.quarantined) == 2
    assert result.audit["exact_duplicate_rows_removed"] == 1
    assert result.audit["conflicting_key_groups"] == 1


def test_fcs_style_rows_and_all_raw_columns_are_preserved():
    frame = _rows([(20, "away", "p", "rushing", "YDS", "5", "P")])
    result = reconcile_frame(frame, {20: 10})
    assert list(result.frame.columns) == list(frame.columns)
    assert result.frame.iloc[0].to_dict() == {**frame.iloc[0].to_dict(), "game_id": 10}


def test_alias_loader_rejects_chains_and_cli_records_hashes(tmp_path):
    raw_path = tmp_path / "player_game_stats.parquet"
    aliases_path = tmp_path / "aliases.json"
    output = tmp_path / "reconciled"
    _rows([(20, "home", "p", "rushing", "YDS", "5", "P")]).to_parquet(raw_path, index=False)
    aliases_path.write_text(json.dumps({"20": 10}))
    audit = reconcile_file(raw_path, aliases_path, output)
    assert audit["input_sha256"] and audit["output_sha256"]
    assert audit["output_rows"] == 1
    assert pd.read_parquet(output / raw_path.name).iloc[0].game_id == 10
    aliases_path.write_text(json.dumps({"20": 10, "10": 1}))
    with pytest.raises(ValueError, match="chains"):
        load_aliases(aliases_path)
