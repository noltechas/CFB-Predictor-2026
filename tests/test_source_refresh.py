"""A daily feature refresh must not redownload the unchanged play corpus."""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfbpred import source


def test_selective_refresh_preserves_play_cache_and_updates_features(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "CACHE", tmp_path)
    play_path = tmp_path / "play_raw.parquet"
    pd.DataFrame({"game_id": [1], "play_index": [0]}).to_parquet(play_path)
    original = play_path.read_bytes()
    calls = []

    def pull(table, order):
        calls.append((table, order))
        return pd.DataFrame({"game_id": [2]})

    monkeypatch.setattr(source, "pull", pull)
    source.refresh_all(["game_scalars", "game_history"])
    assert [table for table, _ in calls] == ["game_scalars", "game_history"]
    assert play_path.read_bytes() == original
    assert source.load("play_raw").game_id.tolist() == [1]
    for table in ("game_scalars", "game_history"):
        assert source.load(table).game_id.tolist() == [2]
    assert len(calls) == 2


def test_missing_play_cache_is_pulled_once(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "CACHE", tmp_path)
    calls = []

    def pull(table, order):
        calls.append(table)
        return pd.DataFrame({"game_id": [1], "play_index": [0]})

    monkeypatch.setattr(source, "pull", pull)
    source.load("play_raw")
    source.load("play_raw")
    assert calls == ["play_raw"]


def test_explicit_full_refresh_still_updates_the_play_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "CACHE", tmp_path)
    calls = []

    def pull(table, order):
        calls.append(table)
        return pd.DataFrame({"game_id": [3]})

    monkeypatch.setattr(source, "pull", pull)
    source.refresh_all()
    assert calls == list(source.TABLES)
    assert source.load("play_raw").game_id.tolist() == [3]
