from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cfbpred.player_timeline import CONTEXT_COLS, normalize
from cfbpred.ranking_player_cutoff import restrict_context, verify_scalar_cutoff


def test_ranking_excludes_later_events_and_facts_but_preserves_prior_current_season(tmp_path):
    cutoff = pd.Timestamp("2026-09-10T12:00Z")
    players, sequences, timeline = [tmp_path / name for name in ("players", "sequences", "timeline")]
    for path in (players, sequences, timeline):
        path.mkdir()
    pd.DataFrame({"person_id": ["p"], "season": [2026]}).to_parquet(players / "player_seasons.parquet")
    kicks = pd.to_datetime(["2026-09-01T12:00Z", "2026-09-15T12:00Z"])
    pd.DataFrame({"person_id": ["p", "p"], "event_id": [1, 2], "kick": kicks}).to_parquet(sequences / "events.parquet")
    pd.DataFrame({"person_id": ["p", "p"], "season": [2026, 2026], "snapshot_id": [1, 2], "kick": kicks}).to_parquet(players / "history_provenance.parquet")
    pd.DataFrame({"game_id": [7], "season": [2026], "offset": [0], "players": [1], "kick": [cutoff + pd.Timedelta(days=7)]}).to_parquet(players / "game_rosters.parquet")
    facts = []
    for known, kind, value in [("2026-09-01T00:00Z", "position", "QB"), ("2026-09-11T00:00Z", "availability", "out")]:
        facts.append(dict(player_ref=1, season=2026, game_id=7, known_at=known, effective_at=known,
                          valid_until="2027-01-01T00:00Z", kind=kind, value=value, source_url="https://example.test", source_sha256="sha"))
    normalize(facts).to_parquet(timeline / "facts.parquet")
    store = SimpleNamespace(player_refs=np.array([1]), history_tips=np.array([2]), current_refs=np.array([2]),
        event_kick=np.r_[0, kicks.astype('int64')], offsets=np.array([[0]]), counts=np.array([[1]]),
        timeline_path=timeline, timeline_context=np.ones((1, len(CONTEXT_COLS)), np.float32),
        timeline_position=np.array([8]), timeline_departed=np.array([True]), timeline_quarantined=np.array([True]),
        roster_refs=lambda _: (np.array([[[0]]]), np.array([[[True]]])))
    report = restrict_context(store, np.array([0]), cutoff, players, sequences)
    assert store.history_tips.tolist() == [1]
    assert store.current_refs.tolist() == [1]
    assert store.timeline_position.tolist() == [1]
    assert store.timeline_context[0, CONTEXT_COLS.index("report_known")] == 0
    assert not store.timeline_departed.any()
    assert report["knowledge_cutoff_strict"]


def test_scalar_replay_rejects_skipping_intervening_game():
    cutoff = pd.Timestamp("2026-09-10T00:00Z")
    schedule = pd.DataFrame({"home_team": ["A", "A"], "away_team": ["B", "C"], "season": [2026, 2026],
                             "kick": pd.to_datetime(["2026-09-12T00:00Z", "2026-09-19T00:00Z"])})
    chosen = pd.DataFrame({"team": ["A"], "season": [2026], "kick": [schedule.kick.iloc[1]]})
    with pytest.raises(ValueError, match="skips a game"):
        verify_scalar_cutoff(chosen, schedule, cutoff, pd.DataFrame({"gp": [1]}), pd.DataFrame({"gp": [1]}, index=["A"]))
