"""Reconstruct player inputs at one ranking cutoff, including bye weeks."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from cfbpred.player_timeline import compile_arrays


def restrict_context(store, game_indices, cutoff, players, sequences):
    refs, valid = store.roster_refs(game_indices)
    packed = np.unique(refs[valid])
    catalog = pd.read_parquet(players / "player_seasons.parquet", columns=["person_id", "season"])
    people = catalog.iloc[store.player_refs[packed] - 1]
    events = pd.read_parquet(sequences / "events.parquet", columns=["event_id", "person_id", "kick"])
    before = events[pd.to_datetime(events.kick, utc=True) < cutoff]
    latest = before.sort_values(["kick", "event_id"]).drop_duplicates("person_id", keep="last").set_index("person_id").event_id
    tips = people.person_id.map(latest).fillna(0).to_numpy(np.int32)
    store.history_tips = np.asarray(store.history_tips).copy()
    store.history_tips[packed] = tips
    history = pd.read_parquet(players / "history_provenance.parquet")
    history = history[pd.to_datetime(history.kick, utc=True) < cutoff]
    latest_current = (history.sort_values(["kick", "snapshot_id"])
                      .drop_duplicates(["person_id", "season"], keep="last")
                      .set_index(["person_id", "season"]).snapshot_id)
    current = latest_current.reindex(pd.MultiIndex.from_frame(people[["person_id", "season"]])).fillna(0).to_numpy(np.int32)
    store.current_refs = np.asarray(store.current_refs).copy()
    store.current_refs[packed] = current
    if (store.event_kick[tips[tips > 0]] >= cutoff.value).any():
        raise ValueError("Ranking contains a later player observation")

    rosters = pd.read_parquet(players / "game_rosters.parquet")
    # Packed offsets unambiguously identify both sides even when game indices
    # differ from the underlying schedule order.
    selected_offsets = store.offsets[game_indices].ravel()
    selected_counts = store.counts[game_indices].ravel()
    offsets = set(map(int, selected_offsets[selected_counts > 0]))
    rosters = rosters[rosters.offset.isin(offsets) & rosters.players.gt(0)].copy()
    if rosters.empty:
        return dict(cutoff=cutoff.isoformat(), player_histories=0, roster_slots=0,
                    latest_player_event=None, knowledge_cutoff_strict=True,
                    empty_rosters_preserved_as_missing=True)
    old_slots = np.concatenate([np.arange(r.offset, r.offset + r.players) for r in rosters.itertuples()])
    rosters["offset"] = np.r_[0, rosters.players.to_numpy().cumsum()[:-1]]
    rosters["kick"] = cutoff
    facts = pd.read_parquet(store.timeline_path / "facts.parquet")
    facts = facts[pd.to_datetime(facts.known_at, utc=True) < cutoff]
    with tempfile.TemporaryDirectory(prefix="cfb-ranking-context-") as temp:
        output = Path(temp)
        report = compile_arrays(facts, rosters, store.player_refs[old_slots], output)
        for name in ("context", "position", "departed", "quarantined"):
            array = np.asarray(getattr(store, "timeline_" + name)).copy()
            array[old_slots] = np.load(output / (name + ".npy"))
            setattr(store, "timeline_" + name, array)
    report.update(cutoff=cutoff.isoformat(), player_histories=int((tips > 0).sum()),
                  latest_player_event=pd.to_datetime(before.kick, utc=True).max().isoformat(),
                  knowledge_cutoff_strict=True)
    return report


def verify_scalar_cutoff(chosen, schedule, cutoff, fields, records):
    """Each selected row must be the next game, with no intervening result."""
    for r in chosen.itertuples():
        team_games = schedule[(schedule.home_team.eq(r.team) | schedule.away_team.eq(r.team))
                              & schedule.season.eq(r.season)]
        intervening = team_games[(team_games.kick >= cutoff) & (team_games.kick < r.kick)]
        if len(intervening):
            raise ValueError(f"Ranking scalar target skips a game for {r.team}")
    expected = records.reindex(chosen.team).gp.to_numpy()
    if not np.array_equal(fields.gp.to_numpy(), expected):
        mismatched = fields.gp.to_numpy() != expected
        detail = [{"team": str(team), "scalar_games": float(actual), "cutoff_games": float(want)}
                  for team, actual, want in zip(chosen.team[mismatched],
                                               fields.gp.to_numpy()[mismatched], expected[mismatched])]
        raise ValueError(f"Ranking scalar result counts disagree with the global cutoff: {detail}")
