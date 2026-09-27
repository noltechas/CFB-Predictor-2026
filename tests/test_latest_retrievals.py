"""Collapsing repeated report captures must not change a single compiled input."""
import random

import numpy as np
import pandas as pd
import pytest

from cfbpred.conference_availability import latest_retrievals
from cfbpred.player_timeline import compile_arrays, name_key, normalize

KICKS = {1: "2026-09-05T18:00:00Z", 2: "2026-09-12T18:00:00Z", 3: "2026-09-19T18:00:00Z"}
NAMES = ["John Smith", "John Smith Jr.", "Ray Jones", "Al Brown"]  # first two share a name key
REFS = {name_key(n): i + 1 for i, n in enumerate(dict.fromkeys(name_key(n) for n in NAMES))}


def row(name, game_id, value, known, *, assumed=False, source="https://conf.example/report"):
    kick = pd.Timestamp(KICKS[game_id])
    return dict(name=name, team="USC", season=2026, game_id=game_id, kind="availability",
                value=value, known_at=known.isoformat(), effective_at=known.isoformat(),
                valid_until=(kick + pd.Timedelta(days=1)).isoformat(), timing_assumed=assumed,
                source_url=source, source_sha256="x", player_ref=REFS[name_key(name)])


def history(rng):
    """Captures every ~30 min, statuses that change and revert, same-capture
    conflicts, post-kickoff T-30 assumptions, and an independent source."""
    facts = []
    for game_id, kick in KICKS.items():
        kick = pd.Timestamp(kick)
        for name in NAMES:
            value = rng.choice(["out", "questionable", "probable"])
            t = kick - pd.Timedelta(hours=rng.randint(4, 60))
            while t < kick:
                if rng.random() < 0.25:
                    value = rng.choice(["out", "doubtful", "questionable", "probable", "available"])
                if rng.random() < 0.8:
                    facts.append(row(name, game_id, value, t))
                if rng.random() < 0.1:  # same capture lists the player twice, differently
                    facts.append(row(name, game_id, rng.choice(["out", "available"]), t))
                t += pd.Timedelta(minutes=rng.choice([29, 30, 31, 35]))
            if rng.random() < 0.5:  # archive captured after kickoff: assumed T-30
                facts.append(row(name, game_id, rng.choice(["out", "questionable"]),
                                 kick - pd.Timedelta(minutes=30), assumed=True))
    return facts


def other_sources(rng):
    """Uncollapsed evidence that can tie with an assumed T-30 conference row."""
    facts = []
    for game_id, kick in KICKS.items():
        for name in NAMES[2:]:
            if rng.random() < 0.5:
                facts.append(row(name, game_id, rng.choice(["out", "questionable"]),
                                 pd.Timestamp(kick) - pd.Timedelta(minutes=30), assumed=True,
                                 source="https://pdf.example/report"))
    facts.append(dict(player_ref=3, season=2026, kind="position", value="WR", game_id=None,
                      known_at="2026-09-10T00:00:00Z", effective_at="2026-09-10T00:00:00Z",
                      valid_until="2027-03-01T00:00:00Z", source_url="https://roster.example", source_sha256="y"))
    return facts


def compiled(facts, output):
    games = pd.DataFrame([dict(game_id=g, season=2026, side="home", offset=i * len(REFS),
                               players=len(REFS), kick=k) for i, (g, k) in enumerate(KICKS.items())])
    refs = np.tile(np.arange(1, len(REFS) + 1), len(KICKS))
    output.mkdir()
    compile_arrays(normalize(facts), games, refs, output)
    return {n: np.load(output / f"{n}.npy") for n in ("context", "position", "departed", "quarantined")}


@pytest.mark.parametrize("seed", range(60))
def test_collapsed_captures_compile_to_identical_arrays(tmp_path, seed):
    rng = random.Random(seed)
    conference, others = history(rng), other_sources(rng)
    kept = latest_retrievals(conference)
    assert len(kept) < len(conference)
    full = compiled(others + conference, tmp_path / "full")
    collapsed = compiled(others + kept, tmp_path / "collapsed")
    for name in full:
        np.testing.assert_array_equal(full[name], collapsed[name], err_msg=name)


def test_conflicting_latest_capture_keeps_the_fallback():
    early = pd.Timestamp("2026-09-12T10:00:00Z"); late = early + pd.Timedelta(hours=1)
    facts = [row("Ray Jones", 2, "out", early), row("Ray Jones", 2, "out", late),
             row("Ray Jones", 2, "available", late)]
    assert latest_retrievals(facts) == facts
    agreeing = [row("Ray Jones", 2, "out", early), row("Ray Jones", 2, "out", late)]
    assert latest_retrievals(agreeing) == agreeing[1:]
