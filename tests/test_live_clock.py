"""Play-clock cleaning for live curves (live_scorer.clean_clock).

supabase/fixtures/clock_cases.json holds real ESPN stamp sequences with the
answers; the edge scorer's cleanClock is pinned to the same file by
supabase/functions/_shared/clock.test.ts, so the two cannot drift apart.
"""
import copy
import json
from pathlib import Path

import pytest

from cfbpred.live_scorer import clean_clock, t_left_of

CASES = json.loads((Path(__file__).resolve().parents[1]
                    / "supabase" / "fixtures" / "clock_cases.json").read_text())


def row(period, clock, t=None):
    return dict(period=period, half=1 if period <= 2 else 2, clock=clock,
                t_left=t_left_of(period, clock) if t is None else t)


def in_quarter(r):
    if r["period"] >= 5:
        return r["t_left"] == 0
    lo = (4 - r["period"]) * 900
    return lo <= r["t_left"] <= lo + 900


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_real_espn_glitches(case):
    got = clean_clock([row(p, c, t) for p, c, t in case["rows"]],
                      now_last=case["now_last"])
    for r, (p, c, t) in zip(got, case["expect"]):
        assert (r["period"], r["clock"]) == (p, c)
        assert r["t_left"] == pytest.approx(t, abs=1e-6)
    assert all(b["t_left"] <= a["t_left"] for a, b in zip(got, got[1:]))
    assert all(in_quarter(r) for r in got)
    assert all(r["half"] == (1 if r["period"] <= 2 else 2) for r in got)


def test_quarter_opening_zero_stamp_cannot_drag_the_quarter():
    rows = clean_clock([row(1, "0:05"), row(1, "0:00"), row(2, "0:00"),
                        row(2, "15:00"), row(2, "15:00"), row(2, "11:54"),
                        row(2, "10:00"), row(4, "0:00", 0.0)], now_last=True)
    assert 2514 <= rows[2]["t_left"] <= 2700
    assert rows[5]["t_left"] == t_left_of(2, "11:54")
    assert rows[6]["t_left"] == t_left_of(2, "10:00")


def test_lone_bad_latest_stamp_loses_to_the_scoreboard():
    rows = clean_clock([row(1, "0:05"), row(1, "0:00"), row(2, "0:00"),
                        row(2, "14:52")], now_last=True)
    assert rows[3]["t_left"] == t_left_of(2, "14:52")


def test_batch_tail_stays_in_its_quarter():
    rows = clean_clock([row(2, "5:00")] + [row(2, "1:00") for _ in range(40)])
    assert all(in_quarter(r) for r in rows)


def test_trailing_scoreboard_sits_after_last_play():
    rows = clean_clock([row(3, "10:00"), row(3, "9:40"), row(3, "9:30"),
                        row(3, "9:50")], now_last=True)
    assert rows[3]["t_left"] == t_left_of(3, "9:30") - 1


def test_clean_is_idempotent_on_clean_monotone_input():
    rows = [row(1, "15:00"), row(1, "14:10"), row(1, "13:00"), row(1, "12:30")]
    assert clean_clock(copy.deepcopy(rows)) == rows
