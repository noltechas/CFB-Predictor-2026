import pandas as pd

from cfbpred.refresh_scheduler import KickoffSlot, dispatch, due_slots, kickoff_slots, load_state, main, missed_slots, resolve_schedule


def schedule():
    return pd.DataFrame({
        "game_id": [1, 2, 3],
        "kick": ["2026-09-21T18:00:00Z", "2026-09-21T18:00:00Z", "2026-09-22T00:00:00Z"],
    })


def test_shared_kickoff_is_one_slot_and_has_all_games():
    slots = kickoff_slots(schedule(), "2026-09-21T12:00:00Z")
    assert len(slots) == 2
    assert slots[0].game_ids == (1, 2)


def test_dispatch_is_idempotent_and_retries_failures(tmp_path):
    state = tmp_path / "state.json"
    calls = []

    def runner(command):
        calls.append(command)
        return 1 if len(calls) == 1 else 0

    first = dispatch(schedule(), state, "2026-09-21T17:40:00Z", ["refresh"], runner=runner)
    assert first["due"][0]["code"] == 1
    second = dispatch(schedule(), state, "2026-09-21T17:45:00Z", ["refresh"], runner=runner)
    assert second["due"][0]["code"] == 0
    third = dispatch(schedule(), state, "2026-09-21T17:50:00Z", ["refresh"], runner=runner)
    assert third["due"] == []
    assert len(calls) == 2


def test_started_slot_is_never_dispatched_after_kickoff(tmp_path):
    result = dispatch(schedule(), tmp_path / "state.json", "2026-09-21T18:00:00Z", ["refresh"], runner=lambda _: 0)
    assert result["due"] == []


def test_latest_pointer_resolves_snapshot(tmp_path):
    run = tmp_path / "run"; (run / "source").mkdir(parents=True)
    schedule_path = run / "source" / "snapshot_schedule.parquet"
    schedule().to_parquet(schedule_path)
    pointer = tmp_path / "latest.json"; pointer.write_text('{"run_directory": "' + str(run) + '"}')
    assert resolve_schedule(pointer) == schedule_path


def test_tbd_and_missing_kickoff_are_skipped():
    frame = schedule().assign(start_time_tbd=[True, False, False])
    frame.loc[2, "kick"] = None
    assert [slot.game_ids for slot in kickoff_slots(frame, "2026-09-21T12:00:00Z")] == [(2,)]


def test_missed_slot_is_reported_but_never_due():
    slot = KickoffSlot(pd.Timestamp("2026-09-21T18:00:00Z"), (1,))
    state = {"slots": {}, "daily": {}}
    now = "2026-09-21T18:01:00Z"
    assert due_slots([slot], now, state) == []
    assert missed_slots([slot], now, state) == [slot]


def test_timeout_is_recorded_as_failed_and_rescheduled_game_retries(tmp_path):
    import subprocess
    state = tmp_path / "state.json"
    schedule_frame = schedule().iloc[[0]].copy()
    calls = []
    def timeout(_):
        calls.append(1); raise subprocess.TimeoutExpired("refresh", 1)
    first = dispatch(schedule_frame, state, "2026-09-21T17:40:00Z", ["refresh"], runner=timeout)
    assert first["due"][0]["code"] == 124
    changed = pd.DataFrame({"game_id": [1, 9], "kick": ["2026-09-21T18:00:00Z"] * 2})
    second = dispatch(changed, state, "2026-09-21T17:45:00Z", ["refresh"], runner=lambda _: 0)
    assert second["due"][0]["game_ids"] == [1, 9]


def test_clock_recheck_skips_slot_that_reaches_kickoff_while_running(tmp_path):
    ticks = iter(["2026-09-21T17:40:00Z", "2026-09-21T18:00:00Z"])
    calls = []
    result = dispatch(schedule().iloc[[0]], tmp_path / "state.json", command=["refresh"], runner=lambda _: calls.append(1) or 0, clock=lambda: next(ticks))
    assert result["due"] == []
    assert calls == []


def test_cli_requires_refresh_command_unless_dry_run(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        main(["--schedule", str(tmp_path / "missing.parquet"), "--state", str(tmp_path / "state.json")])
