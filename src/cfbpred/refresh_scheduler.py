"""Schedule-aware refresh dispatch without changing the refresh scripts.

The launchd entry point calls this module once per minute.  Kickoff slots are
derived from the current schedule, so rescheduled games are picked up without
editing a plist.  A small JSON state file makes each slot idempotent and keeps
retries bounded to the pre-kickoff window.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class KickoffSlot:
    kickoff: pd.Timestamp
    game_ids: tuple[int, ...]

    @property
    def key(self) -> str:
        return self.kickoff.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def kickoff_slots(schedule: pd.DataFrame, now: object | None = None,
                  horizon: timedelta = timedelta(days=8)) -> list[KickoffSlot]:
    """Return unique future kickoff slots, including games sharing a kickoff."""
    current = _utc(now or datetime.now(timezone.utc))
    frame = schedule.copy()
    kick_col = "kick" if "kick" in frame else "start_date"
    frame["_kick"] = pd.to_datetime(frame[kick_col], utc=True, format="mixed", errors="coerce")
    if "start_time_tbd" in frame:
        frame = frame[~frame["start_time_tbd"].fillna(False)]
    frame = frame.dropna(subset=["_kick"])
    frame = frame[(frame["_kick"] > current) & (frame["_kick"] <= current + horizon)]
    groups: dict[pd.Timestamp, list[int]] = {}
    for _, row in frame.iterrows():
        gid = int(row["game_id"])
        groups.setdefault(row["_kick"], []).append(gid)
    return [KickoffSlot(k, tuple(sorted(ids))) for k, ids in sorted(groups.items())]


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"slots": {}, "daily": {}}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("scheduler state must be an object")
    value.setdefault("slots", {})
    value.setdefault("daily", {})
    return value


def resolve_schedule(path: Path) -> Path:
    """Resolve a parquet path or the production ``refreshes/latest.json`` pointer."""
    if path.suffix.lower() != ".json":
        return path
    pointer = json.loads(path.read_text())
    run = Path(pointer.get("run_directory", ""))
    for candidate in (run / "source/snapshot_schedule.parquet", run / "inputs/snapshot_schedule.parquet"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no snapshot schedule in {path}")


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(state, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def due_slots(slots: Iterable[KickoffSlot], now: object,
              state: dict, lead: timedelta = timedelta(minutes=30)) -> list[KickoffSlot]:
    current = _utc(now)
    due = []
    for slot in slots:
        start = slot.kickoff - lead
        record = state["slots"].get(slot.key, {})
        previous_ids = set(record.get("game_ids", []))
        if start <= current < slot.kickoff and not (
                record.get("status") == "succeeded" and set(slot.game_ids) <= previous_ids):
            due.append(slot)
    return due


def missed_slots(slots: Iterable[KickoffSlot], now: object, state: dict) -> list[KickoffSlot]:
    """Return slots whose pregame window passed without a successful run."""
    current = _utc(now)
    return [slot for slot in slots
            if slot.kickoff <= current and state["slots"].get(slot.key, {}).get("status") != "succeeded"]


def daily_due(now: object, state: dict, local_tz: str = "America/Chicago") -> bool:
    """Whether the daily full refresh has not run for the local calendar date."""
    current = _utc(now).tz_convert(local_tz)
    return state["daily"].get(current.date().isoformat()) != "succeeded"


def run_command(command: Sequence[str], timeout: int = 3600) -> int:
    return subprocess.run(list(command), check=False, timeout=timeout).returncode


def dispatch(schedule: pd.DataFrame, state_path: Path, now: object | None = None,
             command: Sequence[str] = (), lead: timedelta = timedelta(minutes=30),
             runner: Callable[[Sequence[str]], int] = run_command,
             clock: Callable[[], object] | None = None) -> dict:
    """Run each due pregame refresh once; failed slots remain retryable."""
    clock = clock or (lambda: now if now is not None else datetime.now(timezone.utc))
    initial = _utc(now if now is not None else clock())
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"checked_at": initial.isoformat(), "due": [], "missed": [], "locked": True}
        try:
            state = load_state(state_path)
            future = kickoff_slots(schedule, initial)
            missed = missed_slots(
                kickoff_slots(schedule, initial - timedelta(hours=1), horizon=timedelta(hours=1)),
                initial,
                state,
            )
            due = due_slots(future, initial, state, lead)
            results = []
            for slot in due:
                current = _utc(clock())
                if not (slot.kickoff - lead <= current < slot.kickoff):
                    continue
                record = state["slots"].setdefault(slot.key, {"game_ids": list(slot.game_ids), "attempts": 0})
                record["game_ids"] = sorted(set(record.get("game_ids", [])) | set(slot.game_ids))
                record["attempts"] = int(record.get("attempts", 0)) + 1
                record["last_attempt"] = current.isoformat()
                try:
                    code = runner(command) if command else 0
                except subprocess.TimeoutExpired:
                    code = 124
                except Exception:
                    code = 1
                record["status"] = "succeeded" if code == 0 else "failed"
                record["last_code"] = code
                results.append({"slot": slot.key, "game_ids": list(slot.game_ids), "code": code})
                save_state(state_path, state)
            missed_keys = [slot.key for slot in missed if slot.kickoff <= initial]
            return {"checked_at": initial.isoformat(), "due": results, "missed": missed_keys, "locked": False}
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--refresh-command", nargs="+", default=[])
    parser.add_argument("--timeout-minutes", type=int, default=60,
                        help="kill a slot's refresh after this long (a CI runner may need the full nightly)")
    parser.add_argument("--lead-minutes", type=int, default=30,
                        help="start a slot's refresh this long before kickoff (slower CI runners need more)")
    parser.add_argument("--dry-run", action="store_true",
                        help="inspect due/missed slots without running a refresh command")
    args = parser.parse_args(argv)
    if not args.refresh_command and not args.dry_run:
        parser.error("--refresh-command is required unless --dry-run is specified")
    schedule = pd.read_parquet(resolve_schedule(args.schedule))
    timeout = args.timeout_minutes * 60
    result = dispatch(schedule, args.state, command=args.refresh_command,
                      lead=timedelta(minutes=args.lead_minutes),
                      runner=lambda command: run_command(command, timeout=timeout))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
