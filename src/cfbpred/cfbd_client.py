"""CollegeFootballData API v2 client with mandatory per-call budget logging.

Every request is appended to logs/api_calls.csv before the response is returned,
so the budget is auditable independently of this process. Reconcile the local
tally against the server with `CFBDClient.quota()`.

The budget is MONTHLY, like the server's (Tier 2: 30,000 calls, reset at 00:00
UTC on the 1st). Counting the whole log instead made it a lifetime cap, which
silently stopped every refresh on 2026-10-07 with 21k calls left on the server.
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

BASE = "https://api.collegefootballdata.com"
ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "logs" / "api_calls.csv"
LOG_FIELDS = ["ts", "endpoint", "params", "status", "rows", "bytes", "elapsed_ms", "note"]


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class CFBDClient:
    api_key: str | None = None
    budget: int = 30_000  # per calendar month (UTC), matching the server quota
    min_interval_s: float = 0.12  # be polite; no documented per-minute throttle
    _spent: int = 0

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("CFBD_API_KEY")
        if not self.api_key:
            raise RuntimeError("CFBD_API_KEY not set")
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        )
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not LOG_PATH.exists():
            with LOG_PATH.open("w", newline="") as fh:
                csv.DictWriter(fh, LOG_FIELDS).writeheader()
        self._spent = self._count_logged()
        self._last_call = 0.0

    # ---------- budget bookkeeping ----------

    def _count_logged(self) -> int:
        """Calls logged since the server's quota last reset (the 1st, 00:00 UTC).

        Log stamps are naive local time, so the reset instant is converted to
        the same local, ISO-sortable form and compared as strings.
        """
        now = datetime.now(timezone.utc)
        reset = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        since = reset.astimezone().strftime("%Y-%m-%dT%H:%M:%S")
        with LOG_PATH.open() as fh:
            return sum(1 for r in csv.DictReader(fh) if r["ts"] >= since)

    @property
    def spent(self) -> int:
        return self._spent

    def _log(self, endpoint: str, params: dict, status: int, rows: int,
             nbytes: int, elapsed_ms: int, note: str = "") -> None:
        with LOG_PATH.open("a", newline="") as fh:
            csv.DictWriter(fh, LOG_FIELDS).writerow({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "endpoint": endpoint,
                "params": urlencode(sorted(params.items())) if params else "",
                "status": status,
                "rows": rows,
                "bytes": nbytes,
                "elapsed_ms": elapsed_ms,
                "note": note,
            })

    # ---------- core ----------

    def get(self, endpoint: str, params: dict | None = None,
            retries: int = 4) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        if self._spent >= self.budget:
            raise BudgetExceeded(f"local budget {self.budget} exhausted")

        gap = time.monotonic() - self._last_call
        if gap < self.min_interval_s:
            time.sleep(self.min_interval_s - gap)

        url = f"{BASE}{endpoint}"
        backoff = 1.5
        for attempt in range(retries):
            t0 = time.monotonic()
            try:
                r = self.session.get(url, params=params, timeout=90)
            except requests.RequestException as exc:
                if attempt == retries - 1:
                    self._spent += 1
                    self._log(endpoint, params, -1, 0, 0, 0, f"network: {exc}")
                    raise
                time.sleep(backoff ** attempt)
                continue

            elapsed = int((time.monotonic() - t0) * 1000)
            self._last_call = time.monotonic()
            # Every request that reached the server counts against quota.
            self._spent += 1

            if r.status_code == 200:
                try:
                    data = r.json()
                except json.JSONDecodeError:
                    self._log(endpoint, params, 200, 0, len(r.content), elapsed, "bad json")
                    raise
                n = len(data) if isinstance(data, list) else 1
                self._log(endpoint, params, 200, n, len(r.content), elapsed)
                return data

            if r.status_code == 429 and attempt < retries - 1:
                # CFBD enforces an undocumented short-window burst limit. Its own
                # 429 body says so, and notes it is separate from the monthly
                # quota. The window is on the order of a minute, so a 2-5s
                # backoff just burns retries — wait it out properly.
                wait = float(r.headers.get("Retry-After") or 0) or 30.0 * (attempt + 1)
                self._log(endpoint, params, 429, 0, len(r.content), elapsed,
                          f"rate-limited, sleeping {wait:.0f}s (attempt {attempt + 1})")
                time.sleep(wait)
                continue

            if r.status_code in (500, 502, 503, 504) and attempt < retries - 1:
                self._log(endpoint, params, r.status_code, 0, len(r.content), elapsed,
                          f"retry {attempt + 1}")
                time.sleep(backoff ** attempt * 2)
                continue

            self._log(endpoint, params, r.status_code, 0, len(r.content), elapsed,
                      r.text[:180].replace("\n", " "))
            r.raise_for_status()

        raise RuntimeError(f"{endpoint} failed after {retries} attempts")

    def quota(self) -> dict:
        """Server-side truth. Counts as one call."""
        return self.get("/info")

    def reconcile(self) -> dict:
        q = self.quota()
        return {
            "server_used": q.get("usedCalls"),
            "server_remaining": q.get("remainingCalls"),
            "local_logged": self._spent,
            "drift": (q.get("usedCalls") or 0) - self._spent,
            "resets_at": q.get("resetAt"),
        }
