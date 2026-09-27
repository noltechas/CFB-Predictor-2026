"""Stage 2: season-keyed CFBD pull into local Parquet.

Budget discipline: every endpoint here takes `year` alone and returns all teams.
No per-team loops. No per-gameId endpoints. Weekly Elo is the one intentional
per-week pull, because season-level ratings are the classic look-ahead leak.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.cfbd_client import CFBDClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "parquet" / "cfbd"
SEASONS = list(range(2001, 2026))
CORE_SEASONS = list(range(2016, 2026))   # /ratings/core only goes back to 2016
WEEKS = list(range(1, 16))


def save(name: str, rows: list, **extra) -> int:
    if not rows:
        return 0
    df = pd.json_normalize(rows)
    for k, v in extra.items():
        df[k] = v
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.parquet"
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path, index=False)
    return len(df)


def main() -> None:
    c = CFBDClient(budget=int(os.environ.get("CFBD_BUDGET", 30_000)))
    start = c.spent
    print(f"starting; {start} calls already logged")

    # --- one-shot reference tables -------------------------------------
    for name, ep, params in [
        ("venues", "/venues", {}),
        ("conferences", "/conferences", {}),
    ]:
        p = OUT / f"{name}.parquet"
        if p.exists():
            print(f"  skip {name} (exists)")
            continue
        n = save(name, c.get(ep, params))
        print(f"  {name:26s} {n:6d} rows   [{c.spent - start} calls]")

    # --- season-keyed, one call per season ------------------------------
    season_eps = [
        ("games",            "/games",                  lambda y: {"year": y}),
        ("teams",            "/teams",                  lambda y: {"year": y}),
        ("calendar",         "/calendar",               lambda y: {"year": y}),
        ("ratings_sp",       "/ratings/sp",             lambda y: {"year": y}),
        ("ratings_srs",      "/ratings/srs",            lambda y: {"year": y}),
        ("ratings_fpi",      "/ratings/fpi",            lambda y: {"year": y}),
        ("recruiting_teams", "/recruiting/teams",       lambda y: {"year": y}),
        ("returning",        "/player/returning",       lambda y: {"year": y}),
        ("lines",            "/lines",                  lambda y: {"year": y}),
        ("weather",          "/games/weather",          lambda y: {"year": y}),
        ("stats_adv",        "/stats/season/advanced",  lambda y: {"year": y}),
        ("ppa_teams",        "/ppa/teams",              lambda y: {"year": y}),
        ("talent",           "/talent",                 lambda y: {"year": y}),
        ("roster",           "/roster",                 lambda y: {"year": y}),
    ]
    for name, ep, mk in season_eps:
        p = OUT / f"{name}.parquet"
        if p.exists():
            print(f"  skip {name} (exists)")
            continue
        total = 0
        for y in SEASONS:
            try:
                total = save(name, c.get(ep, mk(y)), season_pulled=y) or total
            except Exception as exc:
                print(f"    ! {name} {y}: {str(exc)[:110]}")
        print(f"  {name:26s} {total:6d} rows   [{c.spent - start} calls]")

    # --- CORE: CFBD's new opponent-adjusted rating, 2016+ ---------------
    if not (OUT / "ratings_core.parquet").exists():
        total = 0
        for y in CORE_SEASONS:
            try:
                total = save("ratings_core", c.get("/ratings/core", {"year": y}),
                             season_pulled=y) or total
            except Exception as exc:
                print(f"    ! core {y}: {str(exc)[:110]}")
        print(f"  {'ratings_core':26s} {total:6d} rows   [{c.spent - start} calls]")

    # --- coaches: one call spanning all years ---------------------------
    if not (OUT / "coaches.parquet").exists():
        n = save("coaches", c.get("/coaches", {"minYear": 2001, "maxYear": 2025}))
        print(f"  {'coaches':26s} {n:6d} rows   [{c.spent - start} calls]")

    # --- weekly Elo: the point-in-time rating, ~15 calls/season ---------
    if not (OUT / "elo_weekly.parquet").exists():
        total = 0
        for y in SEASONS:
            for w in WEEKS:
                try:
                    total = save("elo_weekly", c.get("/ratings/elo", {"year": y, "week": w}),
                                 season_pulled=y, week_pulled=w) or total
                except Exception as exc:
                    print(f"    ! elo {y} w{w}: {str(exc)[:90]}")
                    break
        print(f"  {'elo_weekly':26s} {total:6d} rows   [{c.spent - start} calls]")

    print(f"\ntotal calls this run: {c.spent - start}")
    print("reconcile:", c.reconcile())


if __name__ == "__main__":
    main()
