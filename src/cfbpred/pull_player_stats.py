"""Pull official per-game player stat lines for every position.

/games/players requires (year, week): ~17 calls per season. The response nests
game -> team -> category -> stat type -> athletes; it is flattened here to one
row per (game, team, category, stat, athlete).

PURITY EXCLUSION: the passing category carries a 'QBR' stat type — that is
ESPN's Total QBR, a model output, not a counting stat. It is dropped at
ingestion so it cannot reach a feature by accident.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.cfbd_client import CFBDClient  # noqa: E402
from cfbpred.config import CFBD  # noqa: E402

SEASONS = range(2002, 2026)
REG_WEEKS = range(1, 17)
BANNED_STAT_TYPES = {"QBR"}          # model outputs masquerading as stats

OUT = CFBD / "player_game_stats.parquet"


def flatten(games: list, season: int, week: int, season_type: str) -> list:
    rows = []
    for g in games or []:
        for t in g.get("teams", []):
            side = t.get("homeAway")
            for cat in t.get("categories", []):
                for ty in cat.get("types", []):
                    if ty["name"] in BANNED_STAT_TYPES:
                        continue
                    for a in ty.get("athletes", []):
                        rows.append((int(g["id"]), season, week, season_type,
                                     side, cat["name"], ty["name"],
                                     a.get("id"), a.get("name"),
                                     a.get("stat")))
    return rows


def main(only_season: int | None = None) -> None:
    """Full pull by default; `python pull_player_stats.py 2026` re-pulls just
    that season and replaces its rows — the weekly in-season refresh."""
    c = CFBDClient()
    c.min_interval_s = 0.6                       # respect the burst limit
    start = c.spent
    all_rows: list = []
    for season in ([only_season] if only_season else SEASONS):
        n_season = 0
        for week in REG_WEEKS:
            try:
                data = c.get("/games/players",
                             {"year": season, "week": week})
            except Exception as exc:
                print(f"  ! {season} wk{week}: {str(exc)[:90]}")
                continue
            rows = flatten(data, season, week, "regular")
            n_season += len(rows)
            all_rows.extend(rows)
        try:
            data = c.get("/games/players", {"year": season, "week": 1,
                                            "seasonType": "postseason"})
            rows = flatten(data, season, 21, "postseason")
            n_season += len(rows)
            all_rows.extend(rows)
        except Exception as exc:
            print(f"  ! {season} post: {str(exc)[:90]}")
        print(f"  {season}: {n_season:,} stat rows "
              f"[{c.spent - start} calls]")

    df = pd.DataFrame(all_rows, columns=[
        "game_id", "season", "week_c", "season_type", "home_away",
        "category", "stat_type", "athlete_id", "athlete", "stat"])
    if only_season and OUT.exists():
        old = pd.read_parquet(OUT)
        df = pd.concat([old[old["season"] != only_season], df],
                       ignore_index=True)
        print(f"  replaced season {only_season} in existing file")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False, compression="zstd")
    print(f"\nwrote {OUT}: {len(df):,} rows, "
          f"{df.game_id.nunique():,} games, "
          f"{OUT.stat().st_size / 1e6:.1f} MB")
    print(f"calls this run: {c.spent - start}")
    print("reconcile:", c.reconcile())


# NOTE (2026-08-27): a classification=fcs backfill was tried and removed —
# the default (year, week) pull already returns every game CFBD has player
# stats for; the classification param only FILTERS that set. FCS-vs-FCS games
# without box scores simply have none at the source.
if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
