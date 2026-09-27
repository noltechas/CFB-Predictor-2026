"""Official per-game player stat lines for the FCS slate.

Coverage the main pull (classification=fbs, the CFBD default) never sees:
  2021+      CFBD /games/players?classification=fcs — the full FCS slate,
             identical response shape, ~17 calls per season.
  2014-2020  ESPN's per-game summary endpoint. Same game ids, same athlete
             ids, and its box-score labels match CFBD's stat_type names
             (C/ATT, YDS, TD, INT, TOT, FG, XP, 'In 20', ...) so rows map
             one-to-one. KNOWN GAP: ESPN FCS box scores carry no 'defensive'
             category, so tackles/TFL/sacks stay absent for those seasons —
             offense, kicking, returns and turnovers are complete.

Output: data/parquet/cfbd/player_game_stats_fcs.parquet, the exact schema of
player_game_stats.parquet, holding ONLY game_ids absent from the main file —
the two concatenate without overlap. Nothing reads it unless CFB_FCS_STATS=1
(see player_features.load_player_stats): the frozen production model's
feature store stays byte-stable until a retrain measures these in.

Purity: box-score counting stats only. ESPN's RTG/QBR-style ratings and
CFBD's QBR never land in a row.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.cfbd_client import CFBDClient  # noqa: E402
from cfbpred.config import CFBD, PQ  # noqa: E402
from cfbpred.pull_player_stats import BANNED_STAT_TYPES, flatten  # noqa: E402

OUT = CFBD / "player_game_stats_fcs.parquet"
CFBD_FCS_SINCE = 2021          # first season CFBD carries the full FCS slate
ESPN_SEASONS = range(2014, 2021)
ESPN_SUMMARY = ("https://site.api.espn.com/apis/site/v2/sports/football/"
                "college-football/summary?event={gid}")
# category -> labels worth keeping (everything team_game_stats consumes)
ESPN_KEEP = {
    "passing": {"C/ATT", "YDS", "TD", "INT", "AVG"},
    "rushing": {"CAR", "YDS", "TD", "AVG", "LONG"},
    "receiving": {"REC", "YDS", "TD", "AVG", "LONG"},
    "defensive": {"TOT", "SOLO", "SACKS", "TFL", "PD", "QB HUR", "TD"},
    "interceptions": {"INT", "YDS", "TD"},
    "fumbles": {"FUM", "LOST", "REC"},
    "kicking": {"FG", "XP", "PTS", "LONG"},
    "punting": {"NO", "YDS", "AVG", "TB", "In 20", "LONG"},
    "kickReturns": {"NO", "YDS", "AVG", "TD", "LONG"},
    "puntReturns": {"NO", "YDS", "AVG", "TD", "LONG"},
}


def espn_rows(s: requests.Session, gid: int, season: int, week_c: int,
              season_type: str) -> list:
    r = s.get(ESPN_SUMMARY.format(gid=gid), timeout=30)
    if r.status_code != 200:
        return []
    d = r.json()
    try:
        side_of = {c["team"]["id"]: c["homeAway"]
                   for c in d["header"]["competitions"][0]["competitors"]}
        blocks = d["boxscore"]["players"]
    except (KeyError, IndexError, TypeError):
        return []
    rows = []
    for blk in blocks:
        side = side_of.get(blk.get("team", {}).get("id"))
        if side is None:
            continue
        for grp in blk.get("statistics", []):
            cat = grp.get("name")
            keep = ESPN_KEEP.get(cat)
            if not keep:
                continue
            labels = grp.get("labels") or []
            for a in grp.get("athletes", []):
                aid = (a.get("athlete") or {}).get("id")
                name = (a.get("athlete") or {}).get("displayName")
                for lab, val in zip(labels, a.get("stats") or []):
                    if lab in keep and lab not in BANNED_STAT_TYPES:
                        rows.append((gid, season, week_c, season_type,
                                     side, cat, lab, aid, name, val))
    return rows


def main() -> None:
    have = set(pd.read_parquet(CFBD / "player_game_stats.parquet",
                               columns=["game_id"]).game_id.unique())
    all_rows: list = []

    # ---- CFBD, full FCS slate 2021+ ----
    c = CFBDClient()
    c.min_interval_s = 0.6
    start = c.spent
    seasons = range(CFBD_FCS_SINCE, 2026)
    for season in seasons:
        n_season = 0
        for week in range(1, 17):
            try:
                data = c.get("/games/players",
                             {"year": season, "week": week,
                              "classification": "fcs"})
            except Exception as exc:
                print(f"  ! {season} wk{week}: {str(exc)[:90]}")
                continue
            rows = [r for r in flatten(data, season, week, "regular")
                    if r[0] not in have]
            n_season += len(rows)
            all_rows.extend(rows)
        try:
            data = c.get("/games/players",
                         {"year": season, "week": 1,
                          "seasonType": "postseason",
                          "classification": "fcs"})
            rows = [r for r in flatten(data, season, 21, "postseason")
                    if r[0] not in have]
            n_season += len(rows)
            all_rows.extend(rows)
        except Exception as exc:
            print(f"  ! {season} post: {str(exc)[:90]}")
        print(f"  cfbd {season}: {n_season:,} rows [{c.spent - start} calls]")

    # ---- ESPN, per game 2014-2020 ----
    got = {r[0] for r in all_rows}
    g = pd.read_parquet(PQ / "games.parquet",
                        columns=["game_id", "season", "week", "season_type",
                                 "home_division", "away_division", "margin"])
    g = g[(g.home_division == "fcs") & (g.away_division == "fcs")
          & g.margin.notna() & g.season.isin(ESPN_SEASONS)]
    # keep requests' default User-Agent: ESPN's WAF 403s custom ones
    s = requests.Session()
    s.headers.update({"Accept": "application/json"})
    for season, gs in g.groupby("season"):
        n_season, n_empty = 0, 0
        for r in gs.itertuples():
            if r.game_id in have or r.game_id in got:
                continue
            wc = int(r.week) + (20 if r.season_type == "postseason" else 0)
            try:
                rows = espn_rows(s, int(r.game_id), int(season), wc,
                                 r.season_type)
            except requests.RequestException:
                rows = []
            time.sleep(0.15)
            if rows:
                all_rows.extend(rows)
                n_season += len(rows)
            else:
                n_empty += 1
        print(f"  espn {season}: {n_season:,} rows "
              f"({n_empty} games without a boxscore)")

    df = pd.DataFrame(all_rows, columns=[
        "game_id", "season", "week_c", "season_type", "home_away",
        "category", "stat_type", "athlete_id", "athlete", "stat"])
    df.to_parquet(OUT, index=False, compression="zstd")
    print(f"\nwrote {OUT}: {len(df):,} rows, {df.game_id.nunique():,} games")


if __name__ == "__main__":
    main()
