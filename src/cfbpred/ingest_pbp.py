"""Stage 1: download, trim, repair and validate the play-by-play corpus.

Three products land in data/parquet/:
  pbp/pbp_{year}.parquet  play level, 2014-2025, allowlisted + repaired
  games.parquet           game level, 2001-2025, AUTHORITATIVE final scores
  lines.parquet           per-book opening and closing prices, 2006-2025

The authoritative score never comes from play-by-play. Deriving it from the
running-score columns disagrees with the schedule on 2.8% of games — truncated
feeds, duplicated plays, and a handful with home and away swapped outright.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.config import (  # noqa: E402
    GAME_SEASONS, LINES_URL, PBP, PBP_KEEP, PBP_SEASONS, PQ, RAW,
    REL_BASE, SCHED_BASE,
)


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


# ------------------------------------------------------------------ timeouts

# The source has NO unique play key. `id_play` is stored as DOUBLE and has
# already lost precision on 18-digit ids, and timeout rows genuinely share both
# game_play_number and id_play with the play they interrupt. So we mint one
# canonical ordering here and every downstream window function uses it.
#
# The ordering is CHRONOLOGICAL — period ascending, game clock descending — not
# game_play_number. That distinction is load-bearing: `game_play_number` is not
# reliably in game order, and in 2025 it scrambles periods badly. Chosen by
# measuring which ordering keeps the score path monotone, since a score can
# never decrease:
#
#   ordering                       impossible   period    clock
#                                  score drops  reversals  reversals
#   game_play_number first             11,620      6,327     43,024
#   period, clock DESC first            4,865          0          0
#
# The residual 4,865 (0.3% of rows) are genuine source defects; games carrying
# them are excluded from expected-points fitting by the clean-game filter.
PLAY_SEQ = """
SELECT *, row_number() OVER (
    PARTITION BY game_id
    ORDER BY COALESCE(period, 0),
             COALESCE(adj_TimeSecsRem, 0) DESC,
             COALESCE(game_play_number, 0),
             COALESCE(half_play_number, 0)
  ) AS play_seq
FROM src
"""

_REPAIR_BODY = """
WITH seq AS (__PLAY_SEQ__), clamped AS (
  SELECT *,
    -- clamp impossible values, then carry the last good value forward within
    -- (game, half, team). Bad rows run 0.3%/season early, 4.6% by 2024.
    CASE WHEN pos_team_timeouts_rem_before BETWEEN 0 AND 3
         THEN CAST(pos_team_timeouts_rem_before AS TINYINT) END AS pto_raw,
    CASE WHEN def_pos_team_timeouts_rem_before BETWEEN 0 AND 3
         THEN CAST(def_pos_team_timeouts_rem_before AS TINYINT) END AS dto_raw
  FROM seq
), filled AS (
  SELECT *,
    COALESCE(
      pto_raw,
      last_value(pto_raw IGNORE NULLS) OVER (
        PARTITION BY game_id, half, pos_team ORDER BY play_seq
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
      3) AS pto_f,
    COALESCE(
      dto_raw,
      last_value(dto_raw IGNORE NULLS) OVER (
        PARTITION BY game_id, half, def_pos_team ORDER BY play_seq
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
      3) AS dto_f
  FROM clamped
), mono AS (
  SELECT *,
    -- timeouts can only decrease within a half; a rise is a data error, so take
    -- the running minimum. 104 of 155,874 transitions in 2019 violate this.
    min(pto_f) OVER (PARTITION BY game_id, half, pos_team
                     ORDER BY play_seq
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS pto,
    min(dto_f) OVER (PARTITION BY game_id, half, def_pos_team
                     ORDER BY play_seq
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS dto
  FROM filled
)
SELECT * EXCLUDE (pto_raw, dto_raw, pto_f, dto_f,
                  pos_team_timeouts_rem_before, def_pos_team_timeouts_rem_before),
       pto AS pos_team_timeouts_rem_before,
       dto AS def_pos_team_timeouts_rem_before
FROM mono
"""

REPAIR_TIMEOUTS = _REPAIR_BODY.replace("__PLAY_SEQ__", PLAY_SEQ)


# The source mixes DOUBLE, BOOLEAN and INTEGER for what are semantically the
# same things — `scoring_play` is DOUBLE while `scoring` is BOOLEAN, `down` is
# DOUBLE. Normalise once here so nothing downstream has to guess.
AS_BOOL = [
    "scoring_play", "scoring", "Under_two", "Goal_To_Go",
    "Goal_To_Go_end", "Under_two_end", "neutral_site",
    "conference_game", "end_of_half", "change_of_pos_team",
    "pos_team_receives_2H_kickoff", "rush", "pass", "sack", "int",
    "fumble_vec", "turnover", "touchdown", "safety", "fg_made", "fg_inds",
    "punt", "kickoff_play", "penalty_flag",
]
AS_TINY = ["period", "half", "down", "down_end", "pos_team_timeouts_rem_before",
           "def_pos_team_timeouts_rem_before", "off_timeouts_rem_before",
           "def_timeouts_rem_before", "week"]
AS_SMALL = ["year", "distance", "yards_to_goal", "yards_gained",
            "distance_end", "yards_to_goal_end", "TimeSecsRem_end",
            "pos_team_score", "def_pos_team_score", "pos_score_diff_start",
            "offense_score", "defense_score", "clock_minutes", "clock_seconds",
            "TimeSecsRem", "adj_TimeSecsRem", "drive_number", "drive_pts"]
AS_INT = ["game_play_number", "half_play_number", "home_team_id",
          "away_team_id", "venue_id"]


def _cast(col: str) -> str:
    q = f'"{col}"'
    if col in AS_BOOL:
        # DOUBLE 0/1 and BOOLEAN both land on BOOLEAN; NULL stays NULL.
        return f"CAST({q} AS BOOLEAN) AS {q}"
    if col in AS_TINY:
        return f"CAST(round({q}) AS TINYINT) AS {q}"
    if col in AS_SMALL:
        return f"CAST(round({q}) AS SMALLINT) AS {q}"
    if col in AS_INT:
        return f"CAST(round({q}) AS INTEGER) AS {q}"
    return q


def ingest_pbp(con: duckdb.DuckDBPyConnection, cache: Path) -> None:
    for year in PBP_SEASONS:
        out = PBP / f"pbp_{year}.parquet"
        if out.exists():
            print(f"  pbp {year}  cached")
            continue
        raw = fetch(f"{REL_BASE}/play_by_play_{year}.parquet",
                    cache / f"play_by_play_{year}.parquet")
        have = {c[0] for c in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{raw}')").fetchall()}
        missing = [c for c in PBP_KEEP if c not in have]
        if missing:
            raise RuntimeError(f"{year}: allowlist columns absent: {missing}")

        sel = ", ".join(_cast(c) for c in PBP_KEEP)
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW src AS
            SELECT {sel} FROM read_parquet('{raw}')
        """)
        # Rename the market columns so they can never be used unlabelled: the
        # CFBD `spread` field is documented as the CLOSING line.
        con.execute(f"""
            COPY (
              SELECT * EXCLUDE (spread, over_under),
                     spread      AS spread_close,
                     over_under  AS over_under_close
              FROM ({REPAIR_TIMEOUTS})
              ORDER BY game_id, play_seq
            ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        n = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
        print(f"  pbp {year}  {n:>7,} rows  {out.stat().st_size / 1e6:6.1f} MB")


def ingest_games(con: duckdb.DuckDBPyConnection, cache: Path) -> None:
    out = PQ / "games.parquet"
    if out.exists():
        print("  games  cached")
        return
    files = []
    for year in GAME_SEASONS:
        files.append(str(fetch(f"{SCHED_BASE}/cfb_schedules_{year}.parquet",
                               cache / f"sched_{year}.parquet")))
    lst = ", ".join(f"'{f}'" for f in files)
    con.execute(f"""
        COPY (
          SELECT game_id, season, week, season_type, start_date,
                 neutral_site, conference_game, venue_id, venue,
                 home_id, home_team, home_conference, home_division, home_points,
                 away_id, away_team, away_conference, away_division, away_points,
                 home_pregame_elo, away_pregame_elo,
                 home_points - away_points AS margin,
                 home_points + away_points AS total
          -- DELIBERATELY DROPPED: home_postgame_elo, away_postgame_elo,
          -- home_post_win_prob, away_post_win_prob, excitement_index,
          -- attendance, highlights, notes. All outcome-derived.
          FROM read_parquet([{lst}], union_by_name=true)
          WHERE home_points IS NOT NULL AND away_points IS NOT NULL
        ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
    print(f"  games  {n:,} scored games 2001-2025")


def ingest_lines(cache: Path) -> None:
    out = PQ / "lines_book.parquet"
    if out.exists():
        print("  lines  cached")
        return
    src = fetch(LINES_URL, cache / "cfb_line_odds.parquet")
    df = pd.read_parquet(src)
    df.to_parquet(out, index=False, compression="zstd")
    print(f"  lines  {len(df):,} rows, {df.game_id.nunique():,} games "
          f"({int(df.season.min())}-{int(df.season.max())})")


def main() -> None:
    cache = RAW / "cfbfastr"
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    print("ingesting...")
    ingest_games(con, cache)
    ingest_lines(cache)
    ingest_pbp(con, cache)
    print("done")


if __name__ == "__main__":
    main()
