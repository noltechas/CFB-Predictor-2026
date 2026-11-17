"""In-season data refresh: pull the current season from CFBD and slot it in.

What it updates, idempotently (re-run any time; each pull replaces that
season's rows rather than appending duplicates):

  data/parquet/cfbd/{games,lines,roster,talent,recruiting_teams,teams,
                     calendar}.parquet      season rows replaced
  data/parquet/cfbd/coaches.parquet         full re-pull (career records need
                                            every prior season in one row)
  data/parquet/schedule.parquet             the season's schedule in the
                                            games.parquet column layout —
                                            unplayed games carry null scores.
                                            games.parquet itself is NOT
                                            touched: it stays the
                                            completed-games corpus the ingest
                                            gates pin.
  Supabase public.games                     season schedule upserted (scores
                                            fill in as games finish)
  Supabase public.teams                     new teams inserted, conference /
                                            classification refreshed; logo
                                            columns are never overwritten

The purity rule stands: lines land in the display-only lines.parquet ->
game_lines path, never the feature store. Elo/attendance/postgame columns in
the raw CFBD payload never reach schedule.parquet.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.cfbd_client import CFBDClient  # noqa: E402
from cfbpred.config import CFBD, CURRENT_SEASON, PQ  # noqa: E402
from cfbpred.load_supabase import dsn  # noqa: E402

SCHEDULE = PQ / "schedule.parquet"


def replace_season(name: str, rows: list, year: int, season_col: str) -> int:
    """Drop the season's old rows from a cfbd parquet, append the new pull."""
    path = CFBD / f"{name}.parquet"
    new = pd.json_normalize(rows)
    new["season_pulled"] = year
    if path.exists():
        old = pd.read_parquet(path)
        keep = old[pd.to_numeric(old[season_col], errors="coerce") != year].copy()
        # parquet round-trips nested lists (e.g. lines.lines) as ndarrays;
        # normalise back to lists so concat + rewrite infers one arrow type
        for col in keep.columns:
            if keep[col].dtype == object:
                keep[col] = keep[col].map(
                    lambda v: v.tolist() if isinstance(v, np.ndarray) else v)
        new = pd.concat([keep, new], ignore_index=True)
        n_repl = len(old) - len(keep)
        print(f"  {name:18s} {len(new):>7,} rows  (replaced {n_repl:,} for {year})")
    else:
        print(f"  {name:18s} {len(new):>7,} rows  (new file)")
    new.to_parquet(path, index=False)
    return len(new)


def schedule_frame(rows: list, year: int) -> pd.DataFrame:
    """CFBD /games -> the games.parquet column layout. Outcome-derived fields
    (postgame elo, win prob, excitement, attendance) are dropped here and the
    market never appears at all."""
    g = pd.json_normalize(rows)
    hp = pd.to_numeric(g["homePoints"], errors="coerce")
    ap = pd.to_numeric(g["awayPoints"], errors="coerce")
    return pd.DataFrame({
        "game_id": g["id"].astype("int64"),
        "season": year,
        "week": pd.to_numeric(g["week"], errors="coerce"),
        "season_type": g["seasonType"],
        "start_date": g["startDate"],
        "start_time_tbd": g.get("startTimeTBD"),
        "neutral_site": g["neutralSite"],
        "conference_game": g["conferenceGame"],
        "venue_id": pd.to_numeric(g.get("venueId"), errors="coerce"),
        "venue": g.get("venue"),
        "home_id": pd.to_numeric(g.get("homeId"), errors="coerce"),
        "home_team": g["homeTeam"],
        "home_conference": g.get("homeConference"),
        "home_division": g.get("homeClassification"),
        "home_points": hp,
        "away_id": pd.to_numeric(g.get("awayId"), errors="coerce"),
        "away_team": g["awayTeam"],
        "away_conference": g.get("awayConference"),
        "away_division": g.get("awayClassification"),
        "away_points": ap,
        "home_pregame_elo": np.nan,      # never pulled; column kept for layout
        "away_pregame_elo": np.nan,
        "margin": hp - ap,
        "total": hp + ap,
    })


def upsert_supabase(sched: pd.DataFrame, teams_rows: list) -> None:
    import psycopg
    with psycopg.connect(dsn(), autocommit=True) as cx, cx.cursor() as cur:
        # venue FK: only ids the venues table actually has
        cur.execute("select venue_id from public.venues")
        known_venues = {r[0] for r in cur.fetchall()}

        # ONE pipelined batch, not one round-trip per row. The old loop issued
        # 3,676 sequential executes against us-east-1: minutes of wall clock,
        # and a multi-minute session is a wide window for a transient drop to
        # kill the run — which it did repeatedly on 2026-08-29 ("server closed
        # the connection unexpectedly"), always partway through, never on the
        # same statement twice. psycopg3 pipelines executemany, so this is the
        # same SQL and the same semantics in a fraction of the exposure.
        params = []
        for r in sched.itertuples():
            vid = int(r.venue_id) if pd.notna(r.venue_id) and int(r.venue_id) in known_venues else None
            params.append((int(r.game_id), int(r.season),
                  int(r.week) if pd.notna(r.week) else None,
                  r.season_type, r.start_date,
                  bool(r.start_time_tbd) if pd.notna(r.start_time_tbd) else None,
                  bool(r.neutral_site) if pd.notna(r.neutral_site) else None,
                  bool(r.conference_game) if pd.notna(r.conference_game) else None,
                  vid,
                  int(r.home_id) if pd.notna(r.home_id) else None,
                  r.home_team, r.home_conference, r.home_division,
                  int(r.home_points) if pd.notna(r.home_points) else None,
                  int(r.away_id) if pd.notna(r.away_id) else None,
                  r.away_team, r.away_conference, r.away_division,
                  int(r.away_points) if pd.notna(r.away_points) else None))
        cur.executemany("""
                insert into public.games (game_id, season, week, season_type,
                  start_date, start_time_tbd, neutral_site, conference_game,
                  venue_id, home_team_id, home_team, home_conference,
                  home_classification, home_points, away_team_id, away_team,
                  away_conference, away_classification, away_points)
                values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                on conflict (game_id) do update set
                  week = excluded.week, season_type = excluded.season_type,
                  start_date = excluded.start_date,
                  start_time_tbd = excluded.start_time_tbd,
                  neutral_site = excluded.neutral_site,
                  conference_game = excluded.conference_game,
                  venue_id = excluded.venue_id,
                  home_conference = excluded.home_conference,
                  home_classification = excluded.home_classification,
                  home_points = excluded.home_points,
                  away_conference = excluded.away_conference,
                  away_classification = excluded.away_classification,
                  away_points = excluded.away_points
            """, params)
        print(f"  public.games       {len(params):,} rows upserted")

        t = pd.json_normalize(teams_rows)
        logo0 = t.get("logos")
        tparams = []
        for i, r in t.iterrows():
            if pd.isna(r.get("school")) or pd.isna(r.get("id")):
                continue
            lg = None
            v = logo0.iloc[i] if logo0 is not None else None
            if isinstance(v, (list, tuple, np.ndarray)) and len(v):
                lg = v[0]
            tparams.append((int(r["id"]), r.get("school"), r.get("mascot"),
                  r.get("abbreviation"), r.get("conference"),
                  r.get("classification"), r.get("color"),
                  r.get("alternateColor"), lg))
        cur.executemany("""
                insert into public.teams (team_id, school, mascot, abbreviation,
                  conference, classification, color, alt_color, logo)
                values (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                on conflict (team_id) do update set
                  school = excluded.school, mascot = excluded.mascot,
                  abbreviation = excluded.abbreviation,
                  conference = excluded.conference,
                  classification = excluded.classification,
                  color = excluded.color, alt_color = excluded.alt_color,
                  logo = coalesce(public.teams.logo, excluded.logo)
            """, tparams)
        print(f"  public.teams       {len(tparams):,} rows upserted")


def main(year: int = CURRENT_SEASON) -> None:
    c = CFBDClient(budget=int(os.environ.get("CFBD_BUDGET", 30_000)))
    start = c.spent
    print(f"season refresh {year}; {start} CFBD calls logged this month")

    games = c.get("/games", {"year": year})
    lines = c.get("/lines", {"year": year})
    roster = c.get("/roster", {"year": year})
    talent = c.get("/talent", {"year": year})
    recruiting = c.get("/recruiting/teams", {"year": year})
    teams = c.get("/teams", {"year": year})
    calendar = c.get("/calendar", {"year": year})
    coaches = c.get("/coaches", {"minYear": 2001, "maxYear": year})

    replace_season("games", games, year, "season")
    replace_season("lines", lines, year, "season")
    replace_season("roster", roster, year, "season_pulled")
    if talent:
        replace_season("talent", talent, year, "year")
    else:
        print(f"  talent             not yet published for {year} (has_talent "
              f"flags cover it)")
    replace_season("recruiting_teams", recruiting, year, "year")
    replace_season("teams", teams, year, "season_pulled")
    replace_season("calendar", calendar, year, "season_pulled")
    pd.json_normalize(coaches).to_parquet(CFBD / "coaches.parquet", index=False)
    print(f"  coaches            {len(coaches):>7,} rows  (full re-pull)")

    sched = schedule_frame(games, year)
    sched.to_parquet(SCHEDULE, index=False, compression="zstd")
    played = int(sched["home_points"].notna().sum())
    print(f"  schedule.parquet   {len(sched):>7,} games, {played:,} completed")

    upsert_supabase(sched, teams)
    print(f"\ncalls this run: {c.spent - start}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else CURRENT_SEASON)
