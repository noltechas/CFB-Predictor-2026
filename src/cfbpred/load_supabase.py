"""Load reference tables (teams, venues, games) from local Parquet into Supabase.

Only reference data and model outputs go to Postgres. The play-level training
corpus stays in local Parquet — see supabase_schema.sql for the rationale.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parents[2]
PQ = ROOT / "data" / "parquet" / "cfbd"


# Supavisor session mode. NOT transaction mode (6543): publish.py runs DDL
# and COPY, which need a real session.
POOLER_PORT = 5432

# TCP keepalives. A half-open connection — laptop sleeps, Wi-Fi flaps, the
# pooler drops us — otherwise leaves a reader blocked FOREVER on a socket the
# kernel still reports as ESTABLISHED. Measured 2026-08-30: publish_lines.py
# sat in exactly that state for 11h21m while Postgres showed no session at
# all, having already dropped game_lines and (by cascade) the Results view, so
# the dashboard 404'd all morning and the 30-minute agent never ran again.
# connect_timeout only bounds the handshake; these bound the reads.
KEEPALIVE = ("keepalives=1&keepalives_idle=30&keepalives_interval=10"
             "&keepalives_count=3")


def dsn() -> str:
    """Connection string for the Supabase Postgres.

    Prefers the Supavisor POOLER over the direct db.<ref>.supabase.co host.
    Direct connections are IPv6-ONLY — Supabase dropped their IPv4 addresses —
    so on any network without IPv6 the direct host resolves to an AAAA record
    this machine cannot route to, and every job that touches Postgres wedges
    in a reconnect loop. Measured 2026-08-29: the live scorer sat in exactly
    that loop for 3h20m through a game day while the REST API (Cloudflare,
    IPv4) kept serving the dashboard, so the site looked healthy and only the
    live reads went stale. The pooler publishes A records and works on both
    stacks; set SUPABASE_DB_POOLER_HOST to use it.
    """
    pw = os.environ["SUPABASE_DB_PASSWORD"]
    host = os.environ["SUPABASE_DB_HOST"]
    pooler = os.environ.get("SUPABASE_DB_POOLER_HOST", "").strip()
    if pooler:
        # the pooler multiplexes tenants, so the user carries the project ref
        ref = host.split(".")[1] if host.startswith("db.") else host.split(".")[0]
        return (f"postgresql://postgres.{ref}:{pw}@{pooler}:{POOLER_PORT}"
                f"/postgres?sslmode=require&connect_timeout=30&{KEEPALIVE}")
    return (f"postgresql://postgres:{pw}@{host}:5432/postgres"
            f"?sslmode=require&connect_timeout=30&{KEEPALIVE}")


INT_TYPES = {"smallint", "integer", "bigint"}
BOOL_TYPES = {"boolean"}


def copy_df(cur, df: pd.DataFrame, table: str, cols: list[str]) -> int:
    """COPY a frame in, casting each column to match the target's Postgres type.

    pandas renders a nullable int column as '4000.0', which COPY rejects for an
    integer target, so the cast has to be driven by the table definition.
    """
    schema, name = table.split(".", 1)
    cur.execute(
        """select column_name, data_type from information_schema.columns
           where table_schema = %s and table_name = %s""", (schema, name))
    pgtype = dict(cur.fetchall())

    df = df[cols].copy()
    for c in cols:
        t = pgtype.get(c, "")
        if t in INT_TYPES:
            df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")
        elif t in BOOL_TYPES:
            df[c] = df[c].map(
                lambda x: None if pd.isna(x) else bool(x)).astype("boolean")

    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="")
    cur.execute(f"truncate {table} cascade")
    with cur.copy(f"copy {table} ({','.join(cols)}) from stdin with (format csv, null '')") as cp:
        cp.write(buf.getvalue())
    return len(df)


def first_col(df: pd.DataFrame, *names, default=None):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([default] * len(df), index=df.index)


def main() -> None:
    with psycopg.connect(dsn(), autocommit=True) as cx, cx.cursor() as cur:

        # ---- venues ----
        v = pd.read_parquet(PQ / "venues.parquet").drop_duplicates("id")
        v = pd.DataFrame({
            "venue_id": v["id"], "name": first_col(v, "name"),
            "city": first_col(v, "city"), "state": first_col(v, "state"),
            "timezone": first_col(v, "timezone"),
            "latitude": first_col(v, "latitude"), "longitude": first_col(v, "longitude"),
            "elevation": pd.to_numeric(first_col(v, "elevation"), errors="coerce"),
            "capacity": first_col(v, "capacity"),
            "dome": first_col(v, "dome"), "grass": first_col(v, "grass"),
        })
        n = copy_df(cur, v, "cfb.venues", list(v.columns))
        print(f"  venues  {n:6d}")

        # ---- teams (latest season per team) ----
        t = pd.read_parquet(PQ / "teams.parquet")
        t = t.sort_values("season_pulled").drop_duplicates("id", keep="last")
        t = pd.DataFrame({
            "team_id": t["id"], "school": first_col(t, "school"),
            "mascot": first_col(t, "mascot"), "abbreviation": first_col(t, "abbreviation"),
            "conference": first_col(t, "conference"),
            "classification": first_col(t, "classification", "division"),
            "color": first_col(t, "color"), "alt_color": first_col(t, "alternateColor"),
            "logo": first_col(t, "logos").apply(
                lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) else None),
        }).dropna(subset=["school"])
        n = copy_df(cur, t, "cfb.teams", list(t.columns))
        print(f"  teams   {n:6d}")

        # ---- games ----
        g = pd.read_parquet(PQ / "games.parquet").drop_duplicates("id")
        known = set(v["venue_id"].dropna().astype("Int64"))
        vid = pd.to_numeric(first_col(g, "venueId"), errors="coerce").astype("Int64")
        g = pd.DataFrame({
            "game_id": g["id"], "season": g["season"], "week": first_col(g, "week"),
            "season_type": first_col(g, "seasonType"),
            "start_date": first_col(g, "startDate"),
            "neutral_site": first_col(g, "neutralSite"),
            "conference_game": first_col(g, "conferenceGame"),
            "venue_id": vid.where(vid.isin(known)),
            "home_team_id": first_col(g, "homeId", "homeTeamId"),
            "home_team": first_col(g, "homeTeam"),
            "home_conference": first_col(g, "homeConference"),
            "home_classification": first_col(g, "homeClassification"),
            "home_points": first_col(g, "homePoints"),
            "away_team_id": first_col(g, "awayId", "awayTeamId"),
            "away_team": first_col(g, "awayTeam"),
            "away_conference": first_col(g, "awayConference"),
            "away_classification": first_col(g, "awayClassification"),
            "away_points": first_col(g, "awayPoints"),
        })
        # NOTE: postgame elo / postgame win prob / excitement index are
        # deliberately NOT loaded. They are outcome-derived and are the
        # classic leakage columns in this dataset.
        n = copy_df(cur, g, "cfb.games", list(g.columns))
        print(f"  games   {n:6d}")

        for tbl in ["cfb.venues", "cfb.teams", "cfb.games"]:
            cur.execute(f"select count(*), pg_size_pretty(pg_total_relation_size('{tbl}')) from {tbl}")
            c, sz = cur.fetchone()
            print(f"  {tbl:18s} {c:7d} rows  {sz}")


if __name__ == "__main__":
    main()
