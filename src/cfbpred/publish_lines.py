"""Publish Vegas closing lines to Supabase — DISPLAY ONLY, never a model input.

The dashboard shows the market's number next to the model's so the two can be
compared game by game. The purity rule stands: betting lines are banned from
the feature store (BANNED_SUBSTRINGS, tests/test_leakage.py), and the training
cache (source.py TABLES) does not include this table. Lines already appear in
one other place with the same status: evaluate.py, as the external benchmark.

Source: data/parquet/cfbd/lines.parquet (CFBD /lines, pulled in stage 2).
One row per game, one book per row, chosen by provider priority — consensus
where it exists (2001-2022 era), then the majors.

Conventions (CFBD): `spread` is the HOME handicap — negative means the home
team is favored; the market's expected home margin is -spread. Moneylines are
American odds; where both sides exist we store them and the dashboard derives
the de-vigged implied home win probability.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.config import CFBD, PQ  # noqa: E402
from cfbpred.load_supabase import dsn  # noqa: E402
from cfbpred.publish import copy_chunked, grant  # noqa: E402

PRIORITY = ["consensus", "ESPN Bet", "DraftKings", "Draft Kings", "Bovada",
            "William Hill (New Jersey)", "Caesars",
            "Caesars Sportsbook (Colorado)", "Caesars (Pennsylvania)",
            "SugarHouse", "teamrankings", "numberfire"]


def best_lines() -> pd.DataFrame:
    df = pd.read_parquet(CFBD / "lines.parquet")
    ex = df.explode("lines").dropna(subset=["lines"])
    L = pd.json_normalize(ex["lines"])
    L["game_id"] = ex["id"].values
    L = L.dropna(subset=["spread"])

    rank = {p: i for i, p in enumerate(PRIORITY)}
    L["rank"] = L["provider"].map(rank).fillna(len(PRIORITY))
    # among equal-priority rows prefer the one carrying a total and moneylines
    L["richness"] = L["overUnder"].notna().astype(int) + \
        (L["homeMoneyline"].notna() & L["awayMoneyline"].notna()).astype(int)
    L = (L.sort_values(["game_id", "rank", "richness"],
                       ascending=[True, True, False])
           .drop_duplicates("game_id"))

    out = pd.DataFrame({
        "game_id": L["game_id"].astype("int64"),
        "provider": L["provider"],
        "spread": L["spread"],
        "spread_open": L["spreadOpen"],
        "over_under": L["overUnder"],
        "over_under_open": L["overUnderOpen"],
        # nullable Int64 so COPY sees '150' / '', never '150.0'
        "home_moneyline": pd.to_numeric(
            L["homeMoneyline"], errors="coerce").round().astype("Int64"),
        "away_moneyline": pd.to_numeric(
            L["awayMoneyline"], errors="coerce").round().astype("Int64"),
    })
    # ESPN lines fill games CFBD's feed doesn't carry (the FCS slate) —
    # display-only like everything here, and never overriding a CFBD line
    espn = PQ / "espn_lines.parquet"
    if espn.exists():
        e = pd.read_parquet(espn)
        if not e.empty:
            e = e[~e["game_id"].isin(out["game_id"])].copy()
        if not e.empty:
            e["game_id"] = e["game_id"].astype("int64")
            e["spread_open"] = pd.NA
            e["over_under_open"] = pd.NA
            for c in ("home_moneyline", "away_moneyline"):
                e[c] = pd.to_numeric(e[c], errors="coerce").round().astype("Int64")
            out = pd.concat([out, e[out.columns]], ignore_index=True)
            print(f"  + {len(e):,} ESPN-only lines (FCS slate)")
    return out


RESULTS_VIEW_SQL = Path(__file__).with_name("sql") / "model_results_view.sql"

def create_results_view(cur) -> None:
    """Replace only the aggregate view; source tables stay untouched."""
    cur.execute(RESULTS_VIEW_SQL.read_text())
    cur.execute("grant select on public.v_model_vs_vegas to anon, authenticated")

def main() -> None:
    gl = best_lines()
    # ONE transaction, NOT autocommit. Postgres DDL is transactional, so the
    # drop-and-rebuild below is invisible to everyone else until it commits:
    # readers see the old game_lines and the old view, then the new ones, and
    # never the gap between. Under autocommit the DROP ... CASCADE published
    # itself immediately and took v_model_vs_vegas with it, so any reader
    # during the rebuild got a 404 on the Results page — and if the run then
    # died (see the 11h socket hang on 2026-08-30) the view stayed gone.
    with psycopg.connect(dsn()) as cx, cx.cursor() as cur:
        cur.execute("drop table if exists public.game_lines cascade")
        cur.execute("""create table public.game_lines (
            game_id bigint primary key,
            provider text,
            spread real,
            spread_open real,
            over_under real,
            over_under_open real,
            home_moneyline integer,
            away_moneyline integer)""")
        cur.execute("""comment on table public.game_lines is
            'Vegas closing lines, dashboard display + benchmark only. '
            'Never joined into training data (purity rule).'""")
        n = copy_chunked(cur, gl, "public.game_lines", list(gl.columns))
        grant(cur, "game_lines")

        # Aggregate scoreboard for the dashboard's Results page: one row per
        # (season, split), model and market judged on the same games.
        create_results_view(cur)

        cur.execute("notify pgrst, 'reload schema'")
        cur.execute("select pg_size_pretty(pg_total_relation_size('public.game_lines'))")
        print(f"  game_lines  {n:,} rows  {cur.fetchone()[0]}")


if __name__ == "__main__":
    main()
