"""Where training data comes from.

Supabase is the system of record. Training pulls from it, then caches locally as
Parquet and trains from the cache.

That last part matters and is not a hedge. A full pull of the 1.57M-row play
table takes ~14 s over the wire against ~1.5 s from local Parquet, which is
irrelevant once per run and ruinous once per epoch — and training rescans the
corpus dozens of times. So: Postgres is the source of truth, the local cache is
a materialised read of it, and `--refresh` re-pulls whenever the database moves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.config import FEAT  # noqa: E402

CACHE = FEAT / "_supabase_cache"
TABLES = {"game_scalars": "game_id",
          "game_history": "game_id, team, slot",
          "play_raw": "game_id, play_index"}


def save_cache(df, path):
    temporary = path.with_suffix(".tmp.parquet")
    df.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _dsn() -> str:
    """One definition of how we reach Postgres — see load_supabase.dsn for
    why the pooler is preferred (the direct host is IPv6-only)."""
    from cfbpred.load_supabase import dsn
    return dsn()


def pull(table: str, order: str) -> pd.DataFrame:
    import psycopg
    with psycopg.connect(_dsn()) as cx:
        return pd.read_sql(f"select * from public.{table} order by {order}", cx)


def load(table: str, refresh: bool = False) -> pd.DataFrame:
    """Read a table, pulling from Supabase if the cache is cold or stale."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{table}.parquet"
    if refresh or not p.exists():
        df = pull(table, TABLES[table])
        save_cache(df, p)
        print(f"  pulled {table}: {len(df):,} rows from Supabase -> cache")
        return df
    return pd.read_parquet(p)


def refresh_all(tables=None) -> None:
    import time
    for t in TABLES if tables is None else tables:
        t0 = time.time()
        df = pull(t, TABLES[t])
        (CACHE).mkdir(parents=True, exist_ok=True)
        save_cache(df, CACHE / f"{t}.parquet")
        print(f"  {t:16s} {len(df):>9,} rows  {time.time() - t0:5.1f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Refresh selected Supabase training caches.")
    parser.add_argument("--tables", nargs="+", choices=list(TABLES), default=list(TABLES),
                        help="Tables changed by the publisher; defaults to all tables.")
    args = parser.parse_args()
    print("refreshing local cache from Supabase...")
    refresh_all(args.tables)
