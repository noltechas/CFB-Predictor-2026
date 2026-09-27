"""Replace feature rows atomically, retaining tables, policies and app views."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.config import FEAT
from cfbpred.load_supabase import dsn
from cfbpred.publish import copy_chunked
from cfbpred.source import CACHE, save_cache


def cache_payload(df, types):
    """Match Postgres REAL rounding before supplying the committed local read."""
    cached = df.copy()
    for col, dtype in types.items():
        if dtype == "real":
            cached[col] = pd.to_numeric(cached[col], errors="raise").astype("float32").astype("float64")
    return cached


def main():
    published = {}
    with psycopg.connect(dsn()) as conn, conn.cursor() as cur:
        cur.execute("set local statement_timeout = '20min'")
        cur.execute("set local lock_timeout = '30s'")
        for table in ("game_scalars", "game_history"):
            df = pd.read_parquet(FEAT / f"{table}.parquet")
            cur.execute("select column_name, data_type from information_schema.columns where table_schema='public' and table_name=%s", (table,))
            types = dict(cur.fetchall())
            if set(types) != set(df):
                raise ValueError(f"{table}: schema differs; refusing to replace feature rows")
            for col, dtype in types.items():
                if dtype in {"smallint", "integer", "bigint"}:
                    df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
            cur.execute(f"truncate public.{table}")
            n = copy_chunked(cur, df, f"public.{table}", list(df))
            cur.execute(f"select count(*) from public.{table}")
            assert cur.fetchone()[0] == n
            published[table] = cache_payload(df, types)
            print(f"{table}: {n:,} verified rows", flush=True)
    print("Feature transaction committed; existing views and permissions retained")
    # Only a successful database COMMIT may advance the model's local cache.
    # This is the exact just-published payload, without a full-table download.
    CACHE.mkdir(parents=True, exist_ok=True)
    for table, frame in published.items():
        save_cache(frame, CACHE / f"{table}.parquet")
        print(f"{table}: committed payload cached locally", flush=True)


if __name__ == "__main__":
    main()
