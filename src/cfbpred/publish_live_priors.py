"""Give the in-game scorer the SAME pregame distribution the app publishes.

The edge function runs only the in-game (state) half of the network and reads each
game's pregame half from public.game_pregame_logits. That table used to be filled by the
retired raw-DNN pregame tower, so the live number at kickoff started from a different
model than the published pregame card: 7.7 points of win probability apart on average,
disagreeing on the favourite in 30 of 251 games.

This writes the active pregame model's published PMFs instead, as log-probabilities,
one copy per state-tower member (softmax(log p) = p, so each member reads exactly the
published distribution). Rows for games that have kicked off are never rewritten.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import psycopg

from cfbpred.config import CURRENT_SEASON, PQ, ROOT
from cfbpred.load_supabase import dsn
from cfbpred.publish_pregame_logits import DDL

EDGE_META = ROOT / "supabase/model/weights.json"


def state_members() -> int:
    meta = json.loads(EDGE_META.read_text()); meta = meta.get("meta", meta)
    return int(meta["n_members"])


def rows_for(window: set[int], season: int, members: int) -> list[tuple]:
    pmfs = pd.read_parquet(PQ / "game_pmfs.parquet")
    preds = pd.read_parquet(PQ / "game_predictions.parquet", columns=["game_id", "model_version"]).set_index("game_id")
    pmfs = pmfs[(pmfs.season == season) & pmfs.game_id.isin(window)]
    out = []
    for r in pmfs.itertuples():
        pm = np.log(np.clip(np.asarray(r.margin_pmf, np.float64), 1e-12, None)).astype(np.float32)
        pt = np.log(np.clip(np.asarray(r.total_pmf, np.float64), 1e-12, None)).astype(np.float32)
        version = f"prior:{preds.model_version.get(r.game_id, 'unknown')}"
        out.append((int(r.game_id), season, version, members, np.tile(pm, members).tolist(), np.tile(pt, members).tolist()))
    return out


def publish(season: int = CURRENT_SEASON, back: int = 3, ahead: int = 10) -> dict:
    members = state_members()
    with psycopg.connect(dsn()) as conn, conn.cursor() as cur:
        cur.execute("""select game_id from public.games where season = %s and start_date between
                       now() - make_interval(days => %s) and now() + make_interval(days => %s)""", (season, back, ahead))
        window = {int(r[0]) for r in cur.fetchall()}
    rows = rows_for(window, season, members)
    if not rows:
        raise SystemExit("no published pregame distributions in the window")
    with psycopg.connect(dsn(), autocommit=False) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        cur.execute("""delete from public.game_pregame_logits p where not (p.game_id = any(%s))
                       and exists (select 1 from public.live_predictions l where l.game_id=p.game_id
                                   and l.state='post' and l.curve_done=true)""", ([r[0] for r in rows],))
        cur.executemany("""insert into public.game_pregame_logits
                             (game_id, season, model_version, n_members, margin_logits, total_logits)
                           values (%s,%s,%s,%s,%s,%s)
                           on conflict (game_id) do update set model_version = excluded.model_version,
                             n_members = excluded.n_members, margin_logits = excluded.margin_logits,
                             total_logits = excluded.total_logits, updated_at = now()
                           where exists (select 1 from public.games g
                                         where g.game_id = excluded.game_id and g.start_date > now())""", rows)
        conn.commit()
        cur.execute("select model_version, count(*) from public.game_pregame_logits where season = %s group by 1", (season,))
        versions = dict(cur.fetchall())
    report = dict(window=len(window), written=len(rows), state_members=members, versions=versions)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("season", nargs="?", type=int, default=CURRENT_SEASON)
    p.add_argument("--days-back", type=int, default=3); p.add_argument("--days-ahead", type=int, default=10)
    a = p.parse_args(); publish(a.season, a.days_back, a.days_ahead)
