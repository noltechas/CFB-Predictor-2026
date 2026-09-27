"""Publish each ensemble member's PREGAME margin/total logits per game.

The edge function runs only the state half of the network (see export_onnx.py);
the pregame half needs team embeddings, 270 scalars and opponent-history slots
that will never live in Deno. Those pregame logits are fixed for a game — the
pipeline locks pregame calls before kickoff by design — so they are computed
here whenever features are rebuilt and read back by the edge function.

Size: 16 members x (115 + 103) float4 = ~14 KB/game. A full season would be
~13 MB against ~38 MB of headroom on the 500 MB tier, so only a rolling
window around today is kept — the edge function never scores a game that is
not about to start, in progress, or just finished. The nightly refresh rolls
the window forward.

    python3 src/cfbpred/publish_pregame_logits.py 2026 [--days-back 3] [--days-ahead 10]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfbpred.ensemble import load_members  # noqa: E402
from cfbpred.load_supabase import dsn  # noqa: E402
from cfbpred.train import build_tensors, game_batch  # noqa: E402

DDL = """
create table if not exists public.game_pregame_logits (
  game_id     bigint primary key,
  season      int  not null,
  model_version text not null,
  n_members   int  not null,
  margin_logits real[] not null,   -- flattened [n_members * 115]
  total_logits  real[] not null,   -- flattened [n_members * 103]
  updated_at  timestamptz not null default now()
);
create index if not exists game_pregame_logits_season_idx
  on public.game_pregame_logits (season);
"""


def _arg(flag: str, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main() -> None:
    season = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 2026
    dev = torch.device("cpu")
    print("loading feature store + ensemble...")
    d = build_tensors(dev, include_plays=False)
    members = load_members(d, dev)
    version = os.environ.get("MODEL_VERSION", "raw-dnn-delta-v22")

    back = int(_arg("--days-back", 3))
    ahead = int(_arg("--days-ahead", 10))

    import psycopg
    with psycopg.connect(dsn()) as conn, conn.cursor() as cur:
        cur.execute(
            """select game_id from public.games
                 where season = %s
                   and start_date between now() - make_interval(days => %s)
                                      and now() + make_interval(days => %s)""",
            (season, back, ahead))
        window = {int(r[0]) for r in cur.fetchall()}
    print(f"{len(window)} games in the -{back}/+{ahead} day window")

    gids = np.asarray(d["G_gid"])
    seasons = np.asarray(d["G_season"])
    keep = np.array([i for i in np.nonzero(seasons == season)[0]
                     if int(gids[i]) in window], dtype=int)
    print(f"{len(keep)} of those are in the feature store")
    if not len(keep):
        raise SystemExit("nothing to publish")

    rows = []
    bs = 512
    with torch.no_grad():
        for i in range(0, len(keep), bs):
            idx = torch.tensor(keep[i:i + bs])
            batch = game_batch(d, idx, dev)
            pm_l, pt_l = [], []
            for m in members:
                _, pm, pt = m.pregame(batch)
                pm_l.append(pm)
                pt_l.append(pt)
            # [n, members, bins] -> flattened row-major per game
            pm_all = torch.stack(pm_l, 1).numpy().astype(np.float32)
            pt_all = torch.stack(pt_l, 1).numpy().astype(np.float32)
            for j, gpos in enumerate(keep[i:i + bs]):
                rows.append((int(gids[gpos]), season, version, len(members),
                             pm_all[j].ravel().tolist(), pt_all[j].ravel().tolist()))
            print(f"  {min(i + bs, len(keep))}/{len(keep)}", end="\r")
    print()

    with psycopg.connect(dsn(), autocommit=False) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        # Preserve locked inputs while a finished game's history is missing.
        # Otherwise a game rolling off the scoreboard/window becomes impossible
        # to repair without recomputing its pregame prediction after kickoff.
        cur.execute("""delete from public.game_pregame_logits p
                       where not (p.game_id = any(%s))
                         and exists (select 1 from public.live_predictions l
                                     where l.game_id=p.game_id and l.state='post'
                                       and l.curve_done=true)""",
                    ([int(r[0]) for r in rows],))
        cur.executemany(
            """insert into public.game_pregame_logits
                 (game_id, season, model_version, n_members, margin_logits, total_logits)
               values (%s,%s,%s,%s,%s,%s)
               on conflict (game_id) do update set
                 model_version = excluded.model_version,
                 n_members     = excluded.n_members,
                 margin_logits = excluded.margin_logits,
                 total_logits  = excluded.total_logits,
               updated_at    = now()
               where exists (select 1 from public.games g
                             where g.game_id = excluded.game_id and g.start_date > now())""",
            rows)
        conn.commit()
        cur.execute("select pg_size_pretty(pg_total_relation_size('public.game_pregame_logits'))")
        print("table size:", cur.fetchone()[0])
    print(f"published {len(rows)} games ({version})")


if __name__ == "__main__":
    main()
