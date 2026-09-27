"""Publish per-game input attributions for the app's pregame "why" section.

One row per game holding the whole payload. Explanations are a readout of an
already-published prediction, so this never writes a prediction, a PMF or a ranking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from cfbpred.config import MODELS
from cfbpred.load_supabase import dsn

DEFAULT = MODELS / "player_sequence_production/explanations"
SCHEMA = """
create table if not exists public.game_explanations (
  game_id        bigint primary key,
  season         int  not null,
  model_version  text not null,
  home_team      text not null,
  away_team      text not null,
  home_win_prob  double precision not null,
  pred_margin    double precision not null,
  generated_at   timestamptz not null,
  payload        jsonb not null
);
create index if not exists game_explanations_season on public.game_explanations (season);
alter table public.game_explanations enable row level security;
"""
READ_POLICY = """
do $$ begin
  if not exists (select 1 from pg_policies
                 where schemaname='public' and tablename='game_explanations' and policyname='anon_read') then
    create policy anon_read on public.game_explanations for select to anon using (true);
  end if;
end $$;
"""


def publish(directory: Path = DEFAULT, *, prune_other_versions: bool = True) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    payloads = [json.loads((directory / f"{gid}.json").read_text()) for gid in manifest["game_ids"]]
    if any(p["model_version"] != manifest["model_version"] for p in payloads):
        raise ValueError("Explanation payloads disagree with the manifest model version")
    with psycopg.connect(dsn()) as conn, conn.cursor() as cur:
        cur.execute(SCHEMA)
        cur.execute(READ_POLICY)
        cur.executemany(
            """insert into public.game_explanations
               (game_id, season, model_version, home_team, away_team, home_win_prob, pred_margin, generated_at, payload)
               values (%s,%s,%s,%s,%s,%s,%s,%s,%s)
               on conflict (game_id) do update set
                 season=excluded.season, model_version=excluded.model_version,
                 home_team=excluded.home_team, away_team=excluded.away_team,
                 home_win_prob=excluded.home_win_prob, pred_margin=excluded.pred_margin,
                 generated_at=excluded.generated_at, payload=excluded.payload""",
            [(p["game_id"], p["season"], p["model_version"], p["home_team"], p["away_team"],
              p["home_win_prob"], p["pred_margin"], p["generated_at"], Jsonb(p)) for p in payloads])
        if prune_other_versions:
            # A stale explanation of a superseded prediction is worse than none.
            cur.execute("delete from public.game_explanations where model_version <> %s",
                        (manifest["model_version"],))
        conn.commit()
        cur.execute("select model_version, count(*) from public.game_explanations group by 1")
        published = dict(cur.fetchall())
    report = dict(passed=True, model_version=manifest["model_version"], written=len(payloads), published=published)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    args = parser.parse_args()
    publish(args.directory)
