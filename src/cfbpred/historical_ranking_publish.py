"""Transactional publication of audited historical weekly reconstructions."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from cfbpred.config import PQ
from cfbpred.load_supabase import dsn
from cfbpred.pregame_experiments import save_json,sha
from cfbpred.player_sequence_production import ACTIVE,REGISTRY,_atomic_json,active_bundle

COLUMNS=['season','ranking_week','snapshot_kind','period_start','as_of','generated_at','model_version','model_fingerprint','method_version','reference_hash','reference_count','n_members','entries','diagnostics']


def publish(directory:Path):
    meta=json.loads((directory/'manifest.json').read_text())
    if meta['completed']!=meta['planned']:raise ValueError('Historical reconstruction incomplete')
    if len(meta['documents'])!=meta['completed']:raise ValueError('Ranking manifest count differs')
    if meta['active_sha256']!=sha(ACTIVE):raise ValueError('Active reconstruction model changed')
    active,_,_=active_bundle(ACTIVE)
    docs=[]
    for row in meta['documents']:
        path=Path(row['path'])
        if sha(path)!=row['sha256']:raise ValueError('Ranking document changed')
        doc=json.loads(path.read_text())
        if doc['model_version']!=active['version'] or not doc['diagnostics']['strict_global_player_cutoff']:
            raise ValueError('Ranking lineage/cutoff gate failed')
        if doc['season']<2026:docs.append(doc)
    keys={(d['season'],d['ranking_week']) for d in docs}
    if len(keys)!=len(docs):raise ValueError('Duplicate ranking snapshot')
    lock=Path('/tmp/cfbpred_publish.lock');lock.mkdir()
    try:
        backup=directory/'backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');backup.mkdir(parents=True)
        with psycopg.connect(dsn(),row_factory=dict_row) as conn:
            before=conn.execute('select * from public.fieldcast_rankings where season between 2001 and 2025 order by season,ranking_week').fetchall()
            (backup/'database_rankings.json').write_text(json.dumps(before,default=str,indent=2))
            conn.execute("set local statement_timeout='120s'")
            # Replace the backed-up historical cohort in one transaction so
            # obsolete provider round labels cannot survive the new calendar.
            conn.execute('delete from public.fieldcast_rankings where season between 2001 and 2025')
            sql=f"insert into public.fieldcast_rankings ({','.join(COLUMNS)}) values ({','.join(['%s']*len(COLUMNS))}) on conflict (season,ranking_week) do update set "+','.join(c+'=excluded.'+c for c in COLUMNS[2:])
            with conn.cursor() as cur:
                cur.executemany(sql,[[Jsonb(d[c]) if c in ('entries','diagnostics') else d[c] for c in COLUMNS] for d in docs])
            actual=conn.execute('select * from public.fieldcast_rankings where season between 2001 and 2025 order by season,ranking_week').fetchall()
            bykey={(x['season'],x['ranking_week']):x for x in actual}
            if set(bykey)!=keys:raise ValueError('Historical ranking key coverage differs')
            for doc in docs:
                got=bykey[(doc['season'],doc['ranking_week'])]
                for key in COLUMNS:
                    same=(pd.Timestamp(got[key])==pd.Timestamp(doc[key])) if key in ('as_of','generated_at','period_start') else got[key]==doc[key]
                    if not same:raise ValueError(f"Ranking database readback differs: {doc['season']}/{doc['ranking_week']} {key}")
            # Connection context commits only after all documents pass.
        for doc in docs:_atomic_json(PQ/f"rankings/{doc['season']}-week-{doc['ranking_week']}.json",doc)
        report=dict(passed=True,model_version=active['version'],historical_snapshots=len(docs),seasons=sorted({d['season'] for d in docs}),team_entries=sum(len(d['entries']) for d in docs),readback_exact=True,current_rankings_preserved=True,backup=str(backup.resolve()),published_at=datetime.now(timezone.utc).isoformat())
        save_json(directory/'publication.json',report)
        return report
    finally:lock.rmdir()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();print(json.dumps(publish(a.directory),indent=2))
