"""Validate offline frozen-model replays and atomically publish each game.

Input files are produced by supabase/replay_week.ts. Existing curves remain
visible until each complete replacement commits. Originals are backed up.
"""
import argparse,json,math,sys
from pathlib import Path
import psycopg
from psycopg.types.json import Jsonb
from psycopg.rows import dict_row
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cfbpred.load_supabase import dsn

def validate(doc):
    s, rows = doc['state'], doc['curve']
    if s['state'] != 'post' or len(rows)<2:
        raise ValueError('A complete finished play history is required')
    prev=3600
    for i,r in enumerate(rows):
        if r['game_id']!=s['game_id'] or r['play_index']!=i or not 0<=r['t_left']<=prev:
            raise ValueError('Invalid game, play order, or clock')
        prev=r['t_left']
        if not all(math.isfinite(r[k]) for k in ('home_win_prob','pred_margin','pred_total')) or not -1e-12<=r['home_win_prob']<=1+1e-12:
            raise ValueError('Invalid model readout')
        r['home_win_prob']=min(1,max(0,r['home_win_prob']))
        if len(r['total_pmf'])!=103 or abs(sum(r['total_pmf'])-1)>0.001:
            raise ValueError('Invalid probability distribution')
        r['total_pmf']=[float(v) for v in r['total_pmf']]
    last=rows[-1]
    if last['t_left']!=0 or last['clock']!='FINAL' or (last['home_points'],last['away_points'])!=(s['home_points'],s['away_points']):
        raise ValueError('History does not reach the observed final score')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('--publish',action='store_true');ap.add_argument('--game',type=int);a=ap.parse_args()
    docs=[json.loads(p.read_text()) for p in sorted(a.directory.glob(f'{a.game}.json' if a.game else '*.json'))]
    if not docs:raise ValueError('No replay files found')
    for d in docs:validate(d)
    print(f'Validated {len(docs)} games, {sum(len(d["curve"]) for d in docs)} points',flush=True)
    if not a.publish:return
    backup=a.directory.parent/'week2_curve_originals';backup.mkdir(exist_ok=True)
    with psycopg.connect(dsn(),row_factory=dict_row) as c:
        for d in docs:
            s,rows=d['state'],d['curve'];gid=s['game_id']
            with c.transaction():
                score=c.execute('select home_points,away_points from games where game_id=%s',(gid,)).fetchone()
                if (score['home_points'],score['away_points'])!=(s['home_points'],s['away_points']):raise ValueError('Final score changed')
                old=c.execute('select * from live_curve where game_id=%s order by play_index',(gid,)).fetchall()
                b=backup/f'{gid}.json'
                if not b.exists():b.write_text(json.dumps(old,default=str))
                c.execute('delete from live_curve where game_id=%s',(gid,))
                cols=list(rows[0]);sql='insert into live_curve ('+','.join(cols)+') values ('+','.join(['%s']*len(cols))+')'
                with c.cursor() as cur:cur.executemany(sql,[tuple(r[k] for k in cols) for r in rows])
                last=rows[-1]
                c.execute('''insert into live_predictions (game_id,updated_at,state,period,clock,home_points,away_points,situation,home_win_prob,pred_margin,pred_total,source,curve_done,live_context)
                values (%s,%s,'post',%s,'FINAL',%s,%s,'',%s,%s,%s,'edge',true,null)
                on conflict(game_id) do update set updated_at=excluded.updated_at,state='post',period=excluded.period,clock='FINAL',home_points=excluded.home_points,away_points=excluded.away_points,situation='',home_win_prob=excluded.home_win_prob,pred_margin=excluded.pred_margin,pred_total=excluded.pred_total,source='edge',curve_done=true,live_context=null''',
                (gid,last['updated_at'],last['period'],s['home_points'],s['away_points'],last['home_win_prob'],last['pred_margin'],last['pred_total']))
                check=c.execute('select count(*) n,max(play_index) idx,min(t_left) t from live_curve where game_id=%s',(gid,)).fetchone()
                if check!={'n':len(rows),'idx':len(rows)-1,'t':0.0}:raise ValueError('Replacement verification failed')
            print(f'Published {gid}: {len(rows)} points',flush=True)
if __name__=='__main__':main()
