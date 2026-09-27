"""Replay every completed weekly ranking using one latest fitted checkpoint."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from cfbpred.config import CURRENT_SEASON
from cfbpred.pregame_experiments import save_json,sha
from cfbpred.rankings import generate_player


def cutoffs(schedule,now=None):
    """Calendar weeks avoid the provider's overlapping postseason round IDs.

    Week one ends at the Tuesday 12:00 UTC before the first FBS Week 2
    kickoff. Subsequent cutoffs are seven days apart, including bowl weeks.
    Only completed results enter the ranking; scheduled rows set the calendar.
    """
    now=pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    frame=schedule[schedule.game_id.gt(0)].copy()
    if all(c in frame for c in ("home_division", "away_division")):
        frame=frame[frame.home_division.isin(["fbs", "fcs"]) | frame.away_division.isin(["fbs", "fcs"])]
    frame['kick']=pd.to_datetime(frame.kick,utc=True)
    rows=[]
    for year,games in frame.groupby('season',sort=True):
        if int(year)>CURRENT_SEASON:continue
        opening=games.kick.min()
        rows.append(dict(season=int(year),week=0,as_of=(opening-pd.Timedelta(seconds=1)).isoformat(),preseason=True))
        done=games[games.home_points.notna() & games.away_points.notna() & games.kick.lt(now)]
        if done.empty:continue
        fbs=games
        if 'home_division' in games:
            fbs=games[games.home_division.eq('fbs') | games.away_division.eq('fbs')]
        second=fbs.loc[fbs.week_c.eq(2),'kick'].min()
        if pd.notna(second):
            anchor=second.normalize()-pd.Timedelta(days=(second.weekday()-1)%7)+pd.Timedelta(hours=12)
            if anchor>=second:anchor-=pd.Timedelta(days=7)
        else:
            first=fbs.loc[fbs.week_c.eq(1),'kick'].min()
            if pd.isna(first):first=opening
            anchor=first.normalize()+pd.Timedelta(days=(1-first.weekday())%7,hours=12)
            if anchor<=first:anchor+=pd.Timedelta(days=7)
        if anchor<=opening:raise ValueError(f'Invalid weekly calendar anchor for {year}')
        end=done.kick.max()+pd.Timedelta(hours=8)
        week=1;seen=set();cutoff=anchor
        while True:
            target=cutoff
            if target>now:
                # Include the just-finished current slate before Tuesday,
                # but never claim a week containing future scheduled games.
                pending=games[games.kick.gt(now)&games.kick.lt(target)]
                if int(year)!=CURRENT_SEASON or end>now or len(pending):break
                target=end
            if target.isoformat() not in seen:
                rows.append(dict(season=int(year),week=week,as_of=target.isoformat(),preseason=False))
                seen.add(target.isoformat())
            if cutoff>=end or target>=end:break
            week+=1;cutoff=anchor+pd.Timedelta(days=7*(week-1))
    return rows


def reconstruct(source,active,output,first_season=None,last_season=None):
    paths={k:Path(v) for k,v in json.loads(source.read_text()).items()}
    schedule=pd.read_parquet(paths['inputs']/'snapshot_schedule.parquet')
    rows=[r for r in cutoffs(schedule) if (first_season is None or r['season']>=first_season)
          and (last_season is None or r['season']<=last_season)]; cache={}
    identity=dict(source=str(source.resolve()),source_sha256=sha(source),active_sha256=sha(active),
                  input_sha256=sha(paths['inputs']/'game_scalars.parquet'),
                  schedule_sha256=sha(paths['inputs']/'snapshot_schedule.parquet'),
                  store_manifests={k:sha(paths[k]/'manifest.json') for k in ('players','sequences','full_history','timeline')},
                  calendar='Tuesday 12:00 UTC; sequential season weeks, postseason rounds normalized',mode='latest_model_reconstruction',historical_accuracy_claim=False)
    output.mkdir(parents=True,exist_ok=True)
    manifest=output/(f'manifest-{first_season}-{last_season}.json' if first_season is not None else 'manifest.json')
    if manifest.exists():
        old=json.loads(manifest.read_text())
        for k,v in identity.items():
            if old[k]!=v:raise ValueError('Replay lineage changed: '+k)
    save_json(manifest,dict(**identity,planned=len(rows),completed=0))
    completed=[]
    for i,row in enumerate(rows):
        path=output/f"{row['season']}-week-{row['week']}.json"
        if path.exists():doc=json.loads(path.read_text())
        else:
            doc=generate_player(row['season'],preseason=row['preseason'],as_of=row['as_of'],active_path=active,input_source=source,replay_cache=cache,ranking_week=row['week'])
            if doc['ranking_week']!=row['week']:raise ValueError(f"Week differs: {row} / {doc['ranking_week']}")
            save_json(path,doc)
        if (doc['season']!=row['season'] or doc['ranking_week']!=row['week']
                or pd.Timestamp(doc['as_of'])!=pd.Timestamp(row['as_of'])
                or doc['model_version']!=json.loads(active.read_text())['version']):
            raise ValueError('Saved reconstruction identity differs')
        if not doc['diagnostics']['strict_global_player_cutoff']:raise ValueError('Missing cutoff audit')
        completed.append(dict(season=row['season'],week=row['week'],path=str(path.resolve()),sha256=sha(path),teams=len(doc['entries']),reference_count=doc['reference_count']))
        save_json(manifest,dict(**identity,planned=len(rows),completed=len(completed),documents=completed))
        print(f"REPLAY {i+1}/{len(rows)} {row['season']} week {row['week']}",flush=True)
    return json.loads(manifest.read_text())


def main():
    p=argparse.ArgumentParser()
    for name in ('source','active','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--first-season',type=int);p.add_argument('--last-season',type=int)
    a=p.parse_args(); reconstruct(a.source,a.active,a.output,a.first_season,a.last_season)
if __name__=='__main__':main()
