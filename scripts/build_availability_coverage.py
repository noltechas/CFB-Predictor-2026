import json, hashlib
from pathlib import Path
import pandas as pd
from cfbpred.pull_espn_roster import team_alias_map

base=Path('models/experiments/complete_history_20260922_v2/availability_coverage')
base.mkdir(parents=True,exist_ok=True)
sched_path=Path('models/experiments/complete_history_20260922_v2/source_schedule.parquet')
facts_path=Path('models/player_sequence_production/availability_complete_v4/20260922T035124704336Z/context/timeline/facts.parquet')
registry_path=Path('models/player_sequence_production/availability_complete_v4/20260922T035124704336Z/context/reviewed_input.json')
sched=pd.read_parquet(sched_path)
sched=sched[sched.game_id>0].copy()
sched['kick']=pd.to_datetime(sched['kick'],utc=True)
alias=team_alias_map()
def norm(x): return alias.get(str(x),str(x))
rows=[]
for r in sched.itertuples(index=False):
    for side,opp in [('home','away'),('away','home')]:
        if getattr(r,f'{side}_division') not in ('fbs','fcs'): continue
        rows.append({'game_id':int(r.game_id),'season':int(r.season),'week':int(r.week),'week_c':int(r.week_c),'kick':r.kick,
                     'season_type':r.season_type,'team':norm(getattr(r,f'{side}_team')),'opponent':norm(getattr(r,f'{opp}_team')),
                     'team_division':getattr(r,f'{side}_division'),'opponent_division':getattr(r,f'{opp}_division'),
                     'conference':getattr(r,f'{side}_conference'),'opponent_conference':getattr(r,f'{opp}_conference'),'side':side})
inventory=pd.DataFrame(rows)
facts=pd.read_parquet(facts_path)
facts=facts[facts.kind.eq('availability') & facts.game_id.notna()].copy()
facts['game_id']=facts.game_id.astype('int64'); facts['team']=facts.team.map(norm)
# Attach aggregate availability facts without converting no-fact rows into healthy.
groups=[]
for (gid,team),d in facts.groupby(['game_id','team'],sort=False):
    players=sorted({int(x) for x in d.player_ref.dropna()})
    statuses=sorted({str(x) for x in d.value.dropna()})
    sources=sorted({str(x) for x in d.source_sha256.dropna()})
    urls=sorted({str(x) for x in d.source_url.dropna()})
    player_game_entries=d[['player_ref','value']].drop_duplicates().shape[0]
    groups.append({'game_id':int(gid),'team':team,'availability_fact_count':int(len(d)),
                   'unique_player_ref_count':len(players),'unique_player_game_entries':int(player_game_entries),'distinct_status_count':len(statuses),
                   'statuses':json.dumps(statuses,separators=(',',':')),
                   'source_count':len(sources),'source_sha256_count':len(sources),'source_url_count':len(urls),
                   'source_sha256s':json.dumps(sources,separators=(',',':'))})
agg=pd.DataFrame(groups)
if agg.empty: agg=pd.DataFrame(columns=['game_id','team'])
inventory=inventory.merge(agg,on=['game_id','team'],how='left')
for c in ['availability_fact_count','unique_player_ref_count','unique_player_game_entries','distinct_status_count','source_count','source_sha256_count','source_url_count']:
    inventory[c]=inventory[c].fillna(0).astype('int64')
inventory['statuses']=inventory['statuses'].fillna('[]'); inventory['source_sha256s']=inventory['source_sha256s'].fillna('[]')
inventory['availability_state']=inventory.availability_fact_count.gt(0).map({True:'observed_published_fact',False:'unknown_no_published_fact'})
inventory['timeline_is_complete_report']=False
inventory.to_parquet(base/'team_game_inventory.parquet',index=False)

def count_summary(d):
 return {'team_game_rows':int(len(d)),'distinct_games':int(d.game_id.nunique()),'games_with_availability_facts':int(d.loc[d.availability_fact_count.gt(0),'game_id'].nunique()),'availability_fact_rows':int(d.availability_fact_count.sum()),'unique_player_game_entries':int(d.loc[d.availability_fact_count.gt(0),'unique_player_game_entries'].sum()),'unique_player_refs':int(facts[facts.team.isin(set(d.team))].player_ref.nunique()),'source_game_rows':int(d.loc[d.availability_fact_count.gt(0),'game_id'].nunique())}
registry=pd.DataFrame(json.load(open(registry_path)))
registry_neb=registry[registry.team.eq('Nebraska')]
registry_neb_av=registry_neb[registry_neb.kind.eq('availability')]
summary={'schedule_path':str(sched_path.resolve()),'schedule_sha256':hashlib.sha256(sched_path.read_bytes()).hexdigest(),'facts_path':str(facts_path.resolve()),'facts_sha256':hashlib.sha256(facts_path.read_bytes()).hexdigest(),'facts_kind':'availability only; absent rows are unknown', 'inventory_path':str((base/'team_game_inventory.parquet').resolve()),'inventory_sha256':hashlib.sha256((base/'team_game_inventory.parquet').read_bytes()).hexdigest(),'all_di_positive_schedule_side_rows':count_summary(inventory),'availability_fact_rows_in_timeline':int(len(facts)),'availability_fact_games_in_timeline':int(facts.game_id.nunique()),'availability_source_hashes_in_timeline':int(facts.source_sha256.nunique()),'availability_source_paths_in_timeline':int(facts.source_path.nunique()),'availability_source_urls_in_timeline':int(facts.source_url.nunique())}
neb=inventory[inventory.team.eq('Nebraska')]
summary['nebraska']={**count_summary(neb),'fixture_denominator_expected':327,'fixture_denominator_matches':int(len(neb)==327),'mapped_game_ids_with_facts':int(neb.loc[neb.availability_fact_count.gt(0),'game_id'].nunique()),'timeline_availability_facts':int(len(facts[(facts.team=='Nebraska')])), 'timeline_availability_game_ids':int(facts[facts.team.eq('Nebraska')].game_id.nunique()), 'timeline_availability_source_paths':int(facts[facts.team.eq('Nebraska')].source_path.nunique()), 'timeline_availability_source_urls':int(facts[facts.team.eq('Nebraska')].source_url.nunique()),'registry_all_fact_rows':int(len(registry_neb)),'registry_all_game_ids':int(registry_neb.game_id.nunique()),'registry_all_source_paths':int(registry_neb.source_path.nunique()),'registry_all_source_urls':int(registry_neb.source_url.nunique()),'registry_availability_facts':int(len(registry_neb_av)),'registry_availability_game_ids':int(registry_neb_av.game_id.nunique()),'registry_availability_source_paths':int(registry_neb_av.source_path.nunique()),'registry_availability_source_urls':int(registry_neb_av.source_url.nunique()),'registry_availability_unique_name_game_status_entries':int(registry_neb_av[['game_id','name','value']].drop_duplicates().shape[0]),'registry_distinction':'All reviewed Nebraska rows include 98 availability and 25 depth facts across 45 game IDs and 34 source paths; availability-only rows cover 98 facts across 34 game IDs and 23 source paths/URLs.'}
summary['season_coverage']={str(int(k)):count_summary(v) for k,v in inventory.groupby('season')}
summary['conference_coverage']={str(k):count_summary(v) for k,v in inventory[inventory.conference.notna()].groupby('conference')}
unmatched=facts.merge(inventory[['game_id','team']].drop_duplicates(),on=['game_id','team'],how='left',indicator=True)
summary['facts_unmatched_to_di_inventory']=int((unmatched._merge=='left_only').sum())
summary['limitation']='A resolved timeline fact proves a published player status for that game; no source row is not evidence of health or a complete report.'
(base/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps({'rows':len(inventory),'games':inventory.game_id.nunique(),'fact_rows':len(facts),'nebraska':summary['nebraska']},indent=2))
