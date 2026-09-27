import pandas as pd
from cfbpred.historical_ranking_inputs import modeled_fixture_mask, terminal_schedule, validate_terminal


def test_modeled_fixture_mask_keeps_fcs_and_lower_division_opponents():
    frame = pd.DataFrame({
        'home_division': ['fbs', 'fcs', 'ii', 'ii'],
        'away_division': ['ii', 'ii', 'fbs', 'ii'],
    })
    assert modeled_fixture_mask(frame).tolist() == [True, True, True, False]

def test_terminal_rows_are_negative_unplayed_and_preserve_real(tmp_path):
    src=tmp_path/'s.parquet'; out=tmp_path/'out'
    pd.DataFrame({'game_id':[1,2],'season':[2001,2001],'week':[1,2],'kick':['2001-08-30T18:00:00Z','2001-09-08T18:00:00Z'],'home_team':['A','B'],'away_team':['B','A'],'home_division':['fbs','fbs'],'away_division':['fbs','fbs'],'home_points':[7,14],'away_points':[3,10],'margin':[4,4],'total':[10,24]}).to_parquet(src)
    terminal_schedule(src,out); report=validate_terminal(out/'snapshot_schedule.parquet',out/'virtual_targets.json')
    assert report['real_rows']==2 and report['virtual_rows']==2 and report['virtual_outcome_rows']==0
    assert all(int(x)<0 for x in pd.read_parquet(out/'snapshot_schedule.parquet').query('game_id < 0').game_id)
