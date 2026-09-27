import numpy as np
import pandas as pd
import pytest
from cfbpred.historical_rankings import cutoffs
from cfbpred.rankings import _current_roster_games


def test_cutoffs_exclude_future_and_virtual_targets():
    frame=pd.DataFrame([
        dict(game_id=1,season=2025,week_c=1,kick='2025-09-01T18:00Z',home_points=20,away_points=10),
        dict(game_id=2,season=2026,week_c=1,kick='2026-09-01T18:00Z',home_points=20,away_points=10),
        dict(game_id=3,season=2026,week_c=2,kick='2026-10-01T18:00Z',home_points=None,away_points=None),
        dict(game_id=-3,season=2026,week_c=999,kick='2027-01-01T18:00Z',home_points=None,away_points=None),
    ])
    rows=cutoffs(frame,now='2026-09-20T00:00Z')
    assert [(r['season'],r['week']) for r in rows]==[(2025,0),(2025,1),(2026,0),(2026,1)]
    assert rows[-1]['as_of']=='2026-09-02T02:00:00+00:00'


def test_historical_roster_selection_preserves_missingness():
    catalog=pd.DataFrame([dict(team='Team',division='fbs')])
    rows=pd.DataFrame([
        dict(team='Team',season=y,kick=pd.Timestamp(f'{y}-09-01',tz='UTC'),game_id=y,players=n,last_history_kick=pd.NaT)
        for y,n in [(2001,0),(2026,40)]
    ])
    got=_current_roster_games(catalog,rows,np.array([2001,2026]),pd.Timestamp('2001-08-01',tz='UTC'),nearest=True,season=2001,allow_empty=True)
    assert got.game_id.tolist()==[2001] and got.players.tolist()==[0]
    with pytest.raises(ValueError,match='empty'):
        _current_roster_games(catalog,rows,np.array([2001,2026]),pd.Timestamp('2001-08-01',tz='UTC'),season=2001)


def test_postseason_round_numbers_do_not_create_nonchronological_weeks():
    frame=pd.DataFrame([
        dict(game_id=i,season=2023,week_c=w,kick=k,home_points=20,away_points=10,home_division=d,away_division=d)
        for i,w,k,d in [(1,1,'2023-08-31T18:00Z','fbs'),(2,2,'2023-09-07T18:00Z','fbs'),(3,33,'2023-11-25T18:00Z','fcs'),(4,21,'2024-01-08T18:00Z','fbs')]
    ])
    rows=cutoffs(frame,now='2026-09-20T00:00Z')
    assert [r['week'] for r in rows]==list(range(len(rows)))
    assert [pd.Timestamp(r['as_of']) for r in rows]==sorted(pd.Timestamp(r['as_of']) for r in rows)
    assert rows[-1]['as_of']=='2024-01-09T12:00:00+00:00'
    assert max(r['week'] for r in rows)<21
