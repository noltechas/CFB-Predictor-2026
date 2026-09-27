import numpy as np
import pandas as pd
import pytest

from cfbpred.qb_roster_prior import annual_passing, roster_priors, attach


def test_preseason_passer_selection_follows_transfers_and_excludes_current_year():
    targets = pd.DataFrame({"season":[2025]*3,"team":["New","Missing","Fresh"]})
    ros = pd.DataFrame({"season":[2025]*3,"team":["New","New","Fresh"],"athlete_id":["A","B","C"]})
    annual = pd.DataFrame([dict(season=s,athlete_id=a,att=n,cmp=n/2,yds=n*8,td=n*.05,ints=n*.02,games=10)
                          for s,a,n in [(2024,"A",100),(2024,"B",20),(2025,"B",9999),(2019,"B",9999)]])
    f = roster_priors(targets,ros,annual).set_index("team")
    assert f.loc["New","prior_passer_id"] == "A"
    assert f.loc["New","source_season"] == 2024
    assert f.loc["New","pre_qb_ypa"] == 8
    assert f.loc["Missing","pre_qb_roster_known"] == 0
    assert np.isnan(f.loc["Missing","pre_qb_passers"])
    assert f.loc["Fresh","pre_qb_roster_known"] == 1
    assert f.loc["Fresh","pre_qb_passer_known"] == 0
    assert np.isnan(f.loc["Fresh","pre_qb_ypa"])
    shuffled = roster_priors(targets,ros.sample(frac=1),annual.sample(frac=1)).set_index("team")
    pd.testing.assert_frame_equal(f,shuffled)


def test_incomplete_box_does_not_get_zero_numerators_or_enter_denominator():
    records = []
    for gid,items in [(1,{"C/ATT":"10/20","YDS":"150","TD":"2","INT":"1"}),
                      (2,{"C/ATT":"10/20","YDS":"150","TD":None,"INT":"0"})]:
        records += [dict(season=2024,game_id=gid,athlete_id="A",category="passing",stat_type=k,stat=v) for k,v in items.items()]
    annual = annual_passing(pd.DataFrame(records)).iloc[0]
    assert annual.att == 20 and annual.games == 1 and annual.td == 2
    duplicated = annual_passing(pd.DataFrame(records+records)).iloc[0]
    pd.testing.assert_series_equal(annual,duplicated)
    conflict = dict(records[0],stat="11/20")
    with pytest.raises(ValueError,match="Conflicting"):
        annual_passing(pd.DataFrame(records+[conflict]))
