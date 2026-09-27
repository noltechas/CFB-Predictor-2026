import pandas as pd

from cfbpred.player_catalog import link_recruits,choose_scouting,attach_history,flag_identity_risks


def fixture():
    roster=pd.DataFrame(dict(athlete_id=["1","2","3"],season=[2025]*3,team=["New"]*3,recruitIds=[["a"],["b"],["c"]]))
    recruits=pd.DataFrame([dict(id=rid,athleteId=aid,year=yr,recruitType="HighSchool",rating=.95,stars=4,ranking=100)
                          for rid,aid,yr in [("a","1",2022),("b","999",2022),("c","3",2026)]])
    return roster,recruits


def test_exact_identity_follows_transfer_but_conflicting_links_and_future_recruits_do_not():
    roster,recruits=fixture()
    r,links,conflicts=link_recruits(roster,recruits)
    assert len(links)==1 and links.iloc[0].athlete_id=="1"
    assert len(conflicts)==1 and conflicts.iloc[0].athlete_id=="2"
    catalog=choose_scouting(r,links)
    history=pd.DataFrame(dict(athlete_id=["1","1"],source_season=[2024,2025],game_id=[10,11],
                             source_team=["Old","Future"],source=["production"]*2))
    out=attach_history(catalog,history).set_index("athlete_id")
    assert out.loc["1","prior_programs"]==["Old"]
    assert out.loc["1","observed_at_other_program"]
    assert out.loc["1","prior_box_games"]==1 and out.loc["1","hs_rating"]==.95
    assert not out.loc["2","any_scouting_known"]


def test_multiple_scouting_records_remain_ambiguous_instead_of_choosing_best_rating():
    roster,recruits=fixture()
    recruits=pd.concat([recruits,pd.DataFrame([dict(recruits.iloc[0],id="second",rating=.99)])],ignore_index=True)
    r,links,_=link_recruits(roster,recruits)
    out=choose_scouting(r,links)
    assert out.iloc[0].hs_recruit_matches==2
    assert not out.iloc[0].hs_scouting_known


def test_old_id_reappearance_is_quarantined_but_continuous_transfer_is_not():
    rows=pd.DataFrame(dict(athlete_id=["1","1","2","2"],season=[2017,2026,2025,2026],
        team=["Old","New","Old","New"],first_prior_season=[2016,2016,2024,2024],
        any_scouting_known=[True]*4,prior_box_observed=[True]*4))
    out=flag_identity_risks(rows).set_index(["athlete_id","season"])
    assert out.loc[("1",2026),"identity_review_required"]
    assert not out.loc[("1",2026),"history_usable_without_identity_review"]
    assert not out.loc[("2",2026),"identity_review_required"]
