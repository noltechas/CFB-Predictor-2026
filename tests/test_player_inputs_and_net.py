import json

import numpy as np
import pandas as pd
import pytest
import torch

from cfbpred.player_identity import canonical_rosters, resolve_history, resolve_rosters, safe_recruit_links
from cfbpred.player_inputs import HISTORY_COLS, PlayerStore, build_snapshots, read_boxes
from cfbpred.player_net import PlayerSetEncoder, make_player_model
from cfbpred.train import game_batch


def roster(rows):
    return pd.DataFrame(rows, columns=["athlete_id", "season", "team"]).assign(
        firstName="Same", lastName="Name", position="QB", recruitIds=lambda d: [[] for _ in range(len(d))])


def test_verified_collision_and_alias_preserve_transfer_without_double_count():
    r = roster([("1", 2017, "Old"), ("1", 2025, "New"), ("2", 2024, "Prior"), ("2", 2025, "New")])
    rule = dict(athlete_id="1", from_season=2024, teams=["New"], person_id="cfbd:2",
                college_start=2024, position="QB", recruit_ids=[], evidence="official")
    identity = resolve_rosters(r, [rule])
    assert len(canonical_rosters(identity)) == 3
    h = resolve_history(r, identity)
    assert h.loc[0, "person_id"] != h.loc[1, "person_id"]
    assert h.loc[1, "person_id"] == h.loc[2, "person_id"] == h.loc[3, "person_id"]
    assert h.history_usable.all()


def test_unverified_gap_does_not_assert_same_person_and_unknown_history_rejected():
    r = roster([("1", 2017, "Old"), ("1", 2025, "New"), ("2", 2024, "Prior"), ("2", 2025, "New")])
    identity = resolve_rosters(r, [])
    assert identity.loc[1, "identity_uncertain"]
    assert identity.loc[0, "person_id"] != identity.loc[1, "person_id"]
    assert identity.loc[2, "person_id"] == identity.loc[3, "person_id"]
    unknown = roster([("1", 2025, "Unobserved")])
    assert not resolve_history(unknown, identity).history_usable.any()


def test_explicit_dated_recruit_bridges_missing_rosters_but_later_class_does_not():
    r = roster([("1",2022,"A"),("1",2026,"B"),("2",2017,"Old"),("2",2023,"New")])
    r["recruitIds"] = [["q1"],["q1"],["q2"],["q2"]]
    q = pd.DataFrame([dict(id=i,athleteId=a,recruitType="HighSchool",year=y,name="Same Name")
                      for i,a,y in [("q1","1",2022),("q2","2",2023)]])
    result = resolve_rosters(r,[],q)
    assert not result.loc[1,"identity_uncertain"]
    assert result.loc[1,"person_id"] == result.loc[0,"person_id"]
    assert result.loc[3,"identity_uncertain"]
    assert result.loc[3,"person_id"] != result.loc[2,"person_id"]


def test_matching_last_name_does_not_link_different_first_names():
    r = roster([("1",2025,"A")]);r["firstName"]="Ross";r["lastName"]="Brown";r["recruitIds"]=[["q"]]
    q = pd.DataFrame([dict(id="q",athleteId="1",recruitType="HighSchool",year=2025,name="Austin Brown")])
    _,accepted,rejected = safe_recruit_links(resolve_rosters(r,[]),q)
    assert accepted.empty and rejected.reject_reason.eq("name_conflict").all()


def test_unrelated_recruit_cannot_quarantine_a_valid_college_career():
    r=roster([("1",2020,"A"),("1",2023,"B")]);r["firstName"]="Tre";r["lastName"]="Smith";r["recruitIds"]=[["q"],["q"]]
    q=pd.DataFrame([dict(id="q",athleteId="1",recruitType="HighSchool",year=2023,name="Genesis Smith")])
    resolved=resolve_rosters(r,[],q)
    assert not resolved.identity_uncertain.any()
    assert resolved.person_id.nunique()==1


def test_confirmed_freshman_gets_his_recruit_but_not_recycled_ids_old_recruit():
    r = roster([("1", 2017, "Old"), ("1", 2026, "New")])
    rule = dict(athlete_id="1", from_season=2026, teams=["New"], person_id="new",
                college_start=2026, position="DL", recruit_ids=["new"], evidence="official")
    r = canonical_rosters(resolve_rosters(r, [rule]))
    q = pd.DataFrame([dict(id=i, athleteId="1", recruitType="HighSchool", year=y,
                           name="Same Name", rating=.85, stars=3, ranking=100, position="DL")
                      for i,y in [("old",2016), ("new",2026)]])
    cat, links, rejected = safe_recruit_links(r,q)
    current = cat.loc[cat.season.eq(2026), "roster_row"].iloc[0]
    assert links.loc[links.roster_row.eq(current), "recruit_id"].tolist() == ["new"]
    old = cat.loc[cat.season.eq(2017), "roster_row"].iloc[0]
    assert links.loc[links.roster_row.eq(old), "recruit_id"].tolist() == ["old"]


def test_duplicate_boxes_do_not_double_production_and_conflicts_stay_unknown(tmp_path):
    d = pd.DataFrame([dict(game_id=1, home_away="home", athlete_id="1", category="passing", stat_type=t, stat=v)
                      for t,v in [("YDS","100"),("YDS","100"),("C/ATT","10/15"),("TD","1"),("TD","2")]])
    p = tmp_path/"raw.parquet"; d.to_parquet(p)
    b = read_boxes([p]).iloc[0]
    assert b.pass_yds == 100 and b.cmp == 10 and b.att == 15
    assert np.isnan(b.pass_td)


def test_snapshot_transfer_same_kickoff_future_and_full_roster(tmp_path):
    cat = pd.DataFrame(dict(season=[2025]*201, team=["A"]*200+["B"],
                            person_id=[f"p{i}" for i in range(200)]+["transfer"]))
    kicks = pd.to_datetime(["2025-08-01", "2025-08-08", "2025-08-08", "2025-08-15", "2025-08-22"], utc=True)
    schedule = pd.DataFrame(dict(game_id=[1,2,3,4,5], season=2025, kick=kicks,
                                 home_team="A", away_team="B", margin=[1,2,3,np.nan,np.nan]))
    boxes = pd.DataFrame([dict(game_id=gid, season=2025, kick=kick, team="Prior", person_id="transfer",
                               **{c: 1. for c in HISTORY_COLS}) for gid,kick in [(1,kicks[0]),(2,kicks[1]),(3,kicks[2])]])
    table = build_snapshots(cat, boxes, schedule, set(schedule.game_id), tmp_path)
    assert table.players.max()==200  # deliberately exceeds usual roster caps
    refs = np.load(tmp_path/"current_refs.npy")
    b = table[table.side.eq("away")]
    assert refs[b.offset].tolist() == [0,1,1,3,3]
    assert (table.last_history_kick.dropna() < table.loc[table.last_history_kick.notna(), "kick"]).all()
    # Every scheduled side must have a reference, even when no roster exists.
    assert len(table)==10


@pytest.mark.parametrize("pooling", ["meanmax", "attention"])
def test_player_pooling_invariant_padding_safe_and_gradients_reach_scouting(pooling):
    torch.manual_seed(1)
    encoder = PlayerSetEncoder(18, len(HISTORY_COLS), pooling=pooling).eval()
    scout = torch.randn(2,2,11,18, requires_grad=True)
    hist = torch.randn(2,2,11,4,len(HISTORY_COLS))
    pos = torch.randint(0,10,(2,2,11)); mask = torch.ones(2,2,11,dtype=torch.bool)
    mask[1] = False
    y = encoder(scout,hist,pos,mask)
    perm = torch.randperm(11)
    assert torch.allclose(y,encoder(scout[:,:,perm],hist[:,:,perm],pos[:,:,perm],mask[:,:,perm]),atol=1e-6)
    padded_scout = torch.cat([scout,torch.full_like(scout,1e6)],2)
    assert torch.allclose(y,encoder(padded_scout,torch.cat([hist,hist],2),torch.cat([pos,pos],2),
                                   torch.cat([mask,torch.zeros_like(mask)],2)),atol=1e-6)
    assert torch.equal(y[1],torch.zeros_like(y[1]))
    y[0].sum().backward()
    assert scout.grad[0].abs().sum()>0
    assert scout.grad[1].abs().sum()==0


def test_position_source_scouting_first_roster_fallback_and_roster_aliases():
    from cfbpred.player_inputs import POSITIONS, assign_positions, map_positions
    catalog = pd.DataFrame(dict(roster_row=[10, 11, 12, 13, 14, 15],
                                position=["PK", "EDGE", "WR", None, "?", "NT"]))
    links = pd.DataFrame(dict(roster_row=[11, 11, 12], recruit_year=[2019, 2023, 2024],
                              position=["ATH", "WDE", "CB"]))
    scouting = assign_positions(catalog, links, "scouting")
    # only linked rows are known; the latest recruiting class wins; ATH stays unknown
    assert scouting.tolist() == [0, POSITIONS["DL"], POSITIONS["DB"], 0, 0, 0]
    fallback = assign_positions(catalog, links, "scouting_then_roster")
    # scouting position kept where present (CB over roster WR), roster fills the rest
    assert fallback.tolist() == [POSITIONS["ST"], POSITIONS["DL"], POSITIONS["DB"], 0, 0, POSITIONS["DL"]]
    assert map_positions(["PK", "G", "T", "KR", "PR", "S", "LS", "K/P", "ATH", None]).tolist() == [
        POSITIONS["ST"], POSITIONS["OL"], POSITIONS["OL"], POSITIONS["ST"], POSITIONS["ST"],
        POSITIONS["DB"], POSITIONS["ST"], POSITIONS["ST"], 0, 0]
    with pytest.raises(ValueError):
        assign_positions(catalog, links, "roster_only")


def test_roster_recruit_pointer_and_name_outvote_a_mismatched_provider_athlete_id():
    # CFBD's five-star record for the player carries someone else's athleteId, while
    # a three-star record with a different first name carries HIS athleteId.
    r = roster([("1", 2026, "Oregon")]); r["firstName"] = "Dante"; r["lastName"] = "Moore"; r["recruitIds"] = [["star"]]
    q = pd.DataFrame([
        dict(id="star", athleteId="999", recruitType="HighSchool", year=2023, name="Dante Moore", rating=.99, stars=5, ranking=3, position="QB"),
        dict(id="other", athleteId="1", recruitType="HighSchool", year=2023, name="Jaeden Moore", rating=.85, stars=3, ranking=400, position="LB"),
    ])
    cat, accepted, rejected = safe_recruit_links(canonical_rosters(resolve_rosters(r, [])), q)
    assert accepted.recruit_id.tolist() == ["star"]
    assert accepted.match_method.tolist() == ["roster_recruit_id_name_verified"]
    assert rejected.reject_reason.tolist() == ["name_conflict"]
    assert rejected.reject_reason.notna().all()   # no more silent, reason-less rejections


def test_mismatched_provider_id_stays_rejected_when_another_player_of_that_name_owns_it():
    # Two Dante Moores: the record's athleteId belongs to the other one, so the roster
    # pointer alone cannot claim it.
    r = roster([("1", 2026, "Oregon"), ("999", 2026, "UCLA")])
    r["firstName"] = "Dante"; r["lastName"] = "Moore"; r["recruitIds"] = [["star"], []]
    q = pd.DataFrame([dict(id="star", athleteId="999", recruitType="HighSchool", year=2023, name="Dante Moore",
                           rating=.99, stars=5, ranking=3, position="QB")])
    cat, accepted, rejected = safe_recruit_links(canonical_rosters(resolve_rosters(r, [])), q)
    first = cat.loc[cat.athlete_id.eq("1"), "roster_row"].iloc[0]
    assert not accepted.roster_row.eq(first).any()
    assert (rejected.loc[rejected.roster_row.eq(first), "reject_reason"] == "provider_athlete_id_conflict").all()
