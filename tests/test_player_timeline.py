import json
import numpy as np
import pandas as pd
import pytest
from cfbpred.player_timeline import normalize,compile_arrays,resolve_names,CONTEXT_COLS


def fact(kind,value,known="2024-09-01T00:00:00Z",**extra):
    return dict(player_ref=1,season=2024,kind=kind,value=value,known_at=known,
                effective_at=known,valid_until="2025-03-01T00:00:00Z",game_id=None,
                source_url="https://official.example/report",source_sha256="abc",**extra)


def test_unknown_pending_is_not_healthy_or_invented_depth():
    for kind,value in [("availability","pending"),("depth","1")]:
        with pytest.raises(ValueError):normalize([fact(kind,value)])


def test_asof_position_departure_status_expiry_and_transfer_timing(tmp_path):
    games=pd.DataFrame([dict(game_id=gid,season=2024,side="home",offset=i,players=1,kick=kick)
        for i,(gid,kick) in enumerate([(1,"2024-09-01T12:00:00Z"),(2,"2024-09-08T12:00:00Z"),(3,"2024-09-15T12:00:00Z")])])
    availability=fact("availability","out");availability["game_id"]=2
    late=fact("position","DB","2024-09-08T12:00:00Z")
    departed=fact("membership","departed","2024-09-10T00:00:00Z")
    records=[fact("position","WR"),availability,late,departed]
    compile_arrays(normalize(records),games,np.array([1,1,1]),tmp_path)
    assert np.load(tmp_path/"position.npy").tolist()==[3,3,8] # exact kickoff is too late
    assert np.load(tmp_path/"departed.npy").tolist()==[False,False,True]
    c=np.load(tmp_path/"context.npy")
    assert c[:,0].tolist()==[0,1,0] # no carrying one game's injury into others
    assert c[:,1].tolist()==[0,1,0]
    assert c[:,CONTEXT_COLS.index("available")].tolist()==[0,0,0]


def test_equal_timestamp_conflicts_and_expired_facts_remain_unknown(tmp_path):
    games=pd.DataFrame([dict(game_id=1,season=2024,side="home",offset=0,players=1,kick="2024-10-01T12:00:00Z")])
    old=fact("membership","departed");old["valid_until"]="2024-09-02T00:00:00Z"
    compile_arrays(normalize([fact("position","QB"),fact("position","WR"),old]),games,np.array([1]),tmp_path)
    assert np.load(tmp_path/"position.npy").tolist()==[-1]
    assert not np.load(tmp_path/"departed.npy").any()


def test_name_selection_never_merges_people_across_programs():
    cat=pd.DataFrame(dict(season=[2024]*3,team=["A","B","B"],firstName=["Joe"]*3,
                         lastName=["Smith"]*3,roster_row=[1,2,3]))
    rows=[dict(season=2024,team=t,name="Joe Smith") for t in ["A","B","C"]]
    linked,rejected=resolve_names(rows,cat)
    assert [x["player_ref"] for x in linked]==[2]
    assert len(rejected)==2
