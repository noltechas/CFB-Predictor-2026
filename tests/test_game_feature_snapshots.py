"""Forecast-time regressions: same-week games, distant targets and transfers."""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfbpred.player_features import build_box_features
from cfbpred.game_feature_snapshots import player_snapshots, pbp_snapshots
from cfbpred.predict import preserve_started_predictions
from cfbpred.backtest_pregame import partitions, fit_inputs
from cfbpred.pull_current_plays import normalize, fetch_game_stats
import torch


def games():
    return pd.DataFrame({"game_id": [1, 2, 3, 4, 5], "season": [2026]*5,
        "week_c": [1, 1, 2, 3, 8], "home_team": ["A"]*5, "away_team": ["Z"]*5,
        "kick": pd.to_datetime(["2026-08-29", "2026-09-05", "2026-09-12", "2026-09-19", "2026-10-24"], utc=True),
        "home_points": [30, 35, np.nan, np.nan, np.nan],
        "away_points": [10, 14, np.nan, np.nan, np.nan]})


def players():
    return pd.DataFrame([{"game_id": gid, "home_away": "home", "category": "passing",
         "stat_type": stat, "stat": value, "athlete_id": "qb1", "athlete": "Quarterback",
         "season": 2026, "week_c": 1}
        for gid, yards in [(1, 200), (2, 300), (3, 9999)]
        for stat, value in [("YDS", str(yards)), ("C/ATT", "20/30")]])


def test_every_future_target_gets_latest_box_without_counting_unplayed_games():
    out = build_box_features(games(), raw=players(), per_game=True).query("team == 'A'").set_index("game_id")
    assert np.isnan(out.loc[1, "bx_pass_ypg"])
    assert out.loc[2, "bx_pass_ypg"] == 200  # same week, strictly earlier game
    assert (out.loc[[3, 4, 5], "bx_pass_ypg"] == 250).all()
    assert (out.loc[[3, 4, 5], "box_gp"] == 2).all()


def test_future_targets_keep_career_and_participation_without_advancing_them():
    out = player_snapshots(games(), players()).query("team == 'A'").set_index("game_id")
    assert np.isnan(out.loc[1, "cx_qb_gms"])
    assert out.loc[2, "cx_qb_gms"] == pytest.approx(1/30)
    assert (out.loc[[3, 4, 5], "cx_qb_gms"] == 2/30).all()
    assert (out.loc[[3, 4, 5], "av_qb_played_last"] == 1).all()
    assert out.loc[5, "cx_qb_vol"] == pytest.approx(500/5000)


def test_unreported_box_fields_remain_unknown_in_future_targets():
    raw = players().query("stat_type != 'YDS'")
    out = build_box_features(games(),raw=raw,per_game=True).query("team == 'A'").set_index("game_id")
    assert out.loc[5,"box_gp"] == 2
    assert np.isnan(out.loc[5,"bx_pass_ypg"])
    assert out.loc[5,"bx_comp_pct"] == pytest.approx(40/61)


def test_transferred_players_cannot_import_future_career_from_alphabetically_earlier_team():
    g = games()
    g.loc[0, ["home_team", "away_team"]] = ["Z", "A"]
    raw = players()
    out = player_snapshots(g, raw).set_index(["game_id", "team"])
    assert np.isnan(out.loc[(1, "Z"), "cx_qb_gms"])
    assert out.loc[(3, "A"), "cx_qb_gms"] == pytest.approx(2/30)


def test_pbp_same_week_and_far_future_use_all_prior_plays(tmp_path):
    rows = []
    for gid, yds in [(1, 5), (2, 15), (3, 9999)]:
        r = dict(game_id=gid, year=2026, week=1, season_type="regular", period=1,
                 pos_team="A", def_pos_team="Z", yards_gained=yds, down=1, distance=10,
                 passer_player_name="Quarterback")
        for k in ["rush", "pass", "sack", "int", "fumble_vec", "turnover", "punt", "penalty_flag", "touchdown", "fg_made"]:
            r[k] = k == "pass"
        rows.append(r)
    p = tmp_path / "plays.parquet"
    pd.DataFrame(rows).to_parquet(p)
    off, defense, qb = pbp_snapshots(games(), duckdb.connect(), p)
    a = off.query("team == 'A'").set_index("game_id")
    assert np.isnan(a.loc[1, "o_ypp"])
    assert a.loc[2, "o_ypp"] == 5/2  # existing +1 denominator convention
    assert (a.loc[[3, 4, 5], "o_plays_cum"] == 2).all()
    assert (a.loc[[3, 4, 5], "o_ypp"] == 20/3).all()
    q = qb.query("team == 'A'").set_index("game_id")
    assert q.loc[5, "qb_ypa"] == 10


def test_refresh_preserves_started_forecasts_and_updates_every_future_target():
    old = pd.DataFrame({"game_id": [1,2], "pred_margin": [3.,7.]})
    new = pd.DataFrame({"game_id": [1,2,3], "pred_margin": [99.,10.,4.]})
    schedule = pd.DataFrame({"game_id": [1,2,3], "start_date": ["2026-09-01", "2026-09-12", "2026-09-19"]})
    out = preserve_started_predictions(new,old,schedule,pd.Timestamp("2026-09-06",tz="UTC")).set_index("game_id")
    assert out.pred_margin.to_dict() == {1:3.,2:10.,3:4.}


def test_qb_leader_ties_do_not_depend_on_source_row_order(tmp_path):
    rows = []
    for gid, qb in [(1, "A"), (1, "B"), (2, "B")]:
        r = dict(game_id=gid, year=2026, week=1, season_type="regular", period=1,
                 pos_team="A", def_pos_team="Z", yards_gained=5, down=1, distance=10,
                 passer_player_name=qb)
        for k in ["rush", "pass", "sack", "int", "fumble_vec", "turnover", "punt", "penalty_flag", "touchdown", "fg_made"]:
            r[k] = k == "pass"
        rows.append(r)
    for order in (rows, rows[::-1]):
        path = tmp_path / "qb_ties.parquet"
        pd.DataFrame(order).to_parquet(path)
        _, _, qb = pbp_snapshots(games(), duckdb.connect(), path)
        assert qb.query("team == 'A' and game_id == 3").iloc[0].qb_changed == 1


def test_chronological_split_and_scaler_exclude_future_values():
    base = {"G_labeled": np.ones(4,bool), "G_season": np.array([2020,2021,2022,2023]),
            "G_gid": np.array([1,2,3,4]), "G_fbs_inv": np.ones(4,bool),
            "G_raw_scalars": np.array([[1.],[3.],[100.],[1000.]],np.float32)}
    for key in ["G_hidx", "G_aidx", "G_hconf", "G_aconf"]:
        base[key] = torch.tensor([1,1,2,3])
    base["G_opp"] = torch.tensor([[[1]],[[1]],[[2]],[[3]]])
    tr,va,te = partitions(base,2023,"chronological")
    assert tr.tolist() == [True,True,False,False]
    assert va.tolist() == [False,False,True,False]
    assert te.tolist() == [False,False,False,True]
    fitted = fit_inputs(base,tr)
    assert fitted["mu"].tolist() == [2.]
    assert fitted["G_hidx"].tolist() == [1,1,0,0]


def test_raw_current_pbp_uses_attribution_and_excludes_nullified_plays():
    plays = [{"gameId":1,"id":"10","playType":"Fumble Recovery (Opponent)",
              "playText":"Quarterback sacked and fumbled", "period":1,"offense":"A","defense":"B",
              "yardsGained":-5,"down":2,"distance":10,"ppa":100},
             {"gameId":1,"id":"11","playType":"Rush","playText":"Rush, penalty, no play",
              "period":1,"offense":"A","defense":"B","yardsGained":90,"down":1,"distance":10,"ppa":200}]
    stats = [{"gameId":1,"playId":"10","statType":"Sack Taken","athleteName":"QB"}]
    p = normalize(plays,stats,2026,1,"regular")
    assert "ppa" not in p
    assert p.loc[0,"pass"] and p.loc[0,"sack"] and p.loc[0,"turnover"]
    assert p.loc[0,"passer_player_name"] == "QB"
    assert not p.loc[1,"rush"]


def test_player_attribution_download_is_game_scoped_and_rejects_truncation():
    class Client:
        def get(self, endpoint, params):
            assert endpoint == "/plays/stats" and params == {"gameId": 123}
            return [{"gameId": 123, "ppa": 9}] * self.rows
    client = Client()
    client.rows = 10
    assert fetch_game_stats(client, 123) == [{"gameId": 123}] * 10
    client.rows = 2000
    with pytest.raises(RuntimeError, match="API cap"):
        fetch_game_stats(client, 123)
