"""Meaningful checks for margin scoring and experiment selection boundaries."""
import numpy as np
import pandas as pd
import pytest
import torch

from cfbpred.model import discrete_crps
from cfbpred.pregame_experiments import choose
from cfbpred.recent_form import recent_snapshots, matchup_features


def test_crps_is_absolute_error_for_a_point_forecast_including_outside_support():
    centres = torch.tensor([-2., 0., 2.])
    logits = torch.tensor([[-1000., 0., -1000.], [-1000., -1000., 0.]])
    score = discrete_crps(logits, torch.tensor([5., 10.]), centres, reduction="none")
    torch.testing.assert_close(score, torch.tensor([5., 8.]))


def test_crps_matches_pairwise_definition_for_unequal_bin_spacing():
    x = torch.tensor([-8., -2., 0., 1., 7.])
    logits = torch.tensor([[0.2, -0.5, 0.1, 1.2, 0.7]], requires_grad=True)
    p = logits.softmax(-1)[0]
    y = torch.tensor([3.])
    expected = (p * abs(x-y)).sum() - 0.5 * (p[:,None]*p[None,:]*abs(x[:,None]-x[None,:])).sum()
    score = discrete_crps(logits, y, x)
    torch.testing.assert_close(score, expected)
    score.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_crps_prefers_less_distant_misses():
    x = torch.tensor([0., 5., 30.])
    near = torch.tensor([[-1000., 0., -1000.]])
    far = torch.tensor([[-1000., -1000., 0.]])
    assert discrete_crps(near, torch.tensor([0.]), x) < discrete_crps(far, torch.tensor([0.]), x)


def test_candidate_selection_requires_improvement_and_probability_guardrails():
    scores = {"base": dict(margin_mae=13., wp_logloss=.5, brier=.17),
              "poor_probabilities": dict(margin_mae=12., wp_logloss=.52, brier=.17),
              "tiny_gain": dict(margin_mae=12.99, wp_logloss=.49, brier=.16),
              "good": dict(margin_mae=12.8, wp_logloss=.501, brier=.171)}
    assert choose(scores, "base") == "good"
    del scores["good"]
    assert choose(scores, "base") == "base"


def form_fixture():
    games = pd.DataFrame({"game_id":range(1,7), "season":[2026]*6, "week_c":[1,1,2,3,4,9],
        "home_team":["A"]*6,"away_team":["Z"]*6,
        "kick":pd.to_datetime(["2026-08-29","2026-09-05","2026-09-12","2026-09-19","2026-09-26","2026-10-31"],utc=True),
        "home_points":[10,10,10,10,np.nan,np.nan],"away_points":[0,0,0,0,np.nan,np.nan]})
    counts=pd.DataFrame([dict(game_id=gid,team="A",plays=1,pass_plays=1,pass_yards=y,
        pass_yards_observed=1,rush_plays=0,rush_yards=np.nan,rush_yards_observed=0,sacks=0,expl20=int(y>=20))
        for gid,y in [(1,10),(2,20),(4,40),(5,9999)]])
    return games,counts


def test_recent_windows_include_missing_games_and_repeat_for_all_future_targets():
    games,counts=form_fixture()
    out=recent_snapshots(games,counts).query("team=='A'").set_index("game_id")
    assert np.isnan(out.loc[1,"r3_pass_ypp"])
    assert out.loc[2,"r3_pass_ypp"]==10  # same-week previous game
    assert (out.loc[[5,6],"r3_pass_ypp"]==30).all()  # IDs 2,3,4; 3 is missing
    assert (out.loc[[5,6],"r3_games"]==2).all()
    assert out.loc[5,"r5_pass_ypp"]==pytest.approx(70/3)
    assert np.isnan(out.loc[5,"r3_rush_ypp"])
    assert out.loc[6,"r3_source_kick"]==games.loc[3,"kick"]


def test_recent_windows_reset_each_season_and_exclude_simultaneous_results():
    games,counts=form_fixture()
    games.loc[2,"kick"]=games.loc[1,"kick"]
    games.loc[5,"season"]=2027
    games.loc[5,"kick"]=pd.Timestamp("2027-09-01",tz="UTC")
    out=recent_snapshots(games,counts).query("team=='A'").set_index("game_id")
    assert out.loc[3,"r3_pass_ypp"]==10
    assert np.isnan(out.loc[6,"r3_pass_ypp"]) and out.loc[6,"r3_games"]==0


def test_matchup_comparisons_use_opponent_defense_and_preserve_unknowns():
    row={f"{side}_{direction}_r3_{stat}":value for side,value in [("h",2.),("a",5.)]
         for direction in ["o","d"] for stat in ["pass_ypp","rush_ypp","sack_rate","expl20_rate","pass_share"]}
    row["h_o_r3_pass_share"]=.6
    row["a_d_r3_rush_ypp"]=np.nan
    result=matchup_features(pd.DataFrame([row]),windows=(3,)).iloc[0]
    assert result.h_match_r3_pass_ypp_diff==-3
    assert result.h_match_r3_pass_style==3
    assert np.isnan(result.h_match_r3_rush_ypp_diff)
    assert np.isnan(result.h_match_r3_rush_style)
