import json
from pathlib import Path

import pytest

from cfbpred.explain_game import FLAG_KEYS, block_of, label_for, strip, value_format


def test_labels_expand_stat_shorthand_and_keep_side_out_of_the_name():
    assert label_for("h_o_ypp") == "Offense: yards per play"
    assert label_for("a_d_expl20_rate") == "Defense: explosive plays (20+ yards) rate allowed"
    assert label_for("h_bx_rush_ypc") == "Box score: rushing yards per carry"
    assert label_for("a_sos_talent") == "Opponents faced: mean talent"
    assert label_for("h_OL_wt") == "OL unit: mean weight"
    assert label_for("is_home_site") == "Home field"
    # the two sides of one field share a label; the side is carried separately
    assert label_for("h_ppg_for") == label_for("a_ppg_for")


def test_only_declared_flags_format_as_yes_no():
    # an undefeated team's win rate is exactly 1.0 and must not read as "yes"
    assert value_format("h_winpct") == "number"
    assert value_format("h_gp") == "number"
    assert value_format("is_home_site") == "flag"
    assert value_format("a_has_box") == "flag"
    assert value_format("h_qb_changed") == "flag"
    assert all(strip(f"h_{k}") == k for k in FLAG_KEYS if not k.startswith("is_"))


def test_quarterback_inputs_land_in_the_quarterback_block():
    for column in ("h_qb_ypa", "a_cx_qb_gms", "h_pre_qb_games", "a_retp_qb1", "h_QB_n"):
        assert block_of(column) == "Quarterback"
    assert block_of("h_o_ypp") == "Offense rates"
    assert block_of("a_match_r5_pass_ypp_diff") == "Recent form"


def _published():
    directory = Path("models/player_sequence_production/explanations")
    manifest = directory / "manifest.json"
    if not manifest.exists():
        pytest.skip("no explanations generated in this checkout")
    ids = json.loads(manifest.read_text())["game_ids"]
    return [json.loads((directory / f"{g}.json").read_text()) for g in ids[:5]]


def test_published_contributions_sum_to_the_prediction():
    """Shapley values are additive by construction, so the published rows must add up.

    The payload stores each contribution at four decimals; ~580 of them accumulate
    about a thousandth of a point of rounding, which is far below anything shown.
    """
    for payload in _published():
        total = sum(i["win_prob_pp"] for b in payload["blocks"] for i in b["inputs"])
        target = (payload["home_win_prob"] - payload["baseline"]["home_win_prob"]) * 100
        assert total == pytest.approx(target, abs=5e-3), payload["game_id"]
        for block in payload["blocks"]:
            assert block["win_prob_pp"] == pytest.approx(
                sum(i["win_prob_pp"] for i in block["inputs"]), abs=5e-3)


def test_player_effects_are_small_and_ranked_within_their_unit():
    for payload in _published():
        players = payload["players"]
        assert players, payload["game_id"]
        # replacement-level swaps are fractions of a point; anything larger means the
        # counterfactual regressed to deleting the roster slot
        assert max(abs(p["win_prob_pp"]) for p in players) < 2.0, payload["game_id"]
        for side in ("home", "away"):
            for unit in {p["position"] for p in players if p["side"] == side}:
                group = sorted((p for p in players if p["side"] == side and p["position"] == unit),
                               key=lambda p: p["rank_in_group"])
                assert [p["rank_in_group"] for p in group] == list(range(1, len(group) + 1))
                # depth order follows the encoder's learned share of the unit, starter first
                shares = [p["unit_share"] for p in group]
                assert shares == sorted(shares, reverse=True), (payload["game_id"], side, unit)
                # shares are stored at four decimals; a 23-man unit accumulates ~1e-3 of rounding
                assert sum(shares) == pytest.approx(1.0, abs=1e-2) or not any(shares)
