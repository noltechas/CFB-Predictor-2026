from cfbpred.live_context import scoreboard_context


def test_possession_is_independent_of_field_side():
    row = dict(home_id="264", away_id="265", possession="264", possession_text="WSU 31", down=2, distance=6)
    assert scoreboard_context(row, "stamp")["possession"] == "home"
    assert scoreboard_context({**row, "possession": 265}, "stamp")["possession"] == "away"
    assert scoreboard_context({**row, "possession": None}, "stamp")["possession"] is None


def test_no_model_defaults_in_display_state():
    context = scoreboard_context(dict(home_id="264", away_id="265"), "stamp")
    assert all(context[k] is None for k in ("possession", "down", "distance", "field_position"))
    assert context["updated_at"] == "stamp"


def test_zero_distance_and_status_are_preserved():
    context = scoreboard_context(dict(home_id="264", away_id="265", down=1, distance=0,
                                     down_distance="1st & Goal", status_name="STATUS_HALFTIME"), "stamp")
    assert context["distance"] == 0
    assert context["down_distance"] == "1st & Goal"
    assert context["status_name"] == "STATUS_HALFTIME"
