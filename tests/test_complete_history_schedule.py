import pandas as pd

from cfbpred.complete_history_schedule import reconcile


def _raw(rows):
    return pd.DataFrame(rows)


def test_reconcile_adds_d1_rows_preserves_pinned_and_quarantines_duplicate(tmp_path):
    pinned = pd.DataFrame({
        "game_id": [10], "season": [2020], "week": [1], "season_type": ["regular"],
        "start_date": ["2020-09-01T00:00:00Z"], "neutral_site": [False],
        "conference_game": [False], "venue_id": [1], "home_team": ["A"],
        "home_conference": ["X"], "home_division": ["fbs"], "home_points": [1],
        "away_team": ["B"], "away_conference": ["Y"], "away_division": ["fbs"],
        "away_points": [0], "margin": [1], "total": [1], "kick": pd.to_datetime(["2020-09-01"], utc=True), "week_c": [1],
    })
    reference = pinned.assign(home_id=[1], away_id=[2])
    raw = _raw([
        {"id": 12, "season": 2020, "week": 1, "seasonType": "regular", "startDate": "2020-09-01T00:00:00Z", "neutralSite": False, "conferenceGame": False, "venueId": 1, "homeId": 1, "homeTeam": "A-old", "homeClassification": "fbs", "homeConference": "X", "homePoints": 1, "awayId": 2, "awayTeam": "B-old", "awayClassification": "fbs", "awayConference": "Y", "awayPoints": 0, "completed": True},
            {"id": 11, "season": 2020, "week": 2, "seasonType": "regular", "startDate": "2020-09-08T00:00:00Z", "neutralSite": False, "conferenceGame": False, "venueId": 1, "homeId": 1, "homeTeam": "A-old", "homeClassification": "fbs", "homeConference": "X", "homePoints": 2, "awayId": 3, "awayTeam": "C-old", "awayClassification": "ii", "awayConference": "Y", "awayPoints": 0, "completed": True},
        {"id": 13, "season": 2020, "week": 1, "seasonType": "postseason", "startDate": "2020-12-20T00:00:00Z", "neutralSite": False, "conferenceGame": False, "venueId": 1, "homeId": 4, "homeTeam": "D-old", "homeClassification": "fbs", "homeConference": "X", "homePoints": 3, "awayId": 5, "awayTeam": "E-old", "awayClassification": "ii", "awayConference": "Y", "awayPoints": 0, "completed": True},
    ])
    out = tmp_path / "out"
    pinned.to_parquet(tmp_path / "pinned.parquet")
    reference.to_parquet(tmp_path / "reference.parquet")
    raw.to_parquet(tmp_path / "raw.parquet")
    audit = reconcile(tmp_path / "pinned.parquet", tmp_path / "raw.parquet", out, reference_games=tmp_path / "reference.parquet", aliases={"A-old": "A", "C-old": "C"})
    assert audit["missing_d1_rows"] == 3
    assert audit["duplicate_fixture_count"] == 1
    result = pd.read_parquet(out / "source_schedule.parquet")
    assert result.game_id.tolist() == [10, 11, 13]
    assert result.loc[result.game_id.eq(13), "week_c"].item() == 21


def test_reconcile_keeps_fcs_and_spring_semantics(tmp_path):
    pinned = pd.DataFrame({"game_id": pd.Series([], dtype="int64"), "season": pd.Series([], dtype="int64"), "week": pd.Series([], dtype="int64"), "season_type": pd.Series([], dtype="object"), "start_date": pd.Series([], dtype="object"), "neutral_site": pd.Series([], dtype="bool"), "conference_game": pd.Series([], dtype="bool"), "venue_id": pd.Series([], dtype="float"), "home_team": pd.Series([], dtype="object"), "home_conference": pd.Series([], dtype="object"), "home_division": pd.Series([], dtype="object"), "home_points": pd.Series([], dtype="float"), "away_team": pd.Series([], dtype="object"), "away_conference": pd.Series([], dtype="object"), "away_division": pd.Series([], dtype="object"), "away_points": pd.Series([], dtype="float"), "margin": pd.Series([], dtype="float"), "total": pd.Series([], dtype="float"), "kick": pd.Series([], dtype="datetime64[ns, UTC]"), "week_c": pd.Series([], dtype="int64")})
    raw = _raw([{"id": 99, "season": 2020, "week": 1, "seasonType": "spring_regular", "startDate": "2021-03-01T00:00:00Z", "neutralSite": False, "conferenceGame": False, "venueId": 1, "homeId": 1, "homeTeam": "F", "homeClassification": "fcs", "homeConference": "X", "homePoints": 2, "awayId": 2, "awayTeam": "II", "awayClassification": "ii", "awayConference": "Y", "awayPoints": 1, "completed": True}])
    raw.to_parquet(tmp_path / "raw.parquet"); pinned.to_parquet(tmp_path / "pinned.parquet"); pinned.assign(home_id=pd.Series([], dtype="float"), away_id=pd.Series([], dtype="float")).to_parquet(tmp_path / "reference.parquet")
    reconcile(tmp_path / "pinned.parquet", tmp_path / "raw.parquet", tmp_path / "out", reference_games=tmp_path / "reference.parquet")
    result = pd.read_parquet(tmp_path / "out" / "source_schedule.parquet")
    assert result.iloc[0].season_type == "spring_regular"
    assert result.iloc[0].home_division == "fcs"


def test_same_fixture_different_kickoff_times_are_quarantined(tmp_path):
    columns = {"game_id": pd.Series([], dtype="int64"), "season": pd.Series([], dtype="int64"), "week": pd.Series([], dtype="int64"), "season_type": pd.Series([], dtype="object"), "start_date": pd.Series([], dtype="object"), "neutral_site": pd.Series([], dtype="bool"), "conference_game": pd.Series([], dtype="bool"), "venue_id": pd.Series([], dtype="float"), "home_team": pd.Series([], dtype="object"), "home_conference": pd.Series([], dtype="object"), "home_division": pd.Series([], dtype="object"), "home_points": pd.Series([], dtype="float"), "away_team": pd.Series([], dtype="object"), "away_conference": pd.Series([], dtype="object"), "away_division": pd.Series([], dtype="object"), "away_points": pd.Series([], dtype="float"), "margin": pd.Series([], dtype="float"), "total": pd.Series([], dtype="float"), "kick": pd.Series([], dtype="datetime64[ns, UTC]"), "week_c": pd.Series([], dtype="int64")}
    pinned = pd.DataFrame(columns)
    raw = _raw([])
    raw = pd.DataFrame([
        {"id": 21, "season": 2019, "week": 2, "seasonType": "regular", "startDate": "2019-09-07T18:00:00Z", "neutralSite": False, "conferenceGame": False, "venueId": 1, "homeId": 1, "homeTeam": "A", "homeClassification": "fbs", "homeConference": "X", "homePoints": 2, "awayId": 2, "awayTeam": "B", "awayClassification": "ii", "awayConference": "Y", "awayPoints": 1, "completed": True},
        {"id": 22, "season": 2019, "week": 2, "seasonType": "regular", "startDate": "2019-09-07T23:00:00Z", "neutralSite": False, "conferenceGame": False, "venueId": 1, "homeId": 1, "homeTeam": "A", "homeClassification": "fbs", "homeConference": "X", "homePoints": 2, "awayId": 2, "awayTeam": "B", "awayClassification": "ii", "awayConference": "Y", "awayPoints": 1, "completed": True},
    ])
    pinned.to_parquet(tmp_path / "pinned.parquet"); raw.to_parquet(tmp_path / "raw.parquet"); pinned.assign(home_id=pd.Series([], dtype="float"), away_id=pd.Series([], dtype="float")).to_parquet(tmp_path / "reference.parquet")
    audit = reconcile(tmp_path / "pinned.parquet", tmp_path / "raw.parquet", tmp_path / "out", reference_games=tmp_path / "reference.parquet")
    assert audit["same_fixture_key_quarantine_count"] == 2
    assert len(pd.read_parquet(tmp_path / "out" / "source_schedule.parquet")) == 0
