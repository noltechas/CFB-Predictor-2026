"""Owner-authored publishing validation; no model or network dependencies."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfbpred.editorial_rankings import build_snapshot

TEAMS = [{"team_id": 194, "school": "Ohio State", "conference": "Big Ten"},
         {"team_id": 61, "school": "Georgia", "conference": "SEC"}]


def document(**extra):
    return {"season": 2026, "ranking_week": 2, "teams": ["Georgia", 194], **extra}


def test_resolves_names_and_ids_without_changing_author_order():
    result = build_snapshot(document(teams=[{"school": "georgia", "note": "My reason"}, 194]), TEAMS,
                            published_at="2026-09-08T20:00:00+00:00")
    assert [(e["team_id"], e["rank"]) for e in result["entries"]] == [(61, 1), (194, 2)]
    assert result["entries"][0]["note"] == "My reason"
    assert result["author"] == "Chas Nolte"
    assert result["published_at"] == "2026-09-08T20:00:00+00:00"


@pytest.mark.parametrize("extra", [
    {"teams": []}, {"teams": [61, "Georgia"]}, {"teams": [99999]}, {"teams": [True]},
    {"teams": [None]}, {"teams": [{"school": "Georgia", "note": 123}]},
    {"season": True}, {"season": 1999}, {"ranking_week": 41}, {"ranking_week": -1},
    {"author": " "}, {"title": None}, {"introduction": []},
])
def test_rejects_invalid_or_ambiguous_publication(extra):
    with pytest.raises(ValueError):
        build_snapshot(document(**extra), TEAMS)


def test_preseason_is_explicit_and_notes_optional():
    result = build_snapshot(document(ranking_week=0), TEAMS)
    assert result["ranking_week"] == 0
    assert result["entries"][0]["note"] is None
