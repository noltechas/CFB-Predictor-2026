"""Live 2026 availability sources: Covers captures and the Sun Belt gist."""
import json

import pandas as pd

from cfbpred.covers_availability import load_facts as covers_facts, parse
from cfbpred.pull_player_context import archive
from cfbpred.pregame_experiments import save_json

PAGE = """<a id="Alpha"></a><section><div class="covers-CoversMatchups-teamName">
<a href="/x">Alpha<br><span>Birds</span></a></div>
<table><tbody><tr><td><span class='player-link'>C. Jones</span></td><td>QB</td>
<td><b>Out - Knee</b><br>(Wed, Sep 23)</td></tr>
<tr><td><span class='player-link'>D. Smith</span></td><td>WR</td>
<td><b>Questionable - Undisclosed</b><br>(Sat, Sep 12)</td></tr></tbody></table></section>"""


def fixtures():
    return pd.DataFrame([dict(game_id=7, season=2026, kick="2026-09-26T18:00:00Z", home_team="Alpha", away_team="Beta"),
                         dict(game_id=8, season=2026, kick="2026-09-24T18:00:00Z", home_team="Gamma", away_team="Alpha")])


def roster():
    return pd.DataFrame([dict(season=2026, team="Alpha", firstName="Cam", lastName="Jones", position="QB",
                              athlete_id=11, identity_uncertain=False),
                         dict(season=2026, team="Alpha", firstName="Dee", lastName="Smith", position="WR",
                              athlete_id=12, identity_uncertain=False)])


def capture(root, when):
    run = root / when.replace(":", "")
    meta = archive(PAGE.encode(), "https://www.covers.com/sport/football/ncaaf/injuries", run,
                   "covers_injuries.html", observed_at=when)
    save_json(run / "manifest.json", dict(run=str(run), sources=[meta]))


def test_covers_uses_only_game_week_entries_captured_before_kickoff(tmp_path):
    capture(tmp_path, "2026-09-25T12:00:00+00:00")
    facts, _ = covers_facts(fixtures(), roster(), root=tmp_path)
    # Game 7 only: game 8 kicked before the capture, and the Sep 12 entry is stale.
    assert [(f["game_id"], f["name"], f["value"]) for f in facts] == [(7, "Cam Jones", "out")]
    assert facts[0]["known_at"] == "2026-09-25T12:00:00+00:00"


def test_covers_never_overrides_an_official_report(tmp_path):
    capture(tmp_path, "2026-09-25T12:00:00+00:00")
    facts, _ = covers_facts(fixtures(), roster(), covered={(7, "Alpha")}, root=tmp_path)
    assert facts == []


def test_covers_parse_keeps_printed_wording():
    teams = parse_text(PAGE)
    assert teams["Alpha"][0]["status"] == "Out - Knee"


def parse_text(text, tmp=None):
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "p.html"
        path.write_text(text)
        return parse(path, pd.Timestamp("2026-09-25T12:00:00Z"))
