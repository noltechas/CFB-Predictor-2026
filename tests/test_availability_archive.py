"""Official conference archive parsing: layouts, reporting side, selection, identity."""
import pandas as pd

from cfbpred.availability_archive import (
    encode, parse_bigten_page, parse_cusa_pdf, parse_mac_page, parse_mwc_page, printed_match,
    printed_update_time, reporting_team, resolve_identities, select_final)


BIGTEN_2024 = """                  AVAILABILITY REPORT
                      Week #2 — Saturday, September 7
Kansas @ ILLINOIS
7 p.m. ET | FS1
OUT                                  QUESTIONABLE
 4 Daniel Brown                      17 Ethan Grunkemeyer
84 Carson Goda
"""

MAC = """Week 6
               AKRON vs. Central michigan
                         10.4.25
               OUT                                  questionablE
   2        Daymon David                     4           Justin Anderson
   7        Alex Adams
GetSomeMACtion.com // @MACSports
"""

CUSA = """                          FIU at WKU
                     Tue., Oct. 14 | 7 p.m. CT
FIU
#     Name                Pos.        Status
8     JuJu Lewis          WR          Out (Full Game)
2     Brian Blades II     DB          Questionable
WKU
#     Name                Pos.        Status
11    Tucker Parks        QB          Out (Full Game)
"""

MWC = """                                     GAME DAY REPORT
                                  Updated as of 10/18/25 at 10:30 A.M. MT
                   WYOMING                                                   AIR FORCE
Player                 Position    Status               Player                     Position   Status
Dante Drake            DT          Out (Season)         TJ Barker                  DB         Out (Full Game)
Evan Svoboda           TE          Questionable
"""


def game(home, away, home_conf="Big Ten", away_conf="Big Ten"):
    return pd.Series(dict(home_team=home, away_team=away, home_conference=home_conf, away_conference=away_conf))


def test_side_by_side_columns_keep_each_players_own_status():
    block = next(parse_bigten_page(BIGTEN_2024))
    assert [(r[0], r[2]) for r in block["rows"]] == [
        ("Daniel Brown", "out"), ("Ethan Grunkemeyer", "questionable"), ("Carson Goda", "out")]
    mac = next(parse_mac_page(MAC))
    assert (mac["team"], mac["printed"]) == ("AKRON", "10.4.25")
    assert dict((r[0], r[2]) for r in mac["rows"]) == {
        "Daymon David": "out", "Justin Anderson": "questionable", "Alex Adams": "out"}


def test_big_ten_reporter_is_the_team_printed_in_capitals_not_an_acronym():
    names = {"Illinois", "Kansas", "USC", "Michigan", "LSU", "Iowa", "UCLA"}
    kansas = dict(team="Kansas", opponent="ILLINOIS")
    assert reporting_team(kansas, game("Illinois", "Kansas", away_conf="Big 12"), names, "bigten_pdf") == "Illinois"
    # Both in capitals: the acronym is how USC is always written; MICHIGAN is shouted.
    shouted = dict(team="USC", opponent="MICHIGAN")
    assert reporting_team(shouted, game("Michigan", "USC"), names, "bigten_pdf") == "Michigan"
    # Neither shouted: the fixture's single Big Ten member reports.
    lsu = dict(team="LSU", opponent="USC")
    assert reporting_team(lsu, game("LSU", "USC", home_conf="SEC"), names, "bigten_pdf") == "USC"
    # Two members and no shouted side: refuse rather than guess.
    both = dict(team="UCLA", opponent="USC")
    assert reporting_team(both, game("USC", "UCLA"), names, "bigten_pdf") is None


def test_cusa_and_mountain_west_layouts():
    blocks = list(parse_cusa_pdf(CUSA))
    assert [(b["team"], b["opponent"], len(b["rows"])) for b in blocks] == [("FIU", "WKU", 2), ("WKU", "FIU", 1)]
    assert blocks[0]["printed"] == "Tue., Oct. 14"
    mwc = list(parse_mwc_page(MWC))
    assert [(b["team"], b["opponent"], len(b["rows"])) for b in mwc] == [
        ("WYOMING", "AIR FORCE", 2), ("AIR FORCE", "WYOMING", 1)]
    stamp = printed_update_time(mwc[0])
    assert stamp == pd.Timestamp("2025-10-18T16:30:00Z")


def test_printed_dates_must_match_the_game_and_its_season():
    assert printed_match("10.4.25", "2025-10-04", "mac_pdf") == "printed_date_exact"
    assert printed_match("10.4.25", "2025-10-11", "mac_pdf") is None
    assert printed_match("Week 1: Aug. 31 - Sept. 3", "2023-09-02", "bigten_pdf", 2023) == "printed_week_span"
    assert printed_match("Week 1: Aug. 31 - Sept. 3", "2023-09-09", "bigten_pdf", 2023) is None
    assert printed_match("Bowls: Dec. 23-Jan. 8", "2024-01-01", "bigten_pdf", 2023) == "printed_week_span"


def test_partial_game_and_unknown_wording_are_never_widened():
    assert encode("Out (Full Game)") == ("out", None)
    assert encode("Game Time Decision") == ("questionable", None)
    assert encode("Out (First Half)")[0] is None
    assert encode("OUT - 1ST HALF")[0] is None
    assert encode("Will play")[0] is None


def test_selection_is_per_team_game_so_stale_midweek_statuses_do_not_survive():
    base = dict(game_id=1, team="A", kind="availability")
    wednesday = [dict(base, name="X", value="questionable", source_url="u1", pdf_page=1,
                      pdf_created_at="2025-10-01T00:00:00+00:00"),
                 dict(base, name="Y", value="out", source_url="u1", pdf_page=1,
                      pdf_created_at="2025-10-01T00:00:00+00:00")]
    gameday = [dict(base, name="Y", value="out", source_url="u2", pdf_page=3,
                    pdf_created_at="2025-10-04T15:00:00+00:00")]
    kept = select_final(wednesday + gameday)
    assert [(r["name"], r["source_url"]) for r in kept] == [("Y", "u2")]


def test_identity_accepts_variant_spelling_only_when_unambiguous():
    catalog = pd.DataFrame([
        dict(season=2025, team="A", firstName="Matthew", lastName="Hibner", jersey=88.0, athlete_id=1, identity_uncertain=False),
        dict(season=2025, team="A", firstName="Chuck", lastName="Montgomery", jersey=3.0, athlete_id=2, identity_uncertain=False),
        dict(season=2025, team="A", firstName="Mike", lastName="Montgomery", jersey=9.0, athlete_id=3, identity_uncertain=False),
        dict(season=2025, team="A", firstName="Will", lastName="Johnson", jersey=2.0, athlete_id=4, identity_uncertain=False),
        dict(season=2025, team="A", firstName="Will", lastName="Johnson", jersey=40.0, athlete_id=5, identity_uncertain=False),
        dict(season=2025, team="A", firstName="Nana", lastName="Burris", jersey=0.0, athlete_id=6, identity_uncertain=False)])
    rows = [dict(name="Matt Hibner", team="A", season=2025, source_quote="88 Matt Hibner"),
            dict(name="Charles Montgomery", team="A", season=2025),
            dict(name="Will Johnson", team="A", season=2025, source_quote="40 Will Johnson"),
            dict(name="Will Johnson", team="A", season=2025),
            dict(name="Nazeviah Burris", team="A", season=2025),
            dict(name="Matt Hibner", team="A", season=2025, source_quote="12 Matt Hibner")]
    ok, bad = resolve_identities(rows, catalog)
    assert [(r["name"], r["athlete_id"], r["identity_method"]) for r in ok] == [
        ("Matthew Hibner", "1", "unique_surname_compatible_first_name"),
        ("Chuck Montgomery", "2", "unique_surname_compatible_first_name"),
        ("Will Johnson", "5", "exact_name_jersey_disambiguated")]
    # Ambiguous without a jersey, an incompatible first name, and a jersey that disagrees.
    assert len(bad) == 3


def test_publisher_versions_accepts_only_a_complete_open_graph_pair(tmp_path):
    from cfbpred.availability_backfill import publisher_versions
    both = tmp_path / "both.html"
    both.write_text('<link rel="canonical" href="https://x.example/a"/>'
                    '<meta property="article:published_time" content="2025-10-01T12:00:00+00:00"/>'
                    '<meta property="article:modified_time" content="2025-10-02T09:00:00+00:00"/>')
    one = tmp_path / "one.html"
    one.write_text('<link rel="canonical" href="https://x.example/b"/>'
                   '<meta property="article:published_time" content="2025-10-01T12:00:00+00:00"/>')
    (key, published, modified), = publisher_versions(both)
    assert key == ("x.example", "/a") and modified > published
    assert publisher_versions(one) == []
