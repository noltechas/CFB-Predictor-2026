"""Official conference availability archives: collect, parse, normalize.

Covers the public, official report families that survive as retrievable
documents, 2023 onward:

* ``bigten_pdf``   2023-2024 weekly compiled PDFs (bigten.org/api/media/file)
* ``mac_pdf``      2025 weekly PDFs (sidearm S3 / getsomemaction.com)
* ``cusa_pdf``     2025 per-game PDFs (sidearm S3 / conferenceusa.com)
* ``mwc_pdf``      2025 per-team pages (storage.googleapis.com/themw-com)
* ``sunbelt_gist`` 2025 GitHub gist revisions (sunbeltsports.org hub feed)

The parsers read the publisher's own explicit status sections only. A player
who is absent from a report is never healthy, and a partial-game designation
is never widened into a full absence: those rows go to quarantine with their
raw text so a later encoding can pick them up.

Timing: PDF metadata is not publication proof, so a PDF's creation time is
used ONLY to ORDER competing files for the same game and to reject a file
demonstrably built after kickoff. The retained knowledge time follows the
user-authorized ``official_dated_report_assumed_pregame`` convention already
in this repository (T-30 minutes), and every fact records which printed date
carried it. Gist revisions are different in kind: GitHub's commit timestamp
is a third-party attestation that the bytes existed then, so those facts
carry the real revision time and are not tagged as assumed.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from cfbpred.config import ROOT
from cfbpred.player_timeline import STATUSES
from cfbpred.pregame_experiments import save_json, sha

ARCHIVE = ROOT / "data/raw/availability_archive"

# Explicit publisher wording -> the five encoded statuses.  "Game Time
# Decision" follows conference_availability.STATUS so a historical row and a
# live 2026 row from the same wording encode identically.
STATUS_TEXT = {
    "out": "out", "out (full game)": "out", "out (season)": "out",
    "out for the season": "out", "unavailable": "out",
    "doubtful": "doubtful",
    "questionable": "questionable", "game time decision": "questionable",
    "game-time decision": "questionable", "gtd": "questionable",
    "probable": "probable",
    "available": "available",
}
# Partial-game designations are real but are NOT a full absence.
PARTIAL = re.compile(r"1st half|first half|2nd half|second half|half\b|suspend", re.I)

ALIASES = {
    "fiu": "Florida International", "florida international": "Florida International",
    "wku": "Western Kentucky", "western kentucky": "Western Kentucky",
    "la tech": "Louisiana Tech", "louisiana tech": "Louisiana Tech",
    "mtsu": "Middle Tennessee", "middle tennessee": "Middle Tennessee",
    "jax state": "Jacksonville State", "jacksonville state": "Jacksonville State",
    "nm state": "New Mexico State", "new mexico state": "New Mexico State",
    "sam houston": "Sam Houston", "sam houston state": "Sam Houston",
    "kennesaw state": "Kennesaw State", "missouri state": "Missouri State",
    "ulm": "UL Monroe", "ul monroe": "UL Monroe", "louisiana-monroe": "UL Monroe",
    "louisiana monroe": "UL Monroe",
    "app state": "App State", "appalachian state": "App State",
    "ull": "Louisiana", "louisiana": "Louisiana", "louisiana lafayette": "Louisiana",
    "southern miss": "Southern Miss", "southern mississippi": "Southern Miss",
    "georgia southern": "Georgia Southern", "georgia state": "Georgia State",
    "old dominion": "Old Dominion", "odu": "Old Dominion",
    "james madison": "James Madison", "jmu": "James Madison",
    "coastal carolina": "Coastal Carolina", "arkansas state": "Arkansas State",
    "south alabama": "South Alabama", "texas state": "Texas State",
    "miami": "Miami (OH)", "miami (oh)": "Miami (OH)", "miami ohio": "Miami (OH)", "miami (ohio)": "Miami (OH)",
    "umass": "Massachusetts", "massachusetts": "Massachusetts",
    "niu": "Northern Illinois", "northern illinois": "Northern Illinois",
    "cmu": "Central Michigan", "central michigan": "Central Michigan",
    "emu": "Eastern Michigan", "eastern michigan": "Eastern Michigan",
    "wmu": "Western Michigan", "western michigan": "Western Michigan",
    "bgsu": "Bowling Green", "bowling green": "Bowling Green",
    "ball state": "Ball State", "kent state": "Kent State", "akron": "Akron",
    "buffalo": "Buffalo", "toledo": "Toledo", "ohio": "Ohio",
    "san jose state": "San José State", "san josé state": "San José State",
    "sjsu": "San José State", "san diego state": "San Diego State",
    "sdsu": "San Diego State", "boise state": "Boise State",
    "colorado state": "Colorado State", "csu": "Colorado State",
    "fresno state": "Fresno State", "air force": "Air Force",
    "hawaii": "Hawai'i", "hawai'i": "Hawai'i", "hawai`i": "Hawai'i",
    "unlv": "UNLV", "utah state": "Utah State", "new mexico": "New Mexico",
    "nevada": "Nevada", "wyoming": "Wyoming",
    "ucla": "UCLA", "usc": "USC", "lsu": "LSU", "smu": "SMU", "byu": "BYU",
    "tcu": "TCU", "ucf": "UCF", "utsa": "UTSA", "utep": "UTEP",
    "uconn": "UConn", "penn state": "Penn State", "ohio state": "Ohio State",
    "michigan state": "Michigan State", "michigan st": "Michigan State",
    "ole miss": "Ole Miss", "mississippi": "Ole Miss",
    "nc state": "NC State", "north carolina state": "NC State",
    "wake forest": "Wake Forest", "delaware": "Delaware", "liberty": "Liberty",
    "marshall": "Marshall", "troy": "Troy",
    "saint francis": "St. Francis (PA)", "st. francis": "St. Francis (PA)",
    "st francis": "St. Francis (PA)", "prairie view": "Prairie View A&M",
    "bryant": "Bryant", "iowa": "Iowa", "missouri": "Missouri",
}
# Wording that means a different program inside one conference's documents.
FAMILY_ALIASES = {"mac_pdf": {"miami": "Miami (OH)"}}


def norm_team(value, names, family=None):
    """Resolve publisher wording to a pinned-schedule team name, or None.

    ``names`` should be the document season's names: CFBD renames programs
    (UMass became Massachusetts), so an all-seasons set resolves wrongly.
    """
    text = re.sub(r"^\s*(?:#\d+\s+|no\.\s*\d+\s+)", "", str(value or "").strip())
    text = re.sub(r"\s+", " ", text).strip(" .,|")
    if not text:
        return None
    special = FAMILY_ALIASES.get(family, {}).get(text.lower())
    if special and special in names:
        return special
    if text in names:
        return text
    key = text.lower().replace("’", "'").replace("‘", "'").strip()
    if key in ALIASES and ALIASES[key] in names:
        return ALIASES[key]
    title = text.title().replace("'S", "'s")
    if title in names:
        return title
    lowered = {n.lower(): n for n in names}
    if key in lowered:
        return lowered[key]
    return None


def clean_name(value):
    """Strip report annotations that are not part of the person's name."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"\((?:season|full game|1st half|2nd half|[^)]*)\)\s*$", "", text).strip()
    text = re.sub(r"[*†#]+$", "", text).strip()
    return text


def encode(raw):
    """Explicit status wording only; partial-game text is never widened."""
    text = re.sub(r"[\x00-\x1f\x7f]", "", str(raw or ""))
    text = re.sub(r"\s+", " ", text).strip()
    key = text.lower().strip(" .")
    if key in STATUS_TEXT:
        value = STATUS_TEXT[key]
        if value == "out" and PARTIAL.search(text):
            return None, "partial_game_designation_requires_distinct_encoding"
        return value, None
    if PARTIAL.search(text):
        return None, "partial_game_designation_requires_distinct_encoding"
    return None, "status_requires_explicit_encoding"


def pdf_text(path, first=None, last=None):
    cmd = ["pdftotext", "-layout"]
    if first:
        cmd += ["-f", str(first), "-l", str(last or first)]
    return subprocess.run(cmd + [str(path), "-"], capture_output=True, text=True).stdout


def pdf_created(path):
    """The file's own build time, with its UTC offset (pdfinfo -isodates)."""
    out = subprocess.run(["pdfinfo", "-isodates", str(path)], capture_output=True, text=True).stdout
    match = re.search(r"CreationDate:\s+(\S+)", out)
    if not match:
        return None
    try:
        stamp = pd.Timestamp(match.group(1))
    except (ValueError, TypeError):
        return None
    return stamp.tz_convert("UTC") if stamp.tzinfo else None


def pdf_pages(path):
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True).stdout
    match = re.search(r"Pages:\s+(\d+)", out)
    return int(match.group(1)) if match else 0


# --------------------------------------------------------------------------
# Parsers.  Each yields report blocks:
#   dict(team=, opponent=, page=, printed=<raw date text>, rows=[(name, pos, raw_status, line)])
# --------------------------------------------------------------------------

HEAD_SPLIT = re.compile(r"\s+(?:vs\.?|at|@)\s+", re.I)
SECTION_WORDS = ("OUT", "QUESTIONABLE", "DOUBTFUL", "PROBABLE", "AVAILABLE")
SECTION_RE = re.compile(r"\b(" + "|".join(SECTION_WORDS) + r")\b", re.I)


def _split_matchup(line):
    parts = HEAD_SPLIT.split(line.strip(), maxsplit=1)
    if len(parts) != 2:
        return None
    left, right = (re.sub(r"\s*\|.*$", "", p).strip(" .,") for p in parts)
    return left, right


def column_rows(lines, entry_re, skip=None):
    """Read status sections laid out as side-by-side columns.

    Both the Big Ten and the MAC print OUT beside QUESTIONABLE, so one text
    line can hold a player from each list. The header line's own character
    offsets establish the column boundaries; reading the line as a single list
    would attach the wrong status to half the roster.
    """
    columns, rows = None, []
    for raw in lines:
        line = raw.expandtabs(8)
        found = [(m.start(), m.end(), m.group(1).lower()) for m in SECTION_RE.finditer(line)]
        if found and re.sub(SECTION_RE, "", line).strip(" \t|-.:") == "":
            columns = sorted(found)
            continue
        if columns is None or (skip and skip.search(line)):
            continue
        # Assign each run of text to the nearest column, because an entry can
        # sit well to the left of the header it belongs under.
        centres = [(start + end) / 2 for start, end, _ in columns]
        buckets = {index: [] for index in range(len(columns))}
        for chunk in re.finditer(r"\S(?:.*?\S)?(?=\s{3,}|$)", line):
            text = chunk.group(0).strip()
            if not text:
                continue
            nearest = min(range(len(centres)), key=lambda i: abs(chunk.start() - centres[i]))
            buckets[nearest].append(text)
        for index, (_, _, section) in enumerate(columns):
            flat = re.sub(r"\s+", " ", " ".join(buckets[index])).strip()
            if not flat or flat.lower() in {"none", "n/a", "-"}:
                continue
            entry = entry_re.match(flat)
            if entry:
                rows.append((clean_name(entry.group(entry.re.groups)), None, section, flat))
    return rows


def _capitals(side):
    return bool(re.search(r"[A-Za-z]", side)) and side.upper() == side


ACRONYMS = {"usc", "lsu", "ucla", "fiu", "ucf", "tcu", "byu", "smu", "utep", "utsa", "unlv", "uab",
            "uconn", "umass", "wku", "mtsu", "niu", "ulm", "jmu", "odu", "sjsu", "sdsu", "bgsu",
            "emu", "cmu", "wmu", "fau", "usf", "vmi", "ecu", "unc", "ncsu", "fsu", "osu", "psu"}


def _naturally_capitals(side, names):
    """An acronym that is simply how the program is written (USC, LSU, FIU)."""
    return side.replace(" ", "").replace(".", "").lower() in ACRONYMS


def reporting_team(block, game, names, family):
    """Which program a Big Ten page reports.

    The page prints its own team in capitals. If only one name is in capitals
    it is the reporter. If both are (USC @ MICHIGAN), the one that is not
    simply an acronym wins. Otherwise (LSU vs. USC, UCLA vs. USC) the Big Ten
    member of the fixture reports, and two members means the page is refused.
    """
    if family != "bigten_pdf":
        return norm_team(block["team"], names, family)
    left, right = block["team"], block.get("opponent") or ""
    caps = [_capitals(left), _capitals(right)]
    loud = [c and not _naturally_capitals(x, names) for c, x in zip(caps, (left, right))]
    pick = None
    if caps.count(True) == 1:
        pick = left if caps[0] else right
    elif loud.count(True) == 1:
        pick = left if loud[0] else right
    if pick is not None:
        return norm_team(pick, names, family)
    sides = {norm_team(left, names, family), norm_team(right, names, family)}
    members = [t for t, conf in ((game.home_team, game.home_conference), (game.away_team, game.away_conference))
               if conf == "Big Ten" and t in sides]
    return members[0] if len(members) == 1 else None


def parse_bigten_page(text):
    """One Big Ten page = one reporting team; the report's team is ALL CAPS."""
    lines = [l.rstrip() for l in text.split("\n") if l.strip()]
    team = opponent = printed = None
    for line in lines[:8]:
        pair = _split_matchup(re.sub(r"\t", " ", line))
        if not pair:
            continue
        left, right = pair
        if not re.search(r"[A-Za-z]{3}", left) or not re.search(r"[A-Za-z]{3}", right):
            continue
        team, opponent = left, right   # reporting_team() decides which side reports
        break
    printed = None
    for line in lines[:8]:
        flat = re.sub(r"\s+", " ", line.replace("\t", " ")).strip()
        if re.match(r"^(week|postseason|bowls?|big ten championship)", flat, re.I):
            printed = flat
            break
    rows = column_rows(lines, re.compile(r"^(\d{1,3})\s+(.+)$"),
                       skip=re.compile(r"AVAILABILITY REPORT|BIG TEN", re.I))
    if team:
        # An empty list is still a filed report; the caller records it as coverage.
        yield dict(team=team, opponent=opponent, page=None, printed=printed, rows=rows)


def parse_mac_page(text):
    """One MAC page = one reporting team, named FIRST in the matchup line.

    The MAC prints OUT and QUESTIONABLE as side-by-side columns, so a single
    text line can carry one player from each. Split on the column offset the
    headers themselves establish rather than reading the line as one list.
    """
    lines = [l.rstrip() for l in text.split("\n") if l.strip()]
    team = opponent = printed = None
    for line in lines[:12]:
        if re.search(r"availability report", line, re.I):
            continue
        pair = _split_matchup(line)
        if pair and all(re.search(r"[A-Za-z]{3}", p) for p in pair):
            team, opponent = pair
            break
    for line in lines[:14]:
        match = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(\d{2})\b", line)
        if match:
            printed = match.group(0)
            break
    rows = column_rows(lines, re.compile(r"^(\d{1,3})\s+(.+)$"),
                       skip=re.compile(r"GetSomeMACtion|AVAILABILITY REPORT", re.I))
    if team:
        # An empty list is still a filed report; the caller records it as coverage.
        yield dict(team=team, opponent=opponent, page=None, printed=printed, rows=rows)


CUSA_ROW = re.compile(r"^\s*(\d{1,3})\s+(.+?)\s{2,}([A-Za-z/]{1,5})\s{2,}(.+?)\s*$")


def parse_cusa_pdf(text):
    """A CUSA report carries both teams; sections start at a bare team name."""
    lines = [l.rstrip() for l in text.split("\n") if l.strip()]
    header = None
    printed = None
    for line in lines[:6]:
        pair = _split_matchup(line)
        if pair and all(re.search(r"[A-Za-z]{3}", p) for p in pair):
            header = pair
            break
    for line in lines[:8]:
        match = re.search(r"([A-Z][a-z]{2,3}\.?,?\s+[A-Z][a-z]{2,8}\.?\s+\d{1,2})", line)
        if match:
            printed = match.group(1)
            break
    # Early-season layout: "School:  X    Opponent:  Y"
    if header is None:
        for line in lines[:8]:
            m = re.search(r"School:\s*(.+?)\s{2,}Opponent:\s*(.+?)\s*$", line)
            if m:
                header = (m.group(1).strip(), m.group(2).strip())
                break
    if header is None:
        return
    blocks, current = [], None
    for line in lines:
        flat = re.sub(r"\s+", " ", line).strip()
        if re.match(r"^#?\s*Name\b", flat, re.I) or re.match(r"^#\s+Name", flat):
            continue
        entry = CUSA_ROW.match(line)
        if entry and current:
            current["rows"].append((clean_name(entry.group(2)), entry.group(3),
                                    entry.group(4).strip(), flat))
            continue
        candidate = flat.strip(" .:")
        if candidate and len(candidate) < 40 and not entry and not re.search(r"\d{1,2}\s*[apAP]\.?m", candidate):
            if candidate.lower() in {h.lower() for h in header} or candidate in header:
                current = dict(team=candidate, opponent=None, page=None,
                               printed=printed, rows=[])
                blocks.append(current)
    for block in blocks:
        other = [h for h in header if h.lower() != block["team"].lower()]
        block["opponent"] = other[0] if other else None
        yield block


MWC_ROW = re.compile(r"^(.+?)\s{2,}([A-Za-z/]{1,5})\s{2,}(.+?)$")


def _mwc_side(segment):
    flat = segment.strip()
    entry = MWC_ROW.match(flat)
    if entry:
        return clean_name(entry.group(1)), entry.group(2), entry.group(3).strip(), re.sub(r"\s+", " ", flat)
    return None


def parse_mwc_page(text):
    """Mountain West report blocks: one team, or two teams side by side.

    Each block prints its own "Updated as of M/D/YY at H:MM A.M. MT" line,
    which is the publisher's own statement of when that report was current.
    Early-season pages name one team and no opponent; later pages print both
    teams, each with its own Player / Position / Status table.
    """
    lines = [l.rstrip().expandtabs(8) for l in text.split("\n") if l.strip()]
    blocks, current = [], None
    for line in lines:
        flat = re.sub(r"\s+", " ", line).strip()
        update = re.search(r"Updated as of\s+(\d{1,2}/\d{1,2}/\d{2})(?:\s+at\s+([0-9:]+\s*[AP]\.?M\.?)\s*([A-Z]{1,3}T)?)?", flat, re.I)
        if re.fullmatch(r"(?:[A-Z]+ )*(?:REPORT|UPDATE)", flat):
            current = dict(printed=None, clock=None, zone=None, teams=None, split=None, rows=[[], []])
            blocks.append(current)
            continue
        if current is None:
            current = dict(printed=None, clock=None, zone=None, teams=None, split=None, rows=[[], []])
            blocks.append(current)
        if update:
            current.update(printed=update.group(1), clock=update.group(2), zone=update.group(3))
            continue
        players = [m.start() for m in re.finditer(r"\bPlayer\b", line)]
        if players and re.search(r"\bStatus\b", line):
            current["split"] = players[1] if len(players) > 1 else None
            continue
        if current["split"] is None and current["teams"] is None and not current["rows"][0]:
            pass
        if flat.upper() == flat and re.search(r"[A-Z]{3}", flat) and not MWC_ROW.match(flat.strip()):
            names = [c.group(0).strip() for c in re.finditer(r"\S(?:.*?\S)?(?=\s{3,}|$)", line.strip())]
            if 1 <= len(names) <= 2 and not re.search(r"REPORT|UPDATED|MOUNTAIN WEST", flat):
                current["teams"] = names
            continue
        if not current["teams"]:
            continue
        split = current["split"]
        halves = [line[:split], line[split:]] if split else [line, ""]
        for side, segment in enumerate(halves):
            parsed = _mwc_side(segment)
            if parsed:
                current["rows"][side].append(parsed)
    for block in blocks:
        teams = block["teams"] or []
        for side, team in enumerate(teams[:2]):
            rows = block["rows"][side]
            if True:
                other = teams[1 - side] if len(teams) == 2 else None
                yield dict(team=team, opponent=other, page=None, printed=block["printed"],
                           clock=block["clock"], zone=block["zone"], rows=rows)


PDF_PARSERS = {"bigten_pdf": parse_bigten_page, "mac_pdf": parse_mac_page,
               "mwc_pdf": parse_mwc_page}


def pdf_blocks(path, family):
    """Yield (page_number, block) for a document, page by page."""
    if family == "cusa_pdf":
        text = pdf_text(path)
        for block in parse_cusa_pdf(text):
            yield 1, block
        return
    parser = PDF_PARSERS[family]
    for page in range(1, pdf_pages(path) + 1):
        for block in parser(pdf_text(path, page, page)):
            yield page, block


# --------------------------------------------------------------------------
# Game matching and fact emission
# --------------------------------------------------------------------------

ZONES = {"ET": "America/New_York", "CT": "America/Chicago", "MT": "America/Denver",
         "PT": "America/Los_Angeles", "HT": "Pacific/Honolulu", "HST": "Pacific/Honolulu"}


def printed_update_time(block):
    """The report's own "Updated as of" time, or None when it is not printed."""
    if not block.get("printed") or not block.get("clock"):
        return None
    zone = ZONES.get((block.get("zone") or "MT").upper())
    if zone is None:
        return None
    clock = block["clock"].upper().replace(".", "").replace(" ", "")
    try:
        local = pd.to_datetime(f"{block['printed']} {clock}", format="%m/%d/%y %I:%M%p")
    except ValueError:
        return None
    return local.tz_localize(zone).tz_convert("UTC")


def fixture_index(schedule):
    games = schedule.copy()
    games["kick"] = pd.to_datetime(games.kick, utc=True)
    games["local"] = games.kick.dt.tz_convert("America/Chicago").dt.strftime("%Y-%m-%d")
    names = set(games.home_team) | set(games.away_team)
    names_by_season = {int(season): set(frame.home_team) | set(frame.away_team)
                       for season, frame in games.groupby("season")}
    by_pair, by_team = {}, {}
    for game in games.itertuples():
        by_pair.setdefault((frozenset((game.home_team, game.away_team)), game.local), []).append(game)
        by_team.setdefault((game.home_team, game.local), []).append(game)
        by_team.setdefault((game.away_team, game.local), []).append(game)
    return names, by_pair, by_team, names_by_season


MONTHS = {m[:3].lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}


def printed_match(printed, kick_local, family, season=None):
    """How the document's own printed date ties it to this game, or None.

    Returns the rule that fired so every retained fact can say why the report
    belongs to the game it was attached to.
    """
    if not printed:
        return None
    day = pd.Timestamp(kick_local)
    if family == "mac_pdf":
        return "printed_date_exact" if printed == f"{day.month}.{day.day}.{day.year % 100:02d}" else None
    if family == "mwc_pdf":
        forms = {f"{day.month}/{day.day}/{day.year % 100:02d}",
                 f"{day.month:02d}/{day.day:02d}/{day.year % 100:02d}"}
        return "printed_date_exact" if printed in forms else None
    text = re.sub(r"^(week\s*#?\s*\d+|postseason|bowls?)\s*[:—-]*\s*", "",
                  printed.lower().replace(".", ""), count=1).strip()
    month = day.strftime("%B").lower()
    if month[:3] in text and re.search(rf"\b{day.day}\b", text):
        return "printed_date_exact"
    if family != "bigten_pdf":
        return None
    # 2023 weekly files print a span: "Week 1: Aug 31 - Sept 3".
    span = [(name[:3], int(number)) for name, number in
            re.findall(r"([a-z]{3,9})\s*(\d{1,2})", text) if name[:3] in MONTHS]
    if len(span) >= 2:
        bounds = []
        base = int(season) if season is not None else (day.year if day.month >= 8 else day.year - 1)
        for stem, number in span[:2]:
            number_month = MONTHS[stem]
            bounds.append(pd.Timestamp(year=base if number_month >= 8 else base + 1,
                                       month=number_month, day=number))
        if bounds[0] <= day <= bounds[1]:
            return "printed_week_span"
    return None


def document_facts(path, url, family, index, receipt, season_hint=None):
    """Turn one archived document into candidate facts plus quarantine rows."""
    names, by_pair, by_team, names_by_season = index
    if season_hint is not None:
        names = names_by_season.get(int(season_hint), names)
    digest = sha(path)
    created = pdf_created(path)
    facts, issues, seen = [], [], []
    for page, block in pdf_blocks(path, family):
        team = norm_team(block["team"], names, family)
        opponent = norm_team(block["opponent"], names, family) if block["opponent"] else None
        if not team:
            issues.append(dict(reason="unresolved_team", raw=block["team"], url=url, page=page))
            continue
        candidates = []
        if opponent:
            for key, games in by_pair.items():
                if key[0] == frozenset((team, opponent)):
                    candidates += games
        else:
            for (name, _), games in by_team.items():
                if name == team:
                    candidates += games
        if season_hint is not None:
            candidates = [g for g in candidates if int(g.season) == int(season_hint)]
        scored = [(g, printed_match(block["printed"], g.local, family, g.season)) for g in candidates]
        matched = [(g, rule) for g, rule in scored if rule]
        if not matched and opponent and len({g.game_id for g in candidates}) == 1:
            # A compiled bowl or championship page names no usable date, but
            # these two programs meet exactly once in the pinned season.
            matched = [(candidates[0], "unique_season_matchup")]
        if len(matched) != 1:
            issues.append(dict(reason="no_unique_printed_date_game_match", team=team,
                               opponent=block["opponent"], printed=block["printed"],
                               candidates=len(candidates), url=url, page=page))
            continue
        game, rule = matched[0]
        reporter = reporting_team(block, game, names, family)
        if reporter is None:
            issues.append(dict(reason="reporting_side_ambiguous", url=url, page=page,
                               team=block["team"], opponent=block["opponent"], game_id=int(game.game_id)))
            continue
        team = reporter
        compiled_later = created is not None and created >= game.kick
        stated = printed_update_time(block)
        if stated is not None and stated >= game.kick:
            issues.append(dict(reason="printed_update_after_kickoff", url=url, page=page,
                               stated=stated.isoformat(), game_id=int(game.game_id)))
            continue
        if compiled_later and family != "bigten_pdf" and stated is None:
            # A game-day document built after this kickoff cannot be this
            # game's pregame report; an earlier document from the same week is.
            issues.append(dict(reason="document_built_after_kickoff", url=url, page=page,
                               created=created.isoformat(), game_id=int(game.game_id)))
            continue
        seen.append(dict(game_id=int(game.game_id), team=team, season=int(game.season), family=family,
                         url=url, page=page, rows=len(block["rows"])))
        policy = ("official_weekly_compilation_assumed_pregame" if family == "bigten_pdf"
                  else "official_report_printed_update_time" if stated is not None
                  else "official_report_printed_date")
        known = game.kick - pd.Timedelta(minutes=30)
        for name, position, raw_status, line in block["rows"]:
            value, reason = encode(raw_status)
            row = dict(name=name, team=team, season=int(game.season), game_id=int(game.game_id),
                       kind="availability", value=value, raw_status=raw_status,
                       known_at=known.isoformat(), effective_at=known.isoformat(),
                       valid_until=(game.kick + pd.Timedelta(days=1)).isoformat(),
                       source_path=str(path.resolve()), source_sha256=digest, source_url=url,
                       source_kind=family, pdf_page=page, source_quote=line.strip(),
                       report_date=game.local, printed_report_date=block["printed"],
                       game_match_rule=rule,
                       printed_update_at=stated.isoformat() if stated is not None else None,
                       pdf_created_at=created.isoformat() if created is not None else None,
                       observed_at=receipt["observed_at"], timing_assumed=True,
                       timing_policy=policy, compiled_after_kickoff=bool(compiled_later),
                       timing_basis=(
                           "User-authorized assumption: the Big Ten's official game-day report for "
                           "this printed week and matchup was public at T-30. The weekly file is a "
                           "later compilation of those reports; its build time is provenance only"
                           if family == "bigten_pdf" else
                           "User-authorized assumption: available at T-30. The report prints its own "
                           "update time, which precedes kickoff; the cumulative weekly file's build "
                           "time is provenance only"
                           if stated is not None else
                           "User-authorized assumption: the official report for the printed game "
                           "date is available at T-30; the document carries no publication "
                           "timestamp, and its own build time precedes kickoff"))
            if not name or value is None:
                issues.append({**row, "reason": reason or "empty_name"})
                continue
            if position:
                row["reported_position"] = position
            facts.append(row)
    return facts, issues, seen


def select_final(facts):
    """Keep only the LAST pre-kickoff report for each team-game.

    A conference files several reports per game (initial, updates, game day).
    Selection is per team-game, never per player: a player who appears
    Wednesday and is gone from the game-day list must not keep Wednesday's
    status, because every retained row carries the same T-30 knowledge time.
    Competing documents are ordered by the build time each file records, which
    is evidence of sequence though not of publication.
    """
    latest = {}
    for row in facts:
        key = (row["game_id"], row["team"])
        order = (row.get("pdf_created_at") or "", row["source_url"], row.get("pdf_page") or 0)
        document = (row["source_url"], row.get("pdf_page"))
        if key not in latest or order[:2] > latest[key][0][:2]:
            latest[key] = (order, {document})
        elif order[:2] == latest[key][0][:2]:
            latest[key][1].add(document)
    kept = [row for row in facts
            if (row["source_url"], row.get("pdf_page")) in latest[(row["game_id"], row["team"])][1]]
    # One status per player within the chosen document.
    unique = {}
    for row in kept:
        unique.setdefault((row["game_id"], row["team"], row["name"]), row)
    return list(unique.values())


def gist_facts(index_path, versions_dir, schedule_index):
    """Sun Belt: each side's list from its last pre-kickoff gist revision.

    Selection is per game side, as with the PDFs, so a status is never carried
    past a later revision that dropped the player. A revision whose list for a
    side is only the empty placeholder the feed uses before a school files is
    not treated as a report.
    """
    _, by_pair, _, names_by_season = schedule_index
    index = json.loads(Path(index_path).read_text())
    chosen, issues = {}, []
    for entry in sorted(index, key=lambda e: e["committed_at"]):
        body = Path(versions_dir) / f"{entry['sha256']}.json"
        if not body.exists():
            continue
        committed = pd.Timestamp(entry["committed_at"])
        try:
            payload = json.loads(body.read_text())
        except ValueError:
            continue
        for game in payload.get("games", []):
            day = game.get("dateISO") or ""
            season = int(day[:4]) if day[5:7] >= "06" else int(day[:4] or 0) - 1
            names = names_by_season.get(season, set())
            home = norm_team((game.get("home") or {}).get("name"), names)
            away = norm_team((game.get("away") or {}).get("name"), names)
            day = game.get("dateISO")
            if not home or not away or not day:
                issues.append(dict(reason="unresolved_team_or_date", home=(game.get("home") or {}).get("name"),
                                   away=(game.get("away") or {}).get("name"), date=day))
                continue
            matched = by_pair.get((frozenset((home, away)), day), [])
            if len(matched) != 1:
                issues.append(dict(reason="no_unique_fixture", home=home, away=away, date=day))
                continue
            fixture = matched[0]
            if committed >= fixture.kick:
                continue
            for side, team in (("home", home), ("away", away)):
                players = [p for p in (game.get(side) or {}).get("players", []) if clean_name(p.get("name"))]
                if not players:
                    continue
                chosen[(int(fixture.game_id), team)] = dict(
                    committed=committed, fixture=fixture, players=players, entry=entry,
                    notes=game.get("notes"), updated=game.get("updatedAtCT"))
    facts = []
    for (game_id, team), pick in chosen.items():
        fixture, entry = pick["fixture"], pick["entry"]
        seen = set()
        for player in pick["players"]:
            name = clean_name(player.get("name"))
            if name in seen:
                continue
            seen.add(name)
            value, reason = encode(player.get("status"))
            row = dict(name=name, team=team, season=int(fixture.season), game_id=game_id,
                       kind="availability", value=value, raw_status=player.get("status"),
                       known_at=pick["committed"].isoformat(), effective_at=pick["committed"].isoformat(),
                       valid_until=(fixture.kick + pd.Timedelta(days=1)).isoformat(),
                       source_path=str((Path(versions_dir) / f"{entry['sha256']}.json").resolve()),
                       source_sha256=entry["sha256"], source_url=entry["url"],
                       source_kind="sunbelt_gist", revision_version=entry["version"],
                       revision_committed_at=entry["committed_at"],
                       revision_index=str(Path(index_path).resolve()),
                       report_kind=pick["notes"], feed_updated_at=pick["updated"],
                       timing_assumed=False, timing_policy="third_party_revision_timestamp",
                       timing_basis=("GitHub commit timestamp for the exact retained bytes; the "
                                     "publisher's own feed served this revision before kickoff"))
            if player.get("position"):
                row["reported_position"] = player["position"]
            if value is None:
                issues.append({**row, "reason": reason})
                continue
            facts.append(row)
    return facts, issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schedule = pd.read_parquet(args.schedule)
    index = fixture_index(schedule)
    manifest = json.loads(args.manifest.read_text())
    facts, issues, reports = [], [], []
    for item in manifest["documents"]:
        path = Path(item["path"])
        receipt = json.loads(path.with_suffix(path.suffix + ".source.json").read_text())
        rows, bad, found = document_facts(path, item["url"], item["family"], index, receipt,
                                          season_hint=item.get("season"))
        facts += rows
        issues += bad
        reports += found
    parsed = len(facts)
    facts = select_final(facts)
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "facts.json", facts)
    save_json(args.output / "quarantine.json", issues)
    save_json(args.output / "reports_seen.json", reports)
    print(json.dumps(dict(parsed_rows=parsed, facts=len(facts), quarantined=len(issues)), indent=2))


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# Identity: attach each reported name to exactly one roster person, or none.
# --------------------------------------------------------------------------

NICKNAMES = [
    {"william", "will", "bill", "billy", "willie"}, {"robert", "rob", "bob", "bobby", "robbie"},
    {"charles", "chuck", "charlie"}, {"michael", "mike"}, {"christopher", "chris"},
    {"joshua", "josh"}, {"matthew", "matt"}, {"nicholas", "nick"}, {"anthony", "tony"},
    {"cameron", "cam"}, {"alexander", "alex"}, {"benjamin", "ben"}, {"daniel", "dan", "danny"},
    {"james", "jim", "jimmy", "jamie"}, {"joseph", "joe", "joey"}, {"thomas", "tom", "tommy"},
    {"zachary", "zach", "zack"}, {"jonathan", "jon"}, {"nathaniel", "nate", "nathan"},
    {"samuel", "sam", "sammy"}, {"timothy", "tim"}, {"edward", "ed", "eddie"},
    {"richard", "rich", "rick", "ricky"}, {"andrew", "drew", "andy"}, {"gregory", "greg"},
    {"jacob", "jake"}, {"kenneth", "ken", "kenny"}, {"patrick", "pat"}, {"jeffrey", "jeff"},
    {"steven", "stephen", "steve"}, {"theodore", "ted", "teddy", "theo"}, {"frederick", "fred"},
    {"gerald", "jerry"}, {"lawrence", "larry"}, {"ronald", "ron", "ronnie"},
    {"donald", "don", "donnie"}, {"raymond", "ray"}, {"douglas", "doug"}, {"peter", "pete"},
    {"phillip", "philip", "phil"}, {"reginald", "reggie"}, {"maximilian", "max", "maxwell"},
    {"jackson", "jack"}, {"david", "dave"}, {"abraham", "abe"}, {"elijah", "eli"},
    {"isaiah", "ike"}, {"jeremiah", "jj"}, {"dominic", "dom"}, {"mitchell", "mitch"},
    {"harrison", "harry"}, {"nicolas", "nico", "nick"}, {"frank", "francis", "frankie"},
]


def _first_compatible(reported, listed):
    """Is a reported first name plausibly the roster's first name for the same person?"""
    from difflib import SequenceMatcher
    a, b = name_key_first(reported), name_key_first(listed)
    if not a or not b:
        return False
    if a == b:
        return True
    if any(a in group and b in group for group in NICKNAMES):
        return True
    if len(a) >= 3 and len(b) >= 3 and (a in b or b in a):
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.75


def name_key_first(value):
    return re.sub(r"[^a-z]", "", str(value or "").lower())


def split_name(value):
    from cfbpred.player_timeline import name_key
    parts = [p for p in re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", str(value), flags=re.I).replace(",", " ").split() if p]
    if len(parts) < 2:
        return None, None
    return parts[0], name_key(parts[-1])


def printed_jersey(row):
    match = re.match(r"^\s*#?(\d{1,3})\s", str(row.get("source_quote") or ""))
    return int(match.group(1)) if match else None


def resolve_identities(facts, catalog):
    """Exact program-season name first; a variant spelling only when unambiguous.

    A fact is attached to a person when (1) the full name matches exactly one
    roster member of that program and season, or (2) exact matches are several
    and the printed jersey picks one, or (3) no exact match exists, exactly one
    roster member shares the surname with a compatible first name, and a
    printed jersey (when there is one) agrees. Everything else is left
    unresolved rather than guessed.
    """
    from cfbpred.player_timeline import name_key
    c = catalog.copy()
    c["full_key"] = (c.firstName.fillna("") + " " + c.lastName.fillna("")).map(name_key)
    c["last_key"] = c.lastName.fillna("").map(name_key)
    by_full = {k: g for k, g in c.groupby(["season", "team", "full_key"])}
    by_last = {k: g for k, g in c.groupby(["season", "team", "last_key"])}
    resolved, unresolved = [], []
    for row in facts:
        season, team = int(row["season"]), row["team"]
        jersey = printed_jersey(row)
        exact = by_full.get((season, team, name_key(row["name"])))
        pick, method = None, None
        if exact is not None and len(exact) == 1:
            pick, method = exact.iloc[0], "exact_name"
        elif exact is not None and jersey is not None:
            narrowed = exact[exact.jersey.eq(jersey)]
            if len(narrowed) == 1:
                pick, method = narrowed.iloc[0], "exact_name_jersey_disambiguated"
        elif exact is None:
            first, last = split_name(row["name"])
            pool = by_last.get((season, team, last)) if last else None
            if pool is not None:
                pool = pool[[_first_compatible(first, f) for f in pool.firstName.fillna("")]]
                if len(pool) == 1:
                    candidate = pool.iloc[0]
                    if jersey is None or pd.isna(candidate.jersey) or int(candidate.jersey) == jersey:
                        pick, method = candidate, "unique_surname_compatible_first_name"
        if pick is None or bool(pick.identity_uncertain):
            unresolved.append({**row, "reason": "identity_unresolved" if pick is None else "identity_uncertain"})
            continue
        out = dict(row)
        out["athlete_id"] = str(pick.athlete_id)
        out["identity_method"] = method
        if method == "unique_surname_compatible_first_name":
            out["reported_name"] = row["name"]
            out["name"] = f"{pick.firstName} {pick.lastName}".strip()
        resolved.append(out)
    return resolved, unresolved
