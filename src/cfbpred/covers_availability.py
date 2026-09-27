"""Covers NCAAF injury page: live captures -> availability facts for uncovered team-games.

Covers republishes team and conference availability statuses for every FBS
program, including conferences with no machine-readable official feed. Each
capture's retrieval time is the knowledge time (never earlier), so a status
counts for a game only when it was captured before that kickoff.

Conservative rules, identical to the historical Wayback backfill:
* per team-game, only the latest capture strictly before kickoff and at most
  96 hours before it;
* only entries updated in the six days before the game, or that state a
  season-ending absence;
* "C. Jones" resolves only when first initial + surname is unique on that
  program-season roster (position breaks a tie when it agrees);
* a team-game already covered by an official conference feed is left alone.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

import pandas as pd
import requests

from cfbpred.config import ROOT as PROJECT_ROOT
from cfbpred.pregame_experiments import save_json, sha
from cfbpred.pull_player_context import archive

ROOT = PROJECT_ROOT / "data/raw/covers_availability"
PAGE = "https://www.covers.com/sport/football/ncaaf/injuries"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"

TEAM = re.compile(r'covers-CoversMatchups-teamName">\s*<a[^>]*>\s*(.*?)<br>', re.S)
ROW = re.compile(r"<tr>\s*<td>\s*<span class='player-link'>(.*?)</span>\s*</td>\s*<td>(.*?)</td>\s*"
                 r"<td><b>(.*?)</b><br>\((.*?)\)</td>", re.S)
MONTH = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
SCHOOLS = {"Miami (FL)": "Miami", "Miami FL": "Miami", "Hawaii": "Hawai'i", "San Jose State": "San José State",
           "Appalachian State": "App State", "Louisiana-Monroe": "UL Monroe", "Louisiana-Lafayette": "Louisiana",
           "UMass": "Massachusetts", "Southern Mississippi": "Southern Miss", "Connecticut": "UConn",
           "FIU": "Florida International", "Florida Intl": "Florida International", "Sam Houston State": "Sam Houston",
           "North Carolina State": "NC State", "Mississippi": "Ole Miss", "Texas-San Antonio": "UTSA",
           "Texas-El Paso": "UTEP", "Central Florida": "UCF", "Southern California": "USC",
           "Brigham Young": "BYU", "Texas Christian": "TCU", "Southern Methodist": "SMU",
           "Louisiana State": "LSU", "Alabama-Birmingham": "UAB", "Middle Tennessee State": "Middle Tennessee"}
POS = {"OG": "OL", "OT": "OL", "C": "OL", "G": "OL", "T": "OL", "DE": "DL", "DT": "DL", "NT": "DL", "EDGE": "DL",
       "OLB": "LB", "ILB": "LB", "MLB": "LB", "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "SAF": "DB",
       "PK": "K", "HB": "RB", "FB": "RB"}


def collect(root: Path = ROOT, session=None) -> Path:
    """Archive one capture of the page, with its retrieval receipt."""
    import hashlib
    session = session or requests.Session()
    response = session.get(PAGE, headers={"User-Agent": UA}, timeout=60)
    response.raise_for_status()
    if b"covers-CoversMatchups-teamName" not in response.content:
        raise ValueError("Covers page did not contain team blocks")
    now = datetime.now(timezone.utc)
    manifests = sorted(root.glob("*/manifest.json"))
    if manifests:
        latest = json.loads(manifests[-1].read_text())
        if latest["sources"][0]["sha256"] == hashlib.sha256(response.content).hexdigest():
            # Unchanged page: log the re-observation instead of storing the bytes again.
            latest.setdefault("reobserved_at", []).append(now.isoformat())
            save_json(manifests[-1], latest)
            return manifests[-1].parent
    run = root / now.strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True, exist_ok=False)
    meta = archive(response.content, PAGE, run, "covers_injuries.html", observed_at=now.isoformat())
    save_json(run / "manifest.json", dict(run=str(run.resolve()), sources=[meta]))
    return run


def parse(path: Path, captured: pd.Timestamp) -> dict:
    """{school: [entry]} exactly as the page printed it."""
    raw = Path(path).read_text(errors="ignore")
    teams = {}
    for part in re.split(r'<a id="[^"]*"></a>', raw):
        head = TEAM.search(part)
        if not head:
            continue
        school = " ".join(unescape(re.sub(r"<[^>]+>", " ", head.group(1))).split())
        entries = []
        for m in ROW.finditer(part):
            player = " ".join(unescape(re.sub(r"<[^>]+>", " ", m.group(1))).split())
            status = " ".join(unescape(re.sub(r"<[^>]+>", " ", m.group(3))).split())
            when = " ".join(unescape(m.group(4)).split())
            day = re.search(r"([A-Z][a-z]{2})\s+(\d{1,2})", when)
            entry_date = None
            if day:
                month = MONTH[day.group(1)]
                year = captured.year if month <= captured.month else captured.year - 1
                entry_date = pd.Timestamp(year=year, month=month, day=int(day.group(2)))
            note = re.search(r'injuryCopy">\s*(.*?)\s*</div>', part[m.end():m.end() + 1500], re.S)
            entries.append(dict(player=player, pos=" ".join(unescape(m.group(2)).split()), status=status,
                                when=when, entry_date=entry_date,
                                note=" ".join(unescape(note.group(1)).split()) if note else ""))
        teams[school] = entries
    return teams


def load_facts(schedule: pd.DataFrame, catalog: pd.DataFrame, covered=frozenset(), root: Path = ROOT):
    """Facts for every team-game with a qualifying capture; unresolved rows are returned too."""
    from cfbpred.availability_archive import encode, norm_team
    from cfbpred.player_timeline import name_key
    games = schedule.copy()
    games["kick"] = pd.to_datetime(games.kick, utc=True)
    captures = []
    for manifest in sorted(root.glob("*/manifest.json")):
        record = json.loads(manifest.read_text())
        for item in record["sources"]:
            path = Path(item["path"])
            if sha(path) != item["sha256"]:
                raise ValueError("Covers capture changed: " + str(path))
            first = pd.Timestamp(item["observed_at"])
            parsed = parse(path, first)
            # Every observation of identical bytes is evidence the page said this then.
            for seen in [item["observed_at"], *record.get("reobserved_at", [])]:
                captures.append((pd.Timestamp(seen), path, item, parsed))
    captures.sort(key=lambda c: c[0])
    if not captures:
        return [], []
    # Only fixtures a capture could inform (captured at most 96 h before kickoff).
    games = games[(games.kick > captures[0][0]) & (games.kick <= captures[-1][0] + pd.Timedelta(hours=96))]
    cat = catalog.copy()
    cat["last_key"] = cat.lastName.fillna("").map(name_key)
    cat["initial"] = cat.firstName.fillna("").str[:1].str.lower()
    by = {k: v for k, v in cat.groupby(["season", "team", "last_key", "initial"])}
    facts, issues = [], []
    for game in games.itertuples():
        season_names = None
        for team in (game.home_team, game.away_team):
            if (int(game.game_id), team) in covered:
                continue
            prior = [c for c in captures if c[0] < game.kick and c[0] >= game.kick - pd.Timedelta(hours=96)]
            if not prior:
                continue
            if season_names is None:
                season_names = set(games[games.season == game.season].home_team) | set(games[games.season == game.season].away_team)
            chosen = None
            for captured, path, item, teams in reversed(prior):
                for school, entries in teams.items():
                    mapped = SCHOOLS.get(school) if SCHOOLS.get(school) in season_names else norm_team(school, season_names)
                    if mapped == team:
                        chosen = (captured, path, item, entries)
                        break
                if chosen:
                    break
            if not chosen:
                continue
            captured, path, item, entries = chosen
            kick_day = game.kick.tz_convert("America/Chicago").normalize().tz_localize(None)
            for e in entries:
                season_ending = bool(re.search(r"remainder of the (?:\d{4} )?season|out for the season|season-ending",
                                               e["note"], re.I)) or e["status"].lower().startswith("out for season")
                fresh = e["entry_date"] is not None and kick_day - pd.Timedelta(days=6) <= e["entry_date"] <= kick_day
                if not (fresh or season_ending):
                    continue
                label = e["status"].split(" - ")[0].strip()
                value, why = encode("out" if label.lower().startswith("out for season") else label)
                if value is None:
                    issues.append(dict(reason=why, team=team, game_id=int(game.game_id), player=e["player"], status=e["status"]))
                    continue
                m = re.match(r"^([A-Za-z])\.?\s+(.+)$", e["player"])
                pool = by.get((int(game.season), team, name_key(m.group(2)), m.group(1).lower())) if m else None
                if pool is not None and len(pool) > 1:
                    pos = POS.get(e["pos"], e["pos"])
                    pool = pool[pool.position.fillna("").map(lambda p: POS.get(p, p)).eq(pos)]
                if pool is None or len(pool) != 1 or bool(pool.iloc[0].identity_uncertain):
                    issues.append(dict(reason="identity_not_unique", team=team, game_id=int(game.game_id), player=e["player"]))
                    continue
                person = pool.iloc[0]
                facts.append(dict(name=f"{person.firstName} {person.lastName}".strip(), reported_name=e["player"],
                                  capture_first_observed_at=item["observed_at"],
                                  athlete_id=str(person.athlete_id), team=team, season=int(game.season),
                                  game_id=int(game.game_id), kind="availability", value=value, raw_status=e["status"],
                                  known_at=captured.isoformat(), effective_at=captured.isoformat(),
                                  valid_until=(game.kick + pd.Timedelta(days=1)).isoformat(),
                                  observed_at=captured.isoformat(), source_path=str(path.resolve()),
                                  source_sha256=item["sha256"], source_url=PAGE,
                                  source_quote=f"{e['player']} {e['pos']} {e['status']}",
                                  source_kind="covers_live_capture", entry_date=str(e["entry_date"].date()) if e["entry_date"] is not None else None,
                                  timing_assumed=False,
                                  timing_basis="Retrieval time of the archived capture; strictly before kickoff"))
    return facts, issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(collect(args.root))
