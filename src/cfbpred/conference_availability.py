"""Public conference statuses with immutable receipts and explicit timing assumptions.

HD Intelligence exposes current-season archives, not a universal historical
college-football database. Unknown programs/players remain unknown.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import pandas as pd
import requests

from cfbpred.pregame_experiments import save_json, sha
from cfbpred.pull_player_context import archive

ROOT = Path("data/raw/conference_availability")
ENDPOINT = "https://app.hdintelligence.com/api/"
# Public conference reporting pages document these providers. Responses are
# retained even when a conference has no records; empty never means healthy.
CONFERENCES = {
    "SEC": "https://www.secsports.com/fbreports-archive",
    "B10": "https://bigten.org/fb/availability-reports/",
    "ACC": "https://theacc.com/sports/2025/8/28/availability-reporting-football-archive.aspx",
    "B12": "https://big12sports.com/sports/2025/8/21/FBReportArchive.aspx",
    "MAC": "https://getsomemaction.com/sports/2026/7/28/FB_0728265050.aspx.aspx",
    # The American's official page embeds HD Intelligence with the long-form
    # organization alias below (``AAC`` is rejected by the public API).
    "American": "https://theamerican.org/sports/2026/8/6/football_player_availability.aspx",
    # CFP uses a separate public HD Intelligence app and endpoint.  Its empty
    # archive is retained as evidence; empty must never be interpreted healthy.
    "CFP": "https://collegefootballplayoff.com/sports/2025/11/12/reports",
}
PUBLIC_ROUTES = {
    "CFP": {"archive": ("https://confinjrepxyz.hdintelligence-app.com/api/M!nD0Mcfp",
                           {"organization": "CFP"})},
}
STATUS = {"out": "out", "doubtful": "doubtful", "questionable": "questionable",
          "probable": "probable", "available": "available", "game time decision": "questionable"}
ALIASES = {"Southern California": "USC", "Southern Methodist": "SMU", "Miami (FL)": "Miami",
           "Connecticut": "UConn", "Massachusetts": "UMass", "Brigham Young": "BYU", "Albany": "UAlbany",
           "San Jose State": "San José State", "Central Connecticut State": "Central Connecticut"}


def collect(root=ROOT, conferences=None, session=None):
    session = session or requests.Session()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = root / stamp
    run.mkdir(parents=True, exist_ok=False)
    sources, failures = [], []
    for conference in conferences or CONFERENCES:
        routes = PUBLIC_ROUTES.get(conference, {})
        if not routes:
            routes = {kind: (ENDPOINT + route,
                             {"sport": "Football", "organization": conference, "conference": conference})
                      for kind, route in (("archive", "get-archive-public"), ("report", "get-publish-public"))}
        for kind, (url, params) in routes.items():
            try:
                response = session.post(url, json=params, timeout=30)
                if response.status_code == 429:
                    failures.append(dict(conference=conference, kind=kind, status=429,
                                         retry_after=response.headers.get("Retry-After")))
                    break
                response.raise_for_status()
                payload = response.json()
                item = archive(response.content, url, run, f"{conference}_{kind}.json", params=params,
                               link_previous=True)
                item.update(conference=conference, kind=kind, official_page=CONFERENCES[conference])
                sources.append(item)
                count = len(payload.get("data", [])) if isinstance(payload, dict) else len(payload)
                print(conference, kind, count, flush=True)
            except (requests.RequestException, ValueError) as error:
                failures.append(dict(conference=conference, kind=kind, error=str(error)))
            time.sleep(1)
        if failures and failures[-1].get("status") == 429:
            break # one reporting host; honor host-wide throttling
    report = dict(run=str(run.resolve()), sources=sources, failures=failures,
                  all_teams_covered=False, historical_timing_policy="assume_official_game_day_report_before_kickoff")
    save_json(run / "manifest.json", report)
    pending = root / "latest.pending.json"
    save_json(pending, report); pending.replace(root / "latest.json")
    return run


def team_name(value, schedule_teams, conference=None):
    value = re.sub(r"^(?:at |vs\.? |@ )", "", str(value)).strip()
    if conference == "MAC" and value == "Miami":
        return "Miami (OH)"
    if value in schedule_teams:
        return value
    return ALIASES.get(value, value)


def build_fixture_index(schedule):
    """Normalize the schedule once for archive matching within one load."""
    games = schedule.copy()
    games["kick"] = pd.to_datetime(games.kick, utc=True)
    names = set(games.home_team) | set(games.away_team)
    by_fixture = {}
    dates = games.kick.dt.tz_convert("America/Chicago").dt.strftime("%Y-%m-%d")
    for (_, game), day in zip(games.iterrows(), dates):
        by_fixture.setdefault((frozenset((game.home_team, game.away_team)), day), []).append(game)
    return names, by_fixture


def archive_facts(path, schedule, *, allow_assumed=True, fixture_index=None):
    """Match exact team/opponent/local date; use only explicit Game Day cells.

    The archive omits report publication timestamps. The user has authorized
    treating its game-day reports as pregame. We assign T-30 solely as a tagged
    modeling assumption, never as a measured publisher timestamp. Initial and
    update cells stay in raw evidence; no invented publication times for them.
    """
    payload = json.loads(path.read_text())
    receipt = json.loads(path.with_suffix(".json.source.json").read_text())
    conference = receipt.get("params", {}).get("conference")
    if fixture_index is None:
        names, by_fixture = build_fixture_index(schedule)
    else:
        names, by_fixture = fixture_index
    observed = pd.Timestamp(receipt["observed_at"])
    digest = sha(path)
    facts, issues = [], []
    rows = payload.get("data", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return [], [dict(reason="unsupported_archive_payload", raw=payload)]
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not row.get("Player"):
            issues.append(dict(source_row=index, raw=row, reason="unsupported_archive_row")); continue
        team = team_name(row.get("TeamDisplay", row.get("Team", "")), names, conference)
        opponent = team_name(row.get("OpponentDisplay", row.get("Opponent", "")), names)
        status_raw = str(row.get("Game Day", "")).strip()
        status = STATUS.get(status_raw.lower())
        reason = None
        if not status:
            reason = "unsupported_or_missing_game_day_status"
        try:
            day = pd.Timestamp(row["Week"]).strftime("%Y-%m-%d")
        except (ValueError, KeyError):
            day = None; reason = "invalid_game_date"
        selected = by_fixture.get((frozenset((team, opponent)), day), [])
        opponent_mismatch = False
        if len(selected) != 1 and day is not None and opponent == team:
            # The publisher's opponent field sometimes repeats the team itself
            # (the 2026 Big 12 feed lists Kansas vs. "Kansas"). A program plays
            # once on a date, so the team and date still identify the fixture.
            # A different printed opponent is never overridden.
            alone = [g for key, games in by_fixture.items() if key[1] == day and team in key[0] for g in games]
            if len(alone) == 1:
                selected, opponent_mismatch = alone, True
        if len(selected) != 1:
            reason = "no_unique_team_opponent_date_match"
        if reason:
            issues.append(dict(source_row=index, raw=row, reason=reason)); continue
        game = selected[0]
        assumed = observed >= game.kick
        if assumed and not allow_assumed:
            issues.append(dict(source_row=index, raw=row, reason="historical_timing_assumption_excluded")); continue
        known = game.kick - pd.Timedelta(minutes=30) if assumed else observed
        facts.append(dict(name=row["Player"], team=team, season=int(game.season), game_id=int(game.game_id),
                          kind="availability", value=status, raw_status=status_raw,
                          known_at=known.isoformat(), effective_at=known.isoformat(),
                          valid_until=(game.kick + pd.Timedelta(days=1)).isoformat(),
                          source_path=str(path.resolve()), source_sha256=digest, source_url=receipt["url"],
                          observed_at=observed.isoformat(), source_kind="conference_public_archive",
                          source_row=index, source_record=row, timing_assumed=assumed,
                          opponent_field_mismatch=opponent_mismatch,
                          timing_basis=("User-authorized assumption: official Game Day status available at T-30; archive has no publication timestamp"
                                        if assumed else "Collector receipt before kickoff")))
    return facts, issues


def current_facts(path, schedule):
    # Same public schema as SEC, with exact team aliases normalized first.
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [], ([dict(reason="unsupported_current_report_payload", raw=payload)] if payload else [])
    from cfbpred.public_player_reports import sec_current
    return sec_current(path, schedule)


def latest_retrievals(facts):
    """Drop re-retrievals the dated timeline can never apply.

    Every capture of an unchanged report restamps each row with its retrieval
    time, so the same status arrives once per capture (37 copies of one Big Ten
    report by 2026-09-25). A game-scoped status applies to one kickoff, and
    ``compile_arrays`` keeps the latest eligible fact before it, so older
    retrievals of a player-game only matter as fallbacks. Walk back from the
    newest retrieval and stop at the first one that cannot be discarded
    downstream: rows that agree, from a real capture, not stamped at kickoff
    minus 30 minutes. Every source that assumes timing assumes exactly T-30, so
    only there can another source's evidence tie with these rows and void both
    (``compile_arrays`` rejects equal-time disagreements). Real captures carry
    microsecond timestamps that no other source shares. Everything a tie
    rejection could fall back to is therefore kept.
    """
    from cfbpred.player_timeline import name_key
    groups = {}
    for i, row in enumerate(facts):
        key = (row["season"], row["team"], name_key(row["name"]), row["game_id"], row["kind"])
        groups.setdefault(key, {}).setdefault(pd.Timestamp(row["known_at"]), []).append(i)
    keep = set()
    for levels in groups.values():
        for known in sorted(levels, reverse=True):
            rows = [facts[i] for i in levels[known]]
            keep.update(levels[known])
            # Both loaders set valid_until to kickoff + 1 day.
            at_assumed_time = any(known == pd.Timestamp(r["valid_until"]) - pd.Timedelta(days=1, minutes=30)
                                  for r in rows)
            if (len({r["value"] for r in rows}) == 1 and not at_assumed_time
                    and not any(r.get("timing_assumed") for r in rows)):
                break
    return [row for i, row in enumerate(facts) if i in keep]


def load_facts(schedule, root=ROOT, allow_assumed=True, latest_only=False):
    """Read all captured archives so removed upstream weeks remain available.

    ``latest_only`` collapses repeated captures (see ``latest_retrievals``).
    Historical rebuilds leave it off so their pinned artifacts stay byte-identical.
    """
    facts, issues, seen = [], [], set()
    fixture_index = build_fixture_index(schedule)
    for manifest in sorted(root.glob("*/manifest.json")):
        for item in json.loads(manifest.read_text())["sources"]:
            path = Path(item["path"])
            if sha(path) != item["sha256"]:
                raise ValueError("Conference archive changed: " + str(path))
            rows, bad = (archive_facts(path, schedule, allow_assumed=allow_assumed,
                                       fixture_index=fixture_index)
                         if item["kind"] == "archive" else current_facts(path, schedule))
            issues.extend({**r, "archive": str(path)} for r in bad)
            for row in rows:
                key = tuple(str(row.get(k)) for k in ("game_id", "team", "name", "value", "known_at", "source_url"))
                if key not in seen:
                    seen.add(key); facts.append(row)
    return (latest_retrievals(facts) if latest_only else facts), issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--conference", action="append", choices=sorted(CONFERENCES))
    args = parser.parse_args()
    print(collect(args.root, args.conference))
