"""Fill roster gaps CFBD hasn't covered, via name aliasing + ESPN's public API.

Why: features_raw.py joins rosters to games by (season, team name). A missing
join leaves the whole roster block NaN, which standardises to the training
MEAN at inference — an FCS opponent with no published roster reads as an
average FBS program and compresses FBS-FCS spreads by ~11 points (measured
2026-08-27). Two distinct causes, two fills:

  alias  games.parquet names a program differently than roster.parquet does
         ('Appalachian State' 2004-13 vs 'App State'). teams.parquet
         alternateNames plus a small manual map resolve the variant; the
         canonical school's CFBD rows for that season are copied under the
         variant name. Local and exact — no network.
  espn   the roster genuinely isn't published (CFBD's FCS lag in the current
         season, a few ancient team-seasons). ESPN serves rosters keyed by
         the SAME team and athlete ids CFBD uses (CFBD is ESPN-sourced), so
         returning_production's exact athlete-id joins work on these rows.
         Current season via the site API (it caps at ~100 players — fine for
         counting features); past seasons via the core API, one request per
         athlete, only ever needed for a handful of team-seasons.

Purity: a roster is a raw fact about who is on a team — same footing as
recruiting points. Nothing here touches lines, ratings, or any outcome data.

Provenance + idempotency: every added row carries source='alias'/'espn'
(native CFBD rows carry null). Each run drops ALL previously added rows and
recomputes from the current gap list, so a pair disappears from the fill the
moment a CFBD re-pull covers it. games.parquet / schedule.parquet are never
modified.

Quality gate: an ESPN roster is used only when it carries >= MIN_PLAYERS
athletes with real names — ESPN's pre-~2010 FCS rosters are often 18-row
placeholder fragments ('- 28') that would poison n_players and
returning_share, and are skipped with a log line instead.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.config import CFBD, CURRENT_SEASON, PQ  # noqa: E402

SITE = ("https://site.api.espn.com/apis/site/v2/sports/football/"
        "college-football/teams/{tid}/roster")
CORE = ("https://sports.core.api.espn.com/v2/sports/football/leagues/"
        "college-football/seasons/{yr}/teams/{tid}/athletes?limit=400")

MIN_PLAYERS = 40          # below this an ESPN roster is a fragment, not a roster
FIRST_SEASON = 2004       # roster features exist 2004+; earlier gaps don't matter

# variants alternateNames doesn't cover, verified against teams.parquet
MANUAL_ALIAS = {
    "Louisiana Monroe": "UL Monroe",
    "Southeastern Louisiana": "SE Louisiana",
    "Presbyterian College": "Presbyterian",
    "UT San Antonio": "UTSA",
    "St. Francis (PA)": "Saint Francis",
}

CLASS_YEAR = {"FR": 1, "SO": 2, "JR": 3, "SR": 4, "GR": 5,
              "Freshman": 1, "Sophomore": 2, "Junior": 3, "Senior": 4,
              "Graduate": 5}

ROSTER_COLS = ["id", "firstName", "lastName", "team", "weight", "height",
               "jersey", "year", "position", "homeCity", "homeState",
               "homeCountry", "homeLatitude", "homeLongitude",
               "homeCountyFIPS", "recruitIds", "season_pulled", "source"]


def team_alias_map() -> dict:
    """variant name -> canonical CFBD school name, from teams.parquet
    alternateNames plus the manual fixes. Shared with features_raw so
    production-vs-roster joins survive a program's rename in the games data
    ('Appalachian State' 2013 -> 'App State' 2014)."""
    teams = pd.read_parquet(CFBD / "teams.parquet",
                            columns=["id", "school", "alternateNames",
                                     "season_pulled"])
    teams = (teams.sort_values("season_pulled")
             .drop_duplicates("school", keep="last"))
    alias = dict(MANUAL_ALIAS)
    for r in teams.itertuples():
        if r.alternateNames is not None:
            for a in list(r.alternateNames):
                alias.setdefault(a, r.school)
    return alias


def fbs_involved_pairs() -> pd.DataFrame:
    """Every (season, team) whose roster features can reach the model:
    FBS-involved games in every season, plus — current season only — any
    game with a Division I side (those are prediction targets on the slate,
    so their D2/D3 opponents deserve rosters too)."""
    frames = []
    for path in (PQ / "games.parquet", PQ / "schedule.parquet"):
        if not path.exists():
            continue
        g = pd.read_parquet(path, columns=["season", "home_team", "away_team",
                                           "home_division", "away_division"])
        keep = ((g.home_division == "fbs") | (g.away_division == "fbs")
                | (((g.home_division == "fcs") | (g.away_division == "fcs"))
                   & (g.season == CURRENT_SEASON)))
        g = g[keep]
        for side in ("home_team", "away_team"):
            frames.append(g[["season", side]].rename(columns={side: "team"}))
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    return out[out.season >= FIRST_SEASON]


def espn_session() -> requests.Session:
    # keep requests' default User-Agent: ESPN's WAF 403s browser-imitation
    # and custom UAs but serves plain python-requests/curl (verified 2026-08-27)
    s = requests.Session()
    s.headers.update({"Accept": "application/json"})
    return s


def _row(team: str, season: int, *, aid, first, last, weight, height,
         jersey, year, position) -> dict:
    return {"id": str(aid) if aid is not None else None,
            "firstName": first, "lastName": last, "team": team,
            "weight": weight, "height": height, "jersey": jersey,
            "year": year, "position": position,
            "homeCity": None, "homeState": None, "homeCountry": None,
            "homeLatitude": None, "homeLongitude": None,
            "homeCountyFIPS": None, "recruitIds": None,
            "season_pulled": season, "source": "espn"}


def _class_year(exp) -> int | None:
    if isinstance(exp, dict):
        exp = exp.get("abbreviation") or exp.get("displayValue")
    return CLASS_YEAR.get(exp)


def fetch_current(s: requests.Session, tid: int, team: str,
                  season: int) -> list[dict]:
    """Site API: the team's current roster in one request (capped ~100)."""
    r = s.get(SITE.format(tid=tid), timeout=30)
    r.raise_for_status()
    rows = []
    for grp in r.json().get("athletes", []):
        for a in grp.get("items", []):
            full = (a.get("fullName") or "").split(" ", 1)
            rows.append(_row(
                team, season, aid=a.get("id"),
                first=a.get("firstName") or (full[0] if full else None),
                last=a.get("lastName") or (full[1] if len(full) > 1 else None),
                weight=a.get("weight"), height=a.get("height"),
                jersey=pd.to_numeric(a.get("jersey"), errors="coerce"),
                year=_class_year(a.get("experience")),
                position=(a.get("position") or {}).get("abbreviation")))
    return rows


def fetch_historical(s: requests.Session, tid: int, team: str,
                     season: int) -> list[dict]:
    """Core API: one listing request, then one request per athlete."""
    r = s.get(CORE.format(yr=season, tid=tid), timeout=30)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    rows = []
    for it in r.json().get("items", []):
        ref = it.get("$ref", "").replace("http://", "https://")
        if not ref:
            continue
        try:
            a = s.get(ref, timeout=30).json()
        except (requests.RequestException, ValueError):
            continue
        time.sleep(0.08)
        rows.append(_row(
            team, season, aid=a.get("id"),
            first=a.get("firstName"), last=a.get("lastName"),
            weight=a.get("weight"), height=a.get("height"),
            jersey=pd.to_numeric(a.get("jersey"), errors="coerce"),
            year=_class_year(a.get("experience")),
            position=(a.get("position") or {}).get("abbreviation")))
    return rows


def plausible(rows: list[dict]) -> bool:
    """A real roster: enough players whose names look like names."""
    real = [r for r in rows
            if r["firstName"] and r["lastName"]
            and any(c.isalpha() for c in str(r["firstName"]))
            and any(c.isalpha() for c in str(r["lastName"]))]
    return len(real) >= MIN_PLAYERS


def main() -> None:
    path = CFBD / "roster.parquet"
    ros = pd.read_parquet(path)
    if "source" not in ros.columns:
        ros["source"] = None
    cfbd = ros[ros["source"].isna()].copy()      # native rows are the base
    n_dropped = len(ros) - len(cfbd)

    have = set(zip(cfbd["season_pulled"].astype(int), cfbd["team"]))
    pairs = fbs_involved_pairs()
    missing = [(int(r.season), r.team) for r in pairs.itertuples()
               if (int(r.season), r.team) not in have]
    print(f"{len(missing)} missing (season, team) pairs "
          f"(recomputing all fills; {n_dropped:,} prior fill rows dropped)")

    teams = pd.read_parquet(CFBD / "teams.parquet",
                            columns=["id", "school", "season_pulled"])
    teams = (teams.sort_values("season_pulled")
             .drop_duplicates("school", keep="last"))
    espn_id = {r.school: int(r.id) for r in teams.itertuples()
               if pd.notna(r.id)}
    alias = team_alias_map()

    s = espn_session()
    fills, n_alias, n_espn, skipped = [], 0, 0, []
    for season, team in sorted(missing):
        canon = alias.get(team)
        if canon and (season, canon) in have:
            src = cfbd[(cfbd["season_pulled"] == season)
                       & (cfbd["team"] == canon)].copy()
            src["team"] = team
            src["source"] = "alias"
            fills.append(src)
            n_alias += 1
            continue
        tid = espn_id.get(canon or team)
        if tid is None:
            skipped.append((season, team, "no ESPN team id"))
            continue
        try:
            rows = (fetch_current(s, tid, team, season)
                    if season == CURRENT_SEASON
                    else fetch_historical(s, tid, team, season))
        except requests.RequestException as e:
            skipped.append((season, team, f"fetch failed: {e}"))
            continue
        time.sleep(0.25)
        if not plausible(rows):
            skipped.append((season, team,
                            f"only {len(rows)} usable ESPN rows"))
            continue
        fills.append(pd.DataFrame(rows))
        n_espn += 1

    out = pd.concat([cfbd, *fills], ignore_index=True) if fills else cfbd
    out.to_parquet(path, index=False)
    print(f"filled {n_alias} team-seasons by alias, {n_espn} from ESPN "
          f"({sum(len(f) for f in fills):,} rows); roster.parquet now "
          f"{len(out):,} rows")
    for season, team, why in skipped:
        print(f"  skipped {season} {team}: {why}")


if __name__ == "__main__":
    main()
