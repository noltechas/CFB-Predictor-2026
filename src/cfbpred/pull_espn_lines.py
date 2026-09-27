"""DraftKings lines for games CFBD's /lines feed doesn't cover (the FCS
slate), read from ESPN's scoreboard. Display-only, exactly like every other
line: rows land in espn_lines.parquet, publish_lines.py appends them to
game_lines for game_ids CFBD has no line for, and nothing on the feature
path can see them (the BANNED_SUBSTRINGS test enforces it).

ESPN's odds.spread is the HOME handicap — the same convention as CFBD
(negative = home favored). Moneylines appear closer to kickoff; whatever is
present is carried.
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.config import PQ  # noqa: E402

OUT = PQ / "espn_lines.parquet"
SB = ("https://site.api.espn.com/apis/site/v2/sports/football/"
     "college-football/scoreboard?groups={grp}&dates={day}&limit=500")
SUMMARY = ("https://site.api.espn.com/apis/site/v2/sports/football/"
           "college-football/summary?event={gid}")
DAYS_AHEAD = 9
# also look BACK: a game that already finished still needs its closing line
# captured (pickcenter keeps it), or it can never be graded on the Results page
DAYS_BACK = 5
PICKCENTER_CAP = 400   # per-run ceiling on per-game fallback calls


def _odds_row(gid: int, o: dict) -> dict:
    return {
        "game_id": gid,
        "provider": (o.get("provider") or {}).get("name", "ESPN"),
        "spread": o.get("spread"),
        "over_under": o.get("overUnder"),
        "home_moneyline": (o.get("homeTeamOdds") or {}).get("moneyLine"),
        "away_moneyline": (o.get("awayTeamOdds") or {}).get("moneyLine"),
    }


def main() -> None:
    s = requests.Session()                      # default UA; ESPN 403s custom
    s.headers.update({"Accept": "application/json"})
    start = date.today() - timedelta(days=DAYS_BACK)
    end = date.today() + timedelta(days=DAYS_AHEAD)
    d0, d1 = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    rows, seen = [], set()
    successful_scoreboards = 0
    for grp in (80, 81):
        # ESPN accepts a single YYYYMMDD here but returns HTTP 400 for a
        # YYYYMMDD-YYYYMMDD range. Query each day so a failed response cannot
        # silently turn the entire odds file into an empty DataFrame.
        for day_offset in range((end - start).days + 1):
            day = (start + timedelta(days=day_offset)).strftime("%Y%m%d")
            try:
                response = s.get(SB.format(grp=grp, day=day), timeout=30)
                response.raise_for_status()
                d = response.json()
            except (requests.RequestException, ValueError) as exc:
                print(f"  ! group {grp} day {day}: {str(exc)[:80]}")
                continue
            successful_scoreboards += 1
            time.sleep(0.3)
            for e in d.get("events", []):
                gid = int(e["id"])
                if gid in seen:
                    continue
                seen.add(gid)
                for o in e["competitions"][0].get("odds", []) or []:
                    if o.get("spread") is None and o.get("overUnder") is None:
                        continue
                    rows.append(_odds_row(gid, o))
                    break                        # first (primary) book only
    if successful_scoreboards == 0:
        raise RuntimeError("all ESPN scoreboard requests failed; preserving previous odds file")
    # the scoreboard drops odds once a game kicks off; the game summary's
    # pickcenter keeps the pregame line during and after — sweep the gaps
    have = {r["game_id"] for r in rows}
    gaps = [g for g in seen if g not in have][:PICKCENTER_CAP]
    n_pc = 0
    for gid in gaps:
        try:
            pc = (s.get(SUMMARY.format(gid=gid), timeout=30).json()
                  .get("pickcenter") or [])
        except (requests.RequestException, ValueError):
            pc = []
        time.sleep(0.15)
        for o in pc:
            if o.get("spread") is None and o.get("overUnder") is None:
                continue
            rows.append(_odds_row(gid, o))
            n_pc += 1
            break
    print(f"  scoreboard odds: {len(have):,} | pickcenter fallback: "
          f"{n_pc:,} of {len(gaps):,} gap games")
    df = pd.DataFrame(rows, columns=["game_id", "provider", "spread", "over_under",
                                     "home_moneyline", "away_moneyline"])
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(df):,} games with ESPN lines "
          f"({d0}..{d1}, FBS+FCS scoreboards)")


if __name__ == "__main__":
    main()
