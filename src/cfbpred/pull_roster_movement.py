"""Stage 2c: offseason roster movement and coaching tenures from CFBD -> parquet.

Three sources the pipeline did not use before September 30, 2026, all raw facts or human
scouting of players (allowed by the purity rule; no outcome prediction enters):

  portal.parquet         /player/portal   every transfer: season it counts toward, origin,
                                          destination, date, 247 transfer rating and stars
  draft_picks.parquet    /draft/picks     every NFL draft pick with college, round, overall
  coach_tenures.parquet  /coaches/tenures dated head-coach tenures incl. interim flags

/coaches/seasons is NOT pulled: it carries SP+, SRS and FPI. From tenures only dates, the
interim flag and the coach identity are used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.cfbd_client import CFBDClient  # noqa: E402
from cfbpred.config import CFBD, CURRENT_SEASON, PQ  # noqa: E402

FIRST_SEASON = 2001


def _frame(rows: list) -> pd.DataFrame:
    return pd.json_normalize(rows, sep="_") if rows else pd.DataFrame()


def _save(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    frame.to_parquet(CFBD / f"{name}.parquet", index=False)
    print(f"{name}: {len(frame):,} rows, columns {list(frame.columns)}", flush=True)
    return frame


def pull(first: int = FIRST_SEASON, last: int = CURRENT_SEASON,
         only: tuple = ("portal", "draft_picks", "coach_tenures")) -> dict:
    client = CFBDClient()
    out = {}
    if "portal" in only:
        portal = [_frame(client.get("/player/portal", {"year": y})) for y in range(2018, last + 1)]
        out["portal"] = _save("portal", pd.concat([p for p in portal if len(p)], ignore_index=True))
    if "draft_picks" in only:
        draft = [_frame(client.get("/draft/picks", {"year": y})) for y in range(first, last + 1)]
        out["draft_picks"] = _save("draft_picks", pd.concat([d for d in draft if len(d)], ignore_index=True).drop(
            columns=["hometownInfo_countyFips", "hometownInfo_latitude", "hometownInfo_longitude"], errors="ignore"))
    if "coach_tenures" in only:
        # the endpoint requires a team (or coach) filter: every Division I program since `first`
        games = pd.concat([pd.read_parquet(PQ / "games.parquet"), pd.read_parquet(PQ / "schedule.parquet")])
        games = games[games.season >= first]
        teams = sorted({t for side in ("home", "away")
                        for t, d in zip(games[f"{side}_team"], games[f"{side}_division"]) if d in ("fbs", "fcs")})
        parts = [_frame(client.get("/coaches/tenures", {"team": t})) for t in teams]
        tenures = pd.concat([x for x in parts if len(x)], ignore_index=True)
        out["coach_tenures"] = _save("coach_tenures", tenures.drop_duplicates("id").reset_index(drop=True))
    return out


if __name__ == "__main__":
    pull(only=tuple(sys.argv[1:]) or ("portal", "draft_picks", "coach_tenures"))
