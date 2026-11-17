"""Current raw PBP for pregame counting features while the release feed lags.

This feed is kept separate from the repaired live-model training corpus.
It contains no PPA/EPA, market data, ratings, or outcome-model outputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.cfbd_client import CFBDClient
from cfbpred.config import CFBD, CURRENT_SEASON, RAW

REFRESH_DAYS = 3


def fetch_game_stats(client, game_id):
    """The endpoint truncates at 2,000 rows; a week-wide query loses games."""
    result = client.get("/plays/stats", {"gameId": int(game_id)})
    if len(result) >= 2000:
        raise RuntimeError(f"Player attribution for {game_id} reached the API cap; refusing truncated inputs")
    if any(int(r["gameId"]) != int(game_id) for r in result):
        raise ValueError(f"Player attribution returned a different game than {game_id}")
    return [{k: v for k, v in r.items() if k != "ppa"} for r in result]


def normalize(plays, stats, season, week, season_type):
    p = pd.DataFrame(plays)
    if p.empty:
        return p
    p = p.drop_duplicates(["gameId", "id"]).copy()
    attribution = pd.DataFrame(stats)
    passer, rusher = {}, {}
    if not attribution.empty:
        for r in attribution.itertuples():
            key = (r.gameId, str(r.playId))
            if r.statType in {"Completion", "Incompletion", "Interception Thrown", "Sack Taken"}:
                passer[key] = r.athleteName
            if r.statType == "Rush":
                rusher[key] = r.athleteName
    ty = p.playType.fillna("")
    txt = p.playText.fillna("").str.lower()
    sack = ty.eq("Sack") | txt.str.contains(r"\bsacked\b")
    interception = ty.str.contains("Interception")
    passed = ty.isin(["Pass", "Pass Reception", "Pass Completion", "Pass Incompletion", "Passing Touchdown"]) | interception | sack
    fumble = ty.str.contains("Fumble") | txt.str.contains(r"\bfumbl(?:e|ed|es)\b")
    # Named play-stat attribution resolves fumbles/safeties that hide the
    # underlying pass or rush type. No player-name inference from prose.
    keys = list(zip(p.gameId, p.id.astype(str)))
    qb = pd.Series([passer.get(k) for k in keys], index=p.index)
    rb = pd.Series([rusher.get(k) for k in keys], index=p.index)
    passed |= qb.notna()
    rush = ty.isin(["Rush", "Rushing Touchdown"]) | (rb.notna() & ~passed)
    penalty = ty.str.contains("Penalty") | txt.str.contains(r"\bpenalty\b")
    no_play = txt.str.contains(r"\bno play\b")
    passed &= ~no_play
    rush &= ~no_play
    punt = ty.str.contains("Punt")
    kickoff = ty.str.contains("Kickoff")
    yards = pd.to_numeric(p.yardsGained, errors="coerce")
    lost = ty.isin(["Fumble Recovery (Opponent)", "Fumble Recovery (Opponent) Touchdown"])
    failed_fourth = p.down.eq(4) & (passed | rush) & yards.lt(pd.to_numeric(p.distance, errors="coerce")) & ~penalty
    turnover = interception | lost | failed_fourth | ty.isin(["Field Goal Missed", "Blocked Field Goal", "Blocked Punt", "Blocked Punt Touchdown", "Blocked Field Goal Touchdown"])
    return pd.DataFrame({
        "game_id": p.gameId.astype("int64"), "play_id": p.id.astype(str),
        "year": season, "week": week, "season_type": season_type,
        "period": p.period, "pos_team": p.offense, "def_pos_team": p.defense,
        "yards_gained": yards, "down": p.down, "distance": p.distance,
        "rush": rush, "pass": passed, "sack": sack & ~no_play,
        "int": interception & ~no_play, "fumble_vec": fumble & ~no_play,
        "turnover": turnover & ~no_play, "punt": punt, "kickoff_play": kickoff,
        "penalty_flag": penalty, "touchdown": ty.str.contains("Touchdown") & ~no_play,
        "fg_made": ty.eq("Field Goal Good"), "passer_player_name": qb,
        "rusher_player_name": rb, "source": "cfbd_raw_plays",
    })


def main(season=CURRENT_SEASON):
    games = pd.read_parquet(CFBD / "games.parquet")
    games = games[(games.season == season) & games.completed.fillna(False)]
    # A finished game's plays are final once CFBD's post-game corrections land,
    # so only games that started within REFRESH_DAYS are re-pulled; older ones
    # come from the cache. Re-pulling every game every night cost ~700
    # /plays/stats calls a night, which would exhaust the monthly quota.
    recent = pd.to_datetime(games.startDate, utc=True) >= (
        pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=REFRESH_DAYS))
    games = games.assign(recent=recent.to_numpy())
    client = CFBDClient()
    start = client.spent
    cache = RAW / "current_plays"
    cache.mkdir(parents=True, exist_ok=True)
    batches = []
    for (kind, week), g in games.groupby(["seasonType", "week"]):
        week_file = cache / f"{season}_{kind}_{week}_plays.json"
        if week_file.exists() and not g.recent.any():
            plays = json.loads(week_file.read_text())
        else:
            params = {"year": season, "week": int(week), "seasonType": kind}
            plays = client.get("/plays", params)
            plays = [{k: v for k, v in r.items() if k != "ppa"} for r in plays]
            week_file.write_text(json.dumps(plays))
        stats = []
        play_games = {int(r["gameId"]) for r in plays}
        targets = g[g.id.isin(play_games)]
        fetched = 0
        for gid, is_recent in zip(targets.id, targets.recent):
            game_file = cache / f"{season}_game_{gid}_stats.json"
            if game_file.exists() and not is_recent:
                result = json.loads(game_file.read_text())
            else:
                result = fetch_game_stats(client, gid)
                game_file.write_text(json.dumps(result))
                fetched += 1
            stats.extend(result)
        print(f"  player attribution: {len(targets)} games ({fetched} fetched, "
              f"{len(targets) - fetched} cached)", flush=True)
        frame = normalize(plays, stats, season, int(week), kind)
        if not frame.empty:
            frame = frame[frame.game_id.isin(g.id)]
            batches.append(frame)
            print(f"{season} {kind} week {week}: {len(frame):,} plays, {frame.game_id.nunique()} completed games", flush=True)
    if not batches:
        raise RuntimeError("No completed-game PBP returned; existing cache retained")
    out = pd.concat(batches, ignore_index=True).drop_duplicates(["game_id", "play_id"])
    dest = CFBD / f"pregame_plays_{season}.parquet"
    tmp = dest.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(dest)
    print(f"Saved {len(out):,} raw plays for {out.game_id.nunique()} games "
          f"[{client.spent - start} CFBD calls]")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else CURRENT_SEASON)
