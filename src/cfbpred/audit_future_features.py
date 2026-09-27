"""Fail a refresh if known observations disappear from any future target."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.config import CFBD, CURRENT_SEASON, FEAT, PBP
from cfbpred.features_raw import load_games
from cfbpred.game_feature_snapshots import game_sides
from cfbpred.train import prep_scalars


def audit():
    frame = pd.read_parquet(FEAT / "game_scalars.parquet")
    with duckdb.connect() as con:
        games = load_games(con, fbs_only=False)
    sides = game_sides(games)
    now = pd.Timestamp.now(tz="UTC")
    future = games[(games.season == CURRENT_SEASON) & (games.kick > now)
                   & (games.home_division.isin(["fbs", "fcs"]) | games.away_division.isin(["fbs", "fcs"]))]
    missing = set(future.game_id) - set(frame.game_id)
    if missing:
        raise AssertionError(f"Future games without feature rows: {sorted(missing)}")
    raw = pd.read_parquet(CFBD / "player_game_stats.parquet", columns=["game_id", "home_away", "category", "stat_type"])
    known = raw[(raw.category == "passing") & (raw.stat_type == "YDS")][["game_id", "home_away"]].drop_duplicates()
    first = sides[sides.completed].merge(known, on=["game_id", "home_away"]).groupby(["team", "season"]).kick.min()
    pbp_paths = sorted(PBP.glob("pbp_*.parquet"))
    extra = CFBD / f"pregame_plays_{CURRENT_SEASON}.parquet"
    if extra.exists():
        pbp_paths.append(extra)
    with duckdb.connect() as con:
        pbp_known = con.execute("""select distinct game_id, pos_team as team,
            bool_or(\"pass\" and passer_player_name is not null and period <= 4) as qb
            from read_parquet(?, union_by_name=true) group by game_id, pos_team""",
            [[str(p) for p in pbp_paths]]).fetchdf()
    played_pbp = sides[sides.completed].merge(pbp_known, on=["game_id", "team"])
    first_pbp = played_pbp.groupby(["team", "season"]).kick.min()
    first_qb = played_pbp[played_pbp.qb].groupby(["team", "season"]).kick.min()
    cur = frame[frame.game_id.isin(future.game_id)].copy()
    failures, gap_rows = [], []
    counts = {}
    for side, pre in (("home", "h_"), ("away", "a_")):
        current = cur.merge(future[["game_id", "kick"]], on="game_id", validate="one_to_one")
        known_before = pd.Series([first.get((r[side+"_team"], r["season"]), pd.NaT) for _, r in current.iterrows()], index=current.index)
        expected = known_before.notna() & (known_before < current.kick)
        missing_box = expected & current[pre+"bx_pass_ypg"].isna()
        failures.extend(current.loc[missing_box, "game_id"].astype(int).tolist())
        source_counts = {}
        for field, source in (("o_ypp", first_pbp), ("qb_ypa", first_qb)):
            source_dates = pd.Series([source.get((r[side+"_team"], r["season"]), pd.NaT)
                                      for _, r in current.iterrows()], index=current.index)
            source_expected = source_dates.notna() & (source_dates < current.kick)
            if (source_expected & current[pre+field].isna()).any():
                raise AssertionError(f"Known prior {side} {field} observations missing from future features")
            source_counts["with_prior_"+field] = int(source_expected.sum())
        counts[side] = {"targets": len(current), "with_prior_passing_box": int(expected.sum()),
                       "with_box_feature": int(current[pre+"bx_pass_ypg"].notna().sum()),
                       "with_pbp_feature": int(current[pre+"o_ypp"].notna().sum()),
                       "with_career_feature": int(current[pre+"cx_qb_gms"].notna().sum()),
                       "with_qb_feature": int(current[pre+"qb_ypa"].notna().sum()), **source_counts}
        for _, r in current[~expected].iterrows():
            gap_rows.append({"game_id": int(r.game_id), "team": r[side+"_team"],
                             "side": side, "games_played": float(r[pre+"gp"]),
                             "reason": "no completed-game passing box available"})
        # Every distant future game gets the current statistical snapshot.
        dynamic = [c for c in cur if c.startswith(pre) and c[len(pre):].startswith(
            ("o_", "d_", "bx_", "qb_", "cx_", "av_", "sos_"))]
        varying = cur.groupby(side+"_team")[dynamic].nunique().gt(1)
        if varying.any().any():
            raise AssertionError(f"Unplayed targets advance {side} statistical snapshots")
    if failures:
        raise AssertionError(f"Known prior box scores missing from future features: {failures}")
    prepared, cols = prep_scalars(cur)
    if len(prepared) != len(future) or cur.game_id.duplicated().any():
        raise AssertionError("Future schedule and feature rows are not one-to-one")
    report = {"generated_at": now.isoformat(), "future_games": len(cur),
              "active_scalar_columns": len(cols), "known_box_join_failures": len(failures),
              "coverage": counts, "source_gaps": gap_rows,
              "missing_by_active_column": {c: int(prepared[c].isna().sum()) for c in cols}}
    (FEAT / "future_coverage.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k,v in report.items() if k not in {"source_gaps", "missing_by_active_column"}}, indent=2))
    return report


if __name__ == "__main__":
    audit()
