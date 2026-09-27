"""Prior-game raw form and matchup features, staged outside production.

Windows refer to completed scheduled games, including missing-feed games.
Unplayed targets never consume a window slot. Each season starts afresh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import deque
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.config import CFBD, CURRENT_SEASON, PBP
from cfbpred.features_raw import load_games
from cfbpred.game_feature_snapshots import game_sides

COUNTS = ["plays", "pass_plays", "pass_yards", "pass_yards_observed", "rush_plays",
          "rush_yards", "rush_yards_observed", "sacks", "expl20"]
RATES = ["pass_ypp", "rush_ypp", "sack_rate", "expl20_rate", "pass_share", "games"]


def recent_snapshots(games, observations, windows=(3, 5)):
    sides = game_sides(games)
    if observations.duplicated(["game_id", "team"]).any():
        raise ValueError("Raw counts must have unique game/team keys")
    obs = {(r.game_id, r.team): np.array([getattr(r, c) for c in COUNTS], dtype=float)
           for r in observations.itertuples()}
    rows = []
    for _, team in sides.groupby(["team", "season"], sort=False):
        history = deque(maxlen=max(windows))
        for _, simultaneous in team.groupby("kick", sort=True):
            for r in simultaneous.itertuples():
                f = {"game_id": r.game_id, "team": r.team}
                for window in windows:
                    previous = list(history)[-window:]
                    known = [o for _, o in previous if o is not None]
                    values = {k: np.nan for k in RATES}
                    values["games"] = float(len(known))
                    if known:
                        c = dict(zip(COUNTS, np.nansum(known, axis=0)))
                        if c["pass_plays"] > 0:
                            if c["pass_yards_observed"] == c["pass_plays"]:
                                values["pass_ypp"] = c["pass_yards"] / c["pass_plays"]
                            values["sack_rate"] = c["sacks"] / c["pass_plays"]
                        if c["rush_plays"] > 0 and c["rush_yards_observed"] == c["rush_plays"]:
                            values["rush_ypp"] = c["rush_yards"] / c["rush_plays"]
                        if c["plays"] > 0:
                            values["pass_share"] = c["pass_plays"] / c["plays"]
                            values["expl20_rate"] = c["expl20"] / c["plays"]
                    f.update({f"r{window}_{key}": value for key, value in values.items()})
                    dated = [kick for kick, observation in previous if observation is not None]
                    f[f"r{window}_source_kick"] = max(dated) if dated else pd.NaT
                    if dated and max(dated) >= r.kick:
                        raise AssertionError("Recent form includes a non-prior game")
                rows.append(f)
            for r in simultaneous.itertuples():
                if r.completed:
                    history.append((r.kick, obs.get((r.game_id, r.team))))
    return pd.DataFrame(rows)


def raw_counts(con, source, team_column):
    return con.execute(f'''SELECT game_id, {team_column} AS team,
        count(*) AS plays,
        count(*) FILTER (WHERE "pass") AS pass_plays,
        sum(yards_gained) FILTER (WHERE "pass") AS pass_yards,
        count(yards_gained) FILTER (WHERE "pass") AS pass_yards_observed,
        count(*) FILTER (WHERE rush) AS rush_plays,
        sum(yards_gained) FILTER (WHERE rush) AS rush_yards,
        count(yards_gained) FILTER (WHERE rush) AS rush_yards_observed,
        count(*) FILTER (WHERE "pass" AND sack) AS sacks,
        count(*) FILTER (WHERE yards_gained >= 20) AS expl20
        FROM {source} WHERE period <= 4 AND (rush OR "pass")
        GROUP BY game_id, {team_column} ORDER BY game_id, team''').fetchdf()


def matchup_features(frame, windows=(3, 5)):
    out = frame.copy()
    for side, other in (("h", "a"), ("a", "h")):
        for w in windows:
            own, defense = f"{side}_o_r{w}_", f"{other}_d_r{w}_"
            prefix = f"{side}_match_r{w}_"
            for stat in ("pass_ypp", "rush_ypp", "sack_rate", "expl20_rate"):
                out[prefix + stat + "_diff"] = out[own+stat] - out[defense+stat]
            out[prefix+"pass_style"] = out[own+"pass_share"] * out[defense+"pass_ypp"]
            out[prefix+"rush_style"] = (1-out[own+"pass_share"]) * out[defense+"rush_ypp"]
    return out


def build(input_dir):
    master = pd.read_parquet(input_dir / "game_scalars.parquet")
    paths = sorted(PBP.glob("pbp_*.parquet"))
    extra = CFBD / f"pregame_plays_{CURRENT_SEASON}.parquet"
    digests = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths + ([extra] if extra.exists() else [])}
    with duckdb.connect() as con:
        schedule_path = input_dir / "snapshot_schedule.parquet"
        if schedule_path.exists():
            games = pd.read_parquet(schedule_path)
        else:
            games = load_games(con, fbs_only=False)
            games.to_parquet(schedule_path, index=False)
        keep = 'game_id, pos_team, def_pos_team, period, "pass", rush, yards_gained, sack'
        con.execute(f"CREATE TEMP TABLE recent_pbp AS SELECT {keep} FROM read_parquet(?, union_by_name=true)", [[str(p) for p in paths]])
        if extra.exists():
            con.execute(f"CREATE TEMP TABLE recent_extra AS SELECT {keep} FROM read_parquet(?)", [str(extra)])
            con.execute("""CREATE TEMP VIEW recent_source AS SELECT * FROM recent_pbp
                UNION ALL BY NAME SELECT * FROM recent_extra
                WHERE game_id NOT IN (SELECT DISTINCT game_id FROM recent_pbp)""")
            source = "recent_source"
        else:
            source = "recent_pbp"
        snapshots = []
        provenance = []
        for prefix, team_col in (("o_", "pos_team"), ("d_", "def_pos_team")):
            counts = raw_counts(con, source, team_col)
            counts.to_parquet(input_dir / f"recent_{prefix}counts.parquet", index=False)
            snap = recent_snapshots(games, counts)
            date_cols = [c for c in snap if c.endswith("source_kick")]
            provenance.append(snap[["game_id", "team"]+date_cols].assign(direction=prefix))
            snap = snap.drop(columns=date_cols)
            snap = snap.rename(columns={c: prefix+c for c in snap if c not in {"game_id", "team"}})
            snapshots.append(snap)
    out = master[["game_id", "home_team", "away_team"]].copy()
    for side, prefix in (("home", "h_"), ("away", "a_")):
        for snap in snapshots:
            block = snap.rename(columns={"team": side+"_team", **{c: prefix+c for c in snap if c not in {"game_id", "team"}}})
            out = out.merge(block, on=["game_id", side+"_team"], how="left", validate="one_to_one")
    out = matchup_features(out).drop(columns=["home_team", "away_team"])
    if len(out) != len(master) or out.game_id.duplicated().any():
        raise AssertionError("Recent-form rows do not match the full model schedule")
    now = pd.Timestamp.now(tz="UTC")
    future_ids = set(games.loc[(games.season==CURRENT_SEASON) & (games.kick>now), "game_id"])
    for side, prefix in (("home", "h_"), ("away", "a_")):
        f = master[["game_id", side+"_team"]].merge(out[out.game_id.isin(future_ids)], on="game_id")
        cols = [c for c in out if c.startswith(prefix) and "_match_" not in c]
        if f.groupby(side+"_team")[cols].nunique().gt(1).any().any():
            raise AssertionError("Unplayed targets advance recent-form windows")
    out.to_parquet(input_dir / "recent_form.parquet", index=False)
    pd.concat(provenance,ignore_index=True).to_parquet(input_dir / "recent_form_provenance.parquet", index=False)
    report = {"generated_at": now.isoformat(), "rows": len(out), "new_columns": len(out.columns)-1,
              "future_rows": int(out.game_id.isin(future_ids).sum()), "pbp_sha256": digests,
              "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "snapshot_schedule_sha256": hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
              "definition": "Three/five prior completed scheduled games within season; missing feeds occupy slots; regular plus postseason, regulation plays only",
              "missing_by_column": {c: int(out[c].isna().sum()) for c in out if c!="game_id"}}
    (input_dir / "recent_form_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in {"pbp_sha256","missing_by_column"}},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.input_dir)
