"""Current, schedule-independent team inputs for the rankings readout.

The existing game feature rows supply season constants only. Results, history,
box scores, participation and play counts are rebuilt at ONE cutoff, including
teams on a bye. This does not modify training tables or run a fitted rating.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from cfbpred.config import CFBD, CURRENT_SEASON, FEAT, PBP, PQ
from cfbpred.features_raw import (N_HIST, PBP_AGG, QB_WINDOW, cum_rates,
                                  load_games, team_log)
from cfbpred.player_features import (availability_features, build_box_features,
                                     career_features)

DYNAMIC_PREFIXES = ("o_", "d_", "bx_", "qb_", "av_", "cx_", "sos_")
DYNAMIC_NAMES = {"gp", "winpct", "ppg_for", "ppg_against", "box_gp"}
TARGET_WEEK = 10000  # an emission marker, never a model input


def completed_before(games: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    final = games["completed"].fillna(False).astype(bool) if "completed" in games else True
    return games[(games.kick < cutoff) & games.home_points.notna()
                 & games.away_points.notna() & final].copy()


def catalog_for(games: pd.DataFrame, season: int, ids: dict) -> pd.DataFrame:
    parts = []
    for side in ("home", "away"):
        part = games.loc[games.season == season,
                         ["kick", f"{side}_team", f"{side}_division", f"{side}_conference"]]
        parts.append(part.rename(columns={f"{side}_team": "team",
                                           f"{side}_division": "division",
                                           f"{side}_conference": "conference"}))
    out = (pd.concat(parts).sort_values("kick").drop_duplicates("team", keep="last"))
    out = out[out.division.isin(["fbs", "fcs"])].copy()
    out["team_id"] = out.team.map(ids)
    if out.team_id.isna().any() or out.team_id.duplicated().any():
        raise ValueError("Division I team IDs are missing or ambiguous")
    out["team_id"] = out.team_id.astype(int)
    return out.sort_values("team").reset_index(drop=True)


def result_inputs(catalog: pd.DataFrame, completed: pd.DataFrame, vocab: dict,
                  season: int, cutoff: pd.Timestamp, priors: dict):
    log = team_log(completed, keep_teams=set(catalog.team))
    histories = np.zeros((len(catalog), N_HIST, 7), np.float32)
    opponents = np.zeros((len(catalog), N_HIST), np.int64)
    rows = []
    for i, team in enumerate(catalog.team):
        d = log[log.team == team].sort_values(["kick", "game_id"])
        current = d[d.season == season]
        gp = len(current)
        wins = int((current.pf > current.pa).sum())
        losses = int((current.pf < current.pa).sum())
        row = {"team": team, "gp": float(gp), "wins": wins, "losses": losses,
               "ties": gp - wins - losses, "winpct": wins / gp if gp else 0.0,
               "ppg_for": current.pf.mean(), "ppg_against": current.pa.mean(),
               "sos_fbs_share": (current.opp_div == "fbs").mean() if gp else np.nan}
        for feature, key in (("sos_talent", "talent_composite"),
                             ("sos_recruit", "recruit_pts_4c")):
            values = [priors.get(t, {}).get(key, np.nan) for t in current.opp]
            row[feature] = pd.Series(values, dtype=float).mean()
        for slot, r in enumerate(d.tail(N_HIST).iloc[::-1].itertuples()):
            histories[i, slot] = [r.pf / 30, r.pa / 30, r.was_home,
                                  float(r.opp_div == "fbs"),
                                  min((cutoff - r.kick).total_seconds() / 86400, 400) / 100,
                                  float(r.season == season), 1]
            opponents[i, slot] = vocab["team"].get(r.opp, 0)
        row["history_games"] = int(histories[i, :, -1].sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("team"), histories, opponents


def pbp_inputs(completed: pd.DataFrame, season: int, con) -> pd.DataFrame:
    path = PBP / f"pbp_{season}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["team"])
    con.register("ranking_completed", completed[["game_id"]])
    parts = []
    for side, prefix in (("pos_team", "o_"), ("def_pos_team", "d_")):
        sql = PBP_AGG.format(side=side, pbp=path).replace(
            "WHERE period", "WHERE game_id IN (SELECT game_id FROM ranking_completed) AND period")
        agg = con.execute(sql).fetchdf()
        if agg.empty:
            continue
        target = agg.drop_duplicates(["team", "season"]).copy()
        target["week"] = TARGET_WEEK
        for col in set(agg) - {"team", "season", "week"}:
            target[col] = 0
        rates = cum_rates(pd.concat([agg, target], ignore_index=True))
        rates = rates[rates.week == TARGET_WEEK].drop(columns=["season", "week"])
        parts.append(rates.set_index("team").add_prefix(prefix))
    q = con.execute(f"""
        SELECT pos_team AS team,
               CASE WHEN season_type = 'postseason' THEN week + 20 ELSE week END AS week,
               passer_player_name AS qb,
               count(*) AS att, sum(yards_gained) AS yds,
               count(*) FILTER (WHERE "int") AS ints,
               count(*) FILTER (WHERE sack) AS sacks,
               count(*) FILTER (WHERE touchdown) AS tds
        FROM read_parquet('{path}')
        WHERE game_id IN (SELECT game_id FROM ranking_completed)
          AND "pass" AND passer_player_name IS NOT NULL AND period <= 4
        GROUP BY 1,2,3
    """).fetchdf()
    qb_rows = []
    for team, d in q.groupby("team"):
        weeks = sorted(d.week.unique())[-QB_WINDOW:]
        p = d[d.week.isin(weeks)]
        starter = p.groupby("qb").att.sum().idxmax()
        p = p[p.qb == starter]
        def top(w):
            v = d[d.week == w]
            return v.loc[v.att.idxmax(), "qb"]
        att = p.att.sum()
        qb_rows.append({"team": team, "qb_att_pg": att / max(len(p), 1),
                        "qb_ypa": p.yds.sum() / max(att, 1),
                        "qb_int_rate": p.ints.sum() / max(att, 1),
                        "qb_sack_rate": p.sacks.sum() / max(att, 1),
                        "qb_td_rate": p.tds.sum() / max(att, 1),
                        "qb_changed": float(len(weeks) > 1 and top(weeks[-1]) != top(weeks[-2])),
                        "qb_known": 1.0})
    if qb_rows:
        parts.append(pd.DataFrame(qb_rows).set_index("team"))
    return pd.concat(parts, axis=1).reset_index() if parts else pd.DataFrame(columns=["team"])


def build_snapshot(season: int, cutoff: pd.Timestamp, *, preseason: bool = False,
                   reconstructed: bool = False, results_only: bool = False,
                   feature_dir: Path = FEAT, allow_short_history: bool = False,
                   schedule_path: Path | None = None) -> dict:
    feature_path = feature_dir / "game_scalars.parquet"
    feature_time = pd.Timestamp(feature_path.stat().st_mtime, unit="s", tz="UTC")
    if cutoff < feature_time and not preseason and not reconstructed:
        raise ValueError("Historical rankings require an archived feature snapshot; backfill is disabled")
    vocab = json.loads((feature_dir / "team_vocab.json").read_text())
    with duckdb.connect() as con:
        if schedule_path is None:
            games = load_games(con, fbs_only=False)
        else:
            # A replay's restored fixtures and result history must come from
            # the same frozen corpus as its scalar and player inputs.
            games = pd.read_parquet(schedule_path)
            games = games[games.game_id.gt(0) & games.week.notna()].copy()
            games["kick"] = pd.to_datetime(games.start_date, utc=True)
            games["week_c"] = np.where(games.season_type.eq("postseason"),
                                       games.week + 20, games.week)
            if games.game_id.duplicated().any():
                raise ValueError("Ranking schedule has duplicate game IDs")
            games = games.sort_values(["season", "kick", "game_id"])
        if preseason:
            opening = games[(games.season == season) & (
                games.home_division.isin(["fbs", "fcs"]) | games.away_division.isin(["fbs", "fcs"]))].kick.min()
            if pd.isna(opening):
                raise ValueError("No opening kickoff for preseason baseline")
            cutoff = min(cutoff, opening)
        status = pd.read_parquet(CFBD / "games.parquet", columns=["id", "completed"])
        known_final = status.drop_duplicates("id", keep="last").set_index("id").completed
        # Current games may have partial score columns. An explicit provider
        # final flag is required; a started game is not a ranking result.
        games["completed"] = games.game_id.map(known_final)
        games.loc[games.season < CURRENT_SEASON, "completed"] = True
        completed = completed_before(games, cutoff)
        if preseason:
            # A replay must never inherit this season's results, even if an
            # earlier lower-division game is present in the source corpus.
            completed = completed[completed.season < season]
        # Schedule IDs and classifications are season-specific. The reference
        # teams file also contains old classifications and must not override them.
        identity = pd.read_parquet(CFBD / "teams.parquet").sort_values("season_pulled")
        ids = dict(zip(identity.school, identity.id))
        schedule = pd.read_parquet(PQ / "schedule.parquet")
        for side in ("home", "away"):
            ids.update(dict(zip(schedule[f"{side}_team"], schedule[f"{side}_id"])))
            if f"{side}_id" in games:
                identified = games.dropna(subset=[f"{side}_id"])
                ids.update(dict(zip(identified[f"{side}_team"], identified[f"{side}_id"])))
        # Historical provider spellings refer to the same public team IDs.
        for old, current in {"Connecticut": "UConn", "Louisiana Monroe": "UL Monroe",
                             "Southern Mississippi": "Southern Miss",
                             "St. Francis (PA)": "Saint Francis",
                             "Savannah St": "Savannah State", "Appalachian State": "App State",
                             "UMass": "Massachusetts", "Sam Houston State": "Sam Houston",
                             "Southeastern Louisiana": "SE Louisiana",
                             "Presbyterian College": "Presbyterian", "Prairie View": "Prairie View A&M",
                             "UT San Antonio": "UTSA"}.items():
            if current in ids:
                ids.setdefault(old, ids[current])
        catalog = catalog_for(games, season, ids)
        if catalog.empty:
            raise ValueError("No Division I schedule for this season")

        g = pd.read_parquet(feature_path)
        g = g[g.season == season]
        sides = []
        for side, prefix in (("home", "h_"), ("away", "a_")):
            cols = [c for c in g if c.startswith(prefix)]
            sides.append(g[[f"{side}_team"] + cols].rename(columns={
                f"{side}_team": "team", **{c: c[2:] for c in cols}}))
        priors = pd.concat(sides).drop_duplicates("team").set_index("team")
        # Never copy a prior game's season-to-date inputs or game context.
        static = [c for c in priors if c not in DYNAMIC_NAMES
                  and not c.startswith(DYNAMIC_PREFIXES)
                  and c not in {"idx", "conf", "rest", "travel", "alt_delta"}]
        fields = priors.reindex(catalog.team)[static].copy()
        records, hist, opp = result_inputs(catalog, completed, vocab, season, cutoff,
                                           priors.to_dict("index"))
        fields = fields.join(records)

        current_games = completed[completed.season == season]
        if current_games.empty or results_only:
            fields["box_gp"] = 0.0
        else:
            print("Building current box-score and participation inputs...", flush=True)
            # Filters exclude uncompleted games even if a partial box score exists.
            raw = pd.read_parquet(CFBD / "player_game_stats.parquet")
            raw = raw[raw.game_id.isin(completed.game_id)]
            current_raw = raw[raw.season == season]
            targets = pd.DataFrame({"team": catalog.team, "season": season,
                                    "week_c": TARGET_WEEK, "game_id": -catalog.team_id})
            box = build_box_features(current_games, raw=current_raw, targets=targets)
            av = availability_features(current_games, raw=current_raw,
                                       target_season=season, target_week=TARGET_WEEK)
            cx = career_features(completed, raw=raw, target_season=season, target_week=TARGET_WEEK)
            for block in (box, av, cx):
                if block.empty:
                    continue
                block = block[(block.season == season) & (block.week_c == TARGET_WEEK)]
                block = block.drop(columns=["season", "week_c"]).set_index("team")
                fields = fields.join(block)
            pbp = pbp_inputs(current_games, season, con).set_index("team")
            fields = fields.join(pbp)
    # Missing blocks keep their training missingness flags. Unknown identities
    # get an explicit unranked row rather than the shared UNK team's rating.
    catalog["eligible"] = catalog.team.map(lambda t: t in vocab["team"] and t in priors.index)
    catalog["history_games"] = records.reindex(catalog.team).history_games.to_numpy()
    if not allow_short_history:
        catalog["eligible"] &= catalog.history_games >= 4
    for col in ("wins", "losses", "ties"):
        catalog[col] = records.reindex(catalog.team)[col].to_numpy().astype(int)
    return {"catalog": catalog, "fields": fields.reindex(catalog.team),
            "history": hist, "opponents": opp, "vocab": vocab,
            "feature_time": feature_time.isoformat(),
            "as_of": cutoff.isoformat(),
            "reconstructed": bool((preseason or reconstructed) and cutoff < feature_time),
            "latest_result": completed.kick.max().isoformat(),
            "source_week": int(current_games.week_c.max()) if len(current_games) else 0}
