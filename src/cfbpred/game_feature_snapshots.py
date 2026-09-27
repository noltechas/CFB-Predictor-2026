"""Game-keyed, strictly prior observations for historical and future targets.

Week labels are metadata, never join keys or a clock. Unplayed targets emit
the current snapshot without advancing any statistical accumulator.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd


def game_sides(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    if "kick" not in g:
        g["kick"] = pd.to_datetime(g["start_date"], format="mixed", utc=True)
    if g.game_id.duplicated().any() or g.kick.isna().any():
        raise ValueError("Game snapshots require unique IDs and known kickoff dates")
    final = g.home_points.notna() & g.away_points.notna()
    if "completed" in g:
        final &= g.completed.fillna(False)
    g["completed"] = final
    return pd.concat([
        g[["game_id", "season", "week_c", "kick", "completed", f"{side}_team"]]
        .rename(columns={f"{side}_team": "team"}).assign(home_away=side)
        for side in ("home", "away")
    ], ignore_index=True).sort_values(["kick", "game_id", "home_away"])


def player_snapshots(games: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Participation and career counts; athlete careers advance globally in time.

Team-alphabetical traversal leaks a transfer's later career into earlier games
at a different school. A single chronological traversal avoids that problem.
"""
    from cfbpred.player_features import VOL_SPECS, _num
    sides = game_sides(games)
    mapping = sides[sides.completed][["game_id", "home_away", "team", "season"]]
    raw = raw.drop(columns=["season", "week_c"], errors="ignore").merge(
        mapping, on=["game_id", "home_away"], how="inner", validate="many_to_one")
    raw = raw[raw.athlete_id.notna()].copy()
    raw["athlete_id"] = raw.athlete_id.astype(str)
    parts = []
    for category, stat, role in VOL_SPECS:
        d = raw[(raw.category == category) & (raw.stat_type == stat)].copy()
        d["vol"] = _num(d.stat).clip(lower=0).fillna(0)
        d["role"] = role
        parts.append(d[["game_id", "team", "athlete_id", "role", "vol"]])
    v = pd.concat(parts, ignore_index=True)
    v = v.groupby(["game_id", "team", "athlete_id", "role"], as_index=False).vol.sum()
    observations = {key: list(d.itertuples(index=False))
                    for key, d in v.groupby(["game_id", "team"], sort=False)}
    career_g, career_v = defaultdict(int), defaultdict(float)
    states = {}
    rows = []
    # Emit every target at a timestamp before consuming any result there.
    for _, simultaneous in sides.groupby("kick", sort=True):
        for r in simultaneous.itertuples():
            state = states.setdefault((r.team, r.season), {
                "volume": defaultdict(float), "active": {}, "tops_before_last": {}})
            volumes, active = state["volume"], state["active"]
            f = {"game_id": r.game_id, "team": r.team}
            if volumes:
                def leader(role):
                    candidates = [(a, vol) for (a, rl), vol in volumes.items() if rl == role]
                    return max(candidates, key=lambda x: x[1])[0] if candidates else None
                for role, field in (("qb", "qb"), ("rb", "rb"), ("wr", "wr"), ("k", "k")):
                    athlete = leader(role)
                    if athlete is not None:
                        f[f"cx_{field}_gms"] = career_g[athlete] / 30.0
                        if role == "qb":
                            f["cx_qb_vol"] = career_v[(athlete, role)] / 5000.0
                if active:
                    f["cx_active_gms"] = sum(career_g[a] * vol for a, vol in active.items()) / (sum(active.values()) or 1) / 30
                defense = sorted([(a, vol) for (a, role), vol in volumes.items() if role == "dfn"], key=lambda x: -x[1])[:5]
                if defense:
                    f["cx_def_gms"] = np.mean([career_g[a] for a, _ in defense]) / 30
                tops = state["tops_before_last"]
                if tops:
                    production = defaultdict(float)
                    for (a, role), vol in volumes.items():
                        if role in {"qb", "rb", "wr"}:
                            production[a] += vol
                    active_offense = state.get("active_offense", set())
                    for role in ("qb", "rb", "wr"):
                        f[f"av_{role}_played_last"] = float(tops[role] in active_offense) if role in tops else np.nan
                    top5 = sorted(production, key=production.get, reverse=True)[:5]
                    f["av_top5_played_last"] = np.mean([a in active_offense for a in top5]) if top5 else np.nan
                    total = sum(production.values())
                    f["av_prod_share"] = sum(vol for a, vol in production.items() if a in active_offense) / total if total else np.nan
                    f["av_known"] = 1.0
            rows.append(f)
        for r in simultaneous.itertuples():
            obs = observations.get((r.game_id, r.team), []) if r.completed else []
            if not obs:
                continue
            state = states[(r.team, r.season)]
            volume = state["volume"]
            state["tops_before_last"] = {}
            for role in ("qb", "rb", "wr"):
                candidates = [(a, vol) for (a, rl), vol in volume.items() if rl == role]
                if candidates:
                    state["tops_before_last"][role] = max(candidates, key=lambda x: x[1])[0]
            active = defaultdict(float)
            for o in obs:
                volume[(o.athlete_id, o.role)] += o.vol
                career_v[(o.athlete_id, o.role)] += o.vol
                active[o.athlete_id] += o.vol
            for a in active:
                career_g[a] += 1
            state["active"] = active
            state["active_offense"] = {o.athlete_id for o in obs if o.role in {"qb", "rb", "wr"}}
    cols = ["game_id", "team", "av_qb_played_last", "av_rb_played_last", "av_wr_played_last",
            "av_top5_played_last", "av_prod_share", "av_known", "cx_qb_gms", "cx_qb_vol",
            "cx_rb_gms", "cx_wr_gms", "cx_k_gms", "cx_active_gms", "cx_def_gms"]
    return pd.DataFrame(rows).reindex(columns=cols)


def pbp_snapshots(games: pd.DataFrame, con, path, extra_path=None) -> tuple:
    """Raw PBP rates and rolling primary-QB counts, emitted for every game."""
    from cfbpred.features_raw import PBP_AGG, QB_WINDOW, cum_rates
    sides = game_sides(games)
    source = f"read_parquet('{path}')"
    if extra_path is not None and extra_path.exists():
        # The repaired release takes precedence once a season appears there.
        con.execute(f"""CREATE OR REPLACE TEMP VIEW pregame_pbp_source AS
            SELECT * FROM {source} UNION ALL BY NAME
            SELECT * FROM read_parquet('{extra_path}') WHERE game_id NOT IN
                (SELECT DISTINCT game_id FROM {source})""")
        source = "pregame_pbp_source"
    rates = []
    for side, prefix in (("pos_team", "o_"), ("def_pos_team", "d_")):
        sql = PBP_AGG.format(side=side, pbp=path).replace(
            'SELECT "year" AS season,', 'SELECT game_id, "year" AS season,').replace(
            "GROUP BY 1, 2, 3", "GROUP BY 1, 2, 3, 4")
        sql = sql.replace(f"read_parquet('{path}')", source)
        agg = con.execute(sql).fetchdf().drop(columns=["season", "week"])
        joined = sides.merge(agg, on=["game_id", "team"], how="left", validate="one_to_one")
        stats = [c for c in agg if c not in {"game_id", "team"}]
        joined.loc[~joined.completed, stats] = np.nan
        joined[stats] = joined[stats].fillna(0)
        joined = joined.sort_values(["team", "season", "kick", "game_id"])
        joined["week"] = joined.groupby(["team", "season"]).cumcount() + 1
        out = cum_rates(joined[["team", "season", "week"] + stats])
        out = out.merge(joined[["team", "season", "week", "game_id"]], on=["team", "season", "week"], validate="one_to_one")
        feature_cols = [c for c in out if c not in {"game_id", "team", "season", "week"}]
        out.loc[out.plays_cum.fillna(0) == 0, feature_cols] = np.nan
        rates.append(out[["game_id", "team"] + feature_cols].rename(columns={c: prefix + c for c in feature_cols}))
    q = con.execute(f'''SELECT game_id, pos_team AS team, passer_player_name AS qb,
        count(*) AS att, sum(yards_gained) AS yds,
        count(*) FILTER (WHERE "int") AS ints,
        count(*) FILTER (WHERE sack) AS sacks,
        count(*) FILTER (WHERE touchdown) AS tds
        FROM {source} WHERE "pass" AND passer_player_name IS NOT NULL
        AND period <= 4 GROUP BY 1,2,3
        ORDER BY game_id, team, att DESC, qb''').fetchdf()
    obs = {key: d for key, d in q.groupby(["game_id", "team"], sort=False)}
    rows = []
    for _, team in sides.groupby(["team", "season"], sort=False):
        history = deque(maxlen=QB_WINDOW)
        for r in team.sort_values(["kick", "game_id"]).itertuples():
            f = {"game_id": r.game_id, "team": r.team}
            if history:
                prior = pd.concat(list(history))
                starter = prior.groupby("qb").att.sum().idxmax()
                p = prior[prior.qb == starter]
                att = p.att.sum()
                top = lambda d: d.loc[d.att.idxmax(), "qb"]
                f.update(qb_att_pg=att / len(p), qb_ypa=p.yds.sum() / att,
                         qb_int_rate=p.ints.sum() / att, qb_sack_rate=p.sacks.sum() / att,
                         qb_td_rate=p.tds.sum() / att,
                         qb_changed=float(len(history) > 1 and top(history[-1]) != top(history[-2])), qb_known=1.0)
            rows.append(f)
            if r.completed and (r.game_id, r.team) in obs:
                history.append(obs[(r.game_id, r.team)])
    qbcols = ["game_id", "team", "qb_att_pg", "qb_ypa", "qb_int_rate", "qb_sack_rate", "qb_td_rate", "qb_changed", "qb_known"]
    return rates[0], rates[1], pd.DataFrame(rows).reindex(columns=qbcols)
