"""Preseason passing experience of rostered players, from public raw boxes.

The most experienced rostered passer is a preseason proxy, not a confirmed
starter. Only the preceding four seasons contribute; athlete IDs follow
transfers. No target-season performance chooses the player or their features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FIELDS = ["roster_known", "passer_known", "passers", "att_log", "games",
          "ypa", "comp_pct", "td_rate", "int_rate"]


def annual_passing(raw):
    raw = raw[raw.category.eq("passing") & raw.stat_type.isin(["C/ATT","YDS","TD","INT"])].copy()
    raw = raw.dropna(subset=["athlete_id"])
    raw["athlete_id"] = raw.athlete_id.astype(str)
    keys = ["season","game_id","athlete_id"]
    # The public file repeats some identical boxes under different week/type
    # metadata (especially postseason). Count a game observation only once.
    raw = raw.drop_duplicates(keys+["stat_type","stat"])
    if raw.duplicated(keys+["stat_type"]).any():
        raise ValueError("Conflicting athlete passing stat observations")
    games = raw.pivot(index=keys,columns="stat_type",values="stat").reset_index()
    pair = games["C/ATT"].astype(str).str.extract(r"^(\d+)/(\d+)$").astype(float)
    games["cmp"],games["att"] = pair[0],pair[1]
    for key,col in [("yds","YDS"),("td","TD"),("ints","INT")]:
        games[key] = pd.to_numeric(games[col],errors="coerce")
    # Each efficiency denominator has a fully observed numerator. Incomplete
    # source games stay out of this prior rather than silently becoming zero.
    games = games.dropna(subset=["cmp","att","yds","td","ints"])
    games = games[(games.att>0) & games.cmp.between(0,games.att)]
    annual = games.groupby(["season","athlete_id"],as_index=False).agg(
        cmp=("cmp","sum"),att=("att","sum"),yds=("yds","sum"),td=("td","sum"),
        ints=("ints","sum"),games=("game_id","nunique"))
    return annual


def roster_priors(targets, roster, annual):
    """One row per season/team, with explicit source provenance and unknowns."""
    targets = targets[["season","team"]].drop_duplicates()
    roster = roster.dropna(subset=["athlete_id"]).copy()
    roster["athlete_id"] = roster.athlete_id.astype(str)
    roster = roster.drop_duplicates(["season","team","athlete_id"])
    rows = []
    for year,tgt in targets.groupby("season",sort=True):
        prior = annual[annual.season.between(year-4,year-1)]
        counts = prior.groupby("athlete_id").agg(
            cmp=("cmp","sum"),att=("att","sum"),yds=("yds","sum"),td=("td","sum"),
            ints=("ints","sum"),games=("games","sum"),source_season=("season","max")).reset_index()
        ros = roster[roster.season.eq(year)]
        observed = ros.merge(counts,on="athlete_id",how="inner")
        covered = set(ros.team)
        groups = {t:f for t,f in observed.groupby("team")}
        for team in tgt.team:
            row = {"season":year,"team":team,**{f"pre_qb_{f}":np.nan for f in FIELDS},
                   "prior_passer_id":None,"source_season":np.nan}
            row["pre_qb_roster_known"] = float(team in covered)
            row["pre_qb_passer_known"] = 0.0
            if team in covered:
                players = groups.get(team)
                row["pre_qb_passers"] = 0.0 if players is None else float(len(players))
                if players is not None:
                    top = players.sort_values(["att","athlete_id"],ascending=[False,True]).iloc[0]
                    row.update(pre_qb_passer_known=1.0,pre_qb_att_log=np.log1p(top.att),
                               pre_qb_games=top.games/30,pre_qb_ypa=top.yds/top.att,
                               pre_qb_comp_pct=top.cmp/top.att,pre_qb_td_rate=top.td/top.att,
                               pre_qb_int_rate=top.ints/top.att,prior_passer_id=top.athlete_id,
                               source_season=int(top.source_season))
            rows.append(row)
    return pd.DataFrame(rows)


def attach(base, features):
    features = features.set_index("game_id")
    if features.index.duplicated().any() or not set(base["G_gid"]).issubset(features.index):
        raise ValueError("Every target must have exactly one preseason prior row")
    cols = sorted(c for c in features if c.startswith(("h_pre_qb_","a_pre_qb_")))
    from cfbpred.train import BANNED_SUBSTRINGS
    if len(cols)!=2*len(FIELDS) or any(any(b in c.lower() for b in BANNED_SUBSTRINGS) for c in cols):
        raise ValueError("Invalid QB prior feature contract")
    out = base.copy()
    combined = base["scalar_cols"] + cols
    if len(set(combined)) != len(combined):
        raise ValueError("QB priors are already attached")
    order = np.argsort(combined)
    out["scalar_cols"] = [combined[i] for i in order]
    values = np.concatenate([base["G_raw_scalars"],features.loc[base["G_gid"],cols].to_numpy(np.float32)],axis=1)
    out["G_raw_scalars"] = values[:,order]
    return out
