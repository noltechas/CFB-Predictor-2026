"""Rebuild identity-dependent legacy features from the corrected player store.

This keeps the new player tower from also consuming contaminated old career,
returner, or portal features through its scalar branch. Original checkpoints
and their frozen feature stores remain reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfbpred.game_feature_snapshots import player_snapshots
from cfbpred.qb_roster_prior import roster_priors
from cfbpred.pull_espn_roster import team_alias_map


def corrected_scalars(master, schedule, player_path):
    roster = pd.read_parquet(player_path/"player_seasons.parquet")
    boxes = pd.read_parquet(player_path/"clean_player_boxes.parquet")
    aliases = team_alias_map()
    schedule = schedule.copy(); out = master.copy()
    for side in ["home", "away"]:
        schedule[side+"_team"] = schedule[side+"_team"].replace(aliases)
    sides = pd.concat([schedule[["game_id", side+"_team"]].rename(columns={side+"_team":"team"}).assign(home_away=side)
                       for side in ["home","away"]])
    lines = []
    for stat, cat, typ in [("pass_yds","passing","YDS"),("rush_yds","rushing","YDS"),
                            ("rec_yds","receiving","YDS"),("tackles","defensive","TOT")]:
        b = boxes[boxes[stat+"_observations"].gt(0)][["game_id","team","person_id",stat]].copy()
        b = b.rename(columns={"person_id":"athlete_id",stat:"stat"}).assign(category=cat,stat_type=typ)
        lines.append(b)
    # Kicker career box presence is available from made+attempt observations.
    b = boxes[boxes.fgm_observations.gt(0)][["game_id","team","person_id","fgm","xpm"]].copy()
    b["stat"] = b.fgm*3+b.xpm
    lines.append(b.rename(columns={"person_id":"athlete_id"}).assign(category="kicking",stat_type="PTS"))
    raw = pd.concat(lines,ignore_index=True).merge(sides,on=["game_id","team"],validate="many_to_one")
    dynamic = player_snapshots(schedule,raw.drop(columns="team"))
    annual = boxes.groupby(["season","team","person_id"])[["pass_yds","rush_yds","rec_yds","tackles"]].sum().reset_index()
    rows = []
    previous_rosters = {(y,t):set(d.person_id) for (y,t),d in roster.groupby(["season","team"])}
    for year, rs in roster.groupby("season"):
        previous = annual[annual.season.eq(year-1)]
        for team, ros in rs.groupby("team"):
            people = set(ros.loc[~ros.identity_uncertain,"person_id"])
            old_roster = previous_rosters.get((year-1,team),set())
            own = previous[previous.team.eq(team)]
            returned = own[own.person_id.isin(people)]
            imported = previous[previous.person_id.isin(people)&previous.team.ne(team)]
            row = dict(season=year,team=team,n_players=len(ros),
                       returning_share=len(people&old_roster)/len(old_roster) if old_roster else np.nan)
            for col,key,floor in [("pass_yds","pass",500),("rush_yds","rush",500),("rec_yds","recv",500),("tackles","tkl",300)]:
                total = own[col].sum()
                row["retp_"+key] = returned[col].sum()/total if total>=floor else np.nan
                if key!="tkl":row["port_"+key] = np.log1p(max(0,imported[col].sum()))
            leader = own.sort_values(["pass_yds","person_id"]).tail(1)
            row["retp_qb1"] = float(leader.person_id.iloc[0] in people) if len(leader) and leader.pass_yds.iloc[0]>=500 else np.nan
            rows.append(row)
    static = pd.DataFrame(rows)
    for side,short in [("home","h"),("away","a")]:
        key = out[["game_id","season",side+"_team"]].rename(columns={side+"_team":"team"})
        key["team"] = key.team.replace(aliases)
        joined = key.merge(dynamic,on=["game_id","team"],how="left",validate="one_to_one").merge(
            static,on=["season","team"],how="left",validate="many_to_one")
        for col in list(dynamic.columns)+list(static.columns):
            if col in ["game_id","season","team"]:continue
            dest = short+"_"+col
            if dest in out:out[dest] = joined[col].to_numpy()
        # Historical class and body measurements can be overwritten upstream.
        # Keep their schema positions for compatibility but never feed them as
        # historical facts into this corrected experiment.
        for col in out:
            if col.startswith(short+"_") and (col.endswith(("_wt","_ht")) or col==short+"_upper_share"):
                out[col] = np.nan
    # Rebuild the retained combination's preseason passing prior with canonical
    # people, complete box components, and the SAME fixed four-season window.
    fields = ["cmp","att","pass_yds","pass_td","pass_int"]
    eligible = boxes[[f+"_observations" for f in fields]].gt(0).all(axis=1) & boxes.att.gt(0) & boxes.cmp.between(0,boxes.att)
    b = boxes[eligible]
    passing = b.groupby(["season","person_id"]).agg(cmp=("cmp","sum"),att=("att","sum"),yds=("pass_yds","sum"),
                  td=("pass_td","sum"),ints=("pass_int","sum"),games=("game_id","nunique")).reset_index().rename(columns={"person_id":"athlete_id"})
    ros = roster[~roster.identity_uncertain][["season","team","person_id"]].rename(columns={"person_id":"athlete_id"})
    priors = roster_priors(roster[["season","team"]],ros,passing)
    qb = out[["game_id"]].copy()
    for side,short in [("home","h"),("away","a")]:
        key = out[["season",side+"_team"]].rename(columns={side+"_team":"team"});key["team"] = key.team.replace(aliases)
        joined = key.merge(priors,on=["season","team"],how="left",validate="many_to_one")
        for c in priors:
            if c.startswith("pre_qb_"):qb[short+"_"+c] = joined[c].to_numpy()
    return out,qb
