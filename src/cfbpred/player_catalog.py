"""Auditable per-player preseason catalog; not yet a production model input.

Roster identity -> individual recruiting records -> prior box-score history.
Only provider ID joins are accepted. Missing, conflicting, and ambiguous
scouting links remain visible. Historical roster/rating publication times are
not supplied, so a season cutoff is not a guarantee of archived availability.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cfbpred.architecture_experiment import REFERENCE
from cfbpred.config import CFBD, MODELS, PQ
from cfbpred.pregame_experiments import save_json, sha
from cfbpred.pull_espn_roster import team_alias_map

OUTPUT=PQ/"player_catalog"


def normalize_id(series):
    return series.astype("string").str.replace(r"\.0$","",regex=True)


def link_recruits(roster,recruits):
    """Return accepted candidates and separate conflicting provider links."""
    r=roster.reset_index(drop=True).copy();r["roster_row"]=np.arange(len(r))
    r["athlete_id"]=normalize_id(r.athlete_id)
    q=recruits.copy();q["recruit_id"]=normalize_id(q.id);q["recruit_athlete_id"]=normalize_id(q.athleteId)
    q=q.rename(columns={"year":"recruit_year"})
    direct=r[["roster_row","season","athlete_id"]].dropna(subset=["athlete_id"]).merge(
        q.dropna(subset=["recruit_athlete_id"]),left_on="athlete_id",right_on="recruit_athlete_id",how="inner")
    direct["match_method"]="athlete_id"
    ids=r[["roster_row","season","athlete_id","recruitIds"]].explode("recruitIds").dropna(subset=["recruitIds"])
    ids["recruit_id"]=normalize_id(ids.recruitIds)
    linked=ids.drop(columns="recruitIds").merge(q,on="recruit_id",how="inner")
    linked["match_method"]="roster_recruit_id"
    conflict=linked[linked.recruit_athlete_id.notna() & linked.recruit_athlete_id.ne(linked.athlete_id)].copy()
    linked=linked.drop(index=conflict.index)
    accepted=pd.concat([direct,linked],ignore_index=True)
    accepted=accepted[accepted.recruit_year<=accepted.season]
    accepted=accepted.sort_values("match_method").drop_duplicates(["roster_row","recruit_id","recruit_year","recruitType"])
    return r,accepted,conflict


def choose_scouting(roster,accepted):
    """Expose one unambiguous record per type; preserve all candidates apart."""
    out=roster.copy()
    for kind,prefix in [("HighSchool","hs"),("JUCO","juco"),("PrepSchool","prep")]:
        rows=accepted[accepted.recruitType.eq(kind)]
        counts=rows.groupby("roster_row").size()
        out[f"{prefix}_recruit_matches"]=out.roster_row.map(counts).fillna(0).astype(int)
        single=rows[rows.roster_row.map(counts).eq(1)].set_index("roster_row")
        for source,dest in [("rating","rating"),("stars","stars"),("ranking","rank"),
                            ("recruit_year","recruit_year"),("recruit_id","recruit_id"),
                            ("height","height"),("weight","weight")]:
            out[f"{prefix}_{dest}"]=out.roster_row.map(single[source]) if source in single else np.nan
        out[f"{prefix}_scouting_known"]=out[f"{prefix}_rating"].notna()
    out["any_scouting_known"]=out[["hs_scouting_known","juco_scouting_known","prep_scouting_known"]].any(axis=1)
    return out


def attach_history(catalog,history):
    """All observed prior college programs follow the same athlete ID."""
    out=catalog.copy()
    history=history.copy();history["athlete_id"]=normalize_id(history.athlete_id)
    blocks=[]
    for year,ros in out.groupby("season",sort=True):
        prior=history[history.source_season<year]
        counts=prior.groupby("athlete_id").agg(prior_box_games=("game_id","nunique"),
                    last_prior_season=("source_season","max"),first_prior_season=("source_season","min"))
        production=prior[prior.source.eq("production")].groupby("athlete_id").game_id.nunique()
        programs=prior.groupby("athlete_id").source_team.agg(lambda s:sorted(set(s.dropna())))
        r=ros.copy()
        for c in counts:
            r[c]=r.athlete_id.map(counts[c])
        r["prior_production_box_games"]=r.athlete_id.map(production).fillna(0).astype(int)
        r["prior_box_games"]=r.prior_box_games.fillna(0).astype(int)
        r["prior_box_observed"]=r.prior_box_games>0
        r["prior_programs"]=r.athlete_id.map(programs).apply(lambda x:x if isinstance(x,list) else [])
        r["observed_at_other_program"]= [any(t!=own for t in old) for own,old in zip(r.team,r.prior_programs)]
        blocks.append(r)
    return pd.concat(blocks,ignore_index=True)


def flag_identity_risks(catalog):
    """Quarantine suspicious ID continuity, without asserting every flag is wrong.

    An observed current freshman sharing an ID with 2017 college boxes exposed
    provider ID reuse. Long gaps/spans can also reflect missing feeds or unusual
    eligibility; they require review, not invented replacement identities.
    """
    out=catalog.copy()
    years=out[["athlete_id","season"]].drop_duplicates().sort_values(["athlete_id","season"])
    years["gap"]=years.groupby("athlete_id").season.diff()
    years["roster_gap_requires_review"]=years.gap.ge(4).groupby(years.athlete_id).cummax()
    out=out.merge(years[["athlete_id","season","roster_gap_requires_review"]],on=["athlete_id","season"],validate="many_to_one")
    out["multiple_rosters_same_season"]=out.groupby(["season","athlete_id"]).team.transform("nunique")>1
    out["long_career_requires_review"]=(out.season-out.first_prior_season).ge(9)
    out["identity_review_required"]=out[["roster_gap_requires_review","multiple_rosters_same_season","long_career_requires_review"]].any(axis=1)
    out["scouting_usable_without_identity_review"]=out.any_scouting_known & ~out.identity_review_required
    out["history_usable_without_identity_review"]=out.prior_box_observed & ~out.identity_review_required
    return out


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    roster=pd.read_parquet(CFBD/"roster.parquet").rename(columns={"id":"athlete_id","season_pulled":"season"})
    roster=roster[["athlete_id","season","team","firstName","lastName","position","year","recruitIds","source"]]
    aliases=team_alias_map();roster["team"]=roster.team.replace(aliases)
    roster=roster.drop_duplicates(["season","team","athlete_id"])
    recruits=pd.read_parquet(CFBD/"recruiting_players.parquet")
    r,accepted,conflicts=link_recruits(roster,recruits)
    catalog=choose_scouting(r,accepted)
    games=pd.read_parquet(REFERENCE/"inputs/snapshot_schedule.parquet")
    parts=[]
    for side in ["home","away"]:
        d=games[["game_id","season",side+"_team"]].rename(columns={"season":"source_season",side+"_team":"source_team"})
        d["home_away"]=side;parts.append(d)
    mapping=pd.concat(parts,ignore_index=True)
    mapping["source_team"]=mapping.source_team.replace(aliases)
    frames=[]
    with duckdb.connect() as con:
        for filename,label in [("player_game_stats.parquet","production"),("player_game_stats_fcs.parquet","extra_fcs")]:
            path=CFBD/filename
            if not path.exists():continue
            d=con.execute("SELECT DISTINCT game_id,home_away,athlete_id,season AS recorded_season FROM read_parquet(?) WHERE athlete_id IS NOT NULL AND season<=2025",[str(path)]).fetchdf()
            d=d.merge(mapping,on=["game_id","home_away"],how="left",validate="many_to_one")
            # Unknown schedule mappings retain the raw source's season, but
            # cannot establish which former college produced those stats.
            d["source_season"]=d.source_season.fillna(d.recorded_season).astype(int)
            d["source"]=label
            frames.append(d)
    history=pd.concat(frames,ignore_index=True)
    catalog=flag_identity_risks(attach_history(catalog,history))
    if not (catalog.loc[catalog.last_prior_season.notna(),"last_prior_season"] < catalog.loc[catalog.last_prior_season.notna(),"season"]).all():
        raise ValueError("Future player history entered catalog")
    catalog.to_parquet(OUTPUT/"player_seasons.parquet",index=False)
    history.to_parquet(OUTPUT/"player_game_history_index.parquet",index=False)
    accepted.to_parquet(OUTPUT/"recruit_links.parquet",index=False)
    conflicts.to_parquet(OUTPUT/"conflicting_recruit_links.parquet",index=False)
    catalog[catalog.identity_review_required].to_parquet(OUTPUT/"identity_review_required.parquet",index=False)
    master=pd.read_parquet(REFERENCE/"inputs/game_scalars.parquet")
    divisions=pd.concat([master[["season",side+"_team",side+"_division"]].rename(
        columns={side+"_team":"team",side+"_division":"division"}) for side in ["home","away"]]).drop_duplicates()
    divisions["team"]=divisions.team.replace(aliases)
    fbs=set(divisions.loc[divisions.season.eq(2026)&divisions.division.eq("fbs"),"team"])
    current=catalog[catalog.season.eq(2026)]
    def coverage(d):
        return {"teams":d.team.nunique(),"rostered_player_rows":len(d),"scouting_known":int(d.any_scouting_known.sum()),
                "scouting_missing_or_ambiguous":int((~d.any_scouting_known).sum()),"prior_box_observed":int(d.prior_box_observed.sum()),
                "observed_at_other_program":int(d.observed_at_other_program.sum()),
                "other_program_with_scouting":int((d.observed_at_other_program&d.any_scouting_known).sum()),
                "multiple_rosters_same_season":int(d.multiple_rosters_same_season.sum()),
                "identity_review_required":int(d.identity_review_required.sum()),
                "scouting_usable_without_identity_review":int(d.scouting_usable_without_identity_review.sum()),
                "history_usable_without_identity_review":int(d.history_usable_without_identity_review.sum()),
                "other_program_and_scouting_without_identity_review":int((d.observed_at_other_program&d.scouting_usable_without_identity_review).sum())}
    report={"production_uses_player_catalog":False,"catalog_rows":len(catalog),"individual_recruits":len(recruits),
            "accepted_links":len(accepted),"conflicting_links":len(conflicts),"2026_all":coverage(current),
            "2026_fbs":coverage(current[current.team.isin(fbs)]),
            "missingness":"No observed box rows does not mean the player never played; OL and other non-stat positions lack complete snap/blocking history",
            "timing":"Prior box seasons strictly earlier; season rosters and recruiting revisions lack archived publication times",
            "identity_limit":"Exact IDs can be reused. Four-year roster gaps, nine-year college-history spans and simultaneous multiple rosters are flagged for review, not automatically declared false",
            "input_sha256":{n:sha(CFBD/n) for n in ["roster.parquet","recruiting_players.parquet","player_game_stats.parquet","player_game_stats_fcs.parquet"]},
            "catalog_sha256":sha(OUTPUT/"player_seasons.parquet"),"generator_sha256":sha(__file__)}
    current.assign(scope=np.where(current.team.isin(fbs),"fbs","other")).groupby(["scope","position"],dropna=False).agg(
        players=("athlete_id","size"),scouting_known=("any_scouting_known","sum"),prior_boxes=("prior_box_observed","sum"),
        other_program=("observed_at_other_program","sum"),identity_review=("identity_review_required","sum"),
        usable_scouting=("scouting_usable_without_identity_review","sum")).reset_index().to_parquet(OUTPUT/"2026_position_coverage.parquet",index=False)
    save_json(OUTPUT/"audit.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":run()
