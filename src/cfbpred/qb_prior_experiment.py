"""Test public roster passing priors after identifying preseason coverage gaps."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import duckdb
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cfbpred.architecture_experiment import REFERENCE
from cfbpred.config import CFBD, MODELS
from cfbpred.pregame_experiments import ASSESSMENT, DEVELOPMENT, choose, paired, save_json, sha, summarize
from cfbpred.pull_espn_roster import team_alias_map
from cfbpred.qb_roster_prior import annual_passing, roster_priors, attach, FIELDS
from cfbpred.shared_team_experiment import run_variant
from cfbpred.train import build_tensors

OUTPUT = MODELS / "experiments" / "qb_prior_20260906"
CODE = ("qb_prior_experiment.py","qb_roster_prior.py","shared_team_experiment.py","shared_team_net.py",
        "architecture_experiment.py","consistent_net.py","train.py","raw_net.py","model.py",
        "backtest_pregame.py","pregame_experiments.py","pull_espn_roster.py")


def build(source, output):
    directory = output / "inputs"
    directory.mkdir(parents=True,exist_ok=True)
    targets = pd.read_parquet(source / "game_scalars.parquet")
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raw = CFBD / "player_game_stats.parquet"
        with duckdb.connect() as con:
            stats = con.execute(f"SELECT season,game_id,athlete_id,category,stat_type,stat FROM read_parquet('{raw}') WHERE category='passing' AND season<=2025").fetchdf()
        annual = annual_passing(stats)
        annual.to_parquet(directory / "annual_passing.parquet",index=False)
        shutil.copy2(CFBD / "roster.parquet",directory / "roster.parquet")
        save_json(directory / "aliases.json",team_alias_map())
        save_json(manifest_path,{
            "raw_stat_sha256":sha(raw),
            "inputs":{p.name:sha(p) for p in directory.iterdir() if p.is_file()},
            "max_source_season":int(annual.season.max()),
            "roster_limit":"Historical season rosters are retrospective provider snapshots, not guaranteed pre-kickoff archives; same limitation as existing roster inputs"})
    manifest = json.loads(manifest_path.read_text())
    for name,digest in manifest["inputs"].items():
        if sha(directory / name)!=digest:
            raise ValueError("Frozen QB input changed")
    annual = pd.read_parquet(directory / "annual_passing.parquet")
    roster = pd.read_parquet(directory / "roster.parquet",columns=["id","team","season_pulled"]).rename(columns={"id":"athlete_id","season_pulled":"season"})
    aliases = json.loads((directory / "aliases.json").read_text())
    roster["team"] = roster.team.replace(aliases)
    teams = pd.concat([targets[["season",f"{side}_team"]].rename(columns={f"{side}_team":"team"}) for side in ["home","away"]]).drop_duplicates()
    teams["team_c"] = teams.team.replace(aliases)
    priors = roster_priors(teams[["season","team_c"]].rename(columns={"team_c":"team"}),roster,annual)
    priors = teams.merge(priors.rename(columns={"team":"team_c"}),on=["season","team_c"],validate="many_to_one").drop(columns="team_c")
    observed = priors.dropna(subset=["source_season"])
    assert (observed.source_season<observed.season).all()
    assert (observed.source_season>=observed.season-4).all()
    priors.to_parquet(directory / "preseason_passing_priors.parquet",index=False)
    rows = targets[["game_id","season","home_team","away_team"]].copy()
    for side,prefix in [("home","h_"),("away","a_")]:
        cols = {"team":f"{side}_team", **{f"pre_qb_{c}":prefix+f"pre_qb_{c}" for c in FIELDS}}
        rows = rows.merge(priors[["season","team"]+[f"pre_qb_{c}" for c in FIELDS]].rename(columns=cols),
                          on=["season",f"{side}_team"],how="left",validate="many_to_one")
    assert len(rows)==len(targets) and rows.game_id.is_unique
    rows = rows.drop(columns=["season","home_team","away_team"])
    rows.to_parquet(directory / "game_qb_priors.parquet",index=False)
    current = rows[targets.season.eq(2026).to_numpy()]
    coverage = {"targets":len(rows),"features":2*len(FIELDS),"2026_targets":len(current),
                "2026_home_prior_known":int(current.h_pre_qb_passer_known.sum()),
                "2026_away_prior_known":int(current.a_pre_qb_passer_known.sum()),
                "no_2026_results":int(annual.season.max())<=2025,
                "game_features_sha256":sha(directory / "game_qb_priors.parquet"),
                "prior_provenance_sha256":sha(directory / "preseason_passing_priors.parquet")}
    save_json(directory / "coverage.json",coverage)
    print(json.dumps(coverage,indent=2),flush=True)
    return rows,manifest,coverage


def compare(frames):
    scores = {n:summarize(f) for n,f in frames.items()}
    pair = paired(frames["baseline"],frames["qb_prior"])
    c,n = scores["baseline"],scores["qb_prior"]
    strong = (n["margin_mae"]<=c["margin_mae"]-.10 and n["wp_logloss"]<=c["wp_logloss"]+.005
              and n["brier"]<=c["brier"]+.002 and n["total_mae"]<=c["total_mae"]+.10)
    return {"scores":scores,"pairing":pair,"passes_point_estimate_gate":bool(strong),
            "passes_promotion_gate":bool(strong and pair["week_cluster_bootstrap_95ci"][1]<0),
            "by_year":{n:{str(y):summarize(f[f.season==y]) for y in sorted(f.season.unique())} for n,f in frames.items()}}


def run(output=OUTPUT):
    output.mkdir(parents=True,exist_ok=True)
    source = REFERENCE / "inputs"
    old = json.loads((REFERENCE / "plan.json").read_text())
    for name,digest in old["code_sha256"].items():
        if sha(Path(__file__).parent/name)!=digest:
            raise ValueError(f"Control code changed: {name}")
    for name,digest in old["input_sha256"].items():
        if sha(source/name)!=digest:
            raise ValueError(f"Frozen input changed: {name}")
    features,manifest,coverage = build(source,output)
    original = REFERENCE / "runs" / "baseline__none"
    plan = {
        "candidate":"18 preseason roster passing features, original DNN and loss; most experienced rostered passer from preceding four seasons, not actual eventual starter",
        "motivation":"Exploratory 2023-2025 residual audit identified no-current-season QB coverage; these later seasons are reused benchmarks, not untouched tests",
        "development":DEVELOPMENT,"assessment":ASSESSMENT,"seeds":[0,1,2],
        "selection":"Development margin MAE gain >=0.03, WP LL <=control+0.005, Brier <=control+0.002",
        "confirmation":"Selected three-seed assessment gain >=0.10, WP LL <=control+0.005, Brier <=control+0.002, total MAE <=control+0.10 extends both models to 16 seeds",
        "promotion":"16-seed gain >=0.10, paired week bootstrap 95CI wholly below zero, same probability/total guardrails; then operational validation",
        "2026_labels_used":False,"input_sha256":old["input_sha256"],"qb_inputs":manifest,"qb_features":coverage,
        "control_sha256":{p.name:sha(p) for p in sorted(original.glob("*.npz"))},
        "code_sha256":{n:sha(Path(__file__).parent/n) for n in CODE}}
    path = output / "plan.json"
    if path.exists() and json.loads(path.read_text())!=plan:
        raise ValueError("QB experiment changed; use a new output directory")
    save_json(path,plan)
    snapshot = output / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for n in CODE:
        shutil.copy2(Path(__file__).parent/n,snapshot/n)
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base = build_tensors(data_dir=source,include_plays=False)
    enriched = attach(base,features)
    def forecasts(years,seeds):
        return {n:run_variant(enriched if n=="qb_prior" else base,source,output,n,years,seeds,device)
                for n in ["baseline","qb_prior"]}
    dev = forecasts(DEVELOPMENT,range(3))
    selected = choose({n:summarize(f) for n,f in dev.items()},"baseline")
    choice = {"development":compare(dev),"selected":selected}
    path = output / "choice.json"
    if path.exists() and json.loads(path.read_text())!=choice:
        raise ValueError("Frozen development choice changed")
    save_json(path,choice)
    print("Development choice: "+json.dumps(choice,indent=2),flush=True)
    report = {**choice,"production_modified":False}
    save_json(output / "report.json",report)
    if selected=="qb_prior":
        report["assessment_3seed"] = compare(forecasts(ASSESSMENT,range(3)))
        save_json(output / "report.json",report)
        if report["assessment_3seed"]["passes_point_estimate_gate"]:
            print("Extending both configurations to 16 seeds",flush=True)
            report["confirmation_16seed"] = compare(forecasts(ASSESSMENT,range(16)))
            save_json(output / "report.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    run()
