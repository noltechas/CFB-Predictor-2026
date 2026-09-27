"""Refresh the complete player candidate from the latest local raw-feature build.

Run after the usual score/roster/box pulls and features_raw.py. New immutable
input directories preserve the backtest, while saved checkpoint scalers and
development-selected ensemble weights remain fixed. No production publishing.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cfbpred.audit_player_forecast import audit
from cfbpred.blend_player_forecasts import blend
from cfbpred.config import FEAT,MODELS
from cfbpred.player_experiment import OUTPUT
from cfbpred.player_forecast import score
from cfbpred.prepare_player_data import prepare
from cfbpred.pregame_experiments import save_json,sha
from cfbpred.recent_form import build as build_form


def refresh(feature_dir=FEAT,model_dir=OUTPUT,root=MODELS/"player_forecast_refreshes"):
    required=[model_dir/"current_2026/bundle.json",model_dir/"current_control_2026/bundle.json",
              model_dir/"blend_selection.json"]
    if any(not p.exists() for p in required):raise ValueError("Train both candidate bundles and freeze development blend selection first")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run=root/stamp;source=run/"source";source.mkdir(parents=True,exist_ok=False)
    for name in ["game_scalars.parquet","game_history.parquet","team_vocab.json"]:
        shutil.copy2(feature_dir/name,source/name)
    # A forecast refresh does not need betting lines. Keep an explicit empty
    # benchmark table for the shared input-directory contract.
    targets=pd.read_parquet(source/"game_scalars.parquet",columns=["game_id"])
    targets.assign(spread_close=np.nan).to_parquet(source/"market.parquet",index=False)
    build_form(source)
    save_json(source/"source_manifest.json",{p.name:sha(p) for p in source.iterdir() if p.is_file()})
    inputs,players=run/"inputs",run/"players"
    prepare(source,inputs,players)
    forecasts=run/"player_predictions.parquet"
    score(inputs,players,required[0],forecasts)
    audit(inputs,players,required[0],forecasts,run/"player_audit.json")
    blended=run/"blended_predictions.parquet"
    blend(inputs,players,required[2],forecasts,required[1],blended)
    audit(inputs,players,required[1],run/"blended_predictions_control.parquet",run/"control_audit.json")
    report={"run_directory":str(run.resolve()),"forecast_path":str(blended.resolve()),
            "forecast_sha256":sha(blended),"feature_source":str(feature_dir.resolve()),
            "feature_source_sha256":sha(source/"game_scalars.parquet"),"production_modified":False,
            "data_and_both_model_audits_passed":True}
    save_json(run/"refresh_report.json",report)
    pending=root/f"latest.{stamp}.pending.json";save_json(pending,report);pending.replace(root/"latest.json")
    print(json.dumps(report,indent=2));return run


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--features",type=Path,default=FEAT)
    ap.add_argument("--model-directory",type=Path,default=OUTPUT)
    ap.add_argument("--output-root",type=Path,default=MODELS/"player_forecast_refreshes")
    a=ap.parse_args();refresh(a.features,a.model_directory,a.output_root)
