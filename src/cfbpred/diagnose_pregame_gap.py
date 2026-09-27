"""Exploratory error slices of archived chronological pregame forecasts.

These retrospective slices describe where error occurs; they do not establish
its cause, prove a feature's value, or constitute an untouched evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cfbpred.architecture_experiment import REFERENCE
from cfbpred.config import MODELS


def run():
    frame = pd.concat([pd.read_parquet(REFERENCE / "runs" / "baseline__none" / f"{y}_predictions.parquet")
                       for y in [2023,2024,2025]],ignore_index=True)
    raw = pd.read_parquet(REFERENCE / "inputs" / "game_scalars.parquet")
    fields = ["game_id","neutral_site","h_gp","a_gp","h_qb_changed","a_qb_changed",
              "h_qb_known","a_qb_known","h_cx_qb_gms","a_cx_qb_gms"]
    frame = frame[frame.fbs_fbs & frame.spread_close.notna()].merge(raw[fields],on="game_id",validate="one_to_one")
    frame["model_ae"] = (frame.em-frame.margin).abs()
    frame["market_ae"] = (-frame.spread_close-frame.margin).abs()
    frame["excess"] = frame.model_ae-frame.market_ae
    frame["season_games"] = pd.cut(frame[["h_gp","a_gp"]].min(axis=1),[-1,0,3,7,100],
                                   labels=["No season games","1-3 season games","4-7 season games","8+ season games"])
    frame["weeks"] = pd.cut(frame.week_c,[0,3,7,13,30],labels=["1-3","4-7","8-13","14+"])
    frame["qb_changed"] = frame[["h_qb_changed","a_qb_changed"]].fillna(0).max(axis=1)>0
    frame["missing_qb"] = frame[["h_qb_known","a_qb_known"]].fillna(0).min(axis=1)==0
    segments = {}
    for col in ["season_games","weeks","neutral_site","qb_changed","missing_qb"]:
        stats = frame.groupby(col,observed=True).agg(n=("game_id","size"),model_mae=("model_ae","mean"),
                    market_mae=("market_ae","mean"),gap=("excess","mean")).reset_index()
        stats[col] = stats[col].astype(str)
        segments[col] = stats.to_dict(orient="records")
    report = {"n":len(frame),"years":[2023,2024,2025],"model":"Three-seed chronological control",
              "interpretation":"Exploratory, overlapping slices; gap is model MAE minus archived market MAE; not causal feature importance",
              "2026_outcomes_used":False,"segments":segments}
    output = MODELS / "experiments" / "architecture_20260906"
    (output / "error_diagnosis.json").write_text(json.dumps(report,indent=2))
    frame.to_parquet(output / "error_diagnosis_games.parquet",index=False)
    return report


if __name__=="__main__":
    print(json.dumps(run(),indent=2))
