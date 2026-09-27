"""Diagnostic 16-seed follow-up for a small, positive aggregate bundle gain.

The initial automatic two-of-three-season rule did not advance this candidate.
The user specifically wants small cumulative improvements investigated. Its
positive margin/total/probability aggregate scores motivate this additional
comparison; year instability is retained as a limitation, not hidden.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cfbpred.combined_experiment import OUTPUT, prepare, run_variant, compare
from cfbpred.pregame_experiments import ASSESSMENT, add_form, save_json, sha
from cfbpred.qb_roster_prior import attach
from cfbpred.train import build_tensors


def run():
    source,qb=prepare()
    plan={"seeds":list(range(16)),"years":ASSESSMENT,"variants":["baseline","combined"],
          "reason":"User requested retaining small cumulative gains; all four pooled margin/total/WP/Brier scores improved at three seeds despite one-of-three season margin wins",
          "selection_limit":"Diagnostic extension after historical assessment; no new untouched years; no 2026 labels used",
          "original_report_sha256":sha(OUTPUT/"report.json"),
          "script_sha256":sha(__file__),"parent_plan_sha256":sha(OUTPUT/"plan.json")}
    path=OUTPUT/"confirmation_plan.json"
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError("Confirmation plan changed")
    save_json(path,plan)
    (OUTPUT/"code_snapshot"/Path(__file__).name).write_text(Path(__file__).read_text())
    torch.set_num_threads(4)
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base=build_tensors(data_dir=source,include_plays=False)
    bundle=attach(add_form(base,source,"recent5_matchup"),pd.read_parquet(qb))
    frames={n:run_variant(base if n=="baseline" else bundle,source,n,ASSESSMENT,range(16),device)
            for n in plan["variants"]}
    result=compare(frames)
    result["production_modified"]=False
    save_json(OUTPUT/"confirmation_report.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":run()
