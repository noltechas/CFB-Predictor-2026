"""Isolate recent-form gains using the unchanged margin-training objective."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.pregame_experiments import (ASSESSMENT, DEFAULT_OUTPUT, DEVELOPMENT,
    FORM_VARIANTS, frozen_inputs, paired, run_variant, save_json, choose, summarize)
from cfbpred.train import build_tensors


def run(output=DEFAULT_OUTPUT):
    source = frozen_inputs(output)
    save_json(output / "form_baseline_plan.json", {
        "reason": "The margin-loss change did not improve later-year margin MAE; also isolate recent-form effects under the unchanged objective.",
        "objective": "baseline", "features": FORM_VARIANTS,
        "development_years": DEVELOPMENT, "assessment_years": ASSESSMENT,
        "selection": "Same development-only criterion and probability guardrails as the original plan",
        "2026_labels_used": False})
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base = build_tensors(data_dir=source, include_plays=False)
    development = {form: run_variant(base, source, output, "baseline", form, DEVELOPMENT, device)
                   for form in FORM_VARIANTS}
    scores = {form: summarize(frame) for form, frame in development.items()}
    selected = choose(scores, "none")
    report = {"stage": "form_with_original_objective", "development": scores,
              "selected_on_development": selected, "control": "none",
              "choice_made_before_assessment": True,
              "selection_pairing": paired(development["none"], development[selected])}
    save_json(output / "form_baseline_report.json", report)
    assessment = {form: run_variant(base, source, output, "baseline", form, ASSESSMENT, device)
                  for form in dict.fromkeys(["none", selected])}
    report["assessment"] = {form: summarize(frame) for form, frame in assessment.items()}
    report["assessment_by_year"] = {form: {str(year): summarize(frame[frame.season==year]) for year in ASSESSMENT}
                                    for form, frame in assessment.items()}
    report["assessment_pairing"] = paired(assessment["none"], assessment[selected])
    save_json(output / "form_baseline_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT)
