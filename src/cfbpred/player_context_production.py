"""Stage the retained full-career checkpoint without changing fitted weights."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import torch

from cfbpred.config import CURRENT_SEASON, ROOT
from cfbpred.player_sequence_production import FULL_CONTEXT_VARIANTS, REGISTRY, _atomic_json
from cfbpred.pregame_experiments import save_json, sha

VERSION = "player-full-context-tabm16-v3"


def stage(candidate: Path, source: Path, registry: Path = REGISTRY, *,
          version: str = VERSION, assessment_mae: float = 12.328706383072717) -> Path:
    """Verify immutable experiment lineage; add inference schema metadata only."""
    plan = json.loads((candidate / "plan.json").read_text())
    original = json.loads((candidate / "bundle.json").read_text())
    paths = {k: Path(v) for k, v in json.loads(source.read_text()).items()}
    if (candidate / "INVALIDATED.json").exists():
        raise ValueError("Cannot install an invalidated experiment")
    if original["architecture"] != "player_full_context_v1" or original["variant"] not in FULL_CONTEXT_VARIANTS:
        raise ValueError("Expected the retained full-context architecture")
    if sha(source) != plan["source_sha256"] or sha(candidate / "plan.json") != original["plan_sha256"]:
        raise ValueError("Training input lineage changed")
    if plan["labels_2026_used"] or plan["refit_through_year"] >= CURRENT_SEASON:
        raise ValueError("Current outcomes entered fitting")
    for name, digest in plan["inputs"].items():
        if sha(paths["inputs"] / name) != digest:
            raise ValueError(f"Pinned input changed: {name}")
    for name, digest in plan["stores"].items():
        if sha(paths[name] / "manifest.json") != digest:
            raise ValueError(f"Pinned player store changed: {name}")
    for name, digest in plan["code"].items():
        if sha(ROOT / "src/cfbpred" / name) != digest:
            raise ValueError(f"Model implementation changed: {name}")
    if sha(candidate / original["member"]) != original["member_sha256"]:
        raise ValueError("Fitted checkpoint changed")
    checkpoint = torch.load(candidate / original["member"], map_location="cpu", weights_only=False)
    if checkpoint["train_max_season"] >= CURRENT_SEASON or checkpoint["plan_sha256"] != sha(candidate / "plan.json"):
        raise ValueError("Invalid checkpoint fit lineage")
    checkpoint["sequence_schema"] = json.loads((paths["sequences"] / "manifest.json").read_text())
    checkpoint["context_cols"] = json.loads((paths["timeline"] / "manifest.json").read_text())["context_cols"]
    checkpoint["full_history_schema"] = json.loads((paths["full_history"] / "manifest.json").read_text())["event_cols"]
    if not version.startswith("player-full-context-") or "/" in version:
        raise ValueError("Invalid release version")
    destination = registry / version
    destination.mkdir(parents=True, exist_ok=True)
    member = destination / "member.pt"
    if member.exists():
        existing = torch.load(member, map_location="cpu", weights_only=False)
        if any(not torch.equal(v, existing["state_dict"][k]) for k, v in checkpoint["state_dict"].items()):
            raise ValueError("Installed weights differ")
    else:
        torch.save(checkpoint, member)
    for name in ("plan.json", "report.json"):
        shutil.copy2(candidate / name, destination / name)
    shutil.copy2(source, destination / "source.json")
    bundle = dict(architecture=original["architecture"], variant=original["variant"],
                  model_version=version, forecast_year=CURRENT_SEASON, pregame_only=True,
                  members=["member.pt"], member_sha256={"member.pt": sha(member)},
                  input_source=str((destination / "source.json").resolve()),
                  original_candidate=str(candidate.resolve()), original_checkpoint_sha256=original["member_sha256"],
                  plan_sha256=sha(destination / "plan.json"), weights_unchanged=True)
    save_json(destination / "bundle.json", bundle)
    active = dict(version=version, scope="production_pregame", architecture=original["architecture"],
                  variant=original["variant"], forecast_year=CURRENT_SEASON, members=1,
                  internal_tabm_members=16, bundle=str((destination / "bundle.json").resolve()),
                  bundle_sha256=sha(destination / "bundle.json"), source_candidate=str(candidate.resolve()),
                  live_model="raw-dnn-delta-v22", historical_market_matched_margin_mae=assessment_mae)
    staged = registry / "staged_full_context.json"
    _atomic_json(staged, active)
    return staged


def _verified_checkpoint(candidate: Path, source: Path) -> tuple[dict, dict, dict]:
    """Every lineage check stage() makes, for one candidate run."""
    plan = json.loads((candidate / "plan.json").read_text())
    original = json.loads((candidate / "bundle.json").read_text())
    paths = {k: Path(v) for k, v in json.loads(source.read_text()).items()}
    if (candidate / "INVALIDATED.json").exists():
        raise ValueError(f"Cannot install an invalidated experiment: {candidate}")
    if original["architecture"] != "player_full_context_v1" or original["variant"] not in FULL_CONTEXT_VARIANTS:
        raise ValueError("Expected the retained full-context architecture")
    if sha(source) != plan["source_sha256"] or sha(candidate / "plan.json") != original["plan_sha256"]:
        raise ValueError(f"Training input lineage changed: {candidate}")
    if plan["labels_2026_used"] or plan["refit_through_year"] >= CURRENT_SEASON:
        raise ValueError("Current outcomes entered fitting")
    for name, digest in plan["inputs"].items():
        if sha(paths["inputs"] / name) != digest:
            raise ValueError(f"Pinned input changed: {name}")
    for name, digest in plan["stores"].items():
        if sha(paths[name] / "manifest.json") != digest:
            raise ValueError(f"Pinned player store changed: {name}")
    for name, digest in plan["code"].items():
        if sha(ROOT / "src/cfbpred" / name) != digest:
            raise ValueError(f"Model implementation changed: {name}")
    if sha(candidate / original["member"]) != original["member_sha256"]:
        raise ValueError(f"Fitted checkpoint changed: {candidate}")
    checkpoint = torch.load(candidate / original["member"], map_location="cpu", weights_only=False)
    if checkpoint["train_max_season"] >= CURRENT_SEASON or checkpoint["plan_sha256"] != sha(candidate / "plan.json"):
        raise ValueError("Invalid checkpoint fit lineage")
    checkpoint["sequence_schema"] = json.loads((paths["sequences"] / "manifest.json").read_text())
    checkpoint["context_cols"] = json.loads((paths["timeline"] / "manifest.json").read_text())["context_cols"]
    checkpoint["full_history_schema"] = json.loads((paths["full_history"] / "manifest.json").read_text())["event_cols"]
    return plan, original, checkpoint


def stage_ensemble(candidates: list[Path], source: Path, registry: Path = REGISTRY, *,
                   version: str, assessment: dict) -> Path:
    """Stage several outer seeds of one variant as a single production bundle.

    Scoring averages the members' probability mass (score_snapshot), which cannot be
    worse than the average single seed on log loss or margin error; it also stops each
    retrain from reshuffling published numbers by more than the change being tested."""
    if len(candidates) < 2:
        raise ValueError("An ensemble needs at least two seeds; use stage() for one")
    if not version.startswith("player-full-context-") or "/" in version:
        raise ValueError("Invalid release version")
    verified = [(c, *_verified_checkpoint(c, source)) for c in candidates]
    variants = {o["variant"] for _, _, o, _ in verified}
    seeds = [int(ck["seed"]) for *_, ck in verified]
    if len(variants) != 1 or len(set(seeds)) != len(seeds):
        raise ValueError(f"Ensemble members must share a variant and differ in seed: {variants}, {seeds}")
    if len({json.dumps(p["code"], sort_keys=True) for _, p, _, _ in verified}) != 1:
        raise ValueError("Ensemble members were trained with different code")
    destination = registry / version
    destination.mkdir(parents=True, exist_ok=True)
    names, digests = [], {}
    for candidate, plan, original, checkpoint in verified:
        name = f"member_s{int(checkpoint['seed'])}.pt"; member = destination / name
        if member.exists():
            existing = torch.load(member, map_location="cpu", weights_only=False)
            if any(not torch.equal(v, existing["state_dict"][k]) for k, v in checkpoint["state_dict"].items()):
                raise ValueError(f"Installed weights differ: {name}")
        else:
            torch.save(checkpoint, member)
        shutil.copy2(candidate / "plan.json", destination / f"plan_s{int(checkpoint['seed'])}.json")
        names.append(name); digests[name] = sha(member)
    first = verified[0][0]
    shutil.copy2(first / "plan.json", destination / "plan.json")   # all members share its source hash
    shutil.copy2(source, destination / "source.json")
    variant = variants.pop()
    extras = {tuple(ck.get("scalar_extras", ())) for *_, ck in verified}
    if len(extras) != 1:
        raise ValueError("Ensemble members were trained on different scalar schemas")
    bundle = dict(architecture="player_full_context_v1", variant=variant, model_version=version,
                  scalar_extras=list(extras.pop()),
                  forecast_year=CURRENT_SEASON, pregame_only=True, members=names, member_sha256=digests,
                  input_source=str((destination / "source.json").resolve()),
                  original_candidates=[str(c.resolve()) for c in candidates], outer_seeds=sorted(seeds),
                  plan_sha256=sha(destination / "plan.json"), weights_unchanged=True)
    save_json(destination / "bundle.json", bundle)
    save_json(destination / "assessment.json", assessment)
    active = dict(version=version, scope="production_pregame", architecture="player_full_context_v1",
                  variant=variant, forecast_year=CURRENT_SEASON, members=len(names), internal_tabm_members=16,
                  outer_seeds=sorted(seeds), bundle=str((destination / "bundle.json").resolve()),
                  bundle_sha256=sha(destination / "bundle.json"),
                  source_candidate=str(first.resolve()), source_candidates=[str(c.resolve()) for c in candidates],
                  live_model="raw-dnn-delta-v22 state tower, prior = this model's published pregame PMF",
                  historical_market_matched_margin_mae=assessment.get("candidate_model_mae"))
    staged = registry / "staged_full_context.json"
    _atomic_json(staged, active)
    return staged
