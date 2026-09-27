"""Publish the audited September 2026 full-context release, with rollback files."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import argparse
from pathlib import Path
import shutil

import pandas as pd
import psycopg
from psycopg.rows import dict_row

from cfbpred.config import PQ, ROOT
from cfbpred.load_supabase import dsn
from cfbpred.player_sequence_production import REGISTRY, active_bundle, promote_local
from cfbpred.pregame_experiments import save_json, sha
from cfbpred.publish_future_predictions import main as publish_predictions
from cfbpred.rankings import publish as publish_rankings
from cfbpred.verify_player_sequence_publish import verify


def _assessment_gate(release: Path, explicit: Path | None = None) -> dict:
    candidates = [explicit] if explicit else [release / "assessment_manifest.json", release / "accepted_assessment.json"]
    for path in candidates:
        if path and path.exists():
            data = json.loads(path.read_text())
            accepted = data.get("accepted") is True or data.get("assessment_accepted") is True
            if not accepted:
                raise ValueError(f"Assessment manifest is not accepted: {path}")
            return data
    raise ValueError("Missing accepted assessment manifest; refusing publication")


def _documents(directory: Path, season: int, active_version: str) -> list[dict]:
    docs = []
    for path in directory.glob(f"{season}-week-*.json"):
        doc = json.loads(path.read_text())
        if doc.get("model_version") != active_version or not doc.get("diagnostics", {}).get("strict_global_player_cutoff"):
            raise ValueError(f"Ranking release gate failed: {path}")
        docs.append(doc)
    docs.sort(key=lambda d: int(d["ranking_week"]))
    if not docs or len({d["ranking_week"] for d in docs}) != len(docs):
        raise ValueError(f"No unique ranking documents found in {directory}")
    return docs


def _runtime_source(release: Path, active: dict) -> Path:
    source_path = release / "source.json"
    if not source_path.exists():
        source_path = Path(active["bundle"] if "bundle" in active else active.get("input_source", ""))
    if source_path.name == "bundle.json":
        source_path = Path(json.loads(source_path.read_text())["input_source"])
    if not source_path.exists():
        source_path = release / "source.json"
    source = json.loads(source_path.read_text())
    runtime = release / "runtime"
    runtime.mkdir(exist_ok=True)
    for name in ("inputs", "players", "sequences"):
        target = Path(source[name]).resolve()
        link = runtime / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target, target_is_directory=True)
    context = runtime / "context"
    context.mkdir(exist_ok=True)
    context_source = dict(source, source_run=str(runtime.resolve()),
                          inputs=str((runtime / "inputs").resolve()),
                          players=str((runtime / "players").resolve()),
                          sequences=str((runtime / "sequences").resolve()))
    save_json(context / "source.json", context_source)
    save_json(runtime / "provenance.json", {"release": str(release.resolve()),
                                             "bundle": active.get("bundle"),
                                             "context_source": str((context / "source.json").resolve()),
                                             "context_source_sha256": sha(context / "source.json")})
    return runtime


def main(*, release_path: Path | None = None, documents_dir: Path | None = None,
         staged_path: Path = REGISTRY / "staged_full_context.json",
         assessment_manifest: Path | None = None, season: int = 2026):
    lock = Path("/tmp/cfbpred_publish.lock")
    lock.mkdir()  # Shared with both scheduled publishers. Never steal an active lock.
    try:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        staged = json.loads(staged_path.read_text())
        VERSION = staged["version"]
        release = (release_path or (REGISTRY / VERSION)).resolve()
        archive = REGISTRY / "release_backups" / stamp
        archive.mkdir(parents=True, exist_ok=False)
        active, _, _ = active_bundle(staged_path)
        if active["version"] != VERSION:
            raise ValueError("Staged active version changed while loading release")
        assessment = _assessment_gate(release, assessment_manifest)
        audit = json.loads((release / "production_audit.json").read_text())
        parity = json.loads((release / "adapter_parity.json").read_text())
        if season != 2026:
            raise ValueError("This publisher handles the current 2026 release; historical predictions use a separate publisher")
        forecasts = release / f"season_{season}_predictions.parquet"
        if not audit["passed"] or not parity["passed"] or sha(forecasts) != audit["forecast_sha256"]:
            raise ValueError("Release audit/parity gates failed")
        documents = _documents(documents_dir or (release / "rankings"), season, active["version"])
        shutil.copy2(REGISTRY / "active.json", archive / "active.json")
        shutil.copy2(REGISTRY / "refreshes/latest.json", archive / "refresh_latest.json")
        shutil.copytree(PQ / "rankings", archive / "rankings")
        with psycopg.connect(dsn(), row_factory=dict_row) as conn:
            previous = conn.execute("select * from public.fieldcast_rankings where season=2026 order by ranking_week").fetchall()
        (archive / "database_rankings.json").write_text(json.dumps(previous, default=str, indent=2))
        save_json(archive / "release_state.json", dict(stage="backed_up", version=VERSION))

        pending = REGISTRY / "active.json.pending"
        shutil.copy2(staged_path, pending)
        pending.replace(REGISTRY / "active.json")
        promotion = promote_local(forecasts, include_started=True)
        save_json(archive / "release_state.json", dict(stage="locally_promoted", promotion=promotion))
        publish_predictions(all_season=True)
        publication = verify(all_season=True)
        promotion = json.loads((REGISTRY / "latest_promotion.json").read_text())
        for doc in documents:
            publish_rankings(doc, replace_preseason=doc["ranking_week"] == 0,
                             replace_reviewed=True)
        with psycopg.connect(dsn(), row_factory=dict_row) as conn:
            actual = conn.execute("select * from public.fieldcast_rankings where season=2026 order by ranking_week").fetchall()
        for expected, got in zip(documents, actual):
            for key in ("ranking_week", "model_version", "model_fingerprint", "reference_count", "entries", "diagnostics"):
                if expected[key] != got[key]:
                    raise ValueError(f"Ranking readback differs: {key}")
        if len(actual) != len(documents):
            raise ValueError("Published ranking coverage differs")
        for doc in documents:
            save_json(PQ / f"rankings/{season}-week-{doc['ranking_week']}.json", doc)
        save_json(PQ / "rankings/latest.json", documents[-1])
        save_json(PQ / f"rankings/{season}-preseason.json", documents[0])
        runtime = _runtime_source(release, active)
        runtime_paths = json.loads((runtime / "context/source.json").read_text())
        refresh = json.loads((REGISTRY / "refreshes/latest.json").read_text())
        refresh.update(forecast_path=str(forecasts.resolve()), forecast_sha256=sha(forecasts),
                       active_model_version=VERSION, production_audit=audit, local_promotion=promotion,
                       run_directory=str(runtime.resolve()),
                       feature_source=runtime_paths["inputs"],
                       feature_source_sha256=sha(Path(runtime_paths["inputs"]) / "game_scalars.parquet"),
                       context_source=str((runtime / "context/source.json").resolve()),
                       context_source_sha256=sha(runtime / "context/source.json"))
        save_json(REGISTRY / "refreshes/latest.json", refresh)
        code = ["player_context_production.py", "player_sequence_production.py", "refresh_player_sequence_production.py",
                "rankings.py", "ranking_snapshot.py", "ranking_player_cutoff.py", "publish_future_predictions.py"]
        report = dict(passed=True, version=VERSION, published_at=datetime.now(timezone.utc).isoformat(),
                      prediction_verification=publication, ranking_weeks=[d["ranking_week"] for d in documents],
                      ranking_teams_per_week=[len(d["entries"]) for d in documents],
                      ranking_readback_exact=True, rollback_archive=str(archive.resolve()),
                      adapter_parity=parity, accepted_assessment=assessment,
                      training_latest_season=2025,
                      historical_predictions_reconstructed=True,
                      release_code_sha256={n: sha(ROOT / "src/cfbpred" / n) for n in code})
        save_json(release / "publication.json", report)
        save_json(archive / "release_state.json", dict(stage="complete", **report))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        lock.rmdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-path", type=Path)
    parser.add_argument("--documents-dir", type=Path)
    parser.add_argument("--staged-path", type=Path, default=REGISTRY / "staged_full_context.json")
    parser.add_argument("--assessment-manifest", type=Path)
    parser.add_argument("--season", type=int, default=2026)
    args = parser.parse_args()
    main(release_path=args.release_path, documents_dir=args.documents_dir,
         staged_path=args.staged_path, assessment_manifest=args.assessment_manifest,
         season=args.season)
