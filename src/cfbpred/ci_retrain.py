"""Retrain the production pregame ensemble on GitHub Actions (cfb-retrain.yml) — no laptop.

    python -m cfbpred.ci_retrain run --version player-full-context-tabm16-v11 [--promote]
    python -m cfbpred.ci_retrain promote --version player-full-context-tabm16-v11   (publish a staged run)

The shipped recipe, end to end, on the state the nightly already restores:

1. stores    build a fresh store chain from the feature store exactly as the nightly
             does (refresh_player_sequence_production.refresh), into
             models/experiments/ci_retrain/stores/<stamp>. Training reads only seasons
             before CURRENT_SEASON (the checkpoint lineage check enforces it).
2. fit       one player_context_experiment.run per outer seed of the ACTIVE variant,
             fixed 14 epochs (the v8-v10 budget), in parallel processes.
3. stage     stage_ensemble -> registry/<version> and staged_full_context.json.
4. score     score + audit the whole season with the staged bundle; adapter parity
             against the refits' own saved distributions (as every release has).
5. guard     score the same season with the ACTIVE bundle on the same store chain and
             compare on completed FBS-involved games. Pre-registered rule (GUARD):
             candidate margin MAE (PMF median) <= active + 0.05 and win-probability
             log loss <= active + 0.002, on at least 200 games. Both models are trained
             on the same earlier seasons, so this checks the new data and code reproduce
             the live model's accuracy; lines are never used.
6. rankings  this season's weekly Fieldcast rankings from the staged bundle.
7. publish   only with --promote AND an accepted guard:
             scripts/publish_full_context_release.py (rollback archive, readback checks).

Without --promote nothing public changes: the staged release and its reports are saved
to CI state for review, and the live model keeps serving.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cfbpred.config import CURRENT_SEASON, ROOT

EXP = ROOT / "models/experiments/ci_retrain"
REGISTRY = ROOT / "models/player_sequence_production"
ACTIVE = REGISTRY / "active.json"
STAGED = REGISTRY / "staged_full_context.json"
GUARD = dict(mae_slack=0.05, ll_slack=0.002, min_games=200,
             cohort="completed FBS-involved games of the forecast season, both models scored on the same store chain")
FIT = dict(threads=1, micro_batch=64, fixed_epochs=14)


def log(msg: str) -> None:
    print(f"[retrain {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str))


def feature_dir() -> Path:
    """The feature files the last nightly built its snapshot from (refresh() copies
    game_scalars/game_history/team_vocab into <run>/source). The feature store itself is
    not CI state; the nightly rebuilds it, so a retrain reuses that night's copy."""
    latest = json.loads((REGISTRY / "refreshes/latest.json").read_text())
    for candidate in (Path(latest.get("run_directory", "")) / "source", ROOT / "data/parquet/features"):
        if all((candidate / f).is_file() for f in ("game_scalars.parquet", "game_history.parquet", "team_vocab.json")):
            return candidate
    raise SystemExit("no feature files: run the nightly first (its snapshot is what a retrain trains on)")


def build_stores() -> Path:
    from cfbpred.refresh_player_sequence_production import refresh
    features = feature_dir()
    log(f"building a store chain from {features}")
    run = refresh(feature_dir=features, output_root=EXP / "stores", active_path=ACTIVE, promote=False)
    log(f"store chain {run}")
    return run


def fit_seeds(source: Path, out: Path, variant: str, seeds: list[int], concurrency: int) -> list[Path]:
    """Each seed in its own process (the fits are single-threaded by design)."""
    jobs, running, failed = [], {}, []
    for s in seeds:
        dest = out / f"refit_{CURRENT_SEASON}_{variant}_s{s}"
        if (dest / "report.json").exists():
            continue
        code = ("from pathlib import Path; from cfbpred.player_context_experiment import run; "
                f"run(Path({str(source)!r}), Path({str(dest)!r}), {variant!r}, year={CURRENT_SEASON}, seed={s}, "
                f"threads={FIT['threads']}, micro_batch={FIT['micro_batch']}, fixed_epochs={FIT['fixed_epochs']})")
        jobs.append((s, dest, code))
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    t0 = time.time()
    while jobs or running:
        while jobs and len(running) < concurrency:
            s, dest, code = jobs.pop(0)
            dest.parent.mkdir(parents=True, exist_ok=True)
            fh = open(out / f"fit_s{s}.log", "w")
            running[s] = (subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, env=env,
                                           stdout=fh, stderr=subprocess.STDOUT), fh)
            log(f"fit seed {s} started")
        time.sleep(15)
        for s, (proc, fh) in list(running.items()):
            if proc.poll() is None:
                continue
            fh.close(); del running[s]
            if proc.returncode:
                failed.append(s)
                tail = (out / f"fit_s{s}.log").read_text()[-3000:]
                log(f"fit seed {s} FAILED (exit {proc.returncode}):\n{tail}")
            else:
                log(f"fit seed {s} done ({time.time() - t0:.0f}s since start)")
    if failed:
        raise SystemExit(f"fits failed for seeds {failed}")
    return [out / f"refit_{CURRENT_SEASON}_{variant}_s{s}" for s in seeds]


def readout(frame: pd.DataFrame) -> pd.DataFrame:
    import torch
    from cfbpred.ensemble import median_readout
    from cfbpred.model import MARGIN_CENTRES
    pm = torch.tensor(np.stack(frame["margin_pmf"]), dtype=torch.float32)
    return pd.DataFrame({"pred_margin": median_readout(pm, MARGIN_CENTRES),
                         "wp": frame["home_win_prob"].to_numpy(float)}, index=frame.index)


def guard(candidate: pd.DataFrame, active: pd.DataFrame) -> dict:
    s = pd.read_parquet(ROOT / "data/parquet/schedule.parquet").set_index("game_id")
    s = s[(s["season"] == CURRENT_SEASON) & s["home_points"].notna() & s["away_points"].notna()]
    s = s[(s["home_division"] == "fbs") | (s["away_division"] == "fbs")]
    c, a = readout(candidate.set_index("game_id")), readout(active.set_index("game_id"))
    ids = s.index.intersection(c.index).intersection(a.index)
    margin = (s.loc[ids, "home_points"] - s.loc[ids, "away_points"]).to_numpy(float)
    won = (margin > 0).astype(float)

    def grade(r):
        p = np.clip(r.loc[ids, "wp"].to_numpy(float), 1e-6, 1 - 1e-6)
        return dict(margin_mae=float(np.abs(margin - r.loc[ids, "pred_margin"].to_numpy(float)).mean()),
                    wp_ll=float(-(won * np.log(p) + (1 - won) * np.log(1 - p)).mean()),
                    straight_up=float(((r.loc[ids, "pred_margin"].to_numpy(float) > 0) == (margin > 0)).mean()))

    gc, ga = grade(c), grade(a)
    accepted = (len(ids) >= GUARD["min_games"] and gc["margin_mae"] <= ga["margin_mae"] + GUARD["mae_slack"]
                and gc["wp_ll"] <= ga["wp_ll"] + GUARD["ll_slack"])
    return dict(games=int(len(ids)), candidate=gc, active=ga, accepted=bool(accepted), rule=GUARD)


def adapter_parity(frame: pd.DataFrame, candidates: list[Path]) -> dict:
    got = frame.set_index("game_id")
    pms, pts, ids = [], [], None
    for c in candidates:
        z = np.load(c / "distributions.npz")
        pos = pd.Series(np.arange(len(z["game_id"])), index=z["game_id"])
        ids = pos.index if ids is None else ids.intersection(pos.index)
        pms.append((pos, z["pm"])); pts.append((pos, z["pt"]))
    ids = ids.intersection(got.index)
    pm = np.mean([m[p.loc[ids].to_numpy()] for p, m in pms], 0)
    pt = np.mean([m[p.loc[ids].to_numpy()] for p, m in pts], 0)
    err_m = float(np.abs(np.stack(got.loc[ids, "margin_pmf"]) - pm).max())
    err_t = float(np.abs(np.stack(got.loc[ids, "total_pmf"]) - pt).max())
    return dict(passed=bool(len(ids) > 0 and max(err_m, err_t) < 1e-6), compared_games=int(len(ids)),
                max_errors=dict(pm=err_m, pt=err_t), weights_unchanged=True)


def run(version: str, seeds: list[int], concurrency: int, promote: bool) -> int:
    from cfbpred.player_context_production import stage_ensemble
    from cfbpred.player_sequence_production import active_bundle, audit_snapshot, score_snapshot
    if (REGISTRY / version).exists():
        raise SystemExit(f"{version} already exists in the registry; pick a new version name")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reports = EXP / "reports" / stamp
    active, _, bundle = active_bundle(ACTIVE)
    variant = bundle["variant"]
    log(f"active {active['version']} ({variant}); retraining {len(seeds)} seeds as {version}, season {CURRENT_SEASON}")

    stores = build_stores()
    source = stores / "context" / "source.json"
    inputs, players, sequences = stores / "inputs", stores / "players", stores / "sequences"
    candidates = fit_seeds(source, EXP / "production" / stamp, variant, seeds, concurrency)

    assessment = dict(accepted=False, arm="ci_retrain", variant=variant, version=version,
                      previous_version=active["version"], recipe=dict(FIT, seeds=seeds),
                      store_chain=str(stores), rule=GUARD, point_readout="interpolated PMF median",
                      production_seeds=len(seeds))
    stage_ensemble(candidates, source, version=version, assessment=assessment)
    release = REGISTRY / version
    forecasts = release / f"season_{CURRENT_SEASON}_predictions.parquet"
    frame = score_snapshot(inputs, players, sequences, forecasts, STAGED, all_season=True)
    audit = audit_snapshot(inputs, players, sequences, forecasts, STAGED)
    save_json(release / "production_audit.json", audit)
    parity = adapter_parity(frame, candidates)
    save_json(release / "adapter_parity.json", parity)
    log(f"audit passed={audit.get('passed')}  adapter parity {parity}")

    current = score_snapshot(inputs, players, sequences, reports / "active_season.parquet", ACTIVE, all_season=True)
    g = guard(frame, current)
    save_json(reports / "guard.json", g)
    log(f"guard on {g['games']} games: candidate {g['candidate']}  active {g['active']}  "
        f"-> {'ACCEPTED' if g['accepted'] else 'REJECTED'}")
    assessment.update(accepted=bool(g["accepted"] and audit.get("passed") and parity["passed"]), guard=g,
                      candidate_model_mae=g["candidate"]["margin_mae"], baseline_model_mae=g["active"]["margin_mae"],
                      candidate_wp_ll=g["candidate"]["wp_ll"], baseline_wp_ll=g["active"]["wp_ll"])
    save_json(release / "assessment_manifest.json", assessment)

    from cfbpred.historical_rankings import reconstruct
    log("rebuilding this season's rankings with the staged bundle")
    reconstruct(source, STAGED, release / "rankings", first_season=CURRENT_SEASON, last_season=CURRENT_SEASON)

    summary = dict(version=version, stamp=stamp, accepted=assessment["accepted"], guard=g, audit_passed=audit.get("passed"),
                   adapter_parity=parity, promote_requested=promote)
    save_json(reports / "summary.json", summary)
    if not assessment["accepted"]:
        log("NOT promoted: the release gates did not pass; the live model is unchanged")
        return 1
    if not promote:
        log(f"staged {version} and saved it for review; not promoted (re-run with promote to publish)")
        return 0
    log(f"promoting {version}")
    publish(release)
    log(f"{version} is live")
    return 0


def publish(release: Path) -> None:
    """scripts/publish_full_context_release.py, which archives the current rankings folder
    for rollback first. That folder is rebuilt by the nightly and is not CI state, so a
    fresh runner may not have it yet."""
    (ROOT / "data/parquet/rankings").mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/publish_full_context_release.py"),
                    "--release-path", str(release), "--staged-path", str(STAGED)],
                   cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), check=True)


def promote_staged(version: str) -> int:
    """Publish a release an earlier stage-only run left staged (and saved to CI state)."""
    staged = json.loads(STAGED.read_text()) if STAGED.exists() else {}
    if staged.get("version") != version:
        raise SystemExit(f"staged release is {staged.get('version')!r}, not {version!r}")
    assessment = json.loads((REGISTRY / version / "assessment_manifest.json").read_text())
    if assessment.get("accepted") is not True:
        raise SystemExit(f"{version} did not pass its release gates; refusing to publish")
    log(f"promoting staged {version} (guard: {assessment.get('guard', {}).get('candidate')})")
    publish(REGISTRY / version)
    log(f"{version} is live")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "promote"])
    ap.add_argument("--version", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--promote", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "promote":
        return promote_staged(a.version)
    return run(a.version, a.seeds, a.concurrency, a.promote)


if __name__ == "__main__":
    raise SystemExit(main())
