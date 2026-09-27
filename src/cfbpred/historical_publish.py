"""Validation and explicitly guarded publication helpers for reconstruction archives.

The default operation is preparation only.  Publication requires the caller to
pass ``allow_publish=True``; every existing row is backed up before mutation
and the inserted rows are read back and compared exactly.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from cfbpred.pregame_experiments import sha

PRED_COLUMNS = ["game_id", "season", "split", "model_version", "home_win_prob",
                "pred_margin", "pred_total", "pred_home_pts", "pred_away_pts"]


def validate_archive(archive: Path, staged: Path | None = None) -> dict:
    archive = Path(archive)
    meta = json.loads((archive / "metadata.json").read_text())
    pred_path, pmf_path = archive / "game_predictions.parquet", archive / "game_pmfs.parquet"
    if meta.get("prediction_sha256") != sha(pred_path) or meta.get("pmf_sha256") != sha(pmf_path):
        raise ValueError("archive parquet hash does not match metadata")
    pred, pmf = pd.read_parquet(pred_path), pd.read_parquet(pmf_path)
    if list(pred.columns) != PRED_COLUMNS:
        raise ValueError("prediction columns are incompatible with game_predictions")
    if pred.game_id.duplicated().any() or pmf.game_id.duplicated().any():
        raise ValueError("archive contains duplicate game IDs")
    if not pred.game_id.equals(pmf.game_id) or (pred.game_id <= 0).any():
        raise ValueError("prediction and PMF game IDs differ or include virtual IDs")
    for col in ("margin_pmf", "total_pmf"):
        vals = np.asarray(pmf[col].tolist(), dtype=float)
        if not np.isfinite(vals).all() or (vals < 0).any() or not np.allclose(vals.sum(1), 1, atol=1e-5):
            raise ValueError(f"invalid {col} distributions")
    if int(meta.get("game_count", -1)) != len(pred):
        raise ValueError("metadata game count differs from archive")
    if staged is not None:
        pointer = json.loads(Path(staged).read_text())
        if pointer.get("version") != meta.get("model_version"):
            raise ValueError("archive model version differs from staged pointer")
    return {"metadata": meta, "predictions": pred, "pmfs": pmf}


def prepare_publish(archive: Path, staged: Path | None = None) -> dict:
    """Validate an archive and return a non-mutating publication plan."""
    checked = validate_archive(archive, staged)
    return {"ready": True, "archive": str(Path(archive).resolve()),
            "model_version": checked["metadata"]["model_version"],
            "game_count": len(checked["predictions"]), "mutated": False}


def historical_cohort(predictions: pd.DataFrame, pmfs: pd.DataFrame,
                      through_season: int = 2025) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the retrospective cohort, preserving current-season rows."""
    pred = predictions.loc[predictions["season"] <= through_season].copy()
    pmf = pmfs.loc[pmfs["season"] <= through_season].copy()
    if set(pred.game_id) != set(pmf.game_id):
        raise ValueError("historical prediction and PMF cohorts differ")
    return pred, pmf


def publish_archive(conn, archive: Path, backup_dir: Path, *, allow_publish: bool = False,
                    staged: Path | None = None, through_season: int = 2025) -> dict:
    """Publish all-year game rows with a backup and exact post-write readback.

    This is intentionally opt-in.  Without ``allow_publish`` it performs no
    database operation, which makes it safe to use as a release gate.
    """
    if not allow_publish:
        raise PermissionError("publication requires allow_publish=True")
    checked = validate_archive(archive, staged)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=False)
    pred, pmf = historical_cohort(checked["predictions"], checked["pmfs"], through_season)
    ids = [int(x) for x in pred.game_id]
    if not ids:
        raise ValueError("historical archive has no rows")
    with conn.cursor() as cur:
        cur.execute("set local extra_float_digits = 3")
        cur.execute("select game_id,season from public.games where game_id=any(%s)", (ids,))
        fixtures = {int(g): int(s) for g, s in cur.fetchall()}
        if set(fixtures) != set(ids) or any(fixtures[g] != int(s) for g, s in zip(pred.game_id, pred.season)):
            raise ValueError("archive contains game IDs absent from public.games or season mismatches")
        cur.execute("select run_id from public.model_runs where model_version=%s order by trained_at desc limit 1",
                    (checked["metadata"]["model_version"],))
        row = cur.fetchone()
        run_id = str(row[0]) if row else str(uuid4())
        old_pred = pd.read_sql_query("select run_id,game_id,season,split,model_version,home_win_prob,"
                                     "pred_margin,pred_total,pred_home_pts,pred_away_pts "
                                     "from public.game_predictions where game_id=any(%s)",
                                     conn, params=(ids,))
        old_pred["run_id"] = old_pred.run_id.astype(str)
        old_pmf = pd.read_sql_query("select game_id,season,margin_pmf,total_pmf from public.game_pmf "
                                   "where game_id=any(%s)", conn, params=(ids,))
    old_pred.to_parquet(backup_dir / "game_predictions.parquet", index=False)
    old_pmf.to_parquet(backup_dir / "game_pmf.parquet", index=False)
    try:
        with conn.cursor() as cur:
            cur.execute("select 1 from public.model_runs where run_id=%s", (run_id,))
            if cur.fetchone() is None:
                cur.execute("insert into public.model_runs "
                            "(run_id,model_version,train_seasons,test_seasons,notes) "
                            "values (%s,%s,int4range(2001,2025,'[]'),int4range(2026,2026,'[]'),%s)",
                            (run_id, checked["metadata"]["model_version"],
                             "retrospective all-year reconstruction; latest weights; no held-out evaluation"))
            rows = [(run_id, int(r.game_id), int(r.season), str(r.split), str(r.model_version),
                     float(r.home_win_prob), float(r.pred_margin), float(r.pred_total),
                     int(r.pred_home_pts), int(r.pred_away_pts)) for r in pred.itertuples()]
            # The public client queries by game ID, without a run filter.
            # Replace only this backed-up cohort across old runs atomically.
            cur.execute("delete from public.game_predictions where game_id=any(%s)", (ids,))
            cur.executemany("insert into public.game_predictions "
                            "(run_id,game_id,season,split,model_version,home_win_prob," 
                            "pred_margin,pred_total,pred_home_pts,pred_away_pts) "
                            "values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                            "on conflict (run_id,game_id) do update set season=excluded.season,split=excluded.split,"
                            "model_version=excluded.model_version,home_win_prob=excluded.home_win_prob,pred_margin=excluded.pred_margin,"
                            "pred_total=excluded.pred_total,pred_home_pts=excluded.pred_home_pts,pred_away_pts=excluded.pred_away_pts", rows)
            cur.executemany("insert into public.game_pmf "
                            "(game_id,season,margin_pmf,total_pmf) values (%s,%s,%s,%s) "
                            "on conflict (game_id) do update set season=excluded.season,margin_pmf=excluded.margin_pmf,total_pmf=excluded.total_pmf",
                            [(int(r.game_id), int(r.season), list(map(float, r.margin_pmf)),
                              list(map(float, r.total_pmf))) for r in pmf.itertuples()])
            # Validate every scalar and every PMF while still inside the
            # transaction.  Any mismatch raises and rolls the write back.
            read_pred = pd.read_sql_query("select game_id,season,split,model_version,home_win_prob,"
                                          "pred_margin,pred_total,pred_home_pts,pred_away_pts from public.game_predictions "
                                          "where game_id=any(%s) order by game_id", conn,
                                          params=(ids,))
            read_pmf = pd.read_sql_query("select game_id,season,margin_pmf,total_pmf from public.game_pmf "
                                         "where game_id=any(%s) order by game_id", conn, params=(ids,))
            exp_pred = pred.sort_values("game_id").reset_index(drop=True)
            got_pred = read_pred.sort_values("game_id").reset_index(drop=True)
            if len(got_pred) != len(exp_pred) or not np.array_equal(got_pred.game_id, exp_pred.game_id):
                raise RuntimeError("game_predictions exact readback failed")
            for col in ("season", "split", "model_version", "pred_home_pts", "pred_away_pts"):
                if not np.array_equal(got_pred[col].to_numpy(), exp_pred[col].to_numpy()):
                    raise RuntimeError(f"game_predictions readback mismatch: {col}")
            for col in ("home_win_prob", "pred_margin", "pred_total"):
                if not np.allclose(got_pred[col], exp_pred[col], rtol=2e-6, atol=2e-6):
                    raise RuntimeError(f"game_predictions readback mismatch: {col}")
            exp_pmf = pmf.sort_values("game_id").reset_index(drop=True)
            got_pmf = read_pmf.sort_values("game_id").reset_index(drop=True)
            if len(got_pmf) != len(exp_pmf) or not np.array_equal(got_pmf.game_id, exp_pmf.game_id):
                raise RuntimeError("game_pmf exact readback failed")
            for col in ("season",):
                if not np.array_equal(got_pmf[col].to_numpy(), exp_pmf[col].to_numpy()):
                    raise RuntimeError(f"game_pmf readback mismatch: {col}")
            for col in ("margin_pmf", "total_pmf"):
                for got, expected in zip(got_pmf[col], exp_pmf[col]):
                    if not np.allclose(got, expected, rtol=2e-6, atol=2e-6):
                        raise RuntimeError(f"game_pmf readback mismatch: {col}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"published": True, "run_id": run_id, "game_count": len(pred),
            "coverage": "historical_only", "database_ids_checked": len(ids),
            "backup": str(backup_dir.resolve())}
