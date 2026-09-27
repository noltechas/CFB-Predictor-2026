"""Read back the active Supabase pregame rows and verify local parity."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

from cfbpred.config import CURRENT_SEASON, MODELS, PQ
from cfbpred.load_supabase import dsn
from cfbpred.pregame_experiments import save_json, sha
from cfbpred.publish_future_predictions import current_production_run


REGISTRY = MODELS / "player_sequence_production"


def verify(registry: Path = REGISTRY, all_season: bool = False) -> dict:
    active = json.loads((registry / "active.json").read_text())
    local_games = pd.read_parquet(PQ / "game_predictions.parquet").set_index("game_id")
    local_pmfs = pd.read_parquet(PQ / "game_pmfs.parquet").set_index("game_id")
    with psycopg.connect(dsn()) as connection, connection.cursor() as cursor:
        cursor.execute("set local extra_float_digits=3")
        run = current_production_run(cursor)
        date_filter = "" if all_season else "and g.start_date>now()"
        cursor.execute(f"""select p.game_id,p.model_version,p.home_win_prob,p.pred_margin,p.pred_total,
                                  p.pred_home_pts,p.pred_away_pts
                           from public.game_predictions p join public.games g using(game_id)
                           where p.run_id=%s and g.season=%s {date_filter}
                           order by p.game_id""", (run, CURRENT_SEASON))
        predictions = cursor.fetchall()
        cursor.execute(f"""select f.game_id,f.margin_pmf,f.total_pmf
                           from public.game_pmf f join public.games g using(game_id)
                           where g.season=%s {date_filter}
                           order by f.game_id""", (CURRENT_SEASON,))
        probabilities = cursor.fetchall()
    prediction_ids = [int(row[0]) for row in predictions]
    pmf_ids = [int(row[0]) for row in probabilities]
    if prediction_ids != pmf_ids or len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("Published predictions and PMFs differ in coverage")
    if not set(prediction_ids).issubset(local_games.index) or not set(pmf_ids).issubset(local_pmfs.index):
        raise ValueError("Published future game is absent locally")

    prediction_error = 0.0
    score_mismatches = 0
    published_versions = set()
    for game_id, model_version, wp, margin, total, home, away in predictions:
        published_versions.add(model_version)
        local = local_games.loc[int(game_id)]
        prediction_error = max(prediction_error, abs(float(wp) - float(local.home_win_prob)),
                               abs(float(margin) - float(local.pred_margin)),
                               abs(float(total) - float(local.pred_total)))
        score_mismatches += int(int(home) != int(local.pred_home_pts) or int(away) != int(local.pred_away_pts))
    pmf_error = 0.0
    for game_id, margin, total in probabilities:
        local = local_pmfs.loc[int(game_id)]
        pmf_error = max(pmf_error,
                        float(np.max(np.abs(np.asarray(margin, np.float64) - np.asarray(local.margin_pmf, np.float64)))),
                        float(np.max(np.abs(np.asarray(total, np.float64) - np.asarray(local.total_pmf, np.float64)))))
    if published_versions != {active["version"]}:
        raise ValueError("Published prediction model version differs from active production")
    if prediction_error > 1e-4 or pmf_error > 1e-6 or score_mismatches:
        raise ValueError("Published predictions differ from the audited local artifacts")

    report = {
        "passed": True, "version": active["version"], "scope": active["scope"],
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "coverage": "full_season" if all_season else "future",
        "published_season_games" if all_season else "published_future_games": len(prediction_ids),
        "prediction_and_pmf_game_ids_match": True,
        "published_model_versions": sorted(published_versions),
        "scoreline_mismatches": score_mismatches,
        "max_prediction_readback_error": prediction_error,
        "max_pmf_readback_error": pmf_error,
        "local_game_predictions_sha256": sha(PQ / "game_predictions.parquet"),
        "local_game_pmfs_sha256": sha(PQ / "game_pmfs.parquet"),
        "external_publish_complete": True,
    }
    save_json(registry / "publish_verification.json", report)
    promotion_path = registry / "latest_promotion.json"
    promotion = json.loads(promotion_path.read_text())
    promotion.update(external_publish_complete=True, publish_verification=report)
    pending = promotion_path.with_name(promotion_path.name + ".pending")
    save_json(pending, promotion)
    pending.replace(promotion_path)
    archived = Path(promotion["rollback_archive"]) / "promotion.json"
    save_json(archived, promotion)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-season", action="store_true")
    args = parser.parse_args()
    verify(all_season=args.all_season)
