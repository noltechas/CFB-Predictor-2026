"""Publish current-season pregame predictions and PMFs.

The default preserves started games.  ``--all-season`` is an explicit audited
backfill path that archives the current database rows before replacing them.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

import pandas as pd
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.config import CURRENT_SEASON, MODELS, PQ
from cfbpred.load_supabase import dsn
from cfbpred.pregame_experiments import save_json, sha

REGISTRY = MODELS / "player_sequence_production"


def active_model_version(registry: Path = REGISTRY) -> str:
    active = json.loads((registry / "active.json").read_text())
    version = active.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("Active production model has no version")
    return version


def current_production_run(cursor):
    # Historical reconstruction can use a different run. Only the current
    # season must have a single destination for scheduled forecast updates.
    cursor.execute("select distinct run_id from public.game_predictions where season=%s", (CURRENT_SEASON,))
    runs = cursor.fetchall()
    if len(runs) != 1:
        raise ValueError("Expected one current-season production run")
    return runs[0][0]


def main(all_season: bool = False):
    gp = pd.read_parquet(PQ / "game_predictions.parquet")
    pf = pd.read_parquet(PQ / "game_pmfs.parquet")
    model_version = active_model_version()
    promotion = json.loads((REGISTRY / "latest_promotion.json").read_text())
    if promotion["version"] != model_version or promotion["game_predictions_sha256"] != sha(PQ / "game_predictions.parquet"):
        raise ValueError("Local forecasts do not match the active audited promotion")
    replacements = {int(k): int(v) for k, v in promotion.get("retired_game_replacements", {}).items()} if all_season else {}
    with psycopg.connect(dsn()) as conn, conn.cursor() as cur:
        cur.execute("set local statement_timeout = '5min'")
        run = current_production_run(cur)
        # Prediction provenance belongs on each row because a current-season
        # promotion can replace forecasts without replacing older seasons.
        cur.execute("alter table public.game_predictions add column if not exists model_version text")
        cur.execute("""update public.game_predictions p set model_version=m.model_version
                       from public.model_runs m
                       where p.run_id=m.run_id and p.model_version is null""")
        cur.execute("alter table public.game_predictions alter column model_version set not null")
        if all_season:
            gp = gp[gp.season.eq(CURRENT_SEASON)]
            pf = pf[pf.season.eq(CURRENT_SEASON)]
            cur.execute("select game_id from public.games where season=%s", (CURRENT_SEASON,))
            season_games = {r[0] for r in cur.fetchall()}
            if not set(gp.game_id).issubset(season_games):
                raise ValueError("Local full-season prediction is absent from public.games")
        else:
            cur.execute("select game_id from public.games where season=%s and start_date>now()", (CURRENT_SEASON,))
            future = {r[0] for r in cur.fetchall()}
            gp = gp[gp.game_id.isin(future)]
            pf = pf[pf.game_id.isin(future)]
        if set(gp.game_id) != set(pf.game_id):
            raise ValueError("Predictions and PMFs must cover identical games")

        archive = None
        if all_season:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            archive = MODELS / "player_sequence_production/database_backups" / stamp
            archive.mkdir(parents=True, exist_ok=False)
            ids = list(map(int, gp.game_id)) + list(replacements)
            cur.execute("""select game_id,season,split,model_version,home_win_prob,pred_margin,pred_total,
                                  pred_home_pts,pred_away_pts
                           from public.game_predictions
                           where run_id=%s and game_id=any(%s) order by game_id""", (run, ids))
            pd.DataFrame(cur.fetchall(), columns=["game_id", "season", "split", "model_version", "home_win_prob",
                         "pred_margin", "pred_total", "pred_home_pts", "pred_away_pts"]).to_parquet(
                             archive / "game_predictions.parquet", index=False, compression="zstd")
            cur.execute("""select game_id,season,margin_pmf,total_pmf from public.game_pmf
                           where game_id=any(%s) order by game_id""", (ids,))
            pd.DataFrame(cur.fetchall(), columns=["game_id", "season", "margin_pmf", "total_pmf"]).to_parquet(
                archive / "game_pmfs.parquet", index=False, compression="zstd")

        for old_id, new_id in replacements.items():
            if old_id in set(gp.game_id) or new_id not in set(gp.game_id):
                raise ValueError("Invalid retired fixture replacement")
            cur.execute("select game_id,season,home_team,away_team,start_date from public.games where game_id=any(%s)", ([old_id, new_id],))
            fixtures = {r[0]: r[1:] for r in cur.fetchall()}
            old, new = fixtures[old_id], fixtures[new_id]
            if old[:3] != new[:3] or abs((old[3] - new[3]).total_seconds()) > 7 * 86400:
                raise ValueError("Retired and replacement fixtures are not the same matchup")
            cur.execute("delete from public.game_predictions where run_id=%s and game_id=%s", (run, old_id))
            cur.execute("delete from public.game_pmf where game_id=%s", (old_id,))

        prediction_sql = """insert into public.game_predictions
            (run_id,game_id,season,split,model_version,home_win_prob,pred_margin,pred_total,pred_home_pts,pred_away_pts)
            select %s,%s,%s,%s,%s,%s,%s,%s,%s,%s"""
        prediction_sql += (" where exists (select 1 from public.games where game_id=%s and season=%s)"
                           if all_season else
                           " where exists (select 1 from public.games where game_id=%s and start_date>now())")
        prediction_sql += """
            on conflict (run_id,game_id) do update set home_win_prob=excluded.home_win_prob,
            pred_margin=excluded.pred_margin,pred_total=excluded.pred_total,
            pred_home_pts=excluded.pred_home_pts,pred_away_pts=excluded.pred_away_pts,
            split=excluded.split,model_version=excluded.model_version"""
        prediction_rows = [
            (run, int(r.game_id), int(r.season), r.split, model_version,
             float(r.home_win_prob), float(r.pred_margin),
             float(r.pred_total), int(r.pred_home_pts), int(r.pred_away_pts), int(r.game_id))
            + ((CURRENT_SEASON,) if all_season else ()) for r in gp.itertuples()]
        cur.executemany(prediction_sql, prediction_rows)

        pmf_sql = """insert into public.game_pmf (game_id,season,margin_pmf,total_pmf)
            select %s,%s,%s,%s where exists
            (select 1 from public.games where game_id=%s"""
        pmf_sql += " and season=%s)" if all_season else " and start_date>now())"
        pmf_sql += " on conflict(game_id) do update set margin_pmf=excluded.margin_pmf,total_pmf=excluded.total_pmf"
        pmf_rows = [
            (int(r.game_id), int(r.season), list(map(float, r.margin_pmf)),
             list(map(float, r.total_pmf)), int(r.game_id))
            + ((CURRENT_SEASON,) if all_season else ()) for r in pf.itertuples()]
        cur.executemany(pmf_sql, pmf_rows)
        cur.execute("select count(*) from public.game_predictions where run_id=%s and game_id=any(%s)",
                    (run, list(map(int,gp.game_id))))
        assert cur.fetchone()[0] == len(gp)
        cur.execute("notify pgrst, 'reload schema'")
    if archive is not None:
        report = {
            "published_at": datetime.now(timezone.utc).isoformat(), "season": CURRENT_SEASON,
            "coverage": "full_season", "published_games": len(gp), "run_id": str(run),
            "model_version": model_version,
            "retired_game_replacements": replacements,
            "database_backup": str(archive.resolve()),
            "backup_game_predictions_sha256": sha(archive / "game_predictions.parquet"),
            "backup_game_pmfs_sha256": sha(archive / "game_pmfs.parquet"),
        }
        save_json(archive / "publish_report.json", report)
    print(f"Published {len(gp):,} {'season' if all_season else 'future'} predictions and distributions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-season", action="store_true")
    args = parser.parse_args()
    main(args.all_season)
