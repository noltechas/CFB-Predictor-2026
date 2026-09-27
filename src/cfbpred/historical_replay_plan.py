"""Durable cutoffs for retrospective ranking/prediction reconstruction.

This module only inventories targets. It does not fit, score, publish, or
claim historical predictive accuracy. ``latest_model_reconstruction`` uses
the active model weights against each cutoff and is explicitly retrospective.
"""
from __future__ import annotations

import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def enumerate_cutoffs(schedule: pd.DataFrame, first_season: int = 2001,
                      last_season: int = 2026) -> list[dict[str, Any]]:
    frame = schedule.copy()
    frame["kick"] = pd.to_datetime(frame["kick" if "kick" in frame else "start_date"], utc=True)
    frame = frame[frame.season.between(first_season, last_season)].copy()
    frame["week_key"] = (frame["week_c"] if "week_c" in frame else frame["week"]).fillna(0).astype(int)
    rows: list[dict[str, Any]] = []
    for season, games in frame.groupby("season", sort=True):
        games = games.sort_values(["kick", "game_id"])
        first = games.kick.min()
        cutoffs = [(0, first - pd.Timedelta(seconds=1), "preseason")]
        for week, group in games.groupby("week_key", sort=True):
            cutoffs.append((int(week), group.kick.max() + pd.Timedelta(seconds=1), "after_completed_week"))
        for week, cutoff, label in cutoffs:
            done = games[games.kick <= cutoff]
            target = games[games.kick > cutoff]
            prior = frame[frame.kick < cutoff]
            no_prior = prior.empty
            rows.append({"season": int(season), "week": int(week), "cutoff": cutoff.isoformat(),
                         "cutoff_label": label, "game_count": int(len(games)),
                         "completed_before_cutoff": int(len(done)),
                         "target_game_count": int(len(target)),
                         "target_game_ids": [int(x) for x in target.game_id],
                         "completed_game_ids": [int(x) for x in done.game_id],
                         "prior_game_count_all_seasons": int(len(prior)),
                         "unsupported_reason": "no_prior_data" if no_prior else None,
                         "reconstruction_state": "pending",
                         "walk_forward_state": "unsupported_current_adapter",
                         "required_train_through": "2025-12-31T23:59:59Z",
                         "mode": "latest_model_reconstruction",
                         "historical_accuracy_claim": False})
    return rows


def build_plan(schedule_path: str | Path, output: str | Path,
               first_season: int = 2001, last_season: int = 2026) -> dict[str, Any]:
    path = Path(schedule_path)
    rows = enumerate_cutoffs(pd.read_parquet(path), first_season, last_season)
    plan = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "schedule_path": str(path), "season_range": [first_season, last_season],
            "mode": "latest_model_reconstruction", "rows": rows,
            "constraints": [
                "Latest weights trained through 2025 are reused; outputs are retrospective reconstructions, not as-issued forecasts.",
                "No external publishing is performed by this plan.",
                "Walk-forward fitting requires a separate arbitrary-cutoff training adapter.",
                "Missing roster/history/source records remain unknown or unsupported."]}
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--first-season", type=int, default=2001)
    ap.add_argument("--last-season", type=int, default=2026)
    a = ap.parse_args()
    plan = build_plan(a.schedule, a.output, a.first_season, a.last_season)
    print(json.dumps({"rows": len(plan["rows"]), "seasons": plan["season_range"]}))


if __name__ == "__main__": main()
