"""Build isolated model inputs from a reviewed, expanded historical schedule."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

from cfbpred.pregame_experiments import save_json, sha


def build(schedule_path: Path, output: Path, reviewed: Path, raw: Path, market: Path):
    if output.exists():
        raise ValueError("Use a new immutable input directory")
    output.mkdir(parents=True)
    base = output / "raw_features"
    base.mkdir()
    schedule = pd.read_parquet(schedule_path)
    if schedule.game_id.duplicated().any() or schedule.game_id.le(0).any():
        raise ValueError("Expanded training schedule needs unique real game IDs")
    schedule["kick"] = pd.to_datetime(schedule.kick, utc=True)
    schedule.to_parquet(base / "snapshot_schedule.parquet", index=False)
    schedule.to_parquet(base / "games.parquet", index=False)
    shutil.copy2(market, base / "market.parquet")
    import cfbpred.features_raw as fr
    original = fr.PQ, fr.FEAT, fr.load_games

    def loader(con, fbs_only=True):
        frame = schedule
        if fbs_only:
            frame = frame[frame.home_division.eq("fbs") | frame.away_division.eq("fbs")]
        return frame.sort_values(["season", "kick", "game_id"]).reset_index(drop=True).copy()

    fr.PQ, fr.FEAT, fr.load_games = base, base, loader
    try:
        fr.main()
    finally:
        fr.PQ, fr.FEAT, fr.load_games = original
    from cfbpred.recent_form import build as recent
    from cfbpred.prepare_player_data import prepare
    from cfbpred.player_sequence_inputs import build as sequences
    from cfbpred.player_full_history import build as history
    from cfbpred.player_timeline import build as timeline
    from cfbpred.player_context_audit import audit
    from cfbpred.conference_availability import load_facts
    recent(base)
    paths = {name: str((output / name).resolve())
             for name in ("inputs", "players", "sequences", "full_history", "timeline")}
    prepare(base, Path(paths["inputs"]), Path(paths["players"]))
    sequences(Path(paths["players"]), Path(paths["inputs"]), Path(paths["sequences"]))
    history(Path(paths["players"]), Path(paths["sequences"]), Path(paths["inputs"]), Path(paths["full_history"]))
    source = output / "source.json"
    save_json(source, paths)
    facts = json.loads(reviewed.read_text())
    extra, issues = load_facts(schedule)
    save_json(output / "reviewed_and_current.json", facts + extra)
    save_json(output / "source_issues.json", issues)
    timeline(source, raw, output / "reviewed_and_current.json", Path(paths["timeline"]))
    audit(source, output / "audit")
    save_json(output / "manifest.json", dict(complete=True, source_sha256=sha(source),
              schedule=str(schedule_path.resolve()), schedule_sha256=sha(schedule_path),
              reviewed_sha256=sha(reviewed), production_modified=False))
    return source


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("schedule", "output", "reviewed", "raw", "market"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    print(build(a.schedule, a.output, a.reviewed, a.raw, a.market))
