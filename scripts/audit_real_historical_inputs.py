"""Audit a real-only historical ranking-input snapshot.

This is deliberately read-only: it writes only the requested JSON report.  The
snapshot may retain negative terminal fixtures in its player schedule because
those are valid ranking targets; real scalar targets are checked separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def same(a, b, tol=1e-8) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        return bool(math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol))
    return a == b


def ids(frame: pd.DataFrame) -> set[int]:
    return set(frame.game_id.astype("int64"))


def audit(output: Path, accepted: Path, report: Path) -> dict:
    report.parent.mkdir(parents=True, exist_ok=True)
    required = [
        output / "raw/real_feature_schedule.parquet",
        output / "raw/game_scalars.parquet",
        output / "inputs/game_scalars.parquet",
        output / "inputs/snapshot_schedule.parquet",
        output / "source.json",
        output / "manifest.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        result = {"status": "pending", "output": str(output), "missing": missing}
        report.write_text(json.dumps(result, indent=2))
        return result

    source = json.loads((output / "source.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    real_schedule = pd.read_parquet(output / "raw/real_feature_schedule.parquet")
    scalars = pd.read_parquet(output / "raw/game_scalars.parquet")
    prepared = pd.read_parquet(output / "inputs/game_scalars.parquet")
    superset = pd.read_parquet(output / "inputs/snapshot_schedule.parquet")
    di = real_schedule.home_division.isin(["fbs", "fcs"]) | real_schedule.away_division.isin(["fbs", "fcs"])
    di_ids = ids(real_schedule.loc[di])
    scalar_ids = ids(scalars)
    prepared_ids = ids(prepared)
    result: dict = {
        "status": "complete",
        "output": str(output),
        "real_schedule_rows": len(real_schedule),
        "real_di_target_ids": len(di_ids),
        "scalar_ids": len(scalar_ids),
        "prepared_scalar_ids": len(prepared_ids),
        "real_schedule_positive_unique": bool(real_schedule.game_id.gt(0).all() and real_schedule.game_id.is_unique),
        "scalar_ids_equal_prepared": scalar_ids == prepared_ids,
        "scalar_ids_equal_di_schedule": scalar_ids == di_ids,
        "scalar_ids_missing_from_di_schedule": sorted(scalar_ids - di_ids)[:20],
        "di_ids_missing_from_scalars": sorted(di_ids - scalar_ids)[:20],
        "superset_negative_rows": int(superset.game_id.le(0).sum()),
        "manifest": manifest,
        "source": source,
    }

    # Source and artifact hashes are useful even when the builder's manifest
    # uses a different hash key naming convention.
    result["hashes"] = {
        str(p.relative_to(output)): digest(p)
        for p in [output / "source.json", output / "manifest.json", output / "raw/real_feature_schedule.parquet", output / "raw/game_scalars.parquet", output / "inputs/game_scalars.parquet"]
    }

    # Scalar vocab semantics: numeric IDs must decode to the row's team names.
    vocab_path = output / "inputs/team_vocab.json"
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text())
        team_to_id = {str(k): int(v) for k, v in vocab.get("team", {}).items()}
        id_to_team = {v: k for k, v in team_to_id.items()}
        mismatches = []
        for side, short in (("home", "h"), ("away", "a")):
            token_col, name_col = short + "_idx", side + "_team"
            if token_col not in prepared or name_col not in prepared:
                continue
            for row in prepared[["game_id", token_col, name_col]].itertuples(index=False):
                expected = team_to_id.get(str(getattr(row, name_col)), 0)
                token = int(getattr(row, token_col))
                if token != expected or (token and id_to_team.get(token) != str(getattr(row, name_col))):
                    mismatches.append({"game_id": int(row.game_id), "side": side, "name": str(getattr(row, name_col)), "token": token, "expected": expected})
                    if len(mismatches) >= 20:
                        break
        result["vocab_semantic_mismatches"] = mismatches

    # No negative IDs may appear in completed box observations or sequence
    # events. Virtual roster references themselves are intentionally allowed.
    players = Path(source.get("players", ""))
    sequences = Path(source.get("sequences", ""))
    negatives = {}
    for label, path in (("clean_player_boxes", players / "clean_player_boxes.parquet"), ("sequence_events", sequences / "events.parquet")):
        if path.exists():
            frame = pd.read_parquet(path, columns=["game_id"])
            negatives[label] = int(frame.game_id.le(0).sum())
    result["negative_observation_ids"] = negatives

    # Verify the sequence tips are strictly before their roster-reference kick.
    tip_check = {"checked": False, "violations": None}
    arr_path, events_path, roster_path = sequences / "arrays.npz", sequences / "events.parquet", players / "game_rosters.parquet"
    if arr_path.exists() and events_path.exists() and roster_path.exists():
        with np.load(arr_path) as arrays:
            tips = arrays["history_tips"]
        events = pd.read_parquet(events_path, columns=["event_id", "kick"]).set_index("event_id")
        rosters = pd.read_parquet(roster_path).sort_values("offset")
        refs = rosters.loc[rosters.index.repeat(rosters.players)]
        ref_kick = pd.to_datetime(refs.kick, utc=True).astype("int64").to_numpy()
        event_kick = pd.to_datetime(events.kick, utc=True).astype("int64")
        valid = tips > 0
        tip_check = {"checked": True, "references": int(len(tips)), "with_history": int(valid.sum()), "violations": int((event_kick.reindex(tips[valid]).to_numpy() >= ref_kick[valid]).sum())}
    result["strict_history_tip_check"] = tip_check

    # Compare core (non-player, non-token) scalar fields against the accepted
    # complete-history context for 2026 and common FBS sides.
    accepted_path = accepted / "game_scalars.parquet"
    if accepted_path.exists():
        old = pd.read_parquet(accepted_path).set_index("game_id")
        new = prepared.set_index("game_id")
        common = new.index.intersection(old.index)
        bad_prefixes = ("av_", "cx_", "retp_", "port_", "n_players", "returning_share", "upper_share", "new_upper_share", "QB_", "OL_", "DL_", "SKILL_", "DB_", "LB_", "talent_", "recruit_", "coach_", "box_", "bx_")
        fields = [c for c in new.columns if (c.startswith("h_") or c.startswith("a_")) and c not in {"h_idx", "a_idx", "h_conf", "a_conf"} and not any(c[2:].startswith(p) for p in bad_prefixes)]
        changed = {"h_fbs": 0, "a_fbs": 0, "h_fcs": 0, "a_fcs": 0, "fields": {}}
        schedule_lookup = real_schedule.set_index("game_id")
        for gid in common:
            if gid not in schedule_lookup.index:
                continue
            row = schedule_lookup.loc[gid]
            for short, div in (("h", row.home_division), ("a", row.away_division)):
                if div not in ("fbs", "fcs"):
                    continue
                for field in fields:
                    if not same(new.at[gid, field], old.at[gid, field]):
                        key = f"{short}_{div}"
                        changed[key] += 1
                        changed["fields"][field] = changed["fields"].get(field, 0) + 1
        result["accepted_core_scalar_parity"] = changed
    else:
        result["accepted_core_scalar_parity"] = {"status": "missing", "path": str(accepted_path)}

    report.write_text(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--accepted", type=Path, default=Path("models/experiments/complete_history_20260922_v2/context/inputs"))
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    report = args.report or args.output / "audit/real_historical_inputs.json"
    print(json.dumps(audit(args.output, args.accepted, report), indent=2, default=str))
