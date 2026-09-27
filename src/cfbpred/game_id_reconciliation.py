"""Audited reconciliation of verified duplicate raw-stat game IDs.

This utility is intentionally independent of the feature and player-input
stores.  It remaps only the supplied raw frame, removes byte-for-byte
duplicate observations created by the remap, and quarantines conflicting
observations instead of choosing, summing, or overwriting one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd


KEY_COLUMNS = ("game_id", "home_away", "athlete_id", "category", "stat_type")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_aliases(path: Path) -> dict[int, int]:
    """Load ``old -> canonical`` IDs from an object or entry-list JSON."""
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        pairs = raw.items()
    elif isinstance(raw, list):
        pairs = ((x["old_game_id"], x["canonical_game_id"]) for x in raw)
    else:
        raise ValueError("aliases JSON must be an object or entry list")
    aliases = {int(old): int(new) for old, new in pairs}
    for old, new in aliases.items():
        if old == new:
            raise ValueError(f"self-alias is not a reconciliation: {old}")
        if new in aliases:
            raise ValueError(f"alias chains are not allowed: {old}->{new}")
    return aliases


@dataclass(frozen=True)
class Reconciliation:
    frame: pd.DataFrame
    quarantined: pd.DataFrame
    audit: dict


def reconcile_frame(frame: pd.DataFrame, aliases: Mapping[int, int]) -> Reconciliation:
    """Apply verified aliases to one raw-stat dataframe.

    Exact duplicate rows are removed after remapping.  A duplicate identity
    key with any differing raw field is quarantined in full; no conflicting
    row reaches the output frame.
    """
    missing = [c for c in KEY_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"raw stats missing reconciliation columns: {missing}")
    aliases = {int(k): int(v) for k, v in aliases.items()}
    out = frame.copy(deep=True)
    original_ids = pd.to_numeric(out["game_id"], errors="raise").astype("int64")
    out["game_id"] = original_ids.map(lambda x: aliases.get(int(x), int(x))).astype("int64")

    exact_duplicate_mask = out.duplicated(keep="first")
    deduped = out.loc[~exact_duplicate_mask].copy()
    # A key conflict is defined by differing values in any non-key raw field.
    # Grouping on all key fields catches conflicts across the production and
    # FCS files while retaining the complete evidence in quarantine.
    non_keys = [c for c in out.columns if c not in KEY_COLUMNS]
    if non_keys:
        varying = deduped.groupby(list(KEY_COLUMNS), dropna=False, sort=False)[non_keys].nunique(dropna=False)
        conflict_keys = varying.gt(1).any(axis=1)
    else:
        conflict_keys = pd.Series(False, index=pd.MultiIndex.from_tuples([], names=KEY_COLUMNS))
    # Existing raw files may legitimately contain repeated identity keys for
    # unrelated game IDs (provider fragments, null identifiers, etc.).  Only
    # groups whose game ID was actually changed can have a new collision from
    # this reconciliation; leave all untouched observations byte-for-byte in
    # the output and report conflicts for the affected canonical game only.
    if len(conflict_keys):
        affected = set(int(v) for v in aliases.values())
        conflict_keys = conflict_keys[
            [int(idx[0]) in affected for idx in conflict_keys.index]
        ]
    if len(conflict_keys):
        key_index = pd.MultiIndex.from_frame(deduped[list(KEY_COLUMNS)])
        conflict_mask = key_index.isin(conflict_keys[conflict_keys].index)
    else:
        conflict_mask = pd.Series(False, index=deduped.index).to_numpy()
    quarantined = deduped.loc[conflict_mask].copy()
    clean = deduped.loc[~conflict_mask].copy()
    audit = {
        "input_rows": int(len(frame)),
        "output_rows": int(len(clean)),
        "quarantined_rows": int(len(quarantined)),
        "exact_duplicate_rows_removed": int(exact_duplicate_mask.sum()),
        "conflicting_key_groups": int(conflict_keys.sum()) if len(conflict_keys) else 0,
        "aliases": {str(k): int(v) for k, v in aliases.items()},
        "remapped_rows": int(original_ids.isin(list(aliases)).sum()),
        "output_preserves_columns": list(frame.columns) == list(clean.columns),
    }
    return Reconciliation(clean, quarantined, audit)


def reconcile_file(raw_path: Path, aliases_path: Path, output: Path) -> dict:
    """Reconcile one parquet file and write an isolated output bundle."""
    aliases = load_aliases(aliases_path)
    raw = pd.read_parquet(raw_path)
    result = reconcile_frame(raw, aliases)
    output.mkdir(parents=True, exist_ok=False)
    clean_path = output / raw_path.name
    quarantine_path = output / (raw_path.stem + ".quarantined.parquet")
    result.frame.to_parquet(clean_path, index=False)
    result.quarantined.to_parquet(quarantine_path, index=False)
    audit = {
        **result.audit,
        "input_path": str(raw_path.resolve()),
        "input_sha256": _sha256(raw_path),
        "output_path": str(clean_path.resolve()),
        "output_sha256": _sha256(clean_path),
        "quarantine_path": str(quarantine_path.resolve()),
        "quarantine_sha256": _sha256(quarantine_path),
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rawpath", type=Path, required=True)
    parser.add_argument("--aliases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reconcile_file(args.rawpath, args.aliases, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
