"""Conservative storage retention for completed refresh runs.

This module is deliberately separate from refresh implementations.  A run is
eligible only after this deployment writes ``retention.json`` with
``retention_managed=true`` and ``completed=true``.  Existing runs therefore
remain untouched until a future refresh opts into retention.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_REQUIRED_FREE = 2 * 1024**3
DAILY_REFRESH_BYTES = 1_100_000_000
LIGHT_REFRESH_BYTES = 240_000_000
RETENTION_FILE = "retention.json"
# These are rebuildable stores.  Predictions, PMFs, reports, reviewed facts,
# and provenance files are intentionally absent from this list.
DERIVED_STORES = ("inputs", "players", "sequences", "full_history", "timeline", "context/full_history", "context/timeline")


class InsufficientStorageError(RuntimeError):
    """Raised when protected data prevents meeting the disk budget."""


@dataclass(frozen=True)
class CleanupResult:
    required_free_bytes: int
    free_before: int
    free_after: int
    pruned_runs: tuple[str, ...]
    pruned_bytes: int


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    # Refresh directories use UTC timestamp names; lexical order is stable
    # across copied filesystems where mtime ordering is not reliable.
    return sorted((p for p in root.iterdir() if p.is_dir() and not p.is_symlink()),
                  key=lambda p: p.name, reverse=True)


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _run_for_path(path: Path, runs: Iterable[Path]) -> Path | None:
    target = _resolved(path)
    for run in runs:
        base = _resolved(run)
        if target == base or base in target.parents:
            return run
    return None


def _referenced_runs(project_root: Path, runs: list[Path]) -> set[Path]:
    """Collect pins from narrow pointer/source files and incoming run refs."""
    protected: set[Path] = set()
    # Seed both retained generations before resolving their dependencies.
    # An obsolete lightweight run's leftover symlinks are provenance only;
    # they must not pin every old daily store forever.
    for parent in {run.parent for run in runs}:
        protected.update(_resolved(run) for run in sorted(
            (r for r in runs if r.parent == parent), key=lambda r: r.name, reverse=True)[:2])
    json_files: list[Path] = []
    models = project_root / "models"
    explicit = [
        # The active bundle and promotion record are live runtime pointers;
        # keep their input stores even when no dated refresh pointer exists.
        models / "player_sequence_production/active.json",
        models / "player_sequence_production/latest_promotion.json",
        models / "player_sequence_production/refreshes/latest.json",
        models / "player_sequence_production/availability_refreshes/latest.json",
        models / "player_sequence_production/validation_refresh/latest.json",
    ]
    if models.exists():
        json_files.extend(models.glob("player_sequence_production/*/source.json"))
        # Historical availability context sources can point at the immutable
        # daily arrays used to produce them.  Only the active and predecessor
        # availability runs are live inputs; scanning every old source here
        # would make every historical array permanently pinned.
        availability_sources = sorted(
            models.glob("player_sequence_production/availability_refreshes/*/context/source.json"),
            key=lambda p: p.parent.parent.name,
            reverse=True,
        )
        json_files.extend(availability_sources[:2])
        json_files.extend(models.glob("player_sequence_production/availability_validation/*/context/source.json"))
        json_files.extend(models.glob("experiments/*/source.json"))
        json_files.extend(models.glob("player_sequence_production/release_backups/*/refresh_latest.json"))
    json_files.extend(path for path in explicit if path.exists())
    for path in json_files:
        data = _load_json(path)
        for value in data.values():
            if isinstance(value, str):
                candidate = _run_for_path(Path(value), runs)
                if candidate:
                    protected.add(_resolved(candidate))
    # A run's symlinks are references to immutable stores.  Keep the run that
    # owns the target if the target is another run under this root.
    # Follow incoming context/source references only from already pinned runs;
    # a run's own self-referential report must not pin unrelated candidates.
    changed = True
    while changed:
        changed = False
        for run in runs:
            if _resolved(run) not in protected:
                continue
            for child in run.rglob("*"):
                if child.is_symlink():
                    target_run = _run_for_path(child, runs)
                    if target_run and _resolved(target_run) not in protected:
                        protected.add(_resolved(target_run)); changed = True
            for pointer in (run / "context_source.json", run / "context" / "source.json"):
                data = _load_json(pointer)
                for value in data.values():
                    if isinstance(value, str):
                        target_run = _run_for_path(Path(value), runs)
                        if target_run and _resolved(target_run) not in protected:
                            protected.add(_resolved(target_run)); changed = True
    return protected


def _size_without_symlink_targets(path: Path) -> int:
    if path.is_symlink() or not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(_size_without_symlink_targets(child) for child in path.iterdir())


def _free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return int(usage.free if hasattr(usage, "free") else usage[2])


def _eligible(run: Path) -> bool:
    meta = _load_json(run / RETENTION_FILE)
    return meta.get("retention_managed") is True and meta.get("completed") is True


def _record_prune(run: Path, stores: list[str], bytes_pruned: int) -> None:
    path = run / RETENTION_FILE
    meta = _load_json(path)
    meta.update({"retention_managed": True, "completed": True,
                 "pruned_at": datetime.now(timezone.utc).isoformat()})
    previous = meta.get("pruned", [])
    if not isinstance(previous, list):
        previous = []
    meta["pruned"] = sorted(set(previous) | set(stores))
    meta["pruned_bytes"] = int(meta.get("pruned_bytes", 0)) + bytes_pruned
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def mark_retention_managed(run: Path, *, completed: bool = True) -> Path:
    """Opt a completed future run into retention, returning its metadata path."""
    original = Path(run).expanduser()
    if original.is_symlink() or not original.is_dir():
        raise ValueError(f"refresh run must be a real directory: {original}")
    run = original.resolve()
    path = run / RETENTION_FILE
    meta = _load_json(path)
    meta.update({"retention_managed": True, "completed": bool(completed),
                 "managed_at": datetime.now(timezone.utc).isoformat()})
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return path


def prune_for_budget(root: Path, *, project_root: Path | None = None,
                     required_free_bytes: int = DEFAULT_REQUIRED_FREE,
                     retention_roots: Iterable[Path] | None = None) -> CleanupResult:
    """Prune only managed, unpinned derived stores until the budget is met."""
    root, project_root = _resolved(root), _resolved(project_root or root)
    roots = [root]
    for sibling in retention_roots or ():
        resolved = _resolved(Path(sibling))
        if resolved not in roots:
            roots.append(resolved)
    free_before = _free_bytes(root)
    if free_before >= required_free_bytes:
        return CleanupResult(required_free_bytes, free_before, free_before, (), 0)
    runs = [run for refresh_root in roots for run in _run_dirs(refresh_root)]
    protected = _referenced_runs(project_root, runs)
    # Keep current and predecessor in *each* managed root, even if their
    # pointers are incomplete.  A daily cleanup may therefore reclaim an old
    # light run (and vice versa) when both roots share one disk budget.
    for refresh_root in roots:
        root_runs = _run_dirs(refresh_root)
        if root_runs:
            protected.add(_resolved(root_runs[0]))
        if len(root_runs) > 1:
            protected.add(_resolved(root_runs[1]))
    candidates = [r for r in runs if _eligible(r) and _resolved(r) not in protected]
    pruned: list[str] = []
    reclaimed = 0
    for run in reversed(candidates):
        stores: list[str] = []
        reclaimed_run = 0
        for name in DERIVED_STORES:
            child = run / name
            if not child.exists() or child.is_symlink():
                continue
            amount = _size_without_symlink_targets(child)
            shutil.rmtree(child)
            stores.append(name)
            reclaimed_run += amount
        if stores:
            _record_prune(run, stores, reclaimed_run)
            pruned.append(str(run))
            reclaimed += reclaimed_run
        if _free_bytes(root) >= required_free_bytes:
            break
    free_after = _free_bytes(root)
    if free_after < required_free_bytes:
        raise InsufficientStorageError(
            f"safe refresh cleanup could free only {free_after - free_before} bytes; "
            f"need {required_free_bytes} bytes free (protected or unmanaged stores remain)")
    return CleanupResult(required_free_bytes, free_before, free_after,
                         tuple(pruned), reclaimed)


def before_refresh(root: Path, *, project_root: Path | None = None,
                   required_free_bytes: int = DEFAULT_REQUIRED_FREE,
                   retention_roots: Iterable[Path] | None = None) -> CleanupResult:
    """Budget disk before creating a daily or light refresh run."""
    return prune_for_budget(root, project_root=project_root,
                            required_free_bytes=required_free_bytes,
                            retention_roots=retention_roots)


def before_daily_refresh(root: Path, *, project_root: Path | None = None,
                        required_free_bytes: int = DEFAULT_REQUIRED_FREE,
                        retention_roots: Iterable[Path] | None = None) -> CleanupResult:
    """Budget space for a daily rebuild and its approximately 1.1GB output."""
    return before_refresh(root, project_root=project_root,
                          required_free_bytes=max(required_free_bytes, DAILY_REFRESH_BYTES),
                          retention_roots=retention_roots)


def before_light_refresh(root: Path, *, project_root: Path | None = None,
                        required_free_bytes: int = 512 * 1024**2,
                        retention_roots: Iterable[Path] | None = None) -> CleanupResult:
    """Budget space for a lightweight refresh and its approximately 240MB output."""
    return before_refresh(root, project_root=project_root,
                          required_free_bytes=max(required_free_bytes, LIGHT_REFRESH_BYTES),
                          retention_roots=retention_roots)


def after_successful_refresh(run: Path) -> Path:
    """Mark a newly completed run as eligible for future cleanup."""
    return mark_retention_managed(run, completed=True)
