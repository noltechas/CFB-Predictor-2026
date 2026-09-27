"""Replace byte-identical archived captures with hard links to one copy.

    python3 src/cfbpred/link_archive_duplicates.py            # report only
    python3 src/cfbpred/link_archive_duplicates.py --apply

Only documents a run manifest lists as a captured source are touched, never
manifests, receipts or latest pointers (some of those are rewritten in place).
Every path keeps its bytes and SHA, so all archive verification still passes.
New captures link at write time (pull_player_context.archive, link_previous).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.config import RAW

ROOTS = (RAW / "conference_availability", RAW / "player_context")
MIN_BYTES = 4096


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def captured_documents(root: Path) -> dict[str, list[Path]]:
    """Manifest-listed documents grouped by recorded SHA, each re-verified."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for manifest in sorted(root.glob("*/manifest.json")):
        for item in json.loads(manifest.read_text()).get("sources", []):
            path = Path(item["path"])
            if path.parent.parent != root.resolve() or path.stat().st_size < MIN_BYTES:
                continue
            if sha(path) != item["sha256"]:
                raise ValueError(f"Archived capture no longer matches its manifest: {path}")
            groups[item["sha256"]].append(path)
    return groups


def link_duplicates(root: Path, apply: bool) -> tuple[int, int]:
    linked = reclaimed = 0
    for digest, paths in captured_documents(root).items():
        canonical, *rest = sorted(set(paths), key=lambda p: (p.parent.name, p.name))
        for path in rest:
            if os.stat(path).st_ino == os.stat(canonical).st_ino:
                continue
            size = path.stat().st_size
            if apply:
                temporary = path.with_name(f".{path.name}.link")
                os.link(canonical, temporary)
                os.replace(temporary, path)  # atomic: readers see old or new, same bytes
                if sha(path) != digest:
                    raise ValueError(f"Linked capture changed: {path}")
            linked += 1; reclaimed += size
        if apply and rest:
            os.chmod(canonical, 0o444)  # shared inode: a write would change every run
    return linked, reclaimed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    for root in ROOTS:
        linked, reclaimed = link_duplicates(root, args.apply)
        verb = "linked" if args.apply else "would link"
        print(f"{root.relative_to(RAW.parent.parent)}: {verb} {linked} duplicate captures, {reclaimed / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
