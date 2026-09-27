"""Sun Belt availability feed: archive every GitHub gist revision, derive facts.

The Sun Belt's report page renders a public gist. GitHub keeps every revision
with its commit time, so a revision collected later still proves what the feed
said before kickoff. Each collection archives the commit listing and any
revision bodies not yet held (content-addressed), then rebuilds the index.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from cfbpred.config import ROOT as PROJECT_ROOT
from cfbpred.pregame_experiments import save_json
from cfbpred.pull_player_context import archive

ROOT = PROJECT_ROOT / "data/raw/sunbelt_availability"
GIST = "d8c9c47e07ad39300ce34096c3abd775"
RAW = "https://gist.githubusercontent.com/SunBeltConference/{gist}/raw/{version}/availability-report.json"


def collect(root: Path = ROOT, session=None) -> Path:
    session = session or requests.Session()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    commits, page = [], 1
    while True:
        url = f"https://api.github.com/gists/{GIST}/commits?per_page=100&page={page}"
        response = session.get(url, timeout=60)
        response.raise_for_status()
        rows = response.json()
        archive(response.content, url, root, f"commits_page_{stamp}_{page}.json")
        commits += rows
        if len(rows) < 100:
            break
        page += 1
    held = {p.stem for p in root.glob("*.json") if len(p.stem) == 64}
    index_path = root / "index.json"
    index = {e["version"]: e for e in (json.loads(index_path.read_text()) if index_path.exists() else [])}
    for commit in commits:
        if commit["version"] in index and index[commit["version"]]["sha256"] in held:
            continue
        url = RAW.format(gist=GIST, version=commit["version"])
        body = session.get(url, timeout=60).content
        digest = hashlib.sha256(body).hexdigest()
        if digest not in held:
            archive(body, url, root, f"{digest}.json")
            held.add(digest)
        index[commit["version"]] = dict(committed_at=commit["committed_at"], version=commit["version"],
                                        url=url, sha256=digest, bytes=len(body))
    save_json(index_path, sorted(index.values(), key=lambda e: e["committed_at"]))
    return root


def load_facts(schedule, root: Path = ROOT):
    from cfbpred.availability_archive import fixture_index, gist_facts
    if not (root / "index.json").exists():
        return [], []
    return gist_facts(root / "index.json", root, fixture_index(schedule))


if __name__ == "__main__":
    print(collect())
