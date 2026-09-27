"""Conservative discovery/cache helpers for official team archives.

This module discovers archive *documents*, not player health.  A provider is
marked supported only after its public route has answered successfully.  A
missing article or unlisted player never becomes a healthy/available fact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import re
import glob
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urljoin

import pandas as pd


NEBRASKA_PROVIDER = "sidearm_nuxt_website_api"


@dataclass(frozen=True)
class TeamArchive:
    team: str
    division: str
    official_url: str | None
    provider: str | None
    state: str  # supported, unsupported, pending, blocked
    reason: str


def modeled_team_manifest(schedule_path: str | Path) -> list[dict[str, Any]]:
    """Return one explicit archive state for every modeled FBS/FCS team."""
    frame = pd.read_parquet(schedule_path)
    teams: dict[str, str] = {}
    for side in ("home", "away"):
        cols = [f"{side}_team", f"{side}_division"]
        # A team can cross the FBS/FCS boundary over time; current model
        # season is the authoritative classification for the manifest.
        ordered_frame = frame.sort_values("season", ascending=False) if "season" in frame.columns else frame
        ordered = ordered_frame[cols].drop_duplicates(f"{side}_team")
        for team, division in ordered.itertuples(index=False):
            teams[str(team)] = str(division)
    return [asdict(TeamArchive(t, d, None, None, "pending", "official domain not yet discovered"))
            for t, d in sorted(teams.items()) if d in {"fbs", "fcs"}]


def _request(session: Any, url: str, **kwargs: Any) -> Any:
    response = session.get(url, timeout=30, **kwargs)
    response.raise_for_status()
    return response


def discover_nebraska(session: Any, base_url: str = "https://huskers.com") -> dict[str, Any]:
    """Detect Nebraska's public Sidearm/Nuxt routes from live API responses."""
    base = base_url.rstrip("/")
    url = f"{base}/website-api/sports?filter[slug]=football"
    response = _request(session, url)
    payload = response.json()
    rows = payload.get("data", payload if isinstance(payload, list) else [])
    football_id = None
    if rows:
        first = rows[0]
        football_id = first.get("id") if isinstance(first, dict) else None
    return {"provider": NEBRASKA_PROVIDER, "team": "Nebraska", "official_url": base,
            "state": "supported", "football_sport_id": football_id,
            "routes": {"sports": url, "schedules": f"{base}/website-api/schedules",
                       "schedule_events": f"{base}/website-api/schedule-events",
                       "article_by_slug": f"{base}/website-api/articles/by/slug/{{slug}}"},
            "observed_at": datetime.now(timezone.utc).isoformat()}


def discover_cached_directory(directory_files: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Extract official athletics links from cached conference-directory HTML.

    The file path is retained as provenance.  A directory link alone only
    establishes a candidate domain; it does not establish an archive provider.
    """
    rows = {}
    pattern = re.compile(r'"title":"([^"]+)"[^{}]{0,1200}?"athletics_website":"(https?[^" ]+)', re.S)
    for filename in directory_files:
        path = Path(filename)
        text = path.read_text(errors="ignore")
        for team, url in pattern.findall(text):
            rows.setdefault(team, {"team": team, "official_url": url,
                                   "source_path": str(path), "state": "pending",
                                   "reason": "official directory link; provider unverified"})
    return sorted(rows.values(), key=lambda row: row["team"])


def probe_official_domains(session: Any, candidates: Iterable[Mapping[str, Any]],
                           output: str | Path, delay_seconds: float = 1.0,
                           sleep: Callable[[float], None] = time.sleep) -> list[dict[str, Any]]:
    """Probe candidate homepages and preserve status; never guesses route support."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    results = []
    for i, candidate in enumerate(candidates):
        if i: sleep(delay_seconds)
        row = dict(candidate); url = str(candidate["official_url"])
        try:
            response = session.get(url, timeout=20, allow_redirects=True)
            row.update({"http_status": response.status_code, "final_url": response.url,
                        "observed_at": datetime.now(timezone.utc).isoformat()})
            if response.status_code < 400:
                row.update({"state": "pending", "reason": "domain reachable; football archive route unverified"})
            else:
                row.update({"state": "blocked", "reason": f"official link HTTP {response.status_code}"})
        except Exception as exc:
            row.update({"state": "blocked", "reason": f"official link probe failed: {exc!r}"})
        results.append(row)
    (output / "domain_probe_receipts.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def cache_directory_pages(session: Any, urls: Iterable[str], output: str | Path,
                          delay_seconds: float = 1.0,
                          sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Cache public conference directory pages, stopping on 429."""
    output = Path(output); raw = output / "conference_pages"; raw.mkdir(parents=True, exist_ok=True)
    urls = list(dict.fromkeys(urls))
    receipts = []
    for i, url in enumerate(urls):
        if i: sleep(delay_seconds)
        try:
            response = session.get(url, timeout=30, allow_redirects=True)
        except Exception as exc:
            receipts.append({"url": url, "status": "error", "error": repr(exc)}); continue
        if response.status_code == 429:
            receipts.append({"url": url, "status": 429, "retry_after": response.headers.get("Retry-After")}); break
        stem = hashlib.sha256(url.encode()).hexdigest()[:16]
        receipts.append(cache_response(response, raw / f"{stem}.html", url))
    done = {r["url"] for r in receipts if r.get("status") == 200}
    payload = {"receipts": receipts, "pending": [u for u in urls if u not in done]}
    (output / "conference_page_receipts.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def cache_response(response: Any, destination: str | Path, url: str,
                   requested_at: str | None = None) -> dict[str, Any]:
    """Write exact response bytes and a receipt suitable for resuming."""
    data = response.content
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"url": url, "status": response.status_code, "path": str(path),
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "requested_at": requested_at or datetime.now(timezone.utc).isoformat(),
            "headers": {k: v for k, v in response.headers.items()
                        if k.lower() in {"content-type", "retry-after", "etag", "last-modified"}}}


def fetch_json_queue(session: Any, urls: Iterable[str], raw_dir: str | Path,
                     delay_seconds: float = 1.0,
                     sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Fetch a queue, stopping at the first 429 and preserving pending URLs."""
    urls = list(dict.fromkeys(urls)); receipts = []
    for i, url in enumerate(urls):
        if i: sleep(delay_seconds)
        try:
            response = session.get(url, timeout=30)
        except Exception as exc:  # retain a resumable failure, don't hide it
            receipts.append({"url": url, "status": "error", "error": repr(exc)})
            continue
        if response.status_code == 429:
            receipts.append({"url": url, "status": 429,
                             "retry_after": response.headers.get("Retry-After")})
            break
        stem = hashlib.sha256(url.encode()).hexdigest()[:16]
        receipt = cache_response(response, Path(raw_dir) / f"{stem}.json", url)
        receipts.append(receipt)
    # Only a cached successful response is complete. Errors and 429s stay
    # resumable, including the URL that triggered the rate-limit boundary.
    done = {r["url"] for r in receipts if r.get("status") == 200}
    return {"receipts": receipts, "pending": [u for u in urls if u not in done],
            "stopped_on_429": any(r.get("status") == 429 for r in receipts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nebraska", action="store_true")
    parser.add_argument("--directory", action="append", default=[],
                        help="cached conference directory HTML (repeatable or glob)")
    parser.add_argument("--probe-domains", action="store_true")
    parser.add_argument("--conference-url", action="append", default=[])
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    manifest = modeled_team_manifest(args.schedule)
    result: dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(),
                              "teams": manifest, "providers": []}
    if args.nebraska:
        import requests
        provider = discover_nebraska(requests.Session())
        result["providers"].append(provider)
        for row in manifest:
            if row["team"] == "Nebraska":
                row.update({"official_url": provider["official_url"], "provider": provider["provider"],
                            "state": provider["state"], "reason": "public football API verified"})
    if args.directory:
        files = [p for pattern in args.directory for p in glob.glob(pattern)]
        candidates = discover_cached_directory(files)
        result["directory_candidates"] = len(candidates)
        result["directory_sources"] = sorted(set(files))
        if args.probe_domains:
            import requests
            probed = probe_official_domains(requests.Session(), candidates, output)
            result["domain_probe_results"] = len(probed)
            candidates = {row["team"]: row for row in probed}
            for row in manifest:
                if row["team"] in candidates:
                    item = candidates[row["team"]]
                    row.update({"official_url": item["official_url"], "state": item["state"],
                                "reason": item["reason"], "domain_source_path": item["source_path"]})
        (output / "directory_candidates.json").write_text(json.dumps(
            list(candidates.values()) if isinstance(candidates, dict) else candidates, indent=2) + "\n")
    if args.conference_url:
        import requests
        cache_directory_pages(requests.Session(), args.conference_url, output)
    (output / "team_manifest.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
