"""Append-only collection of public roster, transfer and availability evidence.

Retrieval time is never substituted for historical publication time. These
archives are evidence, not automatically eligible pregame model features.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs

from bs4 import BeautifulSoup
import requests

from cfbpred.cfbd_client import CFBDClient
from cfbpred.config import ROOT, CURRENT_SEASON

OUTPUT = ROOT / "data/raw/player_context"
PUBLIC_INDEXES = {
    "sec_current": "https://www.secsports.com/fbreports",
    "sec_archive": "https://www.secsports.com/fbreports-archive",
    "bigten_2025": "https://bigten.org/fb/article/37919/",
    "bigten_current": "https://bigten.org/fb/availability-reports/",
    "big12_archive": "https://big12sports.com/sports/2025/8/21/FBReportArchive.aspx",
    "acc_archive": "https://theacc.com/sports/2025/8/28/availability-reporting-football-archive.aspx",
    "mac_current": "https://getsomemaction.com/sports/2026/7/28/FB_0728265050.aspx.aspx",
}


def previous_capture(directory, name, digest, size, *, lookback=3):
    """An earlier run's byte-identical copy of this document, if any.

    Captures live at <root>/<stamp>/<name>; the 30-minute jobs re-capture
    unchanged reports, which had grown the conference archive to 3.6 GB of
    which 3.4 GB were duplicates. Only the newest few runs are checked.
    """
    earlier = sorted((p for p in directory.parent.glob(f"*/{name}") if p.parent != directory), key=lambda p: p.parent.name)
    for candidate in reversed(earlier[-lookback:]):
        if candidate.stat().st_size == size and hashlib.sha256(candidate.read_bytes()).hexdigest() == digest:
            return candidate
    return None


def archive(body, url, directory, name, *, params=None, observed_at=None, link_previous=False):
    """Write one immutable capture plus its receipt.

    With ``link_previous`` an unchanged document becomes a hard link to its
    earlier capture: same path, bytes and SHA, one copy on disk. Linked files are
    made read-only, because a write to one would change every run sharing it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    observed_at = observed_at or datetime.now(timezone.utc).isoformat()
    digest = hashlib.sha256(body).hexdigest()
    path = directory / name
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Refusing to overwrite archived evidence: {path}")
        # Identical bytes: never rewrite in place (the file may be a shared link).
    else:
        previous = previous_capture(directory, name, digest, len(body)) if link_previous else None
        if previous is None:
            path.write_bytes(body)
        else:
            os.link(previous, path)
            os.chmod(path, 0o444)
    meta = dict(url=url, params=params or {}, observed_at=observed_at,
                published_at=None, sha256=digest, path=str(path.resolve()))
    path.with_suffix(path.suffix + ".source.json").write_text(json.dumps(meta, indent=2)+"\n")
    return meta


def html_documents(text, base_url):
    """Read ordinary links and SEC's JSON-encoded public content blocks."""
    soup = BeautifulSoup(text, "html.parser")
    fragments = [text]
    for element in soup.select("[data-page]"):
        data = json.loads(element["data-page"])
        def visit(x):
            if isinstance(x, dict):
                for key, value in x.items():
                    if key == "paragraph" and isinstance(value, str): fragments.append(value)
                    else: visit(value)
            elif isinstance(x, list):
                for value in x: visit(value)
        visit(data)
    links = {}
    for fragment in fragments:
        for element in BeautifulSoup(fragment, "html.parser").find_all(["a", "iframe"]):
            href = element.get("href") or element.get("src")
            if href:
                links[urljoin(base_url, href)] = element.get_text(" ", strip=True)
    return links


def pull(output=OUTPUT, start_year=2018, end_year=CURRENT_SEASON, *, api=True):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = output / stamp
    run.mkdir(parents=True, exist_ok=False)
    sources, failures = [], []
    if api:
        client = CFBDClient()
        for year in range(start_year, end_year+1):
            data = client.get("/player/portal", {"year": year})
            sources.append(archive(json.dumps(data).encode(), "https://api.collegefootballdata.com/player/portal", run,
                                   f"portal_{year}.json", params={"year": year}, link_previous=True))
            print(f"portal {year}: {len(data)}", flush=True)
        for endpoint, params, name in [
            ("/roster", {"year": end_year}, f"roster_{end_year}.json"),
            *[("/recruiting/players", {"year": end_year, "classification": kind},
               f"recruiting_{end_year}_{kind}.json") for kind in ["HighSchool", "JUCO", "PrepSchool"]],
        ]:
            data = client.get(endpoint, params)
            sources.append(archive(json.dumps(data).encode(), "https://api.collegefootballdata.com"+endpoint,
                                   run, name, params=params, link_previous=True))
            print(f"{name}: {len(data)}", flush=True)
    session = requests.Session()
    for name, url in PUBLIC_INDEXES.items():
        try:
            response = session.get(url, timeout=40); response.raise_for_status()
            sources.append(archive(response.content, url, run, name+".html", link_previous=True))
            links = html_documents(response.text, url)
            (run/(name+"_links.json")).write_text(json.dumps(links, indent=2)+"\n")
            for link, label in links.items():
                if ".pdf" not in link.lower(): continue
                response = session.get(link, timeout=40); response.raise_for_status()
                if not response.content.startswith(b"%PDF"): continue
                filename = name+"_"+hashlib.sha256(link.encode()).hexdigest()[:16]+".pdf"
                item = archive(response.content, link, run, filename, link_previous=True); item["label"] = label
                sources.append(item)
            # SEC's public reporting partner. No private/admin routes or keys.
            for link in links:
                parsed=urlparse(link);query=parse_qs(parsed.query)
                report_type=query.get("type",[None])[0];conference=query.get("conf",[None])[0]
                if parsed.netloc=="app.hdintelligence.com" and report_type in {"archive","report"} and conference in {"SEC","B10","ACC","Big12","MAC"}:
                    endpoint="https://app.hdintelligence.com/api/get-"+("archive" if report_type=="archive" else "publish")+"-public"
                    params={"sport":"Football", "organization":conference, "conference":conference}
                    response = session.post(endpoint,
                                            json=params, timeout=40)
                    response.raise_for_status()
                    sources.append(archive(response.content, endpoint,
                                           run, name+"_"+report_type+"_public.json", params=params, link_previous=True))
            print(name, "links", len(links), flush=True)
        except requests.RequestException as error:
            failures.append(dict(source=name, url=url, error=str(error)))
    manifest = dict(run=str(run.resolve()), sources=sources, failures=failures,
                    historical_publication_times_not_assumed=True)
    (run/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (output/"latest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(run, flush=True)
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=CURRENT_SEASON)
    parser.add_argument("--no-api", action="store_true")
    args = parser.parse_args()
    pull(args.output, args.start_year, args.end_year, api=not args.no_api)
