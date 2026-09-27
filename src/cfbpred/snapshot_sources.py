"""Point-in-time archive of every source that overwrites itself.

The project has repeatedly lost pregame knowledge to sources that change in place: a
weekly-overwritten availability sheet, widget archives that now return empty lists,
articles revised after kickoff, and rosters and recruiting with no publication time.
Each run stores what a source says NOW, stamped with when it was seen, so any later
season can be trained on what was actually knowable before each kickoff.

* API payloads (rosters, schedules, weather forecasts, lines) -> public.source_snapshots
* raw documents (availability PDFs, HTML, JSON feeds) -> private Storage bucket
  `source-archive`, indexed in public.source_snapshots
Everything is deduplicated by content hash: an unchanged source adds one row to
public.source_observations, never another copy of its payload. Closing lines are kept
for evaluation only; the purity rule forbids them as model inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import psycopg
import requests
from psycopg.types.json import Jsonb

from cfbpred.config import CURRENT_SEASON, PQ, ROOT
from cfbpred.load_supabase import dsn

CFBD = "https://api.collegefootballdata.com"
BUCKET = "source-archive"
RAW = ROOT / "data/raw"
DDL = """
create table if not exists public.source_snapshots (
  id             bigserial primary key,
  source         text not null,
  scope          text not null,
  captured_at    timestamptz not null,
  content_sha256 text not null,
  n_records      int  not null,
  payload        jsonb not null,
  unique (source, scope, content_sha256)
);
create index if not exists source_snapshots_source_scope on public.source_snapshots (source, scope, captured_at);
create table if not exists public.source_observations (
  source         text not null,
  scope          text not null,
  captured_at    timestamptz not null,
  content_sha256 text not null,
  changed        boolean not null,
  primary key (source, scope, captured_at)
);
-- Private: the app never reads the archive, so no anon policy is created.
alter table public.source_snapshots enable row level security;
alter table public.source_observations enable row level security;
"""


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def cfbd(path: str, **params):
    r = requests.get(CFBD + path, params=params, timeout=90,
                     headers={"Authorization": f"Bearer {os.environ['CFBD_API_KEY']}"})
    r.raise_for_status()
    return r.json()


def current_weeks(season: int) -> list[int]:
    """The week in progress and the one after it, by kickoff time."""
    s = pd.read_parquet(PQ / "schedule.parquet", columns=["season", "week", "start_date"])
    s = s[s.season == season].copy(); s["kick"] = pd.to_datetime(s.start_date, format="mixed", utc=True)
    upcoming = s[s.kick > pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=12)].sort_values("kick")
    return sorted(upcoming.week.drop_duplicates().head(2).astype(int).tolist())


def api_sources(season: int) -> list[tuple[str, str, object]]:
    """(source, scope, payload) for every API source worth a dated copy."""
    out = [("cfbd_roster", str(season), cfbd("/roster", year=season)),
           ("cfbd_games", str(season), cfbd("/games", year=season))]
    for week in current_weeks(season):
        scope = f"{season}-week-{week}"
        out.append(("cfbd_weather", scope, cfbd("/games/weather", year=season, week=week, seasonType="regular")))
        out.append(("cfbd_lines", scope, cfbd("/lines", year=season, week=week)))
    if (PQ / "espn_lines.parquet").exists():
        out.append(("espn_lines", str(season), json.loads(pd.read_parquet(PQ / "espn_lines.parquet").to_json(orient="records", date_format="iso"))))
    roster = PQ / "cfbd/roster.parquet"   # CFBD plus the ESPN backfill: exactly what the model reads
    if roster.exists():
        r = pd.read_parquet(roster)
        r = r[r.get("season_pulled", pd.Series(season, index=r.index)) == season] if "season_pulled" in r else r
        out.append(("model_roster", str(season), json.loads(r.to_json(orient="records", date_format="iso"))))
    # A source with nothing to say yet (next week's forecasts) is not a snapshot.
    return [x for x in out if not (isinstance(x[2], list) and len(x[2]) == 0)]


def document_receipts() -> list[dict]:
    """Raw availability documents the collectors fetched, with their receipts."""
    docs = []
    for pointer in (RAW / "conference_availability/latest.json", RAW / "player_context/latest.json"):
        if pointer.exists():
            for s in json.loads(pointer.read_text()).get("sources", []):
                if s.get("path") and Path(s["path"]).exists():
                    docs.append(dict(url=s.get("url"), path=s["path"], observed_at=s.get("observed_at"), sha256=s.get("sha256")))
    for folder in (RAW / "sunbelt_availability",):
        for p in folder.glob("*.json") if folder.exists() else []:
            if not p.name.endswith(".source.json"):
                meta = p.with_suffix(p.suffix + ".source.json")
                m = json.loads(meta.read_text()) if meta.exists() else {}
                docs.append(dict(url=m.get("url"), path=str(p), observed_at=m.get("observed_at"), sha256=m.get("sha256")))
    covers = RAW / "covers_availability"
    if covers.exists():
        latest = sorted(p for p in covers.iterdir() if p.is_dir())[-1:]
        for run in latest:
            for p in run.rglob("*"):
                if p.is_file() and not p.name.endswith(".source.json"):
                    docs.append(dict(url=None, path=str(p), observed_at=None, sha256=None))
    return docs


class Archive:
    def __init__(self, dry: bool = False):
        self.dry = dry
        self.now = datetime.now(timezone.utc)
        self.conn = None if dry else psycopg.connect(dsn())
        self.url = os.environ["SUPABASE_URL"].rstrip("/")
        self.key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if self.conn:
            with self.conn.cursor() as cur:
                cur.execute(DDL)
            self.conn.commit()

    def record(self, source: str, scope: str, payload, n_records: int | None = None) -> bool:
        """Store the payload if it is new; always log that it was observed."""
        body = canonical(payload); digest = hashlib.sha256(body).hexdigest()
        n = n_records if n_records is not None else (len(payload) if isinstance(payload, list) else 1)
        if self.dry:
            print(f"  [dry] {source:18s} {scope:16s} {n:>7,} records  {len(body)/1e6:6.2f} MB  {digest[:12]}")
            return True
        with self.conn.cursor() as cur:
            cur.execute("""insert into public.source_snapshots (source, scope, captured_at, content_sha256, n_records, payload)
                           values (%s,%s,%s,%s,%s,%s) on conflict (source, scope, content_sha256) do nothing returning id""",
                        (source, scope, self.now, digest, n, Jsonb(payload)))
            changed = cur.fetchone() is not None
            cur.execute("""insert into public.source_observations (source, scope, captured_at, content_sha256, changed)
                           values (%s,%s,%s,%s,%s) on conflict do nothing""", (source, scope, self.now, digest, changed))
        self.conn.commit()
        return changed

    def ensure_bucket(self):
        if self.dry or not self.key:
            return
        h = {"Authorization": f"Bearer {self.key}", "apikey": self.key}
        r = requests.get(f"{self.url}/storage/v1/bucket/{BUCKET}", headers=h, timeout=30)
        if r.status_code != 200:
            requests.post(f"{self.url}/storage/v1/bucket", headers=h, timeout=30,
                          json={"id": BUCKET, "name": BUCKET, "public": False}).raise_for_status()

    def document(self, doc: dict) -> bool:
        data = Path(doc["path"]).read_bytes(); digest = hashlib.sha256(data).hexdigest()
        ext = Path(doc["path"]).suffix or ".bin"; object_path = f"documents/{digest[:2]}/{digest}{ext}"
        meta = dict(url=doc.get("url"), local_path=doc["path"], collector_observed_at=doc.get("observed_at"),
                    collector_sha256=doc.get("sha256"), bytes=len(data), object=f"{BUCKET}/{object_path}")
        if self.dry:
            return True
        with self.conn.cursor() as cur:
            cur.execute("select 1 from public.source_snapshots where source = 'availability_document' and content_sha256 = %s limit 1", (digest,))
            seen = cur.fetchone() is not None
        if not seen and self.key:
            h = {"Authorization": f"Bearer {self.key}", "apikey": self.key,
                 "Content-Type": mimetypes.guess_type(doc["path"])[0] or "application/octet-stream", "x-upsert": "true"}
            requests.post(f"{self.url}/storage/v1/object/{BUCKET}/{object_path}", headers=h, data=data, timeout=120).raise_for_status()
        scope = (doc.get("url") or Path(doc["path"]).name)[:200]
        with self.conn.cursor() as cur:
            cur.execute("""insert into public.source_snapshots (source, scope, captured_at, content_sha256, n_records, payload)
                           values ('availability_document', %s, %s, %s, 1, %s) on conflict (source, scope, content_sha256) do nothing returning id""",
                        (scope, self.now, digest, Jsonb(meta)))
            changed = cur.fetchone() is not None
            cur.execute("""insert into public.source_observations (source, scope, captured_at, content_sha256, changed)
                           values ('availability_document', %s, %s, %s, %s) on conflict do nothing""", (scope, self.now, digest, changed))
        self.conn.commit()
        return changed


def run(season: int = CURRENT_SEASON, dry: bool = False, documents: bool = True) -> dict:
    archive = Archive(dry)
    report = dict(captured_at=archive.now.isoformat(), season=season, api={}, documents=dict(seen=0, new=0), failures=[])
    try:
        for source, scope, payload in api_sources(season):
            try:
                report["api"][f"{source}:{scope}"] = "new" if archive.record(source, scope, payload) else "unchanged"
            except Exception as e:
                report["failures"].append(f"{source}:{scope}: {e}")
    except Exception as e:
        report["failures"].append(f"api fetch: {e}")
    if documents:
        try:
            archive.ensure_bucket()
            for doc in document_receipts():
                try:
                    report["documents"]["seen"] += 1; report["documents"]["new"] += archive.document(doc)
                except Exception as e:
                    report["failures"].append(f"document {doc['path']}: {e}")
        except Exception as e:
            report["failures"].append(f"documents: {e}")
    if archive.conn:
        archive.conn.close()
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", type=int, default=CURRENT_SEASON)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-documents", action="store_true")
    a = p.parse_args()
    r = run(a.season, a.dry_run, not a.no_documents)
    sys.exit(0 if not r["failures"] else 3)
