"""Team logos: ESPN CDN -> Supabase Storage -> public.teams URLs.

CFBD team ids ARE ESPN team ids, so every logo lives at a predictable CDN
path, in a light and a dark variant. This stage downloads both variants for
every team that appears in public.games, uploads them to a public Supabase
Storage bucket (`team-logos`), and points public.teams.logo / .logo_dark at
the storage URLs the dashboard reads.

Storage, not the database: the DB tier is ~434 of 500 MB; the bucket is a
separate 1 GB pool and images don't belong in Postgres anyway.

Upload auth: we hold the publishable (anon) key and the postgres password,
not the service key — so a temporary RLS policy lets anon insert into this
one bucket for the duration of the run, then is dropped. Public read needs
no policy on a public bucket.

Display-only data. Nothing here feeds the model.
"""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.load_supabase import dsn  # noqa: E402

ESPN = "https://a.espncdn.com/i/teamlogos/ncaa/500{dark}/{tid}.png"
CFBD = "https://cdn.collegefootballdata.com/logos{dark}/500/{tid}.png"
BUCKET = "team-logos"
MIN_PNG = 500          # bytes; ESPN serves tiny placeholders for some ids
TIMEOUT = 15


def fetch(url: str) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": "cfbpred/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = r.read()
        return data if len(data) >= MIN_PNG and data[:4] == b"\x89PNG" else None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None


def fetch_logo(tid: int, dark: bool) -> bytes | None:
    d = "-dark" if dark else ""
    return (fetch(ESPN.format(dark=d, tid=tid))
            or fetch(CFBD.format(dark=d, tid=tid)))


def upload(base_url: str, key: str, name: str, data: bytes) -> bool:
    req = urllib.request.Request(
        f"{base_url}/storage/v1/object/{BUCKET}/{name}",
        data=data, method="POST",
        headers={"Authorization": f"Bearer {key}", "apikey": key,
                 "Content-Type": "image/png", "x-upsert": "true",
                 "Cache-Control": "max-age=604800"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return 200 <= r.status < 300
    except urllib.error.HTTPError as e:
        print(f"    upload {name}: HTTP {e.code} {e.read()[:120]!r}")
        return False
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"    upload {name}: {e}")
        return False


def main() -> None:
    url = os.environ["SUPABASE_URL"].rstrip("/")
    if not url.startswith("http"):
        url = "https://" + url
    key = os.environ["SUPABASE_PUBLISHABLE_KEY"]

    with psycopg.connect(dsn(), autocommit=True) as cx, cx.cursor() as cur:
        # bucket + temporary anon write access (dropped in `finally`)
        cur.execute("""insert into storage.buckets (id, name, public)
                       values (%s, %s, true)
                       on conflict (id) do update set public = true""",
                    (BUCKET, BUCKET))
        # insert + update + select: the storage API's insert uses RETURNING,
        # which RLS treats as a read — without the select policy every upload
        # dies with "new row violates row-level security policy".
        cur.execute("drop policy if exists logo_write_tmp on storage.objects")
        cur.execute(f"""create policy logo_write_tmp on storage.objects
                        for insert to anon
                        with check (bucket_id = '{BUCKET}')""")
        cur.execute("drop policy if exists logo_update_tmp on storage.objects")
        cur.execute(f"""create policy logo_update_tmp on storage.objects
                        for update to anon
                        using (bucket_id = '{BUCKET}')
                        with check (bucket_id = '{BUCKET}')""")
        cur.execute("drop policy if exists logo_select_tmp on storage.objects")
        cur.execute(f"""create policy logo_select_tmp on storage.objects
                        for select to anon
                        using (bucket_id = '{BUCKET}')""")

        cur.execute("""select distinct t from (
                         select home_team_id t from public.games
                         union
                         select away_team_id from public.games) x
                       where t is not null order by t""")
        tids = [r[0] for r in cur.fetchall()]
        print(f"teams to fetch: {len(tids)}")

        cur.execute("""alter table public.teams
                       add column if not exists logo_dark text""")

        def work(tid: int) -> tuple[int, dict[str, str]]:
            """Fetch + upload both variants; DB writes stay on the main thread."""
            names: dict[str, str] = {}
            for dark in (False, True):
                data = fetch_logo(tid, dark)
                if data is None and dark:
                    data = fetch_logo(tid, False)       # dark falls back to light
                if data is None:
                    continue
                name = f"{tid}-dark.png" if dark else f"{tid}.png"
                if upload(url, key, name, data):
                    names["logo_dark" if dark else "logo"] = (
                        f"{url}/storage/v1/object/public/{BUCKET}/{name}")
            return tid, names

        ok = miss = 0
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                for i, (tid, names) in enumerate(pool.map(work, tids), 1):
                    if names:
                        sets = ", ".join(f"{c} = %s" for c in names)
                        cur.execute(
                            f"update public.teams set {sets} where team_id = %s",
                            (*names.values(), tid))
                        ok += 1
                    else:
                        miss += 1
                    if i % 50 == 0:
                        print(f"  {i}/{len(tids)}  ok={ok} miss={miss}")
        finally:
            cur.execute("drop policy if exists logo_write_tmp on storage.objects")
            cur.execute("drop policy if exists logo_update_tmp on storage.objects")
            cur.execute("drop policy if exists logo_select_tmp on storage.objects")

        cur.execute("""select count(*) from public.teams
                       where logo like %s""", (f"{url}/storage/%",))
        print(f"done: {ok} teams with logos, {miss} without; "
              f"{cur.fetchone()[0]} rows point at storage")


if __name__ == "__main__":
    main()
