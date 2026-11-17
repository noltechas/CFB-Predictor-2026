"""Mirror the pipeline's on-disk state to Supabase Storage for ephemeral CI runners.

The CFB jobs were written for one persistent Mac: archives append, refresh runs
build on the previous run, caches save CFBD quota. A GitHub Actions runner starts
empty, so each job restores the state it needs before running and saves what it
changed afterwards.

Storage layout (private bucket `cfb-state`):
    chunks/<sha256>                  content-addressed file chunks (<= 45 MB each)
    manifests/<set>/latest.json      path -> size, mtime, mode, chunk list; symlinks
    manifests/<set>/<stamp>.json     history (kept for `gc`)

Sets are declared in `ci/state_sets.json`: each has `roots` (paths relative to the
repo) and optional `exclude` globs. Restore makes the local tree under the roots
match the manifest exactly (extra files are removed, so a stale Actions cache can
never leak deleted files back); identical files are restored as hard links, as
the Mac pipeline stores them. Save uploads only chunks the store lacks.

    python -m cfbpred.ci_state restore <set> [<set> ...]
    python -m cfbpred.ci_state save <set> [<set> ...]
    python -m cfbpred.ci_state status <set>
    python -m cfbpred.ci_state gc --keep 10
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "ci" / "state_sets.json"
LOCAL = ROOT / ".ci_state"
BUCKET = "cfb-state"
CHUNK = 45 * 1024 * 1024
WORKERS = int(os.environ.get("CI_STATE_WORKERS", "4"))   # Storage 429s past ~8 concurrent writes


def _env():
    if "SUPABASE_URL" not in os.environ and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return os.environ["SUPABASE_URL"].rstrip("/"), os.environ["SUPABASE_SERVICE_ROLE_KEY"]


_session = None


def _http():
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=WORKERS, pool_maxsize=WORKERS * 2)
        _session.mount("https://", adapter)
        url, key = _env()
        _session.headers.update({"apikey": key, "Authorization": f"Bearer {key}"})
        _session.base = f"{url}/storage/v1"
    return _session


def _request(method, path, **kw):
    s = _http()
    for attempt in range(9):
        try:
            r = s.request(method, f"{s.base}/{path}", timeout=300, **kw)
        except requests.RequestException:
            if attempt == 8:
                raise
            time.sleep(min(60, 2 ** attempt))
            continue
        if r.status_code in (429, 500, 502, 503, 504) and attempt < 8:
            time.sleep(min(60, 2 ** attempt) + random.random())
            continue
        return r
    raise RuntimeError("unreachable")


def ensure_bucket():
    r = _request("GET", f"bucket/{BUCKET}")
    if r.status_code == 200:
        return
    r = _request("POST", "bucket", json={"id": BUCKET, "name": BUCKET, "public": False})
    if r.status_code not in (200, 201):
        raise RuntimeError(f"create bucket {BUCKET}: {r.status_code} {r.text[:200]}")


def _put(key, data: bytes, content_type="application/octet-stream", upsert=False):
    r = _request("POST", f"object/{BUCKET}/{key}", data=data,
                 headers={"Content-Type": content_type, "x-upsert": "true" if upsert else "false"})
    if r.status_code in (200, 201):
        return True
    if r.status_code in (400, 409) and ("exist" in r.text.lower() or "duplicate" in r.text.lower()):
        return False                      # content-addressed: already stored
    raise RuntimeError(f"upload {key}: {r.status_code} {r.text[:200]}")


def _exists(key) -> bool:
    return _request("HEAD", f"object/{BUCKET}/{key}").status_code == 200


def _get(key) -> bytes:
    r = _request("GET", f"object/{BUCKET}/{key}")
    if r.status_code != 200:
        raise RuntimeError(f"download {key}: {r.status_code} {r.text[:200]}")
    return r.content


# ------------------------------------------------------------------ sets & walking

def load_spec() -> dict:
    return json.loads(SPEC.read_text())["sets"]


def _excluded(rel: str, patterns) -> bool:
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(rel, p.rstrip("/") + "/*")
               for p in patterns)


def _json_strings(path: Path):
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    stack = [data]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
        elif isinstance(x, str):
            yield x


def resolve_roots(spec: dict) -> list:
    """Plain roots, plus {"runs_from": [...], "under": [...], "follow": [...]}: every run
    directory (an immediate child of an `under` dir) named by an absolute path inside
    the pointer files, followed transitively through each run's own pointer files.
    This is how "the latest daily run and whatever an availability run builds on"
    becomes a concrete list without hard-coding timestamps."""
    roots = []
    prefix = str(ROOT) + "/"
    for r in spec["roots"]:
        if isinstance(r, str):
            roots.append(r)
            continue
        seen, queue = set(), [ROOT / f for f in r["runs_from"]]
        while queue:
            for s in _json_strings(queue.pop()):
                if not s.startswith(prefix):
                    continue
                rel = s[len(prefix):]
                for u in r["under"]:
                    if rel.startswith(u + "/"):
                        run = f"{u}/{rel[len(u) + 1:].split('/')[0]}"
                        if run not in seen and (ROOT / run).is_dir():
                            seen.add(run)
                            roots.append(run)
                            queue += [ROOT / run / f for f in r.get("follow", [])]
    return sorted(set(roots))


def walk(spec: dict, roots=None):
    """(relpath, kind) for every file ('f') and symlink ('l') under the set's roots."""
    ex = spec.get("exclude", [])
    for root in (roots if roots is not None else resolve_roots(spec)):
        base = ROOT / root
        if base.is_symlink() or base.is_file():
            if not _excluded(root, ex):
                yield root, "l" if base.is_symlink() else "f"
            continue
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            rel_dir = os.path.relpath(dirpath, ROOT)
            keep = []
            for d in dirnames:
                rel = os.path.join(rel_dir, d)
                if os.path.islink(os.path.join(dirpath, d)):
                    if not _excluded(rel, ex):
                        yield rel, "l"
                elif not _excluded(rel, ex):
                    keep.append(d)
            dirnames[:] = keep
            for f in filenames:
                rel = os.path.join(rel_dir, f)
                if _excluded(rel, ex):
                    continue
                yield rel, "l" if os.path.islink(os.path.join(dirpath, f)) else "f"


def _hash_chunks(path: Path):
    out = []
    with path.open("rb") as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            out.append(hashlib.sha256(b).hexdigest())
    return out


def _local_manifest(name):
    p = LOCAL / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {"files": {}, "links": {}}


def _write_local(name, man):
    LOCAL.mkdir(exist_ok=True)
    (LOCAL / f"{name}.json").write_text(json.dumps(man))


def scan(name: str, spec: dict, prev: dict) -> dict:
    files, links = {}, {}
    todo = []
    roots = resolve_roots(spec)
    for rel, kind in walk(spec, roots):
        p = ROOT / rel
        if kind == "l":
            links[rel] = os.readlink(p)
            continue
        st = p.stat()
        old = prev["files"].get(rel)
        if old and old["size"] == st.st_size and old["mtime_ns"] == st.st_mtime_ns:
            files[rel] = old
        else:
            todo.append((rel, st))
    by_inode = {}
    for rel, e in files.items():
        st = (ROOT / rel).stat()
        by_inode[(st.st_dev, st.st_ino)] = e["chunks"]
    first, rest = {}, []
    for rel, st in todo:                     # one hash per inode
        k = (st.st_dev, st.st_ino)
        if k in by_inode or k in first:
            rest.append((rel, st))
        else:
            first[k] = (rel, st)
    def work(item):
        rel, st = item
        return rel, {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
                     "mode": st.st_mode & 0o777, "chunks": _hash_chunks(ROOT / rel)}
    with ThreadPoolExecutor(max(WORKERS, 8)) as ex:
        for rel, entry in ex.map(work, first.values()):
            files[rel] = entry
            st = (ROOT / rel).stat()
            by_inode[(st.st_dev, st.st_ino)] = entry["chunks"]
    for rel, st in rest:
        files[rel] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "mode": st.st_mode & 0o777,
                      "chunks": by_inode[(st.st_dev, st.st_ino)]}
    return {"set": name, "roots": roots, "files": files, "links": links}


# ------------------------------------------------------------------ save / restore

def save(name: str) -> dict:
    ensure_bucket()
    spec = load_spec()[name]
    t0 = time.time()
    prev_local = _local_manifest(name)
    try:
        remote = json.loads(_get(f"manifests/{name}/latest.json"))
    except RuntimeError:
        remote = {"files": {}, "links": {}}
    known = {c for e in remote["files"].values() for c in e["chunks"]}
    known |= {c for e in prev_local["files"].values() for c in e["chunks"]}
    man = scan(name, spec, prev_local if prev_local["files"] else remote)
    # chunk -> one (path, index) that holds it
    need = {}
    for rel, e in man["files"].items():
        for i, c in enumerate(e["chunks"]):
            if c not in known and c not in need:
                need[c] = (rel, i)
    changed = set()
    def up(item):
        c, (rel, i) = item
        if _exists(f"chunks/{c}"):           # stored by an earlier save or another set
            return 0
        with (ROOT / rel).open("rb") as fh:
            fh.seek(i * CHUNK)
            data = fh.read(CHUNK)
        if hashlib.sha256(data).hexdigest() != c:
            changed.add(rel)                  # written to while we saved: re-hash below
            return 0
        _put(f"chunks/{c}", data)
        return len(data)
    with ThreadPoolExecutor(WORKERS) as ex:
        sent = sum(ex.map(up, need.items()))
    for _ in range(3):                         # files another process rewrote mid-save
        if not changed:
            break
        redo, changed = sorted(changed), set()
        need = {}
        for rel in redo:
            st = (ROOT / rel).stat()
            man["files"][rel] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
                                 "mode": st.st_mode & 0o777, "chunks": _hash_chunks(ROOT / rel)}
            for i, c in enumerate(man["files"][rel]["chunks"]):
                need.setdefault(c, (rel, i))
        with ThreadPoolExecutor(WORKERS) as ex:
            sent += sum(ex.map(up, need.items()))
    if changed:
        raise RuntimeError(f"files kept changing while saving: {sorted(changed)[:5]}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    man["saved_at"] = stamp
    man["host"] = os.environ.get("GITHUB_RUN_ID") and f"actions:{os.environ['GITHUB_RUN_ID']}" or os.uname().nodename
    body = json.dumps(man).encode()
    _put(f"manifests/{name}/{stamp}.json", body, "application/json", upsert=True)
    _put(f"manifests/{name}/latest.json", body, "application/json", upsert=True)
    _write_local(name, man)
    total = sum(e["size"] for e in man["files"].values())
    print(f"[ci_state] saved {name}: {len(man['files'])} files ({total/1e9:.2f} GB), "
          f"checked {len(need)} chunks, uploaded {sent/1e6:.0f} MB in {time.time()-t0:.0f}s", flush=True)
    return man


def restore(name: str, force_local: bool = False) -> dict:
    if os.environ.get("CI") != "true" and not force_local:
        raise SystemExit("ci_state restore deletes local files that the saved state lacks; "
                         "it runs on CI runners only (pass --force-local to mirror onto this machine)")
    spec = load_spec()[name]
    t0 = time.time()
    man = json.loads(_get(f"manifests/{name}/latest.json"))
    prev = _local_manifest(name)
    roots = man.get("roots") or resolve_roots(spec)
    # run directories under a prune parent that the saved state no longer references
    pruned = 0
    for parent in spec.get("prune", []):
        base = ROOT / parent
        if not base.is_dir():
            continue
        for child in base.iterdir():
            rel = f"{parent}/{child.name}"
            if child.is_dir() and not child.is_symlink() and rel not in roots:
                import shutil
                shutil.rmtree(child)
                pruned += 1
    have = {}
    for rel, kind in walk(spec, roots):
        have[rel] = kind
    # remove anything the manifest does not list (exact mirror under the roots)
    removed = 0
    for rel, kind in have.items():
        if rel not in man["files"] and rel not in man["links"]:
            (ROOT / rel).unlink()
            removed += 1
    # decide which files need content
    todo, by_chunks = [], {}
    for rel, e in sorted(man["files"].items()):
        p = ROOT / rel
        key = tuple(e["chunks"])
        ok = False
        if p.exists() and not p.is_symlink():
            st = p.stat()
            if st.st_size == e["size"]:
                old = prev["files"].get(rel)
                if old and old["mtime_ns"] == st.st_mtime_ns and old["chunks"] == e["chunks"]:
                    ok = True
                elif st.st_mtime_ns == e["mtime_ns"] and _hash_chunks(p) == e["chunks"]:
                    ok = True
        if ok:
            by_chunks.setdefault(key, rel)
        else:
            todo.append(rel)
    # files whose content already exists elsewhere locally become hard links
    fetch, linked = [], 0
    for rel in todo:
        key = tuple(man["files"][rel]["chunks"])
        src = by_chunks.get(key)
        dst = ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src and key:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.link(ROOT / src, dst)
            linked += 1
        else:
            fetch.append(rel)            # duplicates among these are linked after download
    def down(rel):
        e = man["files"][rel]
        dst = ROOT / rel
        tmp = dst.with_name(dst.name + ".ci_part")
        with tmp.open("wb") as fh:
            for c in e["chunks"]:
                data = _get(f"chunks/{c}")
                if hashlib.sha256(data).hexdigest() != c:
                    raise RuntimeError(f"chunk {c} for {rel} is corrupt")
                fh.write(data)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        tmp.rename(dst)
        return e["size"]
    first = {}
    for rel in fetch:                       # identical files: download once, link the rest
        first.setdefault(tuple(man["files"][rel]["chunks"]), []).append(rel)
    uniq = [v[0] for v in first.values()]
    with ThreadPoolExecutor(WORKERS) as ex:
        got = sum(ex.map(down, uniq))
    for rels in first.values():
        for rel in rels[1:]:
            dst = ROOT / rel
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.link(ROOT / rels[0], dst)
            linked += 1
    for rel, e in man["files"].items():
        p = ROOT / rel
        os.chmod(p, e["mode"])
        os.utime(p, ns=(e["mtime_ns"], e["mtime_ns"]))
    for rel, target in man["links"].items():
        p = ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_symlink() or p.exists():
            if p.is_symlink() and os.readlink(p) == target:
                continue
            p.unlink()
        os.symlink(target, p)
    missing = [r for r in spec.get("required", []) if not (ROOT / r).exists()]
    if missing:
        raise SystemExit(f"[ci_state] {name}: required files missing after restore: {missing}")
    _write_local(name, man)
    total = sum(e["size"] for e in man["files"].values())
    if pruned:
        print(f"[ci_state] pruned {pruned} stale run directories", flush=True)
    print(f"[ci_state] restored {name} (saved {man.get('saved_at')} by {man.get('host')}): "
          f"{len(man['files'])} files ({total/1e9:.2f} GB); downloaded {len(uniq)} "
          f"({got/1e6:.0f} MB), linked {linked}, removed {removed}, in {time.time()-t0:.0f}s", flush=True)
    return man


def archive(name: str, paths) -> None:
    """Upload outputs nothing reads back (rollback archives, reports) without making
    them part of any restored state: content-addressed chunks + a dated listing."""
    ensure_bucket()
    spec = {"roots": [str(Path(p).resolve().relative_to(ROOT)) for p in paths], "exclude": ["*.ci_part"]}
    man = scan(name, spec, {"files": {}, "links": {}})
    need = {}
    for rel, e in man["files"].items():
        for i, c in enumerate(e["chunks"]):
            need.setdefault(c, (rel, i))
    def up(item):
        c, (rel, i) = item
        if _exists(f"chunks/{c}"):
            return False
        with (ROOT / rel).open("rb") as fh:
            fh.seek(i * CHUNK)
            return _put(f"chunks/{c}", fh.read(CHUNK))
    with ThreadPoolExecutor(WORKERS) as ex:
        new = sum(ex.map(up, need.items()))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _put(f"archives/{name}/{stamp}.json", json.dumps(man).encode(), "application/json", upsert=True)
    print(f"[ci_state] archived {name}: {len(man['files'])} files, {new} new chunks")


def status(name: str):
    spec = load_spec()[name]
    man = scan(name, spec, _local_manifest(name))
    total = sum(e["size"] for e in man["files"].values())
    uniq = {tuple(e["chunks"]): e["size"] for e in man["files"].values()}
    print(f"{name}: {len(man['files'])} files, {len(man['links'])} symlinks, "
          f"{total/1e9:.2f} GB ({sum(uniq.values())/1e9:.2f} GB unique)")


def gc(keep: int):
    """Delete chunks no longer referenced by the newest `keep` manifests of any set."""
    r = _request("POST", f"object/list/{BUCKET}", json={"prefix": "manifests/", "limit": 1000})
    sets = [o["name"] for o in r.json()]
    live = set()
    for s in sets:
        r = _request("POST", f"object/list/{BUCKET}", json={"prefix": f"manifests/{s}/", "limit": 1000,
                                                             "sortBy": {"column": "name", "order": "desc"}})
        names = [o["name"] for o in r.json() if o["name"] != "latest.json"][:keep]
        for n in names + ["latest.json"]:
            m = json.loads(_get(f"manifests/{s}/{n}"))
            live |= {c for e in m["files"].values() for c in e["chunks"]}
    r = _request("POST", f"object/list/{BUCKET}", json={"prefix": "archives/", "limit": 1000})
    for a in [o["name"] for o in r.json()]:
        rr = _request("POST", f"object/list/{BUCKET}", json={"prefix": f"archives/{a}/", "limit": 10000})
        for o in rr.json():
            m = json.loads(_get(f"archives/{a}/{o['name']}"))
            live |= {c for e in m["files"].values() for c in e["chunks"]}
    dead, offset = [], 0
    while True:
        r = _request("POST", f"object/list/{BUCKET}", json={"prefix": "chunks/", "limit": 1000, "offset": offset})
        objs = r.json()
        dead += [f"chunks/{o['name']}" for o in objs if o["name"] not in live]
        if len(objs) < 1000:
            break
        offset += 1000
    for i in range(0, len(dead), 500):
        _request("DELETE", f"object/{BUCKET}", json={"prefixes": dead[i:i + 500]})
    print(f"[ci_state] gc: {len(live)} live chunks, deleted {len(dead)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["save", "restore", "status", "gc", "archive"])
    ap.add_argument("sets", nargs="*", help="set names; for archive: NAME PATH [PATH ...]")
    ap.add_argument("--keep", type=int, default=10)
    ap.add_argument("--force-local", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "gc":
        return gc(a.keep)
    if a.cmd == "archive":
        return archive(a.sets[0], [p for p in a.sets[1:] if Path(p).exists()])
    for s in a.sets:
        if a.cmd == "restore":
            restore(s, a.force_local)
        else:
            {"save": save, "status": status}[a.cmd](s)


if __name__ == "__main__":
    sys.exit(main())
