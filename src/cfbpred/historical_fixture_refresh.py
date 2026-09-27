"""Install the reviewed completed historical schedule overlay.

The overlay is deliberately pointer-gated: no reviewed pointer means no-op.
Only previously absent, scored 2001--2025 DI-involved fixtures are appended;
existing canonical rows are never rewritten.  Database publication is opt-in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POINTER = ROOT / "data/historical_schedule_reviewed_latest.json"
DEFAULT_CANONICAL = ROOT / "data/parquet/games.parquet"
DEFAULT_RAW_CFBD = ROOT / "data/parquet/cfbd/games.parquet"
DEFAULT_PLAYER_STATS = ROOT / "data/parquet/cfbd/player_game_stats.parquet"
DI = {"fbs", "fcs"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _norm(value: object, aliases: dict[str, str] | None = None) -> str:
    value = (aliases or {}).get(str(value), value)
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _fixture_key(row, aliases: dict[str, str] | None = None) -> tuple:
    stamp = pd.Timestamp(row.start_date)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    date = stamp.tz_convert(ZoneInfo("America/Chicago")).date().isoformat()
    return (int(row.season), date, _norm(row.home_team, aliases), _norm(row.away_team, aliases))


def reviewed_source(pointer: Path = DEFAULT_POINTER) -> tuple[Path | None, dict]:
    if not pointer.exists():
        return None, {"status": "no_reviewed_pointer", "added": 0}
    doc = json.loads(pointer.read_text())
    source = Path(doc["schedule"])
    if not source.is_absolute():
        source = ROOT / source
    if not source.is_file():
        raise FileNotFoundError(f"reviewed schedule missing: {source}")
    expected = str(doc.get("sha256", "")).lower()
    if not expected or sha256(source) != expected:
        raise ValueError("reviewed schedule SHA-256 is missing or does not match pointer")
    return source, doc


def _eligible(source: pd.DataFrame) -> pd.DataFrame:
    required = {"game_id", "season", "start_date", "home_team", "away_team",
                "home_points", "away_points", "home_division", "away_division"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"reviewed schedule missing columns: {sorted(missing)}")
    s = source.copy()
    s["game_id"] = pd.to_numeric(s.game_id, errors="raise").astype("int64")
    s["season"] = pd.to_numeric(s.season, errors="raise").astype("int64")
    s["home_points"] = pd.to_numeric(s.home_points, errors="coerce")
    s["away_points"] = pd.to_numeric(s.away_points, errors="coerce")
    s["_kick"] = pd.to_datetime(s.start_date, utc=True, errors="raise")
    s = s[s.season.between(2001, 2025) & s.game_id.gt(0)
          & s.home_points.notna() & s.away_points.notna()
          & (s.home_division.isin(DI) | s.away_division.isin(DI))].copy()
    if s.game_id.duplicated().any():
        raise ValueError("reviewed overlay contains duplicate game IDs")
    try:
        from cfbpred.pull_espn_roster import team_alias_map
        aliases = team_alias_map()
    except Exception:
        aliases = {}
    keys = s.apply(lambda r: _fixture_key(r, aliases), axis=1)
    if keys.duplicated().any():
        raise ValueError("reviewed overlay contains duplicate normalized fixtures")
    s["fixture_key"] = keys
    return s.drop(columns=["_kick"])


def _candidates(source: pd.DataFrame, canonical: pd.DataFrame) -> pd.DataFrame:
    s = _eligible(source)
    try:
        from cfbpred.pull_espn_roster import team_alias_map
        aliases = team_alias_map()
    except Exception:
        aliases = {}
    cids = set(pd.to_numeric(canonical.game_id, errors="coerce").dropna().astype("int64"))
    existing = canonical.copy()
    existing["fixture_key"] = existing.apply(lambda r: _fixture_key(r, aliases), axis=1)
    by_id = existing.set_index("game_id", drop=False)
    existing_keys = set(existing.fixture_key)
    additions = []
    for _, row in s.iterrows():
        gid = int(row.game_id)
        if gid in cids:
            old = by_id.loc[gid]
            if (_fixture_key(row, aliases) != old.fixture_key or
                    int(row.home_points) != int(old.home_points) or int(row.away_points) != int(old.away_points)):
                raise ValueError(f"reviewed overlay conflicts with canonical game ID {gid}")
            continue
        if row.fixture_key in existing_keys:
            raise ValueError("reviewed overlay conflicts with canonical normalized fixture")
        additions.append(row.to_dict())
    return pd.DataFrame(additions).drop(columns=["fixture_key"], errors="ignore")


def _enrich_provider_ids(rows: pd.DataFrame, raw_path: Path = DEFAULT_RAW_CFBD) -> pd.DataFrame:
    """Fill provider team IDs from raw CFBD rows, without changing raw facts."""
    if rows.empty or not raw_path.exists():
        return rows
    raw = pd.read_parquet(raw_path, columns=["id", "homeId", "awayId"])
    raw = raw.rename(columns={"id": "game_id", "homeId": "home_id", "awayId": "away_id"})
    raw = raw.drop_duplicates("game_id").set_index("game_id")
    out = rows.copy()
    out["game_id"] = pd.to_numeric(out.game_id, errors="coerce").astype("int64")
    for col in ("home_id", "away_id"):
        if col not in out:
            out[col] = pd.NA
        values = out.game_id.map(raw[col])
        out[col] = out[col].where(out[col].notna(), values)
    return out


def install_stat_aliases(pointer_doc: dict, *, stats_path: Path = DEFAULT_PLAYER_STATS,
                         backup_dir: Path | None = None) -> dict:
    """Restore reviewed raw-stat game-ID aliases, idempotently."""
    alias_value = pointer_doc.get("stat_aliases", pointer_doc.get("game_aliases"))
    if not alias_value:
        return {"status": "no_stat_alias_pointer", "remapped": 0}
    alias_path = Path(alias_value)
    if not alias_path.is_absolute():
        alias_path = ROOT / alias_path
    expected = str(pointer_doc.get("stat_aliases_sha256",
                                   pointer_doc.get("game_aliases_sha256", ""))).lower()
    if not expected or sha256(alias_path) != expected:
        raise ValueError("stat alias SHA-256 is missing or does not match pointer")
    if not stats_path.exists():
        return {"status": "stats_missing", "remapped": 0}
    from cfbpred.game_id_reconciliation import load_aliases, reconcile_frame
    result = reconcile_frame(pd.read_parquet(stats_path), load_aliases(alias_path))
    if result.audit["quarantined_rows"]:
        raise ValueError("verified stat alias reconciliation produced conflicts")
    if not result.audit["remapped_rows"] and not result.audit["exact_duplicate_rows_removed"]:
        return {"status": "already_current", "remapped": 0}
    if backup_dir is None:
        backup_dir = stats_path.parent / "historical_overlay_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = backup_dir / f"{stats_path.name}.{stamp}.bak"
    shutil.copy2(stats_path, backup)
    fd, tmp_name = tempfile.mkstemp(prefix=stats_path.stem + ".alias.", suffix=".parquet", dir=stats_path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        result.frame.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(stats_path)
    finally:
        tmp.unlink(missing_ok=True)
    return {"status": "installed", "remapped": result.audit["remapped_rows"],
            "duplicates_removed": result.audit["exact_duplicate_rows_removed"],
            "backup": str(backup), "stats_sha256": sha256(stats_path)}


def install(*, pointer: Path = DEFAULT_POINTER, canonical: Path = DEFAULT_CANONICAL,
            backup_dir: Path | None = None, publish: bool = False) -> dict:
    source, pointer_doc = reviewed_source(pointer)
    if source is None:
        return {**pointer_doc, "stat_aliases": install_stat_aliases(pointer_doc)}
    old = pd.read_parquet(canonical)
    source_df = pd.read_parquet(source)
    add = _candidates(source_df, old)
    add = _enrich_provider_ids(add)
    publish_rows = add
    if publish and add.empty:
        # A prior local install may have succeeded while DB publication failed.
        # Reuse the reviewed audit's additions when present; otherwise safely
        # retry the full eligible DI source with DO NOTHING plus readback.
        ids = pointer_doc.get("added_game_ids")
        eligible = _eligible(source_df)
        publish_rows = eligible[eligible.game_id.isin(ids)] if ids else eligible
        publish_rows = _enrich_provider_ids(publish_rows)
    if add.empty:
        if not publish:
            return {"status": "already_current", "added": 0, "source": str(source),
                    "stat_aliases": install_stat_aliases(pointer_doc)}
        # Do not rewrite or create another backup on a publication retry.
        return {"status": "db_retry", "added": 0, "source": str(source),
                "stat_aliases": install_stat_aliases(pointer_doc),
                "published": publish_db(publish_rows)}
    # Reviewed complete-history artifacts intentionally omit provider IDs that
    # are not part of the canonical completed-game layout.  Reindex fills
    # those columns with nulls while preserving the canonical schema/order.
    merged = pd.concat([old, add.reindex(columns=old.columns)], ignore_index=True, sort=False)
    merged = merged.sort_values(["season", "start_date", "game_id"], kind="stable")
    canonical.parent.mkdir(parents=True, exist_ok=True)
    if backup_dir is None:
        backup_dir = canonical.parent / "historical_overlay_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = backup_dir / f"games.parquet.{stamp}.bak"
    shutil.copy2(canonical, backup)
    fd, tmp_name = tempfile.mkstemp(prefix="games.overlay.", suffix=".parquet", dir=canonical.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        merged.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(canonical)
    finally:
        tmp.unlink(missing_ok=True)
    result = {"status": "installed", "added": int(len(add)), "backup": str(backup),
              "source": str(source), "canonical_sha256": sha256(canonical)}
    result["stat_aliases"] = install_stat_aliases(pointer_doc)
    if publish:
        result["published"] = publish_db(publish_rows)
    return result


def _database_fixture_identity(gid, season, kick, home, away, hp, ap, home_id, away_id, aliases):
    return (int(gid), int(season), pd.Timestamp(kick), _norm(home, aliases), _norm(away, aliases),
            int(hp), int(ap), int(home_id) if pd.notna(home_id) else None,
            int(away_id) if pd.notna(away_id) else None)


def publish_db(rows: pd.DataFrame) -> dict:
    """Insert only historical missing rows, preserving existing DB facts."""
    params = []
    try:
        from cfbpred.load_supabase import dsn
        import psycopg
    except ImportError as exc:
        raise RuntimeError("--publish requires psycopg") from exc
    for r in rows.itertuples():
        home_id = getattr(r, "home_team_id", getattr(r, "home_id", None))
        away_id = getattr(r, "away_team_id", getattr(r, "away_id", None))
        params.append((int(r.game_id), int(r.season), int(r.week) if hasattr(r, "week") and pd.notna(r.week) else None,
                       getattr(r, "season_type", None), r.start_date,
                       bool(r.neutral_site) if hasattr(r, "neutral_site") and pd.notna(r.neutral_site) else None,
                       bool(r.conference_game) if hasattr(r, "conference_game") and pd.notna(r.conference_game) else None,
                       int(r.venue_id) if hasattr(r, "venue_id") and pd.notna(r.venue_id) else None,
                       int(home_id) if pd.notna(home_id) else None, r.home_team, getattr(r, "home_conference", None), r.home_division,
                       int(r.home_points), int(away_id) if pd.notna(away_id) else None, r.away_team, getattr(r, "away_conference", None), r.away_division,
                       int(r.away_points)))
    sql = """insert into public.games
      (game_id,season,week,season_type,start_date,neutral_site,conference_game,venue_id,
       home_team_id,home_team,home_conference,home_classification,home_points,
       away_team_id,away_team,away_conference,away_classification,away_points)
      values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
      on conflict (game_id) do nothing"""
    with psycopg.connect(dsn(), autocommit=False) as cx, cx.cursor() as cur:
        cur.execute("select venue_id from public.venues")
        known_venues = {int(x[0]) for x in cur.fetchall() if x[0] is not None}
        params = [p[:7] + ((p[7] if p[7] in known_venues else None),) + p[8:] for p in params]
        cur.executemany(sql, params)
        ids = [p[0] for p in params]
        cur.execute("select game_id,season,start_date,home_team,away_team,home_points,away_points,home_team_id,away_team_id from public.games where game_id=any(%s)", (ids,))
        from cfbpred.pull_espn_roster import team_alias_map
        aliases = team_alias_map()
        # CFBD game 272582413 identifies both spellings as school 2706.
        aliases["West Virginia Tech"] = "WVU Tech"
        # Existing public records use current school spellings; the frozen
        # model corpus keeps historical names. Require both provider IDs and
        # reviewed alias identities, along with exact kickoff and final scores.
        got = {_database_fixture_identity(*r, aliases) for r in cur.fetchall()}
        expected = {_database_fixture_identity(p[0], p[1], p[4], p[9], p[14], p[12], p[17],
                                               p[8], p[13], aliases) for p in params}
        if not expected.issubset(got):
            raise RuntimeError("historical public.games identity/start/score readback failed")
        cx.commit()
    return {"rows": len(params), "readback": len(expected)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    ap.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    ap.add_argument("--backup-dir", type=Path)
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()
    print(json.dumps(install(pointer=args.pointer, canonical=args.canonical,
                             backup_dir=args.backup_dir, publish=args.publish), indent=2))


if __name__ == "__main__":
    main()
