"""Reconcile the cached complete CFBD game corpus into an isolated schedule.

The production schedule intentionally contains a smaller published subset.  This
module adds only completed 2001--2025 games involving an FBS/FCS side to a copy
of a pinned schedule.  Existing rows are byte-preserved at the value level;
ambiguous duplicate fixtures are quarantined instead of being guessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

import pandas as pd


def _hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _name_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _fixture_keys(frame: pd.DataFrame) -> pd.Series:
    """Same-day normalized home/away key; kickoff time is intentionally ignored."""
    kicks = pd.to_datetime(frame["start_date"], utc=True)
    return (frame["season"].astype("int64").astype(str) + "|" + kicks.dt.strftime("%Y-%m-%d") + "|"
            + frame["home_team"].map(_name_key) + "|" + frame["away_team"].map(_name_key))


# Public CFBD duplicate review.  The 638xx records have placeholder kickoff
# times and no venue; the paired IDs carry the same completed result with the
# usable published kickoff metadata.  The Edward Waters ID changed between
# records and is accepted only because this explicit review entry documents it.
REVIEWED_FIXTURE_RESOLUTIONS = {
    63826: {"winner": 242552754, "review": "same_2004_fixture_real_kickoff"},
    63829: {"winner": 242830326, "review": "same_2004_fixture_real_kickoff"},
    63832: {"winner": 242902458, "review": "same_2004_fixture_real_kickoff"},
    63834: {"winner": 242972229, "review": "same_2004_fixture_real_kickoff"},
    63836: {"winner": 243182229, "review": "same_2004_fixture_real_kickoff"},
    63838: {"winner": 243252226, "review": "same_2004_fixture_real_kickoff"},
    63839: {"winner": 243252229, "review": "same_2004_fixture_real_kickoff"},
    63840: {"winner": 243322226, "review": "same_2004_fixture_explicit_away_id_change", "allow_identity_conflict": True},
}


def _canonical_team_names(reference: pd.DataFrame) -> dict[tuple[int, int], str]:
    """Map season/team ID to the name already used by the canonical schedule."""
    values: dict[tuple[int, int], list[str]] = defaultdict(list)
    for side in ("home", "away"):
        ids, names = f"{side}_id", f"{side}_team"
        if ids not in reference or names not in reference:
            continue
        for row in reference[["season", ids, names]].itertuples(index=False):
            if pd.notna(row[1]) and pd.notna(row[2]):
                values[(int(row[0]), int(row[1]))].append(str(row[2]))
    return {key: Counter(names).most_common(1)[0][0] for key, names in values.items()}


def _raw_to_schedule(raw: pd.DataFrame, names: Mapping[tuple[int, int], str], aliases: Mapping[str, str]) -> pd.DataFrame:
    def team(row, side: str) -> str:
        key = (int(row.season), int(getattr(row, f"{side}Id"))) if pd.notna(getattr(row, f"{side}Id")) else None
        # Numeric-ID translation wins; aliases only fill IDs absent from the
        # canonical reference, preserving per-season historical spellings.
        return names.get(key, aliases.get(str(getattr(row, f"{side}Team")), str(getattr(row, f"{side}Team"))))

    out = pd.DataFrame({
        "game_id": raw.id.astype("int64"), "season": raw.season.astype("int64"),
        "week": pd.to_numeric(raw.week, errors="coerce"), "season_type": raw.seasonType,
        "start_date": raw.startDate, "neutral_site": raw.neutralSite,
        "conference_game": raw.conferenceGame, "venue_id": pd.to_numeric(raw.venueId, errors="coerce"),
        "home_team": [team(r, "home") for r in raw.itertuples()],
        "home_conference": raw.homeConference, "home_division": raw.homeClassification,
        "home_points": pd.to_numeric(raw.homePoints, errors="coerce"),
        "away_team": [team(r, "away") for r in raw.itertuples()],
        "away_conference": raw.awayConference, "away_division": raw.awayClassification,
        "away_points": pd.to_numeric(raw.awayPoints, errors="coerce"),
    })
    out["margin"] = out.home_points - out.away_points
    out["total"] = out.home_points + out.away_points
    out["kick"] = pd.to_datetime(out.start_date, utc=True)
    # Match features_raw: postseason numbering restarts at one, so it is
    # shifted out of the regular-season range. Spring types keep their actual
    # dates and are handled separately by the calendar logic.
    out["week_c"] = out.week.fillna(0).astype(int) + out.season_type.eq("postseason").astype(int) * 20
    return out


def reconcile(pinned_schedule: str | Path, raw_games: str | Path,
              output_dir: str | Path, *, reference_games: str | Path | None = None,
              aliases: Mapping[str, str] | None = None,
              reviewed_resolutions: Mapping[int, Mapping[str, object]] | None = None) -> dict:
    outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)
    pinned = pd.read_parquet(pinned_schedule)
    raw = pd.read_parquet(raw_games)
    reference = pd.read_parquet(reference_games or pinned_schedule)
    alias_map = dict(aliases or {})
    if aliases is None:
        try:
            from cfbpred.pull_espn_roster import team_alias_map
            alias_map = team_alias_map()
        except Exception:
            alias_map = {}
    scope = raw[(raw.season.between(2001, 2025)) & raw.completed.eq(True)].copy()
    scope = scope[scope.homeClassification.isin(["fbs", "fcs"]) | scope.awayClassification.isin(["fbs", "fcs"])].copy()
    pinned_ids = set(pd.to_numeric(pinned.game_id, errors="coerce").dropna().astype("int64"))
    candidates = scope[~scope.id.astype("int64").isin(pinned_ids)].copy()
    resolutions = reviewed_resolutions if reviewed_resolutions is not None else REVIEWED_FIXTURE_RESOLUTIONS
    resolved_rows = []
    resolved_losers = set()
    for loser, record in resolutions.items():
        loser, winner = int(loser), int(record["winner"])
        left, right = scope[scope.id.eq(loser)], scope[scope.id.eq(winner)]
        if len(left) == 0 and len(right) == 0:
            continue
        if len(left) != 1 or len(right) != 1:
            raise ValueError(f"reviewed resolution missing raw pair {loser}->{winner}")
        a, b = left.iloc[0], right.iloc[0]
        fields = ["season", "seasonType", "homePoints", "awayPoints"]
        same = all((str(a[f]) == str(b[f])) for f in fields)
        same = same and _name_key(a.homeTeam) == _name_key(b.homeTeam) and _name_key(a.awayTeam) == _name_key(b.awayTeam)
        same_date = pd.Timestamp(a.startDate).date() == pd.Timestamp(b.startDate).date()
        if not same or not same_date:
            raise ValueError(f"reviewed resolution evidence failed for {loser}->{winner}")
        if int(a.homeId) != int(b.homeId) or int(a.awayId) != int(b.awayId):
            if not record.get("allow_identity_conflict"):
                raise ValueError(f"unreviewed team-ID conflict for {loser}->{winner}")
        resolved_losers.add(loser)
        resolved_rows.append({"loser_game_id": loser, "winner_game_id": winner,
                              "review": record.get("review"),
                              "allow_identity_conflict": bool(record.get("allow_identity_conflict")),
                              "loser_start_date": str(a.startDate), "winner_start_date": str(b.startDate),
                              "home_team": str(a.homeTeam), "away_team": str(a.awayTeam),
                              "home_points": float(a.homePoints), "away_points": float(a.awayPoints)})
    candidates = candidates[~candidates.id.isin(resolved_losers)].copy()
    names = _canonical_team_names(reference)
    translated = _raw_to_schedule(candidates, names, alias_map)

    # A second provider ID can represent an existing fixture.  Compare raw
    # numeric IDs and exact kickoff; aliases alone are too weak for lower-level
    # schools.  Only a reference fixture whose ID is already pinned blocks an
    # addition; otherwise the raw row is the missing authoritative ID.
    ref = reference.copy()
    ref["_kick"] = pd.to_datetime(ref["start_date"], utc=True)
    raw_keys = candidates.assign(_kick=pd.to_datetime(candidates.startDate, utc=True))
    key_cols = ["season", "_kick", "homeId", "awayId"]
    ref_keys = ref[["season", "_kick", "home_id", "away_id", "game_id"]].rename(columns={"home_id": "homeId", "away_id": "awayId"})
    matches = raw_keys[key_cols + ["id"]].merge(ref_keys, on=key_cols, how="left")
    pinned_match = matches.game_id.isin(pinned_ids)
    duplicate_ids = set(matches.loc[pinned_match, "id"].astype("int64"))

    # Also detect equivalent fixtures when provider kickoff times differ.
    # Same-day candidate collisions are quarantined rather than guessed.
    translated["_fixture_key"] = _fixture_keys(translated)
    pinned_for_key = pinned.copy()
    pinned_for_key["_fixture_key"] = _fixture_keys(pinned_for_key)
    pinned_keys = set(pinned_for_key._fixture_key)
    candidate_counts = translated.groupby("_fixture_key").game_id.transform("size")
    key_duplicate = translated._fixture_key.isin(pinned_keys) | candidate_counts.gt(1)
    fixture_quarantine = translated.loc[key_duplicate, ["game_id", "_fixture_key"]].copy()
    duplicate_ids.update(fixture_quarantine.game_id.astype("int64"))
    translated = translated.loc[~key_duplicate].drop(columns="_fixture_key").copy()
    merged = pd.concat([pinned, translated[pinned.columns]], ignore_index=True, sort=False)
    merged = merged.sort_values(["season", "kick", "game_id"], kind="stable").reset_index(drop=True)
    output_schedule = outdir / "source_schedule.parquet"
    merged.to_parquet(output_schedule, index=False)
    if merged.game_id.duplicated().any():
        raise AssertionError("reconciled schedule contains duplicate game IDs")
    # Existing source rows are immutable evidence; assert all values survived.
    pinned_cmp = pinned.set_index("game_id")
    check = merged.set_index("game_id").loc[pinned_cmp.index, pinned_cmp.columns]
    if not check.reset_index().equals(pinned_cmp.reset_index()):
        raise AssertionError("pinned schedule rows changed during reconciliation")

    missing = scope[~scope.id.astype("int64").isin(pinned_ids)].copy()
    missing["class_combo"] = missing.homeClassification.fillna("NA") + "-" + missing.awayClassification.fillna("NA")
    d1 = missing[missing.homeClassification.isin(["fbs", "fcs"]) | missing.awayClassification.isin(["fbs", "fcs"])]
    reference_keys = set(names)
    fallback_home = sum((int(row.season), int(row.homeId)) not in reference_keys for row in d1.itertuples() if pd.notna(row.homeId))
    fallback_away = sum((int(row.season), int(row.awayId)) not in reference_keys for row in d1.itertuples() if pd.notna(row.awayId))
    stats_path = Path("data/parquet/cfbd/player_game_stats.parquet")
    stat_ids = set(pd.read_parquet(stats_path, columns=["game_id"]).game_id.astype("int64")) if stats_path.exists() else set()
    nebraska = d1[d1.homeTeam.eq("Nebraska") | d1.awayTeam.eq("Nebraska")]
    audit = {
        "pinned_rows": int(len(pinned)), "raw_completed_d1_rows": int(len(scope)),
        "missing_d1_rows": int(len(d1)), "missing_d1_added_rows": int(len(translated)),
        "duplicate_fixture_ids_quarantined": sorted(duplicate_ids),
        "duplicate_fixture_count": int(len(duplicate_ids)),
        "reviewed_resolution_count": int(len(resolved_rows)),
        "same_fixture_key_quarantine_count": int(len(fixture_quarantine)),
        "numeric_name_fallback_rows": {"home": int(fallback_home), "away": int(fallback_away)},
        "season_class_counts": {f"{int(y)}:{t}:{c}": int(n) for (y, t, c), n in missing.groupby(["season", "seasonType", "class_combo"]).size().items()},
        "missing_d1_with_player_stats": int(d1.id.isin(stat_ids).sum()),
        "nebraska_positive_schedule_rows": int(((merged.home_team.eq("Nebraska") | merged.away_team.eq("Nebraska")) & (merged.game_id > 0)).sum()),
        "nebraska_missing_bowls": int(((nebraska.season_type if "season_type" in nebraska else nebraska.seasonType).eq("postseason")).sum()),
        "name_translation_source": "reference_games numeric season/team ID, aliases fallback",
        "missing_is_not_health_inference": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_paths": {"pinned_schedule": str(Path(pinned_schedule).resolve()), "raw_games": str(Path(raw_games).resolve()), "reference_games": str(Path(reference_games or pinned_schedule).resolve())},
        "input_sha256": {"pinned_schedule": _hash(pinned_schedule), "raw_games": _hash(raw_games), "reference_games": _hash(reference_games or pinned_schedule)},
        "output_sha256": _hash(output_schedule),
        "output_rows": int(len(merged)),
    }
    (outdir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (outdir / "quarantine.json").write_text(json.dumps({
        "duplicate_fixture_ids": sorted(duplicate_ids),
        "same_fixture_keys": fixture_quarantine.to_dict("records"),
    }, indent=2) + "\n")
    (outdir / "source_game_aliases.json").write_text(json.dumps({
        str(row["loser_game_id"]): {"source_game_id": row["winner_game_id"], "reason": row["review"]}
        for row in resolved_rows
    }, indent=2, sort_keys=True) + "\n")
    (outdir / "reviewed_resolutions.json").write_text(json.dumps(resolved_rows, indent=2) + "\n")
    return audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinned", required=True); ap.add_argument("--raw", required=True)
    ap.add_argument("--reference", required=True); ap.add_argument("--output", required=True)
    args = ap.parse_args()
    print(json.dumps(reconcile(args.pinned, args.raw, args.output,
                               reference_games=args.reference), indent=2))


if __name__ == "__main__":
    main()
