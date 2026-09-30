"""Production registry, scoring, and local promotion for the player context model.

The promoted model owns pregame forecasts.  The existing raw-DNN state tower
continues to own in-game updates because the player tower was trained and
validated only for pregame inference.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from cfbpred.config import CURRENT_SEASON, MODELS, PQ, ROOT
from cfbpred.ensemble import median_readout, readouts
from cfbpred.model import MARGIN_CENTRES, TOTAL_CENTRES, scores_from
from cfbpred.player_experiment import load_inputs
from cfbpred.player_sequence_fast_inputs import FastSequenceStore
from cfbpred.player_sequence_train import predict
from cfbpred.pregame_experiments import save_json, sha


REGISTRY = MODELS / "player_sequence_production"
ACTIVE = REGISTRY / "active.json"
CANDIDATE = MODELS / "experiments/architecture_research_20260907_v1/current_2026_tabm16"
VERSION = "player-sequence-tabm16-v2"
ARCHITECTURE = "architecture_research_v1"
VARIANT = "tabm16"
FULL_CONTEXT_VARIANTS = ("full_context", "full_context_rosterhead", "full_context_v2", "full_context_v3",
                         "full_context_v4", "full_context_v4_r4", "full_context_v4_r8")
SUPPORTED = {
    ("player_sequence_v1", "context"),
    (ARCHITECTURE, VARIANT),
    *(("player_full_context_v1", v) for v in FULL_CONTEXT_VARIANTS),
}
PLAYER_SCHEMA_KEYS = ("version", "static_cols", "history_cols", "positions", "history_windows")
SEQUENCE_SCHEMA_KEYS = ("version", "window", "event_cols", "role_cols", "target_cols")


def _atomic_json(path: Path, value: dict) -> None:
    pending = path.with_name(path.name + ".pending")
    save_json(pending, value)
    pending.replace(path)


def _same_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha(source) != sha(destination):
            raise ValueError(f"Installed production artifact differs: {destination.name}")
        return
    shutil.copy2(source, destination)


def install(source: Path = CANDIDATE, registry: Path = REGISTRY) -> dict:
    """Install an already-audited candidate and atomically activate its pointer."""
    plan_path, bundle_path, audit_path = source / "plan.json", source / "bundle.json", source / "forecast_audit.json"
    plan, bundle, audit = (json.loads(p.read_text()) for p in (plan_path, bundle_path, audit_path))
    if not audit.get("passed") or not audit.get("no_2026_outcomes_used"):
        raise ValueError("Candidate has not passed the no-leakage forecast audit")
    identity = (bundle.get("architecture"), bundle.get("variant"))
    if identity != (ARCHITECTURE, VARIANT) or plan.get("variant") != VARIANT:
        raise ValueError("Only the development-selected TabM candidate can be installed")
    if bundle.get("model_version") != VERSION or plan.get("model_version") != VERSION:
        raise ValueError("Candidate model version differs")
    if sha(plan_path) != bundle["plan_sha256"]:
        raise ValueError("Candidate plan changed after training")
    for name, digest in bundle["member_sha256"].items():
        if sha(source / name) != digest:
            raise ValueError(f"Candidate checkpoint changed: {name}")

    destination = registry / VERSION
    destination.mkdir(parents=True, exist_ok=True)
    for name in ["plan.json", "bundle.json", "forecast_audit.json", *bundle["members"]]:
        _same_or_copy(source / name, destination / name)
    report_path = source.parent / "report.json"
    selection_path = source.parent / "selection.json"
    decision_path = source / "adoption_decision.json"
    for path in (report_path, selection_path, decision_path):
        if path.exists():
            _same_or_copy(path, destination / path.name)
    installed_bundle = destination / "bundle.json"
    active = {
        "version": VERSION,
        "scope": "production_pregame",
        "architecture": ARCHITECTURE,
        "variant": VARIANT,
        "forecast_year": int(bundle["forecast_year"]),
        "members": len(bundle["members"]),
        "bundle": str(installed_bundle.resolve()),
        "bundle_sha256": sha(installed_bundle),
        "source_candidate": str(source.resolve()),
        "source_candidate_bundle_sha256": sha(bundle_path),
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "historical_assessment_margin_mae": 12.500528602042952,
        "historical_closing_market_mae": 12.029632721202004,
        "live_model": "raw-dnn-delta-v22",
        "live_model_reason": "The promoted player-sequence checkpoint is pregame-only; the validated state tower continues after kickoff.",
    }
    registry.mkdir(parents=True, exist_ok=True)
    _atomic_json(registry / "active.json", active)
    return active


def active_bundle(active_path: Path = ACTIVE) -> tuple[dict, Path, dict]:
    active = json.loads(active_path.read_text())
    bundle_path = Path(active["bundle"])
    if active["scope"] != "production_pregame":
        raise ValueError("Unsupported active production model")
    if sha(bundle_path) != active["bundle_sha256"]:
        raise ValueError("Active production bundle changed")
    bundle = json.loads(bundle_path.read_text())
    identity = (bundle.get("architecture"), bundle.get("variant"))
    if identity not in SUPPORTED:
        raise ValueError("Active bundle contains an unsupported architecture")
    if active.get("architecture", identity[0]) != identity[0] or active.get("variant", identity[1]) != identity[1]:
        raise ValueError("Active pointer and bundle architecture differ")
    if bundle.get("model_version", active["version"]) != active["version"]:
        raise ValueError("Active pointer and bundle version differ")
    return active, bundle_path, bundle


def _make_model(data: dict, architecture: str, variant: str):
    if (architecture, variant) == ("player_sequence_v1", "context"):
        from cfbpred.player_sequence_net import make
    elif (architecture, variant) == (ARCHITECTURE, VARIANT):
        from cfbpred.architecture_research_net import make
    elif architecture == "player_full_context_v1" and variant in FULL_CONTEXT_VARIANTS:
        from cfbpred.player_full_history_net import make
    else:
        raise ValueError("Unsupported production architecture")
    return make(data, variant)


def _mapping(current: dict, saved: dict, known) -> np.ndarray:
    out = np.zeros(max(current.values(), default=0) + 1, np.int64)
    known = set(map(int, np.asarray(known).ravel()))
    for name, index in current.items():
        old = int(saved.get(name, 0))
        out[int(index)] = old if old in known else 0
    return out


def check_position_source(checkpoint: dict, manifest: dict) -> str:
    """Position groups drive the roster encoder's pooling. A checkpoint fitted on
    recruiting-only positions (63% of players UNK) scored measurably worse when a
    store built with roster fallback was swapped under it (2026 completed games:
    margin MAE 12.61 -> 12.72, WP log loss .3096 -> .3128), so a store may only be
    scored by a checkpoint trained on the same source. Stores and checkpoints that
    predate the field are the recruiting-only source."""
    # The fitted checkpoint embeds the whole player manifest it trained on, so
    # the source is recorded even for checkpoints built before the field existed.
    trained = checkpoint.get("player_schema", {}).get("position_source", "scouting")
    built = manifest.get("position_source", "scouting")
    if built != trained:
        raise ValueError(f"Production player store position source {built!r} differs from the checkpoint's {trained!r}")
    return trained


def _checkpoint_inputs(base: dict, checkpoint: dict, players: Path) -> dict:
    if (checkpoint.get("architecture"), checkpoint.get("variant")) not in SUPPORTED:
        raise ValueError("Unsupported production checkpoint")
    if not checkpoint.get("pregame_only") or checkpoint["train_max_season"] >= CURRENT_SEASON:
        raise ValueError("Production checkpoint is not a pre-2026 pregame fit")
    if checkpoint["scalar_cols"] != base["scalar_cols"]:
        raise ValueError("Production scalar schema changed")
    if any(checkpoint["player_schema"][k] != base["player_store"].manifest[k] for k in PLAYER_SCHEMA_KEYS):
        raise ValueError("Production player schema changed")
    check_position_source(checkpoint, base["player_store"].manifest)
    if any(checkpoint["sequence_schema"][k] != base["player_store"].sequence_manifest[k] for k in SEQUENCE_SCHEMA_KEYS):
        raise ValueError("Production sequence schema changed")
    if checkpoint["architecture"] == "player_full_context_v1":
        store = base["player_store"]
        if not getattr(store, "include_timeline", False):
            raise ValueError("Full-context model requires its dated player store")
        if checkpoint["context_cols"] != store.context_cols or checkpoint["full_history_schema"] != store.full_manifest["event_cols"]:
            raise ValueError("Production full-history/context schema changed")
    mu, sd = np.asarray(checkpoint["mu"]), np.asarray(checkpoint["sd"])
    if mu.shape != (len(base["scalar_cols"]),) or sd.shape != mu.shape or not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd <= 0).any():
        raise ValueError("Invalid production normalization")

    d = base.copy()
    d["mu"], d["sd"] = mu, sd
    d["G_scalars"] = torch.tensor(np.nan_to_num(
        (base["G_raw_scalars"] - mu) / sd, nan=0, posinf=0, neginf=0), dtype=torch.float32)
    team_map = _mapping(base["vocab"]["team"], checkpoint["vocab"]["team"], checkpoint["known_teams"])
    conf_map = _mapping(base["vocab"]["conf"], checkpoint["vocab"]["conf"], checkpoint["known_confs"])
    for field in ("G_hidx", "G_aidx", "G_opp"):
        d[field] = torch.tensor(team_map[base[field].numpy()], dtype=torch.long)
    for field in ("G_hconf", "G_aconf"):
        d[field] = torch.tensor(conf_map[base[field].numpy()], dtype=torch.long)
    d["vocab"] = checkpoint["vocab"]

    store = copy.copy(base["player_store"])
    sequence_vocab = json.loads((store.sequence_path / "team_vocab.json").read_text())["team"]
    # Full-career role features compare program identity before masking learned
    # embeddings. Preserve unseen program IDs for that equality calculation;
    # the encoder still masks their embeddings with known_teams.
    full_context = checkpoint["architecture"] == "player_full_context_v1"
    sequence_known = list(checkpoint["vocab"]["team"].values()) if full_context else checkpoint["known_sequence_teams"]
    sequence_map = _mapping(sequence_vocab, checkpoint["vocab"]["team"], sequence_known)
    store.event_team = sequence_map[store.event_team]
    store.event_opponent = sequence_map[store.event_opponent]
    catalog = pd.read_parquet(players / "player_seasons.parquet", columns=["team"])
    known_sequence = set(map(int, np.asarray(checkpoint["known_sequence_teams"]).ravel()))
    roster_team = catalog.team.map(checkpoint["vocab"]["team"]).fillna(0).to_numpy(np.int64)
    if not full_context:
        roster_team = np.where(np.isin(roster_team, list(known_sequence)), roster_team, 0)
    store.roster_team = np.r_[0, roster_team]
    store.known_teams = np.asarray(checkpoint["known_sequence_teams"])
    store.include_context, store.include_role = True, checkpoint["architecture"] == "player_full_context_v1"
    d["player_store"] = store
    return d


def context_source(inputs: Path, bundle: dict):
    if bundle.get("architecture") != "player_full_context_v1":
        return None
    refreshed = inputs.parent / "context/source.json"
    if refreshed.exists():
        return refreshed
    source = Path(bundle["input_source"])
    plan = source.parent / "plan.json"
    if sha(plan) != bundle["plan_sha256"] or sha(source) != json.loads(plan.read_text())["source_sha256"]:
        raise ValueError("Installed dated source lineage changed")
    return source


def context_lineage(base: dict, source: Path | None) -> dict:
    if source is None:
        return {}
    store = base["player_store"]
    return {"context_source_sha256": sha(source),
            "full_history_manifest_sha256": sha(store.full_path / "manifest.json"),
            "timeline_manifest_sha256": sha(store.timeline_path / "manifest.json"),
            "reviewed_facts_sha256": store.timeline_manifest["reviewed_sha256"]}


def bundle_scalar_extras(bundle: dict) -> tuple:
    """EXCLUDED_SCALARS entries a production bundle was trained with (normally none)."""
    return tuple(bundle.get("scalar_extras", ()))


def load_snapshot(inputs: Path, players: Path, sequences: Path, source: Path | None = None,
                  scalar_extras: tuple = ()) -> dict:
    # The scalar schema must be built exactly as the checkpoint saw it; _checkpoint_inputs
    # then refuses any mismatch, so a wrong setting can never score silently.
    from cfbpred import train as train_module
    previous = train_module.SCALAR_EXTRAS
    train_module.SCALAR_EXTRAS = tuple(scalar_extras)
    try:
        base = load_inputs(inputs, players)
    finally:
        train_module.SCALAR_EXTRAS = previous
    if source is None:
        base["player_store"] = FastSequenceStore(players, sequences, base["G_gid"])
    else:
        from cfbpred.player_timeline import TimelineStore
        paths = json.loads(source.read_text())
        for key, path in (("inputs", inputs), ("players", players), ("sequences", sequences)):
            if Path(paths[key]).resolve() != path.resolve():
                raise ValueError("Dated source does not match the player refresh")
        base["player_store"] = TimelineStore(players, sequences, Path(paths["full_history"]), Path(paths["timeline"]), base["G_gid"])
    manifest = base["player_store"].sequence_manifest
    if manifest["schedule_sha256"] != sha(inputs / "snapshot_schedule.parquet"):
        raise ValueError("Sequence store and schedule differ")
    if manifest["vocab_sha256"] != sha(inputs / "team_vocab.json"):
        raise ValueError("Sequence store and vocabulary differ")
    return base


def score_snapshot(inputs: Path, players: Path, sequences: Path, output: Path,
                   active_path: Path = ACTIVE, cutoff=None,
                   all_season: bool = False) -> pd.DataFrame:
    active, bundle_path, bundle = active_bundle(active_path)
    base = load_snapshot(inputs, players, sequences, context_source(inputs, bundle), bundle_scalar_extras(bundle))
    schedule = pd.read_parquet(inputs / "snapshot_schedule.parquet").set_index("game_id")
    cutoff = pd.Timestamp.now(tz="UTC") if cutoff is None else pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("Production cutoff needs a timezone")
    kicks = pd.to_datetime(schedule.loc[base["G_gid"], "kick"], utc=True)
    season = base["G_season"] == CURRENT_SEASON
    idx = np.where(season if all_season else season & (kicks > cutoff).to_numpy())[0]
    if not len(idx):
        raise ValueError("No production games in requested scope")

    sums = [None, None]
    member_checks = []
    for name in bundle["members"]:
        path = bundle_path.parent / name
        if sha(path) != bundle["member_sha256"][name]:
            raise ValueError(f"Production member changed: {name}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        d = _checkpoint_inputs(base, checkpoint, players)
        model = _make_model(d, checkpoint["architecture"], checkpoint["variant"])
        model.pre.load_state_dict(checkpoint["state_dict"])
        pm, pt = predict(model, d, idx, torch.device("cpu"))
        sums = [pm if sums[0] is None else sums[0] + pm,
                pt if sums[1] is None else sums[1] + pt]
        member_checks.append({"member": name, "seed": int(checkpoint["seed"]),
                              "epochs": int(checkpoint["epochs"]),
                              "latest_training_season": int(checkpoint["train_max_season"])})
        del model, d
    pm, pt = (value / len(bundle["members"]) for value in sums)
    for values in (pm, pt):
        if not torch.isfinite(values).all() or (values < 0).any() or not torch.allclose(values.sum(-1), torch.ones_like(values.sum(-1)), atol=1e-5):
            raise ValueError("Invalid production probability distribution")
    derived = readouts(pm, pt)
    home, away = scores_from(torch.tensor(derived["mm"]), torch.tensor(derived["mt"]))
    frame = pd.DataFrame({
        "game_id": base["G_gid"][idx], "season": base["G_season"][idx],
        # Point numbers are the distribution's (interpolated) median, the MAE-optimal readout and
        # the line at which the model's own P(cover) is 50%. The mean understated every backtest
        # season (2021-25 margin MAE 12.566 mean vs 12.535 median; total 12.883 vs 12.786).
        "home_win_prob": derived["wp"], "pred_margin": median_readout(pm, MARGIN_CENTRES),
        "pred_total": median_readout(pt, TOTAL_CENTRES),
        "pred_home_pts": home.numpy().astype(int), "pred_away_pts": away.numpy().astype(int),
        "split": "test", "margin_pmf": pm.numpy().tolist(), "total_pmf": pt.numpy().tolist(),
        "snapshot_at": cutoff, "model": active["version"], "model_version": active["version"],
    }).merge(schedule[["home_team", "away_team", "kick"]], left_on="game_id", right_index=True,
             validate="one_to_one")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False, compression="zstd")
    save_json(output.with_suffix(".json"), {
        "version": active["version"], "scope": active["scope"],
        "coverage": "full_season" if all_season else "future",
        "season_games" if all_season else "future_games": len(frame),
        "cutoff": cutoff.isoformat(), "forecast_sha256": sha(output),
        "active_manifest_sha256": sha(active_path), "bundle_sha256": sha(bundle_path),
        "input_manifest_sha256": sha(inputs / "player_data_provenance.json"),
        "player_manifest_sha256": sha(players / "manifest.json"),
        "sequence_manifest_sha256": sha(sequences / "manifest.json"),
        "members": member_checks, "no_2026_outcomes_used": True,
        **context_lineage(base, context_source(inputs, bundle)),
    })
    return frame


def audit_snapshot(inputs: Path, players: Path, sequences: Path, forecasts: Path,
                   active_path: Path = ACTIVE) -> dict:
    active, bundle_path, bundle = active_bundle(active_path)
    meta = json.loads(forecasts.with_suffix(".json").read_text())
    frame = pd.read_parquet(forecasts)
    if sha(forecasts) != meta["forecast_sha256"] or sha(active_path) != meta["active_manifest_sha256"]:
        raise ValueError("Production forecast lineage changed")
    base = load_snapshot(inputs, players, sequences, context_source(inputs, bundle), bundle_scalar_extras(bundle))
    for key, digest in context_lineage(base, context_source(inputs, bundle)).items():
        if meta.get(key) != digest:
            raise ValueError(f"Production context lineage changed: {key}")
    schedule = pd.read_parquet(inputs / "snapshot_schedule.parquet").set_index("game_id")
    cutoff = pd.Timestamp(meta["cutoff"])
    kicks = pd.to_datetime(schedule.loc[base["G_gid"], "kick"], utc=True)
    all_season = meta.get("coverage") == "full_season"
    season = base["G_season"] == CURRENT_SEASON
    expected = base["G_gid"][season if all_season else season & (kicks > cutoff).to_numpy()]
    if frame.game_id.duplicated().any() or set(frame.game_id) != set(expected):
        raise ValueError("Production game coverage differs")
    pm = torch.as_tensor(np.stack(frame.margin_pmf), dtype=torch.float32)
    pt = torch.as_tensor(np.stack(frame.total_pmf), dtype=torch.float32)
    if pm.shape != (len(frame), len(MARGIN_CENTRES)) or pt.shape != (len(frame), len(TOTAL_CENTRES)):
        raise ValueError("Production PMF dimensions differ")
    max_mass = max(float((pm.sum(-1) - 1).abs().max()), float((pt.sum(-1) - 1).abs().max()))
    if max_mass > 1e-5 or not torch.isfinite(pm).all() or not torch.isfinite(pt).all() or (pm < 0).any() or (pt < 0).any():
        raise ValueError("Production PMFs are invalid")
    derived = readouts(pm, pt)
    derived["pred_margin"], derived["pred_total"] = median_readout(pm, MARGIN_CENTRES), median_readout(pt, TOTAL_CENTRES)
    max_readout = max(float(np.max(np.abs(frame[col].to_numpy() - derived[key])))
                      for col, key in (("home_win_prob", "wp"), ("pred_margin", "pred_margin"), ("pred_total", "pred_total")))
    if max_readout > 1e-5:
        raise ValueError("Production readout differs from its PMF")
    roster = pd.read_parquet(players / "game_rosters.parquet").set_index(["game_id", "side"])
    target_index = pd.MultiIndex.from_product([expected, ["home", "away"]], names=["game_id", "side"])
    if not target_index.isin(roster.index).all():
        raise ValueError("Future roster reference missing")
    target = roster.loc[target_index].reset_index()
    target["division"] = [schedule.loc[row.game_id, row.side + "_division"] for row in target.itertuples()]
    fbs = target.division.eq("fbs")
    if target.loc[fbs, "players"].eq(0).any():
        raise ValueError("Future FBS roster is empty")
    observed = target.last_history_kick.notna()
    if (pd.to_datetime(target.loc[observed, "last_history_kick"], utc=True).to_numpy()
            >= pd.to_datetime(target.loc[observed, "kick"], utc=True).to_numpy()).any():
        raise ValueError("Future player history is not strictly pregame")
    for name in bundle["members"]:
        checkpoint = torch.load(bundle_path.parent / name, map_location="cpu", weights_only=False)
        if checkpoint["train_max_season"] >= CURRENT_SEASON:
            raise ValueError("2026 outcome entered production fitting")
    report = {
        "passed": True, "version": active["version"], "scope": active["scope"],
        "coverage": "full_season" if all_season else "future",
        "season_games" if all_season else "future_games": int(len(frame)),
        "team_sides": int(len(target)), "fbs_sides": int(fbs.sum()), "missing_fbs_rosters": 0,
        "strict_player_history_before_kickoff": True, "all_expected_future_games_scored": True,
        "no_2026_outcomes_used": True, "max_probability_mass_error": max_mass,
        "max_pmf_readout_error": max_readout, "forecast_sha256": sha(forecasts),
        "production_ready": True,
        **context_lineage(base, context_source(inputs, bundle)),
    }
    save_json(forecasts.with_name("production_audit.json"), report)
    return report


def promote_local(forecasts: Path, registry: Path = REGISTRY,
                  include_started: bool = False) -> dict:
    """Replace every still-upcoming local pregame row, preserving kicked games."""
    active, _, _ = active_bundle(registry / "active.json")
    frame = pd.read_parquet(forecasts)
    meta = json.loads(forecasts.with_suffix(".json").read_text())
    cutoff = pd.Timestamp(meta["cutoff"])
    games_path, pmf_path = PQ / "game_predictions.parquet", PQ / "game_pmfs.parquet"
    old_games, old_pmfs = pd.read_parquet(games_path), pd.read_parquet(pmf_path)
    candidate_ids = set(map(int, frame.game_id))
    schedule = pd.read_parquet(PQ / "schedule.parquet")
    kickoff = pd.to_datetime(schedule.start_date, format="mixed", utc=True)
    replacements = {int(k): int(v) for k, v in meta.get("retired_game_replacements", {}).items()}
    retired_ids = set(replacements) if include_started else set()
    if include_started:
        if meta.get("coverage") != "full_season":
            raise ValueError("Started games may be replaced only by an audited full-season forecast")
        prior_season = set(map(int, old_games.loc[old_games.season.eq(CURRENT_SEASON), "game_id"]))
        if (prior_season - retired_ids != candidate_ids
                or retired_ids & set(schedule.game_id)
                or not set(replacements.values()).issubset(candidate_ids)):
            raise ValueError("Full-season promotion coverage differs from existing production")
    else:
        prior_future = set(map(int, old_games.loc[
            old_games.season.eq(CURRENT_SEASON) & old_games.game_id.isin(schedule.loc[kickoff > cutoff, "game_id"]), "game_id"]))
        if not prior_future.issubset(candidate_ids):
            raise ValueError("Promoted model would drop an existing future production game")
    game_columns = list(old_games.columns)
    if "model_version" in game_columns and "model_version" not in frame:
        if meta.get("version") != active["version"]:
            raise ValueError("Forecast model version differs from active model")
        frame["model_version"] = meta["version"]
    promoted_games = frame[game_columns]
    promoted_pmfs = frame[["game_id", "season", "margin_pmf", "total_pmf"]]
    replaced_ids = candidate_ids | retired_ids
    new_games = pd.concat([old_games[~old_games.game_id.isin(replaced_ids)], promoted_games], ignore_index=True)
    new_pmfs = pd.concat([old_pmfs[~old_pmfs.game_id.isin(replaced_ids)], promoted_pmfs], ignore_index=True)
    if new_games.game_id.duplicated().any() or new_pmfs.game_id.duplicated().any():
        raise ValueError("Promotion produced duplicate game IDs")
    if set(new_games.loc[new_games.season.eq(CURRENT_SEASON), "game_id"]) != set(new_pmfs.loc[new_pmfs.season.eq(CURRENT_SEASON), "game_id"]):
        raise ValueError("Production 2026 predictions and PMFs differ in coverage")
    if not set(new_pmfs.game_id).issubset(set(new_games.game_id)):
        raise ValueError("Production PMFs have no corresponding prediction")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = registry / "promotions" / stamp
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(games_path, archive / games_path.name)
    shutil.copy2(pmf_path, archive / pmf_path.name)
    before_games, before_pmfs = sha(games_path), sha(pmf_path)
    for path, value in ((games_path, new_games), (pmf_path, new_pmfs)):
        pending = path.with_name(path.name + ".pending")
        value.to_parquet(pending, index=False, compression="zstd")
        pending.replace(path)
    report = {
        "passed": True, "version": active["version"], "scope": active["scope"],
        "promoted_at": datetime.now(timezone.utc).isoformat(), "forecast_cutoff": meta["cutoff"],
        "coverage": "full_season" if include_started else "future",
        "promoted_season_games" if include_started else "promoted_future_games": len(frame),
        "started_and_historical_rows_preserved": not include_started,
        "completed_and_future_current_season_rows_replaced": include_started,
        "previous_game_predictions_sha256": before_games, "previous_game_pmfs_sha256": before_pmfs,
        "game_predictions_sha256": sha(games_path), "game_pmfs_sha256": sha(pmf_path),
        "rollback_archive": str(archive.resolve()), "source_forecast_sha256": sha(forecasts),
        "external_publish_complete": False,
        "retired_game_replacements": replacements if include_started else {},
    }
    save_json(archive / "promotion.json", report)
    _atomic_json(registry / "latest_promotion.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["install", "score", "audit", "promote"])
    parser.add_argument("--source", type=Path, default=CANDIDATE)
    parser.add_argument("--active", type=Path, default=ACTIVE)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--players", type=Path)
    parser.add_argument("--sequences", type=Path)
    parser.add_argument("--forecasts", type=Path)
    parser.add_argument("--all-season", action="store_true")
    parser.add_argument("--include-started", action="store_true")
    args = parser.parse_args()
    if args.action == "install":
        print(json.dumps(install(args.source, args.active.parent), indent=2))
    elif args.action == "score":
        score_snapshot(args.inputs, args.players, args.sequences, args.forecasts, args.active,
                       all_season=args.all_season)
    elif args.action == "audit":
        print(json.dumps(audit_snapshot(args.inputs, args.players, args.sequences, args.forecasts, args.active), indent=2))
    else:
        print(json.dumps(promote_local(args.forecasts, args.active.parent, args.include_started), indent=2))
