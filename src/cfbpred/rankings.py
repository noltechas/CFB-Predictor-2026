"""Fieldcast Rankings: a readout of the active pregame ensemble.

    python3 src/cfbpred/rankings.py --season 2026
    python3 src/cfbpred/rankings.py --season 2026 --publish

Every eligible Division I team faces the same FBS reference pool. Matchups
are scored in both input orders on a neutral site, with seven days' rest,
250 miles' travel per side and no altitude change. Antisymmetrized expected
margins are averaged; FBS self-comparisons contribute exactly zero. Thus the
FBS reference mean is zero, and division filters never change rating values.
This output is never consumed by training or the game's prediction pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfbpred.config import CURRENT_SEASON, FEAT, MODELS, PQ
from cfbpred.ensemble import SEEDS
from cfbpred.model import MARGIN_CENTRES
from cfbpred.ranking_snapshot import build_snapshot
from cfbpred.raw_net import RawPregameTower
from cfbpred.train import prep_scalars

METHOD = "neutral-fbs-pool-v2"


def rating_readout(margins: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Common reference pool, no regression or extra fitted model."""
    if margins.ndim != 2 or margins.shape[0] != margins.shape[1]:
        raise ValueError("Expected a square matchup matrix")
    if not reference.any() or not np.isfinite(margins).all():
        raise ValueError("Missing reference opponents or non-finite predictions")
    neutral = (margins - margins.T) * 0.5
    np.fill_diagonal(neutral, 0.0)
    return neutral[:, reference].mean(axis=1)


def pair_frame(fields: pd.DataFrame, catalog: pd.DataFrame,
               h: np.ndarray, a: np.ndarray, week: int) -> pd.DataFrame:
    # Convert raw values before checkpoint normalization. Setting a normalized
    # value to zero would mean training-average travel/rest, not equal context.
    hd = fields.iloc[h].reset_index(drop=True).add_prefix("h_")
    ad = fields.iloc[a].reset_index(drop=True).add_prefix("a_")
    frame = pd.concat([hd, ad], axis=1)
    frame["neutral_site"] = True
    frame["conference_game"] = False
    frame["week_c"] = week
    frame["home_division"] = catalog.division.to_numpy()[h]
    frame["away_division"] = catalog.division.to_numpy()[a]
    for prefix in ("h_", "a_"):
        frame[prefix + "rest"] = 7.0
        frame[prefix + "travel"] = 250.0
        frame[prefix + "alt_delta"] = 0.0
        frame[prefix + "body_clock_shift"] = 0.0
    # Recent-form matchup columns combine the first team's offense with the
    # second team's defense.  They must be rebuilt for these synthetic pairs;
    # copying them from either team's next scheduled opponent is incorrect.
    if all(f"{side}_{unit}_r5_{stat}" in frame
           for side in ("h", "a") for unit in ("o", "d")
           for stat in ("pass_ypp", "rush_ypp", "sack_rate",
                        "expl20_rate", "pass_share")):
        from cfbpred.recent_form import matchup_features
        frame = matchup_features(frame, windows=(5,))
    prepared, _ = prep_scalars(frame)
    return prepared


def period_start(as_of: pd.Timestamp) -> str:
    local = as_of.tz_convert("America/Chicago").normalize()
    # Sunday starts the new football ranking period, after Saturday's slate.
    return (local - pd.Timedelta(days=(local.weekday() + 1) % 7)).date().isoformat()


def generate_raw(season: int, *, batch_size: int = 1024, preseason: bool = False) -> dict:
    as_of = pd.Timestamp.now(tz="UTC")
    snapshot = build_snapshot(season, as_of, preseason=preseason)
    as_of = pd.Timestamp(snapshot["as_of"])
    all_teams = snapshot["catalog"]
    mask = all_teams.eligible.to_numpy()
    cat = all_teams[mask].reset_index(drop=True)
    fields = snapshot["fields"].iloc[np.flatnonzero(mask)].reset_index(drop=True)
    hist = torch.tensor(snapshot["history"][mask])
    opp = torch.tensor(snapshot["opponents"][mask])
    vocab = snapshot["vocab"]
    team_ids = torch.tensor([vocab["team"][t] for t in cat.team], dtype=torch.long)
    conf_ids = torch.tensor([vocab["conf"].get(c, 0) for c in cat.conference], dtype=torch.long)
    gp = torch.tensor(np.clip(fields.gp.to_numpy(np.float32) / 12, 0, 1))
    reference = (cat.division == "fbs").to_numpy()
    if reference.sum() != (all_teams.division == "fbs").sum():
        raise ValueError("An FBS team is missing model coverage; reference pool would change")
    n = len(cat)
    # Only FBS-reference matchups and their reversals are needed.
    hi, ai = np.where((reference[:, None] | reference[None, :]) & ~np.eye(n, dtype=bool))
    week = min(snapshot["source_week"] + 1, 22)
    contract = pd.read_parquet(FEAT / "game_scalars.parquet").head(1)
    _, schema = prep_scalars(contract)
    members = []
    digester = hashlib.sha256()
    torch.set_num_threads(4)
    print(f"Scoring {len(hi):,} ordered matchups, {len(SEEDS)} members, {n} teams...", flush=True)
    for seed in SEEDS:
        path = MODELS / f"two_tower_raw_s{seed}.pt"
        digester.update(path.read_bytes())
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["scalar_cols"] != schema:
            raise ValueError("Feature schema differs from the frozen model contract")
        state = {k[4:]: v for k, v in ck["state_dict"].items() if k.startswith("pre.")}
        if state["team_emb.weight"].shape[0] != len(vocab["team"]) + 1:
            raise ValueError("Team vocabulary differs from the model")
        if state["conf_emb.weight"].shape[0] != len(vocab["conf"]) + 1:
            raise ValueError("Conference vocabulary differs from the model")
        net = RawPregameTower(
            state["team_emb.weight"].shape[0], state["conf_emb.weight"].shape[0],
            len(ck["scalar_cols"]), hidden=ck.get("pre_hidden", 160),
            s2d_idx_h=state["s2d_idx_h"], s2d_idx_a=state["s2d_idx_a"],
            use_s2d_gate=ck.get("s2d_gate", False))
        net.load_state_dict(state)
        net.eval()
        members.append((net, ck["scalar_cols"], ck["mu"], ck["sd"]))
    matrices = np.zeros((len(members), n, n), np.float32)
    with torch.inference_mode():
        for start in range(0, len(hi), batch_size):
            h, a = hi[start:start + batch_size], ai[start:start + batch_size]
            prepared = pair_frame(fields, cat, h, a, week)
            for member, (net, cols, mu, sd) in enumerate(members):
                values = prepared.reindex(columns=cols).to_numpy(np.float32)
                x = torch.tensor(np.nan_to_num((values - mu) / sd, nan=0.0,
                                               posinf=0.0, neginf=0.0))
                _, logits, _ = net(x, team_ids[h], team_ids[a], conf_ids[h], conf_ids[a],
                                   hist[h], opp[h], hist[a], opp[a], gp[h], gp[a])
                mean = (logits.softmax(-1) * MARGIN_CENTRES).sum(-1).numpy()
                matrices[member, h, a] = mean
            if start % (batch_size * 10) == 0:
                print(f"  {min(start + batch_size, len(hi)):,}/{len(hi):,}", flush=True)
    member_ratings = np.stack([rating_readout(m, reference) for m in matrices])
    ratings = member_ratings.mean(0)
    if abs(ratings[reference].mean()) > 1e-4:
        raise ValueError("FBS reference mean is not zero")
    order = sorted(range(n), key=lambda i: (-float(ratings[i]), cat.iloc[i].team))
    ranks = {i: rank + 1 for rank, i in enumerate(order)}
    div_ranks = {}
    for division in ("fbs", "fcs"):
        group = [i for i in order if cat.iloc[i].division == division]
        div_ranks.update({i: rank + 1 for rank, i in enumerate(group)})
    rows = []
    lookup = {t: i for i, t in enumerate(cat.team)}
    for r in all_teams.itertuples():
        i = lookup.get(r.team)
        f = snapshot["fields"].loc[r.team]
        rows.append({"team_id": int(r.team_id), "school": r.team,
                     "conference": None if pd.isna(r.conference) else r.conference,
                     "division": r.division, "rating": float(ratings[i]) if i is not None else None,
                     "rank": ranks.get(i), "division_rank": div_ranks.get(i),
                     "wins": r.wins, "losses": r.losses, "ties": r.ties,
                     "history_games": r.history_games,
                     "roster_available": bool(pd.notna(f.get("n_players"))),
                     "box_games": int(f.get("box_gp", 0)) if pd.notna(f.get("box_gp")) else 0,
                     "member_sd": float(member_ratings[:, i].std()) if i is not None else None,
                     "unranked_reason": None if i is not None else "Insufficient model history"})
    rows.sort(key=lambda r: (r["rank"] is None, r["rank"] or 0, r["school"]))
    raw_mean = matrices.mean(0)
    raw_asymmetry = np.abs(raw_mean[hi, ai] + raw_mean[ai, hi])
    reference_ids = sorted(cat.loc[reference, "team_id"].astype(int).tolist())
    ranking_week = snapshot["source_week"]
    snapshot_kind = ("reconstructed_preseason" if snapshot["reconstructed"] else "preseason") if ranking_week == 0 else "weekly"
    return {"season": season, "ranking_week": ranking_week, "snapshot_kind": snapshot_kind,
            "period_start": period_start(as_of), "as_of": as_of.isoformat(),
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "model_version": os.environ.get("MODEL_VERSION", "raw-dnn-delta-v22"),
            "model_fingerprint": digester.hexdigest(), "method_version": METHOD,
            "reference_hash": hashlib.sha256(json.dumps(reference_ids).encode()).hexdigest(),
            "reference_count": len(reference_ids), "n_members": len(members),
            "entries": rows, "diagnostics": {
                "feature_snapshot_at": snapshot["feature_time"],
                "latest_result_kickoff": snapshot["latest_result"], "week_context": week,
                "ordered_matchups": len(hi), "fbs_mean": float(ratings[reference].mean()),
                "raw_order_bias_mean": float(raw_asymmetry.mean()),
                "raw_order_bias_p95": float(np.quantile(raw_asymmetry, .95)),
                "ranked_teams": n, "unranked_teams": int((~mask).sum()),
                "status": "experimental", "backfilled": snapshot["reconstructed"]}}


def _current_roster_games(catalog: pd.DataFrame, rosters: pd.DataFrame,
                          game_ids: np.ndarray, as_of: pd.Timestamp, *, nearest: bool = False,
                          season: int = CURRENT_SEASON, allow_empty: bool = False) -> pd.DataFrame:
    """Choose one still-upcoming roster snapshot per team at a common cutoff."""
    available = rosters[(rosters.season == season)
                        & (pd.to_datetime(rosters.kick, utc=True) >= as_of)
                        & rosters.game_id.isin(set(map(int, game_ids)))].copy()
    # A team's season roster is repeated per matchup. Prefer a populated row,
    # then the nearest kickoff; no matchup outcome or opponent enters the
    # roster encoding.
    available = (available.sort_values(["team", "kick", "game_id"]) if nearest else
                 available.sort_values(["team", "players", "kick"], ascending=[True, False, True]))
    chosen = available.drop_duplicates("team").set_index("team")
    missing = set(catalog.team) - set(chosen.index)
    if missing:
        raise ValueError(f"No current player snapshot for ranking teams: {sorted(missing)[:5]}")
    chosen = chosen.loc[catalog.team].reset_index()
    if not allow_empty and chosen.loc[catalog.division.eq("fbs"), "players"].eq(0).any():
        raise ValueError("An FBS ranking team has an empty current roster")
    observed = chosen.last_history_kick.notna()
    if (pd.to_datetime(chosen.loc[observed, "last_history_kick"], utc=True) >= as_of).any():
        raise ValueError("Ranking roster history is not before the ranking cutoff")
    return chosen


def _preseason_roster_games(catalog: pd.DataFrame, rosters: pd.DataFrame,
                            game_ids: np.ndarray, as_of: pd.Timestamp, *,
                            season: int = CURRENT_SEASON, allow_empty: bool = False) -> pd.DataFrame:
    """Choose each team's first 2026 pregame snapshot for a Week 0 replay."""
    available = rosters[(rosters.season == season)
                        & (pd.to_datetime(rosters.kick, utc=True) >= as_of)
                        & rosters.game_id.isin(set(map(int, game_ids)))].copy()
    chosen = (available.sort_values(["team", "kick", "game_id"])
              .drop_duplicates("team").set_index("team"))
    missing = set(catalog.team) - set(chosen.index)
    if missing:
        raise ValueError(f"No preseason player snapshot for ranking teams: {sorted(missing)[:5]}")
    chosen = chosen.loc[catalog.team].reset_index()
    if not allow_empty and chosen.loc[catalog.division.eq("fbs"), "players"].eq(0).any():
        raise ValueError("An FBS preseason ranking team has an empty opening roster")
    observed = chosen.last_history_kick.notna()
    if (pd.to_datetime(chosen.loc[observed, "last_history_kick"], utc=True) >= as_of).any():
        raise ValueError("Preseason player history reaches or follows the season cutoff")
    return chosen


def _current_scalar_fields(catalog: pd.DataFrame, base: dict,
                           chosen: pd.DataFrame) -> pd.DataFrame:
    """Extract each team's complete current scalar state from the live snapshot.

    The isolated production refresh already contains the exact corrected
    scalar contract used to score current games, including this season's PBP,
    recent form and quarterback prior.  A future target supplies one side of
    that state for every team.  Game context and opponent-dependent matchup
    columns are replaced later by :func:`pair_frame`.
    """
    raw = pd.DataFrame(base["G_raw_scalars"], columns=base["scalar_cols"])
    game_position = {int(game_id): i for i, game_id in enumerate(base["G_gid"])}
    h_bases = {col[2:] for col in base["scalar_cols"] if col.startswith("h_")}
    a_bases = {col[2:] for col in base["scalar_cols"] if col.startswith("a_")}
    if h_bases != a_bases:
        raise ValueError("Active scalar contract is not side-symmetric")
    rows = []
    for row in chosen.itertuples():
        source = raw.iloc[game_position[int(row.game_id)]]
        prefix = "h_" if row.side == "home" else "a_"
        values = {name: source[prefix + name] for name in h_bases}
        rows.append(values)
    fields = pd.DataFrame(rows, index=catalog.team)
    if len(fields) != len(catalog) or fields.index.duplicated().any():
        raise ValueError("Current scalar snapshot does not cover ranking teams exactly once")
    return fields


def _preseason_scalar_fields(catalog: pd.DataFrame, base: dict,
                             chosen: pd.DataFrame,
                             snapshot_fields: pd.DataFrame) -> pd.DataFrame:
    """Restore the current model's scalar contract at the global Week 0 cutoff."""
    from cfbpred.ranking_snapshot import DYNAMIC_NAMES, DYNAMIC_PREFIXES
    fields = _current_scalar_fields(catalog, base, chosen)
    historical = snapshot_fields.reindex(catalog.team)
    for column in fields:
        dynamic = column in DYNAMIC_NAMES or column.startswith(DYNAMIC_PREFIXES)
        if not dynamic:
            continue
        if column in historical:
            fields[column] = historical[column].to_numpy()
        elif column.endswith(("_r3_games", "_r5_games")):
            fields[column] = 0.0
        else:
            fields[column] = np.nan
    return fields


def _restrict_player_store_to_cutoff(store, game_indices: np.ndarray,
                                     cutoff: pd.Timestamp, players: Path,
                                     sequences: Path) -> int:
    """Remove player events at or after a reconstructed global cutoff."""
    refs, valid = store.roster_refs(game_indices)
    packed_refs = refs[valid]
    player_rows = np.asarray(store.player_refs[packed_refs])
    catalog = pd.read_parquet(players / "player_seasons.parquet",
                              columns=["person_id"])
    people = catalog.person_id.to_numpy()[player_rows - 1]
    events = pd.read_parquet(sequences / "events.parquet",
                             columns=["event_id", "person_id", "kick"])
    before = events[pd.to_datetime(events.kick, utc=True) < cutoff]
    latest = (before.sort_values(["person_id", "kick", "event_id"])
              .drop_duplicates("person_id", keep="last")
              .set_index("person_id").event_id)
    tips = pd.Series(people).map(latest).fillna(0).to_numpy(np.int32)
    store.current_refs = np.asarray(store.current_refs).copy()
    store.current_refs[packed_refs] = 0
    store.history_tips = np.asarray(store.history_tips).copy()
    store.history_tips[packed_refs] = tips
    if (store.event_kick[tips[tips > 0]] >= cutoff.value).any():
        raise ValueError("Player event reached or followed the preseason cutoff")
    return int((tips > 0).sum())


def generate_player(season: int, *, batch_size: int = 1024,
                    preseason: bool = False, as_of=None, active_path=None,
                    input_source: Path | None = None, replay_cache: dict | None = None,
                    ranking_week: int | None = None) -> dict:
    """Neutral common-opponent ratings from the active complete-player model."""
    from cfbpred.player_sequence_production import (
        ACTIVE, _checkpoint_inputs, _make_model, active_bundle, bundle_scalar_extras, context_source, load_snapshot,
    )
    from cfbpred.train import game_batch

    active, bundle_path, bundle = active_bundle(active_path or ACTIVE)
    full_context = bundle.get("architecture") == "player_full_context_v1"
    if not full_context and (bundle.get("architecture"), bundle.get("variant")) != ("architecture_research_v1", "tabm16"):
        return generate_raw(season, batch_size=batch_size, preseason=preseason)
    if input_source is not None:
        paths = json.loads(Path(input_source).read_text())
        inputs, players, sequences = (Path(paths[k]) for k in ("inputs", "players", "sequences"))
        run = inputs.parent
    else:
        refresh = json.loads((MODELS / "player_sequence_production/refreshes/latest.json").read_text())
        run = Path(refresh["run_directory"])
        inputs, players, sequences = run / "inputs", run / "players", run / "sequences"
        if full_context and not (run / "context/source.json").exists():
            paths = json.loads(Path(bundle["input_source"]).read_text())
            inputs, players, sequences = (Path(paths[k]) for k in ("inputs", "players", "sequences"))
            run = inputs.parent
    source = input_source or context_source(inputs, bundle)
    cache_key = (str(inputs.resolve()), str(source), bundle["model_version"])
    if replay_cache is not None and replay_cache.get("key") == cache_key:
        base = replay_cache["base"]
    else:
        base = load_snapshot(inputs, players, sequences, source, bundle_scalar_extras(bundle))
        if replay_cache is not None:
            replay_cache.update(key=cache_key, base=base)

    requested_as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    snapshot = build_snapshot(season, requested_as_of, preseason=preseason,
                              reconstructed=full_context and as_of is not None, results_only=full_context,
                              feature_dir=inputs if full_context or input_source else FEAT,
                              allow_short_history=input_source is not None,
                              schedule_path=inputs / "snapshot_schedule.parquet" if full_context else None)
    provider_week = snapshot["source_week"]
    calendar_override = ranking_week is not None
    if ranking_week is not None:
        if input_source is None or ranking_week < 0:
            raise ValueError("Explicit weekly calendar requires reconstruction inputs")
        snapshot["source_week"] = ranking_week
    as_of = pd.Timestamp(snapshot["as_of"])
    all_teams = snapshot["catalog"]
    mask = all_teams.eligible.to_numpy()
    cat = all_teams[mask].reset_index(drop=True)
    history = torch.tensor(snapshot["history"][mask])
    current_opponents = np.asarray(snapshot["opponents"])[mask]
    reference = cat.division.eq("fbs").to_numpy()
    if reference.sum() != all_teams.division.eq("fbs").sum():
        raise ValueError("An FBS team is missing ranking coverage")

    rosters = pd.read_parquet(players / "game_rosters.parquet")
    if input_source is not None:
        # Player identity resolution canonicalizes old school spellings.
        # Select sides by the immutable game ID and side, then use the scalar
        # schedule's historical spelling for catalog joins and embeddings.
        identity_schedule = pd.read_parquet(inputs / "snapshot_schedule.parquet").set_index("game_id")
        rosters = rosters.copy()
        for side in ("home", "away"):
            side_mask = rosters.side.eq(side)
            rosters.loc[side_mask, "team"] = identity_schedule.loc[
                rosters.loc[side_mask, "game_id"], side + "_team"].to_numpy()
    chosen = (_preseason_roster_games(cat, rosters, base["G_gid"], as_of,
                                     season=season, allow_empty=input_source is not None)
              if preseason else
              _current_roster_games(cat, rosters, base["G_gid"], as_of, nearest=full_context,
                                   season=season, allow_empty=input_source is not None))
    fields = ((_preseason_scalar_fields(cat, base, chosen, snapshot["fields"])
               if preseason and not full_context else _current_scalar_fields(cat, base, chosen))
              .reset_index(drop=True))
    if full_context:
        from cfbpred.ranking_player_cutoff import restrict_context, verify_scalar_cutoff
        schedule = pd.read_parquet(inputs / "snapshot_schedule.parquet")
        verify_scalar_cutoff(chosen, schedule, as_of, fields, snapshot["fields"])
    game_position = {int(game_id): i for i, game_id in enumerate(base["G_gid"])}
    roster_game_indices = np.asarray([game_position[int(game_id)] for game_id in chosen.game_id], np.int64)
    roster_side = chosen.side.map({"home": 0, "away": 1}).to_numpy(np.int64)

    n = len(cat)
    hi, ai = np.where((reference[:, None] | reference[None, :]) & ~np.eye(n, dtype=bool))
    week = min(snapshot["source_week"] + 1, 22)
    prepared = pair_frame(fields, cat, hi, ai, week)
    gp = torch.tensor(np.clip(fields.gp.to_numpy(np.float32) / 12, 0, 1))
    matrices = np.zeros((len(bundle["members"]), n, n), np.float32)
    member_ratings, member_names = [], []
    preseason_player_histories = None
    digester = hashlib.sha256()
    torch.set_num_threads(2 if input_source else 4)
    print(f"Scoring {len(hi):,} ordered player-model matchups, "
          f"{len(bundle['members'])} outer seeds, {n} teams...", flush=True)

    for member_index, name in enumerate(bundle["members"]):
        path = bundle_path.parent / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != bundle["member_sha256"][name]:
            raise ValueError("Active ranking checkpoint changed")
        digester.update(path.read_bytes())
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        data = _checkpoint_inputs(base, checkpoint, players)
        if full_context:
            context_audit = restrict_context(data["player_store"], roster_game_indices, as_of, players, sequences)
            preseason_player_histories = context_audit["player_histories"]
        elif preseason:
            count = _restrict_player_store_to_cutoff(
                data["player_store"], roster_game_indices, as_of, players, sequences)
            if preseason_player_histories is None:
                preseason_player_histories = count
            elif count != preseason_player_histories:
                raise ValueError("Preseason player history coverage changed between members")
        # All teams use the same hypothetical kickoff, seven days after the
        # snapshot. Tips come from each team's next scheduled target and must
        # already be strictly before the current cutoff.
        data["player_store"].kicks = data["player_store"].kicks.copy()
        target_kick = (as_of + pd.Timedelta(days=7)).value
        data["player_store"].kicks[roster_game_indices] = target_kick
        model = _make_model(data, checkpoint["architecture"], checkpoint["variant"])
        model.pre.load_state_dict(checkpoint["state_dict"])
        model.eval()

        saved_team = checkpoint["vocab"]["team"]
        known_team = set(map(int, np.asarray(checkpoint["known_teams"]).ravel()))
        team_ids = torch.tensor([
            saved_team.get(team, 0) if saved_team.get(team, 0) in known_team else 0
            for team in cat.team], dtype=torch.long)
        if (team_ids[torch.tensor(reference)] == 0).any():
            raise ValueError("An FBS ranking team is unknown to the active checkpoint")
        saved_conf = checkpoint["vocab"]["conf"]
        known_conf = set(map(int, np.asarray(checkpoint["known_confs"]).ravel()))
        conf_ids = torch.tensor([
            saved_conf.get(conf, 0) if saved_conf.get(conf, 0) in known_conf else 0
            for conf in cat.conference], dtype=torch.long)
        current_vocab = snapshot["vocab"]["team"]
        opponent_map = np.zeros(max(current_vocab.values(), default=0) + 1, np.int64)
        for team, index in current_vocab.items():
            saved = saved_team.get(team, 0)
            opponent_map[index] = saved if saved in known_team else 0
        opponents = torch.tensor(opponent_map[current_opponents], dtype=torch.long)

        with torch.inference_mode():
            roster_parts = []
            for start in range(0, len(roster_game_indices), 16 if full_context else len(roster_game_indices)):
                ids = roster_game_indices[start:start + (16 if full_context else len(roster_game_indices))]
                roster_batch = game_batch(data, torch.tensor(ids), torch.device("cpu"))
                encoded, _ = model.pre.encoder.forward_batch(roster_batch)
                roster_parts.append(encoded)
            roster_pair = torch.cat(roster_parts)
            roster_latent = roster_pair[torch.arange(n), torch.tensor(roster_side)]
            raw = prepared.reindex(columns=checkpoint["scalar_cols"]).to_numpy(np.float32)
            scalar = torch.tensor(np.nan_to_num(
                (raw - np.asarray(checkpoint["mu"])) / np.asarray(checkpoint["sd"]),
                nan=0.0, posinf=0.0, neginf=0.0))
            for start in range(0, len(hi), batch_size):
                h, a = hi[start:start + batch_size], ai[start:start + batch_size]
                x = torch.cat([scalar[start:start + len(h)], roster_latent[h], roster_latent[a]], -1)
                _, logits, _ = model.pre.core(
                    x, team_ids[h], team_ids[a], conf_ids[h], conf_ids[a],
                    history[h], opponents[h], history[a], opponents[a], gp[h], gp[a])
                matrices[member_index, h, a] = (
                    logits.softmax(-1) * MARGIN_CENTRES).sum(-1).numpy()
                if member_index == 0 and start % (batch_size * 10) == 0:
                    print(f"  {min(start + batch_size, len(hi)):,}/{len(hi):,}", flush=True)
        member_ratings.append(rating_readout(matrices[member_index], reference))
        member_names.append({"checkpoint": name, "seed": int(checkpoint["seed"]),
                             "internal_tabm_members": 16})
        del model, data

    member_ratings = np.stack(member_ratings)
    ratings = member_ratings.mean(0)
    if abs(ratings[reference].mean()) > 1e-4:
        raise ValueError("FBS reference mean is not zero")
    order = sorted(range(n), key=lambda i: (-float(ratings[i]), cat.iloc[i].team))
    ranks = {i: rank + 1 for rank, i in enumerate(order)}
    division_ranks = {}
    for division in ("fbs", "fcs"):
        group = [i for i in order if cat.iloc[i].division == division]
        division_ranks.update({i: rank + 1 for rank, i in enumerate(group)})
    lookup = {team: i for i, team in enumerate(cat.team)}
    rows = []
    for row in all_teams.itertuples():
        i = lookup.get(row.team)
        field = fields.iloc[i] if full_context and i is not None else snapshot["fields"].loc[row.team]
        rows.append({"team_id": int(row.team_id), "school": row.team,
                     "conference": None if pd.isna(row.conference) else row.conference,
                     "division": row.division, "rating": float(ratings[i]) if i is not None else None,
                     "rank": ranks.get(i), "division_rank": division_ranks.get(i),
                     "wins": row.wins, "losses": row.losses, "ties": row.ties,
                     "history_games": row.history_games,
                     "roster_available": bool(pd.notna(field.get("n_players"))),
                     "box_games": int(field.get("box_gp", 0)) if pd.notna(field.get("box_gp")) else 0,
                     "member_sd": float(member_ratings[:, i].std()) if i is not None else None,
                     "unranked_reason": None if i is not None else "Insufficient model history"})
    rows.sort(key=lambda row: (row["rank"] is None, row["rank"] or 0, row["school"]))
    raw_mean = matrices.mean(0)
    raw_asymmetry = np.abs(raw_mean[hi, ai] + raw_mean[ai, hi])
    reference_ids = sorted(cat.loc[reference, "team_id"].astype(int).tolist())
    ranking_week = snapshot["source_week"]
    snapshot_kind = (("reconstructed_preseason" if snapshot["reconstructed"] else "preseason")
                     if ranking_week == 0 else "weekly")
    return {"season": season, "ranking_week": ranking_week, "snapshot_kind": snapshot_kind,
            "period_start": period_start(as_of), "as_of": as_of.isoformat(),
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "model_version": active["version"], "model_fingerprint": digester.hexdigest(),
            "method_version": "neutral-fbs-pool-full-context-v1" if full_context else "neutral-fbs-pool-player-tabm-v1",
            "reference_hash": hashlib.sha256(json.dumps(reference_ids).encode()).hexdigest(),
            "reference_count": len(reference_ids), "n_members": len(bundle["members"]),
            "entries": rows, "diagnostics": {
                "feature_snapshot_at": snapshot["feature_time"],
                "player_snapshot": str(run.resolve()), "latest_result_kickoff": snapshot["latest_result"],
                "week_context": week, "ordered_matchups": len(hi),
                "provider_week_label": provider_week,
                "calendar_week_override": calendar_override,
                "fbs_mean": float(ratings[reference].mean()),
                "raw_order_bias_mean": float(raw_asymmetry.mean()),
                "raw_order_bias_p95": float(np.quantile(raw_asymmetry, .95)),
                "ranked_teams": n, "unranked_teams": int((~mask).sum()),
                "outer_seeds": len(bundle["members"]), "internal_tabm_members_per_seed": 16,
                "effective_probability_members": len(bundle["members"]) * 16,
                "current_roster_snapshots_complete": bool(chosen.players.gt(0).all()),
                "empty_roster_teams": chosen.loc[chosen.players.eq(0), "team"].tolist(),
                "short_history_teams": cat.loc[cat.history_games.lt(4), "team"].tolist(),
                "weights_trained_through": int(checkpoint["train_max_season"]),
                "latest_model_reconstruction": bool(input_source is not None or snapshot["reconstructed"]),
                "historical_accuracy_claim": False,
                "status": "production",
                "preseason_player_histories": preseason_player_histories,
                "strict_global_player_cutoff": bool(preseason or full_context),
                "dated_context_cutoff": context_audit if full_context else None,
                "reconstruction_caveat": "Rebuilt from currently archived raw data; original as-published source vintages are not fully recoverable" if snapshot["reconstructed"] else None,
                "backfilled": snapshot["reconstructed"]}}


def generate(season: int, *, batch_size: int = 1024, preseason: bool = False) -> dict:
    """Dispatch rankings through the currently active pregame architecture."""
    active_path = MODELS / "player_sequence_production/active.json"
    if active_path.exists():
        active = json.loads(active_path.read_text())
        if (active.get("architecture"), active.get("variant")) in {("architecture_research_v1", "tabm16"), ("player_full_context_v1", "full_context")}:
            return generate_player(season, batch_size=batch_size, preseason=preseason)
    return generate_raw(season, batch_size=batch_size, preseason=preseason)


def publish(document: dict, *, replace_preseason: bool = False,
            replace_reviewed: bool = False) -> None:
    """Atomic weekly snapshot; clients can never see half a ranking table."""
    import psycopg
    from psycopg.types.json import Jsonb
    from cfbpred.load_supabase import dsn
    columns = ["season", "ranking_week", "snapshot_kind", "period_start", "as_of", "generated_at", "model_version",
               "model_fingerprint", "method_version", "reference_hash", "reference_count",
               "n_members", "entries", "diagnostics"]
    values = [Jsonb(document[c]) if c in ("entries", "diagnostics") else document[c] for c in columns]
    if replace_preseason and document["ranking_week"] != 0:
        raise ValueError("--replace-preseason may only publish a Week 0 snapshot")
    if replace_reviewed and not document.get("diagnostics", {}).get("strict_global_player_cutoff"):
        raise ValueError("Reviewed replacement requires a verified global player cutoff")
    # Explicit release reconstruction can adopt an earlier calendar cutoff
    # with corrected inputs. Ordinary scheduled updates remain monotonic.
    freshness_policy = "true" if replace_reviewed else "fieldcast_rankings.as_of <= excluded.as_of"
    preseason_policy = ("true" if replace_preseason else
                        "fieldcast_rankings.snapshot_kind = 'preseason' and excluded.snapshot_kind = 'preseason'")
    with psycopg.connect(dsn()) as cx, cx.cursor() as cur:
        cur.execute("set local statement_timeout = '30s'")
        cur.execute(f"""insert into public.fieldcast_rankings ({','.join(columns)})
            values ({','.join(['%s'] * len(columns))})
            on conflict (season, ranking_week) do update set
            {','.join(c + '=excluded.' + c for c in columns[2:])}
            where {freshness_policy}
              and (fieldcast_rankings.ranking_week > 0 or
                   {preseason_policy})""", values)
        print(f"{'Published' if cur.rowcount else 'Retained saved snapshot for'} {len(document['entries'])} teams, week {document['ranking_week']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=CURRENT_SEASON)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--preseason", action="store_true", help="Generate a preseason baseline, explicitly labeled reconstructed after kickoff")
    parser.add_argument("--publish-file", type=Path, help="Publish an already reviewed snapshot without recomputing")
    parser.add_argument("--replace-preseason", action="store_true",
                        help="Explicitly replace an archived Week 0 baseline")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.publish_file:
        document = json.loads(args.publish_file.read_text())
    else:
        document = generate(args.season, preseason=args.preseason)
        output = args.output or PQ / "rankings" / (f"{args.season}-preseason.json" if args.preseason else "latest.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(document, indent=2, allow_nan=False)
        temp = output.with_suffix(".tmp")
        temp.write_text(encoded)
        temp.replace(output)
        archive = output.parent / f"{document['season']}-week-{document['ranking_week']}.json"
        archive.write_text(encoded)
        print(f"Saved {output}; {document['diagnostics']}")
    if args.replace_preseason and not (args.publish or args.publish_file):
        parser.error("--replace-preseason requires --publish or --publish-file")
    if args.publish or args.publish_file:
        publish(document, replace_preseason=args.replace_preseason)


if __name__ == "__main__":
    main()
