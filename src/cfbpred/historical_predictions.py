"""Build an isolated retrospective prediction archive for pinned games."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cfbpred.ensemble import readouts
from cfbpred.model import scores_from
from cfbpred.player_sequence_production import (_checkpoint_inputs, _make_model,
    active_bundle, bundle_scalar_extras, load_snapshot)
from cfbpred.player_sequence_train import predict
from cfbpred.pregame_experiments import save_json, sha

DEFAULT_SOURCE = Path("models/experiments/public_archives_20260922_v3/source.json")
DEFAULT_STAGED = Path("models/player_sequence_production/staged_full_context.json")
DEFAULT_OUTPUT = Path("models/historical_reconstruction/predictions_20260922")


def positive_game_indices(game_ids) -> np.ndarray:
    """Return finite, strictly positive game rows (virtual IDs are excluded)."""
    values = np.asarray(game_ids).astype(np.float64, copy=False)
    return np.flatnonzero(np.isfinite(values) & (values > 0))


def reconstruct(source: Path = DEFAULT_SOURCE, staged: Path = DEFAULT_STAGED,
                output: Path = DEFAULT_OUTPUT, batch_size: int = 256,
                device: str = "cpu", threads: int = 2) -> dict:
    source, staged, output = Path(source), Path(staged), Path(output)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite reconstruction output: {output}")
    pinned = json.loads(source.read_text())
    pointer, bundle_path, bundle = active_bundle(staged)
    if pointer.get("version") != bundle.get("model_version"):
        raise ValueError("staged pointer and bundle versions differ")
    members = list(bundle.get("members", ()))
    if not members:
        raise ValueError("staged bundle has no model members")
    inputs, players, sequences = (Path(pinned[k]) for k in ("inputs", "players", "sequences"))
    base = load_snapshot(inputs, players, sequences, source, bundle_scalar_extras(bundle))
    gids, seasons = np.asarray(base["G_gid"]), np.asarray(base["G_season"])
    idx = positive_game_indices(gids)
    if not len(idx):
        raise ValueError("Pinned source has no positive game IDs")
    selected_ids = gids[idx].astype(np.int64, copy=False)
    if len(np.unique(selected_ids)) != len(selected_ids):
        raise ValueError("Pinned source contains duplicate positive game IDs")
    if threads > 0:
        torch.set_num_threads(threads)
    output.mkdir(parents=True, exist_ok=False)
    margin_members, total_members, member_checks = [], [], []
    for name in members:
        member_path = bundle_path.parent / name
        expected = bundle.get("member_sha256", {}).get(name)
        if expected is None or sha(member_path) != expected:
            raise ValueError(f"Staged member changed or is unpinned: {name}")
        checkpoint = torch.load(member_path, map_location="cpu", weights_only=False)
        d = _checkpoint_inputs(base, checkpoint, players)
        model = _make_model(d, checkpoint["architecture"], checkpoint["variant"])
        model.pre.load_state_dict(checkpoint["state_dict"])
        member_pm = member_pt = None
        with torch.no_grad():
            for start in range(0, len(idx), batch_size):
                take = idx[start:start + batch_size]
                pm, pt = predict(model, d, take, torch.device(device))
                if member_pm is None:
                    member_pm = np.empty((len(idx), pm.shape[-1]), dtype=np.float32)
                    member_pt = np.empty((len(idx), pt.shape[-1]), dtype=np.float32)
                member_pm[start:start + len(take)] = pm.numpy()
                member_pt[start:start + len(take)] = pt.numpy()
        margin_members.append(member_pm)
        total_members.append(member_pt)
        member_checks.append({"member": name, "seed": int(checkpoint.get("seed", -1)),
                              "train_max_season": int(checkpoint["train_max_season"])})
        del model, d, checkpoint
    margin = np.mean(np.stack(margin_members), axis=0, dtype=np.float64).astype(np.float32)
    total = np.mean(np.stack(total_members), axis=0, dtype=np.float64).astype(np.float32)
    tpm, tpt = torch.from_numpy(margin), torch.from_numpy(total)
    for values in (tpm, tpt):
        if not torch.isfinite(values).all() or (values < 0).any() or not torch.allclose(
                values.sum(-1), torch.ones(values.shape[0]), atol=1e-5):
            raise ValueError("Invalid reconstructed probability distribution")
    derived = readouts(tpm, tpt)
    home, away = scores_from(torch.from_numpy(derived["mm"]), torch.from_numpy(derived["mt"]))
    selected_seasons = seasons[idx].astype(np.int64)
    # Keep the production enum-compatible split values.  These are labels for
    # consumers, never claims of historical held-out evaluation: latest weights
    # were fit through 2025, while the live 2026 slate remains the test split.
    split = np.where(selected_seasons == 2026, "test", "train")
    pred = pd.DataFrame({"game_id": selected_ids, "season": selected_seasons,
        "split": split, "model_version": bundle["model_version"],
        "home_win_prob": derived["wp"], "pred_margin": derived["em"],
        "pred_total": derived["et"], "pred_home_pts": home.numpy().astype(int),
        "pred_away_pts": away.numpy().astype(int)})
    pmf = pd.DataFrame({"game_id": selected_ids, "season": selected_seasons,
                        "margin_pmf": margin.tolist(), "total_pmf": total.tolist()})
    pred_path, pmf_path = output / "game_predictions.parquet", output / "game_pmfs.parquet"
    pred.to_parquet(pred_path, index=False, compression="zstd")
    pmf.to_parquet(pmf_path, index=False, compression="zstd")
    metadata = {"passed": True, "scope": "retrospective_reconstruction",
        "coverage": "all_positive_fbs_fcs_involved_games", "game_count": len(pred),
        "seasons": [int(pred.season.min()), int(pred.season.max())],
        "model_version": bundle["model_version"], "members": member_checks,
        "staged_pointer": str(staged.resolve()), "staged_pointer_sha256": sha(staged),
        "bundle_sha256": sha(bundle_path), "source": str(source.resolve()),
        "source_sha256": sha(source), "inputs": str(inputs.resolve()),
        "players": str(players.resolve()), "sequences": str(sequences.resolve()),
        "timeline": pinned.get("timeline"), "full_history": pinned.get("full_history"),
        "fitted_through_season": max(x["train_max_season"] for x in member_checks),
        "retrospective_latest_weights": True, "historical_heldout_evaluation": False,
        "negative_virtual_game_ids_excluded": True, "historical_ui_label": "historical",
        "split_policy": "train_for_2001_2025_test_for_2026",
        "batch_size": batch_size,
        "threads": threads, "prediction_sha256": sha(pred_path), "pmf_sha256": sha(pmf_path),
        "created_at": datetime.now(timezone.utc).isoformat()}
    save_json(output / "metadata.json", metadata)
    return metadata


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--staged", type=Path, default=DEFAULT_STAGED)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    print(json.dumps(reconstruct(args.source, args.staged, args.output, args.batch_size,
                                 threads=args.threads), indent=2))


if __name__ == "__main__":
    main()
