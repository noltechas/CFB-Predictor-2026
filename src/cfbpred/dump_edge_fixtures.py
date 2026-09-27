"""Freeze real ESPN payloads and the features Python derives from them.

The TypeScript port has to reproduce `derive_state` + `prep_play` exactly.
Testing that against a live feed would be testing ESPN too, so this captures
the raw summary JSON alongside the resulting 53-column state matrix. The TS
parity test replays the JSON offline and asserts on these numbers.

    python3 src/cfbpred/dump_edge_fixtures.py [game_id ...]
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from cfbpred.ensemble import load_members, readouts  # noqa: E402
from cfbpred.live_scorer import SUMMARY, fetch_play_rows  # noqa: E402
from cfbpred.train import (STATE_FEATURES, build_tensors,  # noqa: E402
                           game_batch, prep_play)
from cfbpred.verify_edge_parity import pick_games, scoreboard_row  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "supabase" / "fixtures"


def main() -> None:
    gids = [int(a) for a in sys.argv[1:] if a.isdigit()] or pick_games(4)
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    print("loading feature store + ensemble...")
    d = build_tensors(dev)
    models = load_members(d, dev)
    gid_pos = {int(g): i for i, g in enumerate(d["G_gid"])}
    s = requests.Session()
    s.headers.update({"Accept": "application/json"})
    index = []

    for gid in gids:
        raw = s.get(SUMMARY.format(gid=gid), timeout=20).json()
        r = scoreboard_row(s, gid)
        rows = fetch_play_rows(s, r)
        if not rows:
            print(f"{gid}: no plays, skipping")
            continue
        for i, pr in enumerate(rows):
            pr["play_index"] = i
        p = prep_play(pd.DataFrame(rows)).reset_index(drop=True)
        syn = p["synthetic"].to_numpy() == 1
        if syn.any():
            el = 3600.0 - p["t_left"].to_numpy(float)
            p.loc[syn, "ig_plays"] = (el[syn] / 26.0) / 120.0
            p.loc[syn, "ig_poss"] = (el[syn] / 160.0) / 24.0
            p.loc[syn, "ig_pace"] = np.where(el[syn] > 0, 26.0 / 40.0, 0.0)

        # the summary feed is large; gzip keeps the repo sane
        with gzip.open(OUT / f"{gid}.summary.json.gz", "wt") as fh:
            json.dump(raw, fh)
        expect = {
            "game_id": gid,
            # the exact input derive_state saw, so the TS port can be graded
            # on feature construction independently of ESPN parsing
            "rows": rows,
            "scoreboard_row": r,
            "n_rows": len(p),
            "state_features": STATE_FEATURES,
            # the model's five non-pregame inputs, plus the full state matrix
            "t_left": p["t_left"].astype(float).tolist(),
            "home_lead": p["home_lead"].astype(float).tolist(),
            "total_pts_before": p["total_pts_before"].astype(float).tolist(),
            "state": np.round(p[STATE_FEATURES].to_numpy(np.float64), 9).tolist(),
        }
        # the PyTorch answer for every snap, so the TS port is graded against
        # the shipped model rather than against itself
        if gid in gid_pos:
            xs = torch.tensor(p[STATE_FEATURES].to_numpy(np.float32))
            fr = torch.tensor(p["t_left"].to_numpy(np.float32)) / 3600.0
            al = torch.tensor(np.abs(p["home_lead"].to_numpy(np.float32))) / 21.0
            ld = torch.tensor(p["home_lead"].to_numpy(np.float32))
            pz = torch.tensor(p["total_pts_before"].to_numpy(np.float32))
            gj = torch.tensor([gid_pos[gid]] * len(p))
            pm_s = pt_s = None
            pm_mem, pt_mem = [], []
            with torch.no_grad():
                batch = game_batch(d, gj, dev)
                for mdl in models:
                    o = mdl(batch, xs, fr, al, ld, pz)
                    q = torch.softmax(o["margin"], -1)
                    r_t = torch.softmax(o["total"], -1)
                    pm_s = q if pm_s is None else pm_s + q
                    pt_s = r_t if pt_s is None else pt_s + r_t
                    _, gpm, gpt = mdl.pregame(batch)
                    pm_mem.append(gpm[0])
                    pt_mem.append(gpt[0])
            ro = readouts(pm_s / len(models), pt_s / len(models))
            expect["pregame_margin_logits"] = torch.stack(pm_mem).numpy().astype(float).tolist()
            expect["pregame_total_logits"] = torch.stack(pt_mem).numpy().astype(float).tolist()
            expect["wp"] = [float(v) for v in ro["wp"]]
            expect["em"] = [float(v) for v in ro["em"]]
            expect["et"] = [float(v) for v in ro["et"]]
        (OUT / f"{gid}.expected.json").write_text(json.dumps(expect))
        index.append({"game_id": gid, "n_rows": len(p)})
        print(f"{gid}: {len(p)} rows")

    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    total = sum(f.stat().st_size for f in OUT.iterdir()) / 1e6
    print(f"wrote {len(index)} fixtures to {OUT}  ({total:.1f} MB)")


if __name__ == "__main__":
    main()
