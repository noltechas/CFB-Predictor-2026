"""Export the live half of the ensemble as raw float32 weights.

ONNX Runtime cannot load in Supabase's edge sandbox: 1.19 ships only threaded
wasm (no SharedArrayBuffer there) and 1.18's browser build breaks under Deno's
node-compat. The graph is small and entirely standard ops, so the forward pass
is reimplemented in TypeScript instead and fed these weights.

Layout per member (StateTower + Gate), all row-major float32:
    3 x [Linear(w,b) -> LayerNorm(w,b)]   body blocks, GELU between
    Linear margin(w,b), Linear total(w,b) heads
    gate scalars: w_time, w_lead, bias, w_sharp, floor

    python3 src/cfbpred/export_weights.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfbpred.config import N_MARGIN_BINS, N_REM_BINS, N_TOTAL_BINS  # noqa: E402
from cfbpred.ensemble import load_members  # noqa: E402
from cfbpred.train import STATE_FEATURES, build_tensors  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "supabase" / "model"


def main() -> None:
    dev = torch.device("cpu")
    print("loading feature store + ensemble...")
    d = build_tensors(dev)
    members = load_members(d, dev)
    if not all(m.delta_heads for m in members):
        raise SystemExit("expected a delta-head ensemble")

    blob: list[np.ndarray] = []
    offset = 0

    def put(t: torch.Tensor) -> dict:
        nonlocal offset
        a = t.detach().cpu().numpy().astype(np.float32).ravel()
        rec = {"off": offset, "shape": list(t.shape)}
        blob.append(a)
        offset += a.size
        return rec

    members_meta = []
    for m in members:
        body = [l for l in m.state.body if isinstance(l, (nn.Linear, nn.LayerNorm))]
        blocks = []
        for i in range(0, len(body), 2):
            lin, ln = body[i], body[i + 1]
            blocks.append({
                "lin_w": put(lin.weight), "lin_b": put(lin.bias),
                "ln_w": put(ln.weight), "ln_b": put(ln.bias),
                "ln_eps": float(ln.eps),
            })
        members_meta.append({
            "blocks": blocks,
            "margin_w": put(m.state.margin.weight), "margin_b": put(m.state.margin.bias),
            "total_w": put(m.state.total.weight), "total_b": put(m.state.total.bias),
            "gate": {
                "w_time": float(m.gate.w_time), "w_lead": float(m.gate.w_lead),
                "bias": float(m.gate.bias), "w_sharp": float(m.gate.w_sharp),
                "floor": float(m.gate.floor),
            },
        })

    flat = np.concatenate(blob)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "weights.bin").write_bytes(flat.tobytes())
    meta = {
        "model_version": "raw-dnn-delta-v22",
        "n_members": len(members),
        "n_state": len(STATE_FEATURES),
        "state_features": STATE_FEATURES,
        "n_margin_bins": N_MARGIN_BINS,
        "n_total_bins": N_TOTAL_BINS,
        "n_rem_bins": N_REM_BINS,
        "n_floats": int(flat.size),
        "members": members_meta,
    }
    (OUT / "weights.json").write_text(json.dumps(meta))
    print(f"wrote weights.bin ({flat.nbytes/1e6:.2f} MB, {flat.size} floats)")
    print(f"wrote weights.json ({len(members)} members, {len(STATE_FEATURES)} state features)")


if __name__ == "__main__":
    main()
