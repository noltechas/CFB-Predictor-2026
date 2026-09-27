"""Export the live half of the ensemble to a single ONNX graph.

Why only half: RawTwoTower's state tower receives exactly two numbers from
the pregame side — expected pregame margin and expected pregame total (see
the comment in raw_net.forward about memorisation). The pregame tower itself
needs team embeddings, 270 scalars and opponent-history slots, and its output
is LOCKED before kickoff by design. So the pregame logits are computed once
per game here on the Mac and published to Postgres; the edge function only
ever runs what actually changes during a game.

Inputs mirror live_scorer's call site exactly:
    state  [n, 53]   STATE_FEATURES, in order
    fr     [n]       t_left / 3600
    al     [n]       |home_lead| / 21
    lead   [n]       home_lead, raw points
    pts    [n]       total_pts_before, raw points
    pm     [n, M, 115]  per-member pregame margin logits
    pt     [n, M, 103]  per-member pregame total logits
Outputs are the ensemble-averaged PMFs, i.e. exactly what live_scorer feeds
to `readouts`:
    margin_pmf [n, 115]
    total_pmf  [n, 103]

    python3 src/cfbpred/export_onnx.py            # export + verify
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfbpred.config import (MODELS, N_MARGIN_BINS, N_REM_BINS,  # noqa: E402
                            N_TOTAL_BINS)
from cfbpred.ensemble import load_members  # noqa: E402
from cfbpred.model import (MARGIN_CENTRES, REM_CENTRES,  # noqa: E402
                           TOTAL_CENTRES, margin_bin, rem_bin, shift_pmf,
                           total_bin)
from cfbpred.train import STATE_FEATURES, build_tensors  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "supabase" / "model"
EPS = 1e-9


class EdgeEnsemble(nn.Module):
    """The 16 state towers + gates + delta blend, averaged. A faithful copy of
    RawTwoTower.forward's delta branch with the pregame tower lifted out."""

    def __init__(self, members):
        super().__init__()
        self.states = nn.ModuleList([m.state for m in members])
        self.gates = nn.ModuleList([m.gate for m in members])
        self.n = len(members)
        if not all(m.delta_heads for m in members):
            raise SystemExit("expected delta-head members; refusing to export a mixed ensemble")

    def forward(self, state, fr, al, lead, pts, pm, pt):
        m_sum = None
        t_sum = None
        for k in range(self.n):
            pm_k, pt_k = pm[:, k], pt[:, k]
            # cond: the only pregame signal the state tower ever sees
            em = (torch.softmax(pm_k, -1) * MARGIN_CENTRES).sum(-1)
            et = (torch.softmax(pt_k, -1) * TOTAL_CENTRES).sum(-1)
            cond = torch.stack([em / 20.0, (et - 55.0) / 20.0], dim=-1)

            sm, st = self.states[k](state, cond)
            a = self.gates[k](fr, al).unsqueeze(-1)

            # pregame PMFs -> delta space, blend, back to final space
            pre_m = torch.log(shift_pmf(torch.softmax(pm_k, -1), MARGIN_CENTRES,
                                        -lead, margin_bin, N_MARGIN_BINS) + EPS)
            pre_t = torch.log(shift_pmf(torch.softmax(pt_k, -1), TOTAL_CENTRES,
                                        -pts, rem_bin, N_REM_BINS) + EPS)
            bm = a * pre_m + (1 - a) * sm
            bt = a * pre_t + (1 - a) * st
            out_m = torch.log(shift_pmf(torch.softmax(bm, -1), MARGIN_CENTRES,
                                        lead, margin_bin, N_MARGIN_BINS) + EPS)
            out_t = torch.log(shift_pmf(torch.softmax(bt, -1), REM_CENTRES,
                                        pts, total_bin, N_TOTAL_BINS) + EPS)

            # live_scorer softmaxes each member's output before averaging
            p_m, p_t = torch.softmax(out_m, -1), torch.softmax(out_t, -1)
            m_sum = p_m if m_sum is None else m_sum + p_m
            t_sum = p_t if t_sum is None else t_sum + p_t
        return m_sum / self.n, t_sum / self.n


def main() -> None:
    dev = torch.device("cpu")
    print("loading feature store + ensemble...")
    d = build_tensors(dev)
    members = load_members(d, dev)
    net = EdgeEnsemble(members).eval()
    n_state = len(STATE_FEATURES)
    print(f"members={net.n}  n_state={n_state}")

    torch.manual_seed(0)
    n = 5
    args = (
        torch.randn(n, n_state),
        torch.rand(n),
        torch.rand(n) * 2,
        torch.randint(-28, 29, (n,)).float(),
        torch.randint(0, 70, (n,)).float(),
        torch.randn(n, net.n, N_MARGIN_BINS),
        torch.randn(n, net.n, N_TOTAL_BINS),
    )
    with torch.no_grad():
        ref_m, ref_t = net(*args)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "live_state.onnx"
    dyn = {k: {0: "n"} for k in
           ["state", "fr", "al", "lead", "pts", "pm", "pt", "margin_pmf", "total_pmf"]}
    torch.onnx.export(
        net, args, str(path),
        input_names=["state", "fr", "al", "lead", "pts", "pm", "pt"],
        output_names=["margin_pmf", "total_pmf"],
        dynamic_axes=dyn, opset_version=17,
    )
    mb = path.stat().st_size / 1e6
    print(f"wrote {path}  ({mb:.1f} MB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = dict(zip(["state", "fr", "al", "lead", "pts", "pm", "pt"],
                     [a.numpy().astype(np.float32) for a in args]))
    got_m, got_t = sess.run(None, feeds)
    dm = float(np.abs(ref_m.numpy() - got_m).max())
    dt = float(np.abs(ref_t.numpy() - got_t).max())
    print(f"onnx vs torch  margin {dm:.3e}  total {dt:.3e}")
    if max(dm, dt) > 1e-5:
        raise SystemExit("ONNX export diverges from PyTorch — refusing to ship")

    meta = {
        "state_features": STATE_FEATURES,
        "n_members": net.n,
        "n_margin_bins": N_MARGIN_BINS,
        "n_total_bins": N_TOTAL_BINS,
        "margin_centres": MARGIN_CENTRES.tolist(),
        "total_centres": TOTAL_CENTRES.tolist(),
    }
    import json
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {OUT_DIR / 'meta.json'}  ({len(STATE_FEATURES)} state features)")
    print("PARITY OK")


if __name__ == "__main__":
    main()
