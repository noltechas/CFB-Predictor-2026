"""Parity gate for the edge port.

Runs one real game's play sequence through both paths on IDENTICAL prepped
input, so ESPN drift cannot muddy the comparison:

  A. the shipped path — full 16-member PyTorch ensemble (live_scorer's code)
  B. the edge path    — pregame logits read back from Postgres + the ONNX graph

If these disagree, the edge function is not the scorer and must not ship.

    python3 src/cfbpred/verify_edge_parity.py [game_id ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfbpred.ensemble import load_members, readouts  # noqa: E402
from cfbpred.live_scorer import SUMMARY, fetch_play_rows  # noqa: E402
from cfbpred.load_supabase import dsn  # noqa: E402
from cfbpred.model import MARGIN_CENTRES, TOTAL_CENTRES  # noqa: E402
from cfbpred.train import (STATE_FEATURES, build_tensors,  # noqa: E402
                           game_batch, prep_play)

ONNX = Path(__file__).resolve().parents[2] / "supabase" / "model" / "live_state.onnx"
TOL = 2e-4          # win probability is reported to 5dp; this is far inside it


def scoreboard_row(s: requests.Session, gid: int) -> dict | None:
    """Rebuild the scoreboard-shaped dict fetch_play_rows expects. The live
    scoreboard drops finished games, so it is reconstructed from the summary
    header — same fields, sourced from the feed that still has them."""
    try:
        d = s.get(SUMMARY.format(gid=gid), timeout=20).json()
        c = d["header"]["competitions"][0]
        comp = {x["homeAway"]: x for x in c["competitors"]}
        st = c.get("status", {})
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None
    return {
        "game_id": gid, "state": st.get("type", {}).get("state", "post"),
        "period": int(st.get("period") or 4),
        "clock": st.get("displayClock") or "0:00",
        "home_points": int(comp["home"].get("score") or 0),
        "away_points": int(comp["away"].get("score") or 0),
        "home_id": str(comp["home"]["team"]["id"]),
        "away_id": str(comp["away"]["team"]["id"]),
        "home_abbr": comp["home"]["team"].get("abbreviation", "") or "",
        "away_abbr": comp["away"]["team"].get("abbreviation", "") or "",
        "down": None, "distance": None, "possession": None,
        "possession_text": "", "home_to": None, "away_to": None,
    }


def pick_games(n: int) -> list[int]:
    import psycopg
    with psycopg.connect(dsn()) as c, c.cursor() as cur:
        cur.execute(
            """select lc.game_id, count(*) c
                 from public.live_curve lc
                 join public.game_pregame_logits g using (game_id)
                group by 1 having count(*) > 20
                order by c desc limit %s""", (n,))
        return [int(r[0]) for r in cur.fetchall()]


def pregame_logits(gid: int, n_members: int):
    import psycopg
    with psycopg.connect(dsn()) as c, c.cursor() as cur:
        cur.execute("""select n_members, margin_logits, total_logits
                         from public.game_pregame_logits where game_id = %s""", (gid,))
        row = cur.fetchone()
    if not row:
        return None
    m, pm, pt = row
    if m != n_members:
        raise SystemExit(f"member count mismatch: table {m}, ensemble {n_members}")
    return (np.asarray(pm, np.float32).reshape(m, -1),
            np.asarray(pt, np.float32).reshape(m, -1))


def main() -> None:
    dev = torch.device("cpu")
    print("loading feature store + ensemble...")
    d = build_tensors(dev)
    models = load_members(d, dev)
    gid_pos = {int(g): i for i, g in enumerate(d["G_gid"])}

    import onnxruntime as ort
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])

    gids = [int(a) for a in sys.argv[1:] if a.isdigit()] or pick_games(3)
    print("games:", gids)
    s = requests.Session()
    s.headers.update({"Accept": "application/json"})

    worst = 0.0
    for gid in gids:
        if gid not in gid_pos:
            print(f"{gid}: not in feature store, skipping")
            continue
        r = scoreboard_row(s, gid)
        if r is None:
            print(f"{gid}: no summary feed, skipping")
            continue
        rows = fetch_play_rows(s, r)
        if not rows:
            print(f"{gid}: no play rows from ESPN, skipping")
            continue
        # live_scorer assigns play_index after shifting play_text back one row
        for i, pr in enumerate(rows):
            pr["play_index"] = i
        p = prep_play(pd.DataFrame(rows)).reset_index(drop=True)
        # lone synthetic rows have no prior plays: substitute college norms,
        # exactly as the scorer does before inference
        syn = p["synthetic"].to_numpy() == 1
        if syn.any():
            elapsed = 3600.0 - p["t_left"].to_numpy(float)
            p.loc[syn, "ig_plays"] = (elapsed[syn] / 26.0) / 120.0
            p.loc[syn, "ig_poss"] = (elapsed[syn] / 160.0) / 24.0
            p.loc[syn, "ig_pace"] = np.where(elapsed[syn] > 0, 26.0 / 40.0, 0.0)
        n = len(p)

        xs = torch.tensor(p[STATE_FEATURES].to_numpy(np.float32))
        fr = torch.tensor(p["t_left"].to_numpy(np.float32)) / 3600.0
        al = torch.tensor(np.abs(p["home_lead"].to_numpy(np.float32))) / 21.0
        ld = torch.tensor(p["home_lead"].to_numpy(np.float32))
        pz = torch.tensor(p["total_pts_before"].to_numpy(np.float32))
        gj = torch.tensor([gid_pos[int(g)] for g in p["game_id"]])

        # ---- A: the shipped path
        pm_s = pt_s = None
        pm_mem, pt_mem = [], []
        with torch.no_grad():
            batch = game_batch(d, gj, dev)
            for mdl in models:
                o = mdl(batch, xs, fr, al, ld, pz)
                # explicit accumulation: a walrus inside the ternary only binds
                # on the first pass, which silently summed member 0 sixteen times
                q = torch.softmax(o["margin"], -1)
                r_t = torch.softmax(o["total"], -1)
                pm_s = q if pm_s is None else pm_s + q
                pt_s = r_t if pt_s is None else pt_s + r_t
                _, gpm, gpt = mdl.pregame(batch)
                pm_mem.append(gpm[0])          # pregame is constant per game
                pt_mem.append(gpt[0])
        A = readouts(pm_s / len(models), pt_s / len(models))

        # ---- B: the edge path
        stored = pregame_logits(gid, len(models))
        if stored is None:
            print(f"{gid}: no published pregame logits, skipping")
            continue
        pm_pub, pt_pub = stored
        live_pm = torch.stack(pm_mem).numpy()
        live_pt = torch.stack(pt_mem).numpy()
        dl = max(float(np.abs(live_pm - pm_pub).max()), float(np.abs(live_pt - pt_pub).max()))

        feeds = {
            "state": xs.numpy(), "fr": fr.numpy(), "al": al.numpy(),
            "lead": ld.numpy(), "pts": pz.numpy(),
            "pm": np.repeat(pm_pub[None], n, 0), "pt": np.repeat(pt_pub[None], n, 0),
        }
        bm, bt = sess.run(None, feeds)
        B = readouts(torch.tensor(bm), torch.tensor(bt))

        dwp = float(np.abs(A["wp"] - B["wp"]).max())
        dem = float(np.abs(A["em"] - B["em"]).max())
        det = float(np.abs(A["et"] - B["et"]).max())
        worst = max(worst, dwp)
        print(f"{gid}: {n:4d} snaps | published-logit drift {dl:.2e} | "
              f"max |dwp| {dwp:.2e}  |dmargin| {dem:.2e}  |dtotal| {det:.2e}")

    print()
    if worst > TOL:
        raise SystemExit(f"PARITY FAIL — worst win-prob gap {worst:.2e} > {TOL:.0e}")
    print(f"PARITY OK — worst win-prob gap {worst:.2e} (tolerance {TOL:.0e})")


if __name__ == "__main__":
    main()
