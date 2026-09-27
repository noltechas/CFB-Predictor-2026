"""Live in-game reads: replay every play of every game in progress.

Every ~interval seconds:
  1. ESPN scoreboard (FBS group 80 + FCS group 81) → which games are live or
     just finished, current score/clock/situation.
  2. For each live game in the feature store, ESPN's summary feed supplies
     the FULL play-by-play so far — period, clock, down, distance,
     yardsToEndzone (off_ytg exactly), offense, yardage, scores. The whole
     sequence goes through prep_play (so the cumulative in-game block is
     REAL — pace, plays, turnovers, momentum) and the frozen ensemble prices
     EVERY play. public.live_curve is rewritten per game per cycle: the same
     snap-by-snap curve completed games get, drawn live.
  3. The last state (from the scoreboard poll, appended after the final
     play) is the headline read in public.live_predictions. Games ESPN marks
     final get state='post' rows so the dashboard shows finals ahead of the
     weekly official ingest.

PURITY: only raw state is read — scores, clock, down, distance, yard line,
possession, yardage. ESPN's own win probability (lastPlay.probability) is
never parsed. Everything written here is OUR model's output, display-only.

Fallback: a live game whose drives feed is missing gets one synthetic row
from the scoreboard state (with pace/plays estimated at college norms —
elapsed/max(plays,1) explodes on a lone row otherwise).

Usage:
  python3 live_scorer.py             # one pass
  python3 live_scorer.py --loop 60   # poll until interrupted
"""
from __future__ import annotations

import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.ensemble import load_members, readouts  # noqa: E402
from cfbpred.load_supabase import dsn  # noqa: E402
from cfbpred.live_context import scoreboard_context  # noqa: E402
from psycopg.types.json import Jsonb
from cfbpred.train import (STATE_FEATURES, build_tensors,  # noqa: E402
                           game_batch, prep_play)

SCOREBOARD = ("https://site.api.espn.com/apis/site/v2/sports/football/"
              "college-football/scoreboard?groups={grp}&limit=400")
SUMMARY = ("https://site.api.espn.com/apis/site/v2/sports/football/"
           "college-football/summary?event={gid}")

PLAY_COLS = dict(yds=0, is_scrim=0, is_pass=0, is_sack=0,
                 is_to=0, is_pen=0, is_poss_chg=0,
                 h_pqb_play=0, a_pqb_play=0, h_prb_play=0, a_prb_play=0,
                 h_pqb_known=0, a_pqb_known=0)


def clock_seconds(display: str) -> float:
    m = re.match(r"^(\d+):(\d+)$", display or "")
    return int(m.group(1)) * 60 + int(m.group(2)) if m else 0.0


def t_left_of(period: int, clock: str) -> float:
    return (max(0, 4 - period) * 900 + clock_seconds(clock)
            if period <= 4 else 0.0)


def _period_range(p: int) -> tuple[float, float]:
    """A period's slice of regulation time; every overtime period is 0."""
    if p >= 5:
        return 0.0, 0.0
    lo = max(0, 4 - p) * 900.0
    return lo, lo + 900.0


def _quarter_of(t: float, own: int) -> int:
    """The regulation quarter holding t; a boundary goes to the nearest."""
    best = -1
    for q in (1, 2, 3, 4):
        lo, hi = _period_range(q)
        if lo <= t <= hi and (best < 0 or abs(q - own) < abs(best - own)):
            best = q
    return own if best < 0 else best


def clean_clock(rows: list[dict], now_last: bool = False) -> list[dict]:
    """Turn ESPN's per-play clock stamps into a game-time axis, in place.

    Mirrors cleanClock in supabase/functions/_shared/espn.ts EXACTLY (the
    edge scorer is graded against this). Play ORDER is trustworthy; stamps
    are not: drives share their start clock, FCS feeds stamp backward
    mid-backfill, and single rows arrive wildly low (a quarter's first play
    at 0:00, an "End of OT" row inside Q2). A running minimum let any one of
    those drag every later play down to it — a whole quarter piled onto one
    point and the live point quarters ahead of the clock (Nebraska @
    Michigan State, 2026-09-26). Anchors are instead the longest
    non-increasing run of stamps (ties keep higher stamps); every other row
    is placed between its neighbouring anchors, batch-stamped runs spread
    toward the next distinct stamp, and no row leaves its own quarter unless
    its period is itself impossible. `now_last` marks the final row as the
    scoreboard's observed state: it votes like a play, and it is a floor for
    the plays before it, not a spread target.
    """
    n = len(rows)
    if not n:
        return rows
    raw = [float(r["t_left"]) for r in rows]
    ln, prev = [1] * n, [-1] * n
    for i in range(n):
        for j in range(i):
            if raw[j] >= raw[i] and (ln[j] + 1 > ln[i] or (
                    ln[j] + 1 == ln[i] and raw[j] > raw[prev[i]])):
                ln[i], prev[i] = ln[j] + 1, j
    end = 0
    for i in range(1, n):
        if ln[i] > ln[end] or (ln[i] == ln[end] and raw[i] > raw[end]):
            end = i
    anchor = [False] * n
    i = end
    while i >= 0:
        anchor[i] = True
        i = prev[i]
    A = [i for i in range(n) if anchor[i]]
    heads = [a for k, a in enumerate(A) if k == 0 or raw[a] < raw[A[k - 1]]]
    # [start, end) of each group, the stamp it starts from, the next distinct
    segs = []
    if A[0] > 0:
        segs.append((0, A[0], 3600.0, raw[A[0]]))
    for h, a in enumerate(heads):
        b = heads[h + 1] if h + 1 < len(heads) else n
        segs.append((a, b, raw[a], raw[b] if b < n else None))
    t = raw[:]
    now_free = now_last and not anchor[n - 1]
    for a, b, top, nxt in segs:
        if nxt is None and now_free:
            b -= 1                    # a trailing scoreboard row goes last
        m = b - a
        if m <= 0:
            continue
        if nxt is not None and now_last and b == n - 1:
            bottom, floor = max(top - 25.0 * m, nxt), nxt
        elif nxt is not None:
            bottom, floor = nxt, nxt
        elif now_last and anchor[n - 1]:
            bottom, floor = top, top
        else:
            bottom, floor = max(top - 25.0 * m, 0.0), 0.0
        span = max(top - bottom, 0.0)
        for k in range(m):
            pos = a + k
            v = top - span * k / m
            lo_p, hi_p = _period_range(rows[pos]["period"])
            lo, hi = max(lo_p, floor), min(hi_p, top)
            t[pos] = min(max(v, lo), hi) if lo <= hi else v
    if now_free:
        before = t[n - 2] if n >= 2 else raw[n - 1]
        v = max(before - 1.0, 0.0)
        lo_p, hi_p = _period_range(rows[n - 1]["period"])
        lo, hi = lo_p, min(hi_p, before)
        t[n - 1] = min(max(v, lo), hi) if lo <= hi else v
    for k in range(1, n):
        if t[k] > t[k - 1]:
            t[k] = t[k - 1]
    for k, r in enumerate(rows):
        r["t_left"] = t[k]
        lo_p, hi_p = _period_range(r["period"])
        if not lo_p <= t[k] <= hi_p:
            r["period"] = _quarter_of(t[k], r["period"])
            r["half"] = 1 if r["period"] <= 2 else 2
            s = math.floor(t[k] - _period_range(r["period"])[0])
            r["clock"] = f"{s // 60}:{s % 60:02d}"
    return rows


def live_states(s: requests.Session) -> list[dict]:
    """One dict per in-progress or just-finished game, both scoreboards."""
    out, seen = [], set()
    for grp in (80, 81):
        try:
            d = s.get(SCOREBOARD.format(grp=grp), timeout=20).json()
        except (requests.RequestException, ValueError):
            continue
        for e in d.get("events", []):
            # the feed occasionally emits a placeholder event with no id
            try:
                gid = int(e["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if gid in seen:
                continue
            seen.add(gid)
            st = e.get("status", {})
            estate = st.get("type", {}).get("state")
            if estate not in ("in", "post"):
                continue
            c = e["competitions"][0]
            comp = {x["homeAway"]: x for x in c["competitors"]}
            if "home" not in comp or "away" not in comp:
                continue
            sit = c.get("situation") or {}
            period = int(st.get("period") or 1)
            out.append({
                "game_id": gid, "state": estate, "period": period,
                "clock": st.get("displayClock") or "0:00",
                "home_points": int(comp["home"].get("score") or 0),
                "away_points": int(comp["away"].get("score") or 0),
                "home_id": str(comp["home"]["team"]["id"]),
                "away_id": str(comp["away"]["team"]["id"]),
                "home_abbr": comp["home"]["team"].get("abbreviation", ""),
                "away_abbr": comp["away"]["team"].get("abbreviation", ""),
                "down": sit.get("down"), "distance": sit.get("distance"),
                "possession": sit.get("possession"),
                "possession_text": sit.get("possessionText") or "",
                "home_to": sit.get("homeTimeouts"),
                "away_to": sit.get("awayTimeouts"),
                "status_name": st.get("type", {}).get("name"),
                "status_text": st.get("type", {}).get("shortDetail") or st.get("type", {}).get("detail"),
                "down_distance": sit.get("shortDownDistanceText") or re.split(r"\s+at\s+", sit.get("downDistanceText") or "", flags=re.I)[0],
            })
    return out


def _team_aliases(d: dict, r: dict) -> tuple[set, set]:
    """Uppercase name strings ESPN might use for each side in play text."""
    home, away = {r["home_abbr"].upper()}, {r["away_abbr"].upper()}
    try:
        for c in d["header"]["competitions"][0]["competitors"]:
            t = c.get("team") or {}
            names = {str(t.get(k) or "").upper()
                     for k in ("abbreviation", "location", "shortDisplayName",
                               "displayName", "nickname")} - {""}
            (home if c.get("homeAway") == "home" else away).update(names)
    except (KeyError, IndexError, TypeError):
        pass
    return home, away


def fetch_play_rows(s: requests.Session, r: dict,
                    clean: bool = True) -> list[dict]:
    """The game's full play sequence so far as play_raw-shaped rows, state
    strictly BEFORE each play. [] when the drives feed is unavailable."""
    try:
        d = s.get(SUMMARY.format(gid=r["game_id"]), timeout=12).json()
    except (requests.RequestException, ValueError):
        return []
    home_names, away_names = _team_aliases(d, r)
    dr = d.get("drives") or {}
    drives = list(dr.get("previous") or [])
    curd = dr.get("current")
    if curd and all(curd.get("id") != x.get("id") for x in drives):
        drives.append(curd)
    plays = []
    for di, drv in enumerate(drives):
        off_id = str((drv.get("team") or {}).get("id") or "")
        for p in drv.get("plays") or []:
            plays.append((float(p.get("sequenceNumber") or 0), di, off_id, p))
    if not plays:
        return []
    plays.sort(key=lambda x: x[0])
    rows, hp, ap = [], 0, 0
    recv_2h = 0.5
    if plays:
        first_off = plays[0][2]
        # whoever received the opening kick kicks off the second half
        recv_2h = 1.0 if first_off == r["away_id"] else 0.0 \
            if first_off == r["home_id"] else 0.5
    # Timeouts, tracked from the feed itself: 3 per side per half, minus the
    # "Timeout ... by TEAM" plays seen so far. The model learned on real
    # timeout counts (69% of training snaps at 3, the rest below); feeding a
    # constant 3 erased exactly the signal that separates "kneel it out"
    # from "punt coming" in the final minute.
    h_to, a_to, in_2h = 3, 3, False
    prev_off = None
    for seq, di, off_id, p in plays:
        st = p.get("start") or {}
        period = int((p.get("period") or {}).get("number") or 1)
        clock = (p.get("clock") or {}).get("displayValue") or "0:00"
        tl = ((p.get("type") or {}).get("text") or "").lower()
        if period >= 3 and not in_2h:
            h_to, a_to, in_2h = 3, 3, True
        hb = 1.0 if off_id == r["home_id"] else 0.0 \
            if off_id == r["away_id"] else 0.5
        down = st.get("down") or 0
        is_kick = "kickoff" in tl
        off_ytg = float(st.get("yardsToEndzone") or 50)
        is_pass = int("pass" in tl or "sack" in tl or "interception" in tl)
        is_rush = int("rush" in tl)
        rows.append(dict(
            game_id=r["game_id"], season=2026, period=period,
            half=1 if period <= 2 else 2,
            t_left=t_left_of(period, clock), clock=clock,
            home_lead=hp - ap, total_pts_before=hp + ap,
            hp=hp, ap=ap,
            home_has_ball=hb,
            # kickoffs are down=0 in training data (that IS the kick-state
            # flag derive_state reads); other unknown downs default to 1
            down=0 if is_kick else (down if down and down > 0 else 1),
            distance=st.get("distance") or 10,
            home_ytg=off_ytg if hb == 1.0 else 100 - off_ytg
                     if hb == 0.0 else 50.0,
            off_ytg=off_ytg,
            home_to=h_to, away_to=a_to, recv_2h=recv_2h, drive_no=di + 1,
            sit=st.get("downDistanceText") or "",
            play_text=(p.get("text") or "")[:220],
            synthetic=0,
            **{**PLAY_COLS,
               "yds": int(p.get("statYardage") or 0),
               "is_scrim": int(bool(is_pass or is_rush)),
               "is_pass": is_pass,
               "is_sack": int("sack" in tl),
               "is_to": int("interception" in tl or "fumble" in tl),
               "is_pen": int("penalty" in tl)}))
        # possession changes: training flags the LAST play of the old
        # possession (change_of_pos_team); derive_state counts strictly-prior
        # flags into ig_poss, which was silently 0 all game when this was
        # left at the PLAY_COLS default (trained range runs to ~0.8 by Q4)
        if prev_off is not None and off_id and off_id != prev_off \
                and len(rows) >= 2:
            rows[-2]["is_poss_chg"] = 1
        if off_id:
            prev_off = off_id
        if "timeout" in tl or (p.get("text") or "").lower().startswith("timeout"):
            txt = (p.get("text") or "").upper()
            if any(n in txt for n in home_names):
                h_to = max(h_to - 1, 0)
            elif any(n in txt for n in away_names):
                a_to = max(a_to - 1, 0)
        # scores are monotone; max() shrugs off unpopulated feed zeros
        if p.get("homeScore") is not None:
            hp = max(hp, int(p["homeScore"]))
        if p.get("awayScore") is not None:
            ap = max(ap, int(p["awayScore"]))
    # Stamps are untrustworthy (see clean_clock). Callers appending the
    # scoreboard's current state pass clean=False and clean after appending,
    # so that state gets its vote.
    return clean_clock(rows) if clean else rows


def synth_now_row(r: dict, drive_no: int, recv_2h: float,
                  synthetic: int, h_to_fb: int = 3, a_to_fb: int = 3) -> dict:
    """The state RIGHT NOW from the scoreboard poll — appended after the
    play sequence (or standing alone when no drives feed exists)."""
    period = min(r["period"], 5)
    if r["possession"] == r["home_id"]:
        hb = 1.0
    elif r["possession"] == r["away_id"]:
        hb = 0.0
    elif r["possession_text"].startswith(r["home_abbr"] + " "):
        hb = 1.0
    elif r["possession_text"].startswith(r["away_abbr"] + " "):
        hb = 0.0
    else:
        hb = 0.5
    off_ytg = 50.0
    m = re.match(r"^([A-Z&'-]+) (\d+)$", r["possession_text"])
    if m and hb in (0.0, 1.0):
        own = m.group(1) == (r["home_abbr"] if hb == 1.0 else r["away_abbr"])
        yl = float(m.group(2))
        off_ytg = 100 - yl if own else yl
    down = r["down"] if r["down"] and r["down"] > 0 else 1
    dist = r["distance"] if r["distance"] and r["distance"] > 0 else 10
    sit = (f"{down} & {dist}" if r["down"] and r["down"] > 0 else "")
    if r["possession_text"]:
        sit = f"{sit} at {r['possession_text']}" if sit else r["possession_text"]
    return dict(
        game_id=r["game_id"], season=2026, period=period,
        half=1 if period <= 2 else 2,
        t_left=t_left_of(period, r["clock"]), clock=r["clock"],
        home_lead=r["home_points"] - r["away_points"],
        total_pts_before=r["home_points"] + r["away_points"],
        hp=r["home_points"], ap=r["away_points"],
        home_has_ball=hb, down=down, distance=dist,
        home_ytg=off_ytg if hb == 1.0 else 100 - off_ytg
                 if hb == 0.0 else 50.0,
        off_ytg=off_ytg,
        home_to=r["home_to"] if r["home_to"] is not None else h_to_fb,
        away_to=r["away_to"] if r["away_to"] is not None else a_to_fb,
        recv_2h=recv_2h, drive_no=drive_no, sit=sit, play_text="",
        synthetic=synthetic,
        **PLAY_COLS)


def ensure_tables(cur) -> None:
    cur.execute("""
        create table if not exists public.live_predictions (
            game_id bigint primary key,
            updated_at timestamptz not null default now(),
            state text not null default 'in',
            period smallint, clock text,
            home_points smallint, away_points smallint,
            situation text,
            home_win_prob real, pred_margin real, pred_total real)""")
    cur.execute("alter table public.live_predictions "
                "add column if not exists state text not null default 'in'")
    cur.execute("alter table public.live_predictions "
                "add column if not exists live_context jsonb")
    # The live snap-by-snap curve: rewritten per GAME per cycle, never
    # dropped — until cfbfastR ships 2026 play-by-play these are the only
    # curves the season has, so a scorer restart must not erase them.
    cur.execute("""
        create table if not exists public.live_curve (
            game_id bigint, play_index integer,
            updated_at timestamptz not null default now(),
            t_left real, period smallint, clock text,
            down smallint, distance smallint, situation text,
            play_text text,
            home_points smallint, away_points smallint,
            home_win_prob real, pred_margin real, pred_total real,
            primary key (game_id, play_index))""")
    cur.execute("alter table public.live_curve "
                "add column if not exists play_text text")
    # the model's full distribution over the final total AT that snap, so the
    # dashboard's total-points curve can track a hover instead of showing the
    # pregame shape with a moving marker. ~500 B/row, so it is nulled out
    # after 14 days (below) while the lightweight curve rows keep 120.
    cur.execute("alter table public.live_curve "
                "add column if not exists total_pmf real[]")
    for t in ("live_predictions", "live_curve"):
        cur.execute(f"grant select on public.{t} to anon, authenticated")
        cur.execute(f"alter table public.{t} enable row level security")
        cur.execute(f"drop policy if exists anon_read_{t} on public.{t}")
        cur.execute(f"create policy anon_read_{t} on public.{t} "
                    f"for select to anon, authenticated using (true)")


def main() -> None:
    loop = 0
    if "--loop" in sys.argv:
        loop = int(sys.argv[sys.argv.index("--loop") + 1])
    dev = torch.device("cpu")          # small batches; leave MPS to training
    print("loading feature store + 16-member ensemble (once)...")
    d = build_tensors(dev)
    models = load_members(d, dev)
    gid_pos = {int(g): i for i, g in enumerate(d["G_gid"])}
    s = requests.Session()
    s.headers.update({"Accept": "application/json"})
    import psycopg
    finals_curved: set = set()
    while True:
      # one transient Supabase drop must not end a game night: reconnect
      try:
        with psycopg.connect(dsn(), autocommit=True,
                             connect_timeout=30) as cx, cx.cursor() as cur:
          ensure_tables(cur)
          while True:
           # A malformed feed payload must cost one cycle, not the night:
           # DB errors bubble out to reconnect, anything else is logged and
           # retried on the next tick.
           try:
            rows = live_states(s)
            live = [r for r in rows
                    if r["state"] == "in" and r["game_id"] in gid_pos]
            done = [r for r in rows
                    if r["state"] == "post" and r["game_id"] in gid_pos]
            for r in done:
                cur.execute("""
                    insert into public.live_predictions
                      (game_id, updated_at, state, period, clock,
                       home_points, away_points, situation)
                    values (%s, now(), 'post', %s, 'FINAL', %s, %s, '')
                    on conflict (game_id) do update set
                      updated_at=now(), state='post', clock='FINAL',
                      period=excluded.period,
                      home_points=excluded.home_points,
                      away_points=excluded.away_points, situation='', live_context=null
                """, (r["game_id"], r["period"],
                      r["home_points"], r["away_points"]))

            # a game's final curve gets one full pass after it ends
            todo = live + [r for r in done
                           if r["game_id"] not in finals_curved]
            frames = []
            for r in todo:
                prows = fetch_play_rows(s, r, clean=False)
                time.sleep(0.1)
                if prows and r["state"] == "in":
                    now = synth_now_row(
                        r, prows[-1]["drive_no"], prows[0]["recv_2h"], 0,
                        prows[-1]["home_to"], prows[-1]["away_to"])
                    # the scoreboard's clock votes on which stamps to trust
                    # and places itself after the last play when it trails
                    prows.append(now)
                    clean_clock(prows, now_last=True)
                elif prows:
                    clean_clock(prows)
                else:
                    if r["state"] == "post":
                        # nothing to draw, but mark it done: without this the
                        # game is re-fetched every cycle forever (21 finals
                        # with no drives feed stretched a 60 s cycle to 12 min)
                        finals_curved.add(r["game_id"])
                        continue
                    prows = [synth_now_row(r, 1, 0.5, 1)]
                # each row prices the state BEFORE a play; the tooltip should
                # narrate the play that PRODUCED that state, so the score
                # jump and its description land on the same point (a TD's
                # drop otherwise reads as caused by the ensuing kickoff)
                for i in range(len(prows) - 1, 0, -1):
                    prows[i]["play_text"] = prows[i - 1]["play_text"]
                if prows:
                    prows[0]["play_text"] = ""
                for i, pr in enumerate(prows):
                    pr["play_index"] = i
                frames.extend(prows)
                if r["state"] == "post":
                    finals_curved.add(r["game_id"])

            if frames:
                p = prep_play(pd.DataFrame(frames))
                # lone synthetic rows have no prior plays: derive_state's
                # elapsed/max(plays,1) pace explodes; substitute college norms
                syn = p["synthetic"].to_numpy() == 1
                if syn.any():
                    elapsed = 3600.0 - p["t_left"].to_numpy(float)
                    p.loc[syn, "ig_plays"] = (elapsed[syn] / 26.0) / 120.0
                    p.loc[syn, "ig_poss"] = (elapsed[syn] / 160.0) / 24.0
                    p.loc[syn, "ig_pace"] = np.where(
                        elapsed[syn] > 0, 26.0 / 40.0, 0.0)
                xs = torch.tensor(p[STATE_FEATURES].to_numpy(np.float32))
                fr = torch.tensor(p["t_left"].to_numpy(np.float32)) / 3600.0
                al = torch.tensor(
                    np.abs(p["home_lead"].to_numpy(np.float32))) / 21.0
                ld = torch.tensor(p["home_lead"].to_numpy(np.float32))
                pz = torch.tensor(p["total_pts_before"].to_numpy(np.float32))
                gj = torch.tensor([gid_pos[int(g)] for g in p["game_id"]])
                pm_s = pt_s = None
                with torch.no_grad():
                    for mdl in models:
                        o = mdl(game_batch(d, gj, dev), xs, fr, al, ld, pz)
                        pm = torch.softmax(o["margin"], -1)
                        pt = torch.softmax(o["total"], -1)
                        pm_s = pm if pm_s is None else pm_s + pm
                        pt_s = pt if pt_s is None else pt_s + pt
                pt_avg = pt_s / len(models)
                r_out = readouts(pm_s / len(models), pt_avg)
                wp, em, et = r_out["wp"], r_out["em"], r_out["et"]
                tpmf = np.round(pt_avg.numpy(), 6)

                meta = {r["game_id"]: r for r in todo}
                p = p.reset_index(drop=True)
                for gid_i, gp in p.groupby("game_id", sort=False):
                    gid = int(gid_i)
                    r = meta[gid]
                    idx = gp.index.to_numpy()
                    # Games ESPN serves no drives feed for produce ONE
                    # synthetic row per cycle. Rewriting would pin them at a
                    # single point forever, so those APPEND (curve builds a
                    # read at a time); games with real play-by-play rewrite
                    # in full, which also replaces any earlier synthetic rows.
                    fallback = bool(gp["synthetic"].astype(int).min() == 1)
                    if fallback:
                        cur.execute("""select play_index, t_left, home_points,
                                              away_points
                                       from public.live_curve
                                       where game_id=%s
                                       order by play_index desc limit 1""",
                                    (gid,))
                        prev = cur.fetchone()
                        i0 = int(idx[0])
                        if prev is None:
                            # anchor the curve at kickoff with the pregame
                            # call, so one coarse read still draws a line
                            # (and the line starts where the model started)
                            cur.execute("""select home_win_prob, pred_margin,
                                                  pred_total
                                           from public.game_predictions
                                           where game_id=%s""", (gid,))
                            pg = cur.fetchone()
                            if pg is not None:
                                cur.execute("""
                                    insert into public.live_curve
                                      (game_id, play_index, t_left, period,
                                       clock, down, distance, situation,
                                       play_text, home_points, away_points,
                                       home_win_prob, pred_margin, pred_total)
                                    values (%s,0,3600,1,'15:00',1,10,
                                            'Pregame','',0,0,%s,%s,%s)
                                    on conflict do nothing""",
                                    (gid, float(pg[0]), float(pg[1]),
                                     float(pg[2])))
                                gp.at[i0, "play_index"] = 1
                        else:
                            # nothing moved since the last read — don't stack
                            # a duplicate point on the same spot
                            if (abs(float(prev[1]) - float(gp.at[i0, "t_left"])) < 0.5
                                    and int(prev[2]) == int(gp.at[i0, "hp"])
                                    and int(prev[3]) == int(gp.at[i0, "ap"])):
                                idx = np.empty(0, int)
                            else:
                                gp.at[i0, "play_index"] = int(prev[0]) + 1
                    else:
                        cur.execute("delete from public.live_curve "
                                    "where game_id=%s", (gid,))
                    cur.executemany("""
                        insert into public.live_curve
                          (game_id, play_index, t_left, period, clock, down,
                           distance, situation, play_text,
                           home_points, away_points,
                           home_win_prob, pred_margin, pred_total, total_pmf)
                        values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """, [(gid, int(gp.at[i, "play_index"]),
                           float(gp.at[i, "t_left"]),
                           int(gp.at[i, "period"]), gp.at[i, "clock"],
                           int(gp.at[i, "down"]), int(gp.at[i, "distance"]),
                           gp.at[i, "sit"] or "",
                           gp.at[i, "play_text"] or "",
                           int(gp.at[i, "hp"]), int(gp.at[i, "ap"]),
                           float(wp[i]), float(em[i]), float(et[i]),
                           [float(x) for x in tpmf[i]])
                          for i in idx])
                    if r["state"] != "in":
                        continue
                    # the headline read is always this cycle's last row, even
                    # when the curve write above skipped a duplicate point
                    last = int(gp.index.to_numpy()[-1])
                    sit = gp.at[last, "sit"] or ""
                    stamp = datetime.now(timezone.utc).isoformat()
                    cur.execute("""
                        insert into public.live_predictions
                          (game_id, updated_at, state, period, clock,
                           home_points, away_points, situation,
                           home_win_prob, pred_margin, pred_total, live_context)
                        values (%s, %s, 'in', %s,%s,%s,%s,%s,%s,%s,%s,%s)
                        on conflict (game_id) do update set
                          updated_at=excluded.updated_at, state='in',
                          period=excluded.period, clock=excluded.clock,
                          home_points=excluded.home_points,
                          away_points=excluded.away_points,
                          situation=excluded.situation,
                          home_win_prob=excluded.home_win_prob,
                          pred_margin=excluded.pred_margin,
                          pred_total=excluded.pred_total,
                          live_context=excluded.live_context
                    """, (gid, stamp, r["period"], r["clock"],
                          r["home_points"], r["away_points"], sit,
                          float(wp[last]), float(em[last]), float(et[last]),
                          Jsonb(scoreboard_context(r, stamp))))
                    print(f"  {r['away_abbr']} {r['away_points']} @ "
                          f"{r['home_abbr']} {r['home_points']}  "
                          f"Q{r['period']} {r['clock']}  "
                          f"{len(idx)} snaps  [{sit or 'no situation'}]  "
                          f"home wp {wp[last]:.3f}  margin {em[last]:+.1f}  "
                          f"total {et[last]:.1f}")

            # 'post' rows must outlive the gap to the next official ingest;
            # curves keep far longer — they are the season's only
            # snap-by-snap record until 2026 play-by-play is ingestable
            cur.execute("delete from public.live_predictions "
                        "where updated_at < now() - interval '8 days'")
            cur.execute("delete from public.live_curve "
                        "where updated_at < now() - interval '120 days'")
            # drop the heavy per-snap distributions early; the curve itself
            # (scores, situations, readouts) is what keeps for 120 days
            cur.execute("update public.live_curve set total_pmf = null "
                        "where updated_at < now() - interval '14 days' "
                        "and total_pmf is not null")
            n_off = len(rows) - len(live) - len(done)
            print(f"cycle: {len(live)} live games scored, "
                  f"{len(done)} finals carried"
                  f"{f', {n_off} outside the population' if n_off else ''}"
                  f" ({time.strftime('%H:%M:%S')})", flush=True)
           except psycopg.Error:
            raise                    # DB trouble: reconnect below
           except Exception as exc:
            import traceback
            print(f"cycle failed ({type(exc).__name__}: {str(exc)[:120]}) "
                  f"— retrying next tick", flush=True)
            traceback.print_exc()
           if not loop:
               break
           time.sleep(loop)
      except psycopg.OperationalError as exc:
          print(f"db connection lost: {str(exc)[:90]} — reconnecting in 30s",
                flush=True)
          time.sleep(30)
          continue
      if not loop:
          break


if __name__ == "__main__":
    main()
