"""Publish everything to Supabase: raw training tables and all predictions.

Supabase is the system of record. What goes up:
  game_scalars     raw pregame features, every FBS-involved game 2001-2025
  game_history     each side's last 14 raw results per game
  play_features    raw in-game state (unchanged)
  game_predictions pregame prediction for every game, with split labels
  play_predictions the win-probability curve for every play 2014-2025
  model_runs       run metadata with held-out metrics

team_ratings is DROPPED: the project rule is no rating algorithms, and keeping
the table around invites someone to join it back in.
"""
from __future__ import annotations

import io
import json
import sys
import uuid
from pathlib import Path

import pandas as pd
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.config import FEAT, MODELS, PQ, TEST_SEASONS, TRAIN_SEASONS  # noqa: E402
from cfbpred.load_supabase import dsn  # noqa: E402

# earliest season whose snap-by-snap curves get published (space budget)
PUBLISH_PLAYS_SINCE = 2024


def copy_chunked(cur, df, table, cols, chunk=200_000):
    total = 0
    for i in range(0, len(df), chunk):
        buf = io.StringIO()
        df.iloc[i:i + chunk][cols].to_csv(buf, index=False, header=False,
                                          na_rep="")
        qcols = ",".join(f'"{c}"' for c in cols)
        with cur.copy(f"copy {table} ({qcols}) from stdin "
                      f"with (format csv, null '')") as cp:
            cp.write(buf.getvalue())
        total += min(chunk, len(df) - i)
    return total


def grant(cur, table):
    cur.execute(f"grant select on public.{table} to anon, authenticated")
    cur.execute(f"alter table public.{table} enable row level security")
    cur.execute(f"drop policy if exists anon_read_{table} on public.{table}")
    cur.execute(f"create policy anon_read_{table} on public.{table} "
                f"for select to anon, authenticated using (true)")


def main(raw_only: bool = False) -> None:
    """raw_only publishes just the training tables (stage 4a) — predictions
    and the model_runs stamp wait until a model has actually been trained on
    these tables (stage 8)."""
    gs = pd.read_parquet(FEAT / "game_scalars.parquet")
    gh = pd.read_parquet(FEAT / "game_history.parquet")
    pr = pd.read_parquet(FEAT / "play_raw.parquet")
    if not raw_only:
        gp = pd.read_parquet(PQ / "game_predictions.parquet")
        pp = pd.read_parquet(PQ / "predictions.parquet")
        met_all = json.loads((MODELS / "metrics_raw.json").read_text())
        met = met_all["test"]
        graded_on = met_all.get("graded_on", "test")
    run_id = str(uuid.uuid4())
    version = "raw-dnn-delta-v22"

    with psycopg.connect(dsn(), autocommit=True) as cx, cx.cursor() as cur:
        # The server default is statement_timeout = 2min, which the
        # game_scalars COPY (28k rows x 270 columns) now exceeds through the
        # pooler — the nightly refresh died on it 2026-08-30 mid-COPY at line
        # 6325. Raise it for THIS session only; the reads elsewhere keep the
        # server default so a runaway query is still cut off.
        cur.execute("set statement_timeout = '20min'")
        cur.execute("drop table if exists public.team_ratings cascade")
        cur.execute("drop table if exists public.game_features cascade")

        # ---- raw training tables ----
        cur.execute("drop table if exists public.game_scalars cascade")
        cols_sql = []
        for c in gs.columns:
            vals = gs[c].dropna()
            is_bool = (str(gs[c].dtype) in ("bool", "boolean")
                       or (gs[c].dtype == object and len(vals)
                           and all(isinstance(v, (bool,)) for v in
                                   vals.unique()[:20])))
            t = ("bigint" if c == "game_id"
                 else "boolean" if is_bool
                 else "text" if gs[c].dtype == object
                 else "smallint" if c in ("season", "week", "week_c",
                                          "home_points", "away_points",
                                          "margin", "total")
                 else "real")
            cols_sql.append(f'"{c}" {t}')
        cur.execute(f"create table public.game_scalars "
                    f"({', '.join(cols_sql)}, primary key (game_id))")
        # scheduled games carry null scores, which make these columns float —
        # nullable Int64 renders '24' / '' instead of '24.0', which COPY into
        # smallint would reject
        for c in ("season", "week", "week_c", "home_points", "away_points",
                  "margin", "total"):
            gs[c] = pd.to_numeric(gs[c], errors="coerce").round().astype("Int64")
        n = copy_chunked(cur, gs, "public.game_scalars", list(gs.columns))
        print(f"  game_scalars      {n:,}")

        # ---- raw play table (lean: derivable columns reconstructed at load)
        cur.execute("drop table if exists public.play_features cascade")
        cur.execute("drop table if exists public.play_raw cascade")
        PR_COLS = {"game_id": "bigint", "play_index": "integer",
                   "season": "smallint", "period": "smallint",
                   "half": "smallint", "t_left": "smallint",
                   "home_lead": "smallint", "home_has_ball": "smallint",
                   "down": "smallint", "distance": "smallint",
                   "home_ytg": "smallint", "off_ytg": "smallint",
                   "home_to": "smallint", "away_to": "smallint",
                   "recv_2h": "smallint", "yds": "smallint",
                   "is_scrim": "smallint", "is_pass": "smallint",
                   "is_sack": "smallint", "is_to": "smallint",
                   "is_pen": "smallint", "is_poss_chg": "smallint",
                   "drive_no": "smallint", "total_pts_before": "smallint",
                   "h_pqb_play": "smallint", "a_pqb_play": "smallint",
                   "h_prb_play": "smallint", "a_prb_play": "smallint",
                   "h_pqb_known": "smallint", "a_pqb_known": "smallint"}
        cur.execute("create table public.play_raw ("
                    + ", ".join(f'"{c}" {t}' for c, t in PR_COLS.items())
                    + ", primary key (game_id, play_index))")
        prq = pr[list(PR_COLS)].copy()
        for c in PR_COLS:
            if c != "game_id":
                prq[c] = (pd.to_numeric(prq[c], errors="coerce")
                          .fillna(0).round().astype("int64"))
        n = copy_chunked(cur, prq, "public.play_raw", list(PR_COLS))
        print(f"  play_raw          {n:,}")

        cur.execute("drop table if exists public.game_history cascade")
        cur.execute("""create table public.game_history (
            game_id bigint, team text, slot smallint, opp_idx smallint,
            pf real, pa real, was_home real, opp_fbs real,
            days_ago real, same_season real, opp_talent real,
            primary key (game_id, team, slot))""")
        n = copy_chunked(cur, gh, "public.game_history", list(gh.columns))
        print(f"  game_history      {n:,}")

        if raw_only:
            for t in ("game_scalars", "game_history", "play_raw"):
                grant(cur, t)
            cur.execute("notify pgrst, 'reload schema'")
            cur.execute("select pg_size_pretty(pg_database_size(current_database()))")
            print(f"\n  database total: {cur.fetchone()[0]} (limit 500 MB)")
            print("raw tables only — predictions/model_runs unchanged")
            return

        # ---- model run ----
        cur.execute("""
            insert into public.model_runs
              (run_id, model_version, train_seasons, test_seasons, n_params,
               notes, margin_rmse, wp_logloss_all, wp_logloss_q1, wp_ece_q1)
            values (%s,%s,int4range(%s,%s,'[]'),int4range(%s,%s,'[]'),%s,%s,%s,%s,%s,%s)
        """, (run_id, version, 2001, max(TRAIN_SEASONS),
              min(TEST_SEASONS), max(TEST_SEASONS), 235_419,
              "v22: real geography and real rest for every program. Travel "
              "distance is measured from a team's home stadium, but the home "
              "stadium was derived only from games whose HOME side was FBS, "
              "so no FCS or lower-division program had one — and a null "
              "standardises to the training mean, i.e. the net was told an "
              "unknown trip was an average trip. Away-side travel was 40% "
              "null overall and ~50% from 2021 on, once FCS-vs-FCS games "
              "entered the corpus. The base now comes from any hosted "
              "non-neutral game in any division, carried from the nearest "
              "season a team did host: travel null 30/40% -> 2.7/3.0% (home/"
              "away), the remainder being venues with no coordinates. Rest "
              "days were grouped by (team, season), so EVERY season opener "
              "was null and read as an ordinary ~8-day turnaround; the diff "
              "now crosses the season boundary, capped at 21 days so the "
              "offseason saturates without flattening the week-to-week "
              "signal (uncapped, the column sd goes 3.0 -> 10.5 days and a "
              "bye week collapses from +1.1 sd to +0.3). Nulls 8.8/10.5% -> "
              "0.3/0.7%. Altitude delta and body-clock shift improved with "
              "the same base. Architecture, corpus and column count are "
              "unchanged from v21 (delta heads, s2d gate, 270 scalars). "
              "Train 2014-2025, 2026 "
              "is the live held-out season, predicted pregame for the full "
              "schedule and graded as games complete. Metrics stamped here "
              f"are graded on: {graded_on}. All raw counts; no Elo, no "
              "ratings, no lines.",
              met["margin_rmse"], met["logloss"], met["logloss_q1"],
              met["ece_q1"]))

        # ---- predictions ----
        cur.execute("drop table if exists public.game_predictions cascade")
        cur.execute("""create table public.game_predictions (
            run_id uuid, game_id bigint, season smallint, split text,
            model_version text not null,
            home_win_prob real check (home_win_prob between 0 and 1),
            pred_margin real, pred_total real,
            pred_home_pts smallint, pred_away_pts smallint,
            primary key (run_id, game_id))""")
        gp["run_id"] = run_id
        gp["model_version"] = version
        n = copy_chunked(cur, gp, "public.game_predictions",
                         ["run_id", "game_id", "season", "split", "model_version",
                          "home_win_prob", "pred_margin", "pred_total",
                          "pred_home_pts", "pred_away_pts"])
        print(f"  game_predictions  {n:,}")

        # Single-run table: the run's identity lives in model_runs, not on
        # 1.8M rows. A per-row uuid plus a duplicate index cost 145 MB of the
        # 500 MB tier for nothing — the (game_id, play_index) primary key is
        # the only access path the dashboard uses.
        # Only recent seasons are published: snap-by-snap curves for decade-old
        # games are the least valuable bytes in a 500 MB budget, and the v20
        # FCS-target rows + a season of live 2026 pbp need the headroom. The
        # dashboard shows "no curve" for older games; every game-level number
        # stays published for every season.
        pp = pp[pp["season"] >= PUBLISH_PLAYS_SINCE].copy()
        cur.execute("drop table if exists public.play_predictions cascade")
        cur.execute("""create table public.play_predictions (
            game_id bigint, play_index integer,
            game_seconds_left smallint, home_lead smallint,
            home_win_prob real, pred_margin real, pred_total real,
            primary key (game_id, play_index))""")
        pp["game_seconds_left"] = pp["t_left"].astype(int)
        pp["home_lead"] = pp["home_lead"].astype(int)
        n = copy_chunked(cur, pp, "public.play_predictions",
                         ["game_id", "play_index", "game_seconds_left",
                          "home_lead", "home_win_prob", "pred_margin",
                          "pred_total"])
        print(f"  play_predictions  {n:,}")

        # ---- live-season pregame PMFs: lets the app price any current line
        # exactly (cover probability = a partial sum of these bins) ----
        pmf_path = PQ / "game_pmfs.parquet"
        if pmf_path.exists():
            pf = pd.read_parquet(pmf_path)
            cur.execute("drop table if exists public.margin_pmf cascade")
            cur.execute("drop table if exists public.game_pmf cascade")
            cur.execute("""create table public.game_pmf (
                game_id bigint primary key, season smallint,
                margin_pmf real[] not null,   -- bins -56..56 + two tails
                total_pmf real[] not null)    -- bins 10..110 + two tails""")
            cur.executemany(
                "insert into public.game_pmf values (%s,%s,%s,%s)",
                [(int(r.game_id), int(r.season),
                  [float(x) for x in r.margin_pmf],
                  [float(x) for x in r.total_pmf]) for r in pf.itertuples()])
            grant(cur, "game_pmf")
            print(f"  game_pmf          {len(pf):,}")

        for t in ("game_scalars", "game_history", "game_predictions",
                  "play_predictions", "play_raw"):
            grant(cur, t)
        cur.execute("notify pgrst, 'reload schema'")
        cur.execute("select pg_size_pretty(pg_database_size(current_database()))")
        print(f"\n  database total: {cur.fetchone()[0]} (limit 500 MB)")
    print(f"run_id {run_id}")

    # publish.py DROPS game_predictions CASCADE, which takes
    # public.v_model_vs_vegas with it — the Results page 404s until
    # publish_lines.py recreates it. Both refresh_week.sh and train_2026.sh
    # run them in that order, but a manual `python3 publish.py` leaves the
    # site broken with nothing to say so. Fail loudly instead of silently.
    with psycopg.connect(dsn(), autocommit=True) as cx, cx.cursor() as cur:
        cur.execute("select to_regclass('public.v_model_vs_vegas') is null")
        if cur.fetchone()[0]:
            print("\n  !! v_model_vs_vegas was dropped by the cascade above.\n"
                  "  !! The dashboard Results page will 404 until you run:\n"
                  "  !!     python3 src/cfbpred/publish_lines.py",
                  file=sys.stderr)


if __name__ == "__main__":
    main(raw_only="--raw-only" in sys.argv)
