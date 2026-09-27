"""Raw feature store. No rating algorithms, no external model outputs.

Hard rule, per the project owner: every input is raw game / team / coach /
player data, a deterministic count over it, or a human scouting assessment of
PLAYERS (recruiting points, the 247 talent composite — allowed by decision
2026-08-26). Banned outright: CFBD Elo, returning-production PPA, betting
lines, cfbfastR EPA/WP, and our own fitted ratings and expected-points model.
Team strength is the NETWORK's job to learn, via team embeddings and each
team's actual prior results.

Outputs:
  features/game_scalars.parquet   one row per FBS-involved game, raw scalars
  features/game_history.parquet   per game x side x slot: the last N_HIST real
                                  results of that team, opponent identity
                                  included so the net can opponent-adjust
  features/team_vocab.json        team/conference/coach index maps
  (play_features.parquet is already raw state and is reused)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.config import CFBD, CURRENT_SEASON, FEAT, PBP, PQ  # noqa: E402
from cfbpred.player_features import (build_box_features,  # noqa: E402
                                     load_player_stats)

N_HIST = 14          # prior games per team fed to the history encoder
# (20 was measured 2026-08-27: null on val/test, reverted)
QB_WINDOW = 4        # games of rolling raw QB stats
REST_CAP = 21.0      # days; saturates the offseason and bowl layoffs

POS_GROUPS = {
    "QB": ["QB"], "OL": ["OL", "OT", "OG", "C"],
    "DL": ["DL", "DT", "DE", "NT"], "SKILL": ["RB", "WR", "TE", "FB"],
    "DB": ["DB", "CB", "S"], "LB": ["LB", "ILB", "OLB", "EDGE"],
}


def earth_miles(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 3958.8 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_home_base(g_all: pd.DataFrame, ven: pd.DataFrame,
                    keep_teams: set) -> pd.DataFrame:
    """Each team-season's home stadium, as lat/lon/elevation.

    Travel is measured from here, so a team without a base gets a null
    distance — and a null standardises to the training mean, i.e. the model
    is told an unknown trip was an average one. The old rule counted only
    games whose HOME side was FBS, which left every FCS and lower-division
    program baseless: 40% of away-side travel was missing, ~50% from 2021 on
    once FCS-vs-FCS games entered the corpus. Since v16 those programs have
    real identities and real histories; this gives them real geography too.

    Three sources, in order:
      1. the modal venue among games the team actually HOSTED that season,
         any division, neutral sites excluded (a neutral host is not a home);
      2. for a season with no hosted game (a full road slate, or an FCS team
         appearing only as a visitor), that team's base from the nearest
         season it did host;
      3. nothing — a team that never hosts in the corpus keeps a null, which
         is honest: we do not know where it plays.

    Ties (a home slate genuinely split across two stadiums, e.g. UMass
    2014-15 between McGuirk and Gillette) break on the lower venue_id so the
    store is reproducible; the old sort left them to row order.
    """
    host = g_all[~g_all["neutral_site"].fillna(False).astype(bool)]
    base = (host.groupby(["season", "home_team", "venue_id"]).size()
            .reset_index(name="n")
            .sort_values(["n", "venue_id"], ascending=[False, True])
            .drop_duplicates(["season", "home_team"])
            .merge(ven, on="venue_id")
            .rename(columns={"home_team": "team", "latitude": "b_lat",
                             "longitude": "b_lon", "elevation": "b_elev"})
            .dropna(subset=["b_lat", "b_lon"])
            [["season", "team", "b_lat", "b_lon", "b_elev"]])

    appear = pd.concat([
        g_all[["season", "home_team"]].rename(columns={"home_team": "team"}),
        g_all[["season", "away_team"]].rename(columns={"away_team": "team"}),
    ]).drop_duplicates()
    appear = appear[appear["team"].isin(keep_teams)]
    gap = appear.merge(base[["season", "team"]], on=["season", "team"],
                       how="left", indicator=True)
    gap = gap[gap["_merge"] == "left_only"][["season", "team"]]
    carried = gap.merge(base, on="team", suffixes=("", "_src"))
    carried["_d"] = (carried["season"] - carried["season_src"]).abs()
    carried = (carried.sort_values(["_d", "season_src"])
               .drop_duplicates(["season", "team"])
               [["season", "team", "b_lat", "b_lon", "b_elev"]])
    out = pd.concat([base, carried], ignore_index=True)
    print(f"home base: {len(base):,} hosted + {len(carried):,} carried "
          f"= {len(out):,} team-seasons "
          f"({len(appear) - len(out):,} never host, no base)")
    return out


# ------------------------------------------------------------------ game log

def load_games(con, fbs_only: bool = True) -> pd.DataFrame:
    """FBS-involved games are the modeling population; fbs_only=False loads
    EVERY completed game (FCS-vs-FCS included) so lower-division teams get
    real histories and season records instead of a blind spot.

    schedule.parquet (the current season via update_season.py) is unioned in
    for game_ids not already in the completed corpus — those rows carry null
    scores until played, and everything downstream treats a null-score game
    as a prediction target only, never a source of history or stats."""
    div_filter = ("AND (home_division = 'fbs' OR away_division = 'fbs')"
                  if fbs_only else "")
    sched = PQ / "schedule.parquet"
    sched_union = f"""
        UNION ALL BY NAME
        SELECT game_id, season, week, season_type, start_date, neutral_site,
               conference_game, venue_id,
               home_team, home_conference, home_division, home_points,
               away_team, away_conference, away_division, away_points,
               margin, total
        FROM read_parquet('{sched}')
        WHERE week IS NOT NULL {div_filter}
          AND game_id NOT IN (
            SELECT game_id FROM read_parquet('{PQ / "games.parquet"}'))
    """ if sched.exists() else ""
    g = con.execute(f"""
        SELECT game_id, season, week, season_type, start_date, neutral_site,
               conference_game, venue_id,
               home_team, home_conference, home_division, home_points,
               away_team, away_conference, away_division, away_points,
               margin, total
        FROM read_parquet('{PQ / "games.parquet"}')
        WHERE week IS NOT NULL
          {div_filter}
        {sched_union}
    """).fetchdf()
    g["week"] = g["week"].astype(int)
    g["kick"] = pd.to_datetime(g["start_date"], format="mixed", utc=True)
    # A live score is not a completed result. Current-season providers expose
    # an explicit final flag; use it for every history and statistic block.
    status_path = CFBD / "games.parquet"
    if status_path.exists():
        status = pd.read_parquet(status_path, columns=["id", "completed"])
        final = status.drop_duplicates("id", keep="last").set_index("id").completed
        current = g.season == CURRENT_SEASON
        unfinished = current & ~g.game_id.map(final).fillna(False).astype(bool)
        g.loc[unfinished, ["home_points", "away_points", "margin", "total"]] = np.nan
    future = g.kick >= pd.Timestamp.now(tz="UTC")
    g.loc[future, ["home_points", "away_points", "margin", "total"]] = np.nan
    # The source carries occasional duplicate games under two ids (e.g. LSU vs
    # Appalachian State 2008 exists as 400361387 and 282430099). One survives.
    g = g.drop_duplicates(["season", "kick", "home_team", "away_team"],
                          keep="first")
    # Postseason week numbering restarts at 1, which would collide with
    # regular-season weeks in every season-to-date join. week_c is continuous.
    g["week_c"] = np.where(g["season_type"] == "postseason",
                           g["week"] + 20, g["week"])
    return g.sort_values(["season", "kick"]).reset_index(drop=True)


def team_log(g: pd.DataFrame, keep_teams: set | None = None) -> pd.DataFrame:
    """One row per (game, team-side). keep_teams limits which programs get a
    log (every vocab team, FCS included); None keeps the old FBS-only rule."""
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        mask = (g[f"{side}_team"].isin(keep_teams) if keep_teams is not None
                else g[f"{side}_division"] == "fbs")
        d = g[mask][[
            "game_id", "season", "week", "kick", "neutral_site",
            f"{side}_team", f"{opp}_team", f"{opp}_division",
            f"{side}_points", f"{opp}_points"]].copy()
        d.columns = ["game_id", "season", "week", "kick", "neutral",
                     "team", "opp", "opp_div", "pf", "pa"]
        d["was_home"] = 1.0 if side == "home" else 0.0
        rows.append(d)
    log = pd.concat(rows, ignore_index=True)
    log["neutral"] = log["neutral"].fillna(False).astype(bool)
    log.loc[log["neutral"], "was_home"] = 0.5
    return log.sort_values(["team", "kick"]).reset_index(drop=True)


def build_history(log: pd.DataFrame, team_idx: dict,
                  emit_gids: set | None = None,
                  opp_talent: dict | None = None) -> pd.DataFrame:
    """For every appearance, the team's previous N_HIST results — strictly
    earlier by kickoff timestamp, opponent identity attached. The slots draw
    from the FULL log (FCS-vs-FCS included), but rows are only emitted for
    games in emit_gids (the modeling population) so the table stays lean.

    Scheduled-but-unplayed games (null scores) get history rows EMITTED —
    they are the games being predicted — but never fill a slot: only played
    games are results."""
    out = []
    for team, d in log.groupby("team", sort=False):
        d = d.sort_values("kick").reset_index()
        pf = d["pf"].to_numpy(float)
        pa = d["pa"].to_numpy(float)
        wh = d["was_home"].to_numpy(float)
        played = np.isfinite(pf) & np.isfinite(pa)
        fbs_opp = (d["opp_div"] == "fbs").to_numpy()
        # every opponent with a vocab identity is named, FCS included
        oidx = d["opp"].map(lambda t: team_idx.get(t, 0)).to_numpy()
        kick = d["kick"].to_numpy()
        season = d["season"].to_numpy()
        # that season's preseason talent composite of the opponent (scaled),
        # 0 where unpublished — "beat whom" now carries a scouting prior too
        otal = (np.array([opp_talent.get((int(sn), op), 0.0)
                          for sn, op in zip(season, d["opp"])]) / 1000.0
                if opp_talent is not None else np.zeros(len(d)))
        gids = d["game_id"].to_numpy()
        for i in range(len(d)):
            if emit_gids is not None and gids[i] not in emit_gids:
                continue
            s, j = 0, i - 1
            while j >= 0 and s < N_HIST:
                if not played[j]:
                    j -= 1
                    continue
                days = (kick[i] - kick[j]) / np.timedelta64(1, "D")
                out.append((int(d.loc[i, "game_id"]), team, s,
                            int(oidx[j]), pf[j], pa[j], wh[j],
                            float(fbs_opp[j]), float(min(days, 400.0)),
                            1.0 if season[j] == season[i] else 0.0,
                            float(otal[j])))
                s += 1
                j -= 1
    h = pd.DataFrame(out, columns=["game_id", "team", "slot", "opp_idx",
                                   "pf", "pa", "was_home", "opp_fbs",
                                   "days_ago", "same_season", "opp_talent"])
    return h


# ------------------------------------------------- raw pbp counting stats

PBP_AGG = """
WITH sc AS (
  SELECT "year" AS season,
         CASE WHEN season_type = 'postseason' THEN week + 20 ELSE week END AS week,
         {side} AS team,
         count(*) FILTER (WHERE rush OR "pass")                    AS plays,
         sum(yards_gained) FILTER (WHERE rush OR "pass")           AS yards,
         count(*) FILTER (WHERE "pass")                            AS pass_plays,
         count(*) FILTER (WHERE sack)                              AS sacks,
         count(*) FILTER (WHERE "int")                             AS ints,
         count(*) FILTER (WHERE fumble_vec)                        AS fumbles,
         count(*) FILTER (WHERE turnover)                          AS turnovers,
         count(*) FILTER (WHERE (rush OR "pass") AND yards_gained >= 10) AS expl10,
         count(*) FILTER (WHERE (rush OR "pass") AND yards_gained >= 20) AS expl20,
         count(*) FILTER (WHERE down = 3)                          AS third_att,
         count(*) FILTER (WHERE down = 3 AND yards_gained >= distance) AS third_conv,
         count(*) FILTER (WHERE down = 4 AND (rush OR "pass"))     AS fourth_go,
         count(*) FILTER (WHERE punt)                              AS punts,
         count(*) FILTER (WHERE penalty_flag)                      AS pens,
         count(*) FILTER (WHERE touchdown)                         AS tds,
         count(*) FILTER (WHERE fg_made)                           AS fgs
  FROM read_parquet('{pbp}')
  WHERE period <= 4 AND week IS NOT NULL
  GROUP BY 1, 2, 3
)
SELECT * FROM sc
"""

RATE_COLS = ["ypp", "pass_share", "sack_rate", "int_rate", "fumble_rate",
             "to_rate", "expl10_rate", "expl20_rate", "third_pct",
             "fourth_go_rate", "punt_rate", "pen_rate", "td_rate", "fg_rate"]


def cum_rates(agg: pd.DataFrame) -> pd.DataFrame:
    """Cumulative season-to-date rates as of each week — strictly prior weeks."""
    agg = agg.sort_values(["team", "season", "week"])
    sums = agg.groupby(["team", "season"]).apply(
        lambda d: d.set_index("week").cumsum().shift(1), include_groups=False
    ).reset_index()
    s = sums
    eps = 1.0
    out = pd.DataFrame({
        "team": s["team"], "season": s["season"], "week": s["week"],
        "plays_cum": s["plays"],
        "ypp": s["yards"] / (s["plays"] + eps),
        "pass_share": s["pass_plays"] / (s["plays"] + eps),
        "sack_rate": s["sacks"] / (s["plays"] + eps),
        "int_rate": s["ints"] / (s["plays"] + eps),
        "fumble_rate": s["fumbles"] / (s["plays"] + eps),
        "to_rate": s["turnovers"] / (s["plays"] + eps),
        "expl10_rate": s["expl10"] / (s["plays"] + eps),
        "expl20_rate": s["expl20"] / (s["plays"] + eps),
        "third_pct": s["third_conv"] / (s["third_att"] + eps),
        "fourth_go_rate": s["fourth_go"] / (s["plays"] + eps),
        "punt_rate": s["punts"] / (s["plays"] + eps),
        "pen_rate": s["pens"] / (s["plays"] + eps),
        "td_rate": s["tds"] / (s["plays"] + eps),
        "fg_rate": s["fgs"] / (s["plays"] + eps),
    })
    return out


def qb_features(con) -> pd.DataFrame:
    """Rolling raw stats of the team's primary passer, prior games only."""
    q = con.execute(f"""
        SELECT "year" AS season,
               CASE WHEN season_type = 'postseason' THEN week + 20 ELSE week END AS week,
               pos_team AS team, passer_player_name AS qb,
               count(*) AS att, sum(yards_gained) AS yds,
               count(*) FILTER (WHERE "int") AS ints,
               count(*) FILTER (WHERE sack) AS sacks,
               count(*) FILTER (WHERE touchdown) AS tds
        FROM read_parquet('{PBP / "pbp_*.parquet"}')
        WHERE "pass" AND passer_player_name IS NOT NULL AND period <= 4
          AND week IS NOT NULL
        GROUP BY 1, 2, 3, 4
    """).fetchdf().sort_values(["team", "season", "week"])
    rows = []
    for (team, season), d in q.groupby(["team", "season"]):
        weeks = sorted(d["week"].unique())
        for i, w in enumerate(weeks):
            prior = d[d["week"].isin(weeks[max(0, i - QB_WINDOW):i])]
            if not prior.empty:
                by_qb = prior.groupby("qb")["att"].sum()
                starter = by_qb.idxmax()
                # starter in the single most recent game vs the one before
                lastw = weeks[i - 1] if i >= 1 else None
                prevw = weeks[i - 2] if i >= 2 else None
                def top(wk):
                    dd = d[d["week"] == wk]
                    return dd.loc[dd["att"].idxmax(), "qb"] if len(dd) else None
                changed = (top(lastw) != top(prevw)
                           if lastw is not None and prevw is not None else False)
                p = prior[prior["qb"] == starter]
                att = p["att"].sum()
                rows.append((season, w, team, att / max(len(p), 1),
                             p["yds"].sum() / max(att, 1),
                             p["ints"].sum() / max(att, 1),
                             p["sacks"].sum() / max(att, 1),
                             p["tds"].sum() / max(att, 1),
                             float(changed), 1.0))
            # add the row for week w even with no prior (flags handle it)
    return pd.DataFrame(rows, columns=["season", "week_c", "team", "qb_att_pg",
                                       "qb_ypa", "qb_int_rate", "qb_sack_rate",
                                       "qb_td_rate", "qb_changed", "qb_known"])


def schedule_context(log: pd.DataFrame, tal: pd.DataFrame) -> pd.DataFrame:
    """Whom the season-to-date stats were compiled against — raw averaging of
    allowed inputs, no fitted rating anywhere.

    Per (game_id, team), over that team's PRIOR games this season:
      sos_talent     mean preseason talent composite of opponents faced
      sos_recruit    mean 4-class recruiting points of opponents faced
      sos_fbs_share  share of those opponents that were FBS

    Point-in-time: talent/recruiting are preseason constants for the season
    being played, and the expanding mean is shifted one game back. A 6.5
    yards-per-play built against cupcakes now looks different from the same
    number built against a murderers' row."""
    t = tal[["season", "team", "talent_composite", "recruit_pts_4c"]].rename(
        columns={"team": "opp", "talent_composite": "opp_talent",
                 "recruit_pts_4c": "opp_recruit"})
    m = log.merge(t, on=["season", "opp"], how="left")
    m["opp_is_fbs"] = (m["opp_div"] == "fbs").astype(float)
    # an unplayed prior game contributed nothing to the stats being
    # contextualised — blank its opponent so the expanding mean skips it
    unplayed = m["pf"].isna() | m["pa"].isna()
    m.loc[unplayed, ["opp_talent", "opp_recruit", "opp_is_fbs"]] = np.nan
    m = m.sort_values(["team", "season", "kick"])
    grp = m.groupby(["team", "season"], sort=False)
    for src, dst in (("opp_talent", "sos_talent"),
                     ("opp_recruit", "sos_recruit"),
                     ("opp_is_fbs", "sos_fbs_share")):
        m[dst] = grp[src].transform(
            lambda s: s.expanding().mean().shift(1))
    return m[["game_id", "team", "sos_talent", "sos_recruit",
              "sos_fbs_share"]]


# ------------------------------------------------- roster & coach raw facts

def roster_features() -> pd.DataFrame:
    r = pd.read_parquet(CFBD / "roster.parquet")
    r = r.rename(columns={"season_pulled": "season"})
    r["name"] = (r["firstName"].fillna("") + " " + r["lastName"].fillna("")).str.strip()
    rows = []
    names_prev: dict = {}
    prev_groups: dict = {}
    for season in sorted(r["season"].unique()):
        d = r[r["season"] == season]
        for team, t in d.groupby("team"):
            names = set(t["name"])
            prev = names_prev.get(team, set())
            ret = len(names & prev) / len(prev) if prev else np.nan
            feat = {"season": season, "team": team, "n_players": len(t),
                    "returning_share": ret}
            yr = pd.to_numeric(t["year"], errors="coerce")
            feat["upper_share"] = float((yr >= 3).mean()) if yr.notna().any() else np.nan
            # position-group continuity + transfer-influx proxy (new players
            # arriving as upperclassmen are overwhelmingly transfers)
            new = names - prev
            if prev:
                feat["new_upper_share"] = (float(
                    ((t["name"].isin(new)) & (yr >= 2)).sum()) / max(len(t), 1))
            for gname2 in ("OL", "SKILL", "DB"):
                cur_g = set(t.loc[t["position"].isin(POS_GROUPS[gname2]),
                                  "name"])
                prev_g = prev_groups.get((team, gname2), set())
                if prev_g:
                    feat[f"{gname2}_ret_share"] = (
                        len(cur_g & prev_g) / len(prev_g))
                prev_groups[(team, gname2)] = cur_g
            for gname, poss in POS_GROUPS.items():
                grp = t[t["position"].isin(poss)]
                feat[f"{gname}_n"] = len(grp)
                feat[f"{gname}_wt"] = float(pd.to_numeric(
                    grp["weight"], errors="coerce").mean()) if len(grp) else np.nan
                feat[f"{gname}_ht"] = float(pd.to_numeric(
                    grp["height"], errors="coerce").mean()) if len(grp) else np.nan
            rows.append(feat)
            names_prev[team] = names
    return pd.DataFrame(rows)



def returning_production() -> pd.DataFrame:
    """Production-weighted roster continuity — raw counting facts, exact
    athlete-id joins (no name matching).

    For team T entering season Y (roster coverage 2004+, so features 2005+):
      retp_pass/rush/recv/tkl  share of T's season-(Y-1) passing / rushing /
                               receiving yards and total tackles produced by
                               players on T's season-Y roster
      retp_qb1                 last season's leading passer is still rostered
      port_pass/rush/recv      log1p of yards arriving on the roster from
                               OTHER programs' season-(Y-1) production
                               (grad transfers, later the portal)

    Point-in-time by construction: the season-Y roster is preseason
    information and every production number is strictly season Y-1. These are
    counting facts about who is on the team — the same footing as recruiting
    points, not a forecast of any game. Headcount `returning_share` already
    exists; this weighs each head by what it actually produced.
    """
    st = load_player_stats(columns=["game_id", "season", "home_away",
                                    "category", "stat_type", "athlete_id",
                                    "stat"])
    kinds = {("passing", "YDS"): "pass", ("rushing", "YDS"): "rush",
             ("receiving", "YDS"): "recv", ("defensive", "TOT"): "tkl"}
    st = st[st[["category", "stat_type"]].apply(tuple, axis=1).isin(kinds)]
    st["kind"] = [kinds[(c, t)] for c, t in zip(st["category"], st["stat_type"])]
    st["v"] = pd.to_numeric(st["stat"], errors="coerce").fillna(0.0)
    st["athlete_id"] = st["athlete_id"].astype(str)
    st = st[st["athlete_id"].str.len() > 2]          # drop unattributed rows

    gm = pd.read_parquet(PQ / "games.parquet",
                         columns=["game_id", "home_team", "away_team"])
    st = st.merge(gm, on="game_id", how="inner")
    st["team"] = np.where(st["home_away"] == "home",
                          st["home_team"], st["away_team"])
    # production is keyed by the GAMES' team names, rosters by CFBD's current
    # ones; a program renamed between seasons ('Appalachian State' 2013 ->
    # 'App State' 2014) would read as returning nobody and importing its own
    # players. Join in canonical name space; emit under original names below.
    from cfbpred.pull_espn_roster import team_alias_map
    canon = team_alias_map()
    st["team"] = st["team"].map(lambda t: canon.get(t, t))
    ath = (st.groupby(["season", "team", "athlete_id", "kind"],
                      as_index=False)["v"].sum())

    ros = pd.read_parquet(CFBD / "roster.parquet",
                          columns=["id", "team", "season_pulled"])
    ros = (ros.rename(columns={"season_pulled": "season",
                               "id": "athlete_id"})
           .dropna(subset=["athlete_id"]))
    ros["athlete_id"] = ros["athlete_id"].astype(str)
    name_map = ros[["season", "team"]].drop_duplicates()
    name_map["team_c"] = name_map["team"].map(lambda t: canon.get(t, t))
    ros["team"] = ros["team"].map(lambda t: canon.get(t, t))
    ros = ros.drop_duplicates(["season", "team", "athlete_id"])

    # everything below joins season-(Y-1) production to the season-Y roster
    prev = ath.copy()
    prev["season"] = prev["season"] + 1

    team_tot = (prev.pivot_table(index=["season", "team"], columns="kind",
                                 values="v", aggfunc="sum")
                .add_prefix("tot_").reset_index())
    ret = (ros.merge(prev, on=["season", "team", "athlete_id"], how="inner")
           .pivot_table(index=["season", "team"], columns="kind",
                        values="v", aggfunc="sum")
           .add_prefix("ret_").reset_index())

    # production each rostered player logged ANYWHERE last season, minus what
    # they logged here = what the roster imported from other programs
    tot_by_ath = (prev.groupby(["season", "athlete_id", "kind"],
                               as_index=False)["v"].sum()
                  .rename(columns={"v": "v_all"}))
    port = ros.merge(tot_by_ath, on=["season", "athlete_id"], how="inner")
    port = port.merge(prev.rename(columns={"v": "v_here"}),
                      on=["season", "team", "athlete_id", "kind"], how="left")
    port["v_in"] = (port["v_all"] - port["v_here"].fillna(0.0)).clip(lower=0)
    port = (port[port["kind"] != "tkl"]
            .pivot_table(index=["season", "team"], columns="kind",
                         values="v_in", aggfunc="sum")
            .add_prefix("in_").reset_index())

    # last season's leading passer, where the team actually threw the ball
    # (athlete_id settles exact-yardage ties deterministically — feature
    # values must not depend on row order)
    qb = prev[prev["kind"] == "pass"].sort_values(["v", "athlete_id"])
    qb1 = qb.groupby(["season", "team"]).tail(1)
    qb1 = qb1[qb1["v"] >= 500][["season", "team", "athlete_id"]]
    qb1 = qb1.merge(ros.assign(back=1.0),
                    on=["season", "team", "athlete_id"], how="left")
    qb1 = qb1[["season", "team", "back"]].rename(columns={"back": "retp_qb1"})
    qb1["retp_qb1"] = qb1["retp_qb1"].fillna(0.0)

    out = team_tot.merge(ret, on=["season", "team"], how="left")
    # a team with no roster pulled for the season would read as "returned
    # nobody" (share 0.0); restrict to team-seasons the roster feed actually
    # covers so absence stays unknown, not zero. Matters for FCS programs in
    # the current season, whose rosters publish later than FBS ones.
    covered = ros[["season", "team"]].drop_duplicates()
    out = out.merge(covered.assign(_ros=1.0), on=["season", "team"], how="left")
    out = out[out.pop("_ros").notna()].reset_index(drop=True)
    MIN_TOT = {"pass": 500.0, "rush": 500.0, "recv": 500.0, "tkl": 300.0}
    for k, floor in MIN_TOT.items():
        tot = out.get(f"tot_{k}")
        if tot is None:
            continue
        # a roster that returned nobody has share 0.0 — that is a fact, not
        # a missing value; only a sub-floor denominator means "unknown"
        rv = out[f"ret_{k}"].fillna(0.0) if f"ret_{k}" in out else 0.0
        out[f"retp_{k}"] = (rv / tot).where(tot >= floor).clip(0.0, 1.0)
    out = out[["season", "team"]
              + [c for c in out.columns if c.startswith("retp_")]]
    out = out.merge(port, on=["season", "team"], how="left")
    for k in ("pass", "rush", "recv"):
        if f"in_{k}" in out.columns:
            out[f"port_{k}"] = np.log1p(out.pop(f"in_{k}").fillna(0.0))
    out = out.merge(qb1, on=["season", "team"], how="left")
    # roster exists but nobody threw 500 yds last year -> unknown, not 0
    # back to original names: one row per name the games actually use, so the
    # per-side attach in main() joins for variant-named seasons too
    out = (name_map.merge(out.rename(columns={"team": "team_c"}),
                          on=["season", "team_c"], how="inner")
           .drop(columns=["team_c"]))
    return out


def talent_features() -> pd.DataFrame:
    """Human scouting priors for the roster itself — allowed by user decision
    (2026-08-26): recruiting stars and the 247 talent composite are
    assessments of PLAYERS, not forecasts of games. Outcome predictions
    (lines, FPI, Elo, SP+, win probabilities) remain banned.

    This is the one signal a production-history model lacks: a five-star
    true freshman and a walk-on both have zero career games. Point-in-time
    by construction — the talent composite is published preseason for that
    season's roster (2015+), and a recruiting class is fixed on signing day
    before it ever plays (2002+); a season sees only classes <= itself.
    """
    tal = pd.read_parquet(CFBD / "talent.parquet").rename(
        columns={"year": "season", "talent": "talent_composite"})[
        ["season", "team", "talent_composite"]]   # bookkeeping cols stay out
    rec = pd.read_parquet(CFBD / "recruiting_teams.parquet").rename(
        columns={"year": "cls"})
    rows = []
    for season in range(2001, CURRENT_SEASON + 1):
        w = rec[(rec.cls <= season) & (rec.cls >= season - 3)]
        if not len(w):
            continue
        a = (w.groupby("team")
             .agg(recruit_pts_4c=("points", "mean"),
                  recruit_rank_best=("rank", "min"),
                  recruit_classes=("cls", "nunique")).reset_index())
        latest = (w.sort_values("cls").groupby("team").tail(1)
                  [["team", "points"]]
                  .rename(columns={"points": "recruit_pts_latest"}))
        a = a.merge(latest, on="team", how="left")
        a["season"] = season
        rows.append(a)
    out = pd.concat(rows, ignore_index=True)
    out = out.merge(tal, on=["season", "team"], how="outer")
    # the /talent feed ships byte-identical duplicate rows for a few
    # team-seasons; a left-join fan-out would duplicate whole games
    return out.drop_duplicates(["season", "team"]).reset_index(drop=True)


def coach_features() -> pd.DataFrame:
    c = pd.read_parquet(CFBD / "coaches.parquet")
    rows = []
    for _, coach in c.iterrows():
        seasons = coach["seasons"]
        if seasons is None or len(seasons) == 0:
            continue
        seasons = sorted(seasons, key=lambda s: s["year"])
        for s in seasons:
            prior = [x for x in seasons if x["year"] < s["year"]]
            at_school = [x for x in prior if x["school"] == s["school"]]
            g = sum(x["games"] for x in prior)
            w = sum(x["wins"] for x in prior)
            rows.append({"season": s["year"], "team": s["school"],
                         "coach_tenure": len(at_school),
                         "coach_first_year": float(len(at_school) == 0),
                         "coach_career_games": g,
                         "coach_career_winpct": w / g if g else np.nan})
    df = pd.DataFrame(rows)
    # a team occasionally has two entries (midseason change) — keep the more
    # experienced one, a deterministic tie-break
    return (df.sort_values("coach_career_games")
            .drop_duplicates(["season", "team"], keep="last"))


# ------------------------------------------------------------------- main

def main() -> None:
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    g = load_games(con)                       # FBS-involved: the modeling rows
    g_all = load_games(con, fbs_only=False)   # every game, for logs/records
    # Completed FCS-vs-FCS games join the emit population as extra pregame
    # training targets (production default since v20 — measured 2026-08-27:
    # -0.005 val logloss overall, -0.011 on FBS-FCS, every pilot seed beat
    # every control seed). The rows are feature-sparse (no pbp/box/talent),
    # so pretrain samples them at 25% and validation stays FBS-involved.
    # CFB_NO_FCS_TARGETS=1 reverts to the FBS-involved-only population.
    if os.environ.get("CFB_NO_FCS_TARGETS") != "1":
        # completed FCS-vs-FCS games are training targets; the current
        # season, ANY game with a Division I side rides along as a
        # prediction target so the whole slate is visible (an FCS team
        # hosting a D2 school still gets priced — bluntly, and flagged by
        # the roster-gap chip when the lower side is a blank)
        fcs_fcs = ((g_all["home_division"] == "fcs")
                   & (g_all["away_division"] == "fcs"))
        d1_any = ((g_all["home_division"] == "fcs")
                  | (g_all["away_division"] == "fcs"))
        extra = g_all[(fcs_fcs & g_all["margin"].notna())
                      | (d1_any & (g_all["season"] == CURRENT_SEASON))]
        g = (pd.concat([g, extra], ignore_index=True)
             .drop_duplicates("game_id")
             .sort_values(["season", "kick"]).reset_index(drop=True))
        print(f"FCS-vs-FCS targets: +{len(extra):,} lower-division rows")
    fbs_teams = sorted(set(g.loc[g.home_division == "fbs", "home_team"])
                       | set(g.loc[g.away_division == "fbs", "away_team"]))
    team_idx = {t: i + 1 for i, t in enumerate(fbs_teams)}      # 0 = UNK
    # Every other program with a real footprint gets its own embedding —
    # a single shared FCS token made North Dakota State indistinguishable
    # from a winless bottom-feeder, and blowout pricing paid for it.
    real_appearances = g_all[g_all.game_id.gt(0)]
    appear = pd.concat([real_appearances["home_team"], real_appearances["away_team"]]).value_counts()
    for t in sorted(t for t, n in appear.items()
                    if n >= 4 and t not in team_idx):
        team_idx[t] = len(team_idx) + 1
    # conference vocab is part of the frozen model contract: build it from
    # the TRAINING population only (FBS-involved + FCS-vs-FCS). Slate-only
    # extras (FCS hosting D2/D3) keep their emit rows, but their lower-side
    # conferences map to the UNK index instead of growing the embedding.
    vocab_g = g[(g["home_division"] == "fbs") | (g["away_division"] == "fbs")
                | ((g["home_division"] == "fcs")
                   & (g["away_division"] == "fcs"))]
    confs = sorted(set(vocab_g["home_conference"].dropna())
                   | set(vocab_g["away_conference"].dropna()))
    conf_idx = {c: i + 1 for i, c in enumerate(confs)}
    (FEAT / "team_vocab.json").write_text(json.dumps(
        {"team": team_idx, "conf": conf_idx, "n_hist": N_HIST}))
    print(f"{len(g):,} FBS-involved games of {len(g_all):,} total, "
          f"{len(fbs_teams)} FBS + {len(team_idx) - len(fbs_teams)} other "
          f"programs, {len(confs)} conferences")

    log = team_log(g_all, keep_teams=set(team_idx))
    tal = talent_features()
    otal = {(int(r.season), r.team): float(r.talent_composite)
            for r in tal.itertuples()
            if pd.notna(r.talent_composite)}
    hist = build_history(log, team_idx, emit_gids=set(g["game_id"]),
                         opp_talent=otal)
    hist.to_parquet(FEAT / "game_history.parquet", index=False,
                    compression="zstd")
    print(f"history: {len(hist):,} slot rows "
          f"({hist.game_id.nunique():,} games covered)")

    # season-to-date raw record from the game log, strictly prior PLAYED
    # games. A scheduled game is not a result: unplayed rows count zero toward
    # gp/wins/points, so a preseason 2026 row looks exactly like any
    # historical first-game row (gp 0, winpct 0, ppg null).
    log = log.sort_values(["team", "season", "kick"])
    played = log["pf"].notna() & log["pa"].notna()
    log["_gp1"] = played.astype(float)
    log["_win1"] = ((log["pf"] > log["pa"]) & played).astype(float)
    log["_pf1"] = log["pf"].where(played, 0.0)
    log["_pa1"] = log["pa"].where(played, 0.0)
    grp = log.groupby(["team", "season"])
    log["gp"] = grp["_gp1"].transform(lambda s: s.cumsum().shift(1)).fillna(0)
    log["wins_td"] = grp["_win1"].transform(
        lambda s: s.cumsum().shift(1)).fillna(0)
    log["pf_td"] = grp["_pf1"].transform(lambda s: s.cumsum().shift(1))
    log["pa_td"] = grp["_pa1"].transform(lambda s: s.cumsum().shift(1))
    # no played prior games -> points-to-date are unknown, not zero (matches
    # the historical week-1 convention the model was trained on)
    log.loc[log["gp"] == 0, ["pf_td", "pa_td"]] = np.nan
    log = log.drop(columns=["_gp1", "_win1", "_pf1", "_pa1"])
    rec = log[["game_id", "team", "gp", "wins_td", "pf_td", "pa_td"]].copy()
    rec["ppg_for"] = rec["pf_td"] / rec["gp"].clip(lower=1)
    rec["ppg_against"] = rec["pa_td"] / rec["gp"].clip(lower=1)
    rec["win_pct_td"] = rec["wins_td"] / rec["gp"].clip(lower=1)

    from cfbpred.game_feature_snapshots import pbp_snapshots, player_snapshots
    # Every target, including later weeks and byes, sees the same available
    # observations. No completed-week join can drop an upcoming game.
    off, dfn, qb = pbp_snapshots(g, con, PBP / "pbp_*.parquet",
                                CFBD / f"pregame_plays_{CURRENT_SEASON}.parquet")
    ros = roster_features()
    coa = coach_features()
    rp = returning_production()
    sos = schedule_context(log, tal)     # whom the stats were built against
    # NOTE: aggregating these from g_all (FCS slates included) was measured
    # 2026-08-27 alongside the 20-game history — null on val and test — and
    # reverted to keep the store identical to the proven v16 configuration.
    raw_players = load_player_stats()
    box = build_box_features(g, raw=raw_players, per_game=True).drop(
        columns=["season", "week_c"])
    player = player_snapshots(g, raw_players)
    av = player[["game_id", "team"] + [c for c in player if c.startswith("av_")]]
    qbx = player[["game_id", "team"] + [c for c in player if c.startswith("cx_")]]
    print(f"pbp rates: {len(off):,} team-game snapshots | qb: {len(qb):,} | "
          f"box: {len(box):,} | roster: {len(ros):,} | coach: {len(coa):,}")

    # venue geometry + physical attributes (raw facts)
    ven = con.execute(f"""SELECT id AS venue_id, latitude, longitude, elevation,
               CAST(dome AS INT) AS wx_indoor_venue,
               CAST(grass AS INT) AS venue_grass, capacity
        FROM read_parquet('{CFBD / "venues.parquet"}')""").fetchdf()
    ven["venue_capacity"] = np.log1p(pd.to_numeric(ven.pop("capacity"),
                                                   errors="coerce"))
    ven["elevation"] = pd.to_numeric(ven["elevation"], errors="coerce")
    g = g.merge(ven, on="venue_id", how="left")
    home_base = build_home_base(g_all, ven, set(team_idx))

    # ---- rest: days since that team's PREVIOUS game, capped ----
    # Grouping the diff by (team, season) left every season opener null, and
    # a null standardises to the training mean — so the model was told each
    # opener was an ordinary ~8-day turnaround. The diff now runs across the
    # season boundary and saturates at REST_CAP: real football rest (short
    # week 5-6, normal 7, bye 13-14) survives intact, while the offseason
    # (min 222 days observed) and bowl layoffs pin to the cap. The cap is
    # what keeps the week-to-week signal legible: without it the offseason
    # spike inflates the column sd from 3.0 to 10.5 days and a bye week
    # collapses from +1.1 sd to +0.3. Postseason layoff stays available to
    # the net through is_postseason/week_scaled. A team's first-ever
    # appearance has no previous game and stays null.
    log2 = log.sort_values(["team", "kick"])
    log2["rest"] = (log2.groupby("team")["kick"]
                    .diff().dt.total_seconds() / 86400.0).clip(upper=REST_CAP)

    # ---------------- assemble one row per game with h_/a_ blocks ----------
    out = g[["game_id", "season", "week", "week_c", "kick", "neutral_site",
             "conference_game", "home_team", "away_team",
             "home_conference", "away_conference",
             "home_division", "away_division",
             "home_points", "away_points", "margin", "total",
             "latitude", "longitude", "elevation",
             "wx_indoor_venue", "venue_grass", "venue_capacity"]].copy()

    # ---- weather: measured conditions, raw physical data ----
    wx = con.execute(f"""SELECT id AS game_id, temperature AS wx_temp,
               windSpeed AS wx_wind, precipitation AS wx_precip,
               snowfall AS wx_snow, humidity AS wx_humidity,
               CAST(gameIndoors AS INT) AS wx_indoor
        FROM read_parquet('{CFBD / "weather.parquet"}')""").fetchdf()
    wx = wx.drop_duplicates("game_id")
    out = out.merge(wx, on="game_id", how="left")

    # ---- kickoff body clock: venue solar time vs each side's home base ----
    # Longitude/15 is solar-hour offset — deterministic geography, no lookup
    # tables. A 12:00 ET kick is 09:00 body time for a coast-to-coast visitor.
    kick_utc_hour = (g["kick"].dt.hour + g["kick"].dt.minute / 60.0)
    out["kick_local_hour"] = ((kick_utc_hour
                               + out["longitude"].astype(float) / 15.0) % 24.0)

    def attach(side: str) -> None:
        t = f"{side}_team"
        pre = "h_" if side == "home" else "a_"
        m = rec.rename(columns={"team": t})
        nonlocal out
        out = out.merge(m[["game_id", t, "gp", "win_pct_td", "ppg_for",
                           "ppg_against"]].rename(columns={
            "gp": pre + "gp", "win_pct_td": pre + "winpct",
            "ppg_for": pre + "ppg_for", "ppg_against": pre + "ppg_against"}),
            on=["game_id", t], how="left")
        s2 = sos.rename(columns={"team": t})
        out = out.merge(s2.rename(columns={c: pre + c for c in (
            "sos_talent", "sos_recruit", "sos_fbs_share")}),
            on=["game_id", t], how="left")
        for tbl, keys in ((off, ["game_id"]), (dfn, ["game_id"]),
                          (qb, ["game_id"]), (box, ["game_id"]),
                          (av, ["game_id"]),
                          (qbx, ["game_id"]),
                          (ros, ["season"]), (coa, ["season"]),
                          (tal, ["season"]), (rp, ["season"])):
            mm = tbl.rename(columns={"team": t})
            ren = {c: pre + c for c in mm.columns
                   if c not in keys + [t]}
            out = out.merge(mm.rename(columns=ren), on=keys + [t], how="left",
                            validate="many_to_one")
        b = home_base.rename(columns={"team": t, "b_lat": pre + "b_lat",
                                      "b_lon": pre + "b_lon",
                                      "b_elev": pre + "b_elev"})
        out = out.merge(b, on=["season", t], how="left")
        r2 = log2.rename(columns={"team": t})[["game_id", t, "rest"]]
        out = out.merge(r2.rename(columns={"rest": pre + "rest"}),
                        on=["game_id", t], how="left")
        out[pre + "travel"] = earth_miles(
            out[pre + "b_lat"].to_numpy(float), out[pre + "b_lon"].to_numpy(float),
            out["latitude"].to_numpy(float), out["longitude"].to_numpy(float))
        out[pre + "alt_delta"] = (out["elevation"].to_numpy(float)
                                  - out[pre + "b_elev"].to_numpy(float))

    attach("home")
    attach("away")

    # ---- head-to-head: this exact pairing's history ----
    gg = g[["game_id", "kick", "home_team", "away_team", "margin"]].copy()
    gg["pair"] = [tuple(sorted(x)) for x in
                  zip(gg["home_team"], gg["away_team"])]
    gg = gg.sort_values("kick")
    h2h_margin, h2h_days, h2h_n5 = {}, {}, {}
    hist_by_pair: dict = {}
    for r in gg.itertuples():
        past = hist_by_pair.get(r.pair, [])
        recent = [x for x in past
                  if (r.kick - x[0]).days <= 5 * 365]
        h2h_n5[r.game_id] = float(len(recent))
        if past:
            k_last, home_last, m_last = past[-1]
            h2h_days[r.game_id] = min((r.kick - k_last).days, 2000.0)
            h2h_margin[r.game_id] = (m_last if home_last == r.home_team
                                     else -m_last)
        if pd.notna(r.margin):        # only a played meeting is history
            hist_by_pair.setdefault(r.pair, []).append(
                (r.kick, r.home_team, r.margin))
    out["h2h_last_margin"] = out["game_id"].map(h2h_margin)
    out["h2h_days_since"] = out["game_id"].map(h2h_days)
    out["h2h_meetings_5y"] = out["game_id"].map(h2h_n5).fillna(0)

    out["h_idx"] = out["home_team"].map(team_idx).fillna(0).astype(int)
    out["a_idx"] = out["away_team"].map(team_idx).fillna(0).astype(int)
    out["h_conf"] = out["home_conference"].map(conf_idx).fillna(0).astype(int)
    out["a_conf"] = out["away_conference"].map(conf_idx).fillna(0).astype(int)
    for side, pre in (("home", "h_"), ("away", "a_")):
        out[pre + "body_clock_shift"] = (
            (out["longitude"].astype(float)
             - out[pre + "b_lon"].astype(float)) / 15.0)
    out = out.drop(columns=["latitude", "longitude", "elevation",
                            "h_b_lat", "h_b_lon", "h_b_elev",
                            "a_b_lat", "a_b_lon", "a_b_elev", "kick"])
    out.to_parquet(FEAT / "game_scalars.parquet", index=False,
                   compression="zstd")
    ncol = out.shape[1]
    print(f"game scalars: {len(out):,} rows x {ncol} cols")

    n_slot_fields = 8
    total_inputs = (2 * N_HIST * (n_slot_fields + 24)   # history + opp embedding
                    + (ncol - 18)                       # scalar features
                    + 2 * 24 + 2 * 8                    # team + conf embeddings
                    + 15)                               # in-game state
    print(f"input width per prediction: ~{total_inputs} "
          f"(history {2 * N_HIST * (n_slot_fields + 24)}, "
          f"scalars {ncol - 18}, embeddings {2 * 24 + 2 * 8}, state 15)")


if __name__ == "__main__":
    main()
