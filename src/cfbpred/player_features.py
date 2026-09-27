"""Turn raw per-game player stat lines into point-in-time team features.

Every number here is a count or a ratio of counts over official box scores —
season-to-date as of each game, strictly prior games only (cumulative-then-
shift, same discipline as the pbp rates). Positions covered:

  passing      completions, attempts, yards, TD, INT (team QBs)
  rushing      carries, yards, TD, plus top-back concentration
  receiving    receptions, yards, TD, plus top-target concentration
  defensive    tackles, TFL, sacks, passes defended, QB hurries, def TD
  kicking      FG made/att, XP made/att
  punting      punts, gross average, inside-20 share
  returns      kick/punt return yards per attempt, return TD

Concentration features (share of team rushing yards held by the top back, share
of receiving yards held by the top target) are raw ratios, and they carry real
information the totals miss: a one-back offense losing that back is not the
same team.

The 'QBR' stat type was already excluded at ingestion — it is ESPN's model.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.config import CFBD  # noqa: E402


def load_player_stats(columns: list | None = None) -> pd.DataFrame:
    """The official stat lines, plus — behind CFB_FCS_STATS=1 — the FCS-slate
    lines pulled by pull_fcs_stats.py (CFBD classification=fcs 2021+, ESPN
    box scores before that). Kept in a separate file and behind a flag so the
    frozen production model's feature store stays byte-stable until a retrain
    measures and adopts them."""
    df = pd.read_parquet(CFBD / "player_game_stats.parquet", columns=columns)
    extra = CFBD / "player_game_stats_fcs.parquet"
    if os.environ.get("CFB_FCS_STATS") == "1" and extra.exists():
        df = pd.concat([df, pd.read_parquet(extra, columns=columns)],
                       ignore_index=True)
    return df


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _frac_pair(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'19/30' -> (19, 30). Used by C/ATT, FG, XP."""
    parts = s.astype(str).str.extract(r"^(\d+)[/-](\d+)$")
    return _num(parts[0]), _num(parts[1])


def team_game_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, home_away) with raw category totals."""
    out = []

    def team_sum(cat, stat, name, frac=None):
        d = raw[(raw.category == cat) & (raw.stat_type == stat)]
        if frac is None:
            s = (d.assign(v=_num(d["stat"]))
                 .groupby(["game_id", "home_away"])["v"].sum().rename(name))
            out.append(s)
        else:
            a, b = _frac_pair(d["stat"])
            g = d.assign(va=a, vb=b).groupby(["game_id", "home_away"])
            out.append(g["va"].sum().rename(f"{name}_made"))
            out.append(g["vb"].sum().rename(f"{name}_att"))

    def top_share(cat, stat, name):
        """Top athlete's share of the team total — a raw concentration ratio."""
        d = raw[(raw.category == cat) & (raw.stat_type == stat)].copy()
        d["v"] = _num(d["stat"]).clip(lower=0)
        g = d.groupby(["game_id", "home_away"])["v"]
        tot, mx = g.sum(), g.max()
        out.append((mx / tot.replace(0, np.nan)).rename(name))

    team_sum("passing", "C/ATT", "pass_catt", frac=True)
    team_sum("passing", "YDS", "pass_yds")
    team_sum("passing", "TD", "pass_td")
    team_sum("passing", "INT", "pass_int")
    team_sum("rushing", "CAR", "rush_car")
    team_sum("rushing", "YDS", "rush_yds")
    team_sum("rushing", "TD", "rush_td")
    top_share("rushing", "YDS", "rush_top_share")
    team_sum("receiving", "REC", "recv_rec")
    team_sum("receiving", "YDS", "recv_yds")
    team_sum("receiving", "TD", "recv_td")
    top_share("receiving", "YDS", "recv_top_share")
    team_sum("defensive", "TOT", "def_tackles")
    team_sum("defensive", "TFL", "def_tfl")
    team_sum("defensive", "SACKS", "def_sacks")
    team_sum("defensive", "PD", "def_pd")
    team_sum("defensive", "QB HUR", "def_hur")
    team_sum("defensive", "TD", "def_td")
    team_sum("kicking", "FG", "kick_fg", frac=True)
    team_sum("kicking", "XP", "kick_xp", frac=True)
    team_sum("punting", "NO", "punt_no")
    team_sum("punting", "YDS", "punt_yds")
    team_sum("punting", "In 20", "punt_in20")
    team_sum("kickReturns", "NO", "kr_no")
    team_sum("kickReturns", "YDS", "kr_yds")
    team_sum("kickReturns", "TD", "kr_td")
    team_sum("puntReturns", "NO", "pr_no")
    team_sum("puntReturns", "YDS", "pr_yds")
    team_sum("puntReturns", "TD", "pr_td")
    team_sum("interceptions", "INT", "def_int")
    team_sum("fumbles", "LOST", "fum_lost")

    df = pd.concat(out, axis=1).reset_index()
    return df


STAT_COLS = ["pass_catt_made", "pass_catt_att", "pass_yds", "pass_td",
             "pass_int", "rush_car", "rush_yds", "rush_td",
             "recv_rec", "recv_yds", "recv_td",
             "def_tackles", "def_tfl", "def_sacks", "def_pd", "def_hur",
             "def_td", "def_int",
             "kick_fg_made", "kick_fg_att", "kick_xp_made", "kick_xp_att",
             "punt_no", "punt_yds", "punt_in20",
             "kr_no", "kr_yds", "kr_td", "pr_no", "pr_yds", "pr_td",
             "fum_lost"]
SHARE_COLS = ["rush_top_share", "recv_top_share"]


def build_box_features(games: pd.DataFrame, *, raw: pd.DataFrame | None = None,
                       targets: pd.DataFrame | None = None,
                       per_game: bool = False) -> pd.DataFrame:
    """Per (season, week_c, team): season-to-date box-score rates, prior games
    only. `games` supplies the game_id -> (season, week_c, team-name) mapping so
    team naming matches the rest of the feature store exactly."""
    raw = load_player_stats() if raw is None else raw
    tg = team_game_stats(raw)

    side_map = pd.concat([
        games[["game_id", "season", "week_c", "home_team"]]
        .assign(home_away="home").rename(columns={"home_team": "team"}),
        games[["game_id", "season", "week_c", "away_team"]]
        .assign(home_away="away").rename(columns={"away_team": "team"}),
    ])
    if per_game:
        from cfbpred.game_feature_snapshots import game_sides
        side_map = game_sides(games)
        tg["_has_box"] = True
        tg = side_map.merge(tg, on=["game_id", "home_away"], how="left",
                            validate="one_to_one")
        tg["_has_box"] = tg["_has_box"].eq(True) & tg["completed"]
    else:
        tg = tg.merge(side_map, on=["game_id", "home_away"], how="inner")
    # Ranking snapshots can request a read after the last known result,
    # including teams on a bye. Targets carry no stats and cannot add results.
    if targets is not None:
        tg = pd.concat([tg, targets], ignore_index=True)
    for c in STAT_COLS + SHARE_COLS:
        if c not in tg.columns:
            tg[c] = np.nan

    tg = tg.sort_values(["team", "season", "kick" if per_game else "week_c", "game_id"])
    if per_game:
        tg.loc[~tg["_has_box"], STAT_COLS + SHARE_COLS] = np.nan
    grp = tg.groupby(["team", "season"])
    cum = (tg[STAT_COLS].fillna(0).groupby([tg.team, tg.season]).cumsum().shift(1)
           if per_game else grp[STAT_COLS].cumsum().shift(1))
    # a shifted cumsum crosses team boundaries at each first row; blank those
    first = grp.cumcount() == 0
    cum[first.to_numpy()] = np.nan
    if per_game:
        observed = tg[STAT_COLS].notna().groupby([tg.team, tg.season]).cumsum().shift(1)
        observed[first.to_numpy()] = 0
        # A missing field is not a measured zero. Carry known cumulative
        # values across targets, but retain unknowns when no source has it.
        cum = cum.where(observed > 0)
    games_played = (grp["_has_box"].transform(lambda s: s.cumsum().shift(1)).fillna(0)
                    if per_game else grp.cumcount().astype(float))

    eps = 1.0
    f = pd.DataFrame({"team": tg["team"], "season": tg["season"],
                      "week_c": tg["week_c"], "box_gp": games_played})
    if per_game:
        f["game_id"] = tg["game_id"]
        cum.loc[games_played == 0] = np.nan
    gp = games_played.replace(0, np.nan)
    f["bx_comp_pct"] = cum["pass_catt_made"] / (cum["pass_catt_att"] + eps)
    f["bx_pass_ypg"] = cum["pass_yds"] / gp
    f["bx_pass_td_pg"] = cum["pass_td"] / gp
    f["bx_pass_int_pg"] = cum["pass_int"] / gp
    f["bx_rush_ypc"] = cum["rush_yds"] / (cum["rush_car"] + eps)
    f["bx_rush_ypg"] = cum["rush_yds"] / gp
    f["bx_rush_td_pg"] = cum["rush_td"] / gp
    f["bx_recv_ypr"] = cum["recv_yds"] / (cum["recv_rec"] + eps)
    f["bx_tackles_pg"] = cum["def_tackles"] / gp
    f["bx_tfl_pg"] = cum["def_tfl"] / gp
    f["bx_sacks_pg"] = cum["def_sacks"] / gp
    f["bx_pd_pg"] = cum["def_pd"] / gp
    f["bx_hur_pg"] = cum["def_hur"] / gp
    f["bx_def_td_pg"] = (cum["def_td"] + cum["def_int"] * 0) / gp
    f["bx_int_pg"] = cum["def_int"] / gp
    f["bx_fg_pct"] = cum["kick_fg_made"] / (cum["kick_fg_att"] + eps)
    f["bx_fg_att_pg"] = cum["kick_fg_att"] / gp
    f["bx_xp_pct"] = cum["kick_xp_made"] / (cum["kick_xp_att"] + eps)
    f["bx_punt_avg"] = cum["punt_yds"] / (cum["punt_no"] + eps)
    f["bx_punt_in20_share"] = cum["punt_in20"] / (cum["punt_no"] + eps)
    f["bx_kr_avg"] = cum["kr_yds"] / (cum["kr_no"] + eps)
    f["bx_pr_avg"] = cum["pr_yds"] / (cum["pr_no"] + eps)
    f["bx_ret_td_pg"] = (cum["kr_td"] + cum["pr_td"]) / gp
    f["bx_fum_lost_pg"] = cum["fum_lost"] / gp

    # concentration: expanding mean of prior games' top-player share
    for c in SHARE_COLS:
        f[f"bx_{c}"] = (grp[c].apply(
            lambda s: s.expanding().mean().shift(1))
            .reset_index(level=[0, 1], drop=True))

    # a team can appear twice in one week (Week 0); keep the later appearance
    f = f.sort_values(["team", "season", "week_c", "box_gp"])
    if not per_game:
        f = f.drop_duplicates(["team", "season", "week_c"], keep="last")
    return f


if __name__ == "__main__":
    import duckdb
    from cfbpred.config import PQ
    con = duckdb.connect()
    g = con.execute(f"""
        SELECT game_id, season, week,
               CASE WHEN season_type = 'postseason' THEN week + 20
                    ELSE week END AS week_c,
               home_team, away_team
        FROM read_parquet('{PQ / "games.parquet"}')
        WHERE week IS NOT NULL""").fetchdf()
    f = build_box_features(g)
    print(f"{len(f):,} team-week rows, seasons "
          f"{f.season.min()}-{f.season.max()}")
    cov = f.groupby("season")["bx_pass_ypg"].apply(lambda s: s.notna().mean())
    print("coverage of bx_pass_ypg by season:")
    for y, v in cov.items():
        print(f"  {y}: {v:.0%}")


# ------------------------------------------------------------- participation

def _athlete_presence(games: pd.DataFrame, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (game_id, team, athlete) that recorded ANY stat, with a raw
    production number (passing + rushing + receiving yards)."""
    raw = load_player_stats() if raw is None else raw
    side_map = pd.concat([
        games[["game_id", "season", "week_c", "home_team"]]
        .assign(home_away="home").rename(columns={"home_team": "team"}),
        games[["game_id", "season", "week_c", "away_team"]]
        .assign(home_away="away").rename(columns={"away_team": "team"}),
    ])
    raw = raw.merge(side_map, on=["game_id", "home_away"], how="inner",
                    suffixes=("", "_g"))
    yds = raw[(raw.stat_type == "YDS")
              & (raw.category.isin(["passing", "rushing", "receiving"]))].copy()
    yds["v"] = _num(yds["stat"]).fillna(0)
    prod = (yds.groupby(["game_id", "season", "week_c", "team",
                         "category", "athlete"])["v"].sum().reset_index())
    return prod


def availability_features(games: pd.DataFrame, *, raw: pd.DataFrame | None = None,
                          target_season: int | None = None,
                          target_week: int | None = None) -> pd.DataFrame:
    """Pregame star availability, prior games only.

    For each (team, game): identify the team's top producers over games
    strictly before the PREVIOUS game, then check whether they appeared in
    that previous game's box score. "The starting RB missed last week" is a
    real, pregame-knowable fact; current-game participation is not (a box
    score only records players who got stats, which encodes game script).
    """
    prod = _athlete_presence(games, raw)
    rows = []
    for (team, season), d in prod.groupby(["team", "season"]):
        weeks = sorted(d["week_c"].unique())
        if season == target_season and target_week is not None:
            if target_week <= max(weeks):
                raise ValueError("Snapshot target must follow all source weeks")
            weeks.append(target_week)
        cum: dict = {}
        prev_players: set = set()
        prev_tops: dict = {}
        for i, w in enumerate(weeks):
            if i >= 1 and prev_tops:
                played = {c: (1.0 if prev_tops[c] in prev_players else 0.0)
                          for c in prev_tops}
                top5 = sorted(cum.items(), key=lambda kv: -kv[1])[:5]
                share = (np.mean([1.0 if a in prev_players else 0.0
                                  for a, _ in top5]) if top5 else np.nan)
                # production-weighted availability: how much of the season's
                # output belonged to players who appeared in the last game —
                # a persistence signal for absences already visible in a box
                # score (announced-but-unplayed absences stay invisible)
                tot = sum(cum.values())
                prod_share = (sum(v for a, v in cum.items()
                                  if a in prev_players) / tot
                              if tot > 0 else np.nan)
                rows.append((season, int(w), team,
                             played.get("passing", np.nan),
                             played.get("rushing", np.nan),
                             played.get("receiving", np.nan),
                             share, prod_share, 1.0))
            # update state: tops as of BEFORE the game at week w, then presence
            this = d[d["week_c"] == w]
            prev_tops = {}
            for cat, dd in d[d["week_c"] < w].groupby("category"):
                s = dd.groupby("athlete")["v"].sum()
                if len(s):
                    prev_tops[cat] = s.idxmax()
            for _, r in this.iterrows():
                cum[r["athlete"]] = cum.get(r["athlete"], 0.0) + r["v"]
            prev_players = set(this["athlete"])
    return pd.DataFrame(rows, columns=[
        "season", "week_c", "team", "av_qb_played_last", "av_rb_played_last",
        "av_wr_played_last", "av_top5_played_last", "av_prod_share",
        "av_known"])


def qb_career_experience(con) -> pd.DataFrame:
    """Career pass attempts and games (5+ att) of each team's current primary
    QB, cumulative across seasons, prior weeks only. Credit is per (team, QB)
    pair, so a transfer restarts at zero with his new team — a limitation, but
    a name is not a player id in pbp, and cross-team name joins would conflate
    different players with the same name."""
    from cfbpred.config import PBP
    q = con.execute(f"""
        SELECT "year" AS season,
               CASE WHEN season_type = 'postseason' THEN week + 20
                    ELSE week END AS week_c,
               pos_team AS team, passer_player_name AS qb,
               count(*) AS att
        FROM read_parquet('{PBP / "pbp_*.parquet"}')
        WHERE "pass" AND passer_player_name IS NOT NULL AND period <= 4
          AND week IS NOT NULL
        GROUP BY 1, 2, 3, 4
    """).fetchdf().sort_values(["team", "season", "week_c"])
    rows = []
    career: dict = {}
    for team, d in q.groupby("team", sort=False):
        cum_season: dict = {}
        cur_season = None
        for r in d.itertuples():
            if r.season != cur_season:
                cur_season, cum_season = r.season, {}
            # primary = season-to-date leader BEFORE this week
            if cum_season:
                primary = max(cum_season.items(), key=lambda kv: kv[1])[0]
                att_c, gms_c = career.get((team, primary), (0, 0))
                rows.append((r.season, int(r.week_c), team,
                             att_c / 400.0, gms_c / 30.0))
            key = (team, r.qb)
            a, gm = career.get(key, (0, 0))
            career[key] = (a + r.att, gm + (1 if r.att >= 5 else 0))
            cum_season[r.qb] = cum_season.get(r.qb, 0) + r.att
    return (pd.DataFrame(rows, columns=["season", "week_c", "team",
                                        "qb_career_att", "qb_career_games"])
            .drop_duplicates(["season", "week_c", "team"], keep="last"))


def primary_names_pbp(con) -> pd.DataFrame:
    """Season-to-date primary passer and rusher NAMES per (season, week_c,
    team), from prior weeks of play-by-play. Names come from pbp so they match
    the in-game pbp names exactly — box-score name formats can differ."""
    from cfbpred.config import PBP
    q = con.execute(f"""
        SELECT "year" AS season,
               CASE WHEN season_type = 'postseason' THEN week + 20
                    ELSE week END AS week_c,
               pos_team AS team, passer_player_name AS player,
               count(*) AS n, 'qb' AS role
        FROM read_parquet('{PBP / "pbp_*.parquet"}')
        WHERE "pass" AND passer_player_name IS NOT NULL AND period <= 4
          AND week IS NOT NULL
        GROUP BY 1, 2, 3, 4
        UNION ALL
        SELECT "year",
               CASE WHEN season_type = 'postseason' THEN week + 20
                    ELSE week END,
               pos_team, rusher_player_name, count(*), 'rb'
        FROM read_parquet('{PBP / "pbp_*.parquet"}')
        WHERE rush AND rusher_player_name IS NOT NULL AND period <= 4
          AND week IS NOT NULL
        GROUP BY 1, 2, 3, 4
    """).fetchdf()
    rows = []
    for (team, season, role), d in q.groupby(["team", "season", "role"]):
        weeks = sorted(d["week_c"].unique())
        cum: dict = {}
        for w in weeks:
            if cum:
                rows.append((season, int(w), team, role,
                             max(cum.items(), key=lambda kv: kv[1])[0]))
            for _, r in d[d["week_c"] == w].iterrows():
                cum[r["player"]] = cum.get(r["player"], 0) + r["n"]
    df = pd.DataFrame(rows, columns=["season", "week_c", "team", "role",
                                     "player"])
    return df.pivot_table(index=["season", "week_c", "team"], columns="role",
                          values="player", aggfunc="first").reset_index().rename(
        columns={"qb": "primary_qb", "rb": "primary_rb"})


# ------------------------------------------------- career experience, everyone

VOL_SPECS = [("passing", "YDS", "qb"), ("rushing", "YDS", "rb"),
             ("receiving", "YDS", "wr"), ("defensive", "TOT", "dfn"),
             ("kicking", "PTS", "k")]


def career_features(games: pd.DataFrame, *, raw: pd.DataFrame | None = None,
                    target_season: int | None = None,
                    target_week: int | None = None) -> pd.DataFrame:
    """Career-to-date experience for EVERY player, keyed by athlete_id.

    athlete_id follows a player across teams, so a transfer arrives with his
    real career volume — the earlier QB-only version keyed on pbp names and
    silently zeroed transfers. All counting, all prior games only.

    Per (season, week_c, team):
      cx_qb_gms / cx_qb_vol    career games & yards of the current primary QB
      cx_rb_gms, cx_wr_gms     same for the top back and top target
      cx_k_gms                 career games of the primary kicker
      cx_active_gms            production-weighted mean career games of every
                               player who appeared in the previous game
      cx_def_gms               mean career games of the top-5 tacklers
    """
    raw = load_player_stats() if raw is None else raw
    side_map = pd.concat([
        games[["game_id", "season", "week_c", "home_team"]]
        .assign(home_away="home").rename(columns={"home_team": "team"}),
        games[["game_id", "season", "week_c", "away_team"]]
        .assign(home_away="away").rename(columns={"away_team": "team"}),
    ])
    raw = raw[raw.athlete_id.notna()]
    raw = raw.merge(side_map, on=["game_id", "home_away"], how="inner",
                    suffixes=("", "_g"))

    vols = []
    for cat, st, role in VOL_SPECS:
        d = raw[(raw.category == cat) & (raw.stat_type == st)].copy()
        d["vol"] = _num(d["stat"]).clip(lower=0).fillna(0)
        d["role"] = role
        vols.append(d[["game_id", "season", "week_c", "team",
                       "athlete_id", "role", "vol"]])
    V = pd.concat(vols, ignore_index=True)
    appear = (V.groupby(["game_id", "season", "week_c", "team", "athlete_id"])
              ["vol"].sum().reset_index())

    rows = []
    career_g: dict = {}          # athlete -> career games
    career_v: dict = {}          # (athlete, role) -> career volume
    V = V.sort_values(["season", "week_c", "game_id"])
    appear = appear.sort_values(["season", "week_c", "game_id"])
    for (team, season), d in appear.groupby(["team", "season"], sort=True):
        season_vol: dict = {}
        prev_active: dict = {}
        vd = V[(V.team == team) & (V.season == season)]
        visits = list(d.groupby("game_id", sort=False))
        if season == target_season and target_week is not None:
            if target_week <= int(d.week_c.max()):
                raise ValueError("Snapshot target must follow all source weeks")
            visits.append((None, None))
        for gid, dg in visits:
            wk = target_week if gid is None else int(dg["week_c"].iloc[0])
            # ---- features from state BEFORE this game ----
            if season_vol:
                def leader(role):
                    c = {a: v for (a, r), v in season_vol.items() if r == role}
                    return max(c.items(), key=lambda kv: kv[1])[0] if c else None
                feat = {"season": season, "week_c": wk, "team": team}
                for role, gcol, vcol in [("qb", "cx_qb_gms", "cx_qb_vol"),
                                         ("rb", "cx_rb_gms", None),
                                         ("wr", "cx_wr_gms", None),
                                         ("k", "cx_k_gms", None)]:
                    a = leader(role)
                    if a is not None:
                        feat[gcol] = career_g.get(a, 0) / 30.0
                        if vcol:
                            feat[vcol] = career_v.get((a, role), 0.0) / 5000.0
                if prev_active:
                    tot = sum(prev_active.values()) or 1.0
                    feat["cx_active_gms"] = sum(
                        career_g.get(a, 0) * w for a, w in prev_active.items()
                    ) / tot / 30.0
                dtop = sorted(
                    ((a, v) for (a, r), v in season_vol.items() if r == "dfn"),
                    key=lambda kv: -kv[1])[:5]
                if dtop:
                    feat["cx_def_gms"] = float(np.mean(
                        [career_g.get(a, 0) for a, _ in dtop])) / 30.0
                rows.append(feat)
            if gid is None:
                continue
            # ---- update state with THIS game ----
            for a in dg["athlete_id"]:
                career_g[a] = career_g.get(a, 0) + 1
            for r_ in vd[vd.game_id == gid].itertuples():
                career_v[(r_.athlete_id, r_.role)] = (
                    career_v.get((r_.athlete_id, r_.role), 0.0) + r_.vol)
                season_vol[(r_.athlete_id, r_.role)] = (
                    season_vol.get((r_.athlete_id, r_.role), 0.0) + r_.vol)
            prev_active = dict(zip(dg["athlete_id"], dg["vol"]))
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["season", "week_c", "team"])
    return out.drop_duplicates(["season", "week_c", "team"], keep="last")
