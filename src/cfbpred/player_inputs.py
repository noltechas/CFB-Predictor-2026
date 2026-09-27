"""Game-keyed, individual-player inputs with strict pre-kickoff history.

Static roster/person features are stored once. Each game references a ragged
roster and immutable cumulative box snapshots, so no roster is truncated and
future games cannot accidentally consume their own results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import duckdb
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfbpred.config import CFBD, PQ
from cfbpred.player_identity import (OVERRIDES, canonical_rosters, norm_id,
                                    resolve_history, resolve_rosters, safe_recruit_links)
from cfbpred.player_catalog import choose_scouting
from cfbpred.pregame_experiments import save_json, sha
from cfbpred.pull_espn_roster import team_alias_map

OUTPUT = PQ / "player_inputs_v3"
SPECS = [("passing", "C/ATT", "cmp", 1), ("passing", "C/ATT", "att", 2),
         ("passing", "YDS", "pass_yds", 0), ("passing", "TD", "pass_td", 0),
         ("passing", "INT", "pass_int", 0), ("rushing", "CAR", "rush_att", 0),
         ("rushing", "YDS", "rush_yds", 0), ("rushing", "TD", "rush_td", 0),
         ("receiving", "REC", "rec", 0), ("receiving", "YDS", "rec_yds", 0),
         ("receiving", "TD", "rec_td", 0), ("defensive", "TOT", "tackles", 0),
         ("defensive", "TFL", "tfl", 0), ("defensive", "SACKS", "sacks", 0),
         ("defensive", "PD", "pd", 0), ("interceptions", "INT", "def_int", 0),
         ("fumbles", "LOST", "fum_lost", 0), ("kicking", "FG", "fgm", 1),
         ("kicking", "FG", "fga", 2), ("kicking", "XP", "xpm", 1),
         ("kicking", "XP", "xpa", 2), ("punting", "NO", "punts", 0),
         ("punting", "YDS", "punt_yds", 0), ("kickReturns", "NO", "kr", 0),
         ("kickReturns", "YDS", "kr_yds", 0), ("puntReturns", "NO", "pr", 0),
         ("puntReturns", "YDS", "pr_yds", 0)]
STAT_COLS = [x[2] for x in SPECS]
# Observation counts are carried alongside sums. Unobserved is never zero
# production. The game count represents box presence, not snaps or starts.
HISTORY_COLS = STAT_COLS + [x+"_observations" for x in STAT_COLS] + ["box_games", "fbs_box_games"]
SCOUT_COLS = [f"{p}_{c}" for p in ("hs", "juco", "prep")
              for c in ("rating", "stars", "rank", "years_since", "known", "height", "weight", "height_known", "weight_known")]
STATIC_COLS = SCOUT_COLS + ["identity_uncertain", "prior_other_program", "prior_seasons"]
POSITIONS = {"UNK": 0, "QB": 1, "RB": 2, "WR": 3, "TE": 4,
             "OL": 5, "DL": 6, "LB": 7, "DB": 8, "ST": 9}
POSITION_ALIASES = {**{x: "QB" for x in ["PRO", "DUAL"]},
                    **{x: "RB" for x in ["APB", "FB"]},
                    **{x: "OL" for x in ["OT", "OG", "C", "IOL", "OC"]},
                    **{x: "DL" for x in ["DT", "DE", "SDE", "WDE", "EDGE"]},
                    **{x: "LB" for x in ["ILB", "OLB"]},
                    **{x: "DB" for x in ["CB", "S", "FS", "SS"]},
                    **{x: "ST" for x in ["K", "P", "LS", "K/P", "PK", "KR", "PR"]},
                    # roster feeds abbreviate differently from recruiting services
                    "NT": "DL", "G": "OL", "T": "OL"}
# Where a player's position group comes from. "scouting" is the dated recruiting
# position only (the production checkpoints through tabm16-v4 were trained on it;
# 63% of 2026 rostered players, 37% on FBS rosters, fall to UNK because they have no
# recruiting link). "scouting_then_roster" keeps the scouting position where one
# exists and otherwise takes the listed roster position. A store's source is recorded
# in its manifest and a checkpoint must be scored with the source it was trained on.
POSITION_SOURCES = ("scouting", "scouting_then_roster")


def map_positions(values) -> np.ndarray:
    """Position labels -> POSITIONS index via the alias table; anything else is UNK (0)."""
    s = pd.Series(values, dtype="object")
    return s.map(lambda v: POSITIONS.get(POSITION_ALIASES.get(v, v), 0) if isinstance(v, str) else 0).to_numpy(np.int64)


def assign_positions(catalog, links, position_source="scouting") -> np.ndarray:
    """One position-group index per catalog row. The dated scouting position is
    preferred because the roster feed can overwrite a historical listing; with
    "scouting_then_roster" the roster listing fills in only where no scouting link exists."""
    if position_source not in POSITION_SOURCES:
        raise ValueError(f"Unknown position source: {position_source}")
    last = links.sort_values("recruit_year").drop_duplicates("roster_row", keep="last").set_index("roster_row")
    position = map_positions(catalog.roster_row.map(last.position))
    if position_source == "scouting_then_roster":
        position = np.where(position > 0, position, map_positions(catalog.position))
    return position


def transform_history(values):
    """Fixed count transform, no fitted intermediate player ratings."""
    return (np.sign(values)*np.log1p(np.abs(values))/5).astype(np.float32)


def read_boxes(paths):
    paths = [str(p) for p in paths if p.exists()]
    with duckdb.connect() as con:
        # One official stat value per game/side/player/category/type. Identical
        # duplicate imports collapse; conflicting observations stay missing.
        selects = []
        for cat, typ, name, part in SPECS:
            value = f"split_part(stat_value, '/', {part})" if part else "stat_value"
            selects.append(f"max(CASE WHEN category='{cat}' AND stat_type='{typ}' THEN try_cast({value} AS DOUBLE) END) AS {name}")
        sql = """WITH unique_stats AS (
          SELECT game_id,home_away,CAST(athlete_id AS VARCHAR) athlete_id,category,stat_type,
          CASE WHEN count(DISTINCT stat)=1 THEN max(stat) ELSE NULL END AS stat_value
          FROM read_parquet(?) WHERE athlete_id IS NOT NULL GROUP BY ALL)
          SELECT game_id,home_away,athlete_id,""" + ",".join(selects) + " FROM unique_stats GROUP BY ALL"
        return con.execute(sql, [paths]).fetchdf()


def clean_boxes(boxes, schedule, identities):
    mapping = pd.concat([schedule[["game_id", "season", "kick", "margin", side+"_team", side+"_division"]].rename(
        columns={side+"_team": "team", side+"_division": "division"}).assign(home_away=side)
        for side in ["home", "away"]], ignore_index=True)
    b = boxes.merge(mapping, on=["game_id", "home_away"], how="inner", validate="many_to_one")
    b = resolve_history(b, identities)
    rejected = b[~b.history_usable].copy()
    b = b[b.history_usable & b.margin.notna()].copy()
    keys = ["game_id", "season", "kick", "team", "division", "person_id"]
    # Reviewed aliases may duplicate the same box under different provider
    # IDs. Collapse equal observations, quarantine conflicting values.
    grouped = b.groupby(keys, dropna=False)[STAT_COLS]
    values = grouped.first(); values[grouped.nunique().gt(1)] = np.nan
    b = values.reset_index()
    for col in STAT_COLS:
        b[col+"_observations"] = b[col].notna().astype(float)
        b[col] = b[col].fillna(0)
    b["box_games"] = 1.
    b["fbs_box_games"] = b.division.eq("fbs").astype(float)
    return b.sort_values(["person_id", "season", "kick"]), rejected


def static_inputs(catalog, links, boxes, position_source="scouting"):
    catalog = choose_scouting(catalog, links)
    raw = np.zeros((len(catalog), len(STATIC_COLS)), np.float32)
    for p in ("hs", "juco", "prep"):
        j = SCOUT_COLS.index(p+"_rating")
        # Missing fields remain distinguishable; reject partially unrated
        # records as scouting evidence, while preserving them in the audit.
        known = catalog[p+"_scouting_known"].to_numpy()
        raw[:, j] = np.where(known, (catalog[p+"_rating"].fillna(.8)-.8)*5, 0)
        raw[:, j+1] = catalog[p+"_stars"].fillna(0).to_numpy()/5
        raw[:, j+2] = np.log1p(catalog[p+"_rank"].fillna(0).to_numpy())/8
        raw[:, j+3] = (catalog.season-catalog[p+"_recruit_year"]).fillna(0).to_numpy()/6
        raw[:, j+4] = known
        for offset,field,center,scale in [(5,"height",72,12),(6,"weight",220,100)]:
            values = pd.to_numeric(catalog[p+"_"+field],errors="coerce")
            valid = values.between(55,90) if field=="height" else values.between(100,450)
            raw[:,j+offset] = np.where(valid,(values-center)/scale,0)
            raw[:,j+offset+2] = valid
    raw[:, -3] = catalog.identity_uncertain.to_numpy()
    hist = np.zeros((len(catalog), 3, len(HISTORY_COLS)), np.float32)
    annual = boxes.groupby(["season", "person_id"])[HISTORY_COLS].sum().reset_index()
    # Unknown position is a real pool; see assign_positions for the source choice.
    position = assign_positions(catalog, links, position_source)
    for year, ros in catalog.groupby("season"):
        for slot, selection in enumerate([annual.season.eq(year-1), annual.season.eq(year-2), annual.season.le(year-3)]):
            a = annual[selection].groupby("person_id")[HISTORY_COLS].sum()
            hist[ros.index, slot] = a.reindex(ros.person_id).fillna(0).to_numpy(np.float32)
        prior = boxes[boxes.season.lt(year)]
        seen = prior.groupby("person_id").season.nunique()
        raw[ros.index, -1] = ros.person_id.map(seen).fillna(0).to_numpy()/6
        programs = prior[["person_id", "team"]].drop_duplicates().groupby("person_id").team.agg(set)
        raw[ros.index, -2] = [any(t != row.team for t in programs.get(row.person_id, set())) for row in ros.itertuples()]
    # One reserved row is the empty-roster sentinel. Padding contributes zero.
    return catalog, np.vstack([np.zeros_like(raw[:1]), raw]), np.concatenate([np.zeros_like(hist[:1]), transform_history(hist)]), np.r_[0, position]


def build_snapshots(catalog, boxes, schedule, target_ids, output):
    """Emit before consume, grouped by kickoff; future games never advance."""
    # A snapshot references the cumulative record AFTER a prior game. Zero is
    # no observation. Cumulative totals reset every season.
    boxes = boxes.sort_values(["kick", "game_id", "person_id"]).reset_index(drop=True)
    cumulative = boxes.groupby(["person_id", "season"])[HISTORY_COLS].cumsum()
    np.save(output/"current_history.npy", np.vstack([np.zeros((1, len(HISTORY_COLS)), np.float32),
                                                   transform_history(cumulative.to_numpy())]))
    boxes["snapshot_id"] = np.arange(len(boxes))+1
    roster_groups = {(int(y), t): frame for (y, t), frame in catalog.groupby(["season", "team"])}
    updates = {gid: frame for gid, frame in boxes.groupby("game_id")}
    active, rows, player_refs, current_refs = {}, [], [], []
    offset = 0
    for kick, simultaneous in schedule.sort_values("kick").groupby("kick"):
        for game in simultaneous.itertuples():
            if game.game_id not in target_ids:
                continue
            for side in ["home", "away"]:
                team = getattr(game, side+"_team")
                ros = roster_groups.get((game.season, team))
                ids = [] if ros is None else (ros.index.to_numpy()+1).tolist()
                snapshots = [] if ros is None else [active.get((game.season, p), (0, None))[0] for p in ros.person_id]
                dates = [] if ros is None else [active.get((game.season, p), (0, None))[1] for p in ros.person_id]
                dates = [x for x in dates if x is not None]
                if dates and max(dates) >= kick:
                    raise ValueError("Player snapshot uses target or future game")
                player_refs.extend(ids); current_refs.extend(snapshots)
                rows.append({"game_id": game.game_id, "season": game.season, "side": side,
                             "team": team, "offset": offset, "players": len(ids), "kick": kick,
                             "last_history_kick": max(dates) if dates else pd.NaT})
                offset += len(ids)
        for game in simultaneous.itertuples():
            if pd.isna(game.margin):
                continue
            for row in updates.get(game.game_id, pd.DataFrame()).itertuples():
                active[(row.season, row.person_id)] = (row.snapshot_id, kick)
    table = pd.DataFrame(rows)
    table.to_parquet(output/"game_rosters.parquet", index=False)
    np.save(output/"player_refs.npy", np.asarray(player_refs, np.int32))
    np.save(output/"current_refs.npy", np.asarray(current_refs, np.int32))
    boxes[["game_id", "person_id", "season", "team", "kick", "snapshot_id"]].to_parquet(output/"history_provenance.parquet", index=False)
    return table


def build(schedule_path, targets_path, output=OUTPUT, position_source="scouting"):
    if position_source not in POSITION_SOURCES:
        raise ValueError(f"Unknown position source: {position_source}")
    output.mkdir(parents=True, exist_ok=True)
    aliases = team_alias_map()
    schedule = pd.read_parquet(schedule_path).copy()
    schedule["kick"] = pd.to_datetime(schedule.kick, utc=True)
    for side in ["home", "away"]:
        schedule[side+"_team"] = schedule[side+"_team"].replace(aliases)
    targets = pd.read_parquet(targets_path)
    roster = pd.read_parquet(CFBD/"roster.parquet").rename(columns={"id": "athlete_id", "season_pulled": "season"})
    roster["team"] = roster.team.replace(aliases)
    roster = roster.drop_duplicates(["season", "team", "athlete_id"])
    recruits = pd.read_parquet(CFBD/"recruiting_players.parquet")
    identities = resolve_rosters(roster,recruits=recruits)
    identities.to_parquet(output/"identity_crosswalk.parquet", index=False)
    catalog = canonical_rosters(identities)
    catalog, links, rejected = safe_recruit_links(catalog, recruits)
    links.to_parquet(output/"recruit_links.parquet", index=False)
    rejected.to_parquet(output/"rejected_recruit_links.parquet", index=False)
    print(f"Resolved {len(catalog):,} roster people; reading raw individual boxes", flush=True)
    boxes, rejected_history = clean_boxes(read_boxes([CFBD/"player_game_stats.parquet", CFBD/"player_game_stats_fcs.parquet"]), schedule, identities)
    rejected_history[["game_id", "athlete_id", "season", "team", "person_id"]].to_parquet(output/"quarantined_history.parquet", index=False)
    boxes.to_parquet(output/"clean_player_boxes.parquet", index=False)
    catalog, scout, historical, position = static_inputs(catalog, links, boxes, position_source)
    catalog.to_parquet(output/"player_seasons.parquet", index=False)
    unknown = position[1:] == 0
    np.save(output/"scouting.npy", scout); np.save(output/"prior_history.npy", historical); np.save(output/"position.npy", position)
    print("Building pre-kickoff player references", flush=True)
    games = build_snapshots(catalog, boxes, schedule, set(targets.game_id), output)
    if len(games)!=2*len(targets) or games.duplicated(["game_id", "side"]).any():
        raise ValueError("Every target needs exactly two roster references")
    current = games[games.season.eq(2026)]
    report = {"version": 1, "history_cols": HISTORY_COLS, "static_cols": STATIC_COLS,
              "positions": POSITIONS, "history_windows": ["previous_season", "two_seasons_ago", "earlier_career", "current_before_kickoff"],
              "targets": len(targets), "roster_rows": len(catalog), "player_game_references": int(games.players.sum()),
              "untruncated_max_roster": int(games.players.max()), "empty_game_sides": int(games.players.eq(0).sum()),
              "current_empty_game_sides": int(current.players.eq(0).sum()),
              "current_players": len(catalog[catalog.season.eq(2026)]),
              "position_source": position_source,
              "position_unknown_share": float(unknown.mean()),
              "current_position_unknown_share": float(unknown[catalog.season.eq(2026).to_numpy()].mean()),
              "current_uncertain": int(catalog.loc[catalog.season.eq(2026), "identity_uncertain"].sum()),
              "clean_box_rows": len(boxes), "quarantined_history_rows": len(rejected_history),
              "rejected_recruit_links": len(rejected), "no_2026_outcomes_in_model_fitting": True,
              "limitations": ["Annual roster membership and recruiting revisions lack archived publication timestamps", "No complete public snap, starter, injury or OL blocking history", "Unverified identities remain quarantined; provider IDs are not certified globally", "Historical class, height and weight excluded because source can overwrite them"],
              "source_sha256": {str(p): sha(p) for p in [Path(schedule_path), Path(targets_path), OVERRIDES,
                  CFBD/"roster.parquet", CFBD/"recruiting_players.parquet", CFBD/"player_game_stats.parquet", CFBD/"player_game_stats_fcs.parquet"]}}
    report["artifact_sha256"] = {p.name: sha(p) for p in output.iterdir() if p.suffix in [".npy", ".parquet"]}
    save_json(output/"manifest.json", report)
    print(json.dumps({k:v for k,v in report.items() if not k.endswith("sha256")}, indent=2), flush=True)


class PlayerStore:
    def __init__(self, path, game_ids, verify=True):
        self.path = Path(path)
        self.manifest = json.loads((self.path/"manifest.json").read_text())
        if self.manifest["version"] != 1:
            raise ValueError("Unsupported player input schema")
        if verify:
            for name,digest in self.manifest["artifact_sha256"].items():
                if sha(self.path/name)!=digest:
                    raise ValueError(f"Player artifact changed: {name}")
        for name in ["scouting", "prior_history", "position", "current_history", "player_refs", "current_refs"]:
            setattr(self, name, np.load(self.path/(name+".npy"), mmap_mode="r"))
        g = pd.read_parquet(self.path/"game_rosters.parquet").set_index(["game_id", "side"])
        if g.index.duplicated().any():
            raise ValueError("Duplicate player game references")
        # .loc fails if ANY future target is missing, including an empty-roster
        # game. No silently dropped targets or guessed roster joins.
        order = pd.MultiIndex.from_product([game_ids, ["home", "away"]])
        g = g.loc[order]
        self.offsets = g.offset.to_numpy().reshape(-1, 2)
        self.counts = g.players.to_numpy().reshape(-1, 2)
        if (self.offsets<0).any() or (self.counts<0).any() or (self.offsets+self.counts>len(self.player_refs)).any():
            raise ValueError("Invalid ragged roster boundaries")
        if len(self.player_refs)!=len(self.current_refs):
            raise ValueError("Static and current player references differ")
        if self.scouting.shape[-1]!=len(self.manifest["static_cols"]) or self.prior_history.shape[1:]!=(3,len(self.manifest["history_cols"])):
            raise ValueError("Player arrays differ from their schema")

    def batch(self, indices, device):
        indices = np.asarray(indices)
        n = int(max(1, self.counts[indices].max()))
        offsets = self.offsets[indices][..., None]
        valid = np.arange(n)[None, None, :] < self.counts[indices][..., None]
        refs = np.where(valid, offsets + np.arange(n), 0)
        p = np.where(valid, self.player_refs[refs], 0)
        c = np.where(valid, self.current_refs[refs], 0)
        hist = np.concatenate([self.prior_history[p], self.current_history[c][..., None, :]], axis=-2)
        return {"player_scout": torch.tensor(self.scouting[p], device=device),
                "player_history": torch.tensor(hist, device=device),
                "player_position": torch.tensor(self.position[p], device=device),
                "player_mask": torch.tensor(valid, device=device)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", type=Path, required=True); ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    ap.add_argument("--position-source", choices=POSITION_SOURCES, default="scouting",
                    help="scouting = recruiting position only (what deployed checkpoints were trained on); "
                         "scouting_then_roster = fall back to the roster listing (requires a retrain)")
    args = ap.parse_args(); build(args.schedule, args.targets, args.output, args.position_source)
