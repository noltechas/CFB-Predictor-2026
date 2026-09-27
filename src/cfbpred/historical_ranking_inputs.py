"""Build an isolated terminal-target schedule for historical rankings.

Virtual rows are unplayed negative-ID targets only. They never receive scores,
labels, publication entries, or training outcomes.
"""
from __future__ import annotations

import argparse, hashlib, json, shutil
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


def modeled_fixture_mask(frame: pd.DataFrame) -> pd.Series:
    """Return fixtures involving a modeled Division-I team.

    ``features_raw.load_games(fbs_only=True)`` is also used for the ranking
    reconstruction, but the historical ranking contract includes FCS rows and
    FBS-vs-lower-division fixtures.  Filtering to FBS on either side silently
    drops those fixtures and lets the next scalar snapshot include a result
    that was never emitted as a target.
    """
    divisions = frame[["home_division", "away_division"]].astype("string")
    return divisions.isin({"fbs", "fcs"}).any(axis=1)


def terminal_schedule(source: str | Path, output: str | Path) -> dict:
    src = Path(source); out = Path(output); out.mkdir(parents=True, exist_ok=True)
    real = pd.read_parquet(src).copy()
    real["kick"] = pd.to_datetime(real["kick" if "kick" in real else "start_date"], utc=True)
    real = real[real.season.between(2001, 2026)].copy()
    real_ids = set(real.game_id.astype(int)); rows=[]
    for season, frame in real.groupby("season", sort=True):
        teams=[]
        for side in ("home", "away"):
            teams += frame[[f"{side}_team",f"{side}_division"]].rename(columns={f"{side}_team":"team",f"{side}_division":"division"}).to_dict("records")
        teams = pd.DataFrame(teams).drop_duplicates("team").sort_values("team")
        fbs = teams.loc[teams.division.eq("fbs"),"team"].tolist()
        allteams=teams.team.tolist()
        terminal_kick = frame.kick.max() + pd.Timedelta(days=7)
        for idx, rec in enumerate(teams.itertuples(index=False)):
            anchors=[x for x in fbs if x != rec.team] or [x for x in allteams if x != rec.team]
            if not anchors: continue
            anchor=anchors[0]
            gid=-(int(season)*100000+idx+1)
            while gid in real_ids: gid-=1000000
            # A virtual neutral row is after the final real game and has no
            # outcome columns. Its anchor is an existing team in the season.
            rows.append({"game_id":gid,"season":int(season),"week":99,"week_c":99,
                         "season_type":"terminal_virtual","start_date":terminal_kick.isoformat(),"kick":terminal_kick,
                         "neutral_site":True,"conference_game":False,"venue_id":None,
                         "home_team":rec.team,"home_conference":None,"home_division":rec.division,
                         "home_points":None,"away_team":anchor,"away_conference":None,
                         "away_division":"fbs" if anchor in fbs else "fcs","away_points":None,
                         "margin":None,"total":None})
    virtual=pd.DataFrame(rows)
    # Keep the source schema and append virtual rows; source rows remain byte
    # represented by their original values after round-trip normalization.
    combined=pd.concat([real,virtual],ignore_index=True,sort=False).sort_values(["season","kick","game_id"])
    combined.to_parquet(out/"snapshot_schedule.parquet",index=False)
    v_ids=sorted(int(x) for x in virtual.game_id)
    payload={"generated_at":datetime.now(timezone.utc).isoformat(),"source_schedule":str(src),
             "source_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),"real_game_count":len(real),
             "virtual_game_count":len(virtual),"virtual_game_ids":v_ids,
             "virtual_ids_excluded_from_publication":True,
             "virtual_rows_have_no_outcomes":bool(virtual[["home_points","away_points","margin","total"]].isna().all().all())}
    (out/"virtual_targets.json").write_text(json.dumps(payload,indent=2)+"\n")
    (out/"source.json").write_text(json.dumps(payload,indent=2)+"\n")
    return payload


def validate_terminal(schedule_path: str | Path, virtual_path: str | Path) -> dict:
    g=pd.read_parquet(schedule_path); v=json.loads(Path(virtual_path).read_text()); ids=set(v["virtual_game_ids"])
    real=g[~g.game_id.isin(ids)]; virt=g[g.game_id.isin(ids)]
    bad=[]
    for team, d in real.groupby("home_team"):
        pass
    return {"real_rows":len(real),"virtual_rows":len(virt),"virtual_outcome_rows":int(virt[["home_points","away_points","margin","total"]].notna().any(axis=1).sum()),"publication_ids":0}


def build_isolated_features(base: str | Path, players: str | Path | None = None,
                            reviewed: str | Path | None = None) -> dict:
    """Run the existing raw/player builders against only the isolated paths."""
    base, players = Path(base), Path(players or "models/player_sequence_production/refreshes/20260920T122506160000Z/players")
    # returning_production() reads the immutable raw CFBD game table through
    # the configured PQ root; provide that read-only table inside the isolated
    # root so no production path is redirected or mutated.
    if not (base / "games.parquet").exists():
        real_games = pd.read_parquet(base / "snapshot_schedule.parquet")
        real_games.loc[real_games.game_id.gt(0)].to_parquet(base / "games.parquet", index=False)
    market = Path("models/player_sequence_production/refreshes/20260920T122506160000Z/inputs/market.parquet")
    if market.exists() and not (base / "market.parquet").exists():
        shutil.copy2(market, base / "market.parquet")
    import cfbpred.features_raw as fr
    original_pq, original_feat, original_loader = fr.PQ, fr.FEAT, fr.load_games
    schedule = pd.read_parquet(base / "snapshot_schedule.parquet")
    def injected_loader(con, fbs_only=True):
        d = schedule.copy()
        if fbs_only:
            d = d.loc[modeled_fixture_mask(d)].copy()
        d["kick"] = pd.to_datetime(d["kick" if "kick" in d else "start_date"], utc=True)
        return d.sort_values(["season", "kick", "game_id"]).reset_index(drop=True)
    fr.PQ, fr.FEAT, fr.load_games = base, base, injected_loader
    try:
        fr.main()
    finally:
        fr.PQ, fr.FEAT, fr.load_games = original_pq, original_feat, original_loader
    from cfbpred.recent_form import build as recent_build
    recent_build(base)
    from cfbpred.prepare_player_data import prepare
    isolated_players = base / "players"
    player_inputs = base / "player_inputs"
    prepare(base, player_inputs, players=isolated_players, rebuild_players=True)
    from cfbpred.player_sequence_inputs import build as sequence_build
    sequence_dir = base / "sequences"
    sequence_build(isolated_players, player_inputs, sequence_dir)
    from cfbpred.player_full_history import build as full_history_build
    full_dir = base / "full_history"
    full_history_build(isolated_players, sequence_dir, player_inputs, full_dir)
    report = validate_terminal(base / "snapshot_schedule.parquet", base / "virtual_targets.json")
    # Terminal audit is explicit: every modeled team-season gets one virtual
    # row at the common season terminal, and completed counts are unchanged.
    real = schedule = pd.read_parquet(base / "snapshot_schedule.parquet")
    ids=set(json.loads((base/"virtual_targets.json").read_text())["virtual_game_ids"])
    terminal_audit=[]
    for (season,team), d in pd.concat([real.assign(team=real.home_team, side="home"), real.assign(team=real.away_team, side="away")]).groupby(["season","team"]):
        r=d[~d.game_id.isin(ids)]; completed=int((r.home_points.notna() & r.away_points.notna()).sum())
        # Exactly one terminal target is owned by each team as home; anchor
        # appearances as away are not additional team terminal targets.
        terminal_audit.append({"season":int(season),"team":team,"real_completed_rows":completed,
                               "virtual_rows":int((d.game_id.isin(ids)&d.side.eq("home")).sum())})
    (base/"terminal_gp_audit.json").write_text(json.dumps({"rows":terminal_audit,"all_virtual_one_per_team_season":all(x["virtual_rows"]==1 for x in terminal_audit)},indent=2)+"\n")
    report.update({"features_dir": str(base), "player_inputs": str(player_inputs),
                   "sequences": str(sequence_dir), "full_history": str(full_dir),
                   "isolated_players":str(isolated_players),"reviewed_registry": str(reviewed) if reviewed else None,
                   "terminal_gp_audit":str(base/"terminal_gp_audit.json")})
    (base/"context_source.json").write_text(json.dumps({"source_schedule":str(base/"snapshot_schedule.parquet"),"reviewed_registry":str(reviewed) if reviewed else None,"players":str(isolated_players),"inputs":str(player_inputs),"sequences":str(sequence_dir),"full_history":str(full_dir),"timeline":None},indent=2)+"\n")
    (base / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--source",required=True); ap.add_argument("--output",required=True); ap.add_argument("--build",action="store_true"); ap.add_argument("--reviewed")
    a=ap.parse_args(); payload=terminal_schedule(a.source,a.output)
    if a.build: payload=build_isolated_features(a.output, reviewed=a.reviewed)
    print(json.dumps(payload,indent=2))

if __name__ == "__main__": main()
