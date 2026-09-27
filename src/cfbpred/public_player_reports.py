"""Import explicit lists in public, timestamped contemporaneous reports.

Only structured status sections are parsed. Narrative guesses, unlisted
players, and missing timestamps never become available/healthy labels.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
from bs4 import BeautifulSoup
import pandas as pd
from cfbpred.player_inputs import POSITIONS,POSITION_ALIASES
from cfbpred.pregame_experiments import sha,save_json


def status_lists(html):
    soup=BeautifulSoup(html,"html.parser")
    status=None
    for element in soup.find_all(["p","h2","h3","h4","ul"]):
        text=element.get_text(" ",strip=True).rstrip(":").strip().lower()
        if text in {"out","doubtful","questionable","probable","available"}:
            status=text;continue
        if element.name!="ul" or status is None:continue
        for item in element.find_all("li",recursive=False):
            line=item.get_text(" ",strip=True)
            if line.lower().strip() in {"none","n/a"}:continue
            line=re.sub(r"\s*\([^)]*\)\s*"," ",line).strip()
            match=re.fullmatch(r"([A-Z]{1,4})\s+(.+)",line)
            position,name=(match.group(1),match.group(2)) if match else (None,line)
            if position and position not in {*POSITIONS,*POSITION_ALIASES,"SAF","RT","LT","RG","LG"}:
                position=None;name=line
            if position=="SAF":position="DB"
            if position in {"RT","LT","RG","LG"}:position="OL"
            yield name,status,position
        status=None # never propagate a label into another unrelated list


def dated_articles(archive,schedule,revision_times=None):
    meta=json.loads(archive.with_suffix(".json.source.json").read_text())
    games=schedule[(schedule.home_team=="Ohio State")|(schedule.away_team=="Ohio State")].copy()
    games["kick"]=pd.to_datetime(games.kick,utc=True)
    facts,rejected=[],[]
    revisions=({row["link"]:row for row in json.loads(revision_times.read_text())}
               if revision_times is not None else {})
    for article in json.loads(archive.read_text()):
        if not article.get("date_gmt"):continue
        published=pd.Timestamp(article["date_gmt"],tz="UTC")
        rows=list(status_lists(article["content"]["rendered"]))
        if not rows:continue
        candidates=games[(games.kick>published)&(games.kick<=published+pd.Timedelta(hours=36))]
        if len(candidates)!=1:
            rejected.append(dict(url=article["link"],reason="no_unique_upcoming_game_within_36h"));continue
        game=candidates.iloc[0]
        revision=revisions.get(article["link"],article)
        if not revision.get("modified_gmt"):
            rejected.append(dict(url=article["link"],reason="revision_timestamp_unverified"));continue
        modified=pd.Timestamp(revision["modified_gmt"],tz="UTC")
        if modified>=game.kick:
            rejected.append(dict(url=article["link"],reason="article_revised_at_or_after_kickoff"));continue
        known=max(published,modified)
        for name,status,pos in rows:
            base=dict(name=name,team="Ohio State",season=int(game.season),game_id=int(game.game_id),
                      known_at=known.isoformat(),effective_at=known.isoformat(),
                      published_at=published.isoformat(),modified_at=modified.isoformat(),
                      valid_until=(game.kick+pd.Timedelta(days=1)).isoformat(),
                      source_path=str(archive.resolve()),source_sha256=sha(archive),
                      source_url=article["link"],observed_at=meta["observed_at"],
                      timing_basis="Latest publisher publication/revision timestamp, both before kickoff")
            facts.append({**base,"kind":"availability","value":status})
            if pos:facts.append({**base,"kind":"position","value":pos})
    return facts,rejected


def sec_current(archive,schedule):
    """Read the observed public SEC schema, using retrieval as knowledge time.

    The displayed publication schedule can be in the future while pending.
    It is not evidence that the player statuses were published at that time.
    """
    meta=json.loads(archive.with_suffix(".json.source.json").read_text())
    observed=pd.Timestamp(meta["observed_at"])
    games=schedule.copy();games["kick"]=pd.to_datetime(games.kick,utc=True)
    facts,quarantine=[],[]
    for report in json.loads(archive.read_text()).values():
        if report.get("ReportType")=="Report Pending":
            quarantine.append(dict(reason="report_pending",game_date=report.get("footer",{}).get("date")))
            continue
        game_date=report.get("footer",{}).get("date")
        from cfbpred.conference_availability import team_name
        schedule_teams=set(games.home_team)|set(games.away_team)
        conference=meta.get("params",{}).get("conference")
        reported_teams={team_name(t["teamDisplayName"],schedule_teams,conference) for t in report.get("games",[])}
        matches=games[(games.kick.dt.tz_convert("America/Chicago").dt.strftime("%Y-%m-%d")==game_date)
                      &games.home_team.isin(reported_teams)&games.away_team.isin(reported_teams)]
        if len(matches)!=1:
            quarantine.append(dict(reason="unmatched_sec_game",game_date=game_date));continue
        game=matches.iloc[0]
        if observed>=game.kick:
            quarantine.append(dict(reason="report_retrieved_after_kickoff",game_id=int(game.game_id)));continue
        for team in report.get("games",[]):
            for row in team.get("rows",[]):
                from cfbpred.conference_availability import STATUS
                raw_status=str(row.get("status","")).lower()
                status=STATUS.get(raw_status,raw_status)
                if status not in {"out","doubtful","questionable","probable","available"}:
                    # Preserve half-game suspensions and future schema changes
                    # for explicit encoding; never convert them into full out.
                    quarantine.append(dict(reason="status_requires_explicit_encoding",raw=row,team=team["teamDisplayName"]));continue
                if not row.get("name"):raise ValueError("SEC player row has no name")
                facts.append(dict(name=row["name"],team=team_name(team["teamDisplayName"],schedule_teams,conference),season=int(game.season),
                     game_id=int(game.game_id),kind="availability",value=status,
                     known_at=observed.isoformat(),effective_at=observed.isoformat(),
                     valid_until=(game.kick+pd.Timedelta(days=1)).isoformat(),
                     source_path=str(archive.resolve()),source_sha256=sha(archive),source_url=meta["url"],
                     observed_at=observed.isoformat(),raw_status=raw_status,timing_assumed=False,
                     timing_basis="Actual retrieval of public report before kickoff; not a scheduled publication time"))
    return facts,quarantine


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ["archive","schedule","reviewed","output"]:p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--revision-times",type=Path)
    a=p.parse_args();facts,rejected=dated_articles(a.archive,pd.read_parquet(a.schedule),a.revision_times)
    save_json(a.output,json.loads(a.reviewed.read_text())+facts)
    save_json(a.output.with_suffix(".coverage.json"),dict(imported_facts=len(facts),rejected=rejected,
                                                       source_sha256=sha(a.archive)))
    print(f"Imported {len(facts)} dated facts",flush=True)
