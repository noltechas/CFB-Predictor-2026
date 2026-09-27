"""Validate and publish Chas's weekly rankings, independently of the model.

    python3 src/cfbpred/editorial_rankings.py --file data/editorial_rankings.json
    python3 src/cfbpred/editorial_rankings.py --file data/editorial_rankings.json --publish

Load the project's .env into the shell first, as with the existing publishers.
Without --publish this command only reads the team directory and prints a preview.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_snapshot(document: dict, teams: list[dict], *, published_at: str | None = None) -> dict:
    if not isinstance(document, dict):
        raise ValueError("Rankings must be a JSON object")
    for key, low, high in [("season", 2000, 2200), ("ranking_week", 0, 40)]:
        value = document.get(key)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{key} must be an integer from {low} to {high}")
    author = document.get("author", "Chas Nolte")
    title = document.get("title", "Chas’s rankings")
    introduction = document.get("introduction")
    for label, value, limit in [("author", author, 120), ("title", title, 200)]:
        if not isinstance(value, str) or not 1 <= len(value.strip()) <= limit:
            raise ValueError(f"{label} must contain 1–{limit} characters")
    if introduction is not None and (not isinstance(introduction, str) or len(introduction) > 4000):
        raise ValueError("introduction must be text up to 4000 characters")
    selections = document.get("teams")
    if not isinstance(selections, list) or not 1 <= len(selections) <= 1000:
        raise ValueError("Add 1–1000 teams in rank order before publishing")
    by_id = {t["team_id"]: t for t in teams}
    by_name = {t["school"].strip().casefold(): t for t in teams}
    entries, seen = [], set()
    for rank, selection in enumerate(selections, 1):
        note = None
        if isinstance(selection, dict):
            note = selection.get("note")
            selection = selection.get("team_id", selection.get("school"))
        if note is not None and (not isinstance(note, str) or len(note) > 2000):
            raise ValueError(f"Rank {rank}: note must be text up to 2000 characters")
        team = by_id.get(selection) if type(selection) is int else by_name.get(selection.strip().casefold()) if isinstance(selection, str) else None
        if team is None:
            raise ValueError(f"Rank {rank}: unknown team {selection!r}; use --list-teams to find its school name or ID")
        if team["team_id"] in seen:
            raise ValueError(f"Duplicate team: {team['school']}")
        seen.add(team["team_id"])
        entries.append({"team_id": team["team_id"], "school": team["school"],
                        "conference": team.get("conference"), "rank": rank,
                        "note": note.strip() or None if note is not None else None})
    return {"season": document["season"], "ranking_week": document["ranking_week"],
            "author": author.strip(), "title": title.strip(), "introduction": introduction,
            "published_at": published_at or datetime.now(timezone.utc).isoformat(), "entries": entries}


def publish_snapshot(cur, snapshot: dict) -> None:
    """Replace the complete week atomically; an older job cannot overwrite it."""
    from psycopg.types.json import Jsonb
    cur.execute("""insert into public.editorial_rankings
        (season, ranking_week, author, title, introduction, published_at, entries)
        values (%s, %s, %s, %s, %s, %s, %s)
        on conflict (season, ranking_week) do update set
          author = excluded.author, title = excluded.title,
          introduction = excluded.introduction, published_at = excluded.published_at,
          entries = excluded.entries
        where editorial_rankings.published_at <= excluded.published_at
        returning entries""", tuple(snapshot[k] for k in
        ("season", "ranking_week", "author", "title", "introduction", "published_at")) + (Jsonb(snapshot["entries"]),))
    result = cur.fetchone()
    if result is None:
        raise ValueError("A newer version is already published for this week")
    if result[0] != snapshot["entries"]:
        raise ValueError("Published rankings did not match the reviewed list")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--list-teams", metavar="SEARCH", help="Find school names and team IDs")
    args = parser.parse_args()
    if args.publish and (not args.file or args.list_teams is not None):
        parser.error("--publish requires --file and cannot be combined with --list-teams")
    if not args.file and args.list_teams is None:
        parser.error("Use --file or --list-teams")
    import psycopg
    from cfbpred.load_supabase import dsn
    try:
        with psycopg.connect(dsn()) as cx, cx.cursor() as cur:
            cur.execute("set local statement_timeout = '15s'")
            cur.execute("select team_id, school, conference from public.teams order by school")
            teams = [dict(zip(("team_id", "school", "conference"), row)) for row in cur.fetchall()]
            if args.list_teams is not None:
                for team in teams:
                    if args.list_teams.casefold() in team["school"].casefold():
                        print(f"{team['team_id']:>6}  {team['school']} · {team['conference'] or 'Independent'}")
                return
            snapshot = build_snapshot(json.loads(args.file.read_text()), teams)
            print(json.dumps(snapshot, indent=2, ensure_ascii=False))
            if args.publish:
                publish_snapshot(cur, snapshot)
        print(f"Published {len(snapshot['entries'])} teams." if args.publish else "Preview only. Add --publish to publish this list.")
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    except psycopg.Error:
        parser.exit(1, "Database operation failed. Check the connection and editorial_rankings migration.\n")


if __name__ == "__main__":
    main()
