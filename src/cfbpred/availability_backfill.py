"""Public availability ingestion and a schedule-first FBS coverage ledger.

The ledger is deliberately independent of which reports were found: every
FBS-involved scheduled game has two participant rows, including missing rosters.
Source discovery, verified individual facts, and complete roster knowledge are
different things. No missing report is converted to a healthy-player label.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
from bs4 import BeautifulSoup

from cfbpred.player_timeline import STATUSES, resolve_names
from cfbpred.pregame_experiments import save_json, sha


def timestamp(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("missing_or_timezone_naive_timestamp")
    return result.tz_convert("UTC")


def url_key(value):
    parts = urlsplit(str(value))
    return parts.netloc.lower().removeprefix("www."), parts.path.rstrip("/")


def publisher_versions(path):
    """Read revision evidence from the actual archived publisher payload.

    Submissions cannot turn PDF file metadata or an invented date into publisher
    evidence merely by putting those dates in published_at/modified_at fields.
    """
    records = []
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text())
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, dict) and all(item.get(k) for k in ("link", "date_gmt", "modified_gmt")):
                records.append((url_key(item["link"]), pd.Timestamp(item["date_gmt"], tz="UTC"),
                                pd.Timestamp(item["modified_gmt"], tz="UTC")))
    elif path.suffix.lower() in {".html", ".htm"}:
        soup = BeautifulSoup(path.read_text(), "html.parser")
        canonical = soup.find("link", rel="canonical")
        default_url = canonical.get("href", "") if canonical else ""

        def visit(value):
            if isinstance(value, list):
                for item in value:
                    visit(item)
            elif isinstance(value, dict):
                types = value.get("@type", [])
                if isinstance(types, str):
                    types = [types]
                if set(types) & {"NewsArticle", "Article", "BlogPosting", "ReportageNewsArticle"}:
                    if value.get("datePublished") and value.get("dateModified"):
                        url = value.get("url") or value.get("mainEntityOfPage") or default_url
                        if isinstance(url, dict):
                            url = url.get("@id", default_url)
                        records.append((url_key(url), timestamp(value["datePublished"]), timestamp(value["dateModified"])))
                if "@graph" in value:
                    visit(value["@graph"])

        for script in soup.find_all("script", type="application/ld+json"):
            try:
                visit(json.loads(script.get_text()))
            except (ValueError, TypeError):
                continue
        if not records:
            # Open Graph article metadata is the publisher's own statement of
            # both times; accept it only as a complete publication/revision pair.
            published = soup.find("meta", attrs={"property": "article:published_time"})
            modified = soup.find("meta", attrs={"property": "article:modified_time"})
            if published and modified and published.get("content") and modified.get("content"):
                try:
                    records.append((url_key(default_url), timestamp(published["content"]), timestamp(modified["content"])))
                except (ValueError, TypeError):
                    pass
    return records


def archived_text(path, source_url):
    """Extract publisher text for checking an extractor's claimed quotation."""
    if path.suffix.lower() in {".html", ".htm"}:
        html = path.read_text()
    elif path.suffix.lower() == ".json":
        items = json.loads(path.read_text())
        if isinstance(items, dict) and isinstance(items.get("data"), dict):
            article = items["data"]
            if url_key(article.get("permalink", "")) == url_key(source_url):
                return " ".join(BeautifulSoup(json.dumps(article.get("content_blocks", [])), "html.parser").get_text(" ", strip=True).split())
        items = items if isinstance(items, list) else [items]
        selected = [item for item in items if isinstance(item, dict)
                    and url_key(item.get("link", "")) == url_key(source_url)]
        html = " ".join(str(item.get(field, {}).get("rendered", ""))
                        for item in selected for field in ("title", "content", "excerpt"))
    else:
        return None
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
    return " ".join(soup.get_text(" ", strip=True).split())


def validate_fact(row, schedule, hashes=None, versions=None):
    """Gate newly collected game-specific facts; baseline remains immutable.

    A historical publication date by itself cannot attest to the current text.
    Require publisher revision metadata or a demonstrably pregame capture.
    Human interpretation and identity review still accompany this mechanical gate.
    """
    hashes = {} if hashes is None else hashes
    versions = {} if versions is None else versions
    try:
        game = schedule.loc[int(row["game_id"])]
        if isinstance(game, pd.DataFrame):
            return "ambiguous_game_id"
        kick = timestamp(game.kick)
        if row["team"] not in (game.home_team, game.away_team):
            return "team_not_in_game"
        if int(row["season"]) != int(game.season):
            return "wrong_season"
        known, effective, until = [timestamp(row[k]) for k in ("known_at", "effective_at", "valid_until")]
        if not known < kick or not effective <= kick < until:
            return "fact_not_available_at_kickoff"
        if not row.get("name") or not str(row.get("source_url", "")).startswith("https://"):
            return "missing_identity_or_public_source"
        if row["kind"] == "availability":
            if row["value"] not in STATUSES:
                return "status_requires_explicit_encoding"
        elif row["kind"] == "depth":
            import math
            rank = float(row["value"])
            if not math.isfinite(rank) or rank < 1 or rank != int(rank):
                return "invalid_depth_rank"
        elif row["kind"] not in {"position", "depth"}:
            return "unsupported_backfill_fact_kind"
        path = Path(row["source_path"])
        key = str(path.resolve())
        if key not in hashes:
            hashes[key] = sha(path)
        if hashes[key] != row["source_sha256"]:
            return "source_hash_mismatch"
        if row.get("source_quote"):
            text_key = (key, url_key(row["source_url"]), "text")
            if text_key not in versions:
                versions[text_key] = archived_text(path, row["source_url"])
            content = versions[text_key]
            if content is not None and " ".join(row["source_quote"].split()) not in content:
                return "quoted_text_not_in_archived_source"
        if row.get("timing_policy") in ARCHIVE_POLICIES:
            return validate_archived_report(row, game, kick, known, path, key, hashes, versions, schedule)
        if row.get("timing_assumed"):
            if row.get("timing_policy") == "official_dated_report_assumed_pregame":
                if path.suffix.lower() != ".pdf" or known != kick - pd.Timedelta(minutes=30):
                    return "invalid_dated_pdf_assumption"
                if row.get("report_date") != kick.tz_convert("America/Chicago").strftime("%Y-%m-%d"):
                    return "pdf_report_date_not_game_date"
                receipt = json.loads(path.with_suffix(".pdf.source.json").read_text())
                if receipt["sha256"] != hashes[key] or url_key(receipt["url"]) != url_key(row["source_url"]):
                    return "pdf_receipt_mismatch"
                import subprocess
                page = str(int(row["pdf_page"]))
                pdf_key = (key, page, "pdf_page_text")
                if pdf_key not in versions:
                    versions[pdf_key] = " ".join(subprocess.check_output(
                        ["pdftotext", "-f", page, "-l", page, "-layout", str(path), "-"]).decode().split())
                content = versions[pdf_key]
                day = timestamp(row["report_date"] + "T00:00:00Z")
                printed_date = f"{day.month}.{day.day}.{day.year % 100:02d}"
                if printed_date not in content or " ".join(row["source_quote"].split()) not in content:
                    return "dated_pdf_quote_or_date_not_in_source"
                return None
            if row.get("source_kind") == "conference_public_archive":
                payload = json.loads(path.read_text())
                source_row = payload["data"][int(row["source_row"])]
                if source_row != row.get("source_record"):
                    return "archive_record_mismatch"
                from cfbpred.conference_availability import archive_facts
                archive_key = (key, "assumed_archive_facts")
                if archive_key not in versions:
                    versions[archive_key] = archive_facts(path, schedule.reset_index())[0]
                expected = versions[archive_key]
                fields = ("name", "team", "game_id", "value", "known_at", "source_row")
                if not any(all(r.get(k) == row.get(k) for k in fields) for r in expected):
                    return "assumed_archive_fact_mismatch"
                return None
            # Explicit opt-in for dated pregame articles with missing revision
            # metadata. This is an assumption, not a proved historical vintage.
            if row.get("timing_policy") != "user_authorized_pregame_publication":
                return "unrecognized_timing_assumption"
            published = timestamp(row["published_at"])
            if published > known or published >= kick:
                return "publication_not_pregame"
            if row.get("modified_at") and timestamp(row["modified_at"]) >= kick:
                return "known_postgame_revision"
            if path.suffix.lower() == ".pdf":
                parent = Path(row["parent_article_path"])
                if sha(parent) != row["parent_article_sha256"]:
                    return "parent_article_hash_mismatch"
                article = json.loads(parent.read_text())["data"]
                if (timestamp(article["published_at"]) != published or
                        url_key(article["permalink"]) != url_key(row["parent_article_url"])):
                    return "parent_article_publication_mismatch"
                # The dated official article must actually link this PDF.
                import re
                urls = re.findall(r'https?[^\s"<>]+', json.dumps(article).replace("\\/", "/"))
                if url_key(row["source_url"]) not in {url_key(u.rstrip("\\")) for u in urls}:
                    return "pdf_not_linked_by_parent_article"
                import subprocess
                quote_key = (key, "pdf_text")
                if quote_key not in versions:
                    versions[quote_key] = "\n".join(" ".join(subprocess.check_output(
                        ["pdftotext", *flags, str(path), "-"]).decode().split()) for flags in ([], ["-layout"]))
                if " ".join(row["source_quote"].split()) not in versions[quote_key]:
                    return "quoted_text_not_in_pdf"
            elif path.suffix.lower() == ".json":
                article = json.loads(path.read_text()).get("data", {})
                if (url_key(article.get("permalink", "")) != url_key(row["source_url"])
                        or timestamp(article.get("published_at")) != published):
                    return "assumed_publication_not_in_source"
            else:
                return "assumed_source_format_not_supported"
            return None
        if path.suffix.lower() == ".pdf" and not row.get("captured_at"):
            return "pdf_metadata_is_not_publication_evidence"
        if row.get("published_at") and row.get("modified_at"):
            published, modified = timestamp(row["published_at"]), timestamp(row["modified_at"])
            # Scheduled WordPress posts can be last edited as drafts before
            # publication. Knowledge is the later of the two publisher times.
            if max(published, modified) > known:
                return "publisher_timestamp_after_claimed_knowledge"
            if key not in versions:
                versions[key] = publisher_versions(path)
            if (url_key(row["source_url"]), published, modified) not in versions[key]:
                return "publisher_timestamps_not_in_archived_source"
        elif row.get("captured_at"):
            captured = timestamp(row["captured_at"])
            if captured > known or not row.get("capture_url"):
                return "capture_does_not_prove_claimed_knowledge"
        elif row.get("observed_at") and timestamp(row["observed_at"]) <= known:
            # An extractor's proposed observation time needs the collector's
            # archived receipt. It cannot backdate a newly downloaded report.
            receipt = path.with_suffix(path.suffix + ".source.json")
            if not receipt.exists():
                return "collection_timestamp_unverified"
            meta = json.loads(receipt.read_text())
            digest = meta.get("sha256", meta.get("source_sha256"))
            if (timestamp(meta.get("observed_at")) != timestamp(row["observed_at"])
                    or digest != hashes[key]):
                return "collection_receipt_does_not_match_fact"
        else:
            return "original_publication_version_unverified"
        return None
    except (KeyError, ValueError, TypeError, OSError, OverflowError) as exc:
        return "invalid_fact:" + type(exc).__name__


ARCHIVE_POLICIES = {"official_report_printed_date", "official_report_printed_update_time",
                    "official_weekly_compilation_assumed_pregame", "third_party_revision_timestamp",
                    "web_archive_capture"}


def _flat(text):
    return " ".join(str(text or "").split())


def validate_archived_report(row, game, kick, known, path, key, hashes, versions, schedule):
    """Re-derive an official-archive fact from the retained bytes.

    Each policy names its own evidence. Nothing in the submitted row is
    trusted where the archived document can answer: the page is re-read, the
    parser re-run, and the build or revision time recomputed.
    """
    from cfbpred.availability_archive import (pdf_blocks, pdf_created, pdf_text, norm_team,
                                              printed_update_time, reporting_team)
    policy = row["timing_policy"]
    if policy == "web_archive_capture":
        # The Internet Archive's capture time is the knowledge time: it is
        # embedded in the capture URL, which must be the URL actually fetched.
        if row.get("timing_assumed"):
            return "capture_timestamp_is_not_an_assumption"
        import re
        stamp = re.search(r"/web/(\d{14})id_/", str(row.get("capture_url", "")))
        if not stamp:
            return "capture_url_without_timestamp"
        captured = pd.Timestamp(datetime.strptime(stamp.group(1), "%Y%m%d%H%M%S"), tz="UTC")
        if captured != timestamp(row["captured_at"]) or captured != known:
            return "capture_time_not_knowledge_time"
        if not captured < kick:
            return "capture_after_kickoff"
        receipt = json.loads(path.with_suffix(path.suffix + ".source.json").read_text())
        if receipt["sha256"] != hashes[key] or receipt["url"] != row["capture_url"]:
            return "capture_receipt_mismatch"
        quote = _flat(row.get("source_quote"))
        reported = _flat(row.get("reported_name") or row["name"])
        if reported not in quote or _flat(row.get("raw_status")) not in quote:
            return "capture_quote_lacks_name_or_status"
        return None
    if policy == "third_party_revision_timestamp":
        if row.get("timing_assumed"):
            return "revision_timestamp_is_not_an_assumption"
        index_path = Path(row["revision_index"])
        commits = {}
        pages = [p for p in sorted(index_path.parent.glob("commits_page_*.json"))
                 if not p.name.endswith(".source.json")]
        for page in pages:
            receipt = json.loads(page.with_suffix(".json.source.json").read_text())
            if receipt["sha256"] != sha(page):
                return "commit_index_receipt_mismatch"
            for item in json.loads(page.read_text()):
                commits[item["version"]] = pd.Timestamp(item["committed_at"])
        stamp = commits.get(row.get("revision_version"))
        if stamp is None or stamp != known:
            return "revision_not_attested_at_known_time"
        if not known < kick:
            return "revision_after_kickoff"
        receipt = json.loads(path.with_suffix(".json.source.json").read_text())
        if receipt["sha256"] != hashes[key] or row.get("revision_version") not in receipt["url"]:
            return "revision_body_receipt_mismatch"
        body = json.loads(path.read_text())
        day = kick.tz_convert("America/Chicago").strftime("%Y-%m-%d")
        names = {game.home_team, game.away_team}
        reported = row.get("reported_name") or row["name"]
        for entry in body.get("games", []):
            if entry.get("dateISO") != day:
                continue
            for side in ("home", "away"):
                team = entry.get(side) or {}
                if norm_team(team.get("name"), names) != row["team"]:
                    continue
                for player in team.get("players", []):
                    if _flat(player.get("name")) == _flat(reported) and player.get("status") == row.get("raw_status"):
                        return None
        return "player_status_not_in_revision"
    if not row.get("timing_assumed"):
        return "archive_report_timing_must_be_tagged_assumed"
    if path.suffix.lower() != ".pdf":
        return "archive_report_requires_pdf"
    receipt = json.loads(path.with_suffix(".pdf.source.json").read_text())
    if receipt["sha256"] != hashes[key] or url_key(receipt["url"]) != url_key(row["source_url"]):
        return "pdf_receipt_mismatch"
    if known != kick - pd.Timedelta(minutes=30):
        return "archive_report_not_at_t_minus_30"
    if row.get("report_date") != kick.tz_convert("America/Chicago").strftime("%Y-%m-%d"):
        return "report_date_not_game_date"
    page = int(row["pdf_page"])
    page_key = (key, page, "archive_page")
    if page_key not in versions:
        versions[page_key] = _flat(pdf_text(path, page, page) if row["source_kind"] != "cusa_pdf" else pdf_text(path))
    text = versions[page_key]
    if _flat(row["source_quote"]) not in text:
        return "quote_not_in_archived_page"
    printed = row.get("printed_report_date")
    if row.get("game_match_rule") == "unique_season_matchup":
        # Bowl and title-game pages print no usable date; the tie is that the
        # two programs meet exactly once in the season. Re-check that here.
        pair = {game.home_team, game.away_team}
        meetings = schedule[(schedule.season == game.season)
                            & schedule.home_team.isin(pair) & schedule.away_team.isin(pair)]
        if len(meetings) != 1:
            return "season_matchup_not_unique"
        if printed and _flat(printed) not in text:
            return "printed_label_not_in_archived_page"
    elif not printed or _flat(printed) not in text:
        return "printed_date_not_in_archived_page"
    blocks_key = (key, "archive_blocks")
    if blocks_key not in versions:
        versions[blocks_key] = list(pdf_blocks(path, row["source_kind"]))
    names = {game.home_team, game.away_team}
    reported = row.get("reported_name") or row["name"]
    found = False
    for number, block in versions[blocks_key]:
        if number != page or reporting_team(block, game, names, row["source_kind"]) != row["team"]:
            continue
        if any(_flat(n) == _flat(reported) and _flat(status) == _flat(row["raw_status"])
               for n, _, status, _ in block["rows"]):
            found = True
            stated = printed_update_time(block)
            break
    if not found:
        return "parser_does_not_reproduce_fact"
    if policy == "official_report_printed_update_time":
        if stated is None or not stated < kick:
            return "printed_update_time_not_before_kickoff"
    elif policy == "official_report_printed_date":
        created = pdf_created(path)
        if created is None or not created < kick:
            return "document_not_built_before_kickoff"
    elif policy == "official_weekly_compilation_assumed_pregame":
        if url_key(row["source_url"])[0] != "bigten.org" or "/api/media/file/" not in row["source_url"]:
            return "weekly_compilation_policy_is_big_ten_only"
    return None


def fact_key(row):
    return tuple(str(row.get(k, "")) for k in
                 ("season", "team", "game_id", "name", "kind", "value", "known_at", "source_url"))


def merge(source, baseline, additions, output):
    """Freeze an expanded reviewed registry, retaining rejected rows separately."""
    if output.exists():
        raise ValueError("Backfill snapshots are immutable; choose a new output directory")
    paths = json.loads(source.read_text())
    schedule = pd.read_parquet(Path(paths["inputs"]) / "snapshot_schedule.parquet").set_index("game_id")
    catalog = pd.read_parquet(Path(paths["players"]) / "player_seasons.parquet")
    rows = json.loads(baseline.read_text())
    seen = {fact_key(row) for row in rows}
    hashes, versions, accepted, rejected = {}, {}, [], []
    duplicates = 0
    for addition in additions:
        for row in json.loads(addition.read_text()):
            key = fact_key(row)
            if key in seen:
                duplicates += 1
                continue
            reason = validate_fact(row, schedule, hashes, versions)
            if reason:
                rejected.append({**row, "reason": reason, "submission": str(addition.resolve())})
                continue
            seen.add(key)
            accepted.append(row)
    linked, unmatched = resolve_names(accepted, catalog)
    rejected.extend(unmatched)
    accepted = [{k: v for k, v in row.items() if k != "player_ref"} for row in linked]
    output.mkdir(parents=True)
    save_json(output / "reviewed.json", rows + accepted)
    save_json(output / "accepted.json", accepted)
    save_json(output / "quarantine.json", rejected)
    report = dict(baseline_sha256=sha(baseline), baseline_facts=len(rows),
                  added_facts=len(accepted), rejected_facts=len(rejected), duplicates=duplicates,
                  added_availability_facts=sum(r["kind"] == "availability" for r in accepted),
                  additions=[dict(path=str(p.resolve()), sha256=sha(p)) for p in additions],
                  baseline_retained=True, production_modified=False)
    save_json(output / "merge.json", report)
    return report


def coverage_ledger(schedule, coverage, facts, as_of, attempts=None):
    """Create both sides of every FBS-involved game, including unsearched games."""
    schedule = schedule.copy()
    schedule["kick"] = pd.to_datetime(schedule.kick, utc=True)
    if schedule.game_id.duplicated().any():
        raise ValueError("Schedule contains duplicate game IDs")
    schedule = schedule[schedule.home_division.eq("fbs") | schedule.away_division.eq("fbs")]
    pieces = []
    for side, other in [("home", "away"), ("away", "home")]:
        frame = schedule[["game_id", "season", "kick", "conference_game"]].copy()
        frame["side"] = side
        for name in ("team", "conference", "division"):
            frame[name] = schedule[side + "_" + name]
        frame["opponent"] = schedule[other + "_team"]
        frame["opponent_division"] = schedule[other + "_division"]
        frame["historical"] = frame.kick < timestamp(as_of)
        frame["completed"] = frame.historical & schedule.home_points.notna() & schedule.away_points.notna()
        pieces.append(frame)
    ledger = pd.concat(pieces, ignore_index=True)
    keep = ["game_id", "team", "roster_players", "reports_known", "depth_known"]
    ledger = ledger.merge(coverage[keep], on=["game_id", "team"], how="left", validate="one_to_one")
    ledger["roster_in_model"] = ledger.roster_players.notna()
    for col in ("roster_players", "reports_known", "depth_known"):
        ledger[col] = pd.to_numeric(ledger[col], errors="raise").fillna(0).astype(int)
    reports = facts[facts.kind.eq("availability") & facts.game_id.notna()].copy()
    if len(reports):
        counts = reports.groupby(["game_id", "team"]).agg(
            availability_source_urls=("source_url", "nunique"),
            earliest_known_at=("known_at", "min"), latest_known_at=("known_at", "max"))
        ledger = ledger.merge(counts, on=["game_id", "team"], how="left", validate="one_to_one")
    else:
        ledger["availability_source_urls"] = 0
        ledger["earliest_known_at"] = pd.NaT
        ledger["latest_known_at"] = pd.NaT
    ledger["availability_source_urls"] = ledger.availability_source_urls.fillna(0).astype(int)
    ledger["search_status"] = "not_yet_searched"
    ledger["search_attempts"] = 0
    if attempts is not None and len(attempts):
        allowed = {"not_found_in_searched_sources", "source_found_unverified", "source_unreachable", "facts_verified_partial"}
        if not attempts.status.isin(allowed).all():
            raise ValueError("Unknown search-attempt status")
        attempts = attempts.copy()
        attempts["searched_at"] = pd.to_datetime(attempts.searched_at, utc=True)
        attempts = attempts[attempts.searched_at <= timestamp(as_of)]
        counts = attempts.groupby(["game_id", "team"]).size().rename("attempt_count")
        latest = attempts.sort_values("searched_at").drop_duplicates(["game_id", "team"], keep="last")
        ledger = ledger.merge(latest[["game_id", "team", "status"]], how="left", on=["game_id", "team"], validate="one_to_one")
        ledger = ledger.merge(counts, how="left", on=["game_id", "team"], validate="one_to_one")
        ledger["search_status"] = ledger.status.fillna(ledger.search_status)
        ledger["search_attempts"] = ledger.attempt_count.fillna(0).astype(int)
        ledger = ledger.drop(columns=["status", "attempt_count"])
    ledger.loc[ledger.reports_known.gt(0), "search_status"] = "facts_verified_partial"
    ledger["all_roster_availability_verified"] = False
    ledger["unknown_roster_statuses"] = (ledger.roster_players - ledger.reports_known).clip(lower=0)
    return ledger.sort_values(["season", "kick", "game_id", "side"]).reset_index(drop=True)


def ledger(source, audit_path, output, as_of, attempts_path=None):
    paths = json.loads(source.read_text())
    schedule = pd.read_parquet(Path(paths["inputs"]) / "snapshot_schedule.parquet")
    coverage = pd.read_parquet(audit_path / "game_coverage.parquet")
    facts = pd.read_parquet(Path(paths["timeline"]) / "facts.parquet")
    attempts = pd.DataFrame(json.loads(attempts_path.read_text())) if attempts_path else None
    frame = coverage_ledger(schedule, coverage, facts, as_of, attempts)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / "fbs_game_availability.parquet", index=False)
    frame.to_csv(output / "fbs_game_availability.csv", index=False)
    hist = frame[frame.completed & frame.division.eq("fbs")]
    season = hist.groupby("season").agg(
        fbs_games=("game_id", "nunique"), fbs_team_games=("game_id", "size"),
        player_statuses=("reports_known", "sum"),
        team_games_with_statuses=("reports_known", lambda v: int(v.gt(0).sum())))
    season.to_csv(output / "season_coverage.csv")
    game_sides = frame[frame.completed].groupby("game_id").reports_known.apply(lambda v: int(v.gt(0).sum()))
    report = dict(as_of=timestamp(as_of).isoformat(), source_sha256=sha(source),
                  scope="Every FBS-involved game in the pinned schedule; both participants, by historical division",
                  season_min=int(frame.season.min()), season_max=int(frame.season.max()),
                  completed_fbs_involved_games=int(hist.game_id.nunique()),
                  completed_fbs_team_games=len(hist),
                  fbs_team_games_with_verified_statuses=int(hist.reports_known.gt(0).sum()),
                  verified_fbs_player_game_statuses=int(hist.reports_known.sum()),
                  completed_games_with_both_sides_reported=int(game_sides.eq(2).sum()),
                  completed_fbs_team_games_without_statuses=int(hist.reports_known.eq(0).sum()),
                  completed_fbs_team_games_without_roster=int(hist.roster_players.eq(0).sum()),
                  complete_roster_verification_claimed=False,
                  season_coverage=season.reset_index().to_dict("records"))
    save_json(output / "summary.json", report)
    # Source-first batches amortize collection across a conference/team season.
    work = hist.groupby(["season", "conference", "team"], dropna=False).agg(
        games=("game_id", "size"), games_with_statuses=("reports_known", lambda v: int(v.gt(0).sum())),
        game_ids=("game_id", lambda v: [int(x) for x in v]))
    work["priority"] = [0 if season >= 2023 else 1 if season >= 2014 else 2 for season, _, _ in work.index]
    work = work.reset_index().sort_values(["priority", "season", "conference", "team"], ascending=[True, False, True, True])
    work["state"] = "pending"
    work.to_json(output / "team_season_work_queue.jsonl", orient="records", lines=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("merge")
    collect.add_argument("--source", type=Path, required=True)
    collect.add_argument("--baseline", type=Path, required=True)
    collect.add_argument("--additions", type=Path, nargs="+", required=True)
    collect.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("ledger")
    audit.add_argument("--source", type=Path, required=True)
    audit.add_argument("--audit", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--as-of", required=True)
    audit.add_argument("--attempts", type=Path)
    args = parser.parse_args()
    if args.command == "merge":
        result = merge(args.source, args.baseline, args.additions, args.output)
    else:
        result = ledger(args.source, args.audit, args.output, args.as_of, args.attempts)
    print(json.dumps(result, indent=2))
