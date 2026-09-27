# Conference availability archives, 2023–2025

This pass collected every official conference availability report that still
survives as a retrievable document, added timestamped reproductions for the
seasons whose official archives were wiped, validated all of it, and retrained
the live v6 architecture on the result.

## What was added

10,421 validated player-game statuses across 1,278 team-games, all dated 2025
or earlier. The registry is
`data/availability_backfill/conference_archives_20260922_registry/reviewed.json`
(1,594 prior facts plus these additions, zero rejections), and
`data/player_context_reviewed_latest.json` now points to it.

| Source | Seasons | Statuses | Team-games | Timing evidence |
|---|---|---:|---:|---|
| Big Ten weekly PDFs (bigten.org media API) | 2023 | 1,290 | 178 | Printed week and matchup; T-30 assumption |
| Big Ten weekly PDFs | 2024 | 1,504 | 205 | Same |
| MAC PDFs (sidearm S3) | 2025 | 1,365 | 144 | Printed game date; file built before kickoff |
| Conference USA per-game PDFs (sidearm S3) | 2025 | 894 | 80 | Same |
| Mountain West PDFs (Wayback-listed, GCS bucket) | 2025 | 960 | 95 | Report's own "Updated as of" time before kickoff |
| Sun Belt feed (GitHub gist revisions) | 2025 | 1,048 | 104 | GitHub commit time of the exact bytes |
| Covers injury page (Wayback captures) | 2025 | 3,205 | 644 | Internet Archive capture time |
| Reproduction articles (SI, 247Sports, Inside NU, On the Banks, others) | 2024–25 | 155 | 26 | Publisher JSON-LD publication and revision times |

Raw documents, receipts (URL, retrieval time, SHA-256) and parse outputs are in
`data/raw/availability_archive/20260922T170527395474Z/`.

## Coverage against the official report universe

Team-games of each conference-season that fell under a mandatory reporting
policy, and how many now carry at least one filed report or status.

| Conference | Season | Scope | Team-games | Covered | Share |
|---|---|---|---:|---:|---:|
| Big Ten | 2023 | all games | 180 | 180 | 100% |
| Big Ten | 2024 | all games | 235 | 215 | 92% |
| Big Ten | 2025 | all games | 234 | 87 | 37% |
| SEC | 2024 | conference | 130 | 18 | 14% |
| SEC | 2025 | conference | 134 | 92 | 69% |
| ACC | 2025 | conference | 140 | 94 | 67% |
| Big 12 | 2025 | conference | 148 | 107 | 72% |
| MAC | 2025 | all games | 163 | 148 | 91% |
| Conference USA | 2025 | all games | 153 | 94 | 61% |
| Mountain West | 2025 | conference | 98 | 95 | 97% |
| Sun Belt | 2025 | conference | 116 | 106 | 91% |
| **Total** | | | **1,731** | **1,236** | **71%** |

The per-team-game ledger, including every uncovered game, is
`data/availability_backfill/conference_archives_20260922/coverage_ledger.json`.

### Why the rest is missing

- **SEC 2024.** Reports were published in one Google Sheet overwritten weekly.
  Only the final week survives; the 16 per-school workbooks are private (HTTP
  401), and the Wayback captures of the sheet are empty shells.
- **Big Ten, SEC, ACC, Big 12 2025.** Reports lived in the HD Intelligence
  widget, whose archive endpoints now return empty lists. Coverage comes from
  Covers captures (dense in October 2025 only) and articles.
- **Big Ten 2024 Week 6.** The conference never posted that week's file; three
  team-games were recovered from articles.
- **Conference USA non-conference games.** Early-season file names follow no
  guessable pattern.
- **Sun Belt September 2025.** The gist feed began on September 29.
- Article discovery hit a shared 200-search limit for the session. Weekly
  roundups (Saturday Down South, Saturday Tradition) are narrative, and
  narrative sentences are never read as statuses.

## Validation

Every addition passed `availability_backfill.merge`, which now re-derives each
fact from the archived bytes under an explicit policy:

- `official_report_printed_date`: page re-read, parser re-run, quote and printed
  date found on the page, file build time recomputed and before kickoff.
- `official_report_printed_update_time`: the report's own printed update time
  must precede kickoff.
- `official_weekly_compilation_assumed_pregame`: Big Ten only. The weekly file is
  compiled after the week's games, so its build time is provenance, not a gate.
  An independent check found 145 of 145 overlapping statuses identical to
  verified pre-kickoff articles.
- `third_party_revision_timestamp`: GitHub commit time from the archived commit
  pages equals the knowledge time; the player and status are in that revision.
- `web_archive_capture`: capture time embedded in the fetched Wayback URL equals
  the knowledge time and precedes kickoff.

Parsing rules that matter: side-by-side OUT/QUESTIONABLE columns are split by
header position; partial-game designations are quarantined, never widened; for
each team-game only the last pre-kickoff report is kept, so a midweek status
never outlives a later report that dropped the player; Big Ten pages credit the
team printed in capitals (acronyms like USC do not count as shouting), falling
back to the fixture's Big Ten member. Identity: exact program-season name first,
then jersey to break ties, then a unique surname with a compatible first name
(Matt/Matthew); 94.7% of official statuses resolved.

Tests: `tests/test_availability_archive.py`.

## Retrain

Paired 2025 assessment on the 807-game market cohort, variant
`full_context_v2`, fixed 14 epochs, same stores and code as v6. Training is not
bit-for-bit repeatable (re-running v6's exact configuration moved MAE by
0.0035), so both arms ran three seeds.

| Metric | Baseline (3 seeds) | Candidate (3 seeds) | Change |
|---|---:|---:|---:|
| Margin MAE | 12.34736 | 12.34765 | +0.00029 |
| Margin RMSE | 15.72827 | 15.72588 | −0.00239 |
| CRPS | 8.83780 | 8.83658 | −0.00122 |
| WP log loss | 0.53697 | 0.53704 | +0.00007 |

Per-seed MAE changes were +0.0005, +0.0011 and −0.0008; the spread between
seeds is about 0.07. The result is neutral and fails the strictly-lower-MAE
rule, so v6 stays active and nothing was published.

The data now reaches training: team-games with a player reported out rose from
82 to 1,281 for 2014–2025 (51 to 403 inside the 2025 assessment's 2014–2024
training window). The model barely uses it. With a side's most-used QB reported
out, win probability moves 0.1 to 0.3 percentage points depending on the seed,
and the baseline and candidate of the same seed agree to within 0.01. The extra
examples did not change the model's sensitivity. The 2026 refit trained on the full registry is at
`models/experiments/availability_20260922_v1/runs/refit_2026_s0` (not staged).

## Next step

The limit is now architectural, not data: an absence only removes a player from
roster pooling, and a starting quarterback's removal moves win probability by a
fraction of a point. A candidate worth testing is a raw count input per side (the share of
recent passing attempts, carries, targets and tackles belonging to players
reported out), in the same spirit as returning production.

## Follow-up pass (same day): per-game search, sitemaps, and 2026 collection

**Per-game search.** Twelve Haiku agents searched the 828 team-games still
missing (2024–2026). Web search allows about 200 queries per session in total,
shared by all agents, so they checked roughly a quarter of the list and returned
52 verified URLs for 46 team-games. Agents only supplied URLs; every status was
read from the archived page against the program's roster.

**Sitemaps (no search needed).** SI.com team-site and 247Sports monthly
sitemaps listed 2,035 candidate availability articles for 2024–2026. Structured
statuses (never narrative sentences) added 746 validated statuses for 120
otherwise uncovered team-games. Articles only fill team-games with no other
source, and only the latest article per team-game is kept. 448 articles were
revised after kickoff and cannot prove their pregame text; Internet Archive
captures could rescue none of the relevant ones. 247Sports renders article
bodies client-side, so most of its pages are unreadable without a browser.
Registry: `data/availability_backfill/conference_archives_20260922_registry_v2/`.

| Conference | Season | Covered team-games |
|---|---|---|
| Big Ten | 2023 | 180 of 180 |
| Big Ten | 2024 | 215 of 235 |
| Big Ten | 2025 | 89 of 234 |
| SEC | 2024 | 72 of 130 |
| SEC | 2025 | 121 of 134 |
| ACC | 2025 | 101 of 140 |
| Big 12 | 2025 | 110 of 148 |
| MAC | 2025 | 148 of 163 |
| Conference USA | 2025 | 94 of 153 |
| Mountain West | 2025 | 95 of 98 |
| Sun Belt | 2025 | 106 of 116 |
| **2023–2025 total** | | **1,331 of 1,731 (77%)** |

**2026.** Every played conference game in the Big Ten, SEC, ACC, Big 12, MAC and
American has its report (85 team-games). Missing: Conference USA (0 of 31; the
league moved from PDFs to a widget with no readable feed) and the Mountain West's
first conference game (0 of 4). Most unreported 2026 games are non-conference
games, which those leagues do not report.

Collection going forward, all wired into `refresh_player_context` (nightly
refresh and the pre-kickoff dispatcher) and `refresh_scores.sh` (every 30 min):

- Official conference feed (existing), now matching a team whose printed
  opponent is itself (the Big 12 feed's Kansas vs. "Kansas").
- `sunbelt_availability.py`: every GitHub revision of the Sun Belt feed. GitHub
  keeps revisions, so nights the job misses are recovered later.
- `covers_availability.py`: live captures of the Covers injury page for every
  FBS program. Used only for team-games no official source reports; unchanged
  captures are logged as re-observations rather than stored again.

An end-to-end no-publish refresh passed its audit with these sources (1,142
future games, all FBS rosters present, player history strictly pregame).

The remaining gaps are listed per team-game in
`data/availability_backfill/missing_team_games.json` (400 for 2023–2025, 35
played in 2026) for a follow-up search in a fresh session.

## Second per-game pass (2026-09-22, later)

Input: the 435 team-games in `missing_team_games.json` (above), searched P4 first,
then 2026 Conference USA and Mountain West.

- **Haiku game-level search:** six agents used about 180 searches (near the
  session cap) and returned 15 candidate URLs. Nine of them produced validated
  statuses; seven of those nine had also been found in the sitemaps.
- **Team-site sitemaps (no search needed):** SB Nation (`/sitemaps/entries/YYYY/M`)
  and FanSided / Minute Media (`articles-sitemap.xml?quarter_year=qNyYYYY`) team
  sites, alongside the earlier SI.com and 247Sports lists.
- Same rules as before: agents supplied URLs only; every page was re-fetched and
  archived with a receipt; the publisher's publication and revision times both
  had to precede kickoff; statuses were read only in structured forms beside a
  full roster name; only team-games with no other source were filled, using the
  latest article per team-game. Pages with no revision time (Whole Hog Sports,
  Rebel Walk, Nittany Sports Now, older Rocky Top Insider) were excluded.
  `publisher_versions` now also reads a complete Open Graph
  `article:published_time` + `article:modified_time` pair when there is no JSON-LD.

Result: 510 validated statuses (0 rejected) covering 87 team-games. 45 of them
were on the list; the other 42 are uncovered games outside the conference
reporting policy (mostly non-conference). Registry:
`data/availability_backfill/conference_archives_20260922_registry_v3/`
(the pointer's previous target is kept in `data/audit_backups/`).

| Conference | Season | Listed | Found | Still missing |
|---|---|---|---|---|
| Big Ten | 2024 | 20 | 2 | 18 |
| SEC | 2024 | 58 | 14 | 44 |
| ACC | 2025 | 39 | 2 | 37 |
| Big 12 | 2025 | 38 | 3 | 35 |
| Big Ten | 2025 | 145 | 19 | 126 |
| Conference USA | 2025 | 59 | 0 | 59 |
| MAC | 2025 | 15 | 0 | 15 |
| Mountain West | 2025 | 3 | 0 | 3 |
| SEC | 2025 | 13 | 5 | 8 |
| Sun Belt | 2025 | 10 | 0 | 10 |
| Conference USA | 2026 | 31 | 0 | 31 |
| Mountain West | 2026 | 4 | 0 | 4 |
| **Total** | | **435** | **45** | **390** |

Coverage of 2023–2025 conference-policy team-games is now 1,376 of 1,731 (79.5%);
see `coverage_ledger_latest.json`. The 390 still missing are listed in
`data/availability_backfill/missing_team_games_after_pass2.json`. Nothing was
retrained or published.
