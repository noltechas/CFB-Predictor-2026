# Public historical FBS availability

The accepted full-history/context model is retained in `models/experiments/player_context_20260921_v1/RETAINED_FOUNDATION.json`, including hashes of its source and saved models. Its market-matched 2025 margin MAE is 12.3292, versus 12.3632 for the eight-game control. Expanding the data does not overwrite that model or its comparison. This report describes the collection/training pass; the expanded candidate was subsequently [published with the user's authorization](FULL_CONTEXT_PRODUCTION_RELEASE_2026-09-21.md), including a full 2026 prediction rerun and weekly rankings.

## What the original experiment actually used

There were **33 distinct source articles, 32 games, and 220 distinct player-game availability observations**. Thirty-one games concerned Ohio State and one concerned Florida. The normalized fact table has 221 availability rows because two sources describe the same player/game status.

| Season | Games with a reported status | Distinct player-game statuses | Use in the 2025 experiment |
|---|---:|---:|---|
| 2023 | 9 | 64 | Training |
| 2024 | 13 | 63 | Training |
| 2025 | 10 | 93 | Pregame inputs for assessment games |
| Total | 32 | 220 | 127 training statuses, 93 assessment statuses |

The saved 2026 candidate refits through 2025 and therefore can use all 220 historical statuses for training. Availability is only one part of the accepted change: full career encoding, dated positions/membership, and expected contribution weighting changed together. The 0.034-point result cannot be attributed solely to these injury reports.

## Coverage means every scheduled game has a record

The schedule-first ledger covers both participants in every FBS-involved game in the pinned schedule, using each season's division and conference. Its completed-game denominator is **20,077 games and 37,706 FBS team-games**, from 2001 through the September 2026 snapshot. This is the project's collected schedule, not all college football history before 2001. Opponents from other divisions are retained as separate participant rows.

The ledger is `data/availability_backfill/baseline_20260921/fbs_game_availability.parquet` (also CSV). Every row distinguishes roster availability, explicit player statuses, source counts, search attempts and unknowns. Zero reports means unknown. A few known injured players do not establish the status of every roster member. The initial model has no roster players for 4,552 completed FBS team-games, mostly in the earliest years; report discovery and roster recovery are separate gaps.

Work is queued by historical conference, team and season in `team_season_work_queue.jsonl`. Unsearched games remain `not_yet_searched`; finding nothing in a few sources does not prove no report ever existed. Archive failures must not mark an entire conference season as searched.

## Collection and review completed in this pass

Three **gpt-5.6-luna** agents investigated conference archives, additional teams, and older Ohio State reports. Their archived results are under `data/raw/availability_backfill/`. Parent review checks the actual source text and independently verifies the publisher metadata.

The final expanded registry is `data/availability_backfill/merged_20260921_v2/reviewed.json`. It preserves every baseline fact and adds **200 identity-resolved availability facts** from 17 additional articles/games. New observations concern Ohio State, Rutgers, Tennessee, Oklahoma, Arkansas, Alabama and Mississippi State; the retained baseline also includes Florida. The first merge admitted 197 rows; a second merge repaired three explicit `DL` prefixes mistakenly included in names. Seven further rows remain unmatched rather than being merged by a guessed nickname or spelling.

The final compiled timeline contains **420 distinct player-game statuses, 49 games, 53 FBS team-games, eight teams and 50 source articles**. Four games now have at least some verified statuses for both participants. This does not establish every player's availability even in those games. The final ledger is `data/availability_backfill/expanded_20260921/`: 37,622 completed FBS team-games remain unsearched, 31 have a located source awaiting verification, and 53 have partial verified statuses. Its queue contains 3,259 team-season batches across 145 historical FBS programs. **Full historical FBS verification is not complete.**

| Team | Games with verified statuses | Distinct player-game statuses |
|---|---:|---:|
| Ohio State | 39 | 296 |
| Rutgers | 4 | 60 |
| Tennessee | 5 | 28 |
| Oklahoma | 1 | 12 |
| Arkansas | 1 | 10 |
| Mississippi State | 1 | 8 |
| Alabama | 1 | 5 |
| Florida | 1 | 1 |

The game counts in this table sum to team-games; an SEC matchup can appear under both participants. The 2025 training split now has 327 status observations from earlier seasons; its 93 assessment-season observations are unchanged. The current-season refit can use all 420.

Public source families that yielded usable additions:

- [Buckeye Sports public WordPress archive](https://www.buckeyesports.com/wp-json/wp/v2/posts): publication and revision metadata, explicit status lists, and opponent checks. Newly admitted reports include 2022 and 2023.
- [On the Banks Rutgers report](https://www.onthebanks.com/2024/9/7/24238264/rutgers-football-availability-report-scarlet-knights-down-multiple-starters): four 2024 articles with pregame JSON-LD publication/revision timestamps.
- [Rocky Top Talk Tennessee–Oklahoma report](https://www.rockytoptalk.com/2024/9/19/24248904/tennessee-vols-football-oklahoma-sooners-wednesday-injury-report): five 2024 articles, several containing both teams' lists, with pregame JSON-LD timestamps.

The agents also archived official conference PDFs and additional articles. PDF creation/modification dates are **not publication proof**. An official Florida–Kentucky article exposes a story date without timezone or revision history, so its 17 candidate rows remain outside historical training. Nine Game-Time Decision headings remain in raw source/quarantine pending a distinct encoding; they are not silently converted to questionable or out. Some official archives expose only current reports or lack historical timestamps, and some URLs return anti-bot pages or errors. Raw responses and these limitations are retained.

## Retraining result

The expanded snapshot and both trained models are saved under `models/experiments/availability_backfill_20260921_v2/`. A checked comparison (`comparison.json` and `comparison.md`) confirms identical model code, seed, base inputs and the same 14-epoch budget selected by the clean control. Only the availability evidence changes.

| Candidate | Market-matched 2025 margin MAE | Winner log loss |
|---|---:|---:|
| Retained full context | 12.329227 | 0.536959 |
| Expanded public availability | 12.328706 | 0.536794 |

The additional error reduction is about 0.0005 points: effectively unchanged performance on this reused 807-game market-matched benchmark. The paired FBS/FBS MAE change is −0.000586, with a week-cluster 95% interval of [−0.004624, +0.003231]. The previously accepted improvement and saved model remain intact. More coverage is now available, but this small collection pass does not demonstrate a material additional accuracy gain.

The expanded 2026 model refits through 2025, reloads its checkpoint, and scores all **1,142 future modeled FBS/FCS-involved games** after 2026-09-22 00:17:36 UTC in the pinned schedule. This training pass did not use 2026 outcomes or modify production; publication was performed separately afterward. The candidate's role export is `runs/backfill_2026_s0/player_game_roles.parquet`.

The full test suite passed **162 tests with one pre-existing skip**. Dataset audits additionally checked the actual source timestamps, original-model hashes, preservation of all baseline facts, resolved player identities, and prediction lineage. An independent Luna review checked the Buckeye Sports and Rocky Top Talk source/status mappings after parent review.

## The lowest-cost route to broader coverage

1. **Fetch source archives in batches.** Start with conference-season archives and publisher APIs/sitemaps. One downloaded report may cover many teams, and one archive query may cover an entire season. Deduplicate article IDs and content hashes. Cache successes and failures; retry only when there is a new route or changed source. Validate returned date ranges and pagination rather than assuming the server honored query filters.
2. **Use deterministic extraction first.** Parse structured JSON, tables and explicit status lists. Keep the original status, team heading, player spelling, source locator, raw file, URL, publication/revision times, retrieval time and hash. Save revisions separately. A parser must match the game from publication and the stated opponent before checking revision time; matching only the next game after a late revision can put an old report on the wrong game.
3. **Spend Luna effort on exceptions.** Give agents bounded source/team-season batches and a fixed output schema, not one open-ended search per game. Use them for archive discovery, unusual layouts and corroboration. Their proposed facts go through automated timestamp/hash/game/identity checks and source review before training. This pass caught incorrect PDF timing and status mappings, which is why agent assertions alone are insufficient.
4. **Prioritize coverage breadth, then go backward.** Recover recent mandatory-reporting seasons across conferences, then older official team news, game notes and contemporaneous beat reports. The [Big Ten introduced its system in 2023](https://bigten.org/fb/article/blt2856785fb75ee868/); the [Big 12 began in 2025–26](https://big12sports.com/sports/2025/8/21/availability-reporting.aspx). These policy dates guide discovery; they do not supply a publication timestamp for an individual archived report. Include Group of Five programs and nonconference games explicitly.
5. **Keep each unresolved game visible.** Search progress, a recovered report, partial player status coverage and a completely verified roster are separate measures. No public source can guarantee retrospective availability for a game whose report was never published or preserved. The honest outcome for such a game is an explicit unknown, with the searched sources recorded.

## Availability and role training

Pregame availability features use only evidence known by the chosen prediction cutoff. The current historical comparison uses pre-kickoff evidence. Future evaluations should separately freeze daily/24-hour and near-kickoff forecasts and compare each with market lines observed at the same time. A final injury report cannot be used to grade a prediction supposedly made a day earlier.

Gamebooks and official participation/starting-lineup tables are a complementary collection track. They can supply **postgame labels** for an expected-participation or role model, and become prior-game history for later forecasts. They cannot establish what was known before that same game. Absence from a statistical box score is not proof that a lineman did not play. Missing participation lists or snap counts need target masks; estimated shares should not be represented as observed snaps.

After broader coverage, train availability/participation and role heads on past seasons, with explicit masks for unknown evidence. Evaluate full-history only, role weighting, and report features separately; report results by conference, year and report coverage. Maintain chronological splits and the accepted baseline. The current contribution weights remain learned within-position emphasis, not calibrated snap shares or probabilities of playing.

## Reproducible integration

`src/cfbpred/availability_backfill.py` provides two commands:

```sh
PYTHONPATH=src python3 -m cfbpred.availability_backfill merge \
  --source models/experiments/player_context_20260921_v1/source_revision_checked.json \
  --baseline data/availability_backfill/merged_20260921_v2/reviewed.json \
  --additions PATH_TO_NEW_FACTS.json \
  --output data/availability_backfill/NEXT_REVIEWED_SNAPSHOT
```

The merge verifies the source hash, publisher dates in the archived payload, strict pregame knowledge, game/team/season, supported status and a unique program-season player identity. It preserves rejected rows and the baseline. Completed snapshots must use new directories. Current HTML publication dates alone and PDF metadata fail the historical version check.

```sh
PYTHONPATH=src python3 -m cfbpred.refresh_player_context \
  --source models/experiments/player_context_20260921_v1/source_revision_checked.json \
  --reviewed data/availability_backfill/merged_20260921_v2/reviewed.json \
  --raw data/raw/player_context/20260921T224546407855Z \
  --output models/experiments/NEXT_CONTEXT_SNAPSHOT --no-train
```

Omitting `--raw` collects public current sources again. A hash-checked `data/player_context_reviewed_latest.json` pointer supplies the expanded registry when `--reviewed` is omitted. This updates the on-demand data workflow; it does not schedule an automation or activate a production model.

```sh
PYTHONPATH=src python3 -m cfbpred.availability_backfill ledger \
  --source models/experiments/NEXT_CONTEXT_SNAPSHOT/source.json \
  --audit models/experiments/NEXT_CONTEXT_SNAPSHOT/audit \
  --attempts data/availability_backfill/search_attempts_20260921.json \
  --as-of YOUR_UTC_AUDIT_TIME \
  --output data/availability_backfill/NEXT_COVERAGE
```

Coverage and work-queue files are derived outputs. Original public sources, accepted facts, and model snapshots are retained independently so a later parser change cannot silently alter an earlier experiment.
