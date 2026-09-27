# Full-history model production release

Published September 21, 2026 at 9:13 p.m. Central (September 22 at 02:13 UTC).

The active pregame model is **player-full-context-tabm16-v3**. It is the retained expanded-public-availability candidate, refitted through 2025 with the previously selected 14 epochs. The deployed checkpoint has exactly the same fitted weights as `availability_backfill_20260921_v2/runs/backfill_2026_s0`; the production wrapper adds schema metadata. It contains one outer seed and 16 internal TabM members. No 2026 outcomes were used to fit or select it.

The original accepted improvement (2025 market-matched margin MAE 12.3632 to 12.3292) remains saved. The expanded candidate's MAE is 12.3287 on the same reused benchmark. Availability alone did not account for that improvement: complete career encoding, expected-role weighting, and dated player context changed together.

## Published outputs

- Recomputed and published **1,609 current-schedule 2026 predictions and distributions**, including completed games and all modeled future games. All FBS roster sides have nonempty player inputs.
- Recomputed and published **preseason and Weeks 1–3**, with 266 Division I entries and a common 138-team FBS reference pool per snapshot. Preseason has 265 eligible ranked teams and one explicitly unranked team; subsequent weeks rank all 266.
- Earlier weeks retain their saved historical cutoffs. Week 3 uses the current release cutoff and therefore includes the newly observed dated positions and membership information.
- Archived completed-game forecasts are **reconstructions**, not the forecasts originally issued before those games. Original production rows, distributions and ranking documents are backed up. Historical replays use currently archived raw data; complete original source vintages cannot be recovered for every input.
- Kept the separately validated raw-DNN in-game tower. This release changes pregame predictions and their ranking readout.

The old Campbell–Western Carolina fixture ID `401866625` remained in production after postponement. The current schedule contains replacement ID `401917058` for September 6. The obsolete prediction and distribution were archived and removed during publication; the replacement was scored normally. Official [postponement notice](https://catamountsports.com/news/2026/9/5/football-catamounts-camels-postponed-until-sunday.aspx) and [September 6 recap](https://gocamels.com/news/2026/9/6/football-worth-the-wait-camels-down-no-24-wcu-28-19-in-home-opener.aspx) corroborate the match.

## Verification

The production adapter was compared with all 1,142 saved future-game distributions: maximum probability difference was **2.24e-8**. This check caught and fixed a mapping issue: program identities needed for same-team role features must remain available even when the model masks their learned embeddings as unseen in training.

Production readback verified 1,609 prediction/PMF IDs, the new model version on every row, and zero scoreline mismatches. Maximum numeric differences were below the existing database tolerances (0.00005 points and 0.0000005 probability). All four ranking documents matched database readback exactly. The public API used by the app also returned the new version and all four weeks. The Python suite passed **164 tests, one skipped**.

Ranking replays choose the next scheduled target for each team, reject skipped games and mismatched prior result counts, reconstruct both career tips and current-season aggregates before the common cutoff, and recompile dated context using only earlier knowledge. Information observed on September 21 cannot enter preseason or Weeks 1–2. Missing reports remain unknown; role weights are learned contribution weights, not observed snaps or calibrated participation probabilities.

## Operations and rollback

`refresh_week.sh` now dispatches through the full-history production adapter. Its player refresh builds full career and dated context stores from the reviewed registry and newly collected public sources before scoring. The rankings command dispatches through the same active architecture. This changes the existing refresh implementation; no additional scheduled task was created.

An isolated end-to-end refresh also passed: it rebuilt scalar, player, sequence, full-history and dated-context stores, fetched current public sources, and scored/audited all 1,142 future games without promoting those validation outputs. Its report is `models/player_sequence_production/validation_refresh/20260922T021609627536Z/refresh_report.json`.

Release evidence is in `models/player_sequence_production/player-full-context-tabm16-v3/`: `publication.json`, `adapter_parity.json`, `production_audit.json`, `season_2026_predictions.parquet` and `rankings/`. The active pointer is `models/player_sequence_production/active.json`.

The release rollback directory is `models/player_sequence_production/release_backups/20260922T021325Z/`, containing the prior active pointer, refresh pointer, local rankings and database ranking documents. `latest_promotion.json` points to the local prediction/PMF rollback copy; the corresponding database backup is under `database_backups/`. Restore matching model, prediction/PMF and ranking versions together under the shared publish lock if rollback is needed. The published checkpoint's source registry contains 420 distinct historical player-game availability statuses across eight teams. The new Nebraska collection is a subsequent evidence batch, not part of this fitted checkpoint.
