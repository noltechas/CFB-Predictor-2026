# Roster position source release

Published September 22, 2026 at 09:07 a.m. Central (14:07 UTC).

The active pregame model is **player-full-context-tabm16-v5**. The only change from
`player-full-context-tabm16-v4` is how a roster player's position group is assigned. Architecture,
seeds, epoch budget, loss and every other input are unchanged. The live in-game tower
(`raw-dnn-delta-v22`) is untouched; this release changes pregame predictions and their ranking readout.

## The defect

Position groups came only from a player's dated recruiting link. Anyone without one fell to the
`UNK` pool even though the roster catalog lists a position, and six roster abbreviations (`PK`, `NT`,
`G`, `T`, `KR`, `PR`) had no entry in the alias table at all. A per-input attribution over 150 FBS
games found the `UNK` unit was the sixth most influential of 27 input blocks, with a mean absolute
effect of 3.7 percentage points of win probability, larger than the entire quarterback block.

| Population | Unknown before | Unknown after |
| --- | --- | --- |
| Catalog rows (372,454) | 65.5% | 8.5% |
| 2026 rostered players (31,537) | 62.9% | 4.9% |
| 2026 FBS rosters (138 teams) | 37.4% | 7 players |

Where both sources exist they agree 86.8% of the time; disagreements are scheme labels such as
linebacker versus defensive lineman. What remains unknown is rosters carrying no position, a `?`
placeholder, or `ATH`.

## The change

`player_inputs.assign_positions` takes a `position_source`: `scouting` (recruiting link only, what
v4 and every earlier checkpoint were fitted on) or `scouting_then_roster` (the roster listing fills
the gaps). The source is recorded in the player-store manifest. `POSITION_ALIASES` gained the six
missing roster abbreviations, which also affects dated timeline positions: slots carrying a dated
position rose from 245,904 to 253,154. Every other timeline figure is identical, including the
75,473 facts and 2,394 reviewed facts.

Two guards keep stores and checkpoints together. `player_sequence_production.check_position_source`
refuses to score a store whose source differs from the checkpoint's, read from the checkpoint's
embedded player manifest, with an absent field meaning `scouting`.
`refresh_player_sequence_production.active_position_source` makes the daily refresh build the store
the active checkpoint requires, so a rollback to v4 rebuilds a `scouting` store without a second switch.

## Assessment

The fix cannot be applied without refitting: feeding corrected positions to the already-fitted v4
made completed-2026 margin MAE worse, 12.610 to 12.715, because the network had learned with the
`UNK` pool.

The comparison is paired on a single raw vintage. CFBD's `roster.parquet` and
`player_game_stats.parquet` change daily, so v4's recorded 12.3271 is not comparable to anything
measured today; rebuilding v4's exact configuration on the September 22 vintage yields 12.3878.
Both sides therefore use the September 22 nightly refresh as their raw source, the same
`reviewed_and_current.json` availability evidence, the same fixed 14-epoch budget selected earlier by
an independent clean control, the same seed and the same model code.

2025 fixed market cohort, 807 games:

| Metric | Baseline | Candidate | Change |
| --- | --- | --- | --- |
| Margin MAE (primary) | 12.387829 | 12.384872 | −0.002957 |
| Margin RMSE | 15.783399 | 15.782297 | −0.001102 |
| Total MAE | 12.708213 | 12.703792 | −0.004422 |
| CRPS | 8.864826 | 8.864109 | −0.000717 |
| Win probability log loss | 0.537504 | 0.537939 | +0.000435 |
| Brier | 0.181486 | 0.181632 | +0.000146 |
| Calibration error (10 bins) | 0.030976 | 0.034473 | +0.003497 |

The week-cluster bootstrap 95% interval on the paired margin change is [−0.0235, +0.0187], so the
improvement is not distinguishable from noise. Accuracy metrics improve; win-probability calibration
degrades slightly. This matches the v3 to v4 adoption, which was also inside noise and also saw log
loss and Brier rise. Acceptance follows the standing rule of a strictly lower primary MAE.

## Published outputs

All **1,609** 2026 predictions and distributions, covering completed and future games, and ranking
weeks 0 through 3, each with 266 Division I entries against a 138-team FBS reference pool. Archived
completed-game forecasts remain reconstructions, not the forecasts originally issued.

Against v4, published numbers move by a mean 3.33 percentage points of win probability (90th
percentile 7.36, maximum 49.11) and a mean 1.88 points of margin (maximum 26.14). The favourite
changes in 84 of 1,609 games. The largest moves are concentrated in FCS and Division II matchups,
where unknown positions were most common: Virginia Lynchburg at Hampton moves from 0.367 to 0.858.
On the 260 completed FBS-involved 2026 games, margin MAE is 12.610 for v4 and 12.538 for v5; both
models were refit after those games, so this is a reference figure, not an untouched test.

## Verification

The staging gate re-verified pinned inputs, all four store manifests, all fifteen model source files
and the checkpoint hash. The production audit passed for all 1,609 games: no missing FBS rosters,
player history strictly before kickoff, no 2026 outcomes in fitting, maximum probability-mass error
2.4e-7. Adapter parity against the experiment's own frozen distributions covered 1,142 games with a
maximum difference of 3.0e-8. Database readback gave a maximum prediction error of 3.5e-6 and PMF
error of 6.1e-9, and all four ranking documents matched exactly. The Python suite passed 241 tests
with 1 skipped; the 2 failures in `tests/test_leakage.py` predate this work and concern three 2004
to 2007 history slots and Florida Atlantic's 2004 transition season.

## Operations and rollback

The rollback archive is `models/player_sequence_production/release_backups/20260922T140741Z/`,
holding the previous active pointer, refresh pointer, local ranking files and database ranking
documents. Restore matching model, prediction/PMF and ranking versions together under the shared
publish lock.

The release runtime symlinks its stores into
`models/experiments/positions_20260922_v1/`, which is therefore production-critical and must not be
deleted while v5 is active.
