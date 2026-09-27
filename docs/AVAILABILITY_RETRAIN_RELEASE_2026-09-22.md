# Availability retrain — player-full-context-tabm16-v7

Published September 22, 2026 at 3:51 p.m. Central (20:51 UTC). The active pregame model is
**player-full-context-tabm16-v7**: the v6 architecture (`full_context_v2`) trained on the full
reviewed availability registry. The live in-game tower (`raw-dnn-delta-v22`) is unchanged.

## What went in

`data/availability_backfill/conference_archives_20260922_registry_v2/reviewed.json`: 12,761
reviewed facts, 12,496 of them availability statuses. The v6 timeline carried reports on 1,351
roster slots; the rebuilt timeline carries them on 11,436. It was built on v6's exact stores and raw
snapshot, so only the reports differ.

| Season | Out | Doubtful | Questionable | Team-games with QB1 out (FBS-involved) |
| --- | ---: | ---: | ---: | ---: |
| 2022 | 70 | 0 | 0 | 0 |
| 2023 | 1,087 | 0 | 281 | 11 |
| 2024 | 1,652 | 19 | 419 | 8 |
| 2025 | 6,258 | 106 | 1,134 | 23 |
| 2026 | 648 | 0 | 62 | 4 |

"QB1 out" means players reported out held more than half the side's recent pass attempts. The
2025 assessment trains through 2024, so it sees 19 such games; the production refit through 2025
sees 42.

## Assessment

Three arms, three seeds each, paired by seed on the fixed 807-game 2025 market cohort, fixed
14-epoch budget. The baseline arm reuses the three v6-architecture runs from
`availability_20260922_prelim` (v6 timeline); both new arms use the new timeline and share code
with each other. The decision rule was written before any result was seen: adopt an arm only if
its three-seed mean margin MAE is strictly below the baseline's, taking the lower if both qualify.

| Arm | Margin MAE | Seed spread | RMSE | WP log loss | Calibration |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline: v6, previous reports | 12.34736 | 0.071 | 15.72827 | 0.53697 | 0.0311 |
| Retrain: v6, all reports | 12.34728 | 0.066 | 15.72500 | 0.53706 | 0.0326 |
| Absence inputs: v6 + per-side availability inputs | 12.34803 | 0.102 | 15.73815 | 0.53744 | 0.0377 |

The retrain is lower by 0.00008 against a seed-to-seed spread of 0.07: a tie, adopted by the
stated rule, not a demonstrated improvement. The one consistent signal is the 22 cohort games in
which a side's QB1 was reported out, where the retrain beat the baseline on all three seeds by
0.057 on average; 22 games cannot establish it.

## The absence-input arm failed, and why

Variant `full_context_v3` feeds the core seven per-side counts: whether a report exists, the out
share of recent pass attempts, carries, receptions and tackles, the doubtful-or-questionable share
of pass attempts, and the log number of players out
(`architecture_research_net.absence_features`). A side with no report reads differently from a
side reported healthy.

With a side's most-used QB reported out, averaged over 384 team-games:

| Checkpoint | Mean effect | Share where it hurt the team |
| --- | ---: | ---: |
| Baseline, seed 0 | −0.11 pts | 74% |
| Retrain, seed 0 | −0.10 pts | 70% |
| Absence inputs, seed 0 | +0.12 pts | 35% |
| Absence inputs, seed 1 | −0.56 pts (max −2.6) | 87% |

Two seeds of the same arm disagree in sign. With 19 training games in which a starting
quarterback was out, the network cannot identify what that absence is worth, so each seed settles
anywhere. That is the data limit measured directly. The variant remains available in
`FULL_CONTEXT_VARIANTS` but is not used.

## A correction to earlier releases

Seed-to-seed margin MAE spread on this cohort is about 0.07, and a same-seed rerun moved MAE by
0.0035. The single-seed gains reported for v5 (−0.003) and v6 (−0.0025) were inside that noise.
Their structural changes are real and measurable (positions assigned, a roster branch that varies
across teams), but their accuracy gains were not demonstrated. Comparisons now use three-seed means.

## Published outputs

All **1,609** 2026 predictions and distributions, and ranking weeks 0 through 3, each with 266
Division I entries against a 138-team FBS reference pool. Against v6, win probabilities move a
mean 0.15 points (90th percentile 0.35, maximum 1.14) and margins 0.09 points; the favourite changes
in 4 of 1,609 games. Inference already used this registry before the retrain, because the nightly
refresh reads the reviewed pointer; v7 changes what the model learned, not what it sees.

## Verification

Staging re-verified pinned inputs, all store manifests, all fifteen model source files and the
checkpoint hash. The audit passed for 1,609 games. Adapter parity covered 1,142 games with a maximum
difference of 2.6e-8. Database readback: prediction error 3.7e-6, PMF error 7.7e-9, all four ranking
documents exact. The v6 checkpoint reloads under the new code with zero difference, confirming the
added paths are inert by default. Python suite: 261 passed; the 2 failures in `tests/test_leakage.py`
predate this work. New tests: `tests/test_absence_features.py`.

## What would change the answer

More starting-quarterback absences in training. The 400 still-uncovered 2023–2025 team-games in
`data/availability_backfill/missing_team_games.json` are the next source. Alternatively, a sign
constraint on the availability inputs, so an absence can never help its own team; the in-game gate
already uses the same kind of monotone construction. That is a modelling choice, not a fix, so it
has not been made.

## Rollback

`models/player_sequence_production/release_backups/20260922T205128Z/`. The release runtime
symlinks into `models/experiments/rosterv2_20260922_v1/` (inputs, players, sequences) and
`models/experiments/availability_v2_20260922/` (timeline); both are production-critical.
