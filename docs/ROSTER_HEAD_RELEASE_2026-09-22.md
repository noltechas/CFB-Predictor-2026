# Roster-head release — player-full-context-tabm16-v6

Published September 22, 2026 at 11:14 a.m. Central (16:14 UTC). The active pregame model is
**player-full-context-tabm16-v6**, variant `full_context_v2`. The live in-game tower
(`raw-dnn-delta-v22`) is unchanged.

## Why

An input-by-input audit of v5 found the roster branch nearly inert. Its 24-number summary varied
by about 7% of its own magnitude across teams, deleting both rosters moved a game by 0.7 points
on average against 3.1 for team identity, and swapping Ohio State's and Illinois's rosters left
the prediction unchanged. Measured at initialisation on 2023 games, the outcome loss delivers
about 18 units of gradient to the core body and 0.13 to the entire roster encoder; the branch
never had a reason to learn. Separately, a player reported out was removed from only one of the
three pooling statistics per unit, so an "out" flag reached a third of the unit summary.

## The change

Three things, one paired retrain:

1. **Roster-only margin head.** A swap-symmetric auxiliary head predicts the margin distribution
   from the two roster summaries alone, weighted at half the game loss
   (`ResearchPlayerTower.attach_roster_head`, `ROSTER_HEAD_WEIGHT = 0.5`). It delivers roughly
   45 times more gradient to the encoder than the game loss does.
2. **Availability-aware pooling.** Unit mean, maximum and count pool over available players only,
   so a report reaches the whole summary (`FullRosterEncoder.pool_available`).
3. **Recruit-link repair in the stores.** CFBD's recruit records sometimes carry another player's
   athlete id. When the roster's own recruit pointer and the player's name agree, and the
   competing id belongs to no roster row of that name, the record is now accepted
   (`roster_recruit_id_name_verified`); every rejection now carries a reason. 98 roster rows gained
   a recruiting record, among them Dante Moore's five-star profile. The baseline was trained on the
   repaired stores too, so the comparison isolates the architecture.

The deployed v5 variant is untouched: `roster_head_weight` defaults to 0 and `pool_available` to
False, and the v5 checkpoint reloads identically.

## Assessment

Baseline and candidate share the September 22 raw vintage, the repaired stores, the reviewed
availability evidence, the fixed 14-epoch budget selected by the independent clean control, the
seed and the code. Only the variant differs.

| Metric, 2025 fixed market cohort, 807 games | Baseline | v6 | Change |
| --- | --- | --- | --- |
| Margin MAE (primary) | 12.38190 | 12.37946 | −0.00245 |
| Margin RMSE | 15.77993 | 15.77248 | −0.00745 |
| Total MAE | 12.70288 | 12.69799 | −0.00489 |
| Win probability log loss | 0.53794 | 0.53659 | −0.00136 |
| Brier | 0.18164 | 0.18117 | −0.00047 |
| CRPS | 8.86268 | 8.86186 | −0.00082 |
| Calibration error (10 bins) | 0.03667 | 0.02726 | −0.00941 |

Every metric improved. The week-cluster bootstrap interval on the paired margin change is
[−0.0385, +0.0337], so the margin gain is inside noise; this is the second consecutive candidate
in this line where the win-probability metrics moved the right way rather than the wrong way.
`complete_history_comparison.compare` refuses runs whose variant names differ, so the record in
`models/experiments/rosterv2_20260922_v1/comparison/comparison.json` was computed with its own
metric and bootstrap functions.

| Mechanism, 256 FBS games | v5 | v6 |
| --- | --- | --- |
| Roster summary variation across teams | 6.9% | 36.1% |
| Delete both rosters, mean move | 0.81 pts | 0.87 pts |
| QB1 reported out, mean move (384 team-games) | 0.015 pts | 0.124 pts |
| Jayden Maiava out, for USC | −0.04 pts | −0.40 pts |
| Katin Houser out, for Illinois | −0.02 pts | −0.21 pts |
| Dante Moore out, for Oregon | −0.01 pts | −0.05 pts |

"Reported out" and "removed from the roster" now produce identical numbers, as the pooling change
intended. The cost of losing a starter scales with the backup: Oregon loses little because Dylan
Raiola carries 23 career games behind Moore.

## What this does not yet fix

Absence effects remain roughly a tenth of what a real starter is worth, and the reason is data.
The training era holds 58 team-games with any player marked out, 0.11% of 51,769, all from 2022
to 2024. The network cannot learn the size of an absence from that. Historical availability
backfill is the lever; current-season reports supply the flag but not the training signal. One
probe also moved the wrong way: swapping Ohio State's and Illinois's rosters nudges Ohio State up
slightly, consistent with an encoder that reads career production more than pedigree. 18.4% of
FBS roster slots still carry no scouting, history or current box line.

## Published outputs

All **1,609** 2026 predictions and distributions, and ranking weeks 0 through 3, each with 266
Division I entries against a 138-team FBS reference pool. Against v5, published numbers move a
mean 1.70 points of win probability (90th percentile 3.78, maximum 21.3) and 0.93 points of
margin; the favourite changes in 35 of 1,609 games. The largest moves are FCS and Division II
matchups. On the 260 completed FBS-involved games, margin MAE is 12.538 for v5 and 12.574 for v6;
both were refit after those games, so this is a reference figure, not a test.

## Verification

Staging re-verified pinned inputs, all four store manifests, all fifteen model source files and
the checkpoint hash. The production audit passed for 1,609 games with no missing FBS rosters, player
history strictly before kickoff and no 2026 outcomes in fitting. Adapter parity covered 1,142 games
with a maximum difference of 3.7e-8. Database readback gave a maximum prediction error of 3.7e-6
and PMF error of 6.8e-9; all four ranking documents matched exactly. The Python suite passed 248
tests with 1 skipped; the 2 failures in `tests/test_leakage.py` predate this work.

## Operations and rollback

Rollback archive: `models/player_sequence_production/release_backups/20260922T161414Z/`.
The release runtime symlinks its stores into `models/experiments/rosterv2_20260922_v1/`, which is
now production-critical. The daily refresh reads the position source from the active checkpoint
and needs no change. Production accepts variants `full_context`, `full_context_rosterhead` and
`full_context_v2` (`player_sequence_production.FULL_CONTEXT_VARIANTS`).
