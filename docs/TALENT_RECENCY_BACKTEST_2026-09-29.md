# Talent consistency, recency weighting and the median readout — September 29, 2026

The active pregame model is still **player-full-context-tabm16-v8** (same weights, same
distributions, same win probabilities). What changed in production is the point readout: every
2026 prediction's `pred_margin` and `pred_total` is now the **median** of its predicted
distribution instead of the mean. Two model changes aimed at the gap to the market were
backtested and **not** adopted.

## Why

A gap audit of the 331 FBS-involved 2026 games (weeks 1–4) found about 80% of the
model-minus-market margin gap in cross-tier games (power conference vs G5 vs FCS). The model
under-projected the stronger tier by more every season since 2021: in power-conference vs FCS
games, actual minus model went from −3.6 (2021) to +9.4 (2026). Same-tier games after Week 1
were at market parity. Three candidate causes were identified:

1. **The 247 talent composite breaks in 2024.** CFBD publishes FCS composites through 2023
   (71–86% of FCS sides) and none from 2024. A missing value is imputed at the pooled training
   mean (about 410, a low-G5 roster). The 2024 values also lift the FBS floor: the median team
   rose 38 points in one year (usual +8 to +10), the bottom quartile 107. `sos_talent` silently
   skipped talent-less opponents, so an FCS team's schedule read as FBS-level (≈187 → ≈619).
2. **No recency.** The model trains on 2001–2025 with equal weight and static team embeddings,
   while FBS scoring fell from about 57–58 points (2014–20) to 52.4 (2025–26). Totals ran 1.3–2.9
   points high in every backtest season.
3. **Point estimate.** For absolute error the median is the optimal readout; production used
   the mean.

## What was built

- `features_raw.TalentPercentiles`: the composite's within-season percentile among FBS rosters
  (`tpct_talent`), 0.0 for every non-FBS program in every season the composite exists, unknown
  before 2015; `tpct_known`; and `sos_tpct_talent`, the mean over opponents already faced with
  non-FBS opponents at 0.0. All `tpct` columns are opt-in (`train.EXCLUDED_SCALARS`), so v8 reads
  exactly its old inputs. A `-<column>` entry in a variant's scalar extras drops a column.
  Supabase `public.game_scalars` gained the six nullable columns.
- `full_context_v4` (v2 network on the percentile inputs), and `full_context_v4_r4` /
  `_r8` (plus loss weights 0.5^(age / 4 or 8 seasons), mean 1; `player_sequence_train.recency_weights`).
- `ensemble.median_readout`: the PMF median, interpolated within the crossing integer bin, so
  numbers stay fractional. The production scorer and its audit use it.

## Backtest (pre-registered)

Rule written at 00:51 UTC on September 30, before any fold finished:
`models/experiments/talent_recency_20260929/decision_rule.json`. Every arm reads one store
chain built by the nightly refresh code on September 29, with 3 seeds and 14 fixed epochs per
fold, predicting 2021–2025 out of sample. The cohort is the 4,508 FBS-involved games with a
closing spread; points are read as medians.

| 3-seed ensemble | FBS-involved MAE | Log loss | FBS vs FBS MAE | Total MAE | vs base (95% week-cluster CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| base (v8 architecture, same vintage) | 12.5808 | 0.49487 | 12.5325 | 12.733 | |
| v4 talent percentiles | 12.5835 | 0.49518 | 12.5308 | 12.726 | +0.003 [−0.014, +0.020] |
| v4 + recency, half-life 4 | 12.6236 | 0.49661 | 12.5658 | 12.744 | +0.043 [−0.010, +0.098] |
| v4 + recency, half-life 8 | 12.5900 | 0.49534 | 12.5394 | 12.717 | +0.009 [−0.026, +0.044] |
| closing line | 12.1607 | | | 12.475 | |

**No candidate qualified.** The talent fix is neutral. With consistent inputs, the
power-conference vs FCS bias barely moved (+7.2 → +7.1 points on 2024–25). The input break was
real but small next to the 2024+ shift itself, where the market was also 3–4 points short.
Recency helped 2024 (−0.07 at half-life 4) but cost 2022–23 (+0.15 / +0.07). Every arm gained
about 0.036 MAE from the median readout (base mean readout 12.617).

## What shipped

- The median readout, for all 1,609 2026 games, via `median_restatement.py` → `promote_local`
  (full season) → `publish_future_predictions --all-season` → readback verified (maximum error
  3.8e-6, PMFs byte-identical, database rows archived first). No row was re-scored, so each
  completed game keeps the forecast it had. 2026 FBS-involved margin MAE is 12.532 → 12.434
  (market 11.607), total 11.822 → 11.766. The nightly refresh applies the same readout to
  upcoming games.
- The percentile columns flow through the nightly feature build and publish (verified), unused
  by v8, ready for a future variant.
- The live scorer (Supabase edge function `live-scorer`, and the disabled Mac fallback
  `live_scorer.py`) writes the same median readout for `live_predictions` and `live_curve`, so the
  first in-game number starts from the pregame card instead of jumping by the mean-median gap. The
  state tower's pregame conditioning stays the mean it was trained on. Verified before promotion on
  a canary: replaying Washington-Minnesota (167 snaps), win probability and total PMFs are identical
  to the old function; margins move 0.62 points on average. TS `pmfMedian` equals Python
  `median_readout` exactly on 1,200 distributions; the TS-vs-PyTorch parity gate still passes
  (worst |dwp| 2.2e-7). Curves stored before September 30 keep mean-based margins.

## Leakage check for re-scoring completed games

v8 forecast 123 week-4 games prospectively. Re-scored from the September 29 stores, margins
moved an average of 0.52 points. The move's correlation with the outcome surprise was −0.03, and
52% of moves went toward the result: data revisions, not look-ahead
(`leak_check_v8.json`). The v8 regression check under the new code was identical on 79% of
upcoming games; all changes came from newly captured availability reports.

## Artifacts

`models/experiments/talent_recency_20260929/`: `decision_rule.json`, `decision.json`, `runs/`
(60 fits), `stores/` (the shared store chain, 1.2 GB, needed only to reproduce), `run_queue.py`,
`evaluate.py`, `leak_check.py`, `median_restatement.py`, `release.py` (unused staging path for a
future qualifying variant). Local rollback: `models/player_sequence_production/promotions/20260930T021852398647Z/`.
Database rollback: the newest `models/player_sequence_production/database_backups/` entry.
