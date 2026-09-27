# Five-part upgrade — player-full-context-tabm16-v8

Published September 22, 2026 at 5:37 p.m. Central (22:37 UTC). The active pregame model is
**player-full-context-tabm16-v8**: the v7 architecture (`full_context_v2`) shipped as **five outer
seeds**, each with 16 internal TabM members, with probability mass averaged across seeds. The
in-game scorer now starts from this model's published pregame distribution. Five changes shipped
together; each was tested before it was adopted.

## 1. Rolling backtest replaces the single-season test

`src/cfbpred/backtest_eval.py` scores every candidate on the FBS-vs-FBS games with a closing
spread in 2021 through 2025, each season predicted by a model trained only on earlier seasons:
3,942 games against the previous 807. Arms are scored as three-seed ensembles, the way production
ships. Closing totals (median across providers in `data/parquet/cfbd/lines.parquet`) now benchmark
the totals prediction; like spreads, they are scored against and never used as inputs. The
adoption rule was written to `models/experiments/backtest_20260922/decision_rule.json` at
21:24 UTC, before any fold finished.

| 3-seed ensemble, 2021–2025 | Margin MAE | Win-prob log loss | Total MAE |
| --- | ---: | ---: | ---: |
| Baseline (v7 architecture) | 12.5658 | 0.54206 | 12.8831 |
| Per-family early-season gates | 12.5664 | 0.54206 | 12.8834 |
| Re-admitted venue, schedule, head-to-head | 12.5651 | 0.54241 | 12.8753 |
| v7 as it ran, one seed | 12.5931 | 0.54299 | 12.8867 |
| Closing line | 12.1336 | | 12.5396 |

All 49 fits use one fixed 14-epoch budget, chosen once on 2024 validation. Every arm shares it,
so it cannot favour one, but absolute errors in earlier folds are marginally optimistic.

## 2. Seed ensemble

Averaging seeds cannot do worse than the average single seed on log loss or on the error of the
expected margin. Measured on 2024–2025, averaged over every combination of five seeds:

| Seeds | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Margin MAE | 12.5857 | 12.5723 | 12.5680 | 12.5655 | 12.5641 |

Five seeds were fixed in advance; a sixth would add about 0.001. Against v7 as it ran, the
five-season ensemble is 0.027 better on margin MAE and 0.0009 on log loss.
`player_context_production.stage_ensemble` stages the seeds as one bundle, each through the same
lineage checks as a single seed. Scoring, rankings, explanations and the refresh already iterate
bundle members.

## 3. Early season and bowls

The gap to the closing line, 3-seed baseline ensemble:

| 2021–2025 | Games | Gap, pts per game | Share of gap |
| --- | ---: | ---: | ---: |
| Weeks 1–3 | 734 | +0.567 | 24% |
| Weeks 4–6 | 818 | +0.564 | 27% |
| Weeks 7–9 | 805 | +0.090 | 4% |
| Weeks 10–13 | 1,183 | +0.311 | 22% |
| Postseason | 402 | +0.961 | 23% |

The single-season pattern holds across five seasons. Two candidates were tested and neither met
the rule:

- **One season-to-date gate per stat family** (`full_context_gates`): +0.0006 MAE, interval
  [−0.0008, +0.0019]. The learned gates stayed within a few points of their starting value
  (trust after three games 0.73–0.78, against the single gate's 0.74), so this rules out a large
  effect from differential trust without proving the idea useless.
- **Re-admitted head-to-head, kickoff hour, surface, capacity and body-clock columns**
  (`full_context_extras`): −0.0007 MAE with log loss +0.00035 and per-season changes from −0.054 to
  +0.033. Noise.

With stat trust ruled out as the main lever, the early-season and bowl gaps most plausibly come
from information the market has and the model does not: announced starters, injuries and
preseason expectations early, opt-outs in bowls. Coordinator changes have no historical source;
public college depth charts do not exist (ESPN's endpoint returns no chart; CFBD has none). The
archive in section 5 now collects what can be collected going forward.

## 4. The in-game scorer starts from the published pregame number

The edge function runs only the in-game half of the network and reads its pregame half from
`public.game_pregame_logits`. That table was filled by the retired raw-DNN pregame tower, so live
predictions began from a different model: 7.7 points of win probability apart on average, up to
74, disagreeing on the favourite in 30 of 251 games. `src/cfbpred/publish_live_priors.py` now
writes the published pregame PMF instead, one copy per in-game member, and the nightly refresh
runs it in place of the old publisher. The edge function needed no change.

Validated before switching, on 118 held-out 2025 games (20,859 plays) that the in-game model never
trained on:

| Phase | Log loss, old prior | New prior | Change |
| --- | ---: | ---: | ---: |
| All plays | 0.33519 | 0.33142 | −0.0038 |
| 1st quarter | 0.44896 | 0.44627 | −0.0027 |
| 2nd quarter | 0.36985 | 0.36128 | −0.0086 |
| 3rd quarter | 0.30964 | 0.30379 | −0.0059 |
| 4th quarter | 0.21766 | 0.21999 | +0.0023 |

Margin error improved in every quarter. For all 125 upcoming games, the in-game model's pregame
input now equals the published distribution (maximum difference 0.0001 points). The first live
number still blends that input with the in-game half, so it starts close to the pregame card
rather than identical. Games that had kicked off before the switch keep the prior they started with.

## 5. Point-in-time source archive

`src/cfbpred/snapshot_sources.py` stores what each self-overwriting source says, stamped with
when it was seen: CFBD rosters, schedules, weather forecasts and lines, ESPN lines, the model's
merged roster, and every availability document the collectors fetch. Payloads go to
`public.source_snapshots`, raw documents to the private Storage bucket `source-archive`, and an
unchanged source adds only a row to `public.source_observations`. Neither table has a public read
policy. It runs as the last step of the nightly refresh and after each successful pre-kickoff
refresh, never on validation runs, and a failure never changes either script's exit code. First
capture: seven API sources and 116 documents, about 5.6 MB compressed.

## Published outputs

All **1,609** 2026 predictions and distributions, and ranking weeks 0 through 3 scored with all
five seeds, 266 entries against a 138-team reference pool. Against v7, win probabilities move a
mean 1.49 points (90th percentile 3.44, maximum 10.5) and the favourite changes in 30 games; most
of that is one seed's idiosyncrasies averaging out. On the 260 completed FBS-involved 2026 games,
margin MAE is 12.571 for v7 and 12.603 for v8. That is a small sample against a five-season test
of 3,942 games that favours v8 on every metric, and one seed can beat an ensemble on a small
sample by chance.

## Verification

Each seed passed staging's lineage checks. The audit passed for all 1,609 games. Adapter parity
against the average of the five seeds' own saved distributions: maximum difference 1.1e-8 over
1,142 games. Database readback: prediction error 3.4e-6, PMF error 7.8e-9, four ranking documents
exact with `n_members` 5. The v7 checkpoint reloads under the new code to within float rounding.
Python suite: 267 passed; the 2 failures in `tests/test_leakage.py` predate this work.

The nightly pipeline was run end to end against v8 into an isolated directory without promotion:
stores rebuilt with the active checkpoint's position source, dated context refreshed, all 1,142
future games scored by the five seeds, no missing FBS rosters, player history strictly before
kickoff. Its report is `models/experiments/backtest_20260922/validation_refresh/20260922T223804008987Z/`.

## Operations and rollback

Rollback archive: `models/player_sequence_production/release_backups/20260922T223708Z/`. Restoring
the live prior means running `publish_pregame_logits.py` and restoring refresh step 8. The release
runtime symlinks into `models/experiments/rosterv2_20260922_v1/` and
`models/experiments/availability_v2_20260922/`; both are production-critical. Explanations now
average all five seeds, which makes the nightly explanation step about five times slower.
