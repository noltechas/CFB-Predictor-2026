# Feature coverage, chronological backtest, and public availability data

The future-game feature pipeline is repaired. The chronological experiment
slightly reduced average margin error, but the uncertainty interval includes
no improvement. No verified public injury dataset covers this project's full
historical population. Public reports have been archived for further work;
they have not been silently treated as complete, timestamped training inputs.

## Future predictions

Every one of the **1,402 future 2026 games** in the model's D1-involved schedule
has a feature row using the complete **270-column active scalar schema**, plus
the existing history and identity inputs. This is the trained feature set;
features previously removed by model experiments remain excluded.

The former week-based joins lost box-score, player and play-by-play snapshots
for later scheduled weeks. They also mishandled two games sharing a week label.
The replacement constructs snapshots by game ID and kickoff order, emits each
target before consuming its result, and advances accumulators only for games
with completed observations. Athlete career counts advance globally in time
to prevent a transfer's later career at one team appearing in earlier games at
another team. Unreported box fields remain missing instead of becoming zero.
QB leadership ties now have a deterministic name-based tiebreaker; database
row order cannot change the “QB changed” feature between identical rebuilds.

The current cfbfastR release lacks 2026. Raw CFBD plays provide a separate
fallback for pregame counting features, including named player attribution.
The completed import contains **34,711 plays across 198 finished games**, with
named passer observations in all 198. The final audit verifies prior box
features for 1,310 home and 1,289 away target rows; PBP features for 1,310 home
and 1,293 away rows; and QB features for 1,298 home and 1,278 away rows. Each
count matches the applicable available prior source observations. Differences
reflect upstream coverage and teams without qualifying observations.
The attribution endpoint has a 2,000-record response cap: requests are now
scoped to individual games and a cap-sized response fails the refresh.
See the [official CFBD endpoint documentation](https://api.collegefootballdata.com/api/plays).

The adapter uses factual play types and player-stat associations; it does not
consume PPA/EPA, ratings or betting lines. Its flags are not guaranteed to be
identical to cfbfastR's derived flags. The repaired release takes precedence
for overlapping games once available. The fallback never enters the live
play-level training corpus.

`data/parquet/features/future_coverage.json` records coverage for every active
column, checks all known prior passing-box/PBP/QB joins, and rejects changes
in accumulated statistics between unplayed future targets. Missing source
data remains possible: teams without completed games have no current-season
stats; some schools lack individual stat fields, coaching or roster records.
Participation comparisons also need earlier observations to identify leaders.
“Full feature set” does not mean manufacturing values for unavailable facts.

Inference now applies the scaler saved with the model checkpoint and checks
that every ensemble member expects the same columns and preprocessing.
Predictions and probability distributions are refreshed with the existing
16-seed production weights. The weekly workflow publishes features and future
forecasts without dropping application views or rewriting started-game calls.
Current-season prediction snapshots are archived locally with creation times.
These locks apply from this repair onward; older regenerated predictions are
not evidence of what was published before kickoff.

## Chronological experiment

The existing process holds out 2026 by season, but selects checkpoints using
the scattered `game_id % 7 == 3` validation games in earlier seasons. It was
not using a chronological validation season. A held-out historical game's
result can appear in the history of a later training game, so the scattered
validation score is not an independent future-season estimate.

`src/cfbpred/backtest_pregame.py` compares these processes on the same unseen
2023, 2024 and 2025 seasons, with three seeds per method and test season:

- Existing selection: fit on earlier non-validation games and select epochs
  using the earlier hashed-game validation subset.
- Chronological selection: fit before the immediately preceding season,
  select epochs on that season, then refit on all earlier seasons for the
  selected epoch count.

Both methods fit normalization on fitting rows only, map unseen identities to
the unknown token and use only pregame predictions. Model weights do not learn
from the test season. Features can use games completed earlier in that test
season, as an in-season forecaster would. This comparison includes the benefit
of refitting all earlier data; it does not isolate only the choice of validation
split. It is a three-seed process experiment, not a replacement 16-seed live model.

Final results on the **2,396 FBS-vs-FBS games with market lines**:

| Predictor | Margin MAE (points) | Margin RMSE (points) |
| --- | ---: | ---: |
| Existing hashed selection | 12.742 | 16.077 |
| Chronological selection and refit | 12.702 | 16.029 |
| Archived market closing spread | 12.030 | 15.241 |

Across all 2,398 FBS-vs-FBS games, chronological MAE is 12.694 versus 12.735.
The paired change is **−0.041 points**, with a 2,000-resample week-cluster
bootstrap 95% interval of **[−0.151, +0.070]**. Negative favors chronological.
That interval does not establish a reliable improvement. The market still
has lower error. Chronological evaluation is useful for honest measurement,
but this result does not justify replacing production weights on accuracy
grounds. The bootstrap describes these three seasons and does not capture
every source of training or future-season uncertainty.

| Unseen season | Existing MAE, all FBS-vs-FBS | Chronological MAE |
| --- | ---: | ---: |
| 2023 | 12.844 | 12.687 |
| 2024 | 12.774 | 12.814 |
| 2025 | 12.591 | 12.584 |

Results and per-game predictions are in
`models/experiments/chronological_20260906_deterministic/`. The exact scalar input used
for that experiment is retained there as `input_game_scalars.parquet`, with
SHA-256 recorded in `report.json`. This run uses the completed 2026 raw-player
attribution import and deterministic QB leadership tiebreaking.

The general evaluator now calls the actual pregame tower; it previously used
early live-game states for its pregame comparison. Tests no longer regard
beating the market as inherently invalid; they check stage provenance and
finite results. Historical validation output is still labeled validation.
Closing lines are benchmarks, not model inputs. This experiment does not
reconstruct every historical forecast's publication time or measure betting ROI.

## Public injury and availability reports

Following the public-only instruction, no paid feed was purchased or used.
Recent official public archives exist, but coverage is fragmented:

- Big Ten [2023](https://bigten.org/fb/article/blt19caa3aea8cf4525/) and
  [2024](https://bigten.org/fb/article/blt376070b4270ab9d7/) report archives.
- The SEC [introduced public availability reports in 2024](https://www.secsports.com/news/2024/08/sec-to-provide-public-availability-reports-for-football-basketball-and-baseball),
  with football reporting for conference games.
- The MAC has public documents such as its
  [August 29, 2024 report](https://getsomemaction.com/documents/2024/8/29/24_FB_AVAILABILITY_REPORT_8_29.pdf).

**34 Big Ten PDFs covering 2023–2024, containing 443 pages, are archived.**
The manifest includes official source URLs and content hashes:
`data/raw/availability/bigten/manifest.json`. Extracted page text and metadata
are saved in `data/parquet/availability/public_report_pages.parquet` and JSON.
These are source pages, not a finished athlete-to-game feature table.

The archive does not establish original publication timestamps for each
status. For example, the 2023 Week 1 compiled PDF has September 3 creation and
modification metadata while containing a September 2 Illinois–Toledo page.
This does not prove the status was wrong or unavailable pregame; it means the
compiled file timestamp alone cannot establish the pregame information cutoff.
Each staged page therefore has `published_at=null` and `training_eligible=false`.

No verified source was found covering all teams and seasons in this project's
history. No injury features have been added to model training yet. A usable
partial integration needs original publication evidence, player/game identity
matching, coverage indicators and a chronological ablation. Games without
reports must remain “unknown,” not “healthy.” The archived public sources
provide material for that work without contaminating today's training inputs.

## Reproduce and verify

Run the usual refresh with `./refresh_week.sh`. The feature audit runs before
publication. Run `python3 src/cfbpred/backtest_pregame.py --help` for experiment
options; use a new output directory when inputs change.

Validation includes the full test suite, Python compilation, shell syntax,
the future-feature coverage audit, and production readback. Final logs and
readback evidence are retained under `logs/`; the original source and data
backups are in `data/audit_backups/20260906_feature_fix/`.

The final test run passed **59 tests**. Python compilation and the weekly
refresh script's shell syntax check also passed.

Production readback verified **1,402 feature rows, forecasts and probability
distributions** against the local outputs. The cached feature store matches
the published store; all **208 previously started 2026 forecasts** were
preserved. The dashboard comparison view remains available. Evidence is in
`logs/feature_refresh_readback.json`; the verification script is retained as
`logs/verify_feature_refresh.py`.
