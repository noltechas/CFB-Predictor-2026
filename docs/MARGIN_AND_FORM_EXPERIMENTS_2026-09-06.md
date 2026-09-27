# Margin-focused training and recent-form experiments

Both requested experiments are implemented and complete. **Neither established
an improvement sufficient to replace production.** The production checkpoints,
active feature store, and published forecasts were not replaced by these
experimental candidates.

## Results

Three-seed ensembles forecast each of 2023, 2024 and 2025 using only earlier
seasons for fitting and checkpoint selection. On the same **2,396 FBS-vs-FBS
games with archived market lines**, lower margin MAE is better:

| Configuration | Margin MAE | Margin RMSE |
| --- | ---: | ---: |
| Original objective and features, chronological control | **12.702** | **16.029** |
| Selected margin-focused objective | 12.728 | 16.123 |
| Selected margin objective + five-game form/matchups | 12.715 | 16.132 |
| Archived market closing spread | **12.030** | **15.241** |

The standalone recent-form experiment retained the original features during
development selection. Its best alternative improved development MAE by just
**0.005 points**, below the preset 0.03-point advancement threshold. No standalone
form candidate advanced to later-year assessment; its assessment output is the
unchanged control, not evidence that an added-feature model was tested there.

Across all **2,398 FBS-vs-FBS assessment games**, paired changes relative to the
original chronological control were:

| Candidate | Change in margin MAE | Week-cluster bootstrap 95% interval |
| --- | ---: | ---: |
| Margin objective | +0.026 | [−0.040, +0.094] |
| Margin objective + form/matchups | +0.013 | [−0.079, +0.104] |

Positive changes are worse. Both intervals include zero. The combined candidate
improved by 0.012 relative to the margin-objective-only model, but its interval
[−0.084, +0.066] also includes zero. These tested settings did not close the
gap with the market.

Secondary metrics also argue against promotion:

| Configuration | Total MAE | Winner log loss | Brier score | 10-bin calibration error |
| --- | ---: | ---: | ---: | ---: |
| Original control | 13.043 | 0.55450 | 0.18898 | 0.03268 |
| Margin objective | 13.323 | 0.55688 | 0.18988 | 0.02455 |
| Combined | 13.373 | 0.55779 | 0.19005 | 0.02704 |

Calibration error improved, but margin accuracy, total accuracy and the other
probability scores did not. These findings apply to the tested settings;
they do not prove that every margin loss or recent-form design would fail.

## Experiment 1: margin-focused training

The existing network learns a discrete margin distribution using smoothed
cross-entropy and selects checkpoints primarily on winner log loss.
`pretrain()` now supports an optional discrete CRPS component, which accounts
for the distance of a miss while preserving the existing distribution outputs.
Outcomes beyond the tail-bin centres are not clipped in this new loss.

For a distribution with support X and outcome y, the implementation computes
`E|X − y| − E|X − X′| / 2` using its cumulative probabilities. CRPS is a
[proper scoring rule](https://stat.uw.edu/research/tech-reports/strictly-proper-scoring-rules-prediction-and-estimation);
the existing smoothed cross-entropy remains part of the combined training loss.

The following fixed candidates were compared on rolling **2020–2022 development
forecasts**, with three seeds each:

| Candidate | CRPS weight | MAE weight in checkpoint score | Development margin MAE |
| --- | ---: | ---: | ---: |
| Original | 0 | 0 | 13.1991 |
| Margin-aware selection | 0 | 0.05 | 13.1988 |
| CRPS, lighter weight | 0.10 | 0.05 | 13.1773 |
| CRPS, larger weight | 0.30 | 0.05 | **13.1633** |

The larger CRPS weight advanced based on development results. Its later-year
assessment did not reproduce the improvement. Production defaults remain zero
for these optional controls.

## Experiment 2: recent form and matchup comparisons

New staged features describe the last three or five completed games:

- Passing yards per play, rushing yards per play, sacks per passing play,
  explosive-play rate (20+ yards), and passing share, for offense and defense.
- Counts of games with observed data to expose sample size and coverage.
- Offense-versus-opponent-defense differences and pass/run-style interactions.

Each window is evaluated separately. A window adds 24 scalar inputs; its
matchup comparisons add 12 more. The staged file contains all 72 possible new
columns, but an individual candidate uses 24 or 36 of them.

Windows reset by season and include missing-feed games as elapsed games.
They therefore mean the last three/five **scheduled completed games**, rather
than reaching farther back to fill missing data. All games at a shared kickoff
time emit snapshots before their observations are consumed. Unplayed targets
never advance a window. Unobserved passing/rushing rates remain missing.

The staged table has **28,483 rows**, including all **1,402 future targets** in
the frozen schedule. Source timestamps were checked to precede target kickoffs,
and each team's own form remains constant across unplayed future targets.
Matchup comparisons can vary because opponents differ.

| Feature configuration | Development MAE, original objective | Development MAE, selected margin objective |
| --- | ---: | ---: |
| Existing features | 13.1991 | 13.1633 |
| Recent three games | 13.2139 | 13.1260 |
| Recent five games | 13.2559 | 13.1265 |
| Three games + matchups | 13.1989 | 13.1144 |
| Five games + matchups | 13.1939 | **13.0541** |

Five-game form with matchups advanced under the margin-focused objective.
The original-objective control experiment advanced no added-feature candidate.
The latter control was added after experiment 1 failed its later-year check,
to isolate whether form could help without adopting the loss change. Its
settings were still selected entirely on 2020–2022.

All features use existing raw data and counting arithmetic. No market values,
external game predictions, or fitted intermediate ratings enter the inputs.
The current-season raw-play fallback retains the source-compatibility limits
described in the earlier feature audit.

## Selection, limits, and reproducibility

For every forecast season Y, fit before Y−1, select epochs on Y−1, refit all
seasons before Y for that epoch count, then forecast Y. Normalization uses
fitting rows only; future-only identities map to the unknown token.

Configuration selection uses the lowest pooled development FBS-vs-FBS margin
MAE, with an improvement of at least 0.03 points, winner log loss no more than
0.005 worse, and Brier score no more than 0.002 worse than its control. The
choice is written to disk before assessment. Later-year scores do not select
another candidate from the grid. All confidence intervals use 2,000 paired
resamples of season/week clusters.

The 2023–2025 seasons were already examined earlier in this project. They are
later-year historical benchmarks, not a newly untouched final test. The
intervals describe these games and do not incorporate every source of training
uncertainty. **No 2026 outcome was used for fitting or selection.**

The complete experiment is in `models/experiments/margin_form_20260906/`:

- `plan.json`, `form_baseline_plan.json`: candidate settings and selection rules.
- `inputs/`: frozen features, history, vocabulary and benchmark lines; staged
  form features, observation timestamps, and source hashes.
- `code_snapshot/`: the implementation used in this run.
- `runs/`: per-seed probability distributions, selected epochs, and per-game
  predictions for each evaluated candidate and season.
- `objective_report.json`, `form_report.json`, `form_baseline_report.json`,
  and `summary.json`: complete measurements and comparisons.

Commands, run from the project root:

```sh
PYTHONPATH=src python3 src/cfbpred/pregame_experiments.py objective
PYTHONPATH=src python3 src/cfbpred/recent_form.py --input-dir models/experiments/margin_form_20260906/inputs
PYTHONPATH=src python3 src/cfbpred/pregame_experiments.py form
PYTHONPATH=src python3 src/cfbpred/form_baseline_experiment.py
PYTHONPATH=src python3 -m pytest -q
```

Use a new output directory for changed settings, code or inputs. Cached runs
reject mismatched experiment settings rather than silently reusing them.

**66 tests passed**, including CRPS arithmetic/gradients, candidate probability
guardrails, missing-feed windows, same-kickoff ordering, season resets, future
coverage and matchup direction. Python compilation passed. The active feature
store's hash matches the pre-experiment snapshot. The new feature blocks stay
outside the production schema because they did not qualify for promotion.
