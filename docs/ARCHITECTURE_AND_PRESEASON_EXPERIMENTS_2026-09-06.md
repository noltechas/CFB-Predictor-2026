# Architecture and preseason-input experiments

Six additional candidates were implemented and measured. **None established a
margin-accuracy improvement sufficient to replace production.** The live
checkpoints, published predictions and active 270-column feature contract
remain unchanged. The active feature-store SHA-256 still matches the frozen
control: `b6764b0a5e8454447896371a747bbb882219695d283cd9bd46996cbec3ca5616`.

The clearest finding is a preseason information gap. In the chronological
control's 2023–2025 market-matched FBS games, the average error gap with the
market was **1.444 points when either team had no current-season games**, versus
**0.511 after both teams had at least eight games**. This is an exploratory
association, not proof that missing quarterback information causes the gap.

## What was tested

Each candidate retained the original raw inputs and no-market/no-external-model
policy. Configuration selection used three-seed rolling forecasts for
**2020–2022**, covering 1,974 FBS-vs-FBS games. Lower MAE is better.

| Candidate | Development margin MAE | Winner log loss | Decision |
| --- | ---: | ---: | --- |
| Original chronological control | 13.1991 | 0.55570 | Reference |
| Team-order-consistent network | **13.0075** | **0.54643** | Advanced to later-year assessment |
| Original network, last eight training seasons | 13.3644 | 0.55675 | Rejected on development |
| Consistent network, last eight seasons | 13.2406 | 0.55323 | Rejected on development |
| Shared team encoder + consistent matchup decoder | 13.4172 | 0.56346 | Rejected on development |
| Three-component discrete-mixture distribution heads | 13.2498 | 0.55766 | Rejected on development |
| Original network + preseason roster passing priors | 13.1941 | 0.55261 | Gain only 0.0050; rejected on development |

Advancement required at least 0.03 points of development MAE improvement,
winner log loss no more than 0.005 worse and Brier score no more than 0.002 worse.
The shared encoder, mixture heads and preseason priors were sequential follow-up
experiments after earlier results; each had its own saved plan and development
choice. The later-year error audit motivated the preseason-prior experiment,
so it must not be described as a hypothesis developed without seeing those years.

Only the team-order-consistent candidate advanced. On the same **2,396**
2023–2025 FBS-vs-FBS games with archived market spreads:

| Forecast | Margin MAE | Margin RMSE |
| --- | ---: | ---: |
| Original chronological control | 12.7017 | 16.0286 |
| Team-order-consistent candidate | 12.6832 | 16.0151 |
| Archived market closing spread | 12.0296 | 15.2410 |

Across all 2,398 FBS-vs-FBS assessment games, the paired MAE change was
**−0.0186 points**, with a season/week-cluster bootstrap 95% interval of
**[−0.1260, +0.0850]**. Winner log loss improved from 0.55450 to 0.55271 and
total MAE from 13.0426 to 12.9920. These small changes did not meet the saved
promotion criteria. No candidate triggered the planned 16-seed confirmation.
These results compare three-seed experimental ensembles; they are not a new
measurement of the deployed 16-seed ensemble.

## Structural changes

The earlier rankings diagnostic found large input-order inconsistency on
hypothetical neutral matchups. `consistent_net.py` evaluates both orientations
with shared weights, reverses the second margin-logit vector, and averages
corresponding logits. At inference, switching the team labels reverses the
margin PMF exactly and preserves the total PMF.

Home-field advantage follows the physical venue: the venue input is +1 when
the first team is at home, 0 at a neutral site and −1 when the second team is
at home. Per-side scalar normalization uses pooled **training-only** means and
standard deviations. This preserves swap semantics even for missing values.
Team, conference, history, opponent and games-played inputs all switch sides.
New scalar fields without declared reversal semantics raise an error.

`shared_team_net.py` also shares the scalar encoder across both sides. It
learns each team's representation with a 128→64 network before processing the
matchup jointly. Its empty-history branch uses zero evidence instead of the
original max-pooling sentinel of −10000. The matchup remains an end-to-end
DNN, not a separately fitted rating. This architecture failed development.

`mixture_head.py` predicts three logistic-component locations, scales and
weights, integrates their mass over the existing integer bins and open tails,
then applies shared learned bin biases. The biases can learn common final
margins directly from outcomes. Stable log-CDF differences preserve gradients
for distant misses. The output remains the same discrete PMF interface.

## Preseason data and remaining information gap

The current primary-QB career fields are populated only after that team has
observed current-season passing production. The new `qb_roster_prior.py`
experiment adds nine fields per side using the current-season roster and only
the **preceding four seasons** of public passing boxes:

- Roster-known and passer-known flags, plus the number of observed passers.
- Prior attempts and passing-game count of the most experienced rostered passer.
- That player's yards per attempt, completion percentage, TD rate and INT rate.

Athlete IDs carry experience across transfers. The player is selected solely
by prior attempts, with deterministic identity-based tie breaking. This is a
roster-based proxy, **not a confirmed starter**, and is never selected using
the season's eventual passing leader. Missing rosters and unobserved passing
experience remain distinguishable. Incomplete boxes are excluded from rate
denominators; identical duplicate source observations are counted once and
conflicting duplicates are rejected.

The staged features cover all 28,483 model targets. At the original feature
audit cutoff, **all 1,402 future targets** had a row, with a known passing prior
for 1,393 home sides and 1,375 away sides. Each team's preseason inputs remain
constant across its unplayed games. Source-season provenance is strictly
earlier than the target season. No 2026 passing outcome entered these priors.

Historical roster files are retrospective provider snapshots, not guaranteed
archives of the exact roster published before kickoff. This is a limitation
shared with the existing roster inputs. A successful retrospective experiment
would still need genuinely archived prospective validation.

The original-control error audit is retained in `error_diagnosis.json`:

| Minimum current-season games played by either team | Games | Model MAE | Market MAE | Gap |
| --- | ---: | ---: | ---: | ---: |
| 0 | 137 | 13.725 | 12.281 | 1.444 |
| 1–3 | 529 | 12.659 | 11.767 | 0.892 |
| 4–7 | 794 | 12.567 | 11.984 | 0.582 |
| 8+ | 936 | 12.691 | 12.180 | 0.511 |

The next data priority is **dated confirmed-starter announcements and player
availability**, especially before a team has current-season observations. This
is a hypothesis to test, not an established source of a particular accuracy
gain. Public examples exist: Ohio State's official
[August 16, 2024 starter announcement](https://ohiostatebuckeyes.com/news/2024/8/16/football-will-howard-named-ohio-states-starting-quarterback)
predated its opener, and the SEC publishes
[official football availability reports](https://www.secsports.com/fbreports).
These examples do not establish complete historical coverage. Any collection
must preserve publication time, effective matchup, player identity and unknown
status; postgame lineup summaries cannot substitute for pregame announcements.

## Validation and artifacts

**80 tests passed**, and Python compilation passed. New checks cover reversal
at home and neutral venues, gradients, training-only shared normalization,
unknown identities, missing history, complete scalar assignment, chronological
window boundaries, integrated mixture probabilities and extreme-tail
gradients, preseason transfer joins, current-season exclusion, missing rosters,
incomplete boxes and duplicate observations. Future-target coverage and
season-source provenance were also audited against the staged files.

For forecast season Y: fit before Y−1, select epochs on Y−1, refit before Y
for that epoch count, then forecast Y. Each normalization fit uses only its
training rows. All 2023–2025 comparisons are reused historical benchmarks,
not newly untouched tests. **No 2026 outcome was used to fit or select models.**
The bootstrap describes these games and does not incorporate every source of
training uncertainty or repeated research selection.

Artifacts under `models/experiments/`:

- `architecture_20260906/`: three initial candidates, frozen choice, later-year
  results, error slices, and consolidated `round_summary.json`.
- `shared_team_20260906/`: shared-encoder development results.
- `mixture_20260906/`: discrete-mixture development results.
- `qb_prior_20260906/`: frozen roster/raw-stat hashes, annual passing data,
  source provenance, staged priors, and development results.

Each directory contains the exact code snapshot, plan, cached per-seed PMFs,
refitted challenger checkpoints and per-game predictions used in that run.
Controls reuse the unchanged frozen margin/form experiment. The generic runner
later gained an optional model factory; the original shared-encoder snapshot
preserves its exact earlier implementation. Changed code requires a new output
directory; frozen plans reject mismatches.

Entrypoints are `architecture_experiment.py`, `shared_team_experiment.py`,
`mixture_experiment.py`, `qb_prior_experiment.py` and
`diagnose_pregame_gap.py`, run with `PYTHONPATH=src`. The new modules remain
experimental. No database writes or forecast publication were performed.
