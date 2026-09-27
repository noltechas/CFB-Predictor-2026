# Full player histories and dated game context

The accepted model described here is retained. A subsequent public-only collection pass expanded availability from 220 statuses/32 games to 420 statuses/49 games and retrained both historical and current candidates. See [the historical availability backfill report](HISTORICAL_AVAILABILITY_BACKFILL_2026-09-21.md) for exact coverage, results and the remaining FBS-wide work queue. Counts below describe the original accepted experiment.

The candidate can read every collected prior player game, across programs, without an eight-game limit. This does **not** mean the project now possesses every player's complete lifetime record. Public sources do not supply complete historical injury reports, depth charts, offensive-line snaps, or high-school/JUCO game histories. Unknown information remains distinguishable from an observed zero or an available player.

The experiment and source snapshots are in `models/experiments/player_context_20260921_v1`. Production activation has not been changed. Final comparison metrics are saved in `comparison.json`; the 2026 candidate, forecasts, and player-by-game role export are under `runs/full_context_checked_2026_s0`.

The clean 2025 comparison reduced FBS-vs-FBS margin MAE from **12.3531 to 12.3195 points**, a 0.0336-point change whose week-cluster bootstrap interval includes no improvement (−0.1053 to +0.0389). On the same 807 games with market lines, the candidate scored **12.3292** versus **11.8340** for the market. Winner log loss was slightly worse than the control (0.5370 versus 0.5366). The user accepted this reduction on September 21 and asked to retain these changes as the foundation for expanded historical availability. The accepted model and source hashes are pinned in `RETAINED_FOUNDATION.json`. Production activation remains unchanged; the confidence interval describes uncertainty in the estimate and does not undo the decision to keep the changes.

The revision-checked 2026 candidate has been trained, reloaded, and used to score **all 1,142 remaining FBS/FCS-involved scheduled games** in the pinned schedule after 2026-09-21 23:38:54 UTC. Its role export contains **254,254 player-game records**. These are saved research forecasts, not published replacements for the live model.

The full-history extension retains 978,030 resolved player-game records from 2004–2026, covering 86,018 canonical people. The longest observed career is 66 games, and 35,185 people have more than eight collected games. It adds 27 raw fields omitted from the previous event encoder and explicitly accounts for all 50 category/stat-type combinations in the collected feeds. Each value has an observation indicator; conflicting duplicate values remain missing. New, unrecognized stat types fail validation rather than being silently discarded. External prediction metrics and market odds are not model inputs.

The encoder attends over every prior game and can learn how much older performances matter. It also retains existing recruiting/scouting channels and career aggregates. Program, opponent, elapsed time, season, and whether the game was at the current program accompany each observation. There is no player-ID embedding and no hard roster-size or career-length cutoff.

The dated context store represents all 28,482 target games and 56,964 team-game rosters, totaling 4,249,919 player-game slots. It applies a fact only after its knowledge time, during its effective interval, and for its specified season/game. Sources, retrieval times, publication evidence, and hashes accompany the normalized facts.

| Input | What is now available | Important limit |
|---|---|---|
| Career box scores | Every collected, resolved prior game and all supported raw fields | Missing games and unreported snaps are not reconstructed |
| Positions | Dated public announcements and the newly observed current roster | Today's position is not copied into old games |
| Transfers | 18,894 archived portal rows for 2021–2026; 10,623 have matching canonical identities at both roster endpoints | Entry into the portal does not date a destination announcement; these counts do not imply 10,623 historically eligible facts |
| Availability | 220 player-game status observations across 32 games, with verified pregame article timestamps | Almost all historical reports concern Ohio State; there is one Florida game |
| Depth charts | Five explicit game-scoped depth-rank facts plus learned workload/contribution features for every roster | Learned contribution weights are not calibrated snaps or probability of playing |
| Current conference reports | SEC and Big Ten public report/archive collection | The reports observed on September 21 were pending; old archive rows without publication evidence are retained outside historical training |
| Coaches | Existing head-coach tenure and prior-career features remain available | A comprehensive dated coordinator/staff/scheme history has not been added |

Current FBS unknown positions fell from 5,914 of 15,792 roster players to 278 of 15,791 active members. The one removed membership is an erroneous duplicate Georgia State row using the Syracuse receiver's provider ID. Georgia State's cornerback is separately represented by ID 5095071; ID 5212990 is also attached to Syracuse. The official roster metadata differs, and Georgia State's biography contains conflicting copied material. The candidate excludes the duplicate Georgia State membership after the source observation, uses the dated Syracuse receiver position, and quarantines disputed history/scouting rather than asserting an identity merge. Historical rows are not silently rewritten using today's evidence.

Availability and depth facts are game-scoped. A player missing from a report is not automatically marked healthy, and a past injury is not automatically carried into every later game. Confirmed-out players receive zero weight in the available-player contribution pool. Confirmed departed members are excluded from team pooling after the dated evidence. The broader roster representation still records injured players and the team's depth. Historical annual membership without timestamped contrary evidence remains a limitation. The backtest cannot establish the benefit of all newly observed 2026 roster/portal facts, because those facts are correctly unavailable to older games.

The expected-role branch learns contribution weights from full career information, recent usage, position, and the dated context. Recent workload summaries remain useful alongside the full history. Weights sum to one within each nonempty, eligible position group. They should be read as the model's relative emphasis within a unit, not as a depth chart, snap forecast, or medical probability.

The 2025 comparison uses identical immutable source inputs, one outer seed, and 16 internal TabM members for each variant. The control and full-history runs train before 2024, choose epochs on 2024, refit through 2024, and predict 2025. Both independently chose 14 epochs. The revision-checked context run uses the same 14-epoch budget chosen by the clean control and refits from scratch through 2024; no excluded article determines that budget. The three variants are the eight-game control, untruncated history, and untruncated history with dated context and role weighting. The last comparison combines role and context changes; it does not isolate the causal effect of injury data alone. The 2025 benchmark was previously examined elsewhere in this project and is not a new untouched test. Market lines enter only the comparison report.

The current revision-checked candidate also uses the clean control’s frozen 14-epoch budget and refits from scratch through 2025. Its budget was selected on 2024, not on 2026 outcomes. No 2026 outcome is used to fit or select the model. Scoring reloads the saved checkpoint, checks normalization and player vocabulary, validates probability distributions, and targets every modeled FBS/FCS-involved 2026 game after the scoring cutoff. A single outer seed is a research candidate, not equivalent evidence to the deployed three-seed ensemble.

A final publication/revision audit found four news articles whose current text was modified at or after kickoff. Their 56 facts were excluded, and both context candidates were refit from scratch. Earlier context runs are explicitly marked INVALIDATED and must not be used. The retained dated-article facts use the later of publication and modification time.

Validation covers complete-career gradients, cross-player isolation, missing/conflicting stat values, strict pregame cutoffs, game-specific report expiry, dated position changes, departures, ambiguous names, pending reports, and reloaded-checkpoint scoring. The full suite passed 157 tests with one pre-existing skipped test. The audit and export perform additional checks against the actual dataset and saved candidate.

To collect a fresh set of public evidence, build a dated snapshot, and train/score a candidate against a completed compatible source refresh:

```sh
PYTHONPATH=src python3 -m cfbpred.refresh_player_context \
  --source models/experiments/player_context_20260921_v1/source.json \
  --output models/experiments/player_context_NEW_RUN
```

For newly refreshed rosters, schedules, or box scores, supply a new source manifest with those compatible stores; the September 20 source used here is pinned and does not become current merely because public reports are recollected. Omit the `full_history` entry when the full-history extension must be rebuilt for a new base sequence store. The refresh command records unresolved sources and leaves production activation unchanged. It is an on-demand command; no recurring automation was created.

To inspect the saved current model's weights for every roster member in its forecasts:

```sh
PYTHONPATH=src python3 -m cfbpred.export_player_roles \
  --directory models/experiments/player_context_20260921_v1/runs/full_context_checked_2026_s0 \
  --output models/experiments/player_context_20260921_v1/runs/full_context_checked_2026_s0/player_game_roles.parquet
```

Public evidence includes the [Ohio State starting-quarterback announcement](https://ohiostatebuckeyes.com/news/2024/8/16/football-will-howard-named-ohio-states-starting-quarterback), [dated offensive-line update](https://ohiostatebuckeyes.com/news/2024/11/20/football-buckeyes-spoke-to-the-media-before-the-top-5-matchup-against-indiana), [contemporaneous pregame availability lists](https://buckeyesports.com/availability-report-seth-mclaughlin-josh-simmons-only-unavailable-starters-against-no-5-indiana/), [SEC reports](https://www.secsports.com/fbreports), and [Big Ten reports](https://bigten.org/fb/availability-reports/). Exact per-fact attribution is retained in the normalized facts table. Some ACC, Big 12, and MAC pages returned HTTP 404 during collection; this failure is recorded, not interpreted as a healthy roster.

More complete public availability archives and dated gamebook participation/starting-lineup tables would materially improve the role training data. Collecting reports prospectively, including their revisions and actual retrieval times, is necessary for a defensible future evaluation. More information is retained now, but an improvement over Las Vegas must still be demonstrated on games the model has not used for training or selection.
