# CFB Live Prediction Engine

Predicts, pregame and after every snap: home score, away score, and home win
probability — all three read off **one** predicted distribution (a discrete PMF
over integer margins), so they cannot contradict each other.

## The purity rule

**Nothing in the model comes from anyone else's model, or from any rating
algorithm — including our own.** Banned and enforced by tests: Elo, SP+, SRS,
FPI, returning-production PPA, betting lines, cfbfastR EPA/WP, and any
hand-fitted intermediate. The explicit exception is human recruiting/scouting
evaluations and the permitted 247 roster-talent input; the network learns how
these relate to game outcomes.

Team strength is learned by the network itself, three ways at once:
1. **Team & conference embeddings** — fitted during training. Since v16 every
   program with four or more games gets its own vector, FCS included — a
   single shared FCS token had made North Dakota State indistinguishable from
   a winless bottom-feeder, and blowout pricing paid for it.
2. **A 14-game raw-history encoder** — each side's actual prior results, each
   slot carrying the raw score, venue, recency and the *opponent's embedding*,
   so opponent adjustment is learned, not engineered. Histories and season
   records draw from every completed game (36.9k, FCS-vs-FCS included), not
   just the FBS-involved modeling population.
3. **Raw counting stats** — season-to-date yards/play, turnover and explosive
   rates, third-down conversion, roster composition, coach career record, and
   **official per-game stat lines for every position** (6.9M box-score rows,
   2004-2025): passing, rushing, receiving, defense (tackles/TFL/sacks/PD/
   hurries), kicking, punting, returns, plus top-back and top-target
   concentration shares. ESPN's QBR stat type is excluded at ingestion — it is
   a model output, not a stat. Also **returning production** (v15): the share
   of last season's pass/rush/receiving yards and tackles still on the
   current roster via exact athlete-id joins, whether QB1 returned, and
   log-yards imported from other programs through the portal — counting facts
   about people, on the same footing as recruiting points.

A learned **early-season gate** (two parameters, `sigmoid(b + w·gp)`)
multiplies the season-to-date stat block: a 2-game yards-per-play is shrunk
toward the league mean, a 10-game one trusted almost fully. **Schedule
context** (v16) tells the network whom those stats were compiled against —
mean preseason talent/recruiting of opponents faced and their FBS share, raw
averages of allowed inputs. **Production-weighted availability** tracks how
much of the season's output appeared in the previous game.

~1,235 raw inputs per prediction (53 live in-game state). The production
predictor is a 16-seed ensemble of averaged PMFs — single seeds vary by ±0.01
log loss and the mixture of margin PMFs preserves every consistency property.

## Run order

```bash
python3 src/cfbpred/ingest_pbp.py     # stage 1  download, trim, repair, validate
python3 src/cfbpred/pull_cfbd.py      # stage 2  CFBD -> parquet, every call logged
python3 src/cfbpred/pull_player_stats.py #       per-game stat lines, every position
python3 src/cfbpred/features_raw.py   # stage 3  raw feature store (no fits)
python3 src/cfbpred/publish.py        # stage 4a raw tables -> Supabase
python3 src/cfbpred/source.py         # stage 4b Supabase -> local training cache
# stage 5: the production seeds — train(seed=k, tag=f"raw_s{k}", s2d_gate=True)
python3 -c "import sys; sys.path.insert(0,'src'); from cfbpred.train import train; [train(seed=k, tag=f'raw_s{k}', s2d_gate=True) for k in range(16)]"
python3 src/cfbpred/ensemble.py       # stage 5b ensemble test metrics -> metrics_raw.json
python3 src/cfbpred/evaluate.py ensemble # stage 6 benchmarks + market tripwire -> eval_base.json
python3 src/cfbpred/predict.py        # stage 7  every game & play, chart (8-seed ensemble)
python3 src/cfbpred/model_card.py     # stage 7b architecture card -> web/model_card.json
python3 src/cfbpred/publish.py        # stage 8  predictions -> Supabase
python3 src/cfbpred/publish_lines.py  # stage 8b lines + v_model_vs_vegas (ALWAYS after stage 8:
                                      #          publish's cascade drop kills the view)
python3 src/cfbpred/pull_logos.py     # stage 8c ESPN team logos -> Supabase Storage (once)
pytest tests/ -q                      #          gates incl. purity checks
./run_web.sh                          #          dashboard at localhost:8787
```

## The 2026 season

The production ensemble is trained on 2014-2025 (2025 folded in after it
served as the v15-v17 test season); **2026 is the live held-out season**. Its
full schedule is pulled from CFBD into `data/parquet/schedule.parquet` —
separate from `games.parquet`, which stays the completed-game corpus the
ingest gates pin — and every FBS-involved 2026 game gets a pregame prediction
published as split `test`. Unplayed games are prediction targets only: they
never fill a history slot, never count toward a season-to-date stat, and
carry no label (enforced by `tests/test_schedule.py`). As games finish, the
Results page grades them against the line automatically.

```bash
./refresh_week.sh   # weekly: scores + current Vegas lines in, predictions out
```

The Slate marks upcoming games where the model departs from the current
spread by 4+ points (7+ gets the hot badge — the bucket that historically
covers 57.5%), and the rail can sort by disagreement size. Lines shown for
upcoming games are the CURRENT market, not closing; they firm up as kickoff
approaches, so edges shrink or grow with each refresh.

The weekly refresh now pulls current raw CFBD plays and player attribution
as well as box scores. These supply pregame rate/QB inputs while the repaired
cfbfastR release is unavailable. Game-keyed snapshots carry every available
prior observation into every future matchup; scheduled games never advance
the accumulators. `audit_future_features.py` rejects missing known box-score
joins and writes per-column coverage to `features/future_coverage.json`.
Unreported data remains missing. When the repaired 2026 release is published,
it takes precedence over overlapping fallback games. The fallback is separate
from the live-model training corpus.

Forecast refreshes use checkpoint-fitted scalers, preserve predictions after
kickoff, and archive current-season prediction snapshots locally. The weekly
publisher updates future forecasts without dropping app views. These locks
apply from September 6 onward; older regenerated forecasts are not proof of
what was actually published before kickoff.

For an independent pregame comparison, run `python3 src/cfbpred/backtest_pregame.py`.
It compares the existing hashed-game selection with chronological selection
on identical unseen 2023/2024/2025 seasons, with training-only normalization
and a chronological refit after epoch selection. The ordinary training
script still uses the production hashed-game holdout; its validation score
must not be presented as a future-season test score.

See the [September 6 feature and backtest audit](docs/FEATURE_AND_BACKTEST_AUDIT_2026-09-06.md)
for the fixes, matched-game results, and public availability-data coverage.

The subsequent [margin-loss and recent-form experiments](docs/MARGIN_AND_FORM_EXPERIMENTS_2026-09-06.md)
tested distance-sensitive margin loss, margin-aware checkpoint selection, and
three/five-game form with matchup comparisons. Neither established a later-year
improvement. Their code and frozen results are retained; production defaults
and its active feature schema are unchanged.

The next [architecture and preseason-input experiments](docs/ARCHITECTURE_AND_PRESEASON_EXPERIMENTS_2026-09-06.md)
tested six candidates, including team-order consistency, shorter training
windows, a shared team encoder, mixture distribution heads and public roster
passing priors. None qualified to replace production. The error audit identifies
a larger market gap before teams have current-season results; the report
documents that finding, the tested inputs and their limits.

The [combined-improvement and player audit](docs/COMBINED_IMPROVEMENTS_AND_PLAYER_AUDIT_2026-09-06.md)
then tested the compatible changes together with 16 seeds. The retained
candidate improves pooled margin/total/probability scores slightly, with an
uncertain margin gain. The audit also stages individual recruiting and player
history links, identifies limits of the current team-level summaries, and
quarantines suspicious player-ID continuity in the new catalog.

The [conference availability archive backfill](docs/CONFERENCE_AVAILABILITY_ARCHIVES_2026-09-22.md)
adds 10,421 validated 2023–2025 player statuses (official Big Ten, MAC, Conference
USA and Mountain West reports, Sun Belt feed revisions, Wayback captures and
timestamped reproductions), covering 71% of team-games under a conference reporting
policy. A paired three-seed retrain on the v6 architecture was neutral, so v6
remains active; the model does not yet respond materially to reported absences.

The [player identity and individual-input implementation](docs/PLAYER_IDENTITIES_AND_NETWORK_2026-09-06.md)
corrects reviewed player/recruiting identity errors and builds a network input
for every rostered player, combining dated scouting with prior college and
transfer history. Its rolling backtest and current-season pregame forecast
pipeline are separate from the existing live ensemble; the linked report
records coverage, historical comparisons, and adoption status.

The [player sequence experiments](docs/PLAYER_SEQUENCE_EXPERIMENTS_2026-09-07.md)
compare individual game context, recent contribution weighting, and joint
player-stat training separately and together, using chronological folds and
equal-size ensembles. The selected game-context model is now the production
pregame predictor. Its three-member ensemble reads every roster member's dated
scouting, career windows and eight most recent observed individual games. The
existing raw-DNN state tower continues to handle in-game updates because the
player model was trained and validated for pregame inference.

The [architecture study](docs/ARCHITECTURE_RESEARCH_2026-09-07.md) tests a
residual MLP, history attention, opposing-position attention and a TabM-style
parameter-efficient ensemble over that complete input set. The frozen
development rule selected TabM-16. On the 2,398-game 2023-2025 assessment it
reduced margin MAE from 12.6136 to 12.5005, with a paired week-cluster 95%
interval of [-0.2048, -0.0284], while also improving total MAE, winner log
loss, Brier score, CRPS and calibration. On closing-line-matched games it
reduced the remaining model-to-market margin-MAE gap by 19.1%. The refitted
`player-sequence-tabm16-v2` model is now the active pregame predictor. Its
audited predictions and full PMFs cover all 1,610 modeled 2026 games, including
a strictly pre-kickoff reconstruction of the completed Week 1 slate.

After refreshing the local raw data and feature store, run
`python3 src/cfbpred/refresh_player_sequence_production.py --promote` to rebuild
and audit all future player-model forecasts. The latest successful result is
recorded in `models/player_sequence_production/refreshes/latest.json`. The
production refresh replaces only games that have not kicked off. Previous
local files are archived under `models/player_sequence_production/promotions`
for rollback. `src/cfbpred/verify_player_sequence_publish.py` reads the active
Supabase rows back and requires exact game/scoreline coverage plus numeric
agreement within database `real` precision.

v18 (retrained 2026-08-27, 16 seeds): with no completed 2026 games yet, run
metrics are graded on the game-level validation holdout — never in a
gradient, but early stopping selected on it, so read them as slightly
generous until 2026 games grade for real. Ensemble on that holdout: WP log
loss 0.362, ECE 0.017, margin RMSE 11.24 (play level); pregame margin MAE
12.42 vs the closing line's 12.17; the 7+ disagreement bucket covers 61.1%
(201-128). The 2025-test numbers in the table above are v17's, from before
2025 entered training.

## Dashboard

`web/` is a zero-build static app (Archivo + Spline Sans Mono, light and dark
mode) reading Supabase anon REST. Three pages:

- **Slate** — every FBS-involved game by week: real team logos (ESPN ids →
  `team-logos` storage bucket, light + dark variants), the scorebug, a
  *Model vs Vegas* board (margin gauge, total gauge, de-vigged moneyline win
  probability), and the snap-by-snap win-probability curve with hover
  telemetry.
- **Results** — aggregate model-vs-market scoreboard from the
  `v_model_vs_vegas` view: MAE/RMSE on margin and total, straight-up and ATS
  records, Brier vs the de-vigged moneyline, per-season chart and table,
  filterable by held-out / in-sample split.
- **Model** — the architecture, drawn from the live checkpoints via
  `model_card.py` → `web/model_card.json`: input inventory by group, the
  two-tower flow diagram, per-component parameter table, the learned gate
  curves (live handover α and the early-season gate), and a real held-out
  game's margin PMF showing how WP/score/total are read off one distribution.

The closing line appears in the app and in `public.game_lines` strictly as a
BENCHMARK; the purity rule stands — `source.py` never pulls it and the
leakage tests would fail if a lines column reached the feature store.

## Coverage

- **19,817 games, 2001–2025** (every FBS-involved game): pregame prediction.
- **1,574,157 plays, 2014–2025** (8,677 games with play-by-play): full curve.
- Every row carries its split — `pre2014` / `train` / `val` / `test` — and the
  dashboard badges in-sample vs held-out so they can never be confused.

## Held-out results (2025 season, v17 16-seed ensemble)

| bucket | log loss | cfbfastR published | ECE |
|---|---|---|---|
| all | **0.402** | 0.438 | 0.015 |
| Q1  | **0.541** | 0.617 | 0.038 |
| Q2  | **0.453** | 0.491 | 0.023 |
| Q3  | **0.361** | 0.368 | 0.017 |
| Q4  | **0.252** | 0.258 | 0.007 |

Beats the published cfbfastR WP model in **all four quarters** (since v15).
In-game totals: predicted-total RMSE fell **16.36 → 11.87** once the state
gained points-so-far and both teams' running efficiency.

Pregame margin RMSE **16.14** vs the closing line's 15.04 — behind the market,
as an honest no-market model should be. Judged game-by-game against the
closing spread on held-out games, mean absolute margin error is **12.65 vs
the market's 12.12** (+0.53; v15 was +1.06, v14 +1.14). The v16 halving came
from giving FCS programs real identities and real histories — FCS-involved
games went from +3.7 to +0.7 behind the line, 21+ blowouts from +2.7 to +1.3.
On ordinary FBS-vs-FBS games under 21 the model is within ~0.2-0.4 of the
close; what remains there is week-of information — injuries, opt-outs, depth
charts — the deliberate price of the no-market, no-news rule.

Where the model disagrees with the closing spread by 7+ points it covers
**57.5%** of the time (539 held-out games, break-even 52.4%) — the market
wins on average because it has the news, but the model's loudest
disagreements are systematically right. The Results page charts this.

Measured and REVERTED as null (v17 round): 20-game history windows, opponent
talent inside history slots, FCS box-score aggregation, and wider pregame
towers (160/192/224 sweep) — none moved held-out error beyond seed noise. At
~17k games the pregame tower is saturated; the code keeps the knobs
(`pre_hidden`, `margin_aux`, the unused `opp_talent` history column) for
retests if the corpus grows.

## Lessons encoded in the code

- Team embeddings will happily memorise training outcomes: identity dropout
  (30%) + early stopping on held-out games, or validation collapses.
- The state tower gets exactly **two noised scalars** from pregame (E[margin],
  E[total]). Handing it the full embedding gave it a per-game fingerprint and
  it memorised through it.
- Postseason weeks restart at 1 — join season-to-date stats on a continuous
  week or bowls get week-1 stats.
- The source data contains duplicate games and Week-0 games labelled week 1.
- Labels come from schedules, never play-by-play running scores (wrong 2.8%).
