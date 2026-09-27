# Measured benchmarks (all computed locally, FBS-vs-FBS)

## Data inventory (verified)
- schedules 2001-2025: 17,472 FBS-vs-FBS games, 100% score coverage, ZERO ties
- release PBP parquet 2014-2025: 8,677 FBS-vs-FBS games, 1,574,157 regulation plays
- repo PBP parquet 2002-2021: 405-col raw schema, timeouts BROKEN, coverage incomplete pre-2013

## Pregame benchmarks (2014-2025 unless noted)
| model | margin RMSE | WP logloss | SU acc |
|---|---|---|---|
| closing consensus spread (n=8670) | 15.693 | 0.51191 | 0.7404 |
| walk-forward Elo+HFA linear (2004-2025, n=15495) | 16.459 | 0.53053 | 0.7290 |
- closing TOTAL RMSE 16.184 (n=6532)
- LEAKAGE TRIPWIRE: margin RMSE < 15.5 => go find the leak

## Live WP benchmark: published cfbfastR wp_before (n=1,539,455)
| bucket | logloss | brier | acc | ECE | mean WP | actual |
|---|---|---|---|---|---|---|
| ALL | 0.43844 | 0.14566 | 0.7777 | 0.0336 | 0.5482 | 0.5805 |
| Q1 | 0.61661 | 0.21461 | 0.6493 | 0.0598 | 0.5236 | 0.5814 |
| Q2 | 0.49140 | 0.16319 | 0.7568 | 0.0392 | 0.5466 | 0.5818 |
| Q3 | 0.36767 | 0.11848 | 0.8275 | 0.0214 | 0.5608 | 0.5822 |
| Q4 | 0.25762 | 0.07898 | 0.8890 | 0.0145 | 0.5638 | 0.5764 |
NOTE: systematically biased AGAINST home team by 3.2pp overall, 5.8pp in Q1.
Q1 logloss 0.6166 is WORSE than a pregame-only Elo model (0.5305) -> quantifies the gating opportunity.

## Distribution facts
- margin: mean +4.64, SD 21.14 | total: mean 54.39, SD 18.08
- corr(margin,total) = +0.0435 ; corr(home_pts,away_pts) = -0.155
- corr(spread resid, total resid) = +0.0440
- HFA: 58.9% home win / +4.74 margin (home sites); 51.2% / +1.17 (neutral)
- key numbers |margin|: 3 -> 9.54%, 7 -> 8.39%, 10 -> 4.71%, 14 -> 4.44%, 17 -> 3.55%, 21 -> 3.81%
- margin bins: |m|<=56 covers 99.26%, |m|<=70 covers 99.93%
- spread residual: skew -0.045, excess kurtosis +0.175 (near-Gaussian PREGAME)

## Conditional remaining-margin distribution by time (THE key architecture finding)
| bucket | n | SD | excess kurt | P(no further change) |
|---|---|---|---|---|
| Q1 | 360,406 | 19.11 | 0.115 | 3.0% |
| Q2 | 445,143 | 14.71 | 0.205 | 4.9% |
| Q3 | 358,008 | 10.46 | 1.035 | 10.4% |
| Q4 early | 230,981 | 6.92 | 6.024 | 28.0% |
| 2-5 min | 77,753 | 4.77 | 21.617 | 55.1% |
| <2 min | 69,207 | 3.71 | 69.008 | 70.5% |
=> Gaussian fine pregame/Q1-Q2, COLLAPSES in Q4.

## Gaussian vs discrete WP (train 2014-22, test 2023-25)
| bucket | n | Gauss LL | Discrete LL | gain | Gauss ECE | Disc ECE |
|---|---|---|---|---|---|---|
| <2m | 22,228 | 0.31127 | 0.29190 | +0.01937 | 0.0363 | 0.0241 |
| 2-5m | 22,166 | 0.25681 | 0.24739 | +0.00942 | 0.0318 | 0.0249 |
| 5-10m | 30,090 | 0.29694 | 0.29171 | +0.00523 | 0.0353 | 0.0232 |
| Q2 | 121,012 | 0.50059 | 0.49611 | +0.00448 | 0.0475 | 0.0234 |
| Q1 | 101,859 | 0.61782 | 0.61088 | +0.00694 | 0.0479 | 0.0222 |
=> discrete wins on logloss AND uniformly on calibration.

## Time-scaling of margin uncertainty
- brief's sqrt(t): weighted RMSE 1.0262 ; pure power law t^0.4595: 1.3129
- BEST: sd(t) = sqrt(4.00^2 + 0.00197 * t^1.50) -> weighted RMSE 0.2481 (4x better)
- sqrt form fails badly late: at 30s left actual SD 3.86, sqrt predicts 1.76

## Transfer-portal claim (brief says autocorrelation weakened post-2021)
- ridge-SRS year-over-year corr: pre-2021 mean 0.6414 (n=19, sd 0.0443, range .5695-.7276)
- post-2021 mean 0.6028 (n=4, range .5359-.6579)
- t = 1.533, df=21, p ~ 0.14 -> NOT SIGNIFICANT. Do not fit separate era decay.

## EP validation targets (cfbfastR ep_before, 1st & 10)
own 1 (ytg99): -0.731 | own 20 (ytg80): 0.634 | midfield: 2.526 | opp 20: 4.176 | opp 10: 4.693
by down @ own 25-30: 1st 0.806, 2nd -0.038, 3rd -0.958, 4th -1.951
EPA/play: mean -0.0025, SD 1.467 | team-season offense EPA/play 2024 range: +0.218 (Miami) to -0.198 (Houston)

## Timeout data quality (release files)
- start of half at 3 TO: 99.6% (H1), 99.3% (H2) -> resets correct
- monotonicity violations within half: 104 / 155,874 = 0.067%
- bad values (neg/null/>3): 0.3%/season 2014-21, rising to 4.6% in 2024
=> DO NOT need to derive from scratch (contradicts brief); repair only.

## PROTOTYPE RESULTS (validated locally, PyTorch MPS, train 2014-21 / val 2022 / test 2023-25)

### Output head comparison — SAME features, SAME net (2x96 MLP, 15 feats)
| bucket | Gauss LL | Discrete LL | Gauss ECE | Discrete ECE | Gauss mRMSE | Disc mRMSE |
|---|---|---|---|---|---|---|
| ALL | 0.55499 | **0.39772** | 0.0850 | **0.0089** | 11.448 | 11.512 |
| Q1 | 0.83521 | **0.51295** | 0.1481 | **0.0185** | 15.161 | 15.234 |
| Q2 | 0.61712 | **0.43422** | 0.1035 | **0.0134** | 12.484 | 12.552 |
| Q3 | 0.47716 | **0.36975** | 0.0744 | **0.0091** | 9.814 | 9.857 |
| Q4e | 0.31217 | **0.29036** | 0.0361 | 0.0188 | 7.095 | 7.204 |
| <5min | 0.24672 | **0.24290** | 0.0192 | 0.0125 | 4.815 | 4.852 |
=> Discrete margin PMF wins decisively on WP logloss + calibration, loses nothing on margin RMSE.
=> vs cfbfastR published: ALL 0.43844 -> 0.39772 (-9%); Q1 0.61661 -> 0.51295 (-17%)
Training time: 2-3 s for 12 epochs over 1.02M rows on MPS. Compute is a non-issue.

### Pregame decay rate lambda in exp(-lambda * frac_elapsed), logistic, 2022 validation
| lambda | val LL | test LL | test Q1 LL |
|---|---|---|---|
| 0.00 | 0.43432 | 0.39890 | 0.51325 |
| 0.75 | 0.43092 | 0.39709 | 0.51158 |
| **1.50** | **0.42988** | 0.39815 | 0.51358 |
| 4.00 (nflfastR) | 0.43464 | 0.40905 | 0.52086 |
=> optimal lambda ~1, NOT 4. Hard-coding nflfastR's decay costs ~0.012 logloss.
=> GATE MUST BE LEARNED, initialized slow.

### Gating ablation (logistic, test 2023-25)
| model | ALL | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| state only | 0.44467 | 0.60644 | 0.48905 | 0.39695 | 0.28147 |
| + pregame gated (lam=4) | 0.40902 | 0.51962 | 0.44858 | 0.38478 | 0.27944 |
| + pregame ungated | 0.39877 | 0.51404 | 0.43291 | 0.36965 | 0.27525 |
=> pregame strength worth 0.09 logloss in Q1. Aggressive fixed gating HURTS.

## Hard constraints discovered
- PARITY: margin and total ALWAYS share parity — verified 17,472/17,472 games. (T+-M)/2 breaks 50% of the time if drawn independently.
- Team score of exactly 1 is IMPOSSIBLE (0 occurrences); 4 occurs once in 34,944 team-games.
- margin variance depends strongly on total: SD(margin) = 16.58 (total<45) -> 23.85 (total 55-65), +44%.
  Uncorrelated in MEAN (r=0.043) but strongly dependent in VARIANCE.

### Kickoff-state degradation — VALIDATES the brief's auxiliary-loss argument
Prototype (discrete head, trained on ALL plays, no aux pregame loss), evaluated at
near-kickoff states only (t>3400s, tied), test 2023-25, n=19,490:
  margin RMSE 16.191 | WP logloss 0.55518 | SU 0.7157
Compare dedicated pregame walk-forward Elo+HFA model: RMSE 16.459 | LL 0.53053 | SU 0.7290
=> The play-level model is WORSE in logloss at kickoff than a trivial dedicated pregame model,
   because 1.02M mid/late-game rows dominate the loss. The brief's diagnosis is CORRECT.
   Fix = auxiliary pregame head loss + train pregame tower on the full 2001-2025 game corpus.
   (This is separate from, and does not contradict, the finding that lambda=4 gating is too aggressive.)

## Stage-1 data-quality gate (audit vs authoritative schedule scores, FBS-FBS 2014-25, n=8,677)
- exact score match:            8,431 (97.2%)
- home/away ORIENTATION SWAP:       8
- undercount (truncated feed):     86
- overcount (duplicated plays):    41
- total mismatched:               246 (2.8%); 108 off by >=7 points
=> NEVER derive final scores from PBP running-score columns. Use schedules/parquet.
=> cfbfastR NEWS confirms: "A truncated game models perfectly cleanly and produces EPA,
   drive results and a box score that all look reasonable and are all wrong."

## CFBD garbage-time filter (verified from cfb-api server source, ppa.service.js) — KEEP a play if:
  period=1                                        (Q1 never garbage time)
  period=2 AND not scoring AND |margin| <= 38     (scoring: <= 45)
  period=3 AND not scoring AND |margin| <= 28     (scoring: <= 35)
  period=4 AND not scoring AND |margin| <= 22     (scoring: <= 28)
NOTE: /stats/season/advanced uses <=29 (not 28) for Q4 scoring plays — endpoints disagree by 1.

## Ridge opponent-adjustment prototype (2024, garbage-time filtered, rush|pass only)
97,118 plays x 269 columns. Fit time <0.05 s per lambda (scipy lsqr). Compute is a NON-ISSUE.
lambda=200: HFA +0.045 EPA/play, OFF sd 0.0859, DEF sd 0.0817
  best OFF: Ohio State +0.252, Miami +0.233, Oregon +0.179
  best DEF: Notre Dame -0.207, Texas -0.198, Penn State -0.153
Raw (unadjusted) EPA/play 2024 leader was Miami +0.285; adjustment correctly promotes Ohio State.
=> ridge (MAP of Gaussian crossed random effects) is the practical choice; lme4 REML buys
   nothing here and costs orders of magnitude more time. Tune lambda by out-of-sample game
   prediction, not REML.

## Measured team offensive EPA/play range, FBS 2024 (rush|pass, all game states), n=134 teams
  +0.285 (Miami) ... -0.228 (Southern Miss).  Elite is ~+0.25-0.29, NOT +0.30-0.45.

## CONVERGENCE: my lambda measurement vs the literature
Independent research found: sportsdataverse's CFB pipeline inherited nflfastR's
`spread_time = spread * exp(-4 * elapsed_share)` VERBATIM and never re-estimated the constant
for college. lambda=4 => half-life 10.4 game-minutes; only 1.83% of the prior survives to 0:00.
Maddox et al. (2022, arXiv:2207.13747), the one substantive CFB in-game WP paper, fit an
EXPLICIT blend weight on college data: D = 0.09589 + 0.00018*t + 0.02523*|lead|
 -> linear in elapsed time AND in |lead|, with a nonzero floor.
Their blended model beat a Lock-Nettleton random forest 0.1250 vs 0.1705 Brier.
ESPN's live WP (measured from their feed by the researcher): seeds EXACTLY at FPI and decays
  b ~= 0.138 + 0.883 * (fraction of regulation remaining), R^2=0.99, linear in time-remaining,
  with a FLOOR of 0.138 -- matching ESPN's published "prior never completely disappears".
MY INDEPENDENT MEASUREMENT (2022 validation, CFB): optimal lambda ~ 0.75-1.5, NOT 4.
=> Three independent lines of evidence agree: the NFL decay constant is far too fast for CFB.
=> DESIGN: learn the gate; include |lead| as a gating input, not just time; allow a nonzero floor.

## Overtime findings (measured)
- OT games 368/8,677 = 4.24% of FBS-FBS games 2014-25; OT plays 5,025/1,579,182 = 0.32% of rows
- EPA populated on OT plays: ZERO (cfbfastR trains EP on period<=4 only; OT is out of support)
- home win rate in OT games: 50.82% (n=368) -- essentially a coin flip
- by rules era: 2014-18 n=155 home 50.3% | 2019-20 n=44 home 52.3% | 2021+ n=169 home 50.9%
  => no detectable era effect on home win rate
- pregame Elo barely predicts OT outcomes (buckets non-monotone: .44/.58/.60/.44/.57, n=30-143)
- OT final |margin| concentrated on 3 (146) and 7 (94) of 368
=> RECOMMENDATION: exclude period>4 from training (0.32% of rows, no EPA anyway); keep OT points
   in the final-margin LABEL; handle live OT with a separate small model. No rules-era indicator
   needed in the main model because OT is excluded from it.
