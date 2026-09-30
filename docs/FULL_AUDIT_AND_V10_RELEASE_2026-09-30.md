# Full model audit, v9 and v10 releases — September 30, 2026

Overnight audit of the pregame model's architecture, inputs and data sources, followed by three
pre-registered backtest rounds and two releases. The active pregame model is now
**player-full-context-tabm16-v10** (variant `full_context_v5a_fcs`, five outer seeds, published
05:19 UTC): the v8 architecture trained on every FCS-vs-FCS game and given offseason
transfer-portal and NFL-draft inputs. v9 (all-FCS training alone) was live from 04:52 to 05:19 UTC. The purity rule is unchanged: no betting lines, ratings, EPA/PPA or win probabilities
enter any input.

## 1. What the model is (as audited)

- **Network:** 922k parameters. A swap-consistent TabM-16 pregame tower (width 192): 324 scalar
  inputs, 24-d team and 8-d conference embeddings, and a 14-game raw-history encoder. The player
  encoder covers every rostered player: scouting, career events and dated availability/role
  context.
- **Parameter budget:** 73% of the parameters are the 16 members' output heads for the margin and
  total PMFs. The whole player encoder is about 15k parameters (under 2%), consistent with the
  earlier finding that players move this model little.
- **Training:** labeled games 2001–2025 (about 34k). Loss is soft cross-entropy on margin + 0.5
  total + 0.5 roster-only head, with label smoothing 0.1. AdamW (lr 2e-3, weight decay 3e-4), no
  schedule, fixed 14 epochs, and no sample weights. Team embeddings are static across 25 seasons.
- **Epoch budget check (new):** early-stopping selection for the current network (train ≤2023,
  validate 2024) bottoms out at epoch 12 and stays flat to about 20. The inherited 14-epoch budget
  is right, so there is no gain from retuning it.
- **Readout:** since September 29 the published spread/total is the interpolated PMF median, also
  in the live scorer.

## 2. Data audit

A season-by-season scan of all 270 active inputs flagged 33 whose missingness or level shifts in
2024–26. Most are real changes or better coverage: ESPN roster backfill raised returning-production
coverage, the portal era raised imported production, and availability coverage grew. Two are
defects:

| Input | Defect |
| --- | --- |
| `talent_composite`, `has_talent`, `sos_talent` | CFBD stops publishing FCS composites after 2023, and the FBS floor is lifted in 2024 (fixed as opt-in percentiles on Sept 29; backtest-neutral). |
| `upper_share` | CFBD historical rosters carry calendar years or final-career classes in `year`, so the share is 1.0 for every team through 2015 and falls to 0.47 in 2026. It is an era marker, not experience. Dropping it cost 0.014 MAE (round 2), so the network uses it as an era signal. It stays. |

### New data connections (CFBD API has 88 endpoints; these three were unused)

| Source | Rows | Use |
| --- | ---: | --- |
| `/player/portal` 2021–26 | 18,895 | Transfers in/out per team-season: count, stars, 4–5 stars, 247 rating. Only transfers dated before that team's opener count. |
| `/draft/picks` 2001–26 | 6,647 | Players lost to the draft: count, round 1, rounds 1–3. |
| `/coaches` game counts (+ `/coaches/tenures`) | 2,692 tenures | In-season head-coach change flags, dated to the game (e.g. Oklahoma State from Week 5 of 2025, LSU from November 8). |

`pull_roster_movement.py` pulls them. The nightly refreshes portal and draft (about 35 calls).
The features are opt-in (`mvmt_`, `cchg_`) and covered by point-in-time tests. Endpoints that
carry SP+/SRS/FPI/PPA (`/coaches/seasons`, `/ppa/*`, `/ratings/*`, `/metrics/*`, the `ppa` fields of
`/passing/*`) were deliberately not used.

## 3. Backtests (pre-registered; `models/experiments/roster_movement_20260930/decision_rule.json`)

Rolling origin 2021–2025. Every arm uses the same store chain, 3 seeds, 14 epochs and the
median readout. The primary cohort is 4,508 FBS-involved games with a closing spread.

| Round 1 | FBS-involved MAE | Log loss | FBS vs FBS | Total MAE | vs base (95% CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| base (v8 architecture, same data) | 12.5797 | 0.49486 | 12.5312 | 12.732 | |
| v5a portal + draft | 12.8270 | 0.49589 | 12.7600 | 12.793 | +0.247 [+0.071, +0.442] |
| v5b coach-change flags | 12.5897 | 0.49427 | 12.5336 | 12.733 | +0.010 [−0.021, +0.041] |
| v5 both | 12.7264 | 0.49498 | 12.6767 | 12.791 | +0.147 [+0.028, +0.273] |
| **fcs100: every FCS-vs-FCS game (v8: 25%)** | **12.5460** | **0.49389** | **12.5195** | **12.689** | **−0.034 [−0.085, +0.019]** |
| closing line | 12.1607 | | | 12.475 | |

- **fcs100 across seasons:** −0.10 / +0.01 / +0.09 / −0.09 / −0.08 for 2021–2025.
- **fcs100 cross-tier gaps:** power vs FCS 1.10 → 0.77, power vs G5 0.66 → 0.49, G5 vs FCS
  0.09 → 0.03. The Weeks 1–3 gap fell from 0.44 to 0.31.
- **Portal/draft failure:** it collapsed only where the model had never seen portal data. The 2021
  fold went 13.04 → 14.40; the 2022 fold, with one training season, was even. On 2023–25 it was
  −0.04 / −0.11 / +0.03.
- **Round 2:** fcs100 without `upper_share` scored 12.5600, a rejection.
- **Round 3:** portal/draft on top of fcs100, judged on the 2023–25 folds plus a 2026 guard.
  See section 5.

## 4. The release and the 2026 check

- **Staging and audit:** v9 was staged from five production refits (trained through 2025, no 2026
  outcomes) with every lineage check. The full-season audit passed: 1,609 games, 0 missing FBS
  rosters, player history strictly before kickoff.
- **Checks:** adapter parity was 1.5e-8 over 1,019 games. Prediction readback error was 3.4e-6
  and PMF readback 5e-9. Ranking weeks 0–4 (266 teams each) read back exactly. Live priors come
  from v9 for all 131 upcoming games.

**2026 weeks 1–4 (331 FBS-involved games; descriptive, as pre-registered):**

| | Margin MAE | FBS vs FBS | FBS vs non-FBS | Total MAE |
| --- | ---: | ---: | ---: | ---: |
| v9 (fcs100) | 12.540 | 11.573 | 14.331 | 11.730 |
| same-data v8 architecture | 12.445 | 11.570 | 14.068 | |
| v8 as published | 12.434 | 11.558 | 14.056 | 11.766 |
| closing line | 11.607 | 11.042 | 12.655 | 11.618 |

This season disagrees with the backtest. v9 minus the same-data baseline is +0.095, 95% CI
[−0.08, +0.27], all of it in games against non-FBS opponents. The backtest's −0.034 comes from 13.6×
more games and won both most recent seasons (2024, 2025). Weighting both samples by their precision
gives about −0.02. v9 shipped under the rule written before any result. Only 19 regular-season
cross-tier games remain in 2026, so the practical exposure before bowls is small.

**Rollback:** `models/player_sequence_production/release_backups/20260930T045210Z/`, which holds
the v8 active pointer, rankings and database rows. Re-running the release publisher with v8's
staged pointer restores it. Re-evaluate after Week 8 on games v9 forecast before kickoff.

## 5. Round 3: portal/draft on top of fcs100, released as v10

Pre-registered before any round-3 fit: `v5a_fcs` (portal + draft inputs, every FCS-vs-FCS game) vs
fcs100. It was judged only on the 2023–25 folds, whose training holds at least two portal seasons,
plus a 2026 guard. Round-1 per-season portal numbers had been seen, so the rule added the guard.

| Folds 2023–2025 | FBS-involved MAE | Log loss | FBS vs FBS |
| --- | ---: | ---: | ---: |
| fcs100 (v9) | 12.4359 | 0.48768 | 12.4639 |
| **v5a_fcs** | **12.3971** | **0.48739** | **12.4411** |

- **Per season:** −0.081 (2023), −0.063 (2024), +0.025 (2025).
- **Power vs FCS gap:** 0.67 → 0.52.

**2026 weeks 1–4 guard (331 FBS-involved games).** v5a_fcs was scored from five production refits,
full-season re-score:

| | Margin MAE | FBS vs FBS | FBS vs non-FBS | Total MAE | WP log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| **v10 (v5a_fcs)** | **12.217** | **11.301** | **13.915** | 11.740 | **0.3197** |
| v9 (fcs100) | 12.540 | 11.573 | 14.331 | 11.730 | 0.3278 |
| v8 as published | 12.434 | 11.558 | 14.056 | 11.766 | 0.3257 |
| closing line | 11.607 | 11.042 | 12.655 | 11.618 | |

- **Paired vs v9:** −0.323 (game-bootstrap 95% CI [−0.499, −0.148]). vs v8: −0.217.
- **Gap to the line** fell from 0.83 to 0.61. By matchup: power vs power −0.07 (v10 now beats the
  line), power vs G5 0.70 (v8 0.90), power vs FCS 2.21 (2.31), G5 vs FCS 0.54 (0.71).
- **By week:** Week 1 gap 0.88 (v8 1.29), Weeks 2–3 about 0.4 (v8 0.62–0.64), Week 4 0.74 (0.63).
- **Winners picked:** 87.0% (v8 86.1%).
- **Against the spread:** 52.9% (v8 51.4%); 7+ point disagreements 53.0% on 66 (v8 47.3% on 74).
- **Caveat:** the portal idea was partly motivated by 2026 Week 1 misses. Nothing was tuned on 2026,
  and the backtest agrees on direction.

**v10 release checks:**
- The staging lineage checks passed.
- The full-season audit passed (1,609 games, 0 missing FBS rosters, history strictly pregame,
  no 2026 outcomes).
- Parity was 1.5e-8 over 1,019 games, and the season file is identical to the guard's scores.
- Prediction readback 2.5e-6 and PMF 7.9e-9. Ranking weeks 0–4 read back exactly.
- Live priors come from v10 for all 131 upcoming games.
- The nightly's current-week ranking step was dry-run with v10 (week 4, 266 teams).

**Rollback:** `release_backups/20260930T051939Z` restores v9, and `release_backups/20260930T045210Z`
restores v8. Both are the release publisher's own archives.

## 6. Other changes tonight

- "Historical reconstruction" labels were removed from the web dashboard and the mobile app (Expo
  OTA `2d2e1628`, production channel, verified on iOS and Android). The held-out accuracy filter is
  kept; it removes flagged rows from held-out accuracy without showing any label.
- The nightly refresh now pulls portal/draft data. Supabase `game_scalars` has the 34 new opt-in
  columns.

## 7. What did not change, and next steps

- **Past seasons (2001–2025):** the app still shows v4 in-sample reconstructions. Re-scoring them
  with v9 needs a complete-history store chain (about 1.2 GB; disk has about 2.9 GB free) and is
  in-sample either way.
- **Web model card:** the Model page card (`web/model_card.json`) still describes the retired
  raw-DNN from August.
- **Portal features:** keep checking v10 on games it forecasts before kickoff (Weeks 5–13).
  The portal inputs are new, and 2026 was part of their motivation.
- **Still unaddressed:** the bowl gap (+0.93) is still unaddressed: opt-outs have no source. The
  player encoder is too small to matter (2% of parameters); growing it needs the availability
  backfill to reach training first.
