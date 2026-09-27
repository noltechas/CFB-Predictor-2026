# Prediction and player-data gaps — September 21, 2026

The active pregame model has substantial individual player information, but it does not have a complete, dated representation of each upcoming lineup. The highest-priority hypothesis is that confirmed starters, availability, current positions and expected roles will add information that aggregate roster history cannot supply. This audit identifies missing information; it does not establish how many points any proposed addition will improve accuracy.

The audit read the source, the last completed production input snapshot (`20260920T122506160000Z`), retained historical assessments and the live database. Database access was read-only. Production code, model weights and published predictions were not changed. Only this report and two aggregate audit JSON files were added.

## Actual performance and active model

The database confirms `player-sequence-tabm16-v2` for all 409 completed 2026 games with predictions and market spreads at the observation time, September 21 at 22:32 UTC. Pregame uses three outer seeds with sixteen internal TabM members each. The separate `raw-dnn-delta-v22` state model handles live updates; the newer individual player encoder has not been trained as a live model.

| Published 2026 comparison | Matched games | Model margin MAE | Market margin MAE | Model minus market |
| --- | ---: | ---: | ---: | ---: |
| Any FBS participant | 260 | 12.536 | 11.667 | +0.869 |
| FBS versus FBS | 157 | 11.516 | 10.978 | +0.538 |
| FBS versus another division | 103 | 14.091 | 12.718 | +1.373 |
| Neither participant FBS | 149 | 12.286 | 11.903 | +0.383 |

Lower is better. MAE is average absolute error in the final home-minus-away scoring margin. On the 260 FBS-involved games, the model picked 225 winners and the market favorite won 230. The model's spread-side record was 133/256 excluding pushes. Winner selection, margin accuracy, probability calibration and profitable betting are different objectives; no one metric establishes the others.

Within the 157 FBS-versus-FBS games, the model-minus-market MAE differences are +1.652 in Week 1, -0.216 in Week 2 and +0.189 in Week 3. These small slices describe where the current errors occurred, not a proven trend or a causal explanation. The larger gap against lower divisions motivates separate error analysis and coverage work.

The retained 2023–2025 assessment gives a larger historical comparison: on the same 2,396 games with closing lines, TabM margin MAE is **12.5079** versus **12.0296**, a **0.4782-point gap**. The often-quoted 12.5005 is across 2,398 games and must not be directly subtracted from the 2,396-game market result. The current architecture improved on its context control by 0.1130 points over all 2,398 assessment games. Those seasons have been used repeatedly for research and are not untouched tests.

Sources: [published performance audit](audits/2026-09-21/published_performance.json), [active registry](../models/player_sequence_production/active.json), [historical assessment](../models/experiments/architecture_research_20260907_v1/report.json).

## What reaches the player model

| Question | Implemented input | Remaining limitation |
| --- | --- | --- |
| New players and freshmen | Every available roster member is encoded, with separate high-school, JUCO and prep recruiting information when identity-linked and usable. | Recruiting ratings, ranks and dated measurements are not full high-school game histories, practice observations or a forecast of immediate playing time. |
| Full college history | Previous-season, two-seasons-ago, older-career and current-season aggregate windows; 27 counting stats with observation counts. Eight recent observed individual games add opponent, program and timing context. | Only collected and correctly linked observations exist. Earlier career games lose some sequence detail through aggregation. Missing box scores do not establish nonparticipation. |
| Transfers | Corrected person identities retain prior-program production when the player appears on the destination roster. Current roster makeup and imported production reach the model. | There is no ingested portal-event timeline for entry, destination announcement, withdrawal, eligibility or effective membership changes. The roster feed determines the represented destination. |
| Roster composition | All roster members, position-group pooling and counts, recruiting, returning production, team histories and raw performance summaries. No top-N roster cutoff. | A roster is not a depth chart. Starting units, backups, position switches and player combinations are not fully represented. |
| Depth charts and roles | Past box-score workload provides indirect evidence. | No explicit dated depth-chart rank, confirmed starter, expected snap share, routes, receiving targets or calibrated playing-time model. |
| Injuries and availability | Prior-game box presence of key producers and production-weighted presence are used as proxies. | No current official injury/availability feed reaches production. An announced absence before the next game is invisible to that proxy. An absent box row can reflect game script or missing coverage. |
| Coaches | Head-coach tenure, first-year indicator, prior career games and winning percentage. | No structured offensive/defensive coordinator, play-caller, position-coach, scheme or dated staff-change history. Annual multiple-coach entries choose the more experienced coach, which cannot represent a midseason succession accurately. |
| Offensive line and off-ball play | Roster/scouting information, limited individual box observations, and team-level results. | No individual blocking history or full snap participation. Tackles, receptions and carries do not describe every player's contribution. |

The network uses structured inputs. It does not independently read current news, team biographies or press conferences when generating a forecast.

Implementation: [individual inputs](../src/cfbpred/player_inputs.py), [sequence inputs](../src/cfbpred/player_sequence_inputs.py), [active architecture](../src/cfbpred/architecture_research_net.py), [prior-game presence](../src/cfbpred/game_feature_snapshots.py), [coaching features](../src/cfbpred/features_raw.py).

## Measured player coverage

The last completed player snapshot represents **15,792 roster-person rows across all 138 FBS teams**. That verifies coverage of the collected roster feed, not agreement with every official current roster or complete player histories.

| Coverage measure | Players | Share |
| --- | ---: | ---: |
| Usable recruiting/scouting | 9,606 | 60.8% |
| At least one observed prior-season college box record | 6,997 | 44.3% |
| Observed prior production at another program | 3,366 | 21.3% |
| Unknown position in the network's player input | 5,914 | 37.4% |

The prior-history count is not an estimate of missing veterans: freshmen, players without recorded statistics and thinly covered positions also have no history. Likewise, observed prior-program production is not an exhaustive transfer count.

The unknown-position finding is more specific. `player_inputs.py` deliberately takes position from the most recent linked, dated recruiting record instead of the current roster position, because the historical provider feed can overwrite older values. Players without a mapped recruiting position enter an unknown-position pool. In the snapshot's collected 2026 production, those players account for about **15.4% of passing yards, 27.9% of rushing yards, 30.1% of receiving yards and 28.4% of tackles**. This is not confined to unused roster depth. Their statistics still enter the network, but their position grouping is unavailable or may reflect an old recruiting role.

Simply substituting today's position into every historical season would create a timing problem. The repair needs dated position observations, effective dates and provenance, with explicit unknown values for periods without reliable evidence.

The snapshot also contains two FBS roster rows flagged with ambiguous membership. The successful production audit verifies roster presence, model training boundaries and chronology; its pass status is not a claim of zero ambiguous identities. These cases need review under the existing identity rules.

Source and reproducible aggregate counts: [player coverage audit](audits/2026-09-21/player_coverage.json). Arrays contain a reserved zero row; this audit maps roster indices to array indices with the required +1 offset. Prior history is measured from the box-game count channel, not from nonzero yardage.

## Highest-value next work

1. **Build a dated lineup and availability table.** Store canonical player, team, game, reported position, depth-chart rank, confirmed starter status, availability, source publication time, observation time and effective date. Start with quarterbacks and official team/conference reports. Keep missing reports unknown. Forecast expected usage with uncertainty; distinguish a starter's absence from the presence of many healthy backups. The SEC publishes [official availability reports](https://www.secsports.com/fbreports), providing a concrete public starting point, but not complete historical national coverage.

2. **Repair current roles and transfer timing.** Archive official roster positions and changes, collect portal events, and validate destination/eligibility against roster membership. CFBD documents a [transfer endpoint](https://api.collegefootballdata.com/api/players) with origin, destination, transfer date and eligibility fields; the project's ingestion currently does not call it. Review identity collisions and uneven recruiting coverage. Do not force unsafe name matches or overwrite historical roles with current ones.

3. **Represent coaching and unit continuity.** Add dated head-coach, coordinator and play-caller tenures and their earlier teams' raw outcomes/tendencies. Separate the experience of a returning staff from the first weeks of a new system. Add reliable starts, participation, line combinations and position-specific production where sources permit. Prioritize coverage for lower-division opponents, given the observed error slice. Each addition remains an experiment, not a promised gain.

4. **Test the value of the new information with matched chronological experiments.** Retain the current architecture as the control, use equal ensemble sizes, compare identical games and separately report early-season, transfer-QB, missing-starter, new-staff and lower-division slices. Measure margin and total MAE, probability scores, calibration and paired uncertainty. Historical absence reports need publication evidence; do not invent historical labels to fill coverage gaps. Once 2026 outcomes influence development, reserve subsequent weeks or a future season for confirmation.

Adding more inputs is not itself evidence of improvement. Earlier individual-player pooling was essentially level with its corrected control; contribution-weighted pooling and auxiliary player-stat prediction did not improve the later assessment. The successful TabM change produced a modest measured gain. These results favor improving the information about upcoming participation and then testing its use, rather than assuming another larger network or an unrestricted feature dump will close the gap.

## Evaluation and operational details

- Preserve immutable forecasts and market observations at the same cutoff: for example, 24 hours before kickoff and immediately before kickoff. The current `game_lines` table retains one preferred provider plus opening/current fields, without a line observation timestamp. Local forecast snapshots exist, but the database prediction row also lacks a per-game forecast timestamp. The displayed market should not automatically be described as a verified, timestamp-matched closing quote.
- Week 1 includes predictions reconstructed after games were completed during the September 7 deployment. The weights used no 2026 fitting labels and dated histories were cut before kickoff, but annual roster and recruiting information is not fully archived as published at the time. The published 2026 aggregate is consequently not a wholly prospective trading record. See the [deployment report](ARCHITECTURE_RESEARCH_2026-09-07.md).
- Individual recruiting pulls are separate from the routine refresh and reuse cached source files. The current recruiting parquet was last modified September 7; routine roster and coach refreshes do not automatically fetch revised individual recruiting records.
- Weather fields exist in raw data, but `train.py` excludes them following earlier experiments, and the collected weather file stops at 2025. Calling weather an active production input would be incorrect. Forecast-time weather would require a new dated collection and a new experiment.
- The project explicitly excludes market lines and external predictive ratings from model features. That preserves its independent-model objective. If that policy changes, a separately evaluated market-informed model or residual correction becomes an option, but using a market baseline does not establish an ability to beat the market. This audit preserves the existing policy.

The next implementation should make the model aware of the expected participants and their roles for a specific game, with verified timing. Its accuracy benefit must then be measured prospectively.
