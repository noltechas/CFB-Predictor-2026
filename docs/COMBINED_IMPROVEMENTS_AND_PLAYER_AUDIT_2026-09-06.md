# Combined improvements and individual-player audit

Small, repeatable improvements are worth retaining. The earlier minimum-gain
cutoffs were experiment-screening choices, not evidence that several weak
changes could never help together. Their implementations and measured results
were retained. This follow-up explicitly tests their interactions and records
the combined candidate for further work.

**The compatible bundle shows a small pooled gain at 16 seeds, but uncertainty
and season-to-season variation remain. Production has not been replaced.**

## Combined experiment

The bundle uses all 270 active scalar inputs, chronological checkpoint selection
and refitting, team-order consistency, five-game raw form/matchup features and
preseason roster passing priors: **324 scalar columns** in total. A second
fixed candidate adds the previously tested CRPS margin loss and margin-aware
checkpoint score. Both were measured on the development and later-year periods.

Alternative three-game windows, the shared encoder, mixture heads and the
eight-season training window are not cumulative positive changes: they either
replace another setting or previously performed worse. Earlier feature-join,
forecast-locking and scaler fixes already remain in production.

On 1,974 development FBS-vs-FBS games in 2020–2022, three-seed margin MAE was:

| Configuration | MAE |
| --- | ---: |
| Chronological control | 13.1991 |
| Combined features + consistency | **12.9399** |
| Combined + experimental margin loss | 12.9526 |

On all 2,398 2023–2025 FBS-vs-FBS assessment games at three seeds:

| Configuration | Margin MAE | Total MAE | Winner log loss |
| --- | ---: | ---: | ---: |
| Control | 12.6938 | 13.0426 | 0.55450 |
| Combined | **12.6679** | **12.9402** | **0.55258** |
| Combined + margin loss | 12.6782 | 13.1973 | 0.55293 |

The margin-loss addition worsened total forecasts. The original-loss bundle
improved the pooled margin, total, log-loss and Brier metrics. It initially
missed an automatic two-of-three-season advancement rule, but the user's
request to investigate small cumulative gains justified an explicitly recorded
diagnostic extension. The original plan/report remain unchanged, and the
extension has its own `confirmation_plan.json`.

The **16-seed comparison**, on the same 2,398 games:

| Metric | Control | Combined | Change |
| --- | ---: | ---: | ---: |
| Margin MAE | 12.7127 | **12.6746** | −0.0381 |
| Margin RMSE | 16.0350 | **16.0167** | −0.0183 |
| Total MAE | 13.0273 | **12.9128** | −0.1145 |
| Total RMSE | 16.2138 | **16.1042** | −0.1096 |
| Winner log loss | 0.55353 | **0.55247** | −0.00105 |
| Brier score | 0.18853 | **0.18744** | −0.00109 |
| 10-bin calibration error | **0.02661** | 0.02914 | +0.00253 |

Lower is better. The paired season/week-cluster bootstrap 95% interval for
the margin-MAE change is **[−0.1327, +0.0643]**. It includes zero. The larger
ensemble preserves the small pooled gain, but does not prove a future gain.

| Assessment season | Control margin MAE | Combined margin MAE |
| --- | ---: | ---: |
| 2023 | 12.6785 | **12.5267** |
| 2024 | **12.8269** | 12.8291 |
| 2025 | **12.6335** | 12.6671 |

On the 2,396 games with archived market lines, the 16-seed control and combined
MAEs are 12.7205 and 12.6818; the market MAE is 12.0296. The candidate closes a
small portion of that historical gap. These are equal-size chronological
experimental ensembles, not a remeasurement of the deployed hashed-holdout
ensemble.

`models/experiments/combined_20260906/retained_candidate.json` explicitly retains
the bundle. It is a historical-evaluation candidate, not a 2026 model refit or
an active published forecast. There is no minimum required gain in points in
that decision. New work should preserve useful combinations, quantify their
uncertainty and compare secondary outputs; adding reported standalone gains
arithmetically is invalid because changes can overlap, interact or overfit.

All evaluation inputs and code snapshots are frozen. For each forecast year Y,
fit before Y−1, select epochs on Y−1, refit before Y, forecast Y. No 2026
outcome fits or selects these models. The 2023–2025 seasons have been examined
repeatedly and are historical benchmarks, not a new untouched test. Bootstrap
intervals do not account for every training or research-selection uncertainty.

## What production actually knows about players

**Production does not encode every player with an individual recruiting rating
and full playing history.** It aggregates much of that information first.

| Information | Current representation | Important limit |
| --- | --- | --- |
| Recruiting | Team class-point averages, latest class points and best class rank | Original program/class summaries do not explicitly follow each recruit to a new roster |
| Roster talent | One team talent-composite number | Cannot identify which positions or players supply the talent |
| Roster composition | Position-group counts, average size, class-year and continuity summaries | No per-player network input or learned player contribution |
| Returning production | Athlete-ID joins for returning pass/rush/receiving yards and tackles | Team-level summaries of last season's output |
| Incoming production | Imported pass/rush/receiving yards from other programs | No equivalent full defensive-transfer or OL performance representation |
| Career experience | Selected current leaders and active-player/top-defender summaries | Requires observed current-season participation; empty for many opener fields |
| Recent game history | Fourteen prior results per **team**, plus opponent identities | This is not a player's individual history |
| Availability | Prior-game participation proxies | Not a confirmed upcoming lineup or snap projection |

Team talent may reflect roster changes in aggregate, and existing imported
production does carry some transfer signal. Neither establishes correct
individual treatment of every transfer. Roster headcount continuity also uses
names, whereas production/career joins use athlete IDs; these are different
identity mechanisms.

## Individual data now staged

The public CFBD endpoint supplies individual scouting records with recruit and
athlete identifiers, class year, stars, rating and rank. See the provider's
[recruiting API](https://github.com/CFBD/cfbd-python/blob/main/docs/RecruitingApi.md)
and [recruit schema](https://github.com/CFBD/cfbd-python/blob/main/docs/Recruit.md).

`pull_player_recruiting.py` retrieved **90,369 records**, covering the
2000–2026 HighSchool, JUCO and PrepSchool requests. Exact raw response hashes
and retrieval times are recorded. No game-model outputs are imported.

`player_catalog.py` now links season rosters to individual recruiting records
and prior box-score game records across college programs. It uses direct
athlete IDs and explicit roster recruit-ID links; it does not match players
by approximate name or pick whichever rating is highest. Multiple records of
the same recruiting type remain ambiguous. A roster season cannot use a later
recruiting class or target-season box performance.

The staged catalog contains **372,785 canonical roster-season-player rows**.
For the 2026 FBS population:

| Coverage | Count |
| --- | ---: |
| Teams represented | 138 |
| Rostered player rows | 15,773 |
| Unambiguous linked scouting records | 9,457 |
| Prior college box observations | 6,998 |
| Observed at another college program | 3,371 |
| Same-ID continuity requiring review | 18 |
| Linked scouting records passing the identity screen | 9,448 |
| Prior box histories passing the identity screen | 6,991 |
| Other-program history + scouting passing the screen | 2,133 |

Passing this screen is not independent verification of every identity. Across
all divisions in the current catalog, 31,866 roster rows cover 310 teams and
89 rows require identity review.

About 60% of FBS roster rows have scouting information passing this conservative
screen. Missing scouting includes unmatched, unreported and ambiguous records;
it must not automatically become a zero-star or low-ability player. Missing box
history is also not proof of inexperience: only 206 of 2,608 rostered FBS OL
have prior box rows in these files. Ordinary box scores do not provide every
lineman's snaps, blocks or pressure responsibility.

## Confirmed identity problem and quarantine

The cached roster assigns athlete ID `4030388` to both Coastal Carolina's
2016–2018 Preston Carey and Georgia's 2026 Preston Carey. The same cached ID
has 2017 Coastal Carolina box observations. Georgia identifies the current
player as a **freshman from IMG Academy**, which contradicts treating those old
college observations as his experience. See the
[official Georgia player page](https://georgiadogs.com/sports/football/roster/preston-carey/10710).

This shows why a provider ID match alone cannot guarantee correct career
history. The new catalog flags roster gaps of four or more years, college
history spans of nine or more years, and simultaneous multiple rosters.
Suspicious scouting/history connections receive explicit review flags and are
excluded from the catalog's usable-without-review indicators. Raw evidence is
retained in `identity_review_required.parquet`; no replacement identity is
invented. Long gaps can also reflect source coverage or unusual eligibility,
so a flag is not a declaration that every such player is wrong.

The quarantine currently applies to the **new staged catalog**. It has not
retroactively rewritten the production feature store or corrected all its
existing career joins. That is an important remaining implementation task.
Historical recruiting revisions and season-roster publication times are also
not archived by these API records, so class-year filtering alone does not
establish exact pre-kickoff availability.

## Next implementation and artifacts

The next player-model change should resolve and version player identities,
then pass each rostered player's position, scouting, strictly prior production
and explicit coverage flags into a shared player encoder. Position-aware
pooling can learn team contributions while preserving all existing game
features. It must distinguish roster membership from expected participation
and support players with no observed college box stats. This architecture has
**not** been trained or integrated into production by this audit.

Artifacts:

- `models/experiments/combined_20260906/`: both three-seed bundles, 16-seed
  control/bundle comparison, checkpoints, per-game PMFs, plans, code snapshots
  and retained-candidate decision.
- `data/raw/recruiting_players/`: response snapshots and manifest.
- `data/parquet/cfbd/recruiting_players.parquet`: individual scouting records.
- `data/parquet/player_catalog/`: player seasons, exact recruit links, prior
  game-history index, identity-review rows and per-position coverage.
- `logs/test_combined_and_players.log`: **85 tests passed**. New tests include
  advancement of tiny repeated gains, probability regressions, transfers,
  conflicting IDs, future-record exclusion, ambiguous recruiting matches and
  long-gap identity quarantine. Python compilation passed.

The original active feature hash remains unchanged:
`b6764b0a5e8454447896371a747bbb882219695d283cd9bd46996cbec3ca5616`.
No database writes or forecast publication occurred in this follow-up.
