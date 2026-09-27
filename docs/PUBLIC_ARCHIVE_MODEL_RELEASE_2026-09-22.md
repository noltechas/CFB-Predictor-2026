# Public archive model release — 2026-09-22

Published `player-full-context-tabm16-v4` at 2026-09-22 03:29:29 UTC. The model was refitted through 2025 with the fixed 14-epoch budget and no 2026 fitting outcomes.

The historical Slate and Results views are retrospective reconstructions made
with the latest fitted model. They are not as-issued forecasts from each past
game date. The availability archive is an evidence-backed partial registry; it
does not claim that every historical game has complete availability coverage.

## Registry coverage

Source: `models/experiments/public_archives_20260922_v3/registry/reviewed.json`.

- 1,594 total facts; 1,329 availability facts, 223 position facts, 30 depth facts, 10 membership facts, and 2 identity facts.
- 1,571 facts have a game ID, covering 156 distinct games.
- 52 distinct team values.
- 90 distinct source URLs, 57 source paths, and 57 source SHA-256 values.
- Status/value counts across the complete registry: `out` 1,097; `questionable` 140; `available` 70; `probable` 17; `doubtful` 5. Other non-status values are roster position or membership labels (`WR` 51, `DB` 30, `RB` 27, `DL` 26, `1` 21, `OL` 17, `QB` 15, `TE` 13, `DT` 13, `LB` 11, `transfer_arrived` 9, `2` 8, `CB` 7, `DE` 6, `OT` 6, `quarantine_history` 2, `ST` 1, `departed` 1, `3` 1).

For Nebraska, filtering to `team=Nebraska` and `kind=availability` yields 98
facts across 34 distinct actual games, 40 distinct player names, and 23
source URLs. The original modeled schedule contained 313 Nebraska games; the source audit
recovered 14 omitted bowls, raising the available 2001–2026 schedule to 327
games. Thus only 34 of 327 Nebraska games have reviewed availability facts. These 34 games have reviewed source facts, not necessarily a fully
resolved player identity or complete team report for each game. The statuses are `out` 83, `questionable` 11,
`available` 2, and `doubtful` 2. The broader Nebraska team filter (including
depth facts) is 123 facts across 45 games; it should not be presented as
availability-game coverage.

Known gaps include the large set of Nebraska games without a reviewed
availability source, uneven season coverage, and sources that provide a player
status without a complete team report. These gaps must remain visible in any
release UI.

## Published UI disclosure

Historical game cards should carry **“Historical reconstruction · latest
model”** with a tooltip explaining that the value was recomputed with the
latest fitted model and is not an as-issued forecast. Results should say that
historical values are retrospective reconstructions and that the availability
archive is partial. No release claim should imply complete historical
availability for every game.

The mobile production update with these disclosures was published and its iOS/Android manifests verified. See `MOBILE_OTA_HISTORICAL_DISCLOSURES_2026-09-22.md`.

## Assessment and publication

The unchanged model architecture and training settings were compared on the same reused 2025 benchmark. On 807 games with market lines, margin MAE improved from 12.3287063831 to 12.3270852333. Vegas MAE was 11.8339529120. Winner log loss increased slightly; the selected primary criterion was any decrease in market-matched margin MAE. The paired interval crosses zero, so this is a small observed improvement, not conclusive evidence of a durable gain.

The final training registry includes the official MAC 2025 PDF records. The earlier candidate without those records scored 12.3251667488; the final complete accepted registry was retained rather than dropping valid records to optimize the reused benchmark.

The initial 28,482-game v4 reconstruction was published and read back, but is
superseded by the expanded-input current release below. The 26,873 predictions
through 2025 use the existing `train` split so they cannot inflate held-out
accuracy. The 1,609 current-season games retain `test` because their outcomes
were excluded from fitting, while played games are disclosed as retrospective
reconstructions. Probability arrays contain 115 margin and 103 total bins.
Historical reconstructions are not a walk-forward backtest.

Current-season preseason and Weeks 1–3 rankings were published with 266 entries per week. All-year weekly ranking reconstruction is in progress; its final receipt will be recorded below.

## Public feeds and scheduled refresh

The HD Intelligence public API provides individual player status rows for SEC, Big Ten, ACC, Big 12, MAC, and American reports, including FCS opponents when those reports list them. The exposed public archives currently contain 2026 records; this is not a universal historical injury API. The initial five-conference pull archived 794 rows; exact game/team/status validation produced 782 candidate rows and identity review accepted 774. Archived source bytes, raw statuses and hashes are retained. Unsupported partial-game statuses, mismatched fixtures and unresolved identities remain quarantined.

When a historical official report lacks a verifiable publication time, the user-authorized pregame assumption is explicit in each fact; it is never represented as an observed timestamp. Missing reports remain unknown, not healthy. Manually reviewed older articles and PDFs supplement the current API.

The local prediction pipeline is installed for daily 04:30 America/Chicago refresh and a one-minute dispatcher that runs 30 minutes before each scheduled kickoff slot. The dispatcher tracks kickoff changes, avoids duplicate runs and retries failed runs before kickoff. The machine must be awake. Both routes refresh public availability; the score collection job also archives public feeds. Stale Division I schedule/results trigger a complete input rebuild before inference. No daily run automatically refits weights.

A v4 no-publication refresh successfully collected all six feeds, rebuilt dated context and passed the future-game audit: 1,142 games, 1,232 FBS sides, no empty FBS rosters, and strictly pregame player history. Completed managed refresh arrays have bounded retention; prediction outputs and source evidence are retained.

The exact American provider alias was subsequently confirmed from its official embedded report: `American`. Its initial archive adds 29 resolved player/team status facts, all for 2026. They affect inference only; the model fitting cutoff remains 2025. CFP’s separate confirmed public archive endpoint currently returns an empty list, which is retained as an empty receipt and produces no availability assumptions. Other conference reporting providers require separate adapters and are not represented as covered by this API.

## Six-conference inference refresh

The expanded public-feed refresh was published at 2026-09-22 04:17:48 UTC.
All 1,609 2026 predictions and preseason/Weeks 1–3 rankings passed database
readback. The resolved timeline contains 1,351 distinct player-game availability
entries across 147 games and 55 teams. Repeated source captures do not count
as additional player-game coverage. The refresh retains v4 weights.
Receipt: `models/player_sequence_production/availability_complete_v4/20260922T035124704336Z/full_season/publication.json`.

This remains a partial availability archive: 1,351 unique player-game-status
entries cover 147 games and 55 teams, while Nebraska coverage is 34 of 327
games. Missing reports remain unknown and are never treated as healthy.

The later no-publication availability refresh at
`models/player_sequence_production/availability_refreshes/20260922T054631275892Z/availability_refresh.json`
passed its future audit for 1,142 games and 1,232 FBS sides, with no missing
FBS rosters and strictly pregame player history. It was not published. A shell
wrapper status-reporting issue is being fixed separately.

## Historical source reconciliation

A subsequent audit found older public CFBD records cached locally but omitted
from the canonical feature schedule. The reviewed v2 reconciliation restores
7,981 completed Division I fixtures, including 14 Nebraska bowls. Existing
fixture facts are preserved. The expanded source contains 48,566 schedule
rows, including the 7,981 newly restored fixtures. Fixed v4 weights with
expanded inputs improved matched 2025 margin MAE to 12.325721863 on 807 games,
with paired 95% CI [-0.0037857864, +0.0009223801]. This is an input-only
diagnostic, not a refit or publication. The full-history refit failed the
primary margin gate (12.376570102 versus retained v4 12.327085233), so the
latest-model reconstruction mode remains selected.

The current expanded-input release published the 1,609 current-season games
with v4 weights at 2026-09-22 05:45:51 UTC. Its receipt is
`models/player_sequence_production/complete_history_v4/20260922/full_season/publication.json`;
all current-season predictions and PMFs passed exact readback, and the
input-only assessment recorded margin MAE 12.325721865 on 807 matched 2025
games. Preseason and Weeks 1–3 rankings (266 teams per week) also passed
exact readback. The 35,501-game historical receipt at
`models/historical_reconstruction/predictions_complete_v4_20260922/publication.json`
is undergoing a real-only correctness cross-check because virtual rows can
affect rare lower-opponent scalars. The corrected real-only historical export
is still in progress, so historical final publication remains pending.
Approximately 450 of 533 weekly snapshots are in progress; the weekly replay
and its final receipt are also pending. Raw source coverage still does not
imply availability coverage.
