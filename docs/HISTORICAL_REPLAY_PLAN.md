# Historical reconstruction

The selected mode is `latest_model_reconstruction`: one latest published model
reconstructs every archived season. Its weights were fitted through 2025.
Historical pages therefore describe reconstructions, not as-issued forecasts
or a walk-forward accuracy test. No 2026 outcomes enter model fitting.

## Complete source history

`complete_history_schedule.py` reconciles the original canonical schedule
with cached public CFBD records, preserving existing fixture facts and using
numeric program IDs plus historical aliases. The reviewed v2 artifact is
`models/experiments/complete_history_20260922_v2/source_schedule.parquet`.
It restores 7,981 completed Division I fixtures, including 14 Nebraska bowls.
Eight equivalent 2004 duplicate pairs are resolved using the records with real
kickoff metadata; the already-pinned LSU/App State fixture keeps its existing
ID. The immutable audit records source hashes and each duplicate decision.

`complete_history_inputs.py` builds isolated scalar, player, sequence,
full-career and dated-context stores. A fixed 2025 assessment uses the same
807 market-matched games, model architecture, seed and 14-epoch budget as the
published benchmark. The completed source contains 48,566 schedule rows and
restores 7,981 completed Division I fixtures, including 14 Nebraska bowls. The
full-history refit failed the primary margin gate: MAE 12.376570102 versus
12.327085233 for retained v4. The selected path is therefore
`latest_model_reconstruction` using fixed v4 weights.

An input-only diagnostic using those fixed v4 weights and the expanded context
improved margin MAE to 12.325721863 on the same 807 matched 2025 games. Its
paired 95% interval is [-0.0037857864, +0.0009223801], so it is diagnostic
evidence rather than a new fitted candidate or a publication decision. The
latest-model reconstruction currently covers 37,110 historical and current
Division I games. The 533 weekly snapshots remain pending; no claim of
complete publication should be made until their audit and receipt are present.

Availability remains partial: Nebraska has reviewed facts for 34 of 327 games;
the broader resolved timeline contains 1,351 unique player-game-status entries
across 147 games and 55 teams. Missing availability evidence remains unknown,
not healthy.

## Weekly ranking reconstruction

`historical_ranking_inputs.py` adds unplayed negative-ID terminal fixtures so
each team can have a snapshot after its last real game. Virtual fixtures have
no outcome labels and are never published as games. All games involving an
FBS or FCS team must have a scalar target, including historical FCS games
against lower-division opponents. Otherwise a weekly record can include a
result absent from the selectable feature snapshots.

`historical_rankings.py` calls the current player model at a strict global
cutoff for every weekly snapshot. Preseason ends before the first Division I
kickoff. Week 1 ends at Tuesday 12:00 UTC before the first FBS Week 2 game;
subsequent cutoffs are seven days apart through the final completed game,
including bowls and the 2020 spring season. Sequential calendar week labels
avoid CFBD postseason rounds that overlap or reset regular-season numbering.
Completed results, player histories and dated facts are restricted to the
snapshot cutoff. Empty early rosters and short histories remain explicit.

The replay is restartable and partitionable by season. Every document is
recorded with its source, checkpoint, calendar and output hashes. The
historical publisher verifies the complete manifest and database readback,
backs up previous historical rows, and preserves current-season rankings.
Partial earlier attempts under `rankings_partial_v1_20260922` and
`rankings_provider_weeks_partial_20260922` were never published.

## Publication receipts

The initial v4 reconstruction published all 28,482 games in its original model
cohort; its receipt is `models/historical_reconstruction/predictions_v4_20260922/publication.json`.
The expanded-history assessment and final all-year weekly ranking publication
are tracked in `PUBLIC_ARCHIVE_MODEL_RELEASE_2026-09-22.md`.
