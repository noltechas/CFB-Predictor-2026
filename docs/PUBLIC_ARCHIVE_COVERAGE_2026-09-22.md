# Public archive coverage audit — 2026-09-22

This audit reads the frozen registry at
`models/experiments/public_archives_20260922_v2/registry/` and the pinned
Nebraska schedule ledger. The machine-readable results are in
`data/raw/official_team_archives_20260921/frozen_registry_audit.json`.

The frozen registry has 1,558 reviewed records, including 1,293 availability
records. Those represent 1,287 unique player/team/game/status keys, 74 source
report URLs, 142 games, and 51 teams. The accepted subset has 838 availability
records, 833 unique keys, 18 source URLs, 78 games, and 48 teams. The reviewed
registry also contains 223 dated positions, 30 depth records, 10 membership
records, and 2 identity records. Accepted depth records total 25, as expected
from the manually supported Nebraska chart submission.

Nebraska’s pinned ledger contains 313 fixtures: 304 past and 9 future at the
audit cutoff. Accepted status/depth evidence covers 29 past games: 18 with
availability records and 11 with depth records. It covers zero future games;
275 past and all 9 future fixtures have no accepted public status/depth record.
The 11 depth fixtures are the 2019 game-note chart cohort. This is source
coverage, not a claim that other games had no injuries.

The reviewed registry has no invalid `valid_until` values in the timestamp
check, and no conflicting status values for the same player/team/game key.
Carry-forward facts can remain valid through their recorded `valid_until`; a
carry-forward row does not represent a newly published report for each game.

The Nebraska public API is a site-specific schedules/events/articles provider,
not a universal historical availability standard. Its exposed official pages
are strongest for archive enumeration and game-note retrieval; the 2026 pages
do not establish historical status coverage. Publication-before-kickoff timing
assumptions are explicitly tagged, and missing or unlisted players remain
unknown rather than healthy.
