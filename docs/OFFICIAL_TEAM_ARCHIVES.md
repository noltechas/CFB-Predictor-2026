# Official team archive discovery

`src/cfbpred/official_team_archives.py` is a small, resumable discovery and
cache layer for public official team archives. It inventories modeled FBS and
FCS teams, keeps an explicit `pending`, `supported`, `unsupported`, or
`blocked` state, records exact response bytes and SHA-256 hashes, and stops a
queue at the first HTTP 429. Pending URLs remain in the returned queue; the
collector does not rotate hosts or credentials.

Run the initial manifest and Nebraska provider probe with:

```text
PYTHONPATH=src python3 -m cfbpred.official_team_archives \
  --schedule models/player_sequence_production/refreshes/20260920T122506160000Z/inputs/snapshot_schedule.parquet \
  --output data/raw/official_team_archives_20260921 --nebraska
```

The run enumerated 280 modeled programs (142 FBS and 138 FCS). Nebraska is the
only provider marked supported after a live public request; 142 other FBS and
137 other FCS programs remain explicitly pending until their official domain
and archive route are verified. The Nebraska provider is site-specific:
Sidearm/Nuxt exposes `/website-api/sports`, `/schedules`,
`/schedule-events`, and article-by-slug content. This is not a universal
structured injury/status standard, and the collector does not assume that
another school supports these routes.

Archive documents can be tagged `timing_assumed` when the publisher's
publication timestamp is before kickoff but no revision timestamp is exposed.
That tag is provenance metadata only. Postgame recaps are never treated as
pregame evidence, and an absent or unlisted player never becomes a healthy or
available fact. Fact normalization remains the responsibility of the
availability backfill validator, which requires explicit status language.

Focused tests are in `tests/test_official_team_archives.py` and cover rate-limit
stop/resume behavior, response hashing, and explicit team states.

The CLI also accepts repeatable `--directory` and `--conference-url` options.
`--conference-url` caches a bounded public conference-page batch under
`conference_pages/`, writes immutable receipts, and leaves failed or rate
limited URLs pending. A recent batch attempted 11 public conference roots: 2
returned HTTP 200, 8 returned HTTP 404, and 1 failed at the network layer.
Those pages are retained as discovery receipts; because they did not expose a
machine-readable member directory in the cached response, no additional team
domain was marked supported. The MAC directory links remain the only
conference-directory candidates currently merged into the team manifest.

The manifest classification uses the latest season row for each team when the
schedule has repeated teams, avoiding arbitrary `drop_duplicates` ordering.

## Directory-domain discovery batch

A bounded offline extraction from the cached MAC conference directory produced
21 distinct official athletics links, each retaining its source HTML path in
`directory_candidates.json`. One GET probe per link was recorded in
`domain_probe_receipts.json`: all 21 returned reachable responses. Reachability
leaves the teams `pending` because a homepage does not prove a published
football schedule or availability route. The manifest therefore records those
links with provenance while leaving provider support unclaimed. This is the
intended next stage for conference directories that expose equivalent official
school links.
