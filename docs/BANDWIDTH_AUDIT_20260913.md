# Bandwidth audit — September 13, 2026

The original outage was caused by Supabase's cached and uncached egress restrictions. Service is restored on the replacement standard-PostgreSQL Pro project; see `ORIOLEDB_MIGRATION_PLAN_20260913.md` for the completed migration and verification. There is substantial avoidable traffic in the backend, independent of someone checking scores on a phone. The backend, nightly refresh, and mobile changes below are implemented; the cloud function and production phone update are deployed. This audit does not establish that the remaining workload will fit the Free plan.

## Confirmed measurements

Read-only database statistics start August 25, 2026. Counts below include scheduled and manual runs; the database cannot identify which local invocation caused each read. Table replacements leave multiple statistics entries, which were summed by query text.

| Full-table download | Calls | Rows returned |
| --- | ---: | ---: |
| Historical plays (`play_raw`) | 36 | 79,857,140 |
| Opponent history (`game_history`) | 41 | 27,288,974 |
| Game features (`game_scalars`) | 41 | 1,014,494 |

The nightly script publishes only `game_scalars` and `game_history`, then previously downloaded all three tables. Its unchanged historical-play download contains about 2.218 million rows. A bounded sample of 1,000 current rows estimates approximately 177 bytes per row including PostgreSQL row/field framing: roughly 392 MB per full play download, or 11.8 GB over 30 daily runs. The 36 recorded historical-play reads are equivalent to approximately 14.1 GB at that sampled row size.

**These are traffic estimates, not Supabase billing measurements.** The sample is the first 1,000 rows, historical schemas can differ, and JSON text conversion used to estimate field lengths can differ from PostgreSQL text output. Compressed Parquet sizes on disk are not the amount transferred by the SQL queries.

The cloud `live-scorer` cron job is active every minute. There were 10,080 successful cron dispatches during the observed seven-day window. Dispatch success is not proof the function succeeded: recent HTTP responses include 500, 503, and 546 failures as well as 200 responses.

Its model bucket contains two files totaling 4,511,187 bytes, with `no-cache` metadata. The previous function loaded those files before checking whether ESPN reports any games or the Mac scorer already owns the work. The in-memory model cache avoids repeated downloads only while that worker survives. The actual number of cold starts and model downloads is unavailable in this audit; multiplying every invocation by 4.5 MB would be an upper-bound scenario, not observed usage.

The same function reads pregame model arrays for every scoreboard game, including already-completed games, before checking whether their curves are complete. The database recorded 5,145 matching REST query calls. Current payloads average about 30,991 bytes per game; a 100-game response would therefore be approximately 3.1 MB. The number of games in each historical response is unknown, so no cumulative byte total is claimed.

## Phone app

The team-record request previously read all 3,681 season fixtures on each full slate refresh. SQL JSON serialization measured 768,510 bytes. Only 760 started or scored games were needed, totaling 156,435 bytes: approximately 80% less data. These measurements exclude HTTP compression and transport overhead.

The previous Scores tab remained mounted and polling even when another tab or a game-detail screen covers it. Full slate refreshes occur approximately every five minutes, with smaller polls every 20 seconds during live play or 90 seconds otherwise. There is also a timer race where an in-flight poll can rearm after backgrounding/cleanup. The deployed update now stops hidden/background polling and prevents the timer race.

Completed-game sparklines already select only play index and win probability, with a one-hour in-memory cache. Previously they were fetched for the entire slate, including games outside the visible viewport. The deployed update requests visible cards only. Game details retain the full curve but fetch only the latest probability distribution, rather than one distribution per play.

The logo bucket contains 1,374 objects totaling 82.3 MB. The source assets are 500-pixel images even when displayed as small badges. The mobile image component already enables memory/disk caching. Bucket size is **not** downloaded traffic; it does not explain the cached-egress restriction by itself.

## Dashboard usage and billing

The authenticated dashboard reports a September 5–October 5 billing cycle: 41.503 GB cached egress against 5 GB, 9.568 GB uncached egress against 5 GB, and 0.517 GB database usage against 0.5 GB. Storage is 0.086 GB and Edge Function invocations are 12,224. These organization totals establish the restriction but do not attribute cached traffic to individual files.

The user approved Supabase Pro at $25/month plus applicable tax. Checkout shows $25 due today and a $25 monthly estimate. The user corrected the saved-card/postal-code mismatch, but checkout was then rejected because CFB itself uses the OrioleDB preview engine, which is limited to Free organizations. A verified-readable database backup and a standard-PostgreSQL migration plan are prepared in `docs/ORIOLEDB_MIGRATION_PLAN_20260913.md`. The original organization remains Free and restricted. The replacement CFB Production organization is now Pro with Spend Cap enabled; its public score API returns HTTP 200.

## Changes made and deployed

1. The feature publisher creates local caches from the successfully committed publication payloads. It preserves PostgreSQL REAL rounding, writes caches atomically only after commit, and removes the nightly download of the same feature tables. The unchanged historical-play cache is retained; a missing cache still downloads on demand. Explicit manual full refresh remains available.
2. Cloud model weights are bundled unchanged with the function and loaded lazily, after empty-work and Mac-ownership checks. Completed games are checked using small status/score rows before model arrays are fetched. Unchanged finished curves are skipped; missing, incomplete, live, and corrected-score work remains eligible. JWT verification stays enabled.
3. Scores and game details poll only in the foreground. Scores polling stops when another tab or detail screen covers it, and an in-flight request cannot restart a disposed/background timer. Visible cards alone load sparkline history. Details fetch the latest score distribution separately from the full lightweight curve.
4. Team-record requests filter started or scored games on the server, retaining undated finals and reducing the measured record payload by approximately 80%.
5. The default week uses a 48-hour lookback so Sunday's launch stays on Saturday's results. Required prediction-request failures surface as errors rather than silently returning an empty slate.

The original live scorer was deployed as version 13; the replacement project now runs active version 2 with identical embedded weights, a 16-final-game work limit, and a 120-snapshot inference budget. The Expo production update group is `79db0a43-a4f2-4a11-874c-3400b940637d`, runtime `exposdk:57.0.0`. Public production manifests return the expected update IDs for both iOS and Android. Installed apps download this update on launch and apply it on a subsequent restart.

## Verification

- Python tests passed for selective refresh, committed-cache publication, refresh entrypoint, and unchanged model delivery hashes.
- All 91 mobile tests passed; TypeScript checking passed.
- Three scorer-work tests and a bundled-model handler test passed. The latter forbids network requests and exercises the real model-check route.
- Deno type checking and shell syntax checking passed. Numerical parity passed for 803 snapshots; maximum win-probability difference was about 2.24e-7.
- An aggregate production SQL check confirmed the reduced record selection omits **zero recorded finals** and **zero started games awaiting finals**. All 86 Week 2 games involving an FBS team have stored final scores.
- Both production Expo manifests serve the published update, recorded in `logs/score_repair/update-delivery.json`.
- The deployed model check, real scorer writes, and public score API now return HTTP 200. The real mobile data layer and visible mobile web preview show Week 2 finals. Automatic one-minute scorer runs succeed.

## Operational follow-up

Migration, Pro activation, deployed-model checks, Week 2 API requests, production phone-update delivery, and scheduled scorer verification are complete. Storage request attribution can refine long-term measurements, but is not required to restore scores. No claim is made that the remaining workload fits Free limits.

Reducing usage cannot retroactively erase the current quota overage. Supabase states that egress restrictions clear at the next billing cycle or with the applicable plan/spend-cap change: https://supabase.com/docs/guides/platform/manage-your-usage/egress

Evidence: `logs/egress_audit/database.json`, `estimates.json`, `summary.json`, and `verification.json`. Pre-edit source copies are under `logs/egress_audit/before/`.
