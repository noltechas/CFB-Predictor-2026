# Scheduled jobs on GitHub Actions

Since 2026-11-17 the three scheduled CFB jobs run on GitHub-hosted runners, not
the Mac. The live in-game scorer was already a Supabase Edge Function.

| Workflow | Schedule | Replaces |
|---|---|---|
| `scores` (`.github/workflows/cfb-scores.yml`) | :07 and :37 every hour | launchd `com.cfbpred.scores` |
| `nightly` (`cfb-nightly.yml`) | 10:30 UTC (04:30 Central) | launchd `com.cfbpred.refresh` |
| `pregame` (`cfb-pregame.yml`) | check every 10 min; refresh 120 min before each kickoff slot | launchd `com.cfbpred.availability-dispatcher` |
| `state check` (`cfb-state-check.yml`) | manual | read-only rehearsal |

Supabase starts the scheduled runs (next section), not GitHub's `schedule:`
trigger. A scheduled run only does work while the repository variable
`CFB_ACTIONS_ENABLED` is `true`; a run started by hand always works. All jobs
share the `cfb-pipeline` concurrency group, which replaces `/tmp/cfbpred_publish.lock`.

## Who starts the scheduled runs: Supabase pg_cron

GitHub's `schedule:` trigger is best effort. On 2026-11-18 it ran the 30-minute
scores job 3 times in 18 hours, the 10-minute pregame check 3 times, started the
nightly 8.5 hours late, and skipped the basketball morning refresh. So since
that day the workflows have no `schedule:` block, and pg_cron in the CFB
Production Supabase project starts them through the REST API
(`workflow_dispatch`) at the same UTC times. Everything is in `ci/scheduler.sql`;
re-applying it is safe and restores the jobs.

| pg_cron job | When (UTC) | Starts |
|---|---|---|
| `gh-cfb-scores` | :07 and :37 | `cfb-scores.yml` |
| `gh-cfb-pregame` | every 10 min | `cfb-pregame.yml` |
| `gh-cfb-nightly` | 10:30 | `cfb-nightly.yml` |
| `gh-cbb-nightly` | 10:15 and 18:15 | `nightly.yml` in CBB-Predictor-2026 |
| `gh-dispatch-log-trim` | 04:00 | keeps 30 days of `ops.dispatch_log` |

`ops.dispatch(repo, workflow, inputs, ref)` sends the request and logs GitHub's
answer in `ops.dispatch_log` (204 means accepted). Dispatches carry
`inputs.scheduled = "true"`, which is how the football workflows tell a
scheduler run (obeys `CFB_ACTIONS_ENABLED`) from one started by hand (always runs).

**The one-deep queue.** GitHub keeps only one waiting run per concurrency group,
and a newly queued run cancels the one already waiting. A pregame refresh holds
`cfb-pipeline` for about an hour and a half, so without care each 10-minute check
would queue another refresh and a scores run could knock out a waiting nightly
(it happened to the first scheduled scores run, 22:07 UTC on 2026-11-18). So a
scheduled scores run and a pregame refresh first run `ci/pipeline_busy.sh`
outside the queue and stand down while a nightly, a retrain or a pregame refresh
is queued or running; that heavier job republishes everything they would have.
Runs started by hand skip the check.

```sql
-- the last day of dispatches; anything but 204 is a problem
select at, workflow, status, detail from ops.dispatch_log
 where at > now() - interval '1 day' and status is distinct from 204 order by at desc;
```

**The token.** A fine-grained GitHub token (repositories CFB-Predictor-2026 and
CBB-Predictor-2026 only, Actions read and write) in Supabase Vault as
`github_actions_token`, created 2026-11-18 with a one-year expiry. Renew it
before it lapses: make a new token the same way, then run
`cd ~/IdeaProjects/CBB-Predictor-2026 && PYTHONPATH=src python3 scripts/store_github_token.py`
(input hidden; it replaces the stored one). An expired or revoked token shows
as 401 in `ops.dispatch_log`, and no scheduled run starts until it is replaced.

**Pausing.** Football only, keeping the jobs: `gh variable set CFB_ACTIONS_ENABLED
--body false -R noltechas/CFB-Predictor-2026` (runs still start, then skip).
Everything, both sports:

```sql
select cron.unschedule(jobname) from cron.job where jobname like 'gh-%';
```

Re-apply `ci/scheduler.sql` to resume.

## How a runner gets the Mac's disk

The pipeline was written for one persistent machine, so each job restores state,
runs the unchanged shell script, and saves state back:

- `cfbpred.ci_state` mirrors named sets of files (`ci/state_sets.json`) to the
  private Supabase Storage bucket `cfb-state`, content-addressed in 45 MB chunks.
  Hard links (the conference archive is 1.6 GB stored, 21 GB expanded) and
  absolute symlinks are preserved. Restore is an exact mirror, fails if a
  `required` file is missing, and refuses to run outside CI.
  - `static` — inputs that do not change in season (pbp corpus, reviewed
    availability sources, active v10 bundle and its stores).
  - `core` — CFBD tables, published predictions/PMFs, the availability, plays
    and player-context archives, `logs/api_calls.csv`, dispatcher state.
  - `daily` — the run named by `refreshes/latest.json` and
    `availability_refreshes/latest.json` plus every run those point into
    (timelines included: `TimelineStore` reads them).
- The Actions cache (`cfb-core-*`, `cfb-full-*`) keeps a warm copy so a run
  downloads only what changed. A cold restore takes ~10-15 minutes.
- The runner's working copy lives at `/Users/chasnolte/IdeaProjects/CFB-Predictor-2026`
  because stored pointers, manifests and symlinks name that absolute path.
- Rollback promotions are uploaded to `cfb-state/archives/promotions/` and not
  restored. The large context reports (`report_source_issues.json`,
  `reviewed_and_current.json`, `reviewed_input.json`, `context/audit`) are not
  kept; nothing reads them back.
- `TZ=America/Chicago` keeps `logs/api_calls.csv` stamps consistent with the
  Mac's (the CFBD monthly budget counts them).

Never run a job against missing state: `update_season.py` and
`pull_player_stats.py` would write 2026-only tables and `publish_lines.py`
would replace `public.game_lines` with them. A failed restore stops the job.

## Retraining the pregame model (`retrain`, manual)

Training runs on a runner too; nothing needs the laptop. Actions -> **retrain** -> Run workflow:

| mode | what happens |
|---|---|
| `stage` | builds a store chain from the latest nightly snapshot, fits the active variant for five outer seeds (fixed 14 epochs, the shipped recipe), scores and audits the season, grades the new ensemble against the live one, rebuilds this season's rankings, and saves the staged release to CI state. Nothing public changes. |
| `stage-and-promote` | the same, then publishes through `scripts/publish_full_context_release.py` (rollback archive + readback checks), but only if the gates pass. |
| `promote-staged` | publishes the release an earlier `stage` run left staged. |

The guard is pre-registered in `cfbpred/ci_retrain.py` (`GUARD`). On completed FBS-involved games of the season, both models are scored on the same store chain, and the candidate must stay within +0.05 margin MAE (PMF median) and +0.002 win-probability log loss of the live model, over at least 200 games. Lines are never used. A run that fails a gate exits non-zero and saves no state, so `active.json` and every pointer stay as they were.

State: the retrain job is the only writer of the `static` set. `static` follows `active.json` and `staged_full_context.json` to the release folder and its store chain (`runs_from` + `match` in `ci/state_sets.json`), so a promotion survives the next restore and the previous release stays as the rollback target.

When a new season starts, bump `CURRENT_SEASON` in `config.py` (the checkpoint lineage check trains only on earlier seasons) and the season guard in the publisher before retraining.

## No Mac fallback

Nothing in this pipeline runs on the laptop, and there is no rollback to it.
The old launchd agents (`com.cfbpred.scores`, `.availability-dispatcher`,
`.refresh`, `.live`) were only booted out on 2026-11-18 with their plists left
in `~/Library/LaunchAgents`, so the next login reloaded them. For six days they
ran beside the Actions jobs: they published to Supabase on their own and wrote
~30 GB of refresh/promotion archives and availability captures to the Mac's
disk until it was full. On 2026-11-24 the plists were deleted and the agents
disabled (`launchctl print-disabled gui/$(id -u)`), and `refresh_scores.sh`,
`refresh_week.sh`, `refresh_availability.sh` and `run_live_scorer.sh` now exit
unless `GITHUB_ACTIONS=true`.

To pause the football pipeline, set `CFB_ACTIONS_ENABLED=false` and unschedule
the `gh-cfb-*` pg_cron jobs (above). Hand runs from the Actions tab still go.

## Housekeeping

- `python -m cfbpred.ci_state gc --keep 10` deletes chunks no manifest or archive
  references.
- The pregame lead (120 min) and per-slot timeout (300 min) are set in
  `cfb-pregame.yml`; the Mac defaults (30/60) are unchanged.
