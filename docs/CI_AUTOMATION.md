# Scheduled jobs on GitHub Actions

Since 2026-11-17 the three scheduled CFB jobs run on GitHub-hosted runners, not
the Mac. The live in-game scorer was already a Supabase Edge Function.

| Workflow | Schedule | Replaces |
|---|---|---|
| `scores` (`.github/workflows/cfb-scores.yml`) | :07 and :37 every hour | launchd `com.cfbpred.scores` |
| `nightly` (`cfb-nightly.yml`) | 10:30 UTC (04:30 Central) | launchd `com.cfbpred.refresh` |
| `pregame` (`cfb-pregame.yml`) | check every 10 min; refresh 120 min before each kickoff slot | launchd `com.cfbpred.availability-dispatcher` |
| `state check` (`cfb-state-check.yml`) | manual | read-only rehearsal |

Scheduled runs only fire while the repository variable `CFB_ACTIONS_ENABLED` is
`true`; manual runs always work. All jobs share the `cfb-pipeline`
concurrency group, which replaces `/tmp/cfbpred_publish.lock`.

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

## Rolling back to the Mac

```bash
cd ~/IdeaProjects/CFB-Predictor-2026
gh variable set CFB_ACTIONS_ENABLED --body false -R noltechas/CFB-Predictor-2026
PYTHONPATH=src python3 -m cfbpred.ci_state restore core schedule dispatch daily --force-local
for l in scores availability-dispatcher refresh; do
  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.cfbpred.$l.plist
done
```

`--force-local` makes the Mac's files match the saved state exactly (it deletes
local files the state lacks under those roots) — that is the point of a
rollback, but check nothing local is newer first.

## Housekeeping

- `python -m cfbpred.ci_state gc --keep 10` deletes chunks no manifest or archive
  references.
- The pregame lead (120 min) and per-slot timeout (300 min) are set in
  `cfb-pregame.yml`; the Mac defaults (30/60) are unchanged.
