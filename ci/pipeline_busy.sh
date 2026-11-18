#!/usr/bin/env bash
# Prints busy=true when heavier work already holds or waits for the cfb-pipeline
# concurrency group: a nightly, a retrain, or a pregame refresh job.
#
# GitHub keeps only ONE waiting run per concurrency group, so with the Supabase
# scheduler starting scores every 30 minutes and pregame checks every 10, a newly
# queued run would cancel whichever run was already waiting — a scores run could
# knock out a waiting nightly. Scheduled scores runs and pregame refreshes call
# this first and stand down when it says busy; the heavier job republishes
# everything they would have.
#
# Needs GH_TOKEN with actions:read; uses GITHUB_REPOSITORY and GITHUB_RUN_ID.
set -euo pipefail

busy=false
for wf in cfb-nightly.yml cfb-retrain.yml cfb-pregame.yml; do
  for status in queued in_progress waiting pending requested; do
    ids=$(gh api "repos/$GITHUB_REPOSITORY/actions/workflows/$wf/runs?status=$status&per_page=20" \
      -q '.workflow_runs[].id' 2>/dev/null || true)
    for id in $ids; do
      [ "$id" = "$GITHUB_RUN_ID" ] && continue
      if [ "$wf" = cfb-pregame.yml ]; then
        # a pregame run counts only while its refresh job is queued or running;
        # its 20-second check alone holds nothing
        if gh api "repos/$GITHUB_REPOSITORY/actions/runs/$id/jobs" \
             -q '.jobs[] | select(.name == "refresh" and .status != "completed") | .id' 2>/dev/null | grep -q .; then
          busy=true
        fi
      else
        busy=true
      fi
      [ "$busy" = true ] && { echo "busy because of $wf run $id ($status)" >&2; echo "busy=true"; exit 0; }
    done
  done
done
echo "busy=false"
