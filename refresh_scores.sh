#!/bin/zsh
# Light in-season refresh, safe to run often (~2 min): final scores and fresh
# lines in, nothing re-modelled.
#
#   ./refresh_scores.sh
#
# What it does NOT do: rebuild the feature store or re-score games. Pregame
# calls are locked before kickoff by design, so the only thing a finished game
# changes intra-day is the Results ledger — and a team's own next prediction
# is picked up by the nightly refresh_week.sh, which rebuilds features.
#
# Pairing: refresh_scores.sh every 30 min keeps scores/lines/Results current;
# refresh_week.sh nightly folds those results into future predictions.
set -euo pipefail
cd "$(dirname "$0")"
# Hold the Mac awake for exactly this run: launchd often fires jobs inside a
# ~45 s maintenance DarkWake, and the Mac re-sleeping mid-COPY killed the
# 2026-09-26/27 nightlies. -s (system sleep, on AC) is what holds a DarkWake.
caffeinate -i -s -w $$ &
set -a; source .env; set +a

# One run at a time. launchd keeps a single instance per label, but a manual
# run — or a `launchctl kickstart -k` that leaves the previous python child
# alive — can overlap one, and stage 4 DROPS and recreates game_lines: two of
# those at once is a corrupt table. Observed 2026-08-29, when a run wedged on
# a dead database link for 5h33m and the replacement collided with it.
# mkdir is atomic; a lock older than 2h belonged to a run that was killed.
# Fixed path on purpose: $TMPDIR is per-context on macOS (a launchd agent
# and an interactive shell get DIFFERENT ones), so a TMPDIR-based lock
# would never be seen by the other side and the guard would be a no-op.
LOCK="/tmp/cfbpred_publish.lock"   # SHARED with refresh_week.sh
if [ -d "$LOCK" ] && [ -n "$(find "$LOCK" -maxdepth 0 -mmin +120 2>/dev/null)" ]; then
  echo "stale lock (>2h) at $LOCK — reclaiming"
  rmdir "$LOCK" 2>/dev/null || true
fi
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another refresh_scores.sh is already running — skipping ($(date))"
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

echo "== 1/4 CFBD pull (scores, lines, rosters) =="
python3 src/cfbpred/update_season.py 2026

# update_season replaces the season's roster rows with CFBD's pull, which
# drops the ESPN/alias fills; restore them so the next feature build is whole
echo "== 2/4 restore roster fills =="
python3 src/cfbpred/pull_espn_roster.py

echo "== 3/4 ESPN lines (FCS slate + pickcenter backfill) =="
python3 src/cfbpred/pull_espn_lines.py

echo "== 4/4 republish lines + results view =="
python3 src/cfbpred/publish_lines.py

# Preserve fresh public availability on every scheduled score pull. The daily
# and T-30 pipelines consume this append-only archive when scoring forecasts.
PYTHONPATH=src python3 -m cfbpred.conference_availability
PYTHONPATH=src python3 -m cfbpred.covers_availability || echo "covers capture failed (non-fatal)"

echo "scores refreshed ($(date))"
