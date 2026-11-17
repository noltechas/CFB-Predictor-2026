#!/bin/zsh
# Daily in-season refresh: new scores + fresh Vegas lines in, new predictions
# and rankings out. No retraining — the active pregame ensemble stays fixed
# for the season.
#
#   ./refresh_week.sh            # refresh 2026
#
# Box scores and raw CFBD plays/attribution are pulled in step 2. Each game
# gets its own prior-observation snapshot, including every future week.
# The raw-play fallback supplies pregame rates while the repaired cfbfastR
# release is absent; it never enters the live-model training corpus. A later
# repaired release automatically takes precedence over overlapping games.
set -euo pipefail
cd "$(dirname "$0")"
# Hold the Mac awake for exactly this run: launchd often fires jobs inside a
# ~45 s maintenance DarkWake, and the Mac re-sleeping mid-COPY killed the
# 2026-09-26/27 nightlies. -s (system sleep, on AC) is what holds a DarkWake.
caffeinate -i -s -w $$ &
set -a; source .env; set +a

# Exclusive access to the publish pipeline, SHARED with refresh_scores.sh:
# both run update_season / pull_espn_lines / publish_lines, publish_lines DROPS
# and recreates game_lines, and the two live under different launchd labels
# that both fire at 04:30 — launchd's one-instance-per-label rule does not keep
# them apart. This job WAITS (the frequent scores job only takes ~2 min and
# this nightly rebuild must not be silently skipped); refresh_scores.sh skips.
# Fixed path on purpose: $TMPDIR is per-context on macOS (a launchd agent
# and an interactive shell get DIFFERENT ones), so a TMPDIR-based lock
# would never be seen by the other side and the guard would be a no-op.
LOCK="/tmp/cfbpred_publish.lock"
if [ -d "$LOCK" ] && [ -n "$(find "$LOCK" -maxdepth 0 -mmin +120 2>/dev/null)" ]; then
  echo "stale lock (>2h) at $LOCK — reclaiming"
  rmdir "$LOCK" 2>/dev/null || true
fi
cfb_lock_acquired=0
for _i in $(seq 1 40); do
  if mkdir "$LOCK" 2>/dev/null; then
    cfb_lock_acquired=1
    break
  fi
  echo "publish pipeline busy — waiting 30s ($_i/40)"
  sleep 30
done
if [ "$cfb_lock_acquired" -ne 1 ]; then
  echo "could not acquire $LOCK after 20 min — aborting so nothing races" >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

echo "== 0/9 public conference availability archive =="
# Capture the public reports before any later CFBD/plays failure can abort the
# required model refresh.  The collector writes an immutable receipt and keeps
# source failures in latest.json; a collector failure itself remains fatal.
PYTHONPATH=src python3 -m cfbpred.conference_availability
PYTHONPATH=src python3 -m cfbpred.covers_availability || echo "covers capture failed (non-fatal)"

echo "== 1/10 CFBD pull (schedule, scores, lines, rosters) =="
python3 src/cfbpred/update_season.py 2026
python3 src/cfbpred/pull_roster_movement.py portal draft_picks || echo "roster movement pull failed (non-fatal; last pull kept)"
# fill roster gaps CFBD hasn't published yet (FCS lag) from ESPN + aliases;
# idempotent — pairs drop out of the fill as CFBD coverage arrives
python3 src/cfbpred/pull_espn_roster.py

echo "== 2/10 box scores (season-to-date stat features) =="
python3 src/cfbpred/pull_player_stats.py 2026
python3 src/cfbpred/pull_current_plays.py 2026

echo "== 3/10 rebuild feature store =="
# Restore only the reviewed, immutable historical fixture overlay.  The
# pointer is intentionally optional: before review this is a no-op.
PYTHONPATH=src python3 -m cfbpred.historical_fixture_refresh
python3 src/cfbpred/features_raw.py
python3 src/cfbpred/audit_future_features.py

echo "== 4/10 publish raw tables =="
python3 src/cfbpred/publish_feature_refresh.py

# Step 4 caches the committed feature payloads locally. The historical play
# corpus is unchanged and remains cached; a cold cache is loaded on demand.
# A deliberate play-corpus rebuild must still run source.py to refresh it.
echo "== 5/10 verify committed feature caches =="
python3 -c 'import sys; sys.path.insert(0, "src"); from cfbpred.source import CACHE; assert all((CACHE / (t + ".parquet")).is_file() for t in ("game_scalars", "game_history")), "Published feature cache missing"'

echo "== 6/10 refresh pregame forecasts (active player-sequence TabM ensemble; started games locked) =="
python3 src/cfbpred/refresh_player_sequence_production.py --promote

echo "== 7/10 publish predictions + lines =="
python3 src/cfbpred/publish_future_predictions.py
python3 src/cfbpred/pull_espn_lines.py # FCS-slate lines CFBD doesn't carry
python3 src/cfbpred/publish_lines.py

# The scheduled edge scorer remains the validated raw-DNN STATE model. Its
# own pregame tensors are computed here and read back by the function. The
# promoted player-sequence model owns the public pregame predictions above.
# Rolls a -3/+10 day window forward, so a game outside it cannot be scored.
echo "== 8/10 publish pregame logits for the edge scorer =="
# The edge scorer's pregame half is now the PUBLISHED pregame distribution itself
# (publish_live_priors.py), so the live number at kickoff starts where the pregame card
# ends. Validated on held-out 2025 plays: in-game win-probability log loss .3352 -> .3314.
# The retired raw-DNN pregame tower (publish_pregame_logits.py) is kept for rollback only.
python3 src/cfbpred/publish_live_priors.py 2026

echo "== 9/10 publish current Fieldcast rankings =="
python3 src/cfbpred/rankings.py --season 2026 --publish

# Point-in-time archive of every source that overwrites itself (rosters, schedule,
# weather forecasts, lines, availability documents). Best-effort: a failed snapshot
# must never block a published prediction, so it cannot fail this script.
echo "== 10/10 archive dated source snapshots =="
python3 src/cfbpred/snapshot_sources.py || echo "source archive incomplete (non-fatal)"

echo "done — dashboard is current ($(date))"
