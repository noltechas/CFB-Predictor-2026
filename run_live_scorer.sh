#!/bin/zsh
# Wrapper so launchd can run the live scorer with the project's environment.
# The scorer idles cheaply (two scoreboard requests a minute, ~240 MB) when
# nothing is in progress, so it is left resident rather than scheduled around
# kickoff times — and launchd's KeepAlive restarts it if it ever dies.
set -euo pipefail
cd "$(dirname "$0")"
# Retired: live scoring runs in Supabase (pg_cron + edge function). Kept for
# reference only; it refuses to run anywhere but a GitHub Actions runner.
if [ "${GITHUB_ACTIONS:-}" != "true" ]; then
  echo "$(basename "$0") is retired (live scoring runs in Supabase); not running on this machine." >&2
  exit 1
fi
set -a; source .env; set +a
exec python3 src/cfbpred/live_scorer.py --loop 60
