#!/bin/zsh
# Wrapper so launchd can run the live scorer with the project's environment.
# The scorer idles cheaply (two scoreboard requests a minute, ~240 MB) when
# nothing is in progress, so it is left resident rather than scheduled around
# kickoff times — and launchd's KeepAlive restarts it if it ever dies.
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a
exec python3 src/cfbpred/live_scorer.py --loop 60
