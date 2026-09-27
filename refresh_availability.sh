#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"
# Hold the Mac awake for exactly this run: launchd often fires jobs inside a
# ~45 s maintenance DarkWake, and the Mac re-sleeping mid-COPY killed the
# 2026-09-26/27 nightlies. -s (system sleep, on AC) is what holds a DarkWake.
caffeinate -i -s -w $$ &
set -a; source .env; set +a
LOCK=${CFB_PUBLISH_LOCK:-/tmp/cfbpred_publish.lock}
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "publish pipeline busy — refusing to race active lock: $LOCK" >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
set +e
PYTHONPATH=src python3 -m cfbpred.refresh_availability "$@"
cfb_refresh_exit_code=$?
set -e
if [ "$cfb_refresh_exit_code" -eq 42 ]; then
  rmdir "$LOCK" 2>/dev/null || true
  trap - EXIT
  case " $* " in
    *" --no-publish "*)
      echo "availability inputs stale; --no-publish validation stopped (exit 42)" >&2
      exit 42
      ;;
    *)
      exec ./refresh_week.sh
      ;;
  esac
fi
# Pre-kickoff snapshot: weather forecasts, lines and availability as known ~30 minutes
# before each kickoff slot. Best-effort; it never changes this script's exit code.
case " $* " in
  *" --no-publish "*) ;;   # validation runs write nothing external
  *)
    if [ "$cfb_refresh_exit_code" -eq 0 ]; then
      PYTHONPATH=src python3 src/cfbpred/snapshot_sources.py || echo "source archive incomplete (non-fatal)" >&2
    fi
    ;;
esac
exit "$cfb_refresh_exit_code"
