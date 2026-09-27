# Refresh scheduling

The existing launchd jobs remain the source of truth for the current refresh pipeline:

- `com.cfbpred.refresh` runs the daily full refresh at 04:30.
- `com.cfbpred.scores` runs score refreshes every 30 minutes.
- `com.cfbpred.live` runs the persistent live scorer.

`cfbpred.refresh_scheduler` adds a schedule-aware dispatcher. It reads the
current schedule each minute, groups games sharing a kickoff, and invokes the
lightweight `refresh_availability.sh` hook once in the window from kickoff minus 30
minutes through kickoff. It never dispatches at or after kickoff, which keeps
the pregame cutoff intact. The state file records slot attempts and successful
completion; failures retry on the next minute, while completed slots are
idempotent. A changed kickoff naturally produces a new slot key.

Generate the additional launchd plist with:

```sh
PYTHONPATH=src python3 scripts/install_refresh_schedule.py \
  --project "$PWD" \
  --schedule data/parquet/schedule.parquet \
  --state "$PWD/logs/availability_dispatcher_state.json" \
  --kickoff-command "$PWD/refresh_availability.sh"
```

Load it with `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.cfbpred.availability-dispatcher.plist` after reviewing the generated plist. The installer does not modify or duplicate the existing daily, score, or live jobs.
