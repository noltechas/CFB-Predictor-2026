#!/usr/bin/env python3
"""Install the schedule-aware kickoff dispatcher launchd job.

The existing com.cfbpred.refresh job remains the daily full refresh. This
installer refuses to replace it and only writes the additional dispatcher.
"""
from __future__ import annotations
import argparse
import plistlib
import shlex
from pathlib import Path

LABEL = "com.cfbpred.availability-dispatcher"


def plist(project: Path, schedule: Path, state: Path, kickoff_command: Path) -> dict:
    command = ["/usr/bin/python3", "-m", "cfbpred.refresh_scheduler", "--schedule", str(schedule), "--state", str(state), "--refresh-command", str(kickoff_command)]
    shell = "cd %s && PYTHONPATH=src exec %s" % (shlex.quote(str(project)), shlex.join(command))
    return {"Label": LABEL, "ProcessType": "Standard", "ProgramArguments": ["/bin/zsh", "-lc", shell], "StartInterval": 60, "WorkingDirectory": str(project), "StandardOutPath": str(project / "logs/availability_dispatcher.log"), "StandardErrorPath": str(project / "logs/availability_dispatcher.log")}


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--project", type=Path, required=True); p.add_argument("--schedule", type=Path, required=True); p.add_argument("--state", type=Path, required=True); p.add_argument("--kickoff-command", type=Path); p.add_argument("--output", type=Path)
    a = p.parse_args(); out = a.output or Path.home() / "Library/LaunchAgents" / (LABEL + ".plist")
    kickoff = (a.kickoff_command or (a.project / "refresh_availability.sh")).resolve()
    out.parent.mkdir(parents=True, exist_ok=True); out.write_bytes(plistlib.dumps(plist(a.project.resolve(), a.schedule.resolve(), a.state.resolve(), kickoff)))
    print(out); return 0
if __name__ == "__main__": raise SystemExit(main())
