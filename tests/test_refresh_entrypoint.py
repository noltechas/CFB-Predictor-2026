"""The scheduled command must work without an interactive PYTHONPATH."""
import os
from pathlib import Path
import subprocess
import sys


def test_refresh_script_imports_when_invoked_by_the_daily_job():
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run([sys.executable, "src/cfbpred/refresh_player_sequence_production.py", "--help"],
                            cwd=root, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--promote" in result.stdout
