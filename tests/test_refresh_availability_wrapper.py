import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(tmp_path, stub_exit, args=(), fallback_exit=0, archive_exit=0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / ".env").write_text("")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").write_text(
        "#!/bin/zsh\n"
        "if [[ \"$*\" == *snapshot_sources.py* ]]; then\n"
        "  print -r -- \"archive-called\" >> \"$STUB_MARKER\"; exit ${ARCHIVE_EXIT:-0}\n"
        "fi\n"
        "if [[ \"$STUB_MARKER\" != '' ]]; then print -r -- \"python-called\" >> \"$STUB_MARKER\"; fi\n"
        "exit ${STUB_EXIT:-0}\n"
    )
    (bin_dir / "python3").chmod(0o755)
    (tmp_path / "refresh_week.sh").write_text(
        f"#!/bin/zsh\nprint -r -- fallback >> \"$STUB_MARKER\"\nexit {fallback_exit}\n"
    )
    (tmp_path / "refresh_week.sh").chmod(0o755)
    lock = tmp_path / "lock"
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
           "STUB_EXIT": str(stub_exit), "STUB_MARKER": str(tmp_path / "marker"),
           "CFB_PUBLISH_LOCK": str(lock), "ARCHIVE_EXIT": str(archive_exit)}
    wrapper = tmp_path / "refresh_availability.sh"
    shutil.copy2(ROOT / "refresh_availability.sh", wrapper)
    wrapper.chmod(0o755)
    return subprocess.run([str(wrapper), *args], cwd=tmp_path,
                          env=env, text=True, capture_output=True), tmp_path / "marker"


def test_wrapper_success_and_failure_propagate(tmp_path):
    ok, marker = _run(tmp_path / "ok", 0)
    assert ok.returncode == 0
    # a successful pre-kickoff refresh also archives dated source snapshots
    assert marker.read_text().splitlines() == ["python-called", "archive-called"]

    bad, marker = _run(tmp_path / "bad", 7)
    assert bad.returncode == 7
    assert marker.read_text().splitlines() == ["python-called"]   # no archive after a failed refresh


def test_failed_archive_never_changes_the_refresh_exit_code(tmp_path):
    ok, marker = _run(tmp_path / "archive_fails", 0, archive_exit=3)
    assert ok.returncode == 0
    assert marker.read_text().splitlines() == ["python-called", "archive-called"]


def test_no_publish_validation_writes_no_archive(tmp_path):
    ok, marker = _run(tmp_path / "validate", 0, args=("--no-publish",))
    assert ok.returncode == 0
    assert marker.read_text().splitlines() == ["python-called"]


def test_stale_input_falls_back_to_full_refresh(tmp_path):
    result, marker = _run(tmp_path / "fallback", 42, fallback_exit=0)
    assert result.returncode == 0
    assert marker.read_text().splitlines() == ["python-called", "fallback"]


def test_stale_input_no_publish_preserves_42(tmp_path):
    result, marker = _run(tmp_path / "readonly", 42, args=("--no-publish",))
    assert result.returncode == 42
    assert marker.read_text().splitlines() == ["python-called"]
