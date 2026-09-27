"""Unchanged captures share one copy on disk without changing any evidence."""
import hashlib
import os

import pytest

from cfbpred.pull_player_context import archive


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_unchanged_capture_links_to_the_previous_run(tmp_path):
    first = archive(b'{"a": 1}', "https://x.example", tmp_path / "20260925T000000Z", "B10_report.json",
                    link_previous=True)
    second = archive(b'{"a": 1}', "https://x.example", tmp_path / "20260925T003000Z", "B10_report.json",
                     link_previous=True)
    a, b = tmp_path / "20260925T000000Z/B10_report.json", tmp_path / "20260925T003000Z/B10_report.json"
    assert os.stat(a).st_ino == os.stat(b).st_ino and digest(b) == second["sha256"] == first["sha256"]
    assert not os.access(b, os.W_OK)  # a write would change every run sharing the file
    # Each capture keeps its own receipt and retrieval time.
    assert (tmp_path / "20260925T003000Z/B10_report.json.source.json").exists()


def test_changed_capture_and_default_calls_write_their_own_copy(tmp_path):
    archive(b"one", "u", tmp_path / "r1", "doc.json", link_previous=True)
    archive(b"two", "u", tmp_path / "r2", "doc.json", link_previous=True)
    archive(b"one", "u", tmp_path / "r3", "doc.json")
    inodes = {os.stat(tmp_path / r / "doc.json").st_ino for r in ("r1", "r2", "r3")}
    assert len(inodes) == 3
    assert (tmp_path / "r2/doc.json").read_bytes() == b"two"


def test_existing_identical_file_is_never_rewritten(tmp_path):
    archive(b"same", "u", tmp_path / "r1", "doc.json", link_previous=True)
    archive(b"same", "u", tmp_path / "r2", "doc.json", link_previous=True)
    shared = tmp_path / "r2/doc.json"
    before = os.stat(shared)
    archive(b"same", "u", tmp_path / "r2", "doc.json", link_previous=True)  # read-only: a rewrite would raise
    assert os.stat(shared).st_ino == before.st_ino and os.stat(shared).st_mtime_ns == before.st_mtime_ns
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        archive(b"different", "u", tmp_path / "r2", "doc.json", link_previous=True)
