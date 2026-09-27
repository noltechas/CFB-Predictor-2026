import json
from pathlib import Path

from cfbpred.official_team_archives import (discover_cached_directory,
                                            fetch_json_queue, modeled_team_manifest)


class Response:
    def __init__(self, status=200, body=b'{"ok":true}', headers=None):
        self.status_code = status
        self.content = body
        self.headers = headers or {}


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)

    def get(self, url, timeout=30):
        return next(self.responses)


def test_queue_stops_at_first_rate_limit_and_preserves_pending(tmp_path):
    out = fetch_json_queue(Session([Response(), Response(429, headers={"Retry-After": "60"}), Response()]),
                           ["https://a/1", "https://a/2", "https://a/3"], tmp_path, sleep=lambda _: None)
    assert out["stopped_on_429"] is True
    assert out["pending"] == ["https://a/2", "https://a/3"]
    assert out["receipts"][0]["sha256"]


def test_manifest_has_explicit_states(tmp_path):
    pd = __import__("pandas")
    p = tmp_path / "s.parquet"
    pd.DataFrame({"home_team": ["Nebraska"], "away_team": ["Dana College"],
                  "home_division": ["fbs"], "away_division": ["fcs"]}).to_parquet(p)
    rows = modeled_team_manifest(p)
    assert {r["team"] for r in rows} == {"Nebraska", "Dana College"}
    assert all(r["state"] == "pending" for r in rows)


def test_cached_directory_keeps_source_provenance(tmp_path):
    p = tmp_path / "directory.html"
    p.write_text('{"title":"Example U","athletics_website":"https://example.edu"}')
    rows = discover_cached_directory([p])
    assert rows == [{"team": "Example U", "official_url": "https://example.edu",
                     "source_path": str(p), "state": "pending",
                     "reason": "official directory link; provider unverified"}]
