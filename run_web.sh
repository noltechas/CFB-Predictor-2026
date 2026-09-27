#!/usr/bin/env bash
# Serve the dashboard. Needs a real origin — file:// is blocked by CORS.
#
# No-store headers matter: python's plain `http.server` sends no cache headers
# at all, so browsers fall back to HEURISTIC caching and keep running a stale
# app.js after every edit — which looks exactly like "my change did nothing".
# Optional first argument overrides the port (default 8787).
set -euo pipefail
cd "$(dirname "$0")/web"
PORT="${1:-8787}"
echo "http://localhost:$PORT"
exec python3 -c '
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class NoCache(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), NoCache).serve_forever()
' "$PORT"
