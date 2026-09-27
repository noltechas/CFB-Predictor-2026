"""Regression checks for current-season DB run selection.

Historical reconstruction deliberately adds older-season rows and run IDs to
the same tables.  The daily/T30 publisher must therefore select a single run
within the current season, rather than treating the whole table as one run.
"""

import pytest

from cfbpred.config import CURRENT_SEASON
from cfbpred.publish_future_predictions import current_production_run


class _Cursor:
    def __init__(self, rows):
        self.rows = rows
        self.sql = None
        self.params = None

    def execute(self, sql, params=()):
        self.sql = sql
        self.params = params

    def fetchall(self):
        return self.rows


def test_current_run_ignores_historical_run_ids():
    # The query result represents the current-season slice.  An older run is
    # present in the same table in production, but must not affect the choice.
    cursor = _Cursor([("current-run",)])

    assert current_production_run(cursor) == "current-run"
    assert "where season=%s" in cursor.sql.lower()
    assert cursor.params == (CURRENT_SEASON,)


def test_current_run_rejects_multiple_current_season_runs():
    cursor = _Cursor([("current-run",), ("stray-current-run",)])

    with pytest.raises(ValueError, match="one current-season production run"):
        current_production_run(cursor)

