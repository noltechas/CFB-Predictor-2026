import json

from cfbpred.publish_future_predictions import active_model_version
from cfbpred.publish_lines import create_results_view


def test_active_prediction_version_comes_from_production_pointer(tmp_path):
    (tmp_path / "active.json").write_text(json.dumps({"version": "player-model-v3"}))
    assert active_model_version(tmp_path) == "player-model-v3"


def test_results_view_migration_never_mutates_source_tables():
    statements = []

    class Cursor:
        def execute(self, sql):
            statements.append(sql)

    create_results_view(Cursor())
    view_sql = statements[0].lower()
    assert "create or replace view public.v_model_vs_vegas" in view_sql
    assert "model_version" in view_sql
    assert "drop table" not in view_sql
    assert "delete from" not in view_sql
    assert "update " not in view_sql
