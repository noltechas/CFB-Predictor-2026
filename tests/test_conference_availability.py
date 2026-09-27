import json
import pandas as pd
from cfbpred.conference_availability import archive_facts, build_fixture_index, load_facts
from cfbpred.availability_backfill import validate_fact
from cfbpred.pull_player_context import archive
from cfbpred.conference_availability import collect, current_facts


class _Response:
    status_code = 200
    headers = {}
    def __init__(self, payload):
        self.content = json.dumps(payload).encode()
        self._payload = payload
    def raise_for_status(self):
        return None
    def json(self):
        return self._payload


class _Session:
    def __init__(self): self.calls = []
    def post(self, url, json, timeout):
        self.calls.append((url, json))
        return _Response([] if "M!nD0Mcfp" in url else {"data": []})


def test_public_american_alias_and_cfp_empty_receipt(tmp_path):
    session = _Session()
    run = collect(tmp_path, conferences=["American", "CFP"], session=session)
    assert [x[1] for x in session.calls] == [
        {"sport": "Football", "organization": "American", "conference": "American"},
        {"sport": "Football", "organization": "American", "conference": "American"},
        {"organization": "CFP"},
    ]
    manifest = json.loads((run / "manifest.json").read_text())
    assert len(manifest["sources"]) == 3
    cfp = json.loads((run / "CFP_archive.json").read_text())
    assert cfp == []


def test_archive_assumption_preserves_status_and_exact_fixture(tmp_path):
    schedule=pd.DataFrame([dict(game_id=1,season=2025,kick='2025-09-13T19:00:00Z',home_team='USC',away_team='A')])
    raw=dict(Player='John Smith',Team='Southern California',TeamDisplay='USC',Opponent='vs. A',Week='Sat, 13 Sep 2025 00:00:00 GMT',**{'Game Day':'Game Time Decision','Initial':'Out'})
    archive(json.dumps({'data':[raw]}).encode(),'https://app.hdintelligence.com/api/get-archive-public',tmp_path,'report.json',observed_at='2026-09-01T12:00:00Z')
    facts,issues=archive_facts(tmp_path/'report.json',schedule)
    assert not issues and len(facts)==1
    fact=facts[0]
    assert fact['timing_assumed'] and fact['value']=='questionable'
    assert fact['raw_status']=='Game Time Decision'
    assert fact['known_at']=='2025-09-13T18:30:00+00:00'
    assert validate_fact(fact,schedule.set_index('game_id')) is None
    assert validate_fact({**fact,'value':'out'},schedule.set_index('game_id'))=='assumed_archive_fact_mismatch'
    assert not archive_facts(tmp_path/'report.json',schedule,allow_assumed=False)[0]
    raw['Opponent']='vs. B'
    (tmp_path/'report.json').write_text(json.dumps({'data':[raw]}))
    assert not archive_facts(tmp_path/'report.json',schedule)[0]


def test_partial_suspension_does_not_become_full_game_out(tmp_path):
    schedule=pd.DataFrame([dict(game_id=1,season=2025,kick='2025-09-13T19:00:00Z',home_team='USC',away_team='A')])
    raw=dict(Player='John Smith',TeamDisplay='USC',Opponent='vs. A',Week='Sat, 13 Sep 2025 00:00:00 GMT',**{'Game Day':'Out (1st Half)'})
    archive(json.dumps({'data':[raw]}).encode(),'https://app.hdintelligence.com/api/get-archive-public',tmp_path,'report.json',observed_at='2026-09-01T12:00:00Z')
    facts,issues=archive_facts(tmp_path/'report.json',schedule)
    assert not facts and issues[0]['reason']=='unsupported_or_missing_game_day_status'


def test_public_list_payloads_are_not_silently_lost(tmp_path):
    schedule=pd.DataFrame([dict(game_id=1,season=2025,kick='2025-09-13T19:00:00Z',home_team='USC',away_team='A')])
    row=dict(Player='John Smith',TeamDisplay='USC',Opponent='A',Week='2025-09-13',**{'Game Day':'Out'})
    archive(json.dumps([row]).encode(),'https://example.org/public',tmp_path,'report.json',observed_at='2026-09-01T12:00:00Z')
    facts, issues=archive_facts(tmp_path/'report.json',schedule)
    assert len(facts)==1 and not issues
    (tmp_path/'report.json').write_text('[{"unknown_schema": true}]')
    assert archive_facts(tmp_path/'report.json',schedule)[1][0]['reason']=='unsupported_archive_row'
    assert current_facts(tmp_path/'report.json',schedule)[1]
    (tmp_path/'report.json').write_text('[]')
    assert current_facts(tmp_path/'report.json',schedule)==([], [])


def test_archive_fixture_index_reuse_matches_direct_matching(tmp_path):
    schedule = pd.DataFrame([dict(game_id=1, season=2025,
                                  kick='2025-09-13T19:00:00Z', home_team='USC', away_team='A')])
    raw = dict(Player='John Smith', TeamDisplay='USC', Opponent='A', Week='2025-09-13', **{'Game Day': 'Out'})
    archive(json.dumps({'data': [raw]}).encode(), 'https://example.org/public', tmp_path, 'report.json',
            observed_at='2025-09-01T12:00:00Z')
    direct = archive_facts(tmp_path / 'report.json', schedule)
    indexed = archive_facts(tmp_path / 'report.json', schedule,
                            fixture_index=build_fixture_index(schedule))
    assert indexed == direct


def test_load_facts_builds_schedule_fixture_index_once(tmp_path, monkeypatch):
    schedule = pd.DataFrame([dict(game_id=1, season=2025,
                                  kick='2025-09-13T19:00:00Z', home_team='USC', away_team='A')])
    raw = dict(Player='John Smith', TeamDisplay='USC', Opponent='A', Week='2025-09-13', **{'Game Day': 'Out'})
    run_dir = tmp_path / 'run'; run_dir.mkdir()
    archive(json.dumps({'data': [raw]}).encode(), 'https://example.org/public', run_dir, 'report.json',
            observed_at='2025-09-01T12:00:00Z')
    item = json.loads((run_dir / 'report.json.source.json').read_text())
    (run_dir / 'manifest.json').write_text(json.dumps({'sources': [{**item, 'kind': 'archive'}]}))
    calls = []
    import cfbpred.conference_availability as module
    original = module.build_fixture_index
    monkeypatch.setattr(module, 'build_fixture_index', lambda schedule: (calls.append(1) or original(schedule)))
    facts, issues = load_facts(schedule, tmp_path)
    assert len(facts) == 1 and not issues
    assert len(calls) == 1
