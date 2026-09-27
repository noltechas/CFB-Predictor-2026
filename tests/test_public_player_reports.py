from cfbpred.public_player_reports import status_lists
import json
import pandas as pd
from cfbpred.public_player_reports import sec_current
from cfbpred.public_player_reports import dated_articles


def test_explicit_report_lists_do_not_label_unlisted_players_healthy():
    html='<p>Jane may play. Here is the report:</p><p>Questionable:</p><ul><li>QB Alex Smith</li></ul><p>Out</p><ul><li><b>SAF</b> Jo Jones (season)</li><li>TC Caffey</li></ul><p>Other list</p><ul><li>WR Jane Doe</li></ul>'
    assert list(status_lists(html))==[('Alex Smith','questionable','QB'),('Jo Jones','out','DB'),('TC Caffey','out',None)]


def test_empty_report_is_unknown_not_a_roster_of_available_players():
    assert list(status_lists('<p>Questionable</p><ul><li>None</li></ul><p>Out</p><ul><li>None</li></ul>'))==[]


def test_sec_pending_and_retrieved_after_kickoff_never_become_available(tmp_path):
    p=tmp_path/'sec.json';p.with_suffix('.json.source.json').write_text(json.dumps(dict(observed_at='2026-09-21T12:00:00Z',url='https://example.org')))
    report=dict(ReportType='Report Pending',footer=dict(date='2026-09-26'),games=[dict(teamDisplayName='A',rows=[dict(name='Test',status='Out')]),dict(teamDisplayName='B',rows=[])])
    games=pd.DataFrame([dict(game_id=1,season=2026,home_team='A',away_team='B',kick='2026-09-26T19:00:00Z')])
    p.write_text(json.dumps({'one':report}));facts,issues=sec_current(p,games)
    assert not facts and issues[0]['reason']=='report_pending'
    report['ReportType']='Initial';p.write_text(json.dumps({'one':report}))
    facts,issues=sec_current(p,games);assert len(facts)==1 and facts[0]['value']=='out'
    assert facts[0]['known_at']=='2026-09-21T12:00:00+00:00'
    meta=json.loads(p.with_suffix('.json.source.json').read_text());meta['observed_at']='2026-09-27T00:00:00Z';p.with_suffix('.json.source.json').write_text(json.dumps(meta))
    assert not sec_current(p,games)[0]


def test_postgame_article_revision_is_excluded_even_with_pregame_publication(tmp_path):
    p=tmp_path/'articles.json';p.with_suffix('.json.source.json').write_text(json.dumps(dict(observed_at='2026-09-21T12:00:00Z')))
    row=dict(link='https://example.org/report',date_gmt='2024-09-01T10:00:00',modified_gmt='2024-09-01T20:00:00',
             content=dict(rendered='<p>Out</p><ul><li>QB Alex Smith</li></ul>'))
    p.write_text(json.dumps([row]))
    games=pd.DataFrame([dict(game_id=1,season=2024,home_team='Ohio State',away_team='A',kick='2024-09-01T16:00:00Z')])
    facts,issues=dated_articles(p,games)
    assert not facts and issues[0]['reason']=='article_revised_at_or_after_kickoff'
    row['modified_gmt']='2024-09-01T12:00:00';p.write_text(json.dumps([row]))
    facts,issues=dated_articles(p,games)
    assert len(facts)==2 and facts[0]['known_at']=='2024-09-01T12:00:00+00:00'
