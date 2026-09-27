"""Only a successfully committed feature publication may advance local caches."""
from pathlib import Path
import sys
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cfbpred import publish_feature_refresh as publish


def test_cache_matches_postgres_real_precision():
    frame = pd.DataFrame({'rating': [1.234567891, None], 'game_id': [1, 2]})
    cached = publish.cache_payload(frame, {'rating': 'real', 'game_id': 'bigint'})
    assert cached.rating.iloc[0] == float(pd.Series([frame.rating.iloc[0]]).astype('float32').iloc[0])
    assert pd.isna(cached.rating.iloc[1])
    assert frame.rating.iloc[0] == 1.234567891


@pytest.mark.parametrize('commit_fails', [False, True])
def test_publish_commit_precedes_cache_writes(tmp_path, monkeypatch, commit_fails):
    committed = []
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    for table in ('game_scalars', 'game_history'):
        pd.DataFrame({'game_id': [2]}).to_parquet(tmp_path / f'{table}.parquet')
        pd.DataFrame({'game_id': [1]}).to_parquet(cache_dir / f'{table}.parquet')
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, *args): pass
        def fetchall(self): return [('game_id', 'bigint')]
        def fetchone(self): return (1,)
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args):
            if commit_fails: raise RuntimeError('commit failed')
            committed.append(True)
        def cursor(self): return Cursor()
    monkeypatch.setattr(publish, 'FEAT', tmp_path)
    monkeypatch.setattr(publish, 'CACHE', cache_dir)
    monkeypatch.setattr(publish, 'dsn', lambda: '')
    monkeypatch.setattr(publish.psycopg, 'connect', lambda *args: Connection())
    monkeypatch.setattr(publish, 'copy_chunked', lambda *args: 1)
    original_save = publish.save_cache
    def save(frame, path):
        assert committed
        original_save(frame, path)
    monkeypatch.setattr(publish, 'save_cache', save)
    if commit_fails:
        with pytest.raises(RuntimeError, match='commit failed'): publish.main()
    else: publish.main()
    for table in ('game_scalars', 'game_history'):
        assert pd.read_parquet(cache_dir / f'{table}.parquet').game_id.tolist() == ([1] if commit_fails else [2])
