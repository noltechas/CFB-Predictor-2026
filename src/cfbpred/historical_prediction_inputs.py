"""Real-game scalar inputs reusing immutable, superset player stores.

Ranking stores contain terminal targets. They can supply player observations,
but those targets must never count as real team appearances in game features.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import pandas as pd
from cfbpred.pregame_experiments import save_json, sha


def remap_indices(frame, columns, current, saved):
    inverse = {int(value): key for key, value in current.items()}
    mapping = {value: int(saved.get(name, 0)) for value, name in inverse.items()}
    mapping[0] = 0
    out = frame.copy()
    for col in columns:
        out[col] = out[col].map(mapping).fillna(0).astype('int64')
    return out


def build(schedule_path: Path, player_source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    raw = output / 'raw_features'; raw.mkdir()
    paths = json.loads(player_source.read_text())
    schedule = pd.read_parquet(schedule_path)
    if schedule.game_id.le(0).any() or schedule.game_id.duplicated().any():
        raise ValueError('Real prediction inputs require unique positive fixtures')
    schedule['kick'] = pd.to_datetime(schedule.kick, utc=True)
    schedule.to_parquet(raw / 'snapshot_schedule.parquet', index=False)
    schedule.to_parquet(raw / 'games.parquet', index=False)
    shutil.copy2(Path(paths['inputs']) / 'market.parquet', raw / 'market.parquet')
    import cfbpred.features_raw as fr
    old = fr.PQ, fr.FEAT, fr.load_games
    def loader(con, fbs_only=True):
        frame = schedule
        if fbs_only:
            frame = frame[frame.home_division.isin(['fbs','fcs']) | frame.away_division.isin(['fbs','fcs'])]
        return frame.sort_values(['season','kick','game_id']).reset_index(drop=True).copy()
    fr.PQ, fr.FEAT, fr.load_games = raw, raw, loader
    try:
        fr.main()
    finally:
        fr.PQ, fr.FEAT, fr.load_games = old
    from cfbpred.recent_form import build as recent
    recent(raw)
    # Reuse the player store's identity vocabulary. Only token numbering
    # changes; a team absent from the real corpus remains the UNK token.
    current = json.loads((raw / 'team_vocab.json').read_text())
    saved = json.loads((Path(paths['inputs']) / 'team_vocab.json').read_text())
    scalars = pd.read_parquet(raw / 'game_scalars.parquet')
    scalars = remap_indices(scalars, ['h_idx','a_idx'], current['team'], saved['team'])
    scalars = remap_indices(scalars, ['h_conf','a_conf'], current['conf'], saved['conf'])
    scalars.to_parquet(raw / 'game_scalars.parquet', index=False)
    history = remap_indices(pd.read_parquet(raw / 'game_history.parquet'), ['opp_idx'], current['team'], saved['team'])
    history.to_parquet(raw / 'game_history.parquet', index=False)
    shutil.copy2(Path(paths['inputs']) / 'team_vocab.json', raw / 'team_vocab.json')
    # Keep the immutable superset schedule required by the player/sequence
    # lineage check. Scalar targets and observations above use only real games.
    (raw / 'snapshot_schedule.parquet').rename(raw / 'real_feature_schedule.parquet')
    shutil.copy2(Path(paths['inputs']) / 'snapshot_schedule.parquet', raw / 'snapshot_schedule.parquet')
    from cfbpred.prepare_player_data import prepare
    inputs = output / 'inputs'
    prepare(raw, inputs, Path(paths['players']), rebuild_players=False)
    paths['inputs'] = str(inputs.resolve())
    paths['real_feature_schedule'] = str((raw / 'real_feature_schedule.parquet').resolve())
    source = output / 'source.json'; save_json(source, paths)
    save_json(output / 'manifest.json', dict(complete=True, source_sha256=sha(source),
        real_schedule_sha256=sha(schedule_path), player_source_sha256=sha(player_source),
        virtual_rows_used_for_scalar_observations=False, player_stores_reused=True,
        target_games=len(scalars), current_vocabulary=current, saved_vocabulary_sha256=sha(raw / 'team_vocab.json')))
    return source


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('schedule','player-source','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(build(a.schedule,a.player_source,a.output))
