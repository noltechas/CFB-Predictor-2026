#!/bin/zsh
# Full retrain for the 2026 season: 16 production seeds on 2014-2025, then
# ensemble metrics, evaluation, predictions for every game (2026 schedule
# included), and publish. Expects the feature store already rebuilt and the
# Supabase-backed training cache already refreshed (refresh_week.sh steps 1-4
# or: features_raw.py && publish.py --raw-only && source.py).
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a

# delta=True is the v21 architecture (state tower predicts REMAINING margin
# change / points, shifted to final space by the live score). It is NOT a
# default in train(), and this line omitted it until 2026-08-30 — running the
# script as written silently retrained the pre-v21 heads and threw the
# endgame fix away. Keep these flags in sync with the shipped checkpoints
# (torch.load any two_tower_raw_s*.pt and read s2d_gate/pre_hidden/delta).
echo "== training 16 seeds (v21 architecture: s2d gate + delta heads, 2014-2025) =="
python3 -c "import sys; sys.path.insert(0,'src'); from cfbpred.train import train; [train(seed=k, tag=f'raw_s{k}', s2d_gate=True, delta=True) for k in range(16)]"

echo "== ensemble metrics =="
python3 src/cfbpred/ensemble.py

echo "== evaluation harness =="
python3 src/cfbpred/evaluate.py ensemble

echo "== score every game and play =="
python3 src/cfbpred/predict.py

echo "== model card =="
python3 src/cfbpred/model_card.py

echo "== publish =="
python3 src/cfbpred/publish.py
python3 src/cfbpred/publish_lines.py   # ALWAYS after publish.py (view cascade)

echo "done ($(date))"
