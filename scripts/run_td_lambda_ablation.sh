#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0" )/.." && pwd)"
cd "$ROOT"
WAIT_PID="${1:-}"
WAIT_MATCH="${2:-ntuple_fresh_continue1m5.npz}"

if [[ -n "$WAIT_PID" ]]; then
  echo "Waiting for training PID $WAIT_PID ($WAIT_MATCH) to finish..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    CMDLINE="$(ps -p "$WAIT_PID" -o args= 2>/dev/null || true)"
    if [[ "$CMDLINE" != *"$WAIT_MATCH"* ]]; then break; fi
    sleep 20
  done
fi

VENV="${REIWA_ROCM_VENV:-$HOME/reiwa-rocm}"
PYTHON="${PYTHON:-$VENV/bin/python}"
BASE="models/ntuple_fresh_continue1m.npz"
SEED=20261126
COMMON=(--episodes 150000 --seed "$SEED" --alpha 0.04 --gamma 1.0
        --epsilon-start 0.001 --epsilon-end 0.0001
        --checkpoint-every 50000 --log-every 10000
        --eval-every 50000 --eval-games 25 --eval-seed 7000000
        --resume "$BASE" --pool-capacity 10000 --restart-prob 0.0)

for spec in "00:0.0" "03:0.3" "07:0.7"; do
  label="${spec%%:*}"
  lambda="${spec##*:}"
  checkpoint="models/ntuple_td_lambda${label}_150k.npz"
  pool="models/ntuple_td_lambda${label}_pool.npz"
  log="reports/ntuple_td_lambda${label}_150k.log"
  report="reports/ntuple_td_lambda${label}_greedy_100_seed5000000.json"
  echo "=== TD(lambda=$lambda) ==="
  "$PYTHON" scripts/train_ntuple.py "${COMMON[@]}" --td-lambda "$lambda" \
    --output "$checkpoint" --position-pool "$pool" > "$log" 2>&1
  tail -n 8 "$log"
  "$PYTHON" scripts/benchmark_ntuple.py --checkpoint "$checkpoint" \
    --games 100 --seed 5000000 --output "$report"
done

echo "TD(lambda) ablation finished: $(date -Is)" | tee reports/td_lambda_ablation_complete.txt
