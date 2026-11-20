#!/usr/bin/env bash
set -euo pipefail

VENV="${REIWA_ROCM_VENV:-$HOME/reiwa-rocm}"
PROJECT="/mnt/q/PROJECTS/2048-DESTROYER"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "ROCm Python venv not found: $VENV" >&2
  exit 2
fi
if [[ ! -d "$PROJECT/src/destroyer" ]]; then
  echo "Project path not found: $PROJECT" >&2
  exit 2
fi

source "$VENV/bin/activate"
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('PyTorch ROCm GPU is unavailable; refusing silent CPU fallback.')
print(f'GPU: {torch.cuda.get_device_name(0)} | PyTorch {torch.__version__}')
PY
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT"
python "$PROJECT/scripts/gpu_smoke.py"
exec python -m destroyer.app
