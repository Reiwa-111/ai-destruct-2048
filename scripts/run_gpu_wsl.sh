#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REIWA_ROCM_VENV:-$HOME/reiwa-rocm}"
PYTHON="$VENV/bin/python"

fail() {
  echo "[DESTROYER ROCm 6.4.2] ERROR: $*" >&2
  exit 2
}

[[ -e /dev/dxg ]] || fail "WSL GPU device /dev/dxg is missing. Check the Windows AMD driver and WSL update."
[[ -x "$PYTHON" ]] || fail "Python environment not found: $VENV"
[[ -d "$PROJECT/src/destroyer" ]] || fail "Project path not found: $PROJECT"
[[ -e /opt/rocm/lib/libhsa-runtime64.so.1 ]] || fail "System ROCm HSA runtime is missing under /opt/rocm/lib."
[[ -e /usr/lib/wsl/lib/libdxcore.so ]] || fail "WSL DXCore library is missing: /usr/lib/wsl/lib/libdxcore.so"

export HSA_ENABLE_DXG_DETECTION=1
export LD_LIBRARY_PATH="/opt/rocm/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export REIWA_DEVICE=cuda
export REIWA_REQUIRE_ROCM=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$PROJECT"

echo "[DESTROYER ROCm 6.4.2] Runtime diagnostics:"
"$PYTHON" "$PROJECT/scripts/rocm_wsl_diagnostics.py" \
  || fail "Python/ROCm versions or WSL runtime prerequisites are inconsistent."

echo "[DESTROYER ROCm 6.4.2] Running bounded DQN GPU smoke test..."
timeout --foreground 120s "$PYTHON" "$PROJECT/scripts/gpu_smoke.py" \
  || fail "ROCm GPU smoke test failed or exceeded 120 seconds. The GUI was not started and CPU fallback is disabled."

echo "[DESTROYER ROCm 6.4.2] GPU training step verified. Starting DESTROYER."
exec "$PYTHON" -m destroyer.app
