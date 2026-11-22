"""Bounded ROCm/PyTorch smoke test for the DQN compute path."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HSA_ENABLE_DXG_DETECTION", "1")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from destroyer import DQNAgent, Game2048, encode  # noqa: E402


def main() -> None:
    total_started = time.perf_counter()

    def mark(message: str, started: float | None = None) -> None:
        elapsed = time.perf_counter() - (started if started is not None else total_started)
        print(f"[GPU smoke +{elapsed:6.2f}s] {message}", flush=True)

    requested = torch.device(os.environ.get("REIWA_DEVICE", "cuda"))
    if requested.type != "cuda":
        raise SystemExit(f"GPU smoke test requires a CUDA/HIP device, got {requested}.")

    hip_version = torch.version.hip
    if hip_version is None:
        raise SystemExit("This PyTorch build has no ROCm/HIP support; CPU fallback is forbidden.")
    if not hip_version.startswith("6.4."):
        raise SystemExit(
            f"PyTorch was built for HIP {hip_version}; this launcher targets system ROCm 6.4.2."
        )
    system_hsa = Path("/opt/rocm/lib/libhsa-runtime64.so.1")
    if not system_hsa.exists():
        raise SystemExit(f"ROCm 6.4.2 WSL runtime is missing: {system_hsa}")
    torch_hsa = Path(torch.__file__).resolve().parent / "lib" / "libhsa-runtime64.so"
    if torch_hsa.exists() and torch_hsa.resolve() != system_hsa.resolve():
        raise SystemExit(
            "PyTorch is loading a private HSA runtime instead of the system ROCm WSL runtime. "
            "See the ROCm 6.4.2 WSL runtime-library note in README.md."
        )
    if not torch.__version__.startswith("2.6.0+"):
        print(
            "WARNING: AMD's ROCm 6.4.2 WSL matrix officially validates PyTorch 2.6.0; "
            f"this environment has {torch.__version__}. Testing it anyway.",
            flush=True,
        )

    probe_started = time.perf_counter()
    mark("Calling torch.cuda.is_available(); first HIP initialization can be slow.")
    available = torch.cuda.is_available()
    mark(f"torch.cuda.is_available() -> {available}", probe_started)
    if not available:
        raise SystemExit("PyTorch ROCm is installed, but no WSL GPU is available.")

    init_started = time.perf_counter()
    mark("Initializing HIP device...")
    torch.cuda.init()
    mark("HIP device initialized.", init_started)

    name_started = time.perf_counter()
    device_name = torch.cuda.get_device_name(requested)
    print(f"HIP={hip_version} | GPU={device_name} | torch={torch.__version__}", flush=True)
    mark("Device enumeration completed.", name_started)

    # Exercise the same environment-driven device selection used by the GUI.
    os.environ.setdefault("REIWA_DEVICE", str(requested))
    os.environ.setdefault("REIWA_REQUIRE_ROCM", "1")
    agent_started = time.perf_counter()
    mark("Allocating DQN model and replay buffer on GPU...")
    agent = DQNAgent(batch=2, warmup=0, replay_capacity=32)
    mark(f"DQN initialized on {agent.device}.", agent_started)
    if agent.device != requested:
        raise SystemExit(f"DQN selected {agent.device} instead of requested GPU {requested}.")
    game = Game2048()
    state = encode(game.board)
    agent.replay.add(state, 0, 0.25, state, False, [True] * 4)
    agent.replay.add(state, 1, 0.50, state, True, [False] * 4)

    learn_started = time.perf_counter()
    mark("Running a real forward/backward/optimizer step...")
    loss = agent.learn(force=True)
    torch.cuda.synchronize(requested)
    mark("GPU optimizer step and synchronization completed.", learn_started)
    if loss is None:
        raise SystemExit("DQN GPU training step did not produce a loss value.")
    if not agent.online.layers[0].weight.is_cuda or not agent.replay.states.is_cuda:
        raise SystemExit("DQN parameters or replay tensors are not resident on the GPU.")
    print(f"GPU_SMOKE_OK | device={agent.device} | loss={loss:.6f}", flush=True)
    mark("All GPU checks passed.")


if __name__ == "__main__":
    main()
