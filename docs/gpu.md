# GPU training in WSL

The DQN path can use AMD ROCm through WSL. This is an environment-specific development setup, not a general GPU installation guide. N-tuple training, native C++ search, and Expectimax do not use the GPU.

## Local configuration

The tested setup is Windows 11 + WSL Ubuntu 24.04, AMD Radeon RX 7700 XT, system ROCm 6.4.2, and a PyTorch ROCm build. The current local environment uses PyTorch 2.9.1 with HIP 6.4.x. This PyTorch version is not the exact AMD-validated WSL wheel set, and the RX 7700 XT is not listed in the ROCm 6.4.2 WSL hardware matrix. Device enumeration and a real optimizer step were verified locally; this does not imply official vendor support.

The launcher targets the local environment, checks the PyTorch HIP version and runtime linkage, and refuses CPU fallback when GPU mode is explicitly requested. It may set WSL runtime variables but does not delete or replace runtime libraries.

The generic `requirements.txt` is for the standard Python runtime and does not install a ROCm-specific PyTorch build. This repository does not bootstrap or replace ROCm/PyTorch GPU packages. The GPU virtual environment must already contain a compatible ROCm-enabled PyTorch installation; set `REIWA_ROCM_VENV` to its path if it is not `$HOME/reiwa-rocm`.

## Smoke test

Run from Ubuntu WSL with the configured environment:

```bash
bash scripts/run_gpu_wsl.sh
```

The smoke test checks GPU visibility, device placement for the DQN and replay tensors, and performs an actual forward/backward/optimizer step. It is a bounded test and does not train a useful checkpoint.

For environment details, run the diagnostic script:

```bash
python scripts/rocm_wsl_diagnostics.py
```

## Limitations

- ROCm through WSL is experimental for this exact GPU/software combination.
- DQN GPU training does not accelerate the N-tuple or Expectimax hybrid.
- A successful smoke test proves only the tested environment and operations.
- The launcher intentionally fails if ROCm is unavailable instead of silently changing devices.
