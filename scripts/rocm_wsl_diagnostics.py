"""Inspect the ROCm 6.4.2 WSL runtime without probing or initializing the GPU."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

os.environ.setdefault('HSA_ENABLE_DXG_DETECTION', '1')

import numpy as np
import torch


def _describe(path: Path) -> str:
    if not path.exists():
        return f"MISSING: {path}"
    try:
        return f"{path} -> {path.resolve()}"
    except OSError as error:
        return f"{path} (resolve failed: {error})"


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch HIP build: {torch.version.hip}")
    print(f"NumPy: {np.__version__}")
    print(f"HSA_ENABLE_DXG_DETECTION: {os.environ.get('HSA_ENABLE_DXG_DETECTION', '<unset>')}")
    print(f"/dev/dxg: {'present' if Path('/dev/dxg').exists() else 'MISSING'}")
    print(f"/dev/kfd: {'present' if Path('/dev/kfd').exists() else 'not exposed (normal for WSL)'}")

    rocm_root = Path('/opt/rocm')
    print(_describe(rocm_root))
    for candidate in (
        rocm_root / 'lib' / 'libhsa-runtime64.so.1',
        rocm_root / 'lib' / 'libhsa-runtime64.so',
        Path('/usr/lib/wsl/lib/libdxcore.so'),
    ):
        print(_describe(candidate))

    torch_lib = Path(torch.__file__).resolve().parent / 'lib'
    print(f"PyTorch lib directory: {torch_lib}")
    torch_hsa = sorted(torch_lib.glob('libhsa-runtime64.so*'))
    if torch_hsa:
        for candidate in torch_hsa:
            print(_describe(candidate))
    else:
        print('No bundled libhsa-runtime64.so in torch/lib')

    if torch.version.hip is None:
        print('STATUS: PyTorch is not a ROCm build')
        return 2
    if not str(torch.version.hip).startswith('6.4.'):
        print('STATUS: PyTorch HIP runtime is not ROCm 6.4.x')
        return 2
    if not Path('/dev/dxg').exists():
        print('STATUS: WSL DXG device is not exposed')
        return 2

    print('STATUS: version and WSL device prerequisites look plausible; GPU probe not run')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
