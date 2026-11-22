import pytest
import torch

from destroyer.agent import _resolve_device


def test_device_defaults_to_cpu_when_no_accelerator_is_requested(monkeypatch):
    monkeypatch.delenv("REIWA_DEVICE", raising=False)
    assert _resolve_device() == torch.device("cpu")


def test_explicit_gpu_request_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Refusing to silently fall back"):
        _resolve_device("cuda")


def test_explicit_gpu_probe_exception_is_reported(monkeypatch):
    def broken_probe():
        raise RuntimeError("simulated HIP failure")

    monkeypatch.setattr(torch.cuda, "is_available", broken_probe)
    with pytest.raises(RuntimeError, match="runtime probe failed"):
        _resolve_device("cuda")


def test_rocm_mode_rejects_non_rocm_pytorch_build(monkeypatch):
    monkeypatch.setenv("REIWA_REQUIRE_ROCM", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(RuntimeError, match="not compiled with ROCm/HIP"):
        _resolve_device("cuda")


def test_rocm_mode_accepts_available_rocm_device(monkeypatch):
    monkeypatch.setenv("REIWA_REQUIRE_ROCM", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.4.43484-123eb5128")
    assert _resolve_device("cuda") == torch.device("cuda")


def test_environment_selects_rocm_642_for_gui_default(monkeypatch):
    monkeypatch.setenv("REIWA_DEVICE", "cuda")
    monkeypatch.setenv("REIWA_REQUIRE_ROCM", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.4.43484-123eb5128")
    assert _resolve_device() == torch.device("cuda")
