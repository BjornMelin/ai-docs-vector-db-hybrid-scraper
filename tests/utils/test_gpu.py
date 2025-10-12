"""Tests for optional GPU helper utilities."""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any

import pytest

from src.utils import gpu


class _DummyCuda:
    """Minimal CUDA shim used to emulate torch.cuda."""

    def __init__(self) -> None:
        self.device_stats = [
            {
                "name": "Dummy GPU 0",
                "free": float(6 * 1024**3),
                "total": float(8 * 1024**3),
            }
        ]
        self.cache_cleared = False
        self.synchronized = False
        self.mem_fraction: tuple[float, int] | None = None

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return len(self.device_stats)

    def mem_get_info(self, index: int) -> tuple[int, int]:
        stats = self.device_stats[index]
        return int(stats["free"]), int(stats["total"])

    def empty_cache(self) -> None:
        self.cache_cleared = True

    def synchronize(self) -> None:
        self.synchronized = True

    def get_device_name(self, index: int) -> str:
        return self.device_stats[index]["name"]

    def set_per_process_memory_fraction(self, fraction: float, index: int) -> None:
        self.mem_fraction = (fraction, index)


class _DummyTorch:
    """Simplified torch module exposing CUDA helpers."""

    def __init__(self, cuda: _DummyCuda) -> None:
        self.cuda = cuda
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(allow_tf32=False),
        )
        self.requested_devices: list[str] = []

    def device(self, name: str) -> str:
        self.requested_devices.append(name)
        return f"device({name})"


@pytest.fixture(autouse=True)
def _clear_torch_cache() -> Generator[None, None, None]:
    """Reset lazy torch loader cache before and after each test."""

    with contextlib.suppress(AttributeError):  # pragma: no cover - patched loader
        gpu._load_torch.cache_clear()  # type: ignore[attr-defined]
    yield
    with contextlib.suppress(AttributeError):  # pragma: no cover - patched loader
        gpu._load_torch.cache_clear()  # type: ignore[attr-defined]


@pytest.fixture
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> tuple[_DummyTorch, _DummyCuda]:
    """Provide a patched torch module backed by dummy CUDA stats."""

    cuda = _DummyCuda()
    torch_module = _DummyTorch(cuda)

    available_modules = {
        "torch": True,
        "xformers": True,
        "flash_attn": False,
        "vllm": False,
        "deepspeed": True,
    }

    def fake_find_spec(name: str) -> Any:
        return object() if available_modules.get(name, False) else None

    monkeypatch.setattr(gpu, "find_spec", fake_find_spec)

    @functools.cache
    def loader() -> _DummyTorch:
        return torch_module

    monkeypatch.setattr(gpu, "_load_torch", loader, raising=False)
    return torch_module, cuda


def test_is_gpu_unavailable_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return falsey GPU status when torch cannot be imported."""

    monkeypatch.setattr(gpu, "find_spec", lambda name: None, raising=False)
    assert not gpu.is_gpu_available()
    assert gpu.get_gpu_device() is None
    assert gpu.get_gpu_memory_info() == {"total": 0.0, "free": 0.0, "used": 0.0}


def test_get_gpu_device_with_memory_requirements(
    fake_torch: tuple[_DummyTorch, _DummyCuda],
) -> None:
    """Select the first CUDA device that satisfies the memory budget."""

    _torch, cuda = fake_torch

    assert gpu.get_gpu_device(memory_required_gb=4.0) == "cuda:0"

    cuda.device_stats[0]["free"] = float(1 * 1024**3)
    assert gpu.get_gpu_device(memory_required_gb=4.0) is None


def test_get_gpu_memory_info_returns_expected_shape(
    fake_torch: tuple[_DummyTorch, _DummyCuda],
) -> None:
    """Convert CUDA memory info into a floating-point dictionary."""

    expected = gpu.get_gpu_memory_info("cuda:0")
    assert pytest.approx(expected["total"], rel=1e-6) == 8.0
    assert pytest.approx(expected["free"], rel=1e-6) == 6.0
    assert pytest.approx(expected["used"], rel=1e-6) == 2.0


def test_optimize_gpu_memory_triggers_cuda_calls(
    fake_torch: tuple[_DummyTorch, _DummyCuda],
) -> None:
    """Ensure cleanup operations forward to torch.cuda."""

    _torch, cuda = fake_torch
    gpu.optimize_gpu_memory()

    assert cuda.cache_cleared
    assert cuda.synchronized


def test_safe_gpu_operation_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback callable executes when GPU is unavailable."""

    monkeypatch.setattr(gpu, "is_gpu_available", lambda: False)
    sentinel = object()
    result = gpu.safe_gpu_operation(lambda: 1 / 0, fallback=lambda: sentinel)
    assert result is sentinel


def test_safe_gpu_operation_handles_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime errors trigger fallback values when GPUs fail."""

    monkeypatch.setattr(gpu, "is_gpu_available", lambda: True)

    def blow_up() -> None:
        raise RuntimeError("cuda failed")

    fallback = "cpu"
    assert gpu.safe_gpu_operation(blow_up, fallback=fallback) == "cpu"


def test_get_torch_device_requires_installed_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing torch installation raises a runtime error."""

    monkeypatch.setattr(gpu, "find_spec", lambda name: None, raising=False)

    with pytest.raises(RuntimeError, match="PyTorch is not installed"):
        gpu.get_torch_device()


def test_move_to_device_without_torch_returns_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Objects remain untouched when torch cannot be imported."""

    monkeypatch.setattr(gpu, "find_spec", lambda name: None, raising=False)
    payload: dict[str, str] = {"value": "cpu"}
    assert gpu.move_to_device(payload) is payload


def test_move_to_device_invokes_tensor_to(
    fake_torch: tuple[_DummyTorch, _DummyCuda],
) -> None:
    """Delegates to the tensor's ``to`` method with resolved device."""

    torch_module, _cuda = fake_torch

    class DummyTensor:
        def __init__(self) -> None:
            self.targets: list[str] = []

        def to(self, device: Any):
            self.targets.append(str(device))
            return self

    tensor = DummyTensor()
    result = gpu.move_to_device(tensor, device="cuda:0")

    assert result is tensor
    assert torch_module.requested_devices[-1] == "cuda:0"
    assert tensor.targets[-1] == "device(cuda:0)"


def test_get_gpu_stats_reports_optional_modules(
    fake_torch: tuple[_DummyTorch, _DummyCuda],
) -> None:
    """Summarise GPU environment and optional library availability."""

    stats = gpu.get_gpu_stats()

    assert stats["torch_available"] is True
    assert stats["cuda_available"] is True
    assert stats["device_count"] == 1
    assert stats["libraries"]["xformers"] is True
    assert stats["libraries"]["flash_attn"] is False
    assert stats["devices"][0]["name"] == "Dummy GPU 0"


def test_enable_tf32_sets_backend_flags(
    fake_torch: tuple[_DummyTorch, _DummyCuda],
) -> None:
    """Flag toggles propagate to torch backend namespaces."""

    torch_module, _cuda = fake_torch
    gpu.enable_tf32()

    assert torch_module.backends.cuda.matmul.allow_tf32 is True
    assert torch_module.backends.cudnn.allow_tf32 is True


def test_set_memory_fraction_records_fraction(
    fake_torch: tuple[_DummyTorch, _DummyCuda],
) -> None:
    """Record per-process memory fraction against the dummy CUDA shim."""

    _torch, cuda = fake_torch
    gpu.set_memory_fraction(0.5)
    assert cuda.mem_fraction == (0.5, 0)


def test_set_memory_fraction_validates_range() -> None:
    """Reject fractions outside the open interval ``(0, 1]``."""

    with pytest.raises(ValueError, match="between 0 and 1"):
        gpu.set_memory_fraction(0.0)
