"""Tests for the GPU validation harness."""

from __future__ import annotations

import types

import pytest

from scripts.validation import gpu_validation


class _FakeTensor:
    """A fake tensor class for testing purposes."""

    def __init__(self, shape: tuple[int, int]):
        self.shape = shape


class _FakeCudaNoGpu:
    """A fake CUDA module simulating no GPU availability."""

    def is_available(self) -> bool:  # pragma: no cover - trivial
        return False

    def device_count(self) -> int:  # pragma: no cover - trivial
        return 0


class _FakeTorchNoGpu:
    """A fake torch module simulating no GPU support."""

    __version__ = "2.5.1"
    version = types.SimpleNamespace(cuda=None)
    cuda = _FakeCudaNoGpu()
    float32 = "float32"
    float16 = "float16"


class _FakeCuda:
    """A fake CUDA module simulating GPU availability."""

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def get_device_name(self, index: int) -> str:
        return f"Fake GPU {index}"

    def synchronize(self) -> None:  # pragma: no cover - no-op
        return None

    def memory_allocated(self, index: int) -> int:
        return 0

    def memory_reserved(self, index: int) -> int:
        return 0


class _FakeTorchGpu:
    """A fake torch module simulating GPU support."""

    __version__ = "2.5.1"
    version = types.SimpleNamespace(cuda="12.4")

    def __init__(self) -> None:
        self.cuda = _FakeCuda()
        self.float32 = "float32"
        self.float16 = "float16"

    def device(self, name: str) -> types.SimpleNamespace:
        return types.SimpleNamespace(type=name)

    def randn(self, shape: tuple[int, int], *, device=None, dtype=None) -> _FakeTensor:
        _ = (device, dtype)
        return _FakeTensor(shape)

    def matmul(self, lhs: _FakeTensor, rhs: _FakeTensor) -> _FakeTensor:
        return _FakeTensor((lhs.shape[0], rhs.shape[-1]))


class _FakeXformers:
    """A fake xformers module for testing purposes."""

    __version__ = "0.0.32.post1"

    class Ops:  # pylint: disable=too-few-public-methods
        @staticmethod
        def memory_efficient_attention(query, key, value):  # pragma: no cover - trivial
            return query

    ops = Ops()


def test_gpu_validation_fails_without_gpu() -> None:
    """The harness should fail when CUDA is required but unavailable."""

    report = gpu_validation.run_gpu_validation(
        require_gpu=True,
        torch_module=_FakeTorchNoGpu(),
        xformers_module=None,
    )
    assert report.status == "failed"
    checks = {check.name: check for check in report.checks}
    assert checks["torch-cuda"].status == "failed"
    assert checks["flash-attn"].status == "failed"
    assert checks["vllm"].status == "failed"
    assert checks["bitsandbytes"].status == "failed"
    assert checks["triton"].status == "failed"
    assert checks["deepspeed"].status == "failed"


def test_gpu_validation_warns_when_allowing_missing_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opting into degraded mode should downgrade missing GPU errors to warnings."""

    original_import = gpu_validation.importlib.import_module

    def _fake_import(
        name: str, package: str | None = None
    ):  # pragma: no cover - deterministic
        if name == "xformers":
            raise ModuleNotFoundError("xformers missing")
        return original_import(name, package)  # type: ignore[call-arg]

    monkeypatch.setattr(gpu_validation.importlib, "import_module", _fake_import)

    report = gpu_validation.run_gpu_validation(
        require_gpu=False,
        torch_module=_FakeTorchNoGpu(),
        xformers_module=None,
    )
    assert report.status == "warning"
    checks = {check.name: check for check in report.checks}
    assert checks["torch-cuda"].status == "warning"
    assert checks["flash-attn"].status == "warning"
    assert checks["vllm"].status == "warning"
    assert checks["bitsandbytes"].status == "warning"
    assert checks["triton"].status == "warning"
    assert checks["deepspeed"].status == "warning"


def test_gpu_validation_succeeds_with_fake_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """The harness should pass when torch and xformers operations succeed."""

    fake_modules = {
        "flash_attn": types.SimpleNamespace(__version__="2.6.0"),
        "flash_attn.flash_attn_interface": object(),
        "vllm": types.SimpleNamespace(__version__="0.11.0"),
        "bitsandbytes": types.SimpleNamespace(__version__="0.44.0"),
        "triton": types.SimpleNamespace(__version__="3.0.0"),
        "deepspeed": types.SimpleNamespace(__version__="0.14.0"),
    }

    monkeypatch.setattr(
        gpu_validation,
        "_import_optional",
        lambda name: fake_modules.get(name),
    )

    report = gpu_validation.run_gpu_validation(
        require_gpu=True,
        torch_module=_FakeTorchGpu(),
        xformers_module=_FakeXformers(),
    )
    assert report.status == "passed"
    checks = {check.name: check for check in report.checks}
    assert checks["xformers"].status == "passed"
    assert checks["flash-attn"].status == "passed"
    assert checks["vllm"].status == "passed"
    assert checks["bitsandbytes"].status == "passed"
    assert checks["triton"].status == "passed"
    assert checks["deepspeed"].status == "passed"
    assert report.library_versions["flash_attn"] == "2.6.0"
    assert report.library_versions["vllm"] == "0.11.0"
    assert report.library_versions["bitsandbytes"] == "0.44.0"
    assert report.library_versions["triton"] == "3.0.0"
    assert report.library_versions["deepspeed"] == "0.14.0"
