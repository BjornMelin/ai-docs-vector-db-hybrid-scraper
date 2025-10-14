"""GPU validation harness for transformer and CUDA stacks.

The harness checks that PyTorch, xFormers, FlashAttention, vLLM, bitsandbytes,
Triton, and DeepSpeed are importable and able to execute minimal GPU workloads.
By default the script fails when a CUDA-capable device is missing; pass
``--allow-missing-gpu`` to surface a warning instead. Reports are emitted as
JSON for archival in CI jobs.

Usage
-----
    uv run python scripts/validation/gpu_validation.py --output artifacts/gpu.json
"""

# pylint: disable=duplicate-code
from __future__ import annotations

import argparse
import importlib
import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CheckResult:
    """Outcome of an individual GPU validation step."""

    name: str
    status: str
    detail: str | None = None


@dataclass(slots=True)
class ValidationReport:
    """Structured GPU validation artefact."""

    # pylint: disable=too-many-instance-attributes

    status: str
    python_version: str
    platform: str
    torch_version: str | None
    cuda_version: str | None
    device_count: int
    device_names: list[str]
    library_versions: dict[str, str | None] = field(default_factory=dict)
    checks: list[CheckResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the validation report into a serialisable mapping.

        Returns:
            Dict containing validation metadata and check results.
        """
        payload = asdict(self)
        payload["checks"] = [asdict(check) for check in self.checks]
        return payload


def _import_optional(name: str):
    """Import a module optionally, returning None if not found."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


def _tensor_exception_types(torch_module) -> tuple[type[BaseException], ...]:
    """Return tensor execution exceptions worth reporting explicitly."""
    exceptions: list[type[BaseException]] = [RuntimeError]

    out_of_memory = getattr(torch_module, "OutOfMemoryError", None)
    if isinstance(out_of_memory, type):
        exceptions.append(out_of_memory)

    cuda_module = getattr(torch_module, "cuda", None)
    cuda_error = getattr(cuda_module, "CudaError", None) if cuda_module else None
    if isinstance(cuda_error, type):
        exceptions.append(cuda_error)

    return tuple(exceptions)


def _torch_device_checks(
    torch_module, require_gpu: bool
) -> tuple[list[CheckResult], dict[str, Any]]:
    """Perform torch device checks and return results and metadata."""
    checks = []
    cuda_available = torch_module.cuda.is_available()
    device_count = torch_module.cuda.device_count() if cuda_available else 0
    cuda_version = getattr(torch_module.version, "cuda", None)
    torch_version = getattr(torch_module, "__version__", None)

    if not cuda_available:
        checks.append(
            CheckResult(
                name="torch-cuda",
                status="failed" if require_gpu else "warning",
                detail="CUDA device not detected",
            )
        )
        return checks, {
            "torch_version": torch_version,
            "cuda_version": cuda_version,
            "device_count": device_count,
            "device_names": [],
        }

    device_names = [torch_module.cuda.get_device_name(i) for i in range(device_count)]

    # Execute a minimal matrix multiply to confirm kernels are functional.
    try:
        device = torch_module.device("cuda")
        tensor_a = torch_module.randn(
            (256, 128), device=device, dtype=torch_module.float32
        )
        tensor_b = torch_module.randn(
            (128, 64), device=device, dtype=torch_module.float32
        )
        _ = torch_module.matmul(tensor_a, tensor_b)
        if hasattr(torch_module.cuda, "synchronize"):
            torch_module.cuda.synchronize()
        checks.append(CheckResult("torch-matmul", "passed", "GPU matmul succeeded"))
    except _tensor_exception_types(torch_module) as exc:  # pragma: no cover
        detail = f"{type(exc).__name__}: {exc}"
        checks.append(CheckResult("torch-matmul", "failed", detail))
    except Exception as exc:  # pragma: no cover - surfaced in tests
        detail = f"UnexpectedError[{type(exc).__name__}]: {exc}"
        checks.append(CheckResult("torch-matmul", "failed", detail))

    # Capture memory statistics for observability.
    try:
        allocated = getattr(torch_module.cuda, "memory_allocated", lambda *_: None)(0)
        reserved = getattr(torch_module.cuda, "memory_reserved", lambda *_: None)(0)
        detail = f"allocated={allocated} reserved={reserved}"
        checks.append(CheckResult("torch-memory", "passed", detail))
    except Exception as exc:  # pragma: no cover - surfaced in tests
        checks.append(CheckResult("torch-memory", "warning", str(exc)))

    return checks, {
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "device_count": device_count,
        "device_names": device_names,
    }


def _xformers_checks(torch_module, xformers_module, require_gpu: bool) -> CheckResult:
    """Perform xformers checks and return the result."""
    if xformers_module is None:
        status = "failed" if require_gpu else "warning"
        return CheckResult("xformers", status, "xformers not installed")

    ops = getattr(xformers_module, "ops", None)
    if ops is None or not hasattr(ops, "memory_efficient_attention"):
        return CheckResult(
            "xformers", "failed", "xformers.ops.memory_efficient_attention missing"
        )

    cuda_available = bool(torch_module.cuda.is_available())
    if not cuda_available:
        status = "failed" if require_gpu else "warning"
        return CheckResult(
            "xformers", status, "CUDA required for xformers attention test"
        )

    try:
        device = torch_module.device("cuda")
        dtype = getattr(torch_module, "float16", torch_module.float32)
        query = torch_module.randn((1, 4, 64), device=device, dtype=dtype)
        key = torch_module.randn((1, 4, 64), device=device, dtype=dtype)
        value = torch_module.randn((1, 4, 64), device=device, dtype=dtype)
        _ = ops.memory_efficient_attention(query, key, value)
        return CheckResult("xformers", "passed", "Memory efficient attention executed")
    except Exception as exc:  # pragma: no cover - surfaced in tests
        return CheckResult("xformers", "failed", str(exc))


def _flash_attention_check(require_gpu: bool) -> tuple[CheckResult, str | None]:
    """Check if flash attention is available and ready for use."""
    flash_module = _import_optional("flash_attn")
    interface_module = _import_optional("flash_attn.flash_attn_interface")
    if flash_module is None:
        status = "failed" if require_gpu else "warning"
        return (
            CheckResult("flash-attn", status, "flash_attn not installed"),
            None,
        )
    if interface_module is None:
        status = "failed" if require_gpu else "warning"
        return (
            CheckResult(
                "flash-attn",
                status,
                "flash_attn.flash_attn_interface missing",
            ),
            getattr(flash_module, "__version__", None),
        )
    return (
        CheckResult("flash-attn", "passed", "flash_attn kernels available"),
        getattr(flash_module, "__version__", None),
    )


def _import_only_check(
    name: str, module_name: str, *, require_gpu: bool
) -> tuple[CheckResult, str | None]:
    """Import ``module_name`` and return a check result and version metadata."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        status = "failed" if require_gpu else "warning"
        return CheckResult(name, status, f"{module_name} not installed"), None
    except Exception as exc:  # pragma: no cover - surfaced in integration runs
        status = "failed" if require_gpu else "warning"
        detail = f"{module_name} import error [{type(exc).__name__}]: {exc}"
        return CheckResult(name, status, detail), None

    return (
        CheckResult(name, "passed", f"{module_name} import succeeded"),
        getattr(module, "__version__", None),
    )


def run_gpu_validation(
    *,
    require_gpu: bool = True,
    torch_module=None,
    xformers_module=None,
) -> ValidationReport:
    """Execute the GPU validation harness and return a structured report.

    Args:
        require_gpu: Fail if GPU is unavailable (default True).
        torch_module: Optional mock torch module for testing.
        xformers_module: Optional mock xformers module for testing.

    Returns:
        Validation report with check results and metadata.
    """
    checks: list[CheckResult] = []
    library_versions: dict[str, str | None] = {}

    try:
        torch_module = torch_module or importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        return ValidationReport(
            status="failed",
            python_version=platform.python_version(),
            platform=platform.platform(),
            torch_version=None,
            cuda_version=None,
            device_count=0,
            device_names=[],
            library_versions={},
            checks=[CheckResult("torch-import", "failed", str(exc))],
        )

    torch_checks, metadata = _torch_device_checks(torch_module, require_gpu=require_gpu)
    checks.extend(torch_checks)

    try:
        xformers_module = xformers_module or importlib.import_module("xformers")
    except ModuleNotFoundError:
        xformers_module = None
    xformers_check = _xformers_checks(
        torch_module, xformers_module, require_gpu=require_gpu
    )
    checks.append(xformers_check)
    library_versions["xformers"] = (
        getattr(xformers_module, "__version__", None) if xformers_module else None
    )

    flash_check, flash_version = _flash_attention_check(require_gpu=require_gpu)
    checks.append(flash_check)
    library_versions["flash_attn"] = flash_version

    for name, module_name in (
        ("vllm", "vllm"),
        ("bitsandbytes", "bitsandbytes"),
        ("triton", "triton"),
        ("deepspeed", "deepspeed"),
    ):
        check, version = _import_only_check(name, module_name, require_gpu=require_gpu)
        checks.append(check)
        library_versions[name] = version

    if any(check.status == "failed" for check in checks):
        status = "failed"
    elif any(check.status == "warning" for check in checks):
        status = "warning"
    else:
        status = "passed"

    return ValidationReport(
        status=status,
        python_version=platform.python_version(),
        platform=platform.platform(),
        torch_version=metadata.get("torch_version"),
        cuda_version=metadata.get("cuda_version"),
        device_count=metadata.get("device_count", 0),
        device_names=metadata.get("device_names", []),
        library_versions=library_versions,
        checks=checks,
    )


def _write_report(path: Path | None, payload: dict[str, Any], indent: int) -> None:
    """Write the validation report to a file or stdout."""
    if path is None:
        print(json.dumps(payload, indent=indent))
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")


def main() -> int:
    """Main entry point for the GPU validation script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the GPU validation report as JSON.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation level for JSON output.",
    )
    parser.add_argument(
        "--allow-missing-gpu",
        action="store_true",
        help="Demote missing GPU errors to warnings (useful on CPU-only hosts).",
    )
    args = parser.parse_args()

    report = run_gpu_validation(require_gpu=not args.allow_missing_gpu)
    payload = report.to_dict()
    _write_report(args.output, payload, args.indent)

    if report.status == "failed":
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
