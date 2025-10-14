"""Optional GPU helpers with lazy imports."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import cache
from importlib import import_module
from importlib.util import find_spec
from typing import Any, TypeVar, cast


logger = logging.getLogger(__name__)

__all__ = [
    "enable_tf32",
    "get_gpu_device",
    "get_gpu_memory_info",
    "get_gpu_stats",
    "get_torch_device",
    "is_gpu_available",
    "move_to_device",
    "optimize_gpu_memory",
    "safe_gpu_operation",
    "set_memory_fraction",
]

_T = TypeVar("_T")


@cache
def _load_torch() -> Any | None:
    """Load torch lazily and cache the module."""
    if find_spec("torch") is None:
        return None
    try:
        return import_module("torch")
    except Exception as exc:
        logger.info("Torch import failed: %s", exc)
        return None


def _optional_module_available(name: str) -> bool:
    """Return True when the optional module is importable."""
    return find_spec(name) is not None


def is_gpu_available() -> bool:
    """Return True when CUDA-capable devices are available."""
    torch = _load_torch()
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except RuntimeError:
        return False


def _device_count() -> int:
    torch = _load_torch()
    if torch is None:
        return 0
    try:
        return int(torch.cuda.device_count())
    except RuntimeError:
        return 0


def get_gpu_device(memory_required_gb: float | None = None) -> str | None:
    """Return a CUDA device identifier or ``None`` when unavailable."""
    if not is_gpu_available():
        return None
    torch = _load_torch()
    assert torch is not None  # For type-checkers; guarded above.

    if memory_required_gb is not None:
        for index in range(_device_count()):
            try:
                free_bytes, _ = torch.cuda.mem_get_info(index)
                available_gb = free_bytes / 1024**3
            except RuntimeError as exc:
                logger.warning("Failed to read memory info for cuda:%d: %s", index, exc)
                continue
            if available_gb >= memory_required_gb:
                return f"cuda:{index}"
        return None

    return "cuda:0" if _device_count() > 0 else None


def get_gpu_memory_info(device: str | None = None) -> dict[str, float]:
    """Return memory statistics for ``device`` in GiB."""
    torch = _load_torch()
    if torch is None or not is_gpu_available():
        return {"total": 0.0, "free": 0.0, "used": 0.0}

    target = device or get_gpu_device()
    if not target:
        return {"total": 0.0, "free": 0.0, "used": 0.0}

    try:
        index = int(target.split(":")[1]) if ":" in target else int(target)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    except (ValueError, RuntimeError) as exc:
        logger.warning("Failed to query GPU memory for %s: %s", target, exc)
        return {"total": 0.0, "free": 0.0, "used": 0.0}

    total_gb = total_bytes / 1024**3
    free_gb = free_bytes / 1024**3
    used_gb = total_gb - free_gb
    return {"total": total_gb, "free": free_gb, "used": used_gb}


def optimize_gpu_memory(device: str | None = None) -> None:
    """Release cached allocations for ``device``."""
    torch = _load_torch()
    if torch is None or not is_gpu_available():
        return
    target = device or get_gpu_device()
    if not target:
        return
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except RuntimeError as exc:
        logger.warning("Failed to optimize GPU memory for %s: %s", target, exc)


def safe_gpu_operation(
    func: Callable[..., _T],
    *args: Any,
    fallback: Callable[..., _T] | _T | None = None,
    **kwargs: Any,
) -> _T | None:
    """Execute ``func`` when GPUs are available; otherwise return ``fallback``."""
    if not is_gpu_available():
        if isinstance(fallback, Callable):
            return fallback(*args, **kwargs)
        return cast("_T | None", fallback)
    try:
        return func(*args, **kwargs)
    except RuntimeError as exc:
        logger.warning("GPU operation failed, using fallback: %s", exc)
        if isinstance(fallback, Callable):
            return fallback(*args, **kwargs)
        return cast("_T | None", fallback)


def get_torch_device(device: str | None = None) -> Any:
    """Return a torch device instance or raise ``RuntimeError`` when unavailable."""
    torch = _load_torch()
    if torch is None:
        msg = "PyTorch is not installed"
        raise RuntimeError(msg)
    target = device or get_gpu_device() or "cpu"
    return torch.device(target)


def move_to_device(obj: Any, device: str | None = None) -> Any:
    """Move ``obj`` to ``device`` when torch is available."""
    torch = _load_torch()
    if torch is None:
        return obj
    target = get_torch_device(device)
    try:
        return obj.to(target)
    except RuntimeError as exc:
        logger.warning("Failed to move object to %s: %s", target, exc)
        return obj


def get_gpu_stats() -> dict[str, Any]:
    """Return summary information about optional GPU integrations."""
    torch = _load_torch()
    available = is_gpu_available()
    optional_libraries = {
        "xformers": "xformers",
        "flash_attn": "flash_attn",
        "vllm": "vllm",
        "deepspeed": "deepspeed",
        "bitsandbytes": "bitsandbytes",
        "triton": "triton",
    }
    stats: dict[str, Any] = {
        "torch_available": torch is not None,
        "cuda_available": available,
        "device_count": _device_count(),
        "libraries": {
            key: _optional_module_available(module)
            for key, module in optional_libraries.items()
        },
        "devices": [],
    }
    if torch is not None and available:
        for index in range(_device_count()):
            name = torch.cuda.get_device_name(index)
            memory = get_gpu_memory_info(f"cuda:{index}")
            stats["devices"].append({"id": index, "name": name, "memory": memory})
    return stats


def enable_tf32() -> None:
    """Enable TensorFloat-32 optimisations when CUDA is present."""
    torch = _load_torch()
    if torch is None or not is_gpu_available():
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except RuntimeError as exc:
        logger.warning("Failed to enable TF32: %s", exc)


def set_memory_fraction(fraction: float, device: str | None = None) -> None:
    """Limit CUDA memory usage to ``fraction`` of the selected device."""
    if not 0.0 < fraction <= 1.0:
        msg = "fraction must be between 0 and 1"
        raise ValueError(msg)
    torch = _load_torch()
    if torch is None or not is_gpu_available():
        return
    target = device or get_gpu_device()
    if not target:
        return
    try:
        index = int(target.split(":")[1]) if ":" in target else int(target)
        torch.cuda.set_per_process_memory_fraction(fraction, index)
    except (ValueError, RuntimeError) as exc:
        logger.warning("Failed to set memory fraction for %s: %s", target, exc)
