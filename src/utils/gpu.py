"""GPU utilities and helpers for optional GPU acceleration.

This module provides utilities for GPU detection, device management, and
memory optimization when GPU dependencies are available. All functions
are designed to gracefully degrade when GPU libraries are not installed.

Key Features:
- GPU availability detection
- Device selection and management
- Memory optimization helpers
- Safe imports with fallbacks
- Performance monitoring
"""

import logging
from typing import Any


logger = logging.getLogger(__name__)

# GPU availability flags - set during import
TORCH_AVAILABLE = False
CUDA_AVAILABLE = False
XFORMERS_AVAILABLE = False
FLASH_ATTENTION_AVAILABLE = False
VLLM_AVAILABLE = False
DEEPSPEED_AVAILABLE = False

# Attempt safe imports
try:
    import torch

    TORCH_AVAILABLE = True
    try:
        CUDA_AVAILABLE = torch.cuda.is_available()
        if CUDA_AVAILABLE:
            logger.info(f"CUDA available with {torch.cuda.device_count()} device(s)")
    except RuntimeError as e:
        logger.warning(f"CUDA detection failed: {e}")
        CUDA_AVAILABLE = False
except ImportError:
    torch = None
    logger.info("PyTorch not available - GPU features disabled")

try:
    import xformers

    XFORMERS_AVAILABLE = True
    logger.info("xFormers available for memory-efficient attention")
except ImportError:
    xformers = None

try:
    import flash_attn

    FLASH_ATTENTION_AVAILABLE = True
    logger.info("Flash Attention available for optimized transformers")
except ImportError:
    flash_attn = None

try:
    import vllm

    VLLM_AVAILABLE = True
    logger.info("vLLM available for fast LLM inference")
except ImportError:
    vllm = None

try:
    import deepspeed

    DEEPSPEED_AVAILABLE = True
    logger.info("DeepSpeed available for distributed training")
except ImportError:
    deepspeed = None


class GPUManager:
    """Manager for GPU device selection and memory optimization."""

    def __init__(self):
        self._device = None
        self._device_count = 0
        self._memory_info = {}

    @property
    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return TORCH_AVAILABLE and CUDA_AVAILABLE

    @property
    def device_count(self) -> int:
        """Get number of available GPU devices."""
        if not self.is_available:
            return 0
        if self._device_count == 0:
            self._device_count = torch.cuda.device_count()
        return self._device_count

    def get_optimal_device(self, memory_required_gb: float | None = None) -> str | None:
        """Get the optimal GPU device for the current workload.

        Args:
            memory_required_gb: Minimum GPU memory required in GB

        Returns:
            Device string (e.g., 'cuda:0') or None if no suitable device

        """
        if not self.is_available:
            return None

        if memory_required_gb is not None:
            # Find device with sufficient memory
            for i in range(self.device_count):
                device_props = torch.cuda.get_device_properties(i)
                available_memory_gb = device_props.total_memory / (1024**3)

                if available_memory_gb >= memory_required_gb:
                    device = f"cuda:{i}"
                    logger.info(
                        f"Selected {device} with {available_memory_gb:.1f}GB memory"
                    )
                    return device

            logger.warning(f"No GPU with {memory_required_gb}GB+ memory available")
            return None

        # Default to first available device
        device = "cuda:0"
        if self.device_count > 0:
            logger.info(f"Using default GPU device: {device}")
            return device

        return None

    def get_memory_info(self, device: str | None = None) -> dict[str, float]:
        """Get GPU memory information.

        Args:
            device: Device string (e.g., 'cuda:0'), defaults to current device

        Returns:
            Dictionary with memory information in GB

        """
        if not self.is_available:
            return {"total": 0.0, "used": 0.0, "free": 0.0}

        if device is None:
            device = self.get_optimal_device()

        if device and device.startswith("cuda:"):
            device_idx = int(device.split(":")[1])
            try:
                memory_info = torch.cuda.mem_get_info(device_idx)
                total_memory = torch.cuda.get_device_properties(device_idx).total_memory

                return {
                    "total": total_memory / (1024**3),
                    "free": memory_info[0] / (1024**3),
                    "used": (total_memory - memory_info[0]) / (1024**3),
                }
            except RuntimeError as e:
                logger.warning(f"Failed to get GPU memory info: {e}")

        return {"total": 0.0, "used": 0.0, "free": 0.0}

    def optimize_memory(self, device: str | None = None) -> None:
        """Optimize GPU memory usage.

        Args:
            device: Device to optimize, defaults to current device

        """
        if not self.is_available:
            return

        if device is None:
            device = self.get_optimal_device()

        if device and device.startswith("cuda:"):
            try:
                # Clear cache
                torch.cuda.empty_cache()

                # Synchronize to ensure operations complete
                torch.cuda.synchronize()

                logger.info(f"GPU memory optimized for {device}")
            except RuntimeError as e:
                logger.warning(f"GPU memory optimization failed: {e}")

    def set_device(self, device: str) -> None:
        """Set the active GPU device.

        Args:
            device: Device string (e.g., 'cuda:0')

        """
        if not self.is_available:
            logger.warning("GPU not available, cannot set device")
            return

        try:
            torch.cuda.set_device(device)
            self._device = device
            logger.info(f"Active GPU device set to {device}")
        except RuntimeError:
            logger.exception(f"Failed to set GPU device {device}")


# Global GPU manager instance
gpu_manager = GPUManager()


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return gpu_manager.is_available


def get_gpu_device(memory_required_gb: float | None = None) -> str | None:
    """Get optimal GPU device for workload.

    Args:
        memory_required_gb: Minimum memory required in GB

    Returns:
        Device string or None

    """
    return gpu_manager.get_optimal_device(memory_required_gb)


def get_gpu_memory_info(device: str | None = None) -> dict[str, float]:
    """Get GPU memory information.

    Args:
        device: Device string, defaults to optimal device

    Returns:
        Memory information dictionary

    """
    return gpu_manager.get_memory_info(device)


def optimize_gpu_memory(device: str | None = None) -> None:
    """Optimize GPU memory usage.

    Args:
        device: Device to optimize

    """
    gpu_manager.optimize_memory(device)


def safe_gpu_operation(func: callable, fallback: Any = None, *args, **kwargs) -> Any:
    """Execute function safely on GPU with fallback.

    Args:
        func: Function to execute
        fallback: Fallback value/function if GPU unavailable
        *args: Positional arguments for function
        **kwargs: Keyword arguments for function

    Returns:
        Function result or fallback

    """
    if not is_gpu_available():
        if callable(fallback):
            return fallback(*args, **kwargs)
        return fallback

    try:
        return func(*args, **kwargs)
    except RuntimeError as e:
        logger.warning(f"GPU operation failed, using fallback: {e}")
        if callable(fallback):
            return fallback(*args, **kwargs)
        return fallback


def get_torch_device(device: str | None = None) -> Any:
    """Get PyTorch device object.

    Args:
        device: Device string, defaults to optimal GPU or CPU

    Returns:
        PyTorch device object

    """
    if not TORCH_AVAILABLE:
        msg = "PyTorch not available"
        raise RuntimeError(msg)

    if device is None:
        gpu_device = get_gpu_device()
        device = gpu_device or "cpu"

    return torch.device(device)


def move_to_device(tensor_or_model: Any, device: str | None = None) -> Any:
    """Move tensor or model to specified device.

    Args:
        tensor_or_model: PyTorch tensor or model
        device: Target device

    Returns:
        Object moved to device

    """
    if not TORCH_AVAILABLE:
        return tensor_or_model

    target_device = get_torch_device(device)

    try:
        return tensor_or_model.to(target_device)
    except RuntimeError as e:
        logger.warning(f"Failed to move to device {target_device}: {e}")
        return tensor_or_model


def get_gpu_stats() -> dict[str, Any]:
    """Get comprehensive GPU statistics.

    Returns:
        Dictionary with GPU information

    """
    stats = {
        "gpu_available": is_gpu_available(),
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": CUDA_AVAILABLE,
        "xformers_available": XFORMERS_AVAILABLE,
        "flash_attention_available": FLASH_ATTENTION_AVAILABLE,
        "vllm_available": VLLM_AVAILABLE,
        "deepspeed_available": DEEPSPEED_AVAILABLE,
        "device_count": gpu_manager.device_count,
        "devices": [],
    }

    if stats["gpu_available"]:
        for i in range(gpu_manager.device_count):
            device_info = {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory": get_gpu_memory_info(f"cuda:{i}"),
            }
            stats["devices"].append(device_info)

    return stats


# Convenience functions for common GPU operations
def enable_tf32() -> None:
    """Enable TensorFloat-32 precision for faster computations."""
    if TORCH_AVAILABLE and CUDA_AVAILABLE:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 precision enabled for faster GPU computations")
        except RuntimeError as e:
            logger.warning(f"Failed to enable TF32: {e}")


def set_memory_fraction(fraction: float, device: str | None = None) -> None:
    """Set GPU memory fraction limit.

    Args:
        fraction: Memory fraction (0.0 to 1.0)
        device: Device string

    """
    if not TORCH_AVAILABLE or not CUDA_AVAILABLE:
        return

    if device is None:
        device = get_gpu_device()

    if device and device.startswith("cuda:"):
        try:
            torch.cuda.set_per_process_memory_fraction(
                fraction, int(device.split(":")[1])
            )
            logger.info(f"GPU memory fraction set to {fraction} for {device}")
        except RuntimeError as e:
            logger.warning(f"Failed to set GPU memory fraction: {e}")


# Initialize GPU optimizations on import
if TORCH_AVAILABLE and CUDA_AVAILABLE:
    enable_tf32()
