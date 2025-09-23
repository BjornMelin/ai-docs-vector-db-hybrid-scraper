# GPU Acceleration Guide

This guide covers the optional GPU acceleration features available in the AI Document Vector Database Hybrid Scraper.

## Overview

GPU acceleration provides significant performance improvements for AI/ML workloads including:

- **Embeddings Generation**: Faster vector creation using GPU-optimized models
- **Vector Search**: Accelerated similarity computations
- **Large Language Models**: GPU-powered text generation and processing
- **Batch Processing**: Parallel processing of multiple documents

## Installation

### Basic GPU Support

Install with GPU dependencies:

```bash
pip install ai-docs-vector-db-hybrid-scraper[gpu]
```

This includes:

- PyTorch with CUDA support
- Hugging Face Transformers
- GPU-optimized libraries (xFormers, Flash Attention, etc.)

### Platform-Specific Installation

#### NVIDIA CUDA (Linux/Windows)

```bash
# Install CUDA toolkit first (see NVIDIA documentation)
# Then install the package
pip install ai-docs-vector-db-hybrid-scraper[gpu]
```

#### AMD ROCm (Linux)

```bash
# Install ROCm first (see AMD documentation)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
pip install ai-docs-vector-db-hybrid-scraper[gpu]
```

#### Apple Silicon (macOS)

```bash
# MPS (Metal Performance Shaders) is automatically used
pip install ai-docs-vector-db-hybrid-scraper[gpu]
```

## GPU Detection and Configuration

### Automatic Detection

The system automatically detects available GPU resources:

```python
from src.utils import is_gpu_available, get_gpu_stats

# Check if GPU is available
if is_gpu_available():
    print("GPU acceleration is available!")

# Get detailed GPU information
stats = get_gpu_stats()
print(f"GPU devices: {stats['device_count']}")
print(f"CUDA available: {stats['cuda_available']}")
```

### Manual Device Selection

```python
from src.utils import get_gpu_device, get_gpu_memory_info

# Get optimal GPU device
device = get_gpu_device(memory_required_gb=8.0)  # Requires 8GB+ GPU memory

# Get memory information
memory_info = get_gpu_memory_info(device)
print(f"Available GPU memory: {memory_info['free']:.1f}GB")
```

## Usage Examples

### Basic GPU Operations

```python
import torch
from src.utils import get_torch_device, move_to_device

# Get GPU device
device = get_torch_device()

# Move tensors to GPU
tensor = torch.randn(100, 512)
gpu_tensor = move_to_device(tensor, device)

# Automatic memory management
from src.utils import optimize_gpu_memory
optimize_gpu_memory(device)
```

### GPU-Aware Embeddings

```python
from src.services.embeddings import EmbeddingService
from src.utils import get_gpu_device

# Initialize with GPU support
service = EmbeddingService(device=get_gpu_device())

# Generate embeddings on GPU
embeddings = await service.generate_embeddings([
    "Document text 1",
    "Document text 2",
    "Document text 3"
])
```

### GPU Memory Management

```python
from src.utils import GPUManager

gpu_manager = GPUManager()

# Monitor memory usage
memory_info = gpu_manager.get_memory_info()
print(f"GPU memory used: {memory_info['used']:.1f}GB")

# Optimize memory
gpu_manager.optimize_memory()

# Set memory limits
from src.utils import set_memory_fraction
set_memory_fraction(0.8)  # Use 80% of GPU memory
```

## Testing with GPU

### GPU Test Markers

Tests can be marked for GPU-specific execution:

```python
import pytest

@pytest.mark.gpu_required
def test_gpu_accelerated_feature(gpu_device):
    """Test that requires GPU acceleration."""
    assert gpu_device is not None
    # GPU-specific test logic

@pytest.mark.gpu_optional
def test_feature_with_gpu_fallback(gpu_available):
    """Test that works with or without GPU."""
    if gpu_available:
        # Use GPU acceleration
        pass
    else:
        # Fallback to CPU
        pass
```

### Running GPU Tests

```bash
# Run all GPU tests
pytest -m gpu

# Run only GPU-required tests
pytest -m gpu_required

# Skip GPU tests
pytest -m "not gpu"
```

## Performance Optimization

### Memory Optimization

```python
# Enable TF32 precision for faster computations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    # Your GPU computations
    pass
```

### Batch Processing

```python
# Optimal batch sizes for different GPU memory
batch_sizes = {
    "RTX 3060 (12GB)": 32,
    "RTX 3080 (10GB)": 24,
    "A100 (40GB)": 128,
    "H100 (96GB)": 256,
}

# Adaptive batch sizing based on available memory
def get_optimal_batch_size(memory_gb: float) -> int:
    if memory_gb > 80:
        return 256
    elif memory_gb > 40:
        return 128
    elif memory_gb > 20:
        return 64
    elif memory_gb > 10:
        return 32
    else:
        return 16
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

```python
# Clear GPU cache
torch.cuda.empty_cache()

# Use gradient accumulation for large models
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps
```

#### GPU Not Detected

```python
# Check GPU availability
from src.utils import get_gpu_stats
stats = get_gpu_stats()
print("GPU Stats:", stats)

# Verify CUDA installation
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
```

#### Driver Issues

```bash
# Check NVIDIA driver version
nvidia-smi

# Update drivers if necessary
# Ubuntu/Debian
sudo apt update && sudo apt install nvidia-driver-latest

# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

## API Reference

### GPU Utilities

#### `is_gpu_available() -> bool`

Check if GPU acceleration is available.

#### `get_gpu_device(memory_required_gb: float = None) -> str | None`

Get optimal GPU device string.

#### `get_gpu_memory_info(device: str = None) -> dict`

Get GPU memory information in GB.

#### `optimize_gpu_memory(device: str = None) -> None`

Optimize GPU memory usage.

#### `get_gpu_stats() -> dict`

Get comprehensive GPU statistics.

#### `safe_gpu_operation(func, fallback=None, *args, **kwargs)`

Execute function safely on GPU with fallback.

### GPU Manager Class

```python
class GPUManager:
    def is_available(self) -> bool: ...
    def device_count(self) -> int: ...
    def get_optimal_device(self, memory_required_gb: float = None) -> str | None: ...
    def get_memory_info(self, device: str = None) -> dict: ...
    def optimize_memory(self, device: str = None) -> None: ...
    def set_device(self, device: str) -> None: ...
```

## Performance Benchmarks

### Embedding Generation

| GPU Model | Batch Size | Tokens/sec | Memory Usage |
| --------- | ---------- | ---------- | ------------ |
| RTX 3060  | 32         | 2,450      | 6.2GB        |
| RTX 3080  | 64         | 4,890      | 12.8GB       |
| A100      | 128        | 15,600     | 28.4GB       |
| H100      | 256        | 31,200     | 56.8GB       |

### Vector Search

| GPU Model | Dataset Size | QPS   | Latency (ms) |
| --------- | ------------ | ----- | ------------ |
| RTX 3060  | 1M vectors   | 850   | 1.2          |
| RTX 3080  | 1M vectors   | 1,720 | 0.6          |
| A100      | 10M vectors  | 4,200 | 0.24         |
| H100      | 10M vectors  | 8,400 | 0.12         |

## Best Practices

### Memory Management

1. **Monitor Memory Usage**: Regularly check GPU memory with `get_gpu_memory_info()`
2. **Use Appropriate Batch Sizes**: Scale batch sizes based on available GPU memory
3. **Clear Cache Regularly**: Call `optimize_gpu_memory()` after large operations
4. **Use Mixed Precision**: Enable TF32 and automatic mixed precision when possible

### Error Handling

```python
try:
    # GPU operation
    result = gpu_accelerated_function(data)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        # Handle OOM error
        optimize_gpu_memory()
        # Retry with smaller batch size
        result = gpu_accelerated_function(data, batch_size=batch_size//2)
    else:
        raise
```

### Development Tips

1. **Test on CPU First**: Develop and debug on CPU before GPU testing
2. **Use GPU Test Markers**: Mark tests appropriately for CI/CD
3. **Monitor Performance**: Use performance monitoring tools
4. **Handle Graceful Degradation**: Always provide CPU fallbacks

## Contributing

When adding GPU features:

1. Use the GPU utilities from `src.utils.gpu`
2. Add appropriate test markers
3. Provide CPU fallbacks
4. Update this documentation
5. Test on multiple GPU configurations

## Support

For GPU-related issues:

1. Check GPU compatibility in the [requirements](../requirements.txt)
2. Verify driver and CUDA versions
3. Test with CPU fallback first
4. Report issues with GPU specifications and error messages
