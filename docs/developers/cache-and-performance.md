# Cache and Performance

This guide covers the two-tier caching system, GPU acceleration notes, and
performance checkpoints for the retrieval stack.

## 1. Persistent Cache

The platform ships with a local persistent cache (`PersistentCacheManager`) and a
Dragonfly-backed distributed cache (`DragonflyCache`).

### Architecture

- **Local layer** – Stores payloads under a hashed directory structure (SHA-256
  truncated to 16 hex characters) to avoid leaking raw keys and to prevent
  directory fan-out. Safe to delete by namespace without disturbing neighbours.
- **Distributed layer** – DragonflyDB handles shared caches with optional
  compression. Specialised caches (embeddings, search results) reuse this layer
  for read-through performance.
- **Shared helpers** – `_bulk_delete.delete_in_batches` performs batched
  invalidation and keeps deletion metrics consistent.

### Key Helpers

- `embedding_key(text, model, provider, dimensions=None)` – normalises and hashes
  embedding requests.
- `search_key(query, filters=None)` – serialises filters with sorted keys before
  hashing for deterministic lookups.

Refer to tests in `tests/unit/services/cache/` for coverage of hashing, deletion,
negative caching, and batch invalidation.

### Configuration

Configured through `CacheConfig` (`src/config/models.py`):

| Setting | Description |
| --- | --- |
| `enable_local_cache` | Enable/disable the on-disk layer. |
| `local_max_size`, `local_max_memory_mb` | Bound the persistent layer; exceeding the threshold triggers LRU eviction. |
| `ttl_search_results` | Governs populated and empty search result entries. |
| `cache_ttl_seconds` | Per-cache overrides; empty results reuse the search slot. |
| `memory_pressure_threshold` | Triggers eviction when local cache usage grows too high. |

Operational notes:

- Default path is `cache/local`; override via `PersistentCacheManager(local_cache_path=...)`.
- Directories are hardened to `0700` on POSIX hosts.
- Deletion helpers swallow missing-key errors for idempotency.

## 2. GPU Acceleration (Optional)

GPU support remains optional; most workflows run on CPU. When GPUs are present,
utilities in `src.utils.gpu` help detect and manage devices.

### Detection & Device Selection

```python
from src.utils import is_gpu_available, get_gpu_device

if is_gpu_available():
    device = get_gpu_device()
```

`get_gpu_memory_info()` exposes free/used memory for adaptive batching.

### Batch Sizing (Reference)

| Device | Suggested Batch | Notes |
| --- | --- | --- |
| RTX 3060 (12GB) | 32 | Default local profile hardware. |
| RTX 3080 (10GB) | 24 | Reduce when HyDE enabled. |
| A100 (40GB) | 128 | Use mixed precision for best throughput. |
| H100 (96GB) | 256 | Large-scale ingestion. |

Utilities such as `optimize_gpu_memory()` and `GPUManager.optimize_memory()` help
reclaim VRAM. Always implement CPU fallbacks and mark GPU-only tests with
`@pytest.mark.gpu_required`.

## 3. Performance Checklist

- Monitor cache hit ratios (see `docs/observability/query_processing_metrics.md`).
- Before changing cache or GPU settings run:

```bash
uv run pytest tests/unit/services/cache/ -q
uv run pytest tests/unit/services/query_processing/test_pipeline.py -q
```

- For GPU changes, run targeted suites:

```bash
uv run pytest -m gpu_required
```

- Track dependency updates via the compatibility matrix (now summarised in
  `docs/developers/platform-operations.md`).

Keep this document aligned with cache utilities and GPU helper modules whenever
behaviour changes.
