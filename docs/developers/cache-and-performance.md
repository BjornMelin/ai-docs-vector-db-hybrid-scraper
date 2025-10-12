# Cache and Performance

This guide covers the Dragonfly caching layer, GPU acceleration notes, and
performance checkpoints for the retrieval stack.

## 1. Distributed Cache

The platform now relies exclusively on Dragonfly (`DragonflyCache`) for all
runtime caching. Specialised helpers (embeddings, search results, browser cache)
share the same high-performance backend.

### Architecture

- **Dragonfly layer** – Handles all cache storage with optional compression,
  connection pooling, and SCAN-based invalidation.
- **Shared helpers** – `_bulk_delete.delete_in_batches` performs batched
  invalidation and keeps deletion metrics consistent.

### Key Helpers

- `build_embedding_cache_key(text, model, provider, dimensions=None)` –
  normalises and hashes embedding requests.
- `build_search_cache_key(query, filters=None)` – serialises filters with sorted
  keys before hashing for deterministic lookups.

Refer to tests in `tests/services/cache/` for coverage of hashing, deletion, and
batch invalidation.

### Configuration

Configured through `CacheConfig` (`src/config/models.py`):

| Setting | Description |
| --- | --- |
| `enable_caching` | Global kill switch for caching. |
| `enable_dragonfly_cache` | Enable or disable the Dragonfly layer. |
| `dragonfly_url` | Connection string for Dragonfly. |
| `ttl_embeddings`, `ttl_crawl`, `ttl_queries`, `ttl_search_results` | Default TTLs for common cache groups. |
| `cache_ttl_seconds` | Optional per-cache overrides keyed by `embeddings`, `collections`, `search_results`, or `queries`. |

Operational notes:

- Dragonfly keys are automatically prefixed; pattern clearing uses
  `<cache-type>:*` expressions.
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
