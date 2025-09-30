# Persistent Cache Overview

This repository ships a two-tier cache stack combining a disk-backed local cache
with the distributed DragonflyDB layer. The refactor consolidates key-handling,
persistence, and invalidation semantics so that cache behaviour is deterministic
and observable.

## Architecture

- **Local persistent layer** – `PersistentCacheManager` stores payloads under the
  configured `base_path` using SHA-256 key hashes. Directory sharding prevents
  large fan-out and allows safe deletion, even when many keys share a namespace.
- **Distributed layer** – `DragonflyCache` provides Redis-compatible operations
  and optional compression. Specialised caches (embeddings/search results) use
  the distributed layer for shared workloads.
- **Shared helpers** – `_bulk_delete.delete_in_batches` performs batched
  invalidation and is reused by both embedding and search caches to keep
  duplicate-code lint warnings at bay.

## Key determinism

`PersistentCacheManager` exposes helper constructors so keys are stable across
processes:

- `embedding_key(text, model, provider, dimensions=None)` adds the dimensions
  prefix when provided and hashes the payload to avoid leaking the raw text.
- `search_key(query, filters=None)` serialises filters with sorted keys before
  hashing, guaranteeing identical cache hits regardless of dictionary order.

These helpers are now covered by unit tests in `tests/services/cache/` to catch
regressions.

## Deletion semantics

`CacheManager.delete()` now ensures the hashed key used for the local store is
purged alongside the distributed copy. The new test
`test_cache_manager_delete_removes_hashed_local_entry` asserts that the on-disk
artifact is removed from the persistent cache path.

For large-scale invalidation the specialised caches call
`delete_in_batches(cache, keys, batch_size=100)`, providing consistent aggregation
metrics and avoiding `pylint R0801` duplicate-code flags.

## Empty search results

`SearchResultCache.set_search_results` caches empty lists so expensive neg hits
are short-circuited. The TTL obeys the `CacheConfig.ttl_search_results` (or the
popularity-adjusted value) which means negative caching is both bounded and
configurable. The test `test_search_cache_returns_cached_empty_results` verifies
that an empty list is returned from cache without re-hit on the backend stub.

## Configuration knobs

Key configuration lives under `CacheConfig` in `src/config/settings.py`:

- `enable_local_cache` enables the persistent layer (disk writes plus hashed
  keys).
- `local_max_size` and `local_max_memory_mb` bound the persistent layer; when the
  `memory_pressure_threshold` ratio is exceeded the LRU eviction loop runs.
- `ttl_search_results` governs both populated and empty search result entries.
- `cache_ttl_seconds` allows per-cache overrides; empty results reuse the
  `search_results` slot.

## Operational notes

- Persistent paths default to `cache/local` unless the manager is constructed
  with a custom `local_cache_path`.
- All deletion helpers swallow missing-key scenarios, so repeated deletes are
  safe.
- Optional monitoring imports are guarded; metrics wiring only activates when
  `src.services.monitoring.metrics` is installed.

Refer to the tests in `tests/services/cache/` for examples of interacting with
these components.
