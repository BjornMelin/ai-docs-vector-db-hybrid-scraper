"""Cache manager combining in-memory storage with disk persistence.

Entries are held in an LRU-ordered in-memory map and optionally persisted to
disk with pickle serialisation and gzip compression. TTL metadata propagates
through the write and read paths so expired entries are rejected consistently.

Behavioural guarantees provided by this module:

* ``set`` resolves TTL values and records ``expires_at`` in memory and on disk.
* Lock scope stops at metadata copy; persistence work happens outside locks.
* Delete calls always attempt to remove the on-disk record when persistence is
  enabled.
* ``cache_path_for_key`` centralises deterministic path mapping for all disk
  operations.
* :class:`CacheStats` tracks hit-rate information, compression effectiveness,
  and mean access latency.
* ``warm_one_key`` provides an opt-in loader hook with sampled logging.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import pickle
import random
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal


logger = logging.getLogger(__name__)


SERIALIZATION_VERSION: Final[int] = 1
MIN_COMPRESSION_BENEFIT: Final[float] = 0.05  # 5% minimum savings before gzip
DEFAULT_WARM_SAMPLE_RATE: Final[float] = 0.01


@dataclass(slots=True)
class CacheStats:
    """Aggregate cache statistics with lightweight derived metrics.

    Attributes:
        hits: Total successful lookups across memory and persistence.
        misses: Failed lookups after exhausting both tiers.
        sets: Successful ``set`` operations.
        deletes: Successful delete operations.
        errors: Number of exceptions handled safely by the cache.
        disk_reads: Number of persisted entries hydrated into memory.
        disk_writes: Number of persisted writes performed.
        disk_deletes: Number of persisted deletes performed.
        compression_ratio: Running mean compression ratio (payload/original).
        avg_access_time_ms: Exponentially weighted moving average of access
            latency in milliseconds (hits only).
    """

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    disk_reads: int = 0
    disk_writes: int = 0
    disk_deletes: int = 0
    compression_ratio: float = 0.0
    avg_access_time_ms: float = 0.0

    def record_hit(self, latency_ms: float, compression_ratio: float) -> None:
        """Update hit counters and running averages."""

        self.hits += 1
        alpha = 0.2
        if self.hits == 1:
            self.avg_access_time_ms = latency_ms
            self.compression_ratio = compression_ratio
            return
        self.avg_access_time_ms = (
            alpha * latency_ms + (1 - alpha) * self.avg_access_time_ms
        )
        self.compression_ratio = (
            alpha * compression_ratio + (1 - alpha) * self.compression_ratio
        )

    def record_miss(self) -> None:
        """Increment miss counter."""

        self.misses += 1

    def record_set(self) -> None:
        """Increment set counter."""

        self.sets += 1

    def record_delete(self) -> None:
        """Increment delete counter."""

        self.deletes += 1

    def record_error(self) -> None:
        """Increment error counter."""

        self.errors += 1

    def record_disk_read(self) -> None:
        """Increment disk read counter."""

        self.disk_reads += 1

    def record_disk_write(self) -> None:
        """Increment disk write counter."""

        self.disk_writes += 1

    def record_disk_delete(self) -> None:
        """Increment disk delete counter."""

        self.disk_deletes += 1


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """Stored cache entry metadata."""

    payload: bytes
    encoding: Literal["raw", "gz"]
    expires_at: float | None
    created_at: float
    compression_ratio: float
    serialized_size: int
    version: int

    def is_expired(self, now: float) -> bool:
        """Return whether the entry is past its expiry."""

        return self.expires_at is not None and now >= self.expires_at


def cache_path_for_key(base_dir: Path, key: str) -> Path:
    """Return deterministic on-disk path for a cache key.

    The layout shards files by the first two hex characters of the SHA-256 hash
    to avoid large directory fan-out.
    """

    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return base_dir / digest[:2] / f"{digest}.cache"


class PersistentCacheManager:
    """Cache with an in-memory hot set and optional disk persistence."""

    def __init__(
        self,
        base_path: Path,
        *,
        max_entries: int = 2048,
        max_memory_bytes: int = 128 * 1024 * 1024,
        memory_pressure_threshold: float | None = None,
        persistence_enabled: bool = True,
        serialization_version: int = SERIALIZATION_VERSION,
        min_compression_benefit: float = MIN_COMPRESSION_BENEFIT,
        warm_log_sample_rate: float = DEFAULT_WARM_SAMPLE_RATE,
        stats: CacheStats | None = None,
    ) -> None:
        """Initialise a persistent cache instance.

        Args:
            base_path: Root directory used for persistence. Created if missing.
            max_entries: Maximum number of entries held in memory.
            max_memory_bytes: Approximate upper bound for in-memory payload size.
            memory_pressure_threshold: Optional ratio gate applied to memory
                usage. When specified, the cache evicts LRU entries until
                ``usage/max_memory_bytes`` is below the threshold.
            persistence_enabled: Toggle file persistence.
            serialization_version: Marker persisted with each entry.
            min_compression_benefit: Required fractional size reduction before
                applying gzip compression. Example: 0.05 → reuse raw payload
                unless gzip reduces size by at least 5%.
            warm_log_sample_rate: Probability (0–1) of logging a warm success.
            stats: Optional external stats container (primarily for tests).
        """

        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.max_size = max_entries
        self.max_memory_bytes = max_memory_bytes
        self.max_memory_mb = max_memory_bytes / (1024 * 1024)
        self.memory_pressure_threshold = memory_pressure_threshold
        self.persistence_enabled = persistence_enabled
        self.serialization_version = serialization_version
        self.min_compression_benefit = min_compression_benefit
        self.warm_log_sample_rate = warm_log_sample_rate

        self._lock = asyncio.Lock()
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._memory_bytes = 0
        self._stats = stats or CacheStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def stats(self) -> CacheStats:
        """Return read/write statistics."""

        return self._stats

    async def get(self, key: str, default: Any | None = None) -> Any | None:
        """Retrieve value from cache if present and unexpired."""

        start = time.perf_counter()
        now = time.time()
        entry: CacheEntry | None = None

        async with self._lock:
            current = self._entries.get(key)
            if current is not None:
                if current.is_expired(now):
                    self._remove_entry_locked(key, current)
                else:
                    self._entries.move_to_end(key)
                    entry = current

        if entry is not None:
            value = self._deserialize(entry)
            latency = (time.perf_counter() - start) * 1000
            self._stats.record_hit(latency, entry.compression_ratio)
            return value

        if not self.persistence_enabled:
            self._stats.record_miss()
            return default

        disk_entry = await asyncio.to_thread(self._read_from_disk, key, now)
        if disk_entry is None:
            self._stats.record_miss()
            return default

        value = self._deserialize(disk_entry)
        async with self._lock:
            self._insert_entry_locked(key, disk_entry)

        latency = (time.perf_counter() - start) * 1000
        self._stats.record_hit(latency, disk_entry.compression_ratio)
        self._stats.record_disk_read()
        return value

    async def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: int | None = None,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Store ``value`` with optional TTL.

        Both ``ttl`` and ``ttl_seconds`` are accepted for compatibility; when
        both are provided ``ttl_seconds`` wins.
        """

        effective_ttl = ttl_seconds if ttl_seconds is not None else ttl
        now = time.time()
        expires_at = now + effective_ttl if effective_ttl is not None else None
        try:
            entry = self._serialize(value, expires_at, now)
        except Exception as exc:  # pragma: no cover - defensive guard
            self._stats.record_error()
            logger.exception("Failed to serialize cache value for key=%s", key)
            raise RuntimeError("Cache serialization failed") from exc

        async with self._lock:
            self._insert_entry_locked(key, entry)

        if self.persistence_enabled:
            await asyncio.to_thread(self._write_to_disk, key, entry)
            self._stats.record_disk_write()

        self._stats.record_set()
        return True

    async def delete(self, key: str) -> bool:
        """Delete key from cache and attempt persistence delete."""

        removed = False
        async with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._memory_bytes = max(0, self._memory_bytes - entry.serialized_size)
                removed = True

        if self.persistence_enabled:
            deleted = await asyncio.to_thread(self._delete_from_disk, key)
            if deleted:
                self._stats.record_disk_delete()
            removed = removed or deleted

        if removed:
            self._stats.record_delete()
        return removed

    async def clear(self) -> None:
        """Remove all in-memory entries and wipe persistence if enabled."""

        async with self._lock:
            self._entries.clear()
            self._memory_bytes = 0

        if self.persistence_enabled:
            await asyncio.to_thread(self._clear_disk)

    async def initialize(self) -> None:
        """Placeholder to match existing cache manager interface."""

    async def cleanup(self) -> None:
        """Release resources by clearing entries."""

        await self.clear()

    async def warm_one_key(
        self,
        key: str,
        loader: Callable[[], Awaitable[Any]],
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Populate cache for ``key`` using ``loader`` when missing.

        Returns ``True`` if the cache was populated by this call.
        """

        existing = await self.get(key)
        if existing is not None:
            return False

        try:
            value = await loader()
        except Exception:  # pragma: no cover - user loader failure
            self._stats.record_error()
            logger.exception("Warm loader failed for key=%s", key)
            return False

        if value is None:
            return False

        await self.set(key, value, ttl_seconds=ttl_seconds)
        if random.random() < self.warm_log_sample_rate:
            logger.info("Warm populated key hash=%s", self._mask_key(key))
        return True

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def embedding_key(
        text: str,
        model: str,
        provider: str,
        dimensions: int | None = None,
    ) -> str:
        """Return a stable key for embedding cache entries."""

        digest_source = f"{provider}:{model}:{text}"
        digest = hashlib.sha256(digest_source.encode())
        digest_hex = digest.hexdigest()
        if dimensions is not None:
            return f"emb:{dimensions}:{digest_hex}"
        return f"emb:{digest_hex}"

    @staticmethod
    def search_key(query: str, filters: dict[str, Any] | None = None) -> str:
        """Return a stable key for search results."""

        serialized = json.dumps(filters or {}, sort_keys=True, default=str)
        digest = hashlib.sha256(f"{query}:{serialized}".encode()).hexdigest()
        return f"srch:{digest}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _insert_entry_locked(self, key: str, entry: CacheEntry) -> None:
        """Insert entry and enforce limits. Caller must hold the lock."""

        previous = self._entries.pop(key, None)
        if previous is not None:
            self._memory_bytes = max(0, self._memory_bytes - previous.serialized_size)

        self._entries[key] = entry
        self._entries.move_to_end(key)
        self._memory_bytes += entry.serialized_size

        self._evict_if_needed_locked()

    def _remove_entry_locked(self, key: str, entry: CacheEntry) -> None:
        """Remove entry while holding lock."""

        self._entries.pop(key, None)
        self._memory_bytes = max(0, self._memory_bytes - entry.serialized_size)

    def _evict_if_needed_locked(self) -> None:
        """Evict entries until limits are satisfied."""

        while len(self._entries) > self.max_entries:
            _, evicted = self._entries.popitem(last=False)
            self._memory_bytes = max(0, self._memory_bytes - evicted.serialized_size)

        if self.memory_pressure_threshold is None or self.max_memory_bytes <= 0:
            return

        threshold_bytes = self.max_memory_bytes * self.memory_pressure_threshold
        while self._memory_bytes > threshold_bytes and self._entries:
            _, evicted = self._entries.popitem(last=False)
            self._memory_bytes = max(0, self._memory_bytes - evicted.serialized_size)

    def _serialize(
        self, value: Any, expires_at: float | None, created_at: float
    ) -> CacheEntry:
        """Serialize payload and select compression strategy."""

        serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = gzip.compress(serialized)
        compression_ratio = 1.0 if not serialized else len(compressed) / len(serialized)

        benefit = 1.0 - compression_ratio
        if benefit >= self.min_compression_benefit:
            payload = compressed
            encoding: Literal["raw", "gz"] = "gz"
        else:
            payload = serialized
            encoding = "raw"
            compression_ratio = 1.0

        return CacheEntry(
            payload=payload,
            encoding=encoding,
            expires_at=expires_at,
            created_at=created_at,
            compression_ratio=compression_ratio,
            serialized_size=len(payload),
            version=self.serialization_version,
        )

    def _deserialize(self, entry: CacheEntry) -> Any:
        """Materialise Python object from cached entry."""

        if entry.encoding == "gz":
            serialized = gzip.decompress(entry.payload)
        else:
            serialized = entry.payload
        # Cache layer controls the data origin; pickle loads is acceptable here.
        return pickle.loads(serialized)  # noqa: S301

    def _write_to_disk(self, key: str, entry: CacheEntry) -> None:
        """Persist entry to disk synchronously."""

        path = cache_path_for_key(self.base_path, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "version": entry.version,
            "expires_at": entry.expires_at,
            "created_at": entry.created_at,
            "encoding": entry.encoding,
            "compression_ratio": entry.compression_ratio,
            "payload": entry.payload,
        }
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("wb") as fh:
            pickle.dump(record, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)

    def _read_from_disk(self, key: str, now: float) -> CacheEntry | None:
        """Read entry from disk if available and fresh."""

        path = cache_path_for_key(self.base_path, key)
        if not path.exists():
            return None
        try:
            with path.open("rb") as fh:
                record = pickle.load(fh)  # noqa: S301 - trusted persistence files
        except (OSError, pickle.PickleError) as exc:
            self._stats.record_error()
            logger.warning("Failed to read cache entry key=%s: %s", key, exc)
            return None

        if record.get("version") != self.serialization_version:
            logger.debug("Cache entry version mismatch for key=%s", key)
            return None

        expires_at = record.get("expires_at")
        if expires_at is not None and now >= expires_at:
            self._delete_from_disk(key)
            return None

        payload = record["payload"]
        encoding = record["encoding"]
        compression_ratio = float(record.get("compression_ratio", 1.0))
        created_at = float(record.get("created_at", now))
        return CacheEntry(
            payload=payload,
            encoding=encoding,
            expires_at=expires_at,
            created_at=created_at,
            compression_ratio=compression_ratio,
            serialized_size=len(payload),
            version=self.serialization_version,
        )

    def _delete_from_disk(self, key: str) -> bool:
        """Delete persisted entry for ``key``."""

        path = cache_path_for_key(self.base_path, key)
        try:
            path.unlink(missing_ok=True)
            return True
        except OSError as exc:
            self._stats.record_error()
            logger.warning("Failed to delete cache file for key=%s: %s", key, exc)
            return False

    def _clear_disk(self) -> None:
        """Clear the persistence directory."""

        if not self.base_path.exists():
            return
        for child in self.base_path.rglob("*.cache"):
            try:
                child.unlink()
            except OSError:
                logger.debug("Failed to remove cache file %s", child)

    @staticmethod
    def _mask_key(key: str) -> str:
        """Hash key for safe logging."""

        if not key:
            return "empty"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]

    async def size(self) -> int:
        """Return count of in-memory entries."""

        async with self._lock:
            return len(self._entries)

    def get_memory_usage(self) -> float:
        """Return approximate memory usage in megabytes."""

        return self._memory_bytes / (1024 * 1024)


__all__ = [
    "CacheEntry",
    "CacheStats",
    "PersistentCacheManager",
    "cache_path_for_key",
]
