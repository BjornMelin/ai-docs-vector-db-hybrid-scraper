"""HyDE caching implementation using DragonflyDB."""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.config.models import CacheType
from src.services.base import BaseService
from src.services.cache.manager import CacheManager
from src.services.errors import EmbeddingServiceError

from .config import HyDEConfig
from .generator import GenerationResult


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CacheMetrics:
    """Metrics captured for HyDE cache operations."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    errors: int = 0


@dataclass(slots=True)
class CachePrefixes:
    """Prefix values for cache namespaces."""

    embedding: str
    documents: str
    results: str


@dataclass(slots=True)
class CacheEntryContext:
    """Context for cache entries that share domain and metadata."""

    domain: str | None = None
    generation_metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class CacheWriteOptions:
    """Configuration for writing values to the cache manager."""

    cache_type: CacheType
    ttl: int


@dataclass(slots=True)
class SearchResultPayload:
    """Payload for storing search results in the cache."""

    results: list[dict[str, Any]]
    metadata: dict[str, Any] | None = None


def _raise_cache_test_failed() -> None:
    """Raise EmbeddingServiceError for failed cache test."""
    msg = "Cache test failed"
    raise EmbeddingServiceError(msg)


class HyDECache(BaseService):
    """Cache layer for HyDE embeddings and results using DragonflyDB."""

    def __init__(self, config: HyDEConfig, cache_manager: CacheManager) -> None:
        """Initialize HyDE cache.

        Args:
            config: HyDE configuration
            cache_manager: Central cache manager providing get/set/delete APIs
        """

        super().__init__(None)
        self.config = config
        self.hyde_config = config
        self.cache_manager: CacheManager = cache_manager

        # Cache metrics and prefixes
        self.metrics = CacheMetrics()
        self.prefixes = CachePrefixes(
            embedding=f"{config.cache_prefix}:embedding",
            documents=f"{config.cache_prefix}:documents",
            results=f"{config.cache_prefix}:results",
        )

    async def initialize(self) -> None:
        """Initialize cache."""
        if self._initialized:
            return

        try:
            await self._initialize_cache_manager()
            await self._test_cache_functionality()
            self._initialized = True
            logger.info("HyDE cache initialized")
        except Exception as e:
            msg = f"Failed to initialize HyDE cache: {e}"
            raise EmbeddingServiceError(msg) from e

    async def _initialize_cache_manager(self) -> None:
        """Initialize the cache manager."""
        if hasattr(self.cache_manager, "initialize"):
            await self.cache_manager.initialize()

    async def _test_cache_functionality(self) -> None:
        """Test cache functionality with a test key."""
        test_key = f"{self.config.cache_prefix}:test"
        await self.cache_manager.set(
            test_key,
            "test_value",
            cache_type=CacheType.HYDE,
            ttl=60,
        )
        test_value = await self.cache_manager.get(
            test_key,
            cache_type=CacheType.HYDE,
        )

        if test_value != "test_value":
            _raise_cache_test_failed()

        await self.cache_manager.delete(test_key, cache_type=CacheType.HYDE)

    async def cleanup(self) -> None:
        """Cleanup cache resources."""
        if hasattr(self.cache_manager, "cleanup"):
            await self.cache_manager.cleanup()
        self._initialized = False
        logger.info("HyDE cache cleaned up")

    async def _get_from_cache(
        self,
        cache_key: str,
        *,
        cache_type: CacheType,
        log_context: str,
    ) -> Any | None:
        """Retrieve a value from the cache manager with error handling."""

        try:
            return await self.cache_manager.get(
                cache_key,
                cache_type=cache_type,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.metrics.errors += 1
            logger.warning("Cache get error for %s: %s", log_context, exc)
            return None

    async def _set_in_cache(
        self,
        cache_key: str,
        payload: Any,
        *,
        options: CacheWriteOptions,
        log_context: str,
    ) -> bool:
        """Store a value via the cache manager with error handling."""

        try:
            return await self.cache_manager.set(
                cache_key,
                payload,
                cache_type=options.cache_type,
                ttl=options.ttl,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.metrics.errors += 1
            logger.warning("Cache set error for %s: %s", log_context, exc)
            return False

    async def _delete_from_cache(
        self,
        cache_key: str,
        *,
        cache_type: CacheType,
        log_context: str,
    ) -> bool:
        """Delete a value via the cache manager with error handling."""

        try:
            return await self.cache_manager.delete(
                cache_key,
                cache_type=cache_type,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.metrics.errors += 1
            logger.warning("Cache delete error for %s: %s", log_context, exc)
            return False

    async def get_hyde_embedding(
        self, query: str, domain: str | None = None
    ) -> list[float] | None:
        """Get cached HyDE embedding for a query.

        Args:
            query: Search query
            domain: Optional domain hint

        Returns:
            Cached embedding or None if not found

        """
        self._validate_initialized()

        cache_key = self._get_embedding_cache_key(query, domain)

        # Try to get cached data
        cached_data = await self._get_from_cache(
            cache_key,
            cache_type=CacheType.HYDE,
            log_context="HyDE embedding",
        )
        if cached_data is None:
            self.metrics.misses += 1
            return None

        # Process cached data
        embedding = self._deserialize_embedding(cached_data, cache_key)
        if embedding is not None:
            self.metrics.hits += 1
        else:
            self.metrics.misses += 1

        return embedding

    def _deserialize_embedding(
        self, cached_data: Any, cache_key: str
    ) -> list[float] | None:
        """Process cached embedding data into embedding format."""
        if cached_data is None:
            return None

        # Handle different cache formats
        if isinstance(cached_data, dict) and "embedding" in cached_data:
            # Binary format
            if isinstance(cached_data["embedding"], bytes):
                embedding = np.frombuffer(
                    cached_data["embedding"], dtype=np.float32
                ).tolist()
            else:
                embedding = cached_data["embedding"]
        elif isinstance(cached_data, list):
            # Direct embedding format
            embedding = cached_data
        else:
            logger.warning("Unexpected cache format for key %s", cache_key)
            return None

        logger.debug("Cache hit for HyDE embedding key: %s", cache_key)
        return embedding

    async def set_hyde_embedding(
        self,
        query: str,
        embedding: list[float],
        hypothetical_docs: list[str],
        *,
        context: CacheEntryContext | None = None,
    ) -> bool:
        """Cache HyDE embedding with metadata.

        Args:
            query: Search query
            embedding: HyDE embedding vector
            hypothetical_docs: Generated hypothetical documents
            context: Domain and metadata context for the cache entry

        Returns:
            True if cached successfully

        """
        self._validate_initialized()

        resolved_context = context or CacheEntryContext()
        cache_key = self._get_embedding_cache_key(query, resolved_context.domain)

        # Prepare cache data
        cache_data = self._prepare_cache_data(
            query, embedding, hypothetical_docs, resolved_context
        )

        # Store in cache
        success = await self._set_in_cache(
            cache_key,
            cache_data,
            options=CacheWriteOptions(
                cache_type=CacheType.HYDE,
                ttl=self.config.cache_ttl_seconds,
            ),
            log_context="HyDE embedding",
        )

        if success:
            self.metrics.sets += 1
            logger.debug("Cached HyDE embedding for key: %s", cache_key)
        else:
            logger.debug("Failed to cache HyDE embedding for key: %s", cache_key)
        return success

    def _prepare_cache_data(
        self,
        query: str,
        embedding: list[float],
        hypothetical_docs: list[str],
        context: CacheEntryContext,
    ) -> dict[str, Any]:
        """Prepare cache data with binary encoding for embeddings."""
        cache_data = {
            "embedding": embedding,
            "query": query,
            "domain": context.domain,
            "timestamp": time.time(),
            "hypothetical_docs": hypothetical_docs
            if self.config.cache_hypothetical_docs
            else [],
            "metadata": context.generation_metadata or {},
        }

        # Use binary format for embeddings to save space
        if isinstance(embedding, list):
            embedding_array = np.array(embedding, dtype=np.float32)
            cache_data["embedding"] = embedding_array.tobytes()
            cache_data["embedding_shape"] = embedding_array.shape
            cache_data["embedding_dtype"] = str(embedding_array.dtype)

        return cache_data

    async def get_hypothetical_documents(
        self, query: str, domain: str | None = None
    ) -> list[str] | None:
        """Get cached hypothetical documents for a query.

        Args:
            query: Search query
            domain: Optional domain hint

        Returns:
            Cached documents or None if not found
        """

        if not self.config.cache_hypothetical_docs:
            return None

        self._validate_initialized()

        cache_key = self._get_documents_cache_key(query, domain)
        cached_docs = await self._get_from_cache(
            cache_key,
            cache_type=CacheType.HYDE,
            log_context="HyDE hypothetical documents",
        )

        if cached_docs is not None:
            logger.debug("Cache hit for hypothetical documents key: %s", cache_key)
            self.metrics.hits += 1
            return cached_docs

        logger.debug("Cache miss for hypothetical documents key: %s", cache_key)
        self.metrics.misses += 1
        return None

    async def set_hypothetical_documents(
        self,
        query: str,
        documents: list[str],
        generation_result: GenerationResult,
        *,
        context: CacheEntryContext | None = None,
    ) -> bool:
        """Cache hypothetical documents with metadata.

        Args:
            query: Search query
            documents: Generated hypothetical documents
            generation_result: Full generation result with metadata
            context: Optional context providing domain information

        Returns:
            True if cached successfully
        """

        if not self.config.cache_hypothetical_docs:
            return True

        self._validate_initialized()

        resolved_context = context or CacheEntryContext()
        cache_key = self._get_documents_cache_key(query, resolved_context.domain)

        cache_data = {
            "documents": documents,
            "query": query,
            "domain": resolved_context.domain,
            "timestamp": time.time(),
            "generation_time": generation_result.generation_time,
            "tokens_used": generation_result.tokens_used,
            "diversity_score": generation_result.diversity_score,
        }

        success = await self._set_in_cache(
            cache_key,
            cache_data,
            options=CacheWriteOptions(
                cache_type=CacheType.HYDE,
                ttl=self.config.cache_ttl_seconds,
            ),
            log_context="HyDE hypothetical documents",
        )

        query = cache_data.get("query", "<unknown>")
        if success:
            self.metrics.sets += 1
            logger.debug("Cached hypothetical documents for query: %s", query)
        else:
            logger.debug("Failed to cache hypothetical documents for query: %s", query)
        return success

    async def get_search_results(
        self, query: str, collection_name: str, search_params: dict[str, Any]
    ) -> list[dict[str, Any]] | None:
        """Get cached search results.

        Args:
            query: Search query
            collection_name: Target collection
            search_params: Search parameters for cache key

        Returns:
            Cached results or None if not found
        """

        self._validate_initialized()

        cache_key = self._get_results_cache_key(query, collection_name, search_params)
        cached_results = await self._get_from_cache(
            cache_key,
            cache_type=CacheType.SEARCH,
            log_context="HyDE search results",
        )

        if cached_results is not None:
            logger.debug("Cache hit for search results key: %s", cache_key)
            self.metrics.hits += 1
            return cached_results

        logger.debug("Cache miss for search results key: %s", cache_key)
        self.metrics.misses += 1
        return None

    async def set_search_results(
        self,
        query: str,
        collection_name: str,
        search_params: dict[str, Any],
        payload: SearchResultPayload,
    ) -> bool:
        """Cache search results.

        Args:
            query: Search query
            collection_name: Target collection
            search_params: Search parameters
            payload: Search results and optional metadata

        Returns:
            True if cached successfully
        """

        self._validate_initialized()

        cache_key = self._get_results_cache_key(query, collection_name, search_params)

        cache_data = {
            "results": payload.results,
            "query": query,
            "collection_name": collection_name,
            "search_params": search_params,
            "timestamp": time.time(),
            "metadata": payload.metadata or {},
        }

        # Use shorter TTL for search results
        ttl = min(self.config.cache_ttl_seconds // 2, 1800)  # Max 30 minutes

        success = await self._set_in_cache(
            cache_key,
            cache_data,
            options=CacheWriteOptions(
                cache_type=CacheType.SEARCH,
                ttl=ttl,
            ),
            log_context="HyDE search results",
        )

        query = cache_data.get("query", "<unknown>")
        if success:
            self.metrics.sets += 1
            logger.debug("Cached search results for query: %s", query)
        else:
            logger.debug("Failed to cache search results for query: %s", query)
        return success

    async def warm_cache(
        self, common_queries: list[str], domain: str | None = None
    ) -> dict[str, bool]:
        """Pre-warm cache with common queries.

        Args:
            common_queries: List of frequently used queries
            domain: Optional domain hint

        Returns:
            Dictionary mapping queries to warm-up success status
        """

        self._validate_initialized()

        results = {}

        for query in common_queries:
            embedding = await self.get_hyde_embedding(query, domain)
            results[query] = embedding is not None

        already_cached = sum(results.values())
        logger.info(
            "Cache warm-up completed: %d/%d already cached",
            already_cached,
            len(results),
        )

        return results

    async def invalidate_query(self, query: str, domain: str | None = None) -> bool:
        """Invalidate all cached data for a specific query.

        Args:
            query: Query to invalidate
            domain: Optional domain hint

        Returns:
            True if invalidation was successful
        """

        self._validate_initialized()

        try:
            # Get all possible cache keys for this query
            embedding_key = self._get_embedding_cache_key(query, domain)
            documents_key = self._get_documents_cache_key(query, domain)

            # Delete cache entries
            tasks = [
                self._delete_from_cache(
                    embedding_key,
                    cache_type=CacheType.HYDE,
                    log_context="HyDE embedding invalidation",
                ),
                self._delete_from_cache(
                    documents_key,
                    cache_type=CacheType.HYDE,
                    log_context="HyDE documents invalidation",
                ),
            ]

            results = await asyncio.gather(*tasks)

            success_count = sum(1 for result in results if result)

            logger.debug(
                "Invalidated %d cache entries for query: %s", success_count, query
            )

        except Exception as exc:  # pragma: no cover - defensive logging
            self.metrics.errors += 1
            logger.warning("Cache invalidation error for query '%s': %s", query, exc)
            return False

        return success_count > 0

    def _get_embedding_cache_key(self, query: str, domain: str | None = None) -> str:
        """Generate cache key for HyDE embedding."""
        key_data = f"{query}:{domain or 'general'}"
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()
        return f"{self.prefixes.embedding}:{key_hash}"

    def _get_documents_cache_key(self, query: str, domain: str | None = None) -> str:
        """Generate cache key for hypothetical documents."""
        key_data = f"{query}:{domain or 'general'}"
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()
        return f"{self.prefixes.documents}:{key_hash}"

    def _get_results_cache_key(
        self, query: str, collection_name: str, search_params: dict[str, Any]
    ) -> str:
        """Generate cache key for search results."""

        # Create deterministic key from parameters
        params_str = json.dumps(search_params, sort_keys=True)
        key_data = f"{query}:{collection_name}:{params_str}"
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()
        return f"{self.prefixes.results}:{key_hash}"

    def get_cache_metrics(self) -> dict[str, Any]:
        """Get cache performance metrics."""

        total_requests = self.metrics.hits + self.metrics.misses
        hit_rate = self.metrics.hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_hits": self.metrics.hits,
            "cache_misses": self.metrics.misses,
            "cache_sets": self.metrics.sets,
            "cache_errors": self.metrics.errors,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "error_rate": self.metrics.errors / max(total_requests, 1),
        }

    def reset_metrics(self) -> None:
        """Reset cache metrics."""
        self.metrics = CacheMetrics()
        logger.debug("HyDE cache metrics reset")
