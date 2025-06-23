import typing

"""HyDE caching implementation using DragonflyDB."""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any

import numpy as np

from ..base import BaseService
from ..errors import EmbeddingServiceError
from .config import HyDEConfig
from .generator import GenerationResult

logger = logging.getLogger(__name__)


class HyDECache(BaseService):
    """Intelligent caching layer for HyDE embeddings and results."""

    def __init__(self, config: HyDEConfig, cache_manager: Any):
        """Initialize HyDE cache.

        Args:
            config: HyDE configuration
            cache_manager: DragonflyCache or compatible cache manager
        """
        super().__init__(config)
        self.config = config
        self.cache_manager = cache_manager

        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_sets = 0
        self.cache_errors = 0

        # Cache key prefixes
        self.embedding_prefix = f"{config.cache_prefix}:embedding"
        self.documents_prefix = f"{config.cache_prefix}:documents"
        self.results_prefix = f"{config.cache_prefix}:results"

    async def initialize(self) -> None:
        """Initialize cache."""
        if self._initialized:
            return

        try:
            # Test cache connection
            if hasattr(self.cache_manager, "initialize"):
                await self.cache_manager.initialize()

            # Verify cache is working
            test_key = f"{self.config.cache_prefix}:test"
            await self.cache_manager.set(test_key, "test_value", ttl=60)
            test_value = await self.cache_manager.get(test_key)

            if test_value != "test_value":
                raise EmbeddingServiceError("Cache test failed")

            await self.cache_manager.delete(test_key)

            self._initialized = True
            logger.info("HyDE cache initialized")

        except Exception as e:
            raise EmbeddingServiceError(f"Failed to initialize HyDE cache: {e}") from e

    async def cleanup(self) -> None:
        """Cleanup cache resources."""
        if hasattr(self.cache_manager, "cleanup"):
            await self.cache_manager.cleanup()
        self._initialized = False
        logger.info("HyDE cache cleaned up")

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

        try:
            cache_key = self._get_embedding_cache_key(query, domain)
            cached_data = await self.cache_manager.get(cache_key)

            if cached_data is not None:
                self.cache_hits += 1

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
                    logger.warning(f"Unexpected cache format for key {cache_key}")
                    self.cache_misses += 1
                    return None

                logger.debug(f"Cache hit for HyDE embedding: {query}")
                return embedding
            else:
                self.cache_misses += 1
                return None

        except Exception as e:
            self.cache_errors += 1
            logger.warning(f"Cache get error for HyDE embedding: {e}")
            return None

    async def set_hyde_embedding(
        self,
        query: str,
        embedding: list[float],
        hypothetical_docs: list[str],
        generation_metadata: dict[str, Any] | None = None,
        domain: str | None = None,
    ) -> bool:
        """Cache HyDE embedding with metadata.

        Args:
            query: Search query
            embedding: HyDE embedding vector
            hypothetical_docs: Generated hypothetical documents
            generation_metadata: Metadata about generation process
            domain: Optional domain hint

        Returns:
            True if cached successfully
        """
        self._validate_initialized()

        try:
            cache_key = self._get_embedding_cache_key(query, domain)

            # Prepare cache data
            cache_data = {
                "embedding": embedding,
                "query": query,
                "domain": domain,
                "timestamp": time.time(),
                "hypothetical_docs": hypothetical_docs
                if self.config.cache_hypothetical_docs
                else [],
                "metadata": generation_metadata or {},
            }

            # Use binary format for embeddings to save space
            if isinstance(embedding, list):
                embedding_array = np.array(embedding, dtype=np.float32)
                cache_data["embedding"] = embedding_array.tobytes()
                cache_data["embedding_shape"] = embedding_array.shape
                cache_data["embedding_dtype"] = str(embedding_array.dtype)

            success = await self.cache_manager.set(
                cache_key, cache_data, ttl=self.config.cache_ttl_seconds
            )

            if success:
                self.cache_sets += 1
                logger.debug(f"Cached HyDE embedding for query: {query}")

            return success

        except Exception as e:
            self.cache_errors += 1
            logger.warning(f"Cache set error for HyDE embedding: {e}")
            return False

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

        try:
            cache_key = self._get_documents_cache_key(query, domain)
            cached_docs = await self.cache_manager.get(cache_key)

            if cached_docs is not None:
                self.cache_hits += 1
                logger.debug(f"Cache hit for hypothetical documents: {query}")
                return cached_docs
            else:
                self.cache_misses += 1
                return None

        except Exception as e:
            self.cache_errors += 1
            logger.warning(f"Cache get error for hypothetical documents: {e}")
            return None

    async def set_hypothetical_documents(
        self,
        query: str,
        documents: list[str],
        generation_result: GenerationResult,
        domain: str | None = None,
    ) -> bool:
        """Cache hypothetical documents with metadata.

        Args:
            query: Search query
            documents: Generated hypothetical documents
            generation_result: Full generation result with metadata
            domain: Optional domain hint

        Returns:
            True if cached successfully
        """
        if not self.config.cache_hypothetical_docs:
            return True

        self._validate_initialized()

        try:
            cache_key = self._get_documents_cache_key(query, domain)

            cache_data = {
                "documents": documents,
                "query": query,
                "domain": domain,
                "timestamp": time.time(),
                "generation_time": generation_result.generation_time,
                "tokens_used": generation_result.tokens_used,
                "diversity_score": generation_result.diversity_score,
            }

            success = await self.cache_manager.set(
                cache_key, cache_data, ttl=self.config.cache_ttl_seconds
            )

            if success:
                self.cache_sets += 1
                logger.debug(f"Cached hypothetical documents for query: {query}")

            return success

        except Exception as e:
            self.cache_errors += 1
            logger.warning(f"Cache set error for hypothetical documents: {e}")
            return False

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

        try:
            cache_key = self._get_results_cache_key(
                query, collection_name, search_params
            )
            cached_results = await self.cache_manager.get(cache_key)

            if cached_results is not None:
                self.cache_hits += 1
                logger.debug(f"Cache hit for search results: {query}")
                return cached_results
            else:
                self.cache_misses += 1
                return None

        except Exception as e:
            self.cache_errors += 1
            logger.warning(f"Cache get error for search results: {e}")
            return None

    async def set_search_results(
        self,
        query: str,
        collection_name: str,
        search_params: dict[str, Any],
        results: list[dict[str, Any]],
        search_metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Cache search results.

        Args:
            query: Search query
            collection_name: Target collection
            search_params: Search parameters
            results: Search results to cache
            search_metadata: Optional metadata about the search

        Returns:
            True if cached successfully
        """
        self._validate_initialized()

        try:
            cache_key = self._get_results_cache_key(
                query, collection_name, search_params
            )

            cache_data = {
                "results": results,
                "query": query,
                "collection_name": collection_name,
                "search_params": search_params,
                "timestamp": time.time(),
                "metadata": search_metadata or {},
            }

            # Use shorter TTL for search results
            ttl = min(self.config.cache_ttl_seconds // 2, 1800)  # Max 30 minutes

            success = await self.cache_manager.set(cache_key, cache_data, ttl=ttl)

            if success:
                self.cache_sets += 1
                logger.debug(f"Cached search results for query: {query}")

            return success

        except Exception as e:
            self.cache_errors += 1
            logger.warning(f"Cache set error for search results: {e}")
            return False

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
            try:
                # Check if already cached
                embedding = await self.get_hyde_embedding(query, domain)

                if embedding is not None:
                    results[query] = True  # Already cached
                else:
                    results[query] = False  # Needs generation

            except Exception as e:
                logger.warning(f"Cache warm-up error for query '{query}': {e}")
                results[query] = False

        logger.info(
            f"Cache warm-up completed: {sum(results.values())}/{len(results)} already cached"
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
                self.cache_manager.delete(embedding_key),
                self.cache_manager.delete(documents_key),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            success_count = sum(1 for result in results if result is True)

            logger.debug(
                f"Invalidated {success_count} cache entries for query: {query}"
            )

            return success_count > 0

        except Exception as e:
            logger.warning(f"Cache invalidation error for query '{query}': {e}")
            return False

    def _get_embedding_cache_key(self, query: str, domain: str | None = None) -> str:
        """Generate cache key for HyDE embedding."""
        key_data = f"{query}:{domain or 'general'}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{self.embedding_prefix}:{key_hash}"

    def _get_documents_cache_key(self, query: str, domain: str | None = None) -> str:
        """Generate cache key for hypothetical documents."""
        key_data = f"{query}:{domain or 'general'}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{self.documents_prefix}:{key_hash}"

    def _get_results_cache_key(
        self, query: str, collection_name: str, search_params: dict[str, Any]
    ) -> str:
        """Generate cache key for search results."""
        # Create deterministic key from parameters
        params_str = json.dumps(search_params, sort_keys=True)
        key_data = f"{query}:{collection_name}:{params_str}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{self.results_prefix}:{key_hash}"

    def get_cache_metrics(self) -> dict[str, Any]:
        """Get cache performance metrics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_sets": self.cache_sets,
            "cache_errors": self.cache_errors,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "error_rate": self.cache_errors / max(total_requests, 1),
        }

    def reset_metrics(self) -> None:
        """Reset cache metrics."""
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_sets = 0
        self.cache_errors = 0
        logger.debug("HyDE cache metrics reset")
