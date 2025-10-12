"""Specialized cache for embedding vectors with DragonflyDB optimizations."""

import logging
from typing import Any

from ._bulk_delete import delete_in_batches
from .dragonfly_cache import DragonflyCache
from .key_utils import build_embedding_cache_key


logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Specialized cache for embedding vectors using DragonflyDB.

    Optimized for:
    - Long TTL (7 days) since embeddings are expensive to generate
    - Batch operations for multiple embeddings
    - Content-based cache keys for deduplication
    - Efficient storage of floating-point vectors
    """

    def __init__(self, cache: DragonflyCache, default_ttl: int = 86400 * 7):
        """Initialize embedding cache.

        Args:
            cache: DragonflyDB cache instance
            default_ttl: Default TTL in seconds (7 days for embeddings)
        """
        self.cache = cache
        self.default_ttl = default_ttl

    async def get_embedding(
        self,
        text: str,
        model: str,
        provider: str = "openai",
        dimensions: int | None = None,
    ) -> list[float] | None:
        """Get cached embedding for text.

        Args:
            text: Text that was embedded
            model: Model name (e.g., "text-embedding-3-small")
            provider: Embedding provider (e.g., "openai", "fastembed")
            dimensions: Embedding dimensions for validation

        Returns:
            Cached embedding vector or None if not found
        """
        key = build_embedding_cache_key(text, model, provider, dimensions)

        try:
            cached = await self.cache.get(key)
            if cached:
                # Validate dimensions if provided
                if dimensions is not None and len(cached) != dimensions:
                    logger.warning(
                        "Cached embedding dimensions mismatch: expected %d, got %d",
                        dimensions,
                        len(cached),
                    )
                    return None

                # Convert to list of floats for consistency
                return [float(x) for x in cached]

            return None

        except (AttributeError, ConnectionError, RuntimeError, TimeoutError) as e:
            logger.error("Error retrieving embedding from cache: %s", e)
            return None

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    async def set_embedding(
        self,
        text: str,
        model: str,
        embedding: list[float],
        provider: str = "openai",
        dimensions: int | None = None,
        ttl: int | None = None,
    ) -> bool:
        """Cache embedding vector.

        Args:
            text: Text that was embedded
            model: Model name
            embedding: Embedding vector
            provider: Embedding provider
            dimensions: Embedding dimensions
            ttl: Custom TTL (uses default if None)

        Returns:
            Success status
        """
        key = build_embedding_cache_key(text, model, provider, dimensions)
        cache_ttl = ttl or self.default_ttl

        try:
            # Validate embedding
            if not embedding or not isinstance(embedding, list):
                logger.error("Invalid embedding format")
                return False

            # Ensure consistent float format
            normalized_embedding = [float(x) for x in embedding]

            success = await self.cache.set(key, normalized_embedding, ttl=cache_ttl)

            if success:
                logger.debug(
                    "Cached embedding: %sD vector for %s (%s)",
                    len(normalized_embedding),
                    model,
                    provider,
                )

            return success

        except (AttributeError, ImportError, RuntimeError, ValueError) as e:
            logger.error("Error caching embedding: %s", e)
            return False

    def _get_key(
        self,
        text: str,
        model: str,
        provider: str,
        dimensions: int | None = None,
    ) -> str:
        """Return deterministic embedding cache key.

        Args:
            text: Raw text used to generate the embedding.
            model: Embedding model identifier.
            provider: Provider backing the embedding request.
            dimensions: Optional embedding dimensionality when the backend
                exposes multiple sizes for the same model.

        Returns:
            Deterministic cache key derived from canonical hashing utility.
        """
        return build_embedding_cache_key(text, model, provider, dimensions)

    async def get_batch_embeddings(
        self,
        texts: list[str],
        model: str,
        provider: str = "openai",
        dimensions: int | None = None,
    ) -> tuple[dict[str, list[float]], list[str]]:
        """Get batch embeddings efficiently.

        Uses DragonflyDB's superior batch performance for optimal retrieval.

        Args:
            texts: List of texts
            model: Model name
            provider: Embedding provider
            dimensions: Expected embedding dimensions

        Returns:
            Tuple of (cached_embeddings_dict, missing_texts_list)
        """
        if not texts:
            return {}, []

        # Generate cache keys
        keys = [
            build_embedding_cache_key(text, model, provider, dimensions)
            for text in texts
        ]

        try:
            # Use DragonflyDB's optimized MGET
            cached_values = await self.cache.mget(keys)

            cached = {}
            missing = []

            for text, value in zip(texts, cached_values, strict=False):
                if value is not None:
                    # Validate dimensions if provided
                    if dimensions is not None and len(value) != dimensions:
                        logger.warning(
                            "Cached embedding dimensions mismatch for '%s...': "
                            "expected %s, got %s",
                            text[:50],
                            dimensions,
                            len(value),
                        )
                        missing.append(text)
                    else:
                        cached[text] = [float(x) for x in value]
                else:
                    missing.append(text)

            logger.debug(
                "Batch embedding cache: %s hits, %s misses",
                len(cached),
                len(missing),
            )

            return cached, missing

        except (AttributeError, ConnectionError, RuntimeError, TimeoutError) as e:
            logger.error("Error in batch embedding retrieval: %s", e)
            # Return all as missing on error
            return {}, texts

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    async def set_batch_embeddings(
        self,
        embeddings: dict[str, list[float]],
        model: str,
        provider: str = "openai",
        dimensions: int | None = None,
        ttl: int | None = None,
    ) -> bool:
        """Cache multiple embeddings efficiently.

        Uses DragonflyDB's superior batch performance for optimal storage.

        Args:
            embeddings: Dictionary mapping texts to embedding vectors
            model: Model name
            provider: Embedding provider
            dimensions: Expected embedding dimensions
            ttl: Custom TTL (uses default if None)

        Returns:
            Success status
        """
        if not embeddings:
            return True

        cache_ttl = ttl or self.default_ttl

        try:
            # Prepare batch data with cache keys
            mapping = {}

            for text, embedding in embeddings.items():
                # Validate embedding
                if not embedding or not isinstance(embedding, list):
                    logger.warning("Skipping invalid embedding for '%s...'", text[:50])
                    continue

                # Validate dimensions if provided
                if dimensions is not None and len(embedding) != dimensions:
                    logger.warning(
                        "Skipping embedding with wrong dimensions for '%s...': "
                        "expected %s, got %s",
                        text[:50],
                        dimensions,
                        len(embedding),
                    )
                    continue

                key = build_embedding_cache_key(text, model, provider, dimensions)
                mapping[key] = [float(x) for x in embedding]

            if not mapping:
                logger.warning("No valid embeddings to cache")
                return False

            # Use DragonflyDB's optimized MSET
            success = await self.cache.mset(mapping, ttl=cache_ttl)

            if success:
                logger.debug(
                    "Batch cached %s embeddings for %s (%s)",
                    len(mapping),
                    model,
                    provider,
                )

            return success

        except (AttributeError, ConnectionError, ImportError, RuntimeError) as e:
            logger.error("Error in batch embedding caching: %s", e)
            return False

    async def warm_cache(
        self,
        common_queries: list[str],
        model: str,
        provider: str = "openai",
        dimensions: int | None = None,
    ) -> list[str]:
        """Pre-warm cache with common queries.

        Args:
            common_queries: List of common query texts
            model: Model name
            provider: Embedding provider
            dimensions: Expected embedding dimensions

        Returns:
            List of texts that need embedding generation
        """
        if not common_queries:
            return []

        try:
            missing_texts = []

            # Check which queries are not cached
            for query in common_queries:
                key = build_embedding_cache_key(query, model, provider, dimensions)
                exists = await self.cache.exists(key)
                if not exists:
                    missing_texts.append(query)

            if missing_texts:
                logger.info(
                    "Cache warming: %s queries need embedding generation for %s (%s)",
                    len(missing_texts),
                    model,
                    provider,
                )
            else:
                logger.info(
                    "Cache warming: all %s queries already cached for %s (%s)",
                    len(common_queries),
                    model,
                    provider,
                )

            return missing_texts

        except (ConnectionError, ImportError, RuntimeError, TimeoutError) as e:
            logger.error("Error in cache warming: %s", e)
            return common_queries  # Return all as missing on error

    async def invalidate_model(
        self,
        model: str,
        provider: str = "openai",
    ) -> int:
        """Invalidate all cached embeddings for a specific model.

        Args:
            model: Model name to invalidate
            provider: Embedding provider

        Returns:
            Number of invalidated entries
        """
        try:
            # Pattern to match all embeddings for this model/provider
            pattern = f"emb:{provider}:{model}:*"

            # Use cache scan to find matching keys
            keys = await self.cache.scan_keys(pattern)
            deleted_count = await delete_in_batches(self.cache, keys)

            if deleted_count:
                logger.info(
                    "Invalidated %s cached embeddings for %s (%s)",
                    deleted_count,
                    model,
                    provider,
                )

            return deleted_count

        except (AttributeError, ConnectionError, ImportError, RuntimeError) as e:
            logger.error("Error invalidating model cache: %s", e)
            return 0

    async def get_cache_stats(self) -> dict:
        """Get embedding cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        try:
            # Count embedding-specific keys
            pattern = "emb:*"
            keys = await self.cache.scan_keys(pattern)

            stats: dict[str, Any] = {
                "total_embeddings": len(keys),
                "cache_size": await self.cache.size(),
            }

            # Group by provider and model if possible
            providers = {}
            models = {}

            for key in keys:
                try:
                    # Parse key format: emb:{provider}:{model}:{dimensions}:{hash}
                    parts = key.split(":")
                    if len(parts) >= 4:
                        provider = parts[1]
                        model = parts[2]

                        providers[provider] = providers.get(provider, 0) + 1
                        models[f"{provider}:{model}"] = (
                            models.get(f"{provider}:{model}", 0) + 1
                        )

                except (ImportError, RuntimeError, ValueError):
                    continue

            stats["by_provider"] = providers
            stats["by_model"] = models

            return stats

        except (ConnectionError, ImportError, RuntimeError, TimeoutError) as e:
            logger.error("Error getting cache stats: %s", e)
            return {"error": str(e)}

    async def get_stats(self) -> dict:
        """Alias for get_cache_stats for compatibility."""
        return await self.get_cache_stats()
