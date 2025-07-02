import typing


"""Specialized cache for embedding vectors with DragonflyDB optimizations."""

import hashlib
import logging

from .dragonfly_cache import DragonflyCache


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
        key = self._get_key(text, model, provider, dimensions)

        try:
            cached = await self.cache.get(key)
            if cached:
                # Validate dimensions if provided
                if dimensions is not None and len(cached) != dimensions:
                    logger.warning(
                        f"Cached embedding dimensions mismatch: "
                        f"expected {dimensions}, got {len(cached)}"
                    )
                    return None

                # Convert to list of floats for consistency
                return [float(x) for x in cached]

            return None

        except (AttributeError, ConnectionError, RuntimeError, TimeoutError) as e:
            logger.error(f"Error retrieving embedding from cache: {e}")
            return None

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
        key = self._get_key(text, model, provider, dimensions)
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
                    f"Cached embedding: {len(normalized_embedding)}D vector "
                    f"for {model} ({provider})"
                )

            return success

        except (AttributeError, ImportError, RuntimeError, ValueError) as e:
            logger.error(f"Error caching embedding: {e}")
            return False

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
        keys = [self._get_key(text, model, provider, dimensions) for text in texts]

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
                            f"Cached embedding dimensions mismatch for '{text[:50]}...': "
                            f"expected {dimensions}, got {len(value)}"
                        )
                        missing.append(text)
                    else:
                        cached[text] = [float(x) for x in value]
                else:
                    missing.append(text)

            logger.debug(
                f"Batch embedding cache: {len(cached)} hits, {len(missing)} misses"
            )

            return cached, missing

        except (AttributeError, ConnectionError, RuntimeError, TimeoutError) as e:
            logger.error(f"Error in batch embedding retrieval: {e}")
            # Return all as missing on error
            return {}, texts

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
                    logger.warning(f"Skipping invalid embedding for '{text[:50]}...'")
                    continue

                # Validate dimensions if provided
                if dimensions is not None and len(embedding) != dimensions:
                    logger.warning(
                        f"Skipping embedding with wrong dimensions for '{text[:50]}...': "
                        f"expected {dimensions}, got {len(embedding)}"
                    )
                    continue

                key = self._get_key(text, model, provider, dimensions)
                mapping[key] = [float(x) for x in embedding]

            if not mapping:
                logger.warning("No valid embeddings to cache")
                return False

            # Use DragonflyDB's optimized MSET
            success = await self.cache.mset(mapping, ttl=cache_ttl)

            if success:
                logger.debug(
                    f"Batch cached {len(mapping)} embeddings for {model} ({provider})"
                )

            return success

        except (AttributeError, ConnectionError, ImportError, RuntimeError) as e:
            logger.error(f"Error in batch embedding caching: {e}")
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
                key = self._get_key(query, model, provider, dimensions)
                exists = await self.cache.exists(key)
                if not exists:
                    missing_texts.append(query)

            if missing_texts:
                logger.info(
                    f"Cache warming: {len(missing_texts)} queries need embedding "
                    f"generation for {model} ({provider})"
                )
            else:
                logger.info(
                    f"Cache warming: all {len(common_queries)} queries already cached "
                    f"for {model} ({provider})"
                )

            return missing_texts

        except (ConnectionError, ImportError, RuntimeError, TimeoutError) as e:
            logger.error(f"Error in cache warming: {e}")
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

            if keys:
                # Delete in batches for efficiency
                batch_size = 100
                deleted_count = 0

                for i in range(0, len(keys), batch_size):
                    batch = keys[i : i + batch_size]
                    results = await self.cache.delete_many(batch)
                    deleted_count += sum(results.values())

                logger.info(
                    f"Invalidated {deleted_count} cached embeddings for "
                    f"{model} ({provider})"
                )
                return deleted_count

            return 0

        except (AttributeError, ConnectionError, ImportError, RuntimeError) as e:
            logger.error(f"Error invalidating model cache: {e}")
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

            stats = {
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

                except (ImportError, RuntimeError, ValueError) as e:
                    continue

            stats["by_provider"] = providers
            stats["by_model"] = models

            return stats

        except (ConnectionError, ImportError, RuntimeError, TimeoutError) as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}

    async def get_stats(self) -> dict:
        """Alias for get_cache_stats for compatibility."""
        return await self.get_cache_stats()

    def _get_key(
        self,
        text: str,
        model: str,
        provider: str,
        dimensions: int | None = None,
    ) -> str:
        """Generate cache key for embedding.

        Uses content-based hashing to ensure deduplication.

        Args:
            text: Text content
            model: Model name
            provider: Embedding provider
            dimensions: Embedding dimensions

        Returns:
            Cache key
        """
        # Normalize text for consistent hashing (using SHA256 for security)
        normalized_text = text.lower().strip()
        text_hash = hashlib.sha256(normalized_text.encode()).hexdigest()

        # Include dimensions in key if provided
        if dimensions is not None:
            return f"emb:{provider}:{model}:{dimensions}:{text_hash}"
        return f"emb:{provider}:{model}:{text_hash}"
