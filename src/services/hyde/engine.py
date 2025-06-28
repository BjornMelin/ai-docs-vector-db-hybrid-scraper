"""HyDE Query Engine with Query API integration."""

import asyncio
import hashlib
import logging
import time
from typing import Any

import numpy as np

from ..base import BaseService
from ..embeddings.manager import EmbeddingManager
from ..errors import EmbeddingServiceError, QdrantServiceError
from ..vector_db.service import QdrantService
from .cache import HyDECache
from .config import HyDEConfig, HyDEMetricsConfig, HyDEPromptConfig
from .generator import HypotheticalDocumentGenerator


logger = logging.getLogger(__name__)


class HyDEQueryEngine(BaseService):
    """HyDE-enhanced search engine with Query API prefetch and fusion."""

    def __init__(
        self,
        config: HyDEConfig,
        prompt_config: HyDEPromptConfig,
        metrics_config: HyDEMetricsConfig,
        embedding_manager: EmbeddingManager,
        qdrant_service: QdrantService,
        cache_manager: Any,
        llm_client: Any,
    ):
        """Initialize HyDE query engine.

        Args:
            config: HyDE configuration
            prompt_config: Prompt configuration
            metrics_config: Metrics configuration
            embedding_manager: Embedding service manager
            qdrant_service: Qdrant service for search
            cache_manager: Cache manager (DragonflyDB)
            llm_client: LLM client for document generation
        """
        super().__init__(config)
        self.config = config
        self.prompt_config = prompt_config
        self.metrics_config = metrics_config
        self.embedding_manager = embedding_manager
        self.qdrant_service = qdrant_service

        # Initialize components
        self.generator = HypotheticalDocumentGenerator(
            config, prompt_config, llm_client
        )
        self.cache = HyDECache(config, cache_manager)

        # Performance tracking
        self.search_count = 0
        self.total_search_time = 0.0
        self.cache_hit_count = 0
        self.generation_count = 0
        self.fallback_count = 0

        # A/B testing support
        self.control_group_searches = 0
        self.treatment_group_searches = 0

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        try:
            # Initialize components in parallel
            await asyncio.gather(
                self.generator.initialize(),
                self.cache.initialize(),
                self.embedding_manager.initialize()
                if hasattr(self.embedding_manager, "initialize")
                else asyncio.sleep(0),
                self.qdrant_service.initialize()
                if hasattr(self.qdrant_service, "initialize")
                else asyncio.sleep(0),
            )

            self._initialized = True
            logger.info("HyDE query engine initialized")

        except Exception as e:
            raise EmbeddingServiceError("Failed to initialize HyDE engine") from e

    async def cleanup(self) -> None:
        """Cleanup all components."""
        await asyncio.gather(
            self.generator.cleanup(), self.cache.cleanup(), return_exceptions=True
        )
        self._initialized = False
        logger.info("HyDE query engine cleaned up")

    async def enhanced_search(
        self,
        query: str,
        collection_name: str = "documents",
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        search_accuracy: str = "balanced",
        domain: str | None = None,
        use_cache: bool = True,
        force_hyde: bool = False,
    ) -> list[dict[str, Any]]:
        """Perform HyDE-enhanced search with Query API prefetch.

        Args:
            query: Search query
            collection_name: Target collection
            limit: Number of results to return
            filters: Optional filters to apply
            search_accuracy: Search accuracy level
            domain: Optional domain hint for better generation
            use_cache: Whether to use caching
            force_hyde: Force HyDE even if disabled globally

        Returns:
            List of search results with HyDE enhancement

        Raises:
            EmbeddingServiceError: If search fails
        """
        self._validate_initialized()

        start_time = time.time()
        self.search_count += 1

        try:
            # Check if HyDE is enabled
            if not (self.config.enable_hyde or force_hyde):
                logger.debug("HyDE disabled, falling back to regular search")
                return await self._fallback_search(
                    query, collection_name, limit, filters, search_accuracy
                )

            # A/B testing logic
            if self.metrics_config.ab_testing_enabled and not force_hyde:
                use_hyde = await self._should_use_hyde_for_ab_test(query)
                if not use_hyde:
                    self.control_group_searches += 1
                    return await self._fallback_search(
                        query, collection_name, limit, filters, search_accuracy
                    )
                else:
                    self.treatment_group_searches += 1

            # Check cache first if enabled
            if use_cache:
                cached_results = await self._get_cached_results(
                    query, collection_name, limit, filters, search_accuracy, domain
                )
                if cached_results is not None:
                    self.cache_hit_count += 1
                    logger.debug("Cache hit for HyDE search")
                    return cached_results

            # Generate or retrieve HyDE embedding
            hyde_embedding = await self._get_or_generate_hyde_embedding(
                query, domain, use_cache
            )

            # Generate original query embedding
            query_embedding = await self._generate_query_embedding(query)

            # Perform HyDE search with Query API
            results = await self._perform_hyde_search(
                query_embedding,
                hyde_embedding,
                collection_name,
                limit,
                filters,
                search_accuracy,
            )

            # Apply reranking if enabled
            if self.config.enable_reranking and len(results) > 1:
                results = await self._apply_reranking(query, results)

            # Cache results if enabled
            if use_cache:
                await self._cache_search_results(
                    query,
                    collection_name,
                    limit,
                    filters,
                    search_accuracy,
                    domain,
                    results,
                )

            search_time = time.time() - start_time
            self.total_search_time += search_time

            # Log performance metrics
            if self.metrics_config.track_generation_time:
                logger.debug("HyDE search completed in {search_time:.2f}s for query")

            return results

        except Exception as e:
            logger.error("HyDE search failed", exc_info=True)

            # Fallback to regular search if enabled
            if self.config.enable_fallback:
                logger.info("Falling back to regular search due to HyDE failure")
                self.fallback_count += 1
                return await self._fallback_search(
                    query, collection_name, limit, filters, search_accuracy
                )
            else:
                raise EmbeddingServiceError("HyDE search failed") from e

    async def _get_or_generate_hyde_embedding(
        self, query: str, domain: str | None, use_cache: bool
    ) -> list[float]:
        """Get HyDE embedding from cache or generate new one."""

        # Try cache first
        if use_cache:
            cached_embedding = await self.cache.get_hyde_embedding(query, domain)
            if cached_embedding is not None:
                return cached_embedding

        # Generate hypothetical documents
        generation_result = await self.generator.generate_documents(query, domain)

        self.generation_count += 1

        if not generation_result.documents:
            raise EmbeddingServiceError("Failed to generate hypothetical documents")

        # Generate embeddings for hypothetical documents
        embeddings_result = await self.embedding_manager.generate_embeddings(
            texts=generation_result.documents,
            provider_name="openai",  # Use high-quality provider for HyDE
            auto_select=False,
        )

        if "embeddings" not in embeddings_result:
            raise EmbeddingServiceError(
                "Failed to generate embeddings for hypothetical documents"
            )

        # Average embeddings for final HyDE vector
        embeddings_array = np.array(embeddings_result["embeddings"])
        hyde_embedding = np.mean(embeddings_array, axis=0).tolist()

        # Cache the result
        if use_cache:
            await self.cache.set_hyde_embedding(
                query=query,
                embedding=hyde_embedding,
                hypothetical_docs=generation_result.documents,
                generation_metadata={
                    "generation_time": generation_result.generation_time,
                    "tokens_used": generation_result.tokens_used,
                    "diversity_score": generation_result.diversity_score,
                    "model": self.config.generation_model,
                },
                domain=domain,
            )

        return hyde_embedding

    async def _generate_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for the original query."""

        embeddings_result = await self.embedding_manager.generate_embeddings(
            texts=[query], provider_name="openai", auto_select=False
        )

        if "embeddings" not in embeddings_result or not embeddings_result["embeddings"]:
            raise EmbeddingServiceError("Failed to generate query embedding")

        return embeddings_result["embeddings"][0]

    async def _perform_hyde_search(
        self,
        query_embedding: list[float],
        hyde_embedding: list[float],
        collection_name: str,
        limit: int,
        _filters: dict[str, Any] | None,
        search_accuracy: str,
    ) -> list[dict[str, Any]]:
        """Perform search using Query API with HyDE prefetch."""

        try:
            # Use the existing hyde_search method in QdrantService
            # but we need to call it differently since it expects hypothetical_embeddings as a list
            results = await self.qdrant_service.hyde_search(
                collection_name=collection_name,
                query="HyDE search",  # This parameter isn't actually used in the implementation
                query_embedding=query_embedding,
                hypothetical_embeddings=[hyde_embedding],  # Pass as list
                limit=limit,
                fusion_algorithm=self.config.fusion_algorithm,
                search_accuracy=search_accuracy,
            )

            return results

        except Exception as e:
            logger.exception("Query API search failed")
            raise QdrantServiceError("HyDE search execution failed") from e

    async def _apply_reranking(
        self, query: str, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Apply reranking to search results."""

        try:
            # Use embedding manager's reranking if available
            if hasattr(self.embedding_manager, "rerank_results"):
                reranked_results = await self.embedding_manager.rerank_results(
                    query, results
                )
                return reranked_results
            else:
                # Basic reranking fallback (could implement BGE reranking here)
                logger.debug("Reranking not available, returning original results")
                return results

        except Exception:
            logger.warning("Reranking failed, returning original results")
            return results

    async def _fallback_search(
        self,
        query: str,
        collection_name: str,
        limit: int,
        filters: dict[str, Any] | None,
        search_accuracy: str,
    ) -> list[dict[str, Any]]:
        """Fallback to regular search without HyDE."""

        try:
            # Generate query embedding
            query_embedding = await self._generate_query_embedding(query)

            # Perform regular filtered search
            results = await self.qdrant_service.filtered_search(
                collection_name=collection_name,
                query_vector=query_embedding,
                filters=filters or {},
                limit=limit,
                search_accuracy=search_accuracy,
            )

            return results

        except Exception as e:
            logger.exception("Fallback search failed")
            raise EmbeddingServiceError("Both HyDE and fallback search failed") from e

    async def _get_cached_results(
        self,
        query: str,
        collection_name: str,
        limit: int,
        filters: dict[str, Any] | None,
        search_accuracy: str,
        domain: str | None,
    ) -> list[dict[str, Any]] | None:
        """Get cached search results."""

        search_params = {
            "limit": limit,
            "filters": filters,
            "search_accuracy": search_accuracy,
            "domain": domain,
            "hyde_enabled": True,
        }

        return await self.cache.get_search_results(
            query, collection_name, search_params
        )

    async def _cache_search_results(
        self,
        query: str,
        collection_name: str,
        limit: int,
        filters: dict[str, Any] | None,
        search_accuracy: str,
        domain: str | None,
        results: list[dict[str, Any]],
    ) -> None:
        """Cache search results."""

        search_params = {
            "limit": limit,
            "filters": filters,
            "search_accuracy": search_accuracy,
            "domain": domain,
            "hyde_enabled": True,
        }

        search_metadata = {"result_count": len(results), "cached_at": time.time()}

        await self.cache.set_search_results(
            query, collection_name, search_params, results, search_metadata
        )

    async def _should_use_hyde_for_ab_test(self, query: str) -> bool:
        """Determine if query should use HyDE for A/B testing."""

        # Simple hash-based assignment for consistent user experience

        query_hash = int(hashlib.sha256(query.encode()).hexdigest(), 16)

        # Use control group percentage from config
        threshold = self.metrics_config.control_group_percentage

        return (query_hash % 100) >= (threshold * 100)

    async def batch_search(
        self,
        queries: list[str],
        collection_name: str = "documents",
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        search_accuracy: str = "balanced",
        domain: str | None = None,
        max_concurrent: int = 5,
    ) -> list[list[dict[str, Any]]]:
        """Perform batch HyDE search with concurrency control.

        Args:
            queries: List of search queries
            collection_name: Target collection
            limit: Number of results per query
            filters: Optional filters to apply
            search_accuracy: Search accuracy level
            domain: Optional domain hint
            max_concurrent: Maximum concurrent searches

        Returns:
            List of search results for each query
        """
        self._validate_initialized()

        semaphore = asyncio.Semaphore(max_concurrent)

        async def search_single(query: str) -> list[dict[str, Any]]:
            async with semaphore:
                return await self.enhanced_search(
                    query=query,
                    collection_name=collection_name,
                    limit=limit,
                    filters=filters,
                    search_accuracy=search_accuracy,
                    domain=domain,
                )

        # Execute searches concurrently
        tasks = [search_single(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Batch search failed for query {i}")
                processed_results.append([])
            else:
                processed_results.append(result)

        return processed_results

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics."""

        # Basic metrics
        avg_search_time = (
            self.total_search_time / self.search_count if self.search_count > 0 else 0.0
        )

        cache_hit_rate = (
            self.cache_hit_count / self.search_count if self.search_count > 0 else 0.0
        )

        fallback_rate = (
            self.fallback_count / self.search_count if self.search_count > 0 else 0.0
        )

        metrics = {
            "search_performance": {
                "total_searches": self.search_count,
                "avg_search_time": avg_search_time,
                "total_search_time": self.total_search_time,
                "cache_hit_rate": cache_hit_rate,
                "fallback_rate": fallback_rate,
            },
            "generation_metrics": self.generator.get_metrics(),
            "cache_metrics": self.cache.get_cache_metrics(),
        }

        # A/B testing metrics
        if self.metrics_config.ab_testing_enabled:
            total_ab_searches = (
                self.control_group_searches + self.treatment_group_searches
            )
            metrics["ab_testing"] = {
                "control_group_searches": self.control_group_searches,
                "treatment_group_searches": self.treatment_group_searches,
                "total_ab_searches": total_ab_searches,
                "treatment_percentage": (
                    self.treatment_group_searches / total_ab_searches
                    if total_ab_searches > 0
                    else 0.0
                ),
            }

        return metrics

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.search_count = 0
        self.total_search_time = 0.0
        self.cache_hit_count = 0
        self.generation_count = 0
        self.fallback_count = 0
        self.control_group_searches = 0
        self.treatment_group_searches = 0

        self.cache.reset_metrics()
        logger.info("HyDE engine metrics reset")
