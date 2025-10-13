"""HyDE Query Engine with Query API integration."""

# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments,too-many-locals

import asyncio
import hashlib
import logging
import time
from typing import Any, cast

import numpy as np

from src.services.base import BaseService
from src.services.embeddings.manager import EmbeddingManager
from src.services.errors import EmbeddingServiceError, QdrantServiceError
from src.services.vector_db.service import VectorStoreService

from .cache import CacheEntryContext, HyDECache, SearchResultPayload
from .config import HyDEConfig, HyDEMetricsConfig, HyDEPromptConfig
from .generator import HypotheticalDocumentGenerator


logger = logging.getLogger(__name__)


class HyDEQueryEngine(BaseService):
    """HyDE search engine with vector store integration."""

    def __init__(
        self,
        config: HyDEConfig,
        prompt_config: HyDEPromptConfig,
        metrics_config: HyDEMetricsConfig,
        embedding_manager: EmbeddingManager,
        vector_store: VectorStoreService,
        cache_manager: Any,
        openai_api_key: str | None,
    ):
        """Initialize HyDE query engine.

        Args:
            config: HyDE configuration
            prompt_config: Prompt configuration
            metrics_config: Metrics configuration
            embedding_manager: Embedding service manager
            vector_store: Vector service for search
            cache_manager: Cache manager (DragonflyDB)
            openai_api_key: OpenAI API key for document generation

        """
        super().__init__(None)
        self.config = config
        self.prompt_config = prompt_config
        self.metrics_config = metrics_config
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

        # Initialize components
        self.generator = HypotheticalDocumentGenerator(
            config, prompt_config, openai_api_key
        )
        self.cache = HyDECache(config, cache_manager)

        # Performance tracking
        self.search_count = 0
        self._total_search_time = 0.0
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
                self.vector_store.initialize()
                if hasattr(self.vector_store, "initialize")
                else asyncio.sleep(0),
            )

            self._initialized = True
            logger.info("HyDE query engine initialized")

        except Exception as e:
            msg = "Failed to initialize HyDE engine"
            raise EmbeddingServiceError(msg) from e

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
        """Perform HyDE search with vector store integration.

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
            results = await self._perform_hybrid_search(
                query,
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

        except Exception as e:
            logger.exception("HyDE search failed")

            # Fallback to regular search if enabled
            if self.config.enable_fallback:
                logger.info("Falling back to regular search due to HyDE failure")
                self.fallback_count += 1
                return await self._fallback_search(
                    query, collection_name, limit, filters, search_accuracy
                )
            msg = "HyDE search failed"
            raise EmbeddingServiceError(msg) from e

        return results

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
            msg = "Failed to generate hypothetical documents"
            raise EmbeddingServiceError(msg)

        # Generate embeddings for hypothetical documents
        embeddings_result = await self.embedding_manager.generate_embeddings(
            texts=generation_result.documents,
            provider_name="openai",  # Use high-quality provider for HyDE
            auto_select=False,
        )

        if "embeddings" not in embeddings_result:
            msg = "Failed to generate embeddings for hypothetical documents"
            raise EmbeddingServiceError(msg)

        # Average embeddings for final HyDE vector
        embeddings_array = np.array(embeddings_result["embeddings"])
        hyde_embedding = np.mean(embeddings_array, axis=0).tolist()

        # Cache the result
        if use_cache:
            await self.cache.set_hyde_embedding(
                query=query,
                embedding=hyde_embedding,
                hypothetical_docs=generation_result.documents,
                context=CacheEntryContext(
                    domain=domain,
                    generation_metadata={
                        "generation_time": generation_result.generation_time,
                        "tokens_used": generation_result.tokens_used,
                        "diversity_score": generation_result.diversity_score,
                        "model": self.config.generation_model,
                    },
                ),
            )

        return hyde_embedding

    async def _generate_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for the original query."""
        embeddings_result = await self.embedding_manager.generate_embeddings(
            texts=[query], provider_name="openai", auto_select=False
        )

        embeddings_list: list[list[float]] | None = None
        if isinstance(embeddings_result, dict):
            embeddings_list = cast(
                list[list[float]] | None, embeddings_result.get("embeddings")
            )
        else:
            embeddings_list = cast(
                list[list[float]] | None,
                getattr(embeddings_result, "embeddings", None),
            )

        if not embeddings_list:
            msg = "Failed to generate query embedding"
            raise EmbeddingServiceError(msg)

        return list(embeddings_list[0])

    async def _perform_hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        hyde_embedding: list[float],
        collection_name: str,
        limit: int,
        filters: dict[str, Any] | None,
        search_accuracy: str,
    ) -> list[dict[str, Any]]:
        """Perform search using HyDE embedding."""
        try:
            # Use HyDE embedding for search
            matches = await self.vector_store.search_vector(
                collection=collection_name,
                vector=hyde_embedding,
                limit=limit,
                filters=filters,
            )
            results: list[dict[str, Any]] = []
            for match in matches:
                record = {
                    "id": match.id,
                    "score": match.score,
                    "content": match.content,
                    "title": match.title,
                    "url": match.url,
                    "collection": match.collection,
                    "metadata": dict(match.metadata or {}),
                }
                if match.normalized_score is not None:
                    record["normalized_score"] = match.normalized_score
                if match.raw_score is not None:
                    record["raw_score"] = match.raw_score
                results.append(record)
            return results

        except Exception as e:
            logger.exception("HyDE search execution failed")
            msg = "HyDE search execution failed"
            raise QdrantServiceError(msg) from e

    async def _apply_reranking(
        self, query: str, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Apply reranking to search results."""
        try:
            # Use embedding manager's reranking if available
            if hasattr(self.embedding_manager, "rerank_results"):
                return await self.embedding_manager.rerank_results(query, results)
            # Basic reranking fallback (could implement BGE reranking here)
            logger.debug("Reranking not available, returning original results")

        except (TimeoutError, AttributeError, RuntimeError, ValueError):
            logger.warning("Reranking failed, returning original results")
            return results

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

            matches = await self.vector_store.search_vector(
                collection=collection_name,
                vector=query_embedding,
                limit=limit,
                filters=filters,
            )
            results: list[dict[str, Any]] = []
            for match in matches:
                record = {
                    "id": match.id,
                    "score": match.score,
                    "content": match.content,
                    "title": match.title,
                    "url": match.url,
                    "collection": match.collection,
                    "metadata": dict(match.metadata or {}),
                }
                if match.normalized_score is not None:
                    record["normalized_score"] = match.normalized_score
                if match.raw_score is not None:
                    record["raw_score"] = match.raw_score
                results.append(record)
            return results

        except Exception as e:
            logger.exception("Fallback search failed")
            msg = "Both HyDE and fallback search failed"
            raise EmbeddingServiceError(msg) from e

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

        payload = SearchResultPayload(
            results=results,
            metadata={"result_count": len(results), "cached_at": time.time()},
        )

        await self.cache.set_search_results(
            query, collection_name, search_params, payload
        )

    async def _should_use_hyde_for_ab_test(self, query: str) -> bool:
        """Determine if query should use HyDE for A/B testing."""
        # Simple hash-based assignment for consistent user experience

        query_hash = int(hashlib.sha256(query.encode()).hexdigest(), 16)

        # Use control group percentage from config
        threshold = self.metrics_config.control_group_percentage

        return (query_hash % 100) >= (threshold * 100)

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
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

    @property
    def total_search_time(self) -> float:
        """Aggregate search latency recorded by the engine."""

        return self._total_search_time

    @total_search_time.setter
    def total_search_time(self, value: float) -> None:
        self._total_search_time = value

        self.cache.reset_metrics()
        logger.info("HyDE engine metrics reset")
