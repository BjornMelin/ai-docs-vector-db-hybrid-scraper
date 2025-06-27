"""Search orchestrator with essential features.

This module provides a streamlined search orchestrator that maintains core functionality
and valuable V2 features. Includes query expansion, clustering, personalization, and federation.
"""

import logging  # noqa: PLC0415
import time  # noqa: PLC0415
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ..base import BaseService
from ..rag import RAGGenerator
from ..rag.models import RAGRequest
from .clustering import ResultClusteringRequest, ResultClusteringService
from .expansion import QueryExpansionRequest, QueryExpansionService
from .federated import FederatedSearchService
from .ranking import PersonalizedRankingRequest, PersonalizedRankingService


logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    """Search execution modes."""

    BASIC = "basic"  # Basic search without advanced features
    ENHANCED = "enhanced"  # Search with expansion and filtering
    FULL = "full"  # All features enabled


class SearchPipeline(str, Enum):
    """Pipeline configurations."""

    FAST = "fast"  # Optimized for speed
    BALANCED = "balanced"  # Balance between speed and quality
    COMPREHENSIVE = "comprehensive"  # Maximum quality and features


class SearchRequest(BaseModel):
    """Search request parameters."""

    # Core parameters
    query: str = Field(..., description="Search query")
    collection_name: str | None = Field(None, description="Target collection")
    limit: int = Field(10, ge=1, le=1000, description="Maximum results")
    offset: int = Field(0, ge=0, description="Result offset")

    # Feature flags
    mode: SearchMode = Field(SearchMode.ENHANCED, description="Search mode")
    pipeline: SearchPipeline = Field(
        SearchPipeline.BALANCED, description="Pipeline config"
    )

    # Optional features
    enable_expansion: bool = Field(True, description="Enable query expansion")
    enable_clustering: bool = Field(False, description="Enable result clustering")
    enable_personalization: bool = Field(
        False, description="Enable personalized ranking"
    )
    enable_federation: bool = Field(False, description="Enable federated search")

    # RAG features (NEW V1 PORTFOLIO FEATURE)
    enable_rag: bool = Field(False, description="Enable RAG answer generation")
    rag_max_tokens: int | None = Field(
        None, gt=0, le=4000, description="Override RAG max tokens"
    )
    rag_temperature: float | None = Field(
        None, ge=0.0, le=2.0, description="Override RAG temperature"
    )
    require_high_confidence: bool = Field(
        False, description="Require high confidence RAG answers"
    )

    # User context
    user_id: str | None = Field(None, description="User ID for personalization")
    session_id: str | None = Field(None, description="Session ID")

    # Performance
    enable_caching: bool = Field(True, description="Enable result caching")
    max_processing_time_ms: float = Field(5000.0, description="Max processing time")


class SearchResult(BaseModel):
    """Search result with metadata."""

    results: list[dict[str, Any]] = Field(..., description="Search results")
    total_results: int = Field(..., ge=0, description="Total results found")
    query_processed: str = Field(..., description="Final processed query")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time")

    # Feature metadata
    expanded_query: str | None = Field(None, description="Expanded query if applicable")
    clusters: list[dict[str, Any]] | None = Field(None, description="Result clusters")
    features_used: list[str] = Field(
        default_factory=list, description="Features applied"
    )

    # RAG-generated answer (NEW V1 PORTFOLIO FEATURE)
    generated_answer: str | None = Field(
        None, description="RAG-generated contextual answer"
    )
    answer_confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Answer confidence score"
    )
    answer_sources: list[dict[str, Any]] | None = Field(
        None, description="Sources used for answer"
    )
    answer_metrics: dict[str, Any] | None = Field(
        None, description="Answer generation metrics"
    )

    # Caching
    cache_hit: bool = Field(False, description="Whether result was cached")


class SearchOrchestrator(BaseService):
    """Search orchestrator with essential features."""

    def __init__(
        self,
        cache_size: int = 1000,
        enable_performance_optimization: bool = True,
    ):
        """Initialize search orchestrator.

        Args:
            cache_size: Size of result cache
            enable_performance_optimization: Enable performance optimizations
        """
        super().__init__()
        self._logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

        # Configuration
        self.enable_performance_optimization = enable_performance_optimization

        # Initialize services (lazy loading)
        self._query_expansion_service: QueryExpansionService | None = None
        self._clustering_service: ResultClusteringService | None = None
        self._ranking_service: PersonalizedRankingService | None = None
        self._federated_service: FederatedSearchService | None = None
        self._rag_generator: RAGGenerator | None = None

        # Pipeline configurations
        self.pipeline_configs = {
            SearchPipeline.FAST: {
                "enable_expansion": False,
                "enable_clustering": False,
                "enable_personalization": False,
                "max_processing_time_ms": 1000.0,
            },
            SearchPipeline.BALANCED: {
                "enable_expansion": True,
                "enable_clustering": False,
                "enable_personalization": False,
                "max_processing_time_ms": 3000.0,
            },
            SearchPipeline.COMPREHENSIVE: {
                "enable_expansion": True,
                "enable_clustering": True,
                "enable_personalization": True,
                "max_processing_time_ms": 10000.0,
            },
        }

        # Simple caching
        self.cache = {}
        self.cache_size = cache_size

        # Performance tracking
        self.stats = {
            "total_searches": 0,
            "avg_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    @property
    def query_expansion_service(self) -> QueryExpansionService:
        """Lazy load query expansion service."""
        if self._query_expansion_service is None:
            self._query_expansion_service = QueryExpansionService()
        return self._query_expansion_service

    @property
    def clustering_service(self) -> ResultClusteringService:
        """Lazy load clustering service."""
        if self._clustering_service is None:
            self._clustering_service = ResultClusteringService()
        return self._clustering_service

    @property
    def ranking_service(self) -> PersonalizedRankingService:
        """Lazy load ranking service."""
        if self._ranking_service is None:
            self._ranking_service = PersonalizedRankingService()
        return self._ranking_service

    @property
    def federated_service(self) -> FederatedSearchService:
        """Lazy load federated search service."""
        if self._federated_service is None:
            self._federated_service = FederatedSearchService()
        return self._federated_service

    @property
    def rag_generator(self) -> RAGGenerator:
        """Lazy load RAG generator service."""
        if self._rag_generator is None:
            from src.config import get_config  # noqa: PLC0415

            config = get_config()
            self._rag_generator = RAGGenerator(config.rag)
        return self._rag_generator

    async def search(self, request: SearchRequest) -> SearchResult:
        """Execute search with optimized pipeline.

        Args:
            request: Search request

        Returns:
            SearchResult with processed results
        """
        start_time = time.time()
        features_used = []

        try:
            # Check cache first
            if request.enable_caching:
                cache_key = self._get_cache_key(request)
                if cache_key in self.cache:
                    self.stats["cache_hits"] += 1
                    self.stats["total_searches"] += 1  # Count cache hits too
                    cached_result = self.cache[cache_key]
                    cached_result.cache_hit = True
                    return cached_result
                else:
                    self.stats["cache_misses"] += 1

            # Apply pipeline configuration
            config = self._apply_pipeline_config(request)

            # Process query
            processed_query = request.query
            expanded_query = None

            # Step 1: Query expansion (if enabled)
            if (
                config.get("enable_expansion", False)
                and request.mode != SearchMode.BASIC
            ):
                try:
                    expansion_request = QueryExpansionRequest(
                        original_query=request.query,
                        max_expanded_terms=5,  # Keep it simple
                        min_confidence=0.7,
                    )
                    expansion_result = await self.query_expansion_service.expand_query(
                        expansion_request
                    )
                    if expansion_result.expanded_query:
                        expanded_query = expansion_result.expanded_query
                        processed_query = expanded_query
                        features_used.append("query_expansion")
                except Exception as e:
                    self._logger.warning("Query expansion failed")

            # Step 2: Execute search (would call actual search service)
            search_results = await self._execute_search(
                processed_query, request, config
            )

            # Step 3: Post-processing (clustering, ranking, etc.)
            if search_results:
                # Clustering (if enabled)
                if config.get("enable_clustering", False) and len(search_results) > 5:
                    try:
                        clustering_request = ResultClusteringRequest(
                            results=search_results,
                            num_clusters=min(5, len(search_results) // 3),
                        )
                        clustering_result = (
                            await self.clustering_service.cluster_results(
                                clustering_request
                            )
                        )
                        # Add cluster info to results
                        for cluster in clustering_result.clusters:
                            for result in cluster.results:
                                for sr in search_results:
                                    if sr.get("id") == result.id:
                                        sr["cluster_id"] = cluster.cluster_id
                                        sr["cluster_label"] = cluster.label
                        features_used.append("result_clustering")
                    except Exception as e:
                        self._logger.warning("Clustering failed")

                # Personalized ranking (if enabled)
                if config.get("enable_personalization", False) and request.user_id:
                    try:
                        ranking_request = PersonalizedRankingRequest(
                            user_id=request.user_id,
                            query=request.query,
                            results=search_results,
                        )
                        ranking_result = await self.ranking_service.rank_results(
                            ranking_request
                        )
                        # Re-order results based on ranking
                        search_results = self._apply_ranking(
                            search_results, ranking_result
                        )
                        features_used.append("personalized_ranking")
                    except Exception as e:
                        self._logger.warning("Personalized ranking failed")

            # Step 4: RAG answer generation (if enabled)
            rag_answer = None
            rag_confidence = None
            rag_sources = None
            rag_metrics = None

            if request.enable_rag and search_results:
                try:
                    from src.config import get_config  # noqa: PLC0415

                    config = get_config()

                    # Only generate RAG answer if globally enabled or explicitly requested
                    if config.rag.enable_rag or request.enable_rag:
                        rag_request = RAGRequest(
                            query=request.query,
                            search_results=search_results[
                                : config.rag.max_results_for_context
                            ],
                            max_tokens=request.rag_max_tokens,
                            temperature=request.rag_temperature,
                            require_high_confidence=request.require_high_confidence,
                        )

                        # Initialize RAG generator if needed
                        if not self._rag_generator._initialized:
                            await self.rag_generator.initialize()

                        rag_result = await self.rag_generator.generate_answer(
                            rag_request
                        )

                        # Only include answer if confidence meets threshold
                        min_confidence = config.rag.min_confidence_threshold
                        if rag_result.confidence_score >= min_confidence:
                            rag_answer = rag_result.answer
                            rag_confidence = rag_result.confidence_score
                            rag_sources = [
                                {
                                    "source_id": source.source_id,
                                    "title": source.title,
                                    "url": source.url,
                                    "relevance_score": source.relevance_score,
                                    "excerpt": source.excerpt,
                                }
                                for source in rag_result.sources
                            ]
                            rag_metrics = (
                                rag_result.metrics.model_dump()
                                if rag_result.metrics
                                else None
                            )
                            features_used.append("rag_answer_generation")

                except Exception as e:
                    self._logger.warning("RAG answer generation failed")
                    # Continue without RAG - don't fail the entire search

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            # Build result
            result = SearchResult(
                results=search_results[: request.limit],
                total_results=len(search_results),
                query_processed=processed_query,
                processing_time_ms=processing_time,
                expanded_query=expanded_query,
                features_used=features_used,
                generated_answer=rag_answer,
                answer_confidence=rag_confidence,
                answer_sources=rag_sources,
                answer_metrics=rag_metrics,
                cache_hit=False,
            )

            # Cache result
            if request.enable_caching and len(self.cache) < self.cache_size:
                cache_key = self._get_cache_key(request)
                self.cache[cache_key] = result

            # Update stats
            self.stats["total_searches"] += 1
            self.stats["avg_processing_time"] = (
                self.stats["avg_processing_time"] * (self.stats["total_searches"] - 1)
                + processing_time
            ) / self.stats["total_searches"]

            return result

        except Exception as e:
            self._logger.exception("Search failed")
            # Return minimal result on error
            return SearchResult(
                results=[],
                total_results=0,
                query_processed=request.query,
                processing_time_ms=(time.time() - start_time) * 1000,
                features_used=features_used,
            )

    def _apply_pipeline_config(self, request: SearchRequest) -> dict[str, Any]:
        """Apply pipeline configuration to request."""
        # Start with pipeline defaults - these should take precedence
        config = self.pipeline_configs.get(request.pipeline, {}).copy()

        # Check which fields were explicitly set using Pydantic's __pydantic_fields_set__
        if hasattr(request, "__pydantic_fields_set__"):
            explicitly_set = request.__pydantic_fields_set__
        else:
            # Fallback: assume fields that differ from pipeline defaults were explicitly set
            explicitly_set = set()
            pipeline_defaults = self.pipeline_configs.get(request.pipeline, {})
            for field in [
                "enable_expansion",
                "enable_clustering",
                "enable_personalization",
                "max_processing_time_ms",
            ]:
                request_value = getattr(request, field)
                pipeline_default = pipeline_defaults.get(field)
                if pipeline_default is not None and request_value != pipeline_default:
                    explicitly_set.add(field)

        # Override pipeline settings for explicitly set fields
        for field in explicitly_set:
            if hasattr(request, field):
                config[field] = getattr(request, field)

        return config

    async def _execute_search(
        self, query: str, request: SearchRequest, config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Execute the actual search (calls federated search if enabled)."""

        # Check if federated search is enabled
        if request.enable_federation:
            try:
                from .federated import (  # noqa: PLC0415
                    CollectionSelectionStrategy,
                    FederatedSearchRequest,
                    ResultMergingStrategy,
                    SearchMode as FedSearchMode,
                )

                # Create federated search request
                fed_request = FederatedSearchRequest(
                    query=query,
                    limit=request.limit,
                    offset=request.offset,
                    target_collections=[request.collection_name]
                    if request.collection_name
                    else None,
                    collection_selection_strategy=CollectionSelectionStrategy.SMART_ROUTING,
                    search_mode=FedSearchMode.PARALLEL,
                    result_merging_strategy=ResultMergingStrategy.SCORE_BASED,
                    timeout_ms=config.get("max_processing_time_ms", 5000.0),
                    enable_deduplication=True,
                )

                # Execute federated search
                fed_result = await self.federated_service.search(fed_request)

                # Convert federated results to standard format
                results = []
                for result in fed_result.results:
                    results.append(
                        {
                            "id": result.get("id", f"fed_{len(results)}"),
                            "title": result.get("title", "Federated Result"),
                            "content": result.get("content", ""),
                            "score": result.get("score", 0.0),
                            "metadata": result.get("metadata", {}),
                            "collection": result.get("collection", "unknown"),
                        }
                    )

                return results

            except Exception as e:
                self._logger.warning("Federated search failed")
                # Fall back to mock results

        # Default mock search implementation for non-federated search
        results = []
        for i in range(20):  # Mock 20 results
            results.append(
                {
                    "id": f"doc_{i}",
                    "title": f"Document {i} Title",  # Required by ranking/clustering
                    "content": "Result {i} for query",
                    "score": 0.9 - (i * 0.04),
                    "metadata": {"source": f"collection_{i % 3}"},
                }
            )
        return results

    def _apply_ranking(
        self, results: list[dict[str, Any]], ranking_result: Any
    ) -> list[dict[str, Any]]:
        """Apply personalized ranking to results."""
        # Create a mapping of result IDs to results
        result_map = {r["id"]: r for r in results}

        # Re-order based on ranking
        ranked_results = []
        for ranked in ranking_result.ranked_results:
            if ranked.result_id in result_map:
                result = result_map[ranked.result_id]
                result["personalized_score"] = ranked.final_score
                ranked_results.append(result)

        # Add any results not in ranking (shouldn't happen, but be safe)
        for result in results:
            if result["id"] not in [r["id"] for r in ranked_results]:
                ranked_results.append(result)

        return ranked_results

    def _get_cache_key(self, request: SearchRequest) -> str:
        """Generate cache key for request."""
        # Simple cache key based on essential parameters
        key_parts = [
            request.query,
            request.collection_name or "default",
            str(request.limit),
            str(request.offset),
            request.mode.value,
            request.user_id or "anonymous",
        ]
        return "|".join(key_parts)

    def get_stats(self) -> dict[str, Any]:
        """Get orchestrator statistics."""
        return self.stats.copy()

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self.cache.clear()
        self._logger.info("Cache cleared")

    async def initialize(self) -> None:
        """Initialize the orchestrator."""
        self._logger.info("SearchOrchestrator initialized")

    async def cleanup(self) -> None:
        """Cleanup orchestrator resources."""
        self.clear_cache()

        # Cleanup RAG generator if initialized
        if self._rag_generator and self._rag_generator._initialized:
            await self._rag_generator.cleanup()

        self._logger.info("SearchOrchestrator cleaned up")
