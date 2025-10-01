"""Federated search service for cross-collection search orchestration.

Federated search across multiple Qdrant collections with routing, result merging,
load balancing, and distributed query optimization.
"""

import asyncio
import logging
import math
import statistics
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from enum import Enum
from typing import Any, cast

import httpx
import numpy as np
from pydantic import BaseModel, Field, field_validator

from src.config import get_config
from src.config.models import QueryProcessingConfig, ScoreNormalizationStrategy
from src.services.errors import EmbeddingServiceError
from src.services.vector_db.adapter_base import VectorMatch
from src.services.vector_db.service import VectorStoreService

from .utils import (
    STOP_WORDS,
    CacheManager,
    PerformanceTracker,
    build_cache_key,
    deduplicate_results,
    merge_performance_metadata,
    performance_snapshot,
)


logger = logging.getLogger(__name__)


def _raise_no_collections_selected() -> None:
    """Raise ValueError for no collections selected for search."""
    msg = "No collections selected for search"
    raise ValueError(msg)


def _raise_insufficient_successful_collections(successful_count: int) -> None:
    """Raise ValueError for insufficient successful collections."""
    msg = f"Only {successful_count} collections succeeded, minimum required"
    raise ValueError(msg)


class SearchMode(str, Enum):
    """Search execution modes for federated queries."""

    PARALLEL = "parallel"  # Search all collections simultaneously
    SEQUENTIAL = "sequential"  # Search collections one by one
    ADAPTIVE = "adaptive"  # Adapt strategy based on conditions
    PRIORITIZED = "prioritized"  # Search high-priority collections first
    ROUND_ROBIN = "round_robin"  # Distribute queries across collections


class CollectionSelectionStrategy(str, Enum):
    """Strategies for selecting collections to search."""

    ALL = "all"  # Search all available collections
    SMART_ROUTING = "smart_routing"  # Route based on query analysis
    EXPLICIT = "explicit"  # Use explicitly specified collections
    CONTENT_BASED = "content_based"  # Select based on content type
    PERFORMANCE_BASED = "performance_based"  # Select based on performance metrics


class ResultMergingStrategy(str, Enum):
    """Strategies for merging results from multiple collections."""

    SCORE_BASED = "score_based"  # Merge based on relevance scores
    ROUND_ROBIN = "round_robin"  # Interleave results from collections
    COLLECTION_PRIORITY = "collection_priority"  # Priority-based merging
    TEMPORAL = "temporal"  # Merge based on timestamps
    DIVERSITY_OPTIMIZED = "diversity_optimized"  # Optimize for result diversity


class FederatedSearchScope(str, Enum):
    """Scope of federated search operations."""

    COMPREHENSIVE = "comprehensive"  # Search all relevant collections thoroughly
    EFFICIENT = "efficient"  # Balance coverage and performance
    TARGETED = "targeted"  # Search specific collections only
    EXPLORATORY = "exploratory"  # Search broadly for discovery


class CollectionMetadata(BaseModel):
    """Metadata about a searchable collection."""

    collection_name: str = Field(..., description="Collection identifier")
    display_name: str | None = Field(None, description="Human-readable collection name")
    description: str | None = Field(None, description="Collection description")

    # Collection characteristics
    document_count: int = Field(..., ge=0, description="Number of documents")
    vector_size: int = Field(..., ge=1, description="Vector dimensionality")
    indexed_fields: list[str] = Field(
        default_factory=list, description="Indexed metadata fields"
    )

    # Content metadata
    content_types: list[str] = Field(
        default_factory=list, description="Document types in collection"
    )
    domains: list[str] = Field(default_factory=list, description="Content domains")
    languages: list[str] = Field(
        default_factory=list, description="Supported languages"
    )

    # Performance characteristics
    avg_search_time_ms: float = Field(0.0, ge=0.0, description="Average search latency")
    availability_score: float = Field(
        1.0, ge=0.0, le=1.0, description="Collection availability"
    )
    quality_score: float = Field(
        1.0, ge=0.0, le=1.0, description="Content quality score"
    )

    # Search capabilities
    supports_hybrid_search: bool = Field(True, description="Supports hybrid search")
    supports_filtering: bool = Field(True, description="Supports metadata filtering")
    supports_clustering: bool = Field(False, description="Supports result clustering")

    # Administrative metadata
    priority: int = Field(1, ge=1, le=10, description="Collection search priority")
    last_updated: datetime = Field(
        default_factory=datetime.now, description="Last update time"
    )
    access_restrictions: dict[str, Any] = Field(
        default_factory=dict, description="Access control metadata"
    )


class CollectionSearchResult(BaseModel):
    """Results from a single collection search."""

    collection_name: str = Field(..., description="Source collection")
    results: list[dict[str, Any]] = Field(..., description="Search results")
    total_hits: int = Field(..., ge=0, description="Total matching documents")
    search_time_ms: float = Field(..., ge=0.0, description="Search execution time")

    # Quality metrics
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Result confidence"
    )
    coverage_score: float = Field(..., ge=0.0, le=1.0, description="Query coverage")

    # Search metadata
    query_used: str = Field(..., description="Actual query executed")
    filters_applied: dict[str, Any] = Field(
        default_factory=dict, description="Filters applied"
    )
    search_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Collection-specific metadata"
    )

    # Error handling
    has_errors: bool = Field(False, description="Whether errors occurred")
    error_details: dict[str, Any] = Field(
        default_factory=dict, description="Error information"
    )


class FederatedSearchRequest(BaseModel):
    """Request for federated search across collections."""

    # Core search parameters
    query: str = Field(..., description="Search query")
    vector: list[float] | None = Field(
        None, description="Query vector for similarity search"
    )
    limit: int = Field(10, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Result offset for pagination")

    # Collection selection
    target_collections: list[str] | None = Field(
        None, description="Specific collections to search"
    )
    collection_selection_strategy: CollectionSelectionStrategy = Field(
        CollectionSelectionStrategy.SMART_ROUTING,
        description="Collection selection strategy",
    )
    max_collections: int | None = Field(
        None, ge=1, le=50, description="Maximum collections to search"
    )

    # Search execution
    search_mode: SearchMode = Field(
        SearchMode.PARALLEL, description="Search execution mode"
    )
    federated_scope: FederatedSearchScope = Field(
        FederatedSearchScope.EFFICIENT, description="Search scope"
    )
    timeout_ms: float = Field(10000.0, ge=1000.0, description="Overall search timeout")
    per_collection_timeout_ms: float = Field(
        5000.0, ge=500.0, description="Per-collection timeout"
    )

    # Result processing
    result_merging_strategy: ResultMergingStrategy = Field(
        ResultMergingStrategy.SCORE_BASED, description="Result merging strategy"
    )
    enable_deduplication: bool = Field(
        True, description="Enable cross-collection deduplication"
    )
    deduplication_threshold: float = Field(
        0.9, ge=0.0, le=1.0, description="Similarity threshold for deduplication"
    )
    overfetch_multiplier: float | None = Field(
        None,
        ge=1.0,
        le=5.0,
        description="Optional multiplier applied to per-collection limits",
    )
    enable_score_normalization: bool = Field(
        True, description="Enable score normalization prior to merging results"
    )
    score_normalization_strategy: ScoreNormalizationStrategy | None = Field(
        None,
        description="Normalization strategy overriding the configured default",
    )
    score_normalization_epsilon: float | None = Field(
        None,
        gt=0.0,
        le=0.1,
        description="Variance guard used when normalizing scores",
    )

    # Quality controls
    min_collection_confidence: float = Field(
        0.3, ge=0.0, le=1.0, description="Minimum collection confidence"
    )
    require_minimum_collections: int | None = Field(
        None, ge=1, description="Minimum successful collections required"
    )

    # Filtering and processing
    global_filters: dict[str, Any] = Field(
        default_factory=dict, description="Filters applied to all collections"
    )
    collection_specific_filters: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Collection-specific filters"
    )

    # Performance settings
    enable_caching: bool = Field(True, description="Enable result caching")
    enable_load_balancing: bool = Field(True, description="Enable load balancing")
    failover_enabled: bool = Field(True, description="Enable automatic failover")

    @field_validator("target_collections")
    @classmethod
    def validate_target_collections(cls, v):
        """Validate target collection list."""
        if v is not None and len(v) == 0:
            msg = "Target collections list cannot be empty"
            raise ValueError(msg)
        return v


class FederatedSearchResult(BaseModel):
    """Result of federated search operations."""

    # Merged results
    results: list[dict[str, Any]] = Field(..., description="Merged search results")
    total_results: int = Field(
        ...,
        ge=0,
        description="Total results found",
    )

    # Collection results
    collection_results: list[CollectionSearchResult] = Field(
        ..., description="Per-collection search results"
    )
    collections_searched: list[str] = Field(
        ..., description="Collections that were searched"
    )
    collections_failed: list[str] = Field(
        default_factory=list, description="Collections that failed"
    )

    # Search metadata
    search_strategy: CollectionSelectionStrategy = Field(
        ..., description="Strategy used for collection selection"
    )
    merging_strategy: ResultMergingStrategy = Field(
        ..., description="Strategy used for result merging"
    )
    search_mode: SearchMode = Field(..., description="Search execution mode used")

    # Performance metrics
    total_search_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total search time",
    )
    fastest_collection_ms: float = Field(
        ..., ge=0.0, description="Fastest collection time"
    )
    slowest_collection_ms: float = Field(
        ..., ge=0.0, description="Slowest collection time"
    )

    # Quality metrics
    overall_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Overall result confidence"
    )
    coverage_score: float = Field(
        ..., ge=0.0, le=1.0, description="Query coverage across collections"
    )
    diversity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Result diversity score"
    )

    # Processing metadata
    deduplication_stats: dict[str, Any] = Field(
        default_factory=dict, description="Deduplication statistics"
    )
    load_balancing_stats: dict[str, Any] = Field(
        default_factory=dict, description="Load balancing information"
    )

    # Cache and performance
    cache_hit: bool = Field(False, description="Whether result was cached")
    federated_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional federated search metadata"
    )


class FederatedSearchService:
    """Advanced federated search service for cross-collection coordination."""

    def __init__(
        self,
        enable_intelligent_routing: bool = True,
        enable_adaptive_load_balancing: bool = True,
        enable_result_caching: bool = True,
        cache_size: int = 100,
        max_concurrent_searches: int = 5,
        *,
        query_processing_config: QueryProcessingConfig | None = None,
        vector_store_service: VectorStoreService | None = None,
        config=None,
    ):
        """Initialize federated search service.

        Args:
            enable_intelligent_routing: Enable smart collection routing
            enable_adaptive_load_balancing: Enable adaptive load balancing
            enable_result_caching: Enable federated result caching
            cache_size: Size of result cache
            max_concurrent_searches: Maximum concurrent collection searches
        """

        self._logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

        # Configuration
        resolved_config = config or get_config()
        self._query_processing_config = query_processing_config or getattr(
            resolved_config, "query_processing", QueryProcessingConfig()
        )
        self.enable_intelligent_routing = enable_intelligent_routing
        self.enable_adaptive_load_balancing = enable_adaptive_load_balancing
        self.enable_result_caching = enable_result_caching
        self.max_concurrent_searches = max_concurrent_searches
        self._vector_service: VectorStoreService | None = vector_store_service
        self._default_overfetch_multiplier = (
            self._query_processing_config.federated_overfetch_multiplier
        )
        self._default_normalization_enabled = (
            self._query_processing_config.enable_score_normalization
        )
        self._default_normalization_strategy = (
            self._query_processing_config.score_normalization_strategy
        )
        self._default_normalization_epsilon = (
            self._query_processing_config.score_normalization_epsilon
        )

        # Collection registry and metadata
        self.collection_registry = {}  # Collection name -> metadata
        self.collection_clients = {}  # Collection name -> Qdrant client
        self.collection_performance_stats = {}  # Performance tracking

        # Load balancing and routing
        self.collection_load_scores = {}
        self.routing_intelligence = {}

        self._cache = CacheManager(cache_size)
        self._cache_size = cache_size
        self._performance = PerformanceTracker()
        self._successful_searches = 0
        self._failed_searches = 0

    @property
    def cache_size(self) -> int:
        """Return the configured cache size."""

        return self._cache_size

    @cache_size.setter
    def cache_size(self, value: int) -> None:
        """Resize the result cache and reset trackers."""

        self._cache_size = int(value)
        self._cache = CacheManager(self._cache_size)

    @property
    def cache_stats(self) -> dict[str, int]:
        """Expose cache hit and miss counters."""

        return {
            "hits": self._cache.tracker.hits,
            "misses": self._cache.tracker.misses,
        }

    @property
    def performance_stats(self) -> dict[str, Any]:
        """Expose aggregated performance statistics."""

        return self.get_performance_stats()

    @property
    def federated_cache(self) -> dict[str, FederatedSearchResult]:
        """Return a snapshot of cached federated search results."""

        return self._cache.snapshot()

    async def search(self, request: FederatedSearchRequest) -> FederatedSearchResult:
        """Execute federated search across multiple collections.

        Args:
            request: Federated search request

        Returns:
            FederatedSearchResult with merged results and metadata

        """
        start_time = time.time()

        try:
            # Check cache first
            if request.enable_caching and self.enable_result_caching:
                cached_result = self._get_cached_result(request)
                if cached_result:
                    cached_result.cache_hit = True
                    return cached_result

            # Select collections to search
            target_collections = await self._select_collections(request)

            if not target_collections:
                _raise_no_collections_selected()

            vector_service = await self._get_vector_service()
            query_vector: Sequence[float] | None = request.vector
            if query_vector is None:
                query_vector = await vector_service.embed_query(request.query)

            # Execute searches across collections
            collection_results = await self._execute_search(
                request, target_collections, list(query_vector)
            )

            # Filter successful results
            successful_results = [
                result
                for result in collection_results
                if not result.has_errors
                and result.confidence_score >= request.min_collection_confidence
            ]

            # Check minimum collection requirement
            if (
                request.require_minimum_collections
                and len(successful_results) < request.require_minimum_collections
            ):
                _raise_insufficient_successful_collections(len(successful_results))

            # Merge and deduplicate results
            merged_results = await self._merge_results(successful_results, request)

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                successful_results, merged_results, request
            )

            total_search_time = (time.time() - start_time) * 1000

            # Build federated result
            result = FederatedSearchResult(
                results=merged_results[: request.limit],
                total_results=len(merged_results),
                collection_results=collection_results,
                collections_searched=[r.collection_name for r in successful_results],
                collections_failed=[
                    r.collection_name for r in collection_results if r.has_errors
                ],
                search_strategy=request.collection_selection_strategy,
                merging_strategy=request.result_merging_strategy,
                search_mode=request.search_mode,
                total_search_time_ms=total_search_time,
                fastest_collection_ms=min(
                    (r.search_time_ms for r in successful_results), default=0.0
                ),
                slowest_collection_ms=max(
                    (r.search_time_ms for r in successful_results), default=0.0
                ),
                overall_confidence=quality_metrics["overall_confidence"],
                coverage_score=quality_metrics["coverage_score"],
                diversity_score=quality_metrics["diversity_score"],
                deduplication_stats=quality_metrics.get("deduplication_stats", {}),
                load_balancing_stats=self._get_load_balancing_stats(target_collections),
                cache_hit=False,
                federated_metadata={
                    "target_collections_count": len(target_collections),
                    "successful_collections_count": len(successful_results),
                    "total_hits": sum(r.total_hits for r in successful_results),
                    "grouping_applied": self._should_skip_dedup(successful_results),
                    "score_normalization": quality_metrics.get(
                        "score_normalization", {}
                    ),
                    "search_efficiency": len(successful_results)
                    / len(target_collections)
                    if target_collections
                    else 0,
                },
            )

            # Cache result
            if request.enable_caching and self.enable_result_caching:
                self._cache_result(request, result)

            # Update performance stats
            self._update_performance_stats(request, result, total_search_time)

            self._logger.info(
                "Federated search completed: "
                "%d/%d collections, "
                "%d total results in %.1f ms",
                len(successful_results),
                len(target_collections),
                len(merged_results),
                total_search_time,
            )

            return result

        except Exception as e:
            total_search_time = (time.time() - start_time) * 1000
            self._logger.exception("Federated search failed")

            # Return fallback result
            return FederatedSearchResult(
                results=[],
                total_results=0,
                collection_results=[],
                collections_searched=[],
                collections_failed=[],
                search_strategy=request.collection_selection_strategy,
                merging_strategy=request.result_merging_strategy,
                search_mode=request.search_mode,
                total_search_time_ms=total_search_time,
                fastest_collection_ms=0.0,
                slowest_collection_ms=0.0,
                overall_confidence=0.0,
                coverage_score=0.0,
                diversity_score=0.0,
                cache_hit=False,
                federated_metadata={"error": str(e)},
            )

    async def register_collection(
        self, collection_name: str, metadata: CollectionMetadata, client: Any = None
    ) -> None:
        """Register a collection for federated search.

        Args:
            collection_name: Unique collection identifier
            metadata: Collection metadata and capabilities
            client: Qdrant client for the collection

        """
        try:
            self.collection_registry[collection_name] = metadata
            if client:
                self.collection_clients[collection_name] = client

            # Initialize performance tracking
            self.collection_performance_stats[collection_name] = {
                "_total_searches": 0,
                "total_searches": 0,
                "avg_latency_ms": 0.0,
                "avg_response_time": 0.0,
                "success_rate": 1.0,
                "last_updated": datetime.now(tz=UTC),
            }

            self.collection_load_scores[collection_name] = 0.0

            self._logger.info("Registered collection %s", collection_name)

        except (TimeoutError, OSError, PermissionError):
            self._logger.exception("Failed to register collection %s", collection_name)
            raise

    async def unregister_collection(self, collection_name: str) -> None:
        """Unregister a collection from federated search.

        Args:
            collection_name: Collection to unregister

        """
        try:
            self.collection_registry.pop(collection_name, None)
            self.collection_clients.pop(collection_name, None)
            self.collection_performance_stats.pop(collection_name, None)
            self.collection_load_scores.pop(collection_name, None)

            self._logger.info("Unregistered collection %s", collection_name)

        except (ConnectionError, OSError, PermissionError):
            self._logger.exception(
                "Failed to unregister collection %s", collection_name
            )

    async def _select_collections(self, request: FederatedSearchRequest) -> list[str]:
        """Select collections to search based on strategy."""
        if (
            request.collection_selection_strategy
            == CollectionSelectionStrategy.EXPLICIT
        ):
            return request.target_collections or []

        if request.collection_selection_strategy == CollectionSelectionStrategy.ALL:
            collections = list(self.collection_registry.keys())

        elif (
            request.collection_selection_strategy
            == CollectionSelectionStrategy.SMART_ROUTING
        ):
            collections = await self._smart_route_collections(request)

        elif (
            request.collection_selection_strategy
            == CollectionSelectionStrategy.CONTENT_BASED
        ):
            collections = self._select_by_content_type(request)

        elif (
            request.collection_selection_strategy
            == CollectionSelectionStrategy.PERFORMANCE_BASED
        ):
            collections = self._select_by_performance(request)

        else:
            collections = list(self.collection_registry.keys())

        # Apply max collections limit
        if request.max_collections and len(collections) > request.max_collections:
            # Sort by priority and take top collections
            sorted_collections = sorted(
                collections,
                key=lambda c: self.collection_registry[c].priority,
                reverse=True,
            )
            collections = sorted_collections[: request.max_collections]

        return collections

    async def _smart_route_collections(
        self, request: FederatedSearchRequest
    ) -> list[str]:
        """Intelligently route query to most suitable collections."""
        if not self.enable_intelligent_routing:
            return list(self.collection_registry.keys())

        suitable_collections = []
        query_lower = request.query.lower()

        for collection_name, metadata in self.collection_registry.items():
            # Calculate suitability score
            suitability_score = 0.0

            # Content type matching
            if metadata.content_types:
                for content_type in metadata.content_types:
                    if content_type.lower() in query_lower:
                        suitability_score += 0.3

            # Domain matching
            if metadata.domains:
                for domain in metadata.domains:
                    if domain.lower() in query_lower:
                        suitability_score += 0.4

            # Performance factors
            suitability_score += metadata.quality_score * 0.2
            suitability_score += metadata.availability_score * 0.1

            # Load balancing consideration
            current_load = self.collection_load_scores.get(collection_name, 0.0)
            suitability_score *= max(0.1, 1.0 - current_load)

            if suitability_score > 0.2:  # Minimum threshold
                suitable_collections.append((collection_name, suitability_score))

        # Sort by suitability and return collection names
        suitable_collections.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in suitable_collections]

    def _select_by_content_type(self, request: FederatedSearchRequest) -> list[str]:
        """Select collections based on content type analysis."""
        selected_collections = []

        # Simple content type detection from query
        query_lower = request.query.lower()
        detected_types = []

        if any(
            term in query_lower for term in ["code", "programming", "function", "api"]
        ):
            detected_types.append("code")
        if any(term in query_lower for term in ["documentation", "docs", "manual"]):
            detected_types.append("documentation")
        if any(term in query_lower for term in ["tutorial", "guide", "how to"]):
            detected_types.append("tutorial")

        for collection_name, metadata in self.collection_registry.items():
            # Check if collection supports detected content types
            if not detected_types or any(
                dt in metadata.content_types for dt in detected_types
            ):
                selected_collections.append(collection_name)

        return selected_collections or list(self.collection_registry.keys())

    def _select_by_performance(self, _request: FederatedSearchRequest) -> list[str]:
        """Select collections based on performance characteristics."""
        performance_ranked = []

        for collection_name, metadata in self.collection_registry.items():
            # Calculate performance score
            perf_stats = self.collection_performance_stats.get(collection_name, {})
            success_rate = perf_stats.get("success_rate", 1.0)
            avg_time = perf_stats.get("avg_latency_ms", metadata.avg_search_time_ms)

            # Performance score (lower time and higher success rate is better)
            performance_score = success_rate * metadata.availability_score
            if avg_time > 0:
                performance_score /= avg_time / 1000.0  # Normalize to seconds

            performance_ranked.append((collection_name, performance_score))

        # Sort by performance (higher is better)
        performance_ranked.sort(key=lambda x: x[1], reverse=True)

        return [name for name, _ in performance_ranked]

    def _get_system_load(self) -> float:
        """Return a heuristic system load metric (placeholder implementation)."""

        return 0.5

    async def _execute_search(
        self,
        request: FederatedSearchRequest,
        target_collections: list[str],
        query_vector: Sequence[float] | None,
    ) -> list[CollectionSearchResult]:
        """Execute search with adaptive system-load handling."""

        if (
            request.search_mode == SearchMode.ADAPTIVE
            and self.enable_adaptive_load_balancing
        ) and self._get_system_load() >= 0.75:
            return await self._execute_sequential_search(
                request, target_collections, query_vector
            )

        return await self._execute_federated_search(
            request, target_collections, query_vector
        )

    async def _execute_federated_search(
        self,
        request: FederatedSearchRequest,
        target_collections: list[str],
        query_vector: Sequence[float] | None,
    ) -> list[CollectionSearchResult]:
        """Execute search across target collections."""
        if request.search_mode == SearchMode.PARALLEL:
            return await self._execute_parallel_search(
                request, target_collections, query_vector
            )
        if request.search_mode == SearchMode.SEQUENTIAL:
            return await self._execute_sequential_search(
                request, target_collections, query_vector
            )
        if request.search_mode == SearchMode.PRIORITIZED:
            return await self._execute_prioritized_search(
                request, target_collections, query_vector
            )
        # ADAPTIVE or ROUND_ROBIN
        return await self._execute_adaptive_search(
            request, target_collections, query_vector
        )

    async def _execute_parallel_search(
        self,
        request: FederatedSearchRequest,
        target_collections: list[str],
        query_vector: Sequence[float] | None,
    ) -> list[CollectionSearchResult]:
        """Execute searches in parallel across collections."""
        semaphore = asyncio.Semaphore(self.max_concurrent_searches)

        async def search_collection(collection_name: str) -> CollectionSearchResult:
            async with semaphore:
                try:
                    return await self._search_single_collection(
                        collection_name, request, query_vector
                    )
                except Exception as exc:  # pragma: no cover - defensive guard
                    self._logger.exception(
                        "Collection %s search failed", collection_name
                    )
                    return self._create_error_result(str(exc), collection_name)

        # Create tasks for all collections
        tasks = [
            search_collection(collection_name) for collection_name in target_collections
        ]

        # Execute with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=request.timeout_ms / 1000.0,
            )

        except TimeoutError:
            self._logger.warning(
                "Parallel search timed out after %d ms", request.timeout_ms
            )
            return [
                self._create_error_result("Search timeout", collection)
                for collection in target_collections
            ]
        return results

    async def _execute_sequential_search(
        self,
        request: FederatedSearchRequest,
        target_collections: list[str],
        query_vector: Sequence[float] | None,
    ) -> list[CollectionSearchResult]:
        """Execute searches sequentially across collections."""
        results = []

        for collection_name in target_collections:
            try:
                result = await asyncio.wait_for(
                    self._search_single_collection(
                        collection_name, request, query_vector
                    ),
                    timeout=request.per_collection_timeout_ms / 1000.0,
                )
                results.append(result)

            except TimeoutError:
                self._logger.warning(
                    "Search timeout for collection %s", collection_name
                )
                results.append(
                    self._create_error_result("Collection timeout", collection_name)
                )
            except Exception as e:
                self._logger.exception(
                    "Search failed for collection %s", collection_name
                )
                results.append(self._create_error_result(str(e), collection_name))

        return results

    async def _execute_prioritized_search(
        self,
        request: FederatedSearchRequest,
        target_collections: list[str],
        query_vector: Sequence[float] | None,
    ) -> list[CollectionSearchResult]:
        """Execute searches with priority ordering."""
        # Sort collections by priority
        prioritized_collections = sorted(
            target_collections,
            key=lambda c: self.collection_registry[c].priority,
            reverse=True,
        )

        # Execute high-priority collections first, then parallel for others
        high_priority = []
        normal_priority = []

        for collection in prioritized_collections:
            metadata = self.collection_registry[collection]
            if metadata.priority >= 8:  # High priority threshold
                high_priority.append(collection)
            else:
                normal_priority.append(collection)

        results = []

        # Execute high priority sequentially
        if high_priority:
            sequential_results = await self._execute_sequential_search(
                request, high_priority, query_vector
            )
            results.extend(sequential_results)

        # Execute normal priority in parallel
        if normal_priority:
            parallel_results = await self._execute_parallel_search(
                request, normal_priority, query_vector
            )
            results.extend(parallel_results)

        return results

    async def _execute_adaptive_search(
        self,
        request: FederatedSearchRequest,
        target_collections: list[str],
        query_vector: Sequence[float] | None,
    ) -> list[CollectionSearchResult]:
        """Execute adaptive search based on conditions."""
        # Simple adaptive logic - use parallel for small sets, sequential for large
        if len(target_collections) <= 5:
            return await self._execute_parallel_search(
                request, target_collections, query_vector
            )
        return await self._execute_sequential_search(
            request, target_collections, query_vector
        )

    async def _search_single_collection(
        self,
        collection_name: str,
        request: FederatedSearchRequest,
        query_vector: Sequence[float] | None,
    ) -> CollectionSearchResult:
        """Search a single collection."""
        start_time = time.time()

        try:
            vector_service = await self._get_vector_service()
            vector = list(query_vector) if query_vector is not None else None
            if vector is None:
                vector = list(await vector_service.embed_query(request.query))

            effective_limit = self._calculate_collection_limit(request)
            filters = self._build_collection_filters(collection_name, request)

            matches = await vector_service.search_vector(
                collection_name,
                vector,
                limit=effective_limit,
                filters=filters,
            )

            search_time = (time.time() - start_time) * 1000
            self._update_collection_performance(collection_name, search_time, True)

            hits = [
                {
                    "id": match.id,
                    "score": float(match.score),
                    "payload": self._copy_payload(match),
                }
                for match in matches
            ]

            grouping_applied = any(
                isinstance(hit.get("payload"), dict)
                and bool((hit["payload"].get("_grouping") or {}).get("applied", False))
                for hit in hits
            )

            confidence_score = self._estimate_confidence(hits)
            coverage_score = 1.0 if hits else 0.0

            search_metadata = {
                "grouping_applied": grouping_applied,
                "fetched_limit": effective_limit,
                "raw_hit_count": len(hits),
            }

            return CollectionSearchResult(
                collection_name=collection_name,
                results=hits,
                total_hits=len(hits),
                search_time_ms=search_time,
                confidence_score=confidence_score,
                coverage_score=coverage_score,
                query_used=request.query,
                filters_applied=filters or {},
                search_metadata=search_metadata,
                has_errors=False,
            )

        except (
            httpx.HTTPError,
            httpx.RequestError,
            ConnectionError,
            TimeoutError,
            EmbeddingServiceError,
        ) as e:
            search_time = (time.time() - start_time) * 1000
            self._update_collection_performance(collection_name, search_time, False)

            return CollectionSearchResult(
                collection_name=collection_name,
                results=[],
                total_hits=0,
                search_time_ms=search_time,
                confidence_score=0.0,
                coverage_score=0.0,
                query_used=request.query,
                has_errors=True,
                error_details={"error": str(e)},
            )

    async def _merge_results(
        self,
        collection_results: list[CollectionSearchResult],
        request: FederatedSearchRequest,
    ) -> list[dict[str, Any]]:
        """Merge results from multiple collections."""
        if request.result_merging_strategy == ResultMergingStrategy.SCORE_BASED:
            return self._merge_by_score(collection_results, request)
        if request.result_merging_strategy == ResultMergingStrategy.ROUND_ROBIN:
            return self._merge_round_robin(collection_results, request)
        if request.result_merging_strategy == ResultMergingStrategy.COLLECTION_PRIORITY:
            return self._merge_by_priority(collection_results, request)
        if request.result_merging_strategy == ResultMergingStrategy.TEMPORAL:
            return self._merge_by_time(collection_results, request)
        # DIVERSITY_OPTIMIZED
        return self._merge_for_diversity(collection_results, request)

    def _merge_by_score(
        self,
        collection_results: list[CollectionSearchResult],
        request: FederatedSearchRequest,
    ) -> list[dict[str, Any]]:
        """Merge results by relevance score."""
        all_results: list[dict[str, Any]] = []
        skip_dedup = self._should_skip_dedup(collection_results)

        for result in collection_results:
            for item in self._prepare_hits(result, request):
                enhanced_item = {**item}
                enhanced_item["_collection"] = result.collection_name
                enhanced_item["_collection_confidence"] = result.confidence_score
                all_results.append(enhanced_item)

        # Sort by score (descending)
        all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        # Apply deduplication if enabled
        if request.enable_deduplication and not skip_dedup:
            all_results = self._deduplicate_results(
                all_results, request.deduplication_threshold
            )

        return all_results

    def _merge_round_robin(
        self,
        collection_results: list[CollectionSearchResult],
        request: FederatedSearchRequest,
    ) -> list[dict[str, Any]]:
        """Merge results using round-robin strategy."""
        prepared = [
            (result, self._prepare_hits(result, request))
            for result in collection_results
        ]
        merged_results: list[dict[str, Any]] = []
        max_len = max((len(hits) for _, hits in prepared), default=0)

        for i in range(max_len):
            for result, hits in prepared:
                if i < len(hits):
                    enhanced_item = {**hits[i]}
                    enhanced_item["_collection"] = result.collection_name
                    enhanced_item["_collection_confidence"] = result.confidence_score
                    merged_results.append(enhanced_item)

        if request.enable_deduplication and not self._should_skip_dedup(
            collection_results
        ):
            merged_results = self._deduplicate_results(
                merged_results, request.deduplication_threshold
            )

        return merged_results

    def _merge_by_priority(
        self,
        collection_results: list[CollectionSearchResult],
        request: FederatedSearchRequest,
    ) -> list[dict[str, Any]]:
        """Merge results by collection priority."""
        # Sort collections by priority
        sorted_results = sorted(
            collection_results,
            key=lambda r: self.collection_registry[r.collection_name].priority,
            reverse=True,
        )

        merged_results: list[dict[str, Any]] = []

        for result in sorted_results:
            priority = self.collection_registry[result.collection_name].priority
            for item in self._prepare_hits(result, request):
                enhanced_item = {**item}
                enhanced_item["_collection"] = result.collection_name
                enhanced_item["_collection_confidence"] = result.confidence_score
                enhanced_item["_collection_priority"] = priority
                merged_results.append(enhanced_item)

        if request.enable_deduplication and not self._should_skip_dedup(sorted_results):
            merged_results = self._deduplicate_results(
                merged_results, request.deduplication_threshold
            )

        return merged_results

    def _merge_by_time(
        self,
        collection_results: list[CollectionSearchResult],
        request: FederatedSearchRequest,
    ) -> list[dict[str, Any]]:
        """Merge results by temporal order."""
        all_results: list[dict[str, Any]] = []

        for result in collection_results:
            for item in self._prepare_hits(result, request):
                enhanced_item = {**item}
                enhanced_item["_collection"] = result.collection_name
                enhanced_item["_collection_confidence"] = result.confidence_score

                payload = enhanced_item.get("payload") or {}
                timestamp = (
                    payload.get("timestamp") if isinstance(payload, dict) else None
                )
                enhanced_item["_sort_timestamp"] = (
                    timestamp if timestamp else datetime.now(tz=UTC).isoformat()
                )

                all_results.append(enhanced_item)

        # Sort by timestamp (most recent first)
        all_results.sort(key=lambda x: x.get("_sort_timestamp", ""), reverse=True)

        if request.enable_deduplication and not self._should_skip_dedup(
            collection_results
        ):
            all_results = self._deduplicate_results(
                all_results, request.deduplication_threshold
            )

        return all_results

    def _merge_for_diversity(
        self,
        collection_results: list[CollectionSearchResult],
        request: FederatedSearchRequest,
    ) -> list[dict[str, Any]]:
        """Merge results optimizing for diversity."""
        all_results: list[dict[str, Any]] = []

        for result in collection_results:
            for item in self._prepare_hits(result, request):
                enhanced_item = {**item}
                enhanced_item["_collection"] = result.collection_name
                enhanced_item["_collection_confidence"] = result.confidence_score
                all_results.append(enhanced_item)

        # Simple diversity optimization: alternate between collections
        collections_seen = set()
        diverse_results = []
        remaining_results = all_results.copy()

        while remaining_results and len(diverse_results) < request.limit * 2:
            # Find next result from unseen collection
            for i, item in enumerate(remaining_results):
                collection = item["_collection"]
                if collection not in collections_seen:
                    diverse_results.append(item)
                    remaining_results.pop(i)
                    collections_seen.add(collection)
                    break
            else:
                # All collections seen, reset and continue
                collections_seen.clear()
                if remaining_results:
                    diverse_results.append(remaining_results.pop(0))

        if request.enable_deduplication and not self._should_skip_dedup(
            collection_results
        ):
            diverse_results = self._deduplicate_results(
                diverse_results, request.deduplication_threshold
            )

        return diverse_results

    async def _get_vector_service(self) -> VectorStoreService:
        """Return an initialized vector store service instance."""

        if self._vector_service is None:
            self._vector_service = VectorStoreService()

        service = self._vector_service
        initialize_candidate = getattr(service, "initialize", None)
        if callable(initialize_candidate):
            initialize_coro = cast(Callable[[], Awaitable[Any]], initialize_candidate)
            is_initialized_candidate = getattr(service, "is_initialized", None)
            needs_initialize = True
            if callable(is_initialized_candidate):
                try:
                    callable_is_initialized = cast(
                        Callable[[], bool], is_initialized_candidate
                    )
                    needs_initialize = not bool(
                        callable_is_initialized()  # pylint: disable=not-callable
                    )
                except Exception:  # pragma: no cover - defensive
                    needs_initialize = True
            if needs_initialize:
                await initialize_coro()  # pylint: disable=not-callable
        return service

    @staticmethod
    def _copy_payload(match: VectorMatch) -> dict[str, Any]:
        """Return a shallow copy of a match payload."""

        if match.payload is None:
            return {}
        if isinstance(match.payload, dict):
            return dict(match.payload)
        return dict(match.payload)

    def _calculate_collection_limit(self, request: FederatedSearchRequest) -> int:
        """Calculate per-collection limit applying configured over-fetch."""

        multiplier = self._resolve_request_value(
            request, "overfetch_multiplier", self._default_overfetch_multiplier
        )
        if multiplier is None:
            multiplier = self._default_overfetch_multiplier
        multiplier = max(1.0, float(multiplier))
        base_limit = max(1, int(request.limit))
        computed_limit = int(math.ceil(base_limit * multiplier))
        max_limit = min(base_limit * 5, 1000)
        return max(base_limit, min(computed_limit, max_limit))

    def _build_collection_filters(
        self, collection_name: str, request: FederatedSearchRequest
    ) -> dict[str, Any] | None:
        """Merge global and collection-specific filters."""

        combined: dict[str, Any] = {}
        if request.global_filters:
            combined.update(request.global_filters)
        specific = request.collection_specific_filters.get(collection_name, {})
        if specific:
            combined.update(specific)
        return combined or None

    @staticmethod
    def _estimate_confidence(hits: list[dict[str, Any]]) -> float:
        """Estimate confidence score from raw hit scores."""

        scores: list[float] = []
        for hit in hits:
            raw_score = hit.get("score")
            if isinstance(raw_score, (int, float)):
                scores.append(float(raw_score))
        if not scores:
            return 0.0
        max_score = max(scores)
        try:
            logistic = 1.0 / (1.0 + math.exp(-max_score))
        except OverflowError:  # pragma: no cover - extreme values
            logistic = 1.0 if max_score > 0 else 0.0
        return float(max(0.0, min(1.0, logistic)))

    @staticmethod
    def _is_field_explicit(request: FederatedSearchRequest, field_name: str) -> bool:
        """Return True if the request explicitly set the field value."""

        fields_set = getattr(request, "__pydantic_fields_set__", None)
        if fields_set is None:
            fields_set = getattr(request, "model_fields_set", None)
        return isinstance(fields_set, set) and field_name in fields_set

    def _resolve_request_value(
        self, request: FederatedSearchRequest, field_name: str, fallback: Any
    ) -> Any:
        """Resolve a request field honoring explicit overrides."""

        if self._is_field_explicit(request, field_name):
            return getattr(request, field_name)
        return fallback

    def _resolve_normalization_strategy(
        self, request: FederatedSearchRequest
    ) -> ScoreNormalizationStrategy:
        """Resolve the score normalization strategy for this request."""

        enabled = bool(
            self._resolve_request_value(
                request,
                "enable_score_normalization",
                self._default_normalization_enabled,
            )
        )
        if not enabled:
            return ScoreNormalizationStrategy.NONE

        strategy = self._resolve_request_value(
            request,
            "score_normalization_strategy",
            self._default_normalization_strategy,
        )
        if strategy is None:
            return self._default_normalization_strategy
        return strategy

    def _resolve_normalization_epsilon(self, request: FederatedSearchRequest) -> float:
        """Resolve epsilon guard used during normalization."""

        epsilon = self._resolve_request_value(
            request,
            "score_normalization_epsilon",
            self._default_normalization_epsilon,
        )
        if epsilon is None:
            epsilon = self._default_normalization_epsilon
        return float(epsilon)

    def _prepare_hits(
        self, result: CollectionSearchResult, request: FederatedSearchRequest
    ) -> list[dict[str, Any]]:
        """Copy and normalize result hits for downstream merging."""

        hits = [dict(item) for item in result.results]
        raw_scores: list[float] = []
        for hit in hits:
            raw_score = hit.get("score")
            if isinstance(raw_score, (int, float)):
                raw_scores.append(float(raw_score))
        if raw_scores:
            result.search_metadata.setdefault(
                "raw_score_stats",
                {"min": min(raw_scores), "max": max(raw_scores)},
            )

        strategy = self._resolve_normalization_strategy(request)
        epsilon = self._resolve_normalization_epsilon(request)
        normalized = self._normalize_scores_in_place(hits, strategy, epsilon)
        result.search_metadata["score_normalized"] = normalized
        result.search_metadata["score_normalization_strategy"] = strategy.value
        return hits

    @staticmethod
    def _should_skip_dedup(
        collection_results: list[CollectionSearchResult],
    ) -> bool:
        """Return True when server-side grouping already deduplicated results."""

        return any(
            bool(result.search_metadata.get("grouping_applied"))
            for result in collection_results
        )

    def _normalize_scores_in_place(
        self,
        hits: list[dict[str, Any]],
        strategy: ScoreNormalizationStrategy,
        epsilon: float,
    ) -> bool:
        """Normalize scores within the supplied hit list."""

        if strategy == ScoreNormalizationStrategy.NONE or not hits:
            return False

        scores: list[float] = []
        for hit in hits:
            score_value = hit.get("score")
            if isinstance(score_value, (int, float)):
                scores.append(float(score_value))
        if len(scores) < 2:
            return False

        if strategy == ScoreNormalizationStrategy.MIN_MAX:
            min_score = min(scores)
            max_score = max(scores)
            span = max_score - min_score
            if span <= epsilon:
                for hit in hits:
                    score = hit.get("score")
                    if isinstance(score, (int, float)):
                        raw = float(score)
                        hit["_raw_score"] = raw
                        hit["score"] = 1.0
                return True
            for hit in hits:
                score = hit.get("score")
                if isinstance(score, (int, float)):
                    raw = float(score)
                    hit["_raw_score"] = raw
                    hit["score"] = (raw - min_score) / span
            return True

        if strategy == ScoreNormalizationStrategy.Z_SCORE:
            mean = statistics.fmean(scores)
            stdev = statistics.pstdev(scores)
            if stdev <= epsilon:
                for hit in hits:
                    score = hit.get("score")
                    if isinstance(score, (int, float)):
                        raw = float(score)
                        hit["_raw_score"] = raw
                        hit["score"] = 0.0
                return True
            for hit in hits:
                score = hit.get("score")
                if isinstance(score, (int, float)):
                    raw = float(score)
                    hit["_raw_score"] = raw
                    hit["score"] = (raw - mean) / stdev
            return True

        return False

    def _deduplicate_results(
        self, results: list[dict[str, Any]], threshold: float
    ) -> list[dict[str, Any]]:
        """Remove duplicate results based on similarity threshold."""

        return deduplicate_results(
            results,
            content_getter=lambda item: str(item.get("payload", {}).get("content", "")),
            threshold=threshold,
            stop_words=STOP_WORDS,
            embedding_getter=lambda item: (
                np.asarray(item.get("vector"))
                if item.get("vector") is not None
                else None
            ),
        )

    def _calculate_quality_metrics(
        self,
        collection_results: list[CollectionSearchResult],
        merged_results: list[dict[str, Any]],
        request: FederatedSearchRequest,
    ) -> dict[str, Any]:
        """Calculate quality metrics for federated search."""
        if not collection_results:
            return {
                "overall_confidence": 0.0,
                "coverage_score": 0.0,
                "diversity_score": 0.0,
            }

        # Overall confidence (weighted by collection results)
        total_results = sum(len(r.results) for r in collection_results)
        if total_results > 0:
            weighted_confidence = (
                sum(r.confidence_score * len(r.results) for r in collection_results)
                / total_results
            )
        else:
            weighted_confidence = 0.0

        # Coverage score (how many collections provided results)
        target_collections = len(self.collection_registry)
        searched_collections = len(collection_results)
        coverage_score = (
            searched_collections / target_collections if target_collections > 0 else 0.0
        )

        # Diversity score (collection representation in results)
        if merged_results:
            collection_counts = {}
            for result in merged_results[:20]:  # Check top 20 results
                collection = result.get("_collection", "unknown")
                collection_counts[collection] = collection_counts.get(collection, 0) + 1

            # Calculate diversity as entropy
            total_checked = sum(collection_counts.values())
            diversity_score = 0.0
            if total_checked > 0:
                for count in collection_counts.values():
                    if count > 0:
                        prob = count / total_checked
                        diversity_score -= prob * (
                            prob**0.5
                        )  # Modified entropy calculation

                # Normalize diversity score and ensure it's positive
                max_possible_diversity = (
                    len(collection_counts) ** 0.5 if len(collection_counts) > 1 else 1.0
                )
                diversity_score = max(
                    0.0, min(1.0, abs(diversity_score) / max_possible_diversity)
                )
        else:
            diversity_score = 0.0

        grouping_applied = self._should_skip_dedup(collection_results)
        deduplication_applied = request.enable_deduplication and not grouping_applied
        normalization_applied = any(
            bool(result.search_metadata.get("score_normalized"))
            for result in collection_results
        )
        normalization_strategies = sorted(
            {
                str(strategy)
                for strategy in (
                    result.search_metadata.get("score_normalization_strategy")
                    for result in collection_results
                )
                if strategy
            }
        )

        return {
            "overall_confidence": weighted_confidence,
            "coverage_score": coverage_score,
            "diversity_score": diversity_score,
            "deduplication_stats": {
                "original_count": sum(len(r.results) for r in collection_results),
                "deduplicated_count": len(merged_results),
                "applied": deduplication_applied,
                "skip_reason": "grouping_applied" if grouping_applied else None,
            },
            "score_normalization": {
                "applied": normalization_applied,
                "strategies": normalization_strategies,
            },
        }

    def _create_error_result(
        self, error_message: str, collection_name: str = "unknown"
    ) -> CollectionSearchResult:
        """Create an error result for failed collection search."""
        return CollectionSearchResult(
            collection_name=collection_name,
            results=[],
            total_hits=0,
            search_time_ms=0.0,
            confidence_score=0.0,
            coverage_score=0.0,
            query_used="",
            has_errors=True,
            error_details={"error": error_message},
        )

    def _update_collection_performance(
        self, collection_name: str, search_time_ms: float, success: bool
    ) -> None:
        """Update performance statistics for a collection."""
        stats = self.collection_performance_stats.setdefault(
            collection_name,
            {
                "_total_searches": 0,
                "total_searches": 0,
                "avg_latency_ms": 0.0,
                "avg_response_time": 0.0,
                "success_rate": 1.0,
                "last_updated": datetime.now(tz=UTC),
            },
        )

        stats["total_searches"] += 1
        stats["_total_searches"] = stats["total_searches"]

        total = stats["total_searches"]
        current_avg = stats["avg_latency_ms"]
        stats["avg_latency_ms"] = (current_avg * (total - 1) + search_time_ms) / total
        stats["avg_response_time"] = stats["avg_latency_ms"]

        success_value = 1.0 if success else 0.0
        stats["success_rate"] = (
            (stats["success_rate"] * (total - 1)) + success_value
        ) / total

        stats["last_updated"] = datetime.now(tz=UTC)

        # Update load balancing scores
        if self.enable_adaptive_load_balancing:
            # Simple load calculation based on recent performance
            load_score = min(1.0, search_time_ms / 5000.0)  # Normalize to 5 second max
            self.collection_load_scores[collection_name] = load_score

    def _get_load_balancing_stats(
        self, target_collections: list[str]
    ) -> dict[str, Any]:
        """Get load balancing statistics."""
        return {
            "target_collections": target_collections,
            "load_scores": {
                collection: self.collection_load_scores.get(collection, 0.0)
                for collection in target_collections
            },
            "load_balancing_enabled": self.enable_adaptive_load_balancing,
        }

    def _get_cached_result(
        self, request: FederatedSearchRequest
    ) -> FederatedSearchResult | None:
        """Get cached federated search result."""
        cache_key = self._generate_cache_key(request)
        return self._cache.get(cache_key)

    def _cache_result(
        self, request: FederatedSearchRequest, result: FederatedSearchResult
    ) -> None:
        """Cache federated search result."""
        cache_key = self._generate_cache_key(request)
        self._cache.set(cache_key, result)

    def _generate_cache_key(self, request: FederatedSearchRequest) -> str:
        """Generate cache key for federated search request."""
        return build_cache_key(
            request.query,
            ",".join(sorted(request.target_collections or [])),
            request.collection_selection_strategy.value,
            request.result_merging_strategy.value,
            request.search_mode.value,
            str(request.limit),
            str(request.offset),
            str(request.enable_deduplication),
            f"{request.deduplication_threshold:.3f}",
        )

    def _update_performance_stats(
        self,
        _request: FederatedSearchRequest,
        result: FederatedSearchResult,
        search_time_ms: float,
    ) -> None:
        """Update overall performance statistics."""
        self._performance.record(search_time_ms, label=result.search_strategy)
        if result.results:
            self._successful_searches += 1
        else:
            self._failed_searches += 1

    def get_collection_registry(self) -> dict[str, CollectionMetadata]:
        """Get the current collection registry."""
        return self.collection_registry.copy()

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""
        snapshot = performance_snapshot(self._performance)
        stats = {
            "_total_searches": snapshot["total_operations"],
            "total_searches": snapshot["total_operations"],
            "avg_search_time": snapshot["avg_processing_time"],
            "successful_searches": self._successful_searches,
            "failed_searches": self._failed_searches,
            "strategy_usage": snapshot["counters"],
            "collection_performance": self.collection_performance_stats,
            "load_balancing": {
                "enabled": self.enable_adaptive_load_balancing,
                "current_loads": self.collection_load_scores,
            },
        }

        return merge_performance_metadata(
            performance_stats=stats,
            cache_tracker=self._cache.tracker,
            cache_size=len(self._cache),
        )

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
