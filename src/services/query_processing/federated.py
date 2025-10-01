"""Federated search service for cross-collection search orchestration.

This module provides sophisticated federated search capabilities enabling unified
search across multiple Qdrant collections with intelligent routing, result merging,
load balancing, and distributed query optimization.
"""

import asyncio
import logging
import time
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator


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
    total_results: int = Field(..., ge=0, description="Total results found")

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
    total_search_time_ms: float = Field(..., ge=0.0, description="Total search time")
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
        cache_size: int = 1000,
        max_concurrent_searches: int = 10,
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
        self.enable_intelligent_routing = enable_intelligent_routing
        self.enable_adaptive_load_balancing = enable_adaptive_load_balancing
        self.enable_result_caching = enable_result_caching
        self.max_concurrent_searches = max_concurrent_searches

        # Collection registry and metadata
        self.collection_registry = {}  # Collection name -> metadata
        self.collection_clients = {}  # Collection name -> Qdrant client
        self.collection_performance_stats = {}  # Performance tracking

        # Load balancing and routing
        self.collection_load_scores = {}
        self.routing_intelligence = {}

        # Caching
        self.federated_cache = {}
        self.cache_size = cache_size
        self.cache_stats = {"hits": 0, "misses": 0}

        # Performance tracking
        self.performance_stats = {
            "total_searches": 0,
            "avg_search_time": 0.0,
            "successful_searches": 0,
            "failed_searches": 0,
            "collections_usage": {},
        }

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

            # Execute searches across collections
            collection_results = await self._execute_federated_search(
                request, target_collections
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
                federated_metadata={"error": str(e)},
            )

        else:
            return result

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
                "total_searches": 0,
                "avg_response_time": 0.0,
                "success_rate": 1.0,
                "last_updated": datetime.now(tz=UTC),
            }

            self.collection_load_scores[collection_name] = 0.0

            self._logger.info("Registered collection")

        except (TimeoutError, OSError, PermissionError):
            self._logger.exception("Failed to register collection {collection_name}")
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

            self._logger.info("Unregistered collection")

        except (ConnectionError, OSError, PermissionError):
            self._logger.exception("Failed to unregister collection {collection_name}")

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
            avg_time = perf_stats.get("avg_response_time", metadata.avg_search_time_ms)

            # Performance score (lower time and higher success rate is better)
            performance_score = success_rate * metadata.availability_score
            if avg_time > 0:
                performance_score /= avg_time / 1000.0  # Normalize to seconds

            performance_ranked.append((collection_name, performance_score))

        # Sort by performance (higher is better)
        performance_ranked.sort(key=lambda x: x[1], reverse=True)

        return [name for name, _ in performance_ranked]

    async def _execute_federated_search(
        self, request: FederatedSearchRequest, target_collections: list[str]
    ) -> list[CollectionSearchResult]:
        """Execute search across target collections."""
        if request.search_mode == SearchMode.PARALLEL:
            return await self._execute_parallel_search(request, target_collections)
        if request.search_mode == SearchMode.SEQUENTIAL:
            return await self._execute_sequential_search(request, target_collections)
        if request.search_mode == SearchMode.PRIORITIZED:
            return await self._execute_prioritized_search(request, target_collections)
        # ADAPTIVE or ROUND_ROBIN
        return await self._execute_adaptive_search(request, target_collections)

    async def _execute_parallel_search(
        self, request: FederatedSearchRequest, target_collections: list[str]
    ) -> list[CollectionSearchResult]:
        """Execute searches in parallel across collections."""
        semaphore = asyncio.Semaphore(self.max_concurrent_searches)

        async def search_collection(collection_name: str) -> CollectionSearchResult:
            async with semaphore:
                return await self._search_single_collection(collection_name, request)

        # Create tasks for all collections
        tasks = [
            search_collection(collection_name) for collection_name in target_collections
        ]

        # Execute with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=request.timeout_ms / 1000.0,
            )

            # Process results and handle exceptions
            collection_results = []
            for result in results:
                if isinstance(result, Exception):
                    # Create error result
                    collection_results.append(self._create_error_result(str(result)))
                else:
                    collection_results.append(result)

        except TimeoutError:
            self._logger.warning(
                "Parallel search timed out after %d ms", request.timeout_ms
            )
            return [
                self._create_error_result("Search timeout") for _ in target_collections
            ]

        else:
            return collection_results

    async def _execute_sequential_search(
        self, request: FederatedSearchRequest, target_collections: list[str]
    ) -> list[CollectionSearchResult]:
        """Execute searches sequentially across collections."""
        results = []

        for collection_name in target_collections:
            try:
                result = await asyncio.wait_for(
                    self._search_single_collection(collection_name, request),
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
                self._logger.exception("Search failed for collection {collection_name}")
                results.append(self._create_error_result(str(e), collection_name))

        return results

    async def _execute_prioritized_search(
        self, request: FederatedSearchRequest, target_collections: list[str]
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
                request, high_priority
            )
            results.extend(sequential_results)

        # Execute normal priority in parallel
        if normal_priority:
            parallel_results = await self._execute_parallel_search(
                request, normal_priority
            )
            results.extend(parallel_results)

        return results

    async def _execute_adaptive_search(
        self, request: FederatedSearchRequest, target_collections: list[str]
    ) -> list[CollectionSearchResult]:
        """Execute adaptive search based on conditions."""
        # Simple adaptive logic - use parallel for small sets, sequential for large
        if len(target_collections) <= 5:
            return await self._execute_parallel_search(request, target_collections)
        return await self._execute_sequential_search(request, target_collections)

    async def _search_single_collection(
        self, collection_name: str, request: FederatedSearchRequest
    ) -> CollectionSearchResult:
        """Search a single collection."""
        start_time = time.time()

        try:
            # This would integrate with actual Qdrant client
            # For now, creating a mock implementation

            # Simulate search execution
            await asyncio.sleep(0.1)  # Simulate network latency

            # Mock results
            mock_results = [
                {
                    "id": f"{collection_name}_result_{i}",
                    "score": 0.9 - (i * 0.1),
                    "payload": {
                        "title": f"Result {i} from {collection_name}",
                        "content": f"Mock content from collection {collection_name}",
                        "collection": collection_name,
                    },
                }
                for i in range(min(request.limit, 5))
            ]

            search_time = (time.time() - start_time) * 1000

            # Update collection performance stats
            self._update_collection_performance(collection_name, search_time, True)

            return CollectionSearchResult(
                collection_name=collection_name,
                results=mock_results,
                total_hits=len(mock_results),
                search_time_ms=search_time,
                confidence_score=0.85,
                coverage_score=0.9,
                query_used=request.query,
                filters_applied=request.global_filters,
                search_metadata={"mock": True},
                has_errors=False,
            )

        except (
            httpx.HTTPError,
            httpx.RequestError,
            ConnectionError,
            TimeoutError,
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
        all_results = []

        for result in collection_results:
            for item in result.results:
                # Add collection context to result
                enhanced_item = {**item}
                enhanced_item["_collection"] = result.collection_name
                enhanced_item["_collection_confidence"] = result.confidence_score
                all_results.append(enhanced_item)

        # Sort by score (descending)
        all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        # Apply deduplication if enabled
        if request.enable_deduplication:
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
        merged_results = []
        max_len = max((len(r.results) for r in collection_results), default=0)

        for i in range(max_len):
            for result in collection_results:
                if i < len(result.results):
                    enhanced_item = {**result.results[i]}
                    enhanced_item["_collection"] = result.collection_name
                    enhanced_item["_collection_confidence"] = result.confidence_score
                    merged_results.append(enhanced_item)

        if request.enable_deduplication:
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

        merged_results = []
        for result in sorted_results:
            for item in result.results:
                enhanced_item = {**item}
                enhanced_item["_collection"] = result.collection_name
                enhanced_item["_collection_confidence"] = result.confidence_score
                enhanced_item["_collection_priority"] = self.collection_registry[
                    result.collection_name
                ].priority
                merged_results.append(enhanced_item)

        if request.enable_deduplication:
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
        all_results = []

        for result in collection_results:
            for item in result.results:
                enhanced_item = {**item}
                enhanced_item["_collection"] = result.collection_name
                enhanced_item["_collection_confidence"] = result.confidence_score

                # Extract timestamp if available
                timestamp = item.get("payload", {}).get("timestamp")
                if timestamp:
                    enhanced_item["_sort_timestamp"] = timestamp
                else:
                    enhanced_item["_sort_timestamp"] = datetime.now(tz=UTC).isoformat()

                all_results.append(enhanced_item)

        # Sort by timestamp (most recent first)
        all_results.sort(key=lambda x: x.get("_sort_timestamp", ""), reverse=True)

        if request.enable_deduplication:
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
        all_results = []

        for result in collection_results:
            for item in result.results:
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

        if request.enable_deduplication:
            diverse_results = self._deduplicate_results(
                diverse_results, request.deduplication_threshold
            )

        return diverse_results

    def _deduplicate_results(
        self, results: list[dict[str, Any]], threshold: float
    ) -> list[dict[str, Any]]:
        """Remove duplicate results based on similarity threshold."""
        if threshold >= 1.0:
            return results

        deduplicated = []

        for result in results:
            is_duplicate = False
            result_content = str(result.get("payload", {}).get("content", "")).lower()

            for existing in deduplicated:
                existing_content = str(
                    existing.get("payload", {}).get("content", "")
                ).lower()

                # Simple similarity check based on content overlap
                if result_content and existing_content:
                    # Calculate Jaccard similarity
                    result_words = set(result_content.split())
                    existing_words = set(existing_content.split())

                    if result_words and existing_words:
                        intersection = len(result_words & existing_words)
                        union = len(result_words | existing_words)
                        similarity = intersection / union if union > 0 else 0.0

                        if similarity >= threshold:
                            is_duplicate = True
                            break

            if not is_duplicate:
                deduplicated.append(result)

        return deduplicated

    def _calculate_quality_metrics(
        self,
        collection_results: list[CollectionSearchResult],
        merged_results: list[dict[str, Any]],
        _request: FederatedSearchRequest,
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

        return {
            "overall_confidence": weighted_confidence,
            "coverage_score": coverage_score,
            "diversity_score": diversity_score,
            "deduplication_stats": {
                "original_count": sum(len(r.results) for r in collection_results),
                "deduplicated_count": len(merged_results),
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
        if collection_name not in self.collection_performance_stats:
            self.collection_performance_stats[collection_name] = {
                "total_searches": 0,
                "avg_response_time": 0.0,
                "success_rate": 1.0,
                "last_updated": datetime.now(tz=UTC),
            }

        stats = self.collection_performance_stats[collection_name]
        stats["total_searches"] += 1

        # Update average response time
        total = stats["total_searches"]
        current_avg = stats["avg_response_time"]
        stats["avg_response_time"] = (
            current_avg * (total - 1) + search_time_ms
        ) / total

        # Update success rate
        if success:
            stats["success_rate"] = (stats["success_rate"] * (total - 1) + 1.0) / total
        else:
            stats["success_rate"] = (stats["success_rate"] * (total - 1) + 0.0) / total

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

        if cache_key in self.federated_cache:
            self.cache_stats["hits"] += 1
            return self.federated_cache[cache_key]

        self.cache_stats["misses"] += 1
        return None

    def _cache_result(
        self, request: FederatedSearchRequest, result: FederatedSearchResult
    ) -> None:
        """Cache federated search result."""
        if len(self.federated_cache) >= self.cache_size:
            # Simple LRU eviction
            oldest_key = next(iter(self.federated_cache))
            del self.federated_cache[oldest_key]

        cache_key = self._generate_cache_key(request)
        self.federated_cache[cache_key] = result

    def _generate_cache_key(self, request: FederatedSearchRequest) -> str:
        """Generate cache key for federated search request."""
        key_components = [
            request.query,
            str(sorted(request.target_collections or [])),
            request.collection_selection_strategy.value,
            request.result_merging_strategy.value,
            str(request.limit),
            str(request.offset),
        ]
        return "|".join(key_components)

    def _update_performance_stats(
        self,
        _request: FederatedSearchRequest,
        result: FederatedSearchResult,
        search_time_ms: float,
    ) -> None:
        """Update overall performance statistics."""
        self.performance_stats["total_searches"] += 1

        if result.results:
            self.performance_stats["successful_searches"] += 1
        else:
            self.performance_stats["failed_searches"] += 1

        # Update average search time
        total = self.performance_stats["total_searches"]
        current_avg = self.performance_stats["avg_search_time"]
        self.performance_stats["avg_search_time"] = (
            current_avg * (total - 1) + search_time_ms
        ) / total

        # Update collection usage stats
        for collection in result.collections_searched:
            usage_stats = self.performance_stats["collections_usage"]
            usage_stats[collection] = usage_stats.get(collection, 0) + 1

    def get_collection_registry(self) -> dict[str, CollectionMetadata]:
        """Get the current collection registry."""
        return self.collection_registry.copy()

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            **self.performance_stats,
            "cache_stats": self.cache_stats,
            "collection_performance": self.collection_performance_stats,
            "load_balancing": {
                "enabled": self.enable_adaptive_load_balancing,
                "current_loads": self.collection_load_scores,
            },
        }

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.federated_cache.clear()
        self.cache_stats = {"hits": 0, "misses": 0}
