"""Advanced search orchestrator for unified filtering and query processing.

This module provides a sophisticated search orchestrator that combines advanced filtering
capabilities with query processing features including expansion, clustering, personalized
ranking, and federated search to deliver comprehensive search experiences.
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

from ..vector_db.filters.composer import FilterComposer
from ..vector_db.filters.content_type import ContentTypeFilter
from ..vector_db.filters.metadata import MetadataFilter
from ..vector_db.filters.similarity import SimilarityThresholdManager
from ..vector_db.filters.temporal import TemporalFilter
from .clustering import ResultClusteringRequest
from .clustering import ResultClusteringService
from .expansion import QueryExpansionRequest
from .expansion import QueryExpansionService
from .federated import FederatedSearchRequest
from .federated import FederatedSearchService
from .ranking import PersonalizedRankingRequest
from .ranking import PersonalizedRankingService

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    """Search execution modes."""

    SIMPLE = "simple"  # Basic search without advanced features
    ENHANCED = "enhanced"  # Search with filtering and expansion
    INTELLIGENT = "intelligent"  # AI-driven optimization
    FEDERATED = "federated"  # Cross-collection search
    PERSONALIZED = "personalized"  # User-specific optimization
    COMPREHENSIVE = "comprehensive"  # All features enabled


class ProcessingStage(str, Enum):
    """Processing stages in the search pipeline."""

    PREPROCESSING = "preprocessing"  # Query preprocessing and validation
    EXPANSION = "expansion"  # Query expansion and enhancement
    FILTERING = "filtering"  # Filter application and optimization
    EXECUTION = "execution"  # Core search execution
    CLUSTERING = "clustering"  # Result clustering and grouping
    RANKING = "ranking"  # Personalized ranking and scoring
    FEDERATION = "federation"  # Cross-collection coordination
    POSTPROCESSING = "postprocessing"  # Final result processing


class SearchPipeline(str, Enum):
    """Predefined search pipeline configurations."""

    FAST = "fast"  # Optimized for speed
    BALANCED = "balanced"  # Balance between speed and quality
    COMPREHENSIVE = "comprehensive"  # Maximum quality and features
    DISCOVERY = "discovery"  # Exploration and diversity focus
    PRECISION = "precision"  # High precision and accuracy
    PERSONALIZED = "personalized"  # User-centric optimization


class AdvancedSearchRequest(BaseModel):
    """Comprehensive search request with advanced options."""

    # Core search parameters
    query: str = Field(..., description="Search query")
    collection_name: str | None = Field(None, description="Target collection name")
    limit: int = Field(10, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Result offset for pagination")

    # Search mode and pipeline
    search_mode: SearchMode = Field(SearchMode.ENHANCED, description="Search execution mode")
    pipeline: SearchPipeline = Field(SearchPipeline.BALANCED, description="Processing pipeline")

    # Filtering criteria
    temporal_criteria: dict[str, Any] | None = Field(None, description="Temporal filtering criteria")
    content_type_criteria: dict[str, Any] | None = Field(None, description="Content type filtering criteria")
    metadata_criteria: dict[str, Any] | None = Field(None, description="Metadata filtering criteria")
    similarity_threshold_criteria: dict[str, Any] | None = Field(None, description="Similarity threshold criteria")
    filter_composition: dict[str, Any] | None = Field(None, description="Filter composition logic")

    # Query processing options
    enable_expansion: bool = Field(True, description="Enable query expansion")
    expansion_config: dict[str, Any] | None = Field(None, description="Query expansion configuration")

    enable_clustering: bool = Field(False, description="Enable result clustering")
    clustering_config: dict[str, Any] | None = Field(None, description="Clustering configuration")

    enable_personalization: bool = Field(False, description="Enable personalized ranking")
    ranking_config: dict[str, Any] | None = Field(None, description="Ranking configuration")

    enable_federation: bool = Field(False, description="Enable federated search")
    federation_config: dict[str, Any] | None = Field(None, description="Federated search configuration")

    # User context
    user_id: str | None = Field(None, description="User identifier for personalization")
    session_id: str | None = Field(None, description="Session identifier")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context")

    # Performance and quality controls
    max_processing_time_ms: float = Field(5000.0, ge=100.0, description="Maximum processing time")
    enable_caching: bool = Field(True, description="Enable result caching")
    quality_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Minimum quality threshold")
    diversity_factor: float = Field(0.1, ge=0.0, le=1.0, description="Result diversity importance")

    # Pipeline customization
    skip_stages: list[ProcessingStage] | None = Field(None, description="Stages to skip")
    stage_timeouts: dict[str, float] | None = Field(None, description="Per-stage timeout overrides")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """Validate search query."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class StageResult(BaseModel):
    """Result from a processing stage."""

    stage: ProcessingStage = Field(..., description="Processing stage")
    success: bool = Field(..., description="Whether stage completed successfully")
    processing_time_ms: float = Field(..., ge=0.0, description="Stage processing time")
    results_count: int = Field(..., ge=0, description="Number of results after stage")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Stage-specific metadata")
    error_details: dict[str, Any] | None = Field(None, description="Error information if failed")


class AdvancedSearchResult(BaseModel):
    """Comprehensive search result with pipeline metadata."""

    # Core results
    results: list[dict[str, Any]] = Field(..., description="Search results")
    total_results: int = Field(..., ge=0, description="Total results found")

    # Search metadata
    search_mode: SearchMode = Field(..., description="Search mode used")
    pipeline: SearchPipeline = Field(..., description="Pipeline configuration used")
    query_processed: str = Field(..., description="Final processed query")

    # Processing pipeline results
    stage_results: list[StageResult] = Field(..., description="Results from each processing stage")
    total_processing_time_ms: float = Field(..., ge=0.0, description="Total processing time")

    # Quality metrics
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall result quality")
    diversity_score: float = Field(..., ge=0.0, le=1.0, description="Result diversity")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Average relevance score")

    # Feature usage
    features_used: list[str] = Field(..., description="Advanced features that were applied")
    optimizations_applied: list[str] = Field(..., description="Performance optimizations used")

    # Performance and caching
    cache_hit: bool = Field(False, description="Whether result was cached")
    performance_metrics: dict[str, Any] = Field(
        default_factory=dict, description="Detailed performance metrics"
    )

    # Search metadata
    search_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional search metadata"
    )


class AdvancedSearchOrchestrator:
    """Unified orchestrator for advanced search with filtering and query processing."""

    def __init__(
        self,
        enable_all_features: bool = True,
        enable_performance_optimization: bool = True,
        cache_size: int = 1000,
        max_concurrent_stages: int = 5
    ):
        """Initialize advanced search orchestrator.
        
        Args:
            enable_all_features: Enable all advanced features by default
            enable_performance_optimization: Enable performance optimizations
            cache_size: Size of result cache
            max_concurrent_stages: Maximum concurrent processing stages
        """
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        # Configuration
        self.enable_all_features = enable_all_features
        self.enable_performance_optimization = enable_performance_optimization
        self.max_concurrent_stages = max_concurrent_stages

        # Initialize component services
        self._initialize_services()

        # Pipeline configurations
        self.pipeline_configs = self._initialize_pipeline_configs()

        # Caching
        self.search_cache = {}
        self.cache_size = cache_size
        self.cache_stats = {"hits": 0, "misses": 0}

        # Performance tracking
        self.performance_stats = {
            "total_searches": 0,
            "avg_processing_time": 0.0,
            "feature_usage": {},
            "pipeline_usage": {},
            "stage_performance": {}
        }

        # Quality tracking
        self.quality_stats = {
            "avg_quality_score": 0.0,
            "avg_diversity_score": 0.0,
            "avg_relevance_score": 0.0
        }

    def _initialize_services(self):
        """Initialize component services."""
        # Filter services
        self.temporal_filter = TemporalFilter()
        self.content_type_filter = ContentTypeFilter()
        self.metadata_filter = MetadataFilter()
        self.similarity_threshold_manager = SimilarityThresholdManager()
        self.filter_composer = FilterComposer()

        # Query processing services
        self.query_expansion_service = QueryExpansionService()
        self.clustering_service = ResultClusteringService()
        self.ranking_service = PersonalizedRankingService()
        self.federated_service = FederatedSearchService()

        self._logger.info("Initialized all component services")

    def _initialize_pipeline_configs(self) -> dict[str, dict[str, Any]]:
        """Initialize predefined pipeline configurations."""
        return {
            SearchPipeline.FAST.value: {
                "enable_expansion": False,
                "enable_clustering": False,
                "enable_personalization": False,
                "enable_federation": False,
                "max_processing_time_ms": 1000.0,
                "quality_threshold": 0.4
            },
            SearchPipeline.BALANCED.value: {
                "enable_expansion": True,
                "enable_clustering": False,
                "enable_personalization": False,
                "enable_federation": False,
                "max_processing_time_ms": 3000.0,
                "quality_threshold": 0.6
            },
            SearchPipeline.COMPREHENSIVE.value: {
                "enable_expansion": True,
                "enable_clustering": True,
                "enable_personalization": True,
                "enable_federation": True,
                "max_processing_time_ms": 10000.0,
                "quality_threshold": 0.8
            },
            SearchPipeline.DISCOVERY.value: {
                "enable_expansion": True,
                "enable_clustering": True,
                "enable_personalization": False,
                "enable_federation": True,
                "diversity_factor": 0.3,
                "max_processing_time_ms": 5000.0
            },
            SearchPipeline.PRECISION.value: {
                "enable_expansion": False,
                "enable_clustering": False,
                "enable_personalization": True,
                "enable_federation": False,
                "quality_threshold": 0.9,
                "max_processing_time_ms": 2000.0
            },
            SearchPipeline.PERSONALIZED.value: {
                "enable_expansion": True,
                "enable_clustering": False,
                "enable_personalization": True,
                "enable_federation": False,
                "max_processing_time_ms": 4000.0
            }
        }

    async def search(self, request: AdvancedSearchRequest) -> AdvancedSearchResult:
        """Execute advanced search with comprehensive pipeline processing.
        
        Args:
            request: Advanced search request
            
        Returns:
            AdvancedSearchResult with processed results and metadata
        """
        import time

        start_time = time.time()
        stage_results = []

        try:
            # Check cache first
            if request.enable_caching:
                cached_result = self._get_cached_result(request)
                if cached_result:
                    cached_result.cache_hit = True
                    return cached_result

            # Apply pipeline configuration
            effective_config = self._apply_pipeline_config(request)

            # Stage 1: Preprocessing
            stage_result = await self._execute_preprocessing_stage(request, effective_config)
            stage_results.append(stage_result)

            if not stage_result.success:
                return self._build_error_result(request, stage_results, "Preprocessing failed")

            processed_query = stage_result.metadata.get("processed_query", request.query)

            # Stage 2: Query Expansion (if enabled)
            if effective_config.get("enable_expansion", True) and ProcessingStage.EXPANSION not in (request.skip_stages or []):
                stage_result = await self._execute_expansion_stage(processed_query, request, effective_config)
                stage_results.append(stage_result)

                if stage_result.success:
                    processed_query = stage_result.metadata.get("expanded_query", processed_query)

            # Stage 3: Filter Application
            if ProcessingStage.FILTERING not in (request.skip_stages or []):
                stage_result = await self._execute_filtering_stage(request, effective_config)
                stage_results.append(stage_result)

                if not stage_result.success:
                    self._logger.warning("Filtering stage failed, continuing with base search")

            # Stage 4: Core Search Execution
            stage_result = await self._execute_search_stage(processed_query, request, effective_config)
            stage_results.append(stage_result)

            if not stage_result.success:
                return self._build_error_result(request, stage_results, "Search execution failed")

            search_results = stage_result.metadata.get("results", [])

            # Stage 5: Result Clustering (if enabled)
            if effective_config.get("enable_clustering", False) and ProcessingStage.CLUSTERING not in (request.skip_stages or []):
                stage_result = await self._execute_clustering_stage(search_results, request, effective_config)
                stage_results.append(stage_result)

                if stage_result.success:
                    search_results = stage_result.metadata.get("clustered_results", search_results)

            # Stage 6: Personalized Ranking (if enabled)
            if effective_config.get("enable_personalization", False) and ProcessingStage.RANKING not in (request.skip_stages or []):
                stage_result = await self._execute_ranking_stage(search_results, request, effective_config)
                stage_results.append(stage_result)

                if stage_result.success:
                    search_results = stage_result.metadata.get("ranked_results", search_results)

            # Stage 7: Federation (if enabled)
            if effective_config.get("enable_federation", False) and ProcessingStage.FEDERATION not in (request.skip_stages or []):
                stage_result = await self._execute_federation_stage(processed_query, request, effective_config)
                stage_results.append(stage_result)

                if stage_result.success:
                    federated_results = stage_result.metadata.get("federated_results", [])
                    search_results = self._merge_federated_results(search_results, federated_results)

            # Stage 8: Post-processing
            stage_result = await self._execute_postprocessing_stage(search_results, request, effective_config)
            stage_results.append(stage_result)

            if stage_result.success:
                final_results = stage_result.metadata.get("final_results", search_results)
            else:
                final_results = search_results

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(final_results, request)

            total_processing_time = (time.time() - start_time) * 1000

            # Build comprehensive result
            result = AdvancedSearchResult(
                results=final_results[:request.limit],
                total_results=len(final_results),
                search_mode=request.search_mode,
                pipeline=request.pipeline,
                query_processed=processed_query,
                stage_results=stage_results,
                total_processing_time_ms=total_processing_time,
                quality_score=quality_metrics["quality_score"],
                diversity_score=quality_metrics["diversity_score"],
                relevance_score=quality_metrics["relevance_score"],
                features_used=self._get_features_used(effective_config, stage_results),
                optimizations_applied=self._get_optimizations_applied(effective_config),
                cache_hit=False,
                performance_metrics={
                    "stage_count": len(stage_results),
                    "successful_stages": sum(1 for sr in stage_results if sr.success),
                    "avg_stage_time": sum(sr.processing_time_ms for sr in stage_results) / len(stage_results) if stage_results else 0,
                    "processing_efficiency": len(final_results) / total_processing_time if total_processing_time > 0 else 0
                },
                search_metadata={
                    "original_query": request.query,
                    "pipeline_config": effective_config,
                    "user_context": bool(request.user_id),
                    "federation_enabled": effective_config.get("enable_federation", False)
                }
            )

            # Cache result
            if request.enable_caching:
                self._cache_result(request, result)

            # Update performance stats
            self._update_performance_stats(request, result, total_processing_time)

            self._logger.info(
                f"Advanced search completed: {len(final_results)} results, "
                f"{len(stage_results)} stages, {total_processing_time:.1f}ms"
            )

            return result

        except Exception as e:
            total_processing_time = (time.time() - start_time) * 1000
            self._logger.error(f"Advanced search failed: {e}", exc_info=True)

            return self._build_error_result(
                request, stage_results, f"Search pipeline failed: {e}",
                total_processing_time
            )

    async def _execute_preprocessing_stage(
        self, request: AdvancedSearchRequest, config: dict[str, Any]
    ) -> StageResult:
        """Execute preprocessing stage."""
        import time

        start_time = time.time()

        try:
            # Query validation and normalization
            processed_query = request.query.strip().lower()

            # Query enhancement based on context
            if request.context:
                processed_query = self._enhance_query_with_context(processed_query, request.context)

            processing_time = (time.time() - start_time) * 1000

            return StageResult(
                stage=ProcessingStage.PREPROCESSING,
                success=True,
                processing_time_ms=processing_time,
                results_count=1,  # One processed query
                metadata={
                    "processed_query": processed_query,
                    "original_query": request.query,
                    "context_enhanced": bool(request.context)
                }
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000

            return StageResult(
                stage=ProcessingStage.PREPROCESSING,
                success=False,
                processing_time_ms=processing_time,
                results_count=0,
                error_details={"error": str(e)}
            )

    async def _execute_expansion_stage(
        self, query: str, request: AdvancedSearchRequest, config: dict[str, Any]
    ) -> StageResult:
        """Execute query expansion stage."""
        import time

        start_time = time.time()

        try:
            # Build expansion request
            expansion_request = QueryExpansionRequest(
                original_query=query,
                query_context=request.context,
                **config.get("expansion_config", {})
            )

            # Apply query expansion
            expansion_result = await self.query_expansion_service.expand_query(expansion_request)

            processing_time = (time.time() - start_time) * 1000

            return StageResult(
                stage=ProcessingStage.EXPANSION,
                success=True,
                processing_time_ms=processing_time,
                results_count=len(expansion_result.expanded_terms),
                metadata={
                    "expanded_query": expansion_result.expanded_query,
                    "expanded_terms": len(expansion_result.expanded_terms),
                    "expansion_confidence": expansion_result.confidence_score,
                    "expansion_strategy": expansion_result.expansion_strategy.value
                }
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000

            return StageResult(
                stage=ProcessingStage.EXPANSION,
                success=False,
                processing_time_ms=processing_time,
                results_count=0,
                error_details={"error": str(e)}
            )

    async def _execute_filtering_stage(
        self, request: AdvancedSearchRequest, config: dict[str, Any]
    ) -> StageResult:
        """Execute filtering stage."""
        import time

        start_time = time.time()

        try:
            applied_filters = []
            filter_conditions = []

            # Apply temporal filtering
            if request.temporal_criteria:
                filter_result = await self.temporal_filter.apply(
                    request.temporal_criteria, request.context
                )
                if filter_result.filter_conditions:
                    filter_conditions.append(filter_result.filter_conditions)
                    applied_filters.append("temporal")

            # Apply content type filtering
            if request.content_type_criteria:
                filter_result = await self.content_type_filter.apply(
                    request.content_type_criteria, request.context
                )
                if filter_result.filter_conditions:
                    filter_conditions.append(filter_result.filter_conditions)
                    applied_filters.append("content_type")

            # Apply metadata filtering
            if request.metadata_criteria:
                filter_result = await self.metadata_filter.apply(
                    request.metadata_criteria, request.context
                )
                if filter_result.filter_conditions:
                    filter_conditions.append(filter_result.filter_conditions)
                    applied_filters.append("metadata")

            # Apply similarity threshold management
            if request.similarity_threshold_criteria:
                await self.similarity_threshold_manager.optimize_threshold(
                    request.similarity_threshold_criteria, request.context
                )
                applied_filters.append("similarity_threshold")

            # Compose filters if multiple are present
            composed_filter = None
            if len(filter_conditions) > 1 and request.filter_composition:
                composition_result = await self.filter_composer.compose_filters(
                    request.filter_composition, request.context
                )
                if composition_result.filter_conditions:
                    composed_filter = composition_result.filter_conditions
                    applied_filters.append("composition")

            processing_time = (time.time() - start_time) * 1000

            return StageResult(
                stage=ProcessingStage.FILTERING,
                success=True,
                processing_time_ms=processing_time,
                results_count=len(filter_conditions),
                metadata={
                    "applied_filters": applied_filters,
                    "filter_conditions": filter_conditions,
                    "composed_filter": composed_filter is not None,
                    "total_filters": len(filter_conditions)
                }
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000

            return StageResult(
                stage=ProcessingStage.FILTERING,
                success=False,
                processing_time_ms=processing_time,
                results_count=0,
                error_details={"error": str(e)}
            )

    async def _execute_search_stage(
        self, query: str, request: AdvancedSearchRequest, config: dict[str, Any]
    ) -> StageResult:
        """Execute core search stage."""
        import time

        start_time = time.time()

        try:
            # This would integrate with the actual vector search service
            # For now, creating mock results to demonstrate the pipeline

            # Simulate search execution
            await asyncio.sleep(0.1)  # Simulate search latency

            # Mock search results
            mock_results = [
                {
                    "id": f"result_{i}",
                    "title": f"Search Result {i}",
                    "content": f"Content for result {i} matching query: {query}",
                    "score": 0.9 - (i * 0.1),
                    "content_type": "documentation" if i % 2 == 0 else "code",
                    "published_date": "2024-01-01T00:00:00Z",
                    "metadata": {
                        "source": "mock_search",
                        "processing_stage": "core_search"
                    }
                }
                for i in range(min(request.limit * 2, 20))  # Generate more than needed for processing
            ]

            processing_time = (time.time() - start_time) * 1000

            return StageResult(
                stage=ProcessingStage.EXECUTION,
                success=True,
                processing_time_ms=processing_time,
                results_count=len(mock_results),
                metadata={
                    "results": mock_results,
                    "query_used": query,
                    "collection": request.collection_name,
                    "search_type": "vector_search"
                }
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000

            return StageResult(
                stage=ProcessingStage.EXECUTION,
                success=False,
                processing_time_ms=processing_time,
                results_count=0,
                error_details={"error": str(e)}
            )

    async def _execute_clustering_stage(
        self, results: list[dict[str, Any]], request: AdvancedSearchRequest, config: dict[str, Any]
    ) -> StageResult:
        """Execute result clustering stage."""
        import time

        start_time = time.time()

        try:
            # Build clustering request
            clustering_request = ResultClusteringRequest(
                results=[
                    {
                        "id": result["id"],
                        "content": result.get("content", ""),
                        "metadata": result.get("metadata", {})
                    }
                    for result in results
                ],
                **config.get("clustering_config", {})
            )

            # Apply clustering
            clustering_result = await self.clustering_service.cluster_results(clustering_request)

            # Enhance results with cluster information
            clustered_results = []
            for result in results:
                enhanced_result = result.copy()

                # Find cluster for this result
                for cluster in clustering_result.clusters:
                    if result["id"] in [item.get("id") for item in cluster.items]:
                        enhanced_result["cluster_id"] = cluster.cluster_id
                        enhanced_result["cluster_label"] = cluster.label
                        enhanced_result["cluster_score"] = cluster.coherence_score
                        break

                clustered_results.append(enhanced_result)

            processing_time = (time.time() - start_time) * 1000

            return StageResult(
                stage=ProcessingStage.CLUSTERING,
                success=True,
                processing_time_ms=processing_time,
                results_count=len(clustered_results),
                metadata={
                    "clustered_results": clustered_results,
                    "clusters_found": len(clustering_result.clusters),
                    "clustering_algorithm": clustering_result.algorithm_used.value,
                    "clustering_confidence": clustering_result.overall_coherence
                }
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000

            return StageResult(
                stage=ProcessingStage.CLUSTERING,
                success=False,
                processing_time_ms=processing_time,
                results_count=len(results),
                error_details={"error": str(e)}
            )

    async def _execute_ranking_stage(
        self, results: list[dict[str, Any]], request: AdvancedSearchRequest, config: dict[str, Any]
    ) -> StageResult:
        """Execute personalized ranking stage."""
        import time

        start_time = time.time()

        try:
            if not request.user_id:
                # Skip personalization if no user context
                return StageResult(
                    stage=ProcessingStage.RANKING,
                    success=True,
                    processing_time_ms=0.0,
                    results_count=len(results),
                    metadata={"ranked_results": results, "personalization": "skipped_no_user"}
                )

            # Build ranking request
            ranking_request = PersonalizedRankingRequest(
                user_id=request.user_id,
                session_id=request.session_id,
                query=request.query,
                results=results,
                context=request.context,
                **config.get("ranking_config", {})
            )

            # Apply personalized ranking
            ranking_result = await self.ranking_service.rank_results(ranking_request)

            # Convert ranked results back to dict format
            ranked_results = []
            for ranked_result in ranking_result.ranked_results:
                result_dict = ranked_result.metadata.copy()
                result_dict.update({
                    "id": ranked_result.result_id,
                    "title": ranked_result.title,
                    "content": ranked_result.content,
                    "score": ranked_result.final_score,
                    "original_score": ranked_result.original_score,
                    "personalization_boost": ranked_result.personalization_boost,
                    "ranking_factors": ranked_result.ranking_factors
                })
                ranked_results.append(result_dict)

            processing_time = (time.time() - start_time) * 1000

            return StageResult(
                stage=ProcessingStage.RANKING,
                success=True,
                processing_time_ms=processing_time,
                results_count=len(ranked_results),
                metadata={
                    "ranked_results": ranked_results,
                    "ranking_strategy": ranking_result.strategy_used.value,
                    "personalization_applied": ranking_result.personalization_applied,
                    "reranking_impact": ranking_result.reranking_impact,
                    "user_profile_confidence": ranking_result.user_profile_confidence
                }
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000

            return StageResult(
                stage=ProcessingStage.RANKING,
                success=False,
                processing_time_ms=processing_time,
                results_count=len(results),
                error_details={"error": str(e)}
            )

    async def _execute_federation_stage(
        self, query: str, request: AdvancedSearchRequest, config: dict[str, Any]
    ) -> StageResult:
        """Execute federated search stage."""
        import time

        start_time = time.time()

        try:
            # Build federated search request
            federation_request = FederatedSearchRequest(
                query=query,
                limit=request.limit,
                offset=request.offset,
                **config.get("federation_config", {})
            )

            # Execute federated search
            federation_result = await self.federated_service.search(federation_request)

            processing_time = (time.time() - start_time) * 1000

            return StageResult(
                stage=ProcessingStage.FEDERATION,
                success=True,
                processing_time_ms=processing_time,
                results_count=len(federation_result.results),
                metadata={
                    "federated_results": federation_result.results,
                    "collections_searched": federation_result.collections_searched,
                    "collections_failed": federation_result.collections_failed,
                    "federation_strategy": federation_result.search_strategy.value,
                    "total_hits": federation_result.federated_metadata.get("total_hits", 0)
                }
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000

            return StageResult(
                stage=ProcessingStage.FEDERATION,
                success=False,
                processing_time_ms=processing_time,
                results_count=0,
                error_details={"error": str(e)}
            )

    async def _execute_postprocessing_stage(
        self, results: list[dict[str, Any]], request: AdvancedSearchRequest, config: dict[str, Any]
    ) -> StageResult:
        """Execute post-processing stage."""
        import time

        start_time = time.time()

        try:
            # Apply final filters and optimizations
            final_results = results.copy()

            # Apply quality threshold
            quality_threshold = config.get("quality_threshold", 0.6)
            final_results = [
                result for result in final_results
                if result.get("score", 0.0) >= quality_threshold
            ]

            # Apply diversity optimization if requested
            if config.get("diversity_factor", 0.0) > 0:
                final_results = self._apply_diversity_optimization(
                    final_results, config["diversity_factor"]
                )

            # Sort by final score
            final_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

            # Add final metadata
            for i, result in enumerate(final_results):
                result["final_rank"] = i + 1
                result["pipeline"] = request.pipeline.value
                result["processing_timestamp"] = datetime.now().isoformat()

            processing_time = (time.time() - start_time) * 1000

            return StageResult(
                stage=ProcessingStage.POSTPROCESSING,
                success=True,
                processing_time_ms=processing_time,
                results_count=len(final_results),
                metadata={
                    "final_results": final_results,
                    "quality_filtered": len(results) - len(final_results),
                    "diversity_applied": config.get("diversity_factor", 0.0) > 0,
                    "final_processing": True
                }
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000

            return StageResult(
                stage=ProcessingStage.POSTPROCESSING,
                success=False,
                processing_time_ms=processing_time,
                results_count=len(results),
                error_details={"error": str(e)}
            )

    def _apply_pipeline_config(self, request: AdvancedSearchRequest) -> dict[str, Any]:
        """Apply pipeline configuration to request."""
        # Start with base pipeline config
        config = self.pipeline_configs.get(request.pipeline.value, {}).copy()

        # Override with request-specific settings
        if hasattr(request, 'enable_expansion'):
            config["enable_expansion"] = request.enable_expansion
        if hasattr(request, 'enable_clustering'):
            config["enable_clustering"] = request.enable_clustering
        if hasattr(request, 'enable_personalization'):
            config["enable_personalization"] = request.enable_personalization
        if hasattr(request, 'enable_federation'):
            config["enable_federation"] = request.enable_federation

        # Add configuration objects
        if request.expansion_config:
            config["expansion_config"] = request.expansion_config
        if request.clustering_config:
            config["clustering_config"] = request.clustering_config
        if request.ranking_config:
            config["ranking_config"] = request.ranking_config
        if request.federation_config:
            config["federation_config"] = request.federation_config

        # Apply request-level overrides
        config.update({
            "max_processing_time_ms": request.max_processing_time_ms,
            "quality_threshold": request.quality_threshold,
            "diversity_factor": request.diversity_factor
        })

        return config

    def _enhance_query_with_context(self, query: str, context: dict[str, Any]) -> str:
        """Enhance query with contextual information."""
        enhanced_query = query

        # Add domain context if available
        if "domain" in context:
            domain = context["domain"]
            enhanced_query = f"{query} domain:{domain}"

        # Add intent context if available
        if "intent" in context:
            intent = context["intent"]
            enhanced_query = f"{enhanced_query} intent:{intent}"

        return enhanced_query

    def _merge_federated_results(
        self, primary_results: list[dict[str, Any]], federated_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Merge primary and federated search results."""
        # Simple merge strategy - interleave results
        merged = []
        primary_idx = 0
        federated_idx = 0

        while primary_idx < len(primary_results) and federated_idx < len(federated_results):
            # Add 2 primary results, then 1 federated result
            if len(merged) % 3 < 2:
                merged.append(primary_results[primary_idx])
                primary_idx += 1
            else:
                merged.append(federated_results[federated_idx])
                federated_idx += 1

        # Add remaining results
        merged.extend(primary_results[primary_idx:])
        merged.extend(federated_results[federated_idx:])

        return merged

    def _apply_diversity_optimization(
        self, results: list[dict[str, Any]], diversity_factor: float
    ) -> list[dict[str, Any]]:
        """Apply diversity optimization to results."""
        if diversity_factor <= 0 or len(results) <= 1:
            return results

        # Simple diversity implementation based on content types
        diversified = []
        content_type_counts = {}

        for result in results:
            content_type = result.get("content_type", "unknown")
            current_count = content_type_counts.get(content_type, 0)

            # Apply diversity penalty based on how many of this type we've seen
            diversity_penalty = current_count * diversity_factor * 0.1
            result["score"] = result.get("score", 0.0) - diversity_penalty

            diversified.append(result)
            content_type_counts[content_type] = current_count + 1

        # Re-sort by adjusted scores
        diversified.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        return diversified

    def _calculate_quality_metrics(
        self, results: list[dict[str, Any]], request: AdvancedSearchRequest
    ) -> dict[str, float]:
        """Calculate quality metrics for search results."""
        if not results:
            return {
                "quality_score": 0.0,
                "diversity_score": 0.0,
                "relevance_score": 0.0
            }

        # Quality score - average of result scores
        scores = [result.get("score", 0.0) for result in results]
        quality_score = sum(scores) / len(scores) if scores else 0.0

        # Diversity score - based on content type variety
        content_types = {result.get("content_type", "unknown") for result in results[:10]}
        diversity_score = min(1.0, len(content_types) / 5.0)  # Normalize to max 5 types

        # Relevance score - same as quality for now
        relevance_score = quality_score

        return {
            "quality_score": quality_score,
            "diversity_score": diversity_score,
            "relevance_score": relevance_score
        }

    def _get_features_used(
        self, config: dict[str, Any], stage_results: list[StageResult]
    ) -> list[str]:
        """Get list of features that were used."""
        features = []

        if config.get("enable_expansion"):
            features.append("query_expansion")
        if config.get("enable_clustering"):
            features.append("result_clustering")
        if config.get("enable_personalization"):
            features.append("personalized_ranking")
        if config.get("enable_federation"):
            features.append("federated_search")

        # Add features based on successful stages
        for stage_result in stage_results:
            if stage_result.success and stage_result.stage == ProcessingStage.FILTERING:
                applied_filters = stage_result.metadata.get("applied_filters", [])
                features.extend(f"{filter_type}_filtering" for filter_type in applied_filters)

        return list(set(features))  # Remove duplicates

    def _get_optimizations_applied(self, config: dict[str, Any]) -> list[str]:
        """Get list of optimizations that were applied."""
        optimizations = []

        if self.enable_performance_optimization:
            optimizations.append("performance_optimization")

        if config.get("enable_caching", True):
            optimizations.append("result_caching")

        if config.get("diversity_factor", 0.0) > 0:
            optimizations.append("diversity_optimization")

        return optimizations

    def _build_error_result(
        self,
        request: AdvancedSearchRequest,
        stage_results: list[StageResult],
        error_message: str,
        processing_time: float = 0.0
    ) -> AdvancedSearchResult:
        """Build error result when pipeline fails."""
        return AdvancedSearchResult(
            results=[],
            total_results=0,
            search_mode=request.search_mode,
            pipeline=request.pipeline,
            query_processed=request.query,
            stage_results=stage_results,
            total_processing_time_ms=processing_time,
            quality_score=0.0,
            diversity_score=0.0,
            relevance_score=0.0,
            features_used=[],
            optimizations_applied=[],
            cache_hit=False,
            performance_metrics={"error": error_message},
            search_metadata={"error": error_message, "failed": True}
        )

    def _get_cached_result(self, request: AdvancedSearchRequest) -> AdvancedSearchResult | None:
        """Get cached search result."""
        cache_key = self._generate_cache_key(request)

        if cache_key in self.search_cache:
            self.cache_stats["hits"] += 1
            return self.search_cache[cache_key]

        self.cache_stats["misses"] += 1
        return None

    def _cache_result(self, request: AdvancedSearchRequest, result: AdvancedSearchResult) -> None:
        """Cache search result."""
        if len(self.search_cache) >= self.cache_size:
            # Simple LRU eviction
            oldest_key = next(iter(self.search_cache))
            del self.search_cache[oldest_key]

        cache_key = self._generate_cache_key(request)
        self.search_cache[cache_key] = result

    def _generate_cache_key(self, request: AdvancedSearchRequest) -> str:
        """Generate cache key for search request."""
        key_components = [
            request.query,
            request.search_mode.value,
            request.pipeline.value,
            str(request.limit),
            str(request.offset),
            str(request.enable_expansion),
            str(request.enable_clustering),
            str(request.enable_personalization),
            str(request.enable_federation)
        ]
        return "|".join(key_components)

    def _update_performance_stats(
        self,
        request: AdvancedSearchRequest,
        result: AdvancedSearchResult,
        processing_time: float
    ) -> None:
        """Update performance statistics."""
        self.performance_stats["total_searches"] += 1

        # Update average processing time
        total = self.performance_stats["total_searches"]
        current_avg = self.performance_stats["avg_processing_time"]
        self.performance_stats["avg_processing_time"] = (
            (current_avg * (total - 1) + processing_time) / total
        )

        # Update feature usage
        for feature in result.features_used:
            if feature not in self.performance_stats["feature_usage"]:
                self.performance_stats["feature_usage"][feature] = 0
            self.performance_stats["feature_usage"][feature] += 1

        # Update pipeline usage
        pipeline_key = request.pipeline.value
        if pipeline_key not in self.performance_stats["pipeline_usage"]:
            self.performance_stats["pipeline_usage"][pipeline_key] = 0
        self.performance_stats["pipeline_usage"][pipeline_key] += 1

        # Update stage performance
        for stage_result in result.stage_results:
            stage_key = stage_result.stage.value
            if stage_key not in self.performance_stats["stage_performance"]:
                self.performance_stats["stage_performance"][stage_key] = {
                    "total_time": 0.0,
                    "count": 0,
                    "success_rate": 0.0
                }

            stage_stats = self.performance_stats["stage_performance"][stage_key]
            stage_stats["total_time"] += stage_result.processing_time_ms
            stage_stats["count"] += 1

            # Update success rate
            current_rate = stage_stats["success_rate"]
            stage_stats["success_rate"] = (
                (current_rate * (stage_stats["count"] - 1) + (1.0 if stage_result.success else 0.0)) /
                stage_stats["count"]
            )

        # Update quality stats
        total = self.performance_stats["total_searches"]
        self.quality_stats["avg_quality_score"] = (
            (self.quality_stats["avg_quality_score"] * (total - 1) + result.quality_score) / total
        )
        self.quality_stats["avg_diversity_score"] = (
            (self.quality_stats["avg_diversity_score"] * (total - 1) + result.diversity_score) / total
        )
        self.quality_stats["avg_relevance_score"] = (
            (self.quality_stats["avg_relevance_score"] * (total - 1) + result.relevance_score) / total
        )

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            **self.performance_stats,
            **self.quality_stats,
            "cache_stats": self.cache_stats,
            "cache_size": len(self.search_cache)
        }

    def clear_cache(self) -> None:
        """Clear search cache."""
        self.search_cache.clear()
        self.cache_stats = {"hits": 0, "misses": 0}
