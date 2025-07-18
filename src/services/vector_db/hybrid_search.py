"""Advanced hybrid search service - backward compatibility wrapper.

This module provides a backward compatibility wrapper around the new
AdvancedSearchOrchestrator.
It maps the old AdvancedHybridSearchService API to the new unified orchestrator.
"""

import asyncio
import logging
import time
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient

from src.config import (
    ABTestVariant,
    Config,
    OptimizationStrategy,
    SearchMode,
    SearchPipeline,
)
from src.models.vector_search import (
    HybridSearchRequest,
    HybridSearchResponse,
    RetrievalMetrics,
    SearchResult,
)
from src.services.errors import ServiceError
from src.services.query_processing import (
    AdvancedSearchOrchestrator,
    AdvancedSearchRequest,
)

from .adaptive_fusion_tuner import AdaptiveFusionTuner
from .model_selector import ModelSelector
from .query_classifier import QueryClassifier
from .search import QdrantSearch
from .splade_provider import SPLADEProvider


logger = logging.getLogger(__name__)


class HybridSearchService:
    """Backward compatibility wrapper for hybrid search.

    This class provides the same interface as the old HybridSearchService
    but delegates to the new AdvancedSearchOrchestrator for actual search operations.
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        config: Config,
        qdrant_search: QdrantSearch,
    ):
        """Initialize advanced hybrid search service.

        Args:
            client: Qdrant client instance
            config: Unified configuration
            qdrant_search: Base Qdrant search service

        """
        self.client = client
        self.config = config
        self.qdrant_search = qdrant_search

        # Initialize the new orchestrator
        self.orchestrator = AdvancedSearchOrchestrator(
            cache_size=1000, enable_performance_optimization=True
        )

        # Initialize ML components for backward compatibility
        self.query_classifier = QueryClassifier(config)
        self.adaptive_fusion_tuner = AdaptiveFusionTuner(config)
        self.model_selector = ModelSelector(config)
        self.splade_provider = SPLADEProvider(config)

        # Performance tracking
        self.search_metrics: dict[str, RetrievalMetrics] = {}
        self.ab_test_assignments: dict[str, ABTestVariant] = {}

        # Fallback configurations
        self.enable_fallback = True
        self.fallback_timeout_ms = 5000

    async def initialize(self) -> None:
        """Initialize all ML components."""
        try:
            await self.splade_provider.initialize()
            logger.info("Advanced hybrid search service initialized successfully")
        except Exception:
            logger.exception("Failed to initialize advanced search service: %s")
            if not self.enable_fallback:
                raise

    async def hybrid_search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        """Perform hybrid search with ML optimization.

        Args:
            request: Hybrid search request

        Returns:
            HybridSearchResponse with results and optimization metadata

        """
        start_time = time.time()
        query_id = str(uuid.uuid4())

        try:
            advanced_request = self._build_advanced_search_request(request)
        except Exception as e:
            logger.exception("Failed to build advanced search request: %s")
            if self.enable_fallback:
                return await self._perform_fallback_search(request, start_time, str(e))
            msg = f"Request preparation failed: {e}"
            raise ServiceError(msg) from e

        try:
            await self._prepare_search_context(request, advanced_request)
        except Exception as e:
            logger.exception("Failed to prepare search context: %s")
            if self.enable_fallback:
                return await self._perform_fallback_search(request, start_time, str(e))
            msg = f"Context preparation failed: {e}"
            raise ServiceError(msg) from e

        try:
            orchestrator_result = await self.orchestrator.search(advanced_request)
        except Exception as e:
            logger.exception("Orchestrator search failed: %s")
            if self.enable_fallback:
                return await self._perform_fallback_search(request, start_time, str(e))
            msg = f"Search execution failed: {e}"
            raise ServiceError(msg) from e

        try:
            response = self._build_hybrid_search_response(
                request, advanced_request, orchestrator_result
            )
        except Exception as e:
            logger.exception("Failed to build search response: %s")
            if self.enable_fallback:
                return await self._perform_fallback_search(request, start_time, str(e))
            msg = f"Response construction failed: {e}"
            raise ServiceError(msg) from e

        try:
            await self._store_search_for_learning(query_id, request, response)
        except (OSError, AttributeError, ConnectionError) as e:
            logger.warning("Failed to store search for learning: %s", e)
            # Don't fail the request for learning storage issues

        return response

    def _build_advanced_search_request(
        self, request: HybridSearchRequest
    ) -> AdvancedSearchRequest:
        """Build advanced search request from hybrid search request."""
        return AdvancedSearchRequest(
            query=request.query,
            collection_name=request.collection_name,
            limit=request.limit,
            score_threshold=request.score_threshold,
            user_id=request.user_id,
            session_id=request.session_id,
            # Select appropriate search mode based on enabled features
            search_mode=self._determine_search_mode(request),
            # Select appropriate pipeline based on optimization strategy
            pipeline=self._determine_pipeline(request),
            # Enable features based on request
            enable_expansion=request.enable_adaptive_fusion,
            enable_clustering=False,  # Not enabled in old API
            enable_personalization=request.enable_adaptive_fusion,
            enable_federation=False,  # Not enabled in old API
            enable_caching=True,
            # Pass through existing configurations
            max_processing_time_ms=self.fallback_timeout_ms,
            quality_threshold=0.7,
        )

    async def _prepare_search_context(
        self, request: HybridSearchRequest, advanced_request: AdvancedSearchRequest
    ) -> None:
        """Prepare search context with classification, model selection, and SPLADE."""
        # Handle query classification if enabled
        if request.enable_query_classification:
            query_classification = await self._classify_query_with_timeout(
                request.query,
                {"user_id": request.user_id, "session_id": request.session_id},
            )
            if query_classification:
                # Store classification in context for later use
                advanced_request.context["query_classification"] = query_classification

        # Handle model selection if enabled
        if (
            request.enable_model_selection
            and "query_classification" in advanced_request.context
        ):
            model_selection = await self._select_model_with_timeout(
                advanced_request.context["query_classification"], request
            )
            if model_selection:
                # Pass model selection to orchestrator via context
                advanced_request.context["model_selection"] = model_selection

        # Handle SPLADE if enabled
        if request.enable_splade:
            sparse_vector = await self._generate_sparse_vector_with_timeout(
                request.query, request.splade_config
            )
            if sparse_vector:
                # Pass sparse vector configuration via context
                advanced_request.context["splade_config"] = {
                    "enabled": True,
                    "vector": sparse_vector,
                    "config": request.splade_config,
                }

    def _build_hybrid_search_response(
        self,
        request: HybridSearchRequest,
        advanced_request: AdvancedSearchRequest,
        orchestrator_result,
    ) -> HybridSearchResponse:
        """Build hybrid search response from orchestrator result."""
        # Map orchestrator result to old response format
        response = HybridSearchResponse(
            results=self._format_search_results(orchestrator_result.results),
            retrieval_metrics=RetrievalMetrics(
                query_vector_time_ms=orchestrator_result.search_metadata.get(
                    "vector_generation_time_ms", 50.0
                ),
                search_time_ms=orchestrator_result.total_processing_time_ms - 50.0,
                total_time_ms=orchestrator_result.total_processing_time_ms,
                results_count=len(orchestrator_result.results),
                filtered_count=len(orchestrator_result.results),
                cache_hit=orchestrator_result.search_metadata.get("cache_hit", False),
                hnsw_ef_used=request.search_params.hnsw_ef,
            ),
            query_classification=advanced_request.context.get(
                "query_classification", None
            ),
            model_selection=advanced_request.context.get("model_selection", None),
            optimization_applied=orchestrator_result.optimizations_applied != [],
            # A/B test handling
            ab_test_variant=self._assign_ab_test_variant(request)
            if request.ab_test_config
            else None,
        )

        # Handle adaptive fusion if enabled
        if (
            request.enable_adaptive_fusion
            and "query_classification" in advanced_request.context
        ):
            # Get fusion weights from orchestrator metadata
            fusion_metadata = orchestrator_result.search_metadata.get("fusion_weights")
            if fusion_metadata:
                response.fusion_weights = fusion_metadata
                response.effectiveness_score = orchestrator_result.search_metadata.get(
                    "effectiveness_score"
                )

        return response

    def _determine_search_mode(self, request: HybridSearchRequest) -> SearchMode:
        """Determine the appropriate search mode based on request features."""
        if request.enable_adaptive_fusion and request.enable_query_classification:
            return SearchMode.INTELLIGENT
        if request.enable_model_selection:
            return SearchMode.PERSONALIZED
        if request.enable_splade:
            return SearchMode.ENHANCED
        return SearchMode.SIMPLE

    def _determine_pipeline(self, request: HybridSearchRequest) -> SearchPipeline:
        """Determine the appropriate pipeline based on request configuration."""
        if request.enable_adaptive_fusion:
            return SearchPipeline.COMPREHENSIVE
        if request.fusion_config.algorithm.value == "dbsf":
            return SearchPipeline.PRECISION
        if request.search_params.accuracy_level.value == "fast":
            return SearchPipeline.FAST
        return SearchPipeline.BALANCED

    async def _classify_query_with_timeout(
        self, query: str, context: dict[str, Any] | None
    ) -> Any:
        """Classify query with timeout protection."""
        try:
            return await asyncio.wait_for(
                self.query_classifier.classify_query(query, context),
                timeout=2.0,  # 2 second timeout
            )
        except TimeoutError:
            logger.warning(
                "Query classification timed out, using default classification"
            )
            return None
        except (asyncio.CancelledError, RuntimeError) as e:
            logger.warning("Query classification failed: %s", e)
            return None

    async def _select_model_with_timeout(
        self, query_classification: Any, _request: HybridSearchRequest
    ) -> Any:
        """Select model with timeout protection."""
        try:
            return await asyncio.wait_for(
                self.model_selector.select_optimal_model(
                    query_classification, OptimizationStrategy.BALANCED
                ),
                timeout=1.0,  # 1 second timeout
            )
        except TimeoutError:
            logger.warning("Model selection timed out, using default model")
            return None
        except (ValueError, ConnectionError, RuntimeError) as e:
            logger.warning("Model selection failed: %s", e)
            return None

    async def _generate_sparse_vector_with_timeout(
        self, query: str, _splade_config: Any
    ) -> dict[int, float] | None:
        """Generate sparse vector with timeout protection."""
        try:
            return await asyncio.wait_for(
                self.splade_provider.generate_sparse_vector(query),
                timeout=3.0,  # 3 second timeout
            )
        except TimeoutError:
            logger.warning("SPLADE generation timed out, skipping sparse vector")
            return None
        except (ValueError, ConnectionError, RuntimeError) as e:
            logger.warning("SPLADE generation failed: %s", e)
            return None

    def _assign_ab_test_variant(self, request: HybridSearchRequest) -> ABTestVariant:
        """Assign A/B test variant for the request."""
        if not request.ab_test_config or not request.user_id:
            return ABTestVariant.CONTROL

        # Simple hash-based assignment for consistent user experience
        user_hash = hash(request.user_id) % 100

        # Get traffic allocation
        allocations = request.ab_test_config.traffic_allocation
        cumulative = 0

        for variant_name, allocation in allocations.items():
            cumulative += allocation * 100
            if user_hash < cumulative:
                try:
                    return ABTestVariant(variant_name)
                except ValueError:
                    return ABTestVariant.CONTROL

        return ABTestVariant.CONTROL

    async def _perform_fallback_search(
        self, request: HybridSearchRequest, start_time: float, error_msg: str
    ) -> HybridSearchResponse:
        """Perform fallback search when advanced features fail."""
        try:
            # Use simple search through orchestrator
            fallback_request = AdvancedSearchRequest(
                query=request.query,
                collection_name=request.collection_name,
                limit=request.limit,
                score_threshold=request.score_threshold,
                search_mode=SearchMode.SIMPLE,
                pipeline=SearchPipeline.FAST,
                enable_caching=True,
            )

            result = await self.orchestrator.search(fallback_request)

            end_time = time.time()
            return HybridSearchResponse(
                results=self._format_search_results(result.results),
                retrieval_metrics=RetrievalMetrics(
                    total_time_ms=(end_time - start_time) * 1000,
                    results_count=len(result.results),
                    filtered_count=len(result.results),
                ),
                fallback_reason=f"Advanced search failed: {error_msg}",
            )

        except Exception:
            logger.exception("Fallback search also failed: %s")
            return HybridSearchResponse(
                results=[],
                retrieval_metrics=RetrievalMetrics(
                    total_time_ms=(time.time() - start_time) * 1000,
                ),
                fallback_reason=f"All search attempts failed: {error_msg}",
            )

    def _format_search_results(
        self, results: list[dict[str, Any]]
    ) -> list[SearchResult]:
        """Format search results to standard format."""
        formatted_results = []

        for result in results:
            search_result = SearchResult(
                id=str(result.get("id", "")),
                score=float(result.get("score", 0.0)),
                payload=result.get("payload", result),
                vector=result.get("vector"),
            )
            formatted_results.append(search_result)

        return formatted_results

    async def _store_search_for_learning(
        self,
        query_id: str,
        request: HybridSearchRequest,
        response: HybridSearchResponse,
    ) -> None:
        """Store search results for continuous learning."""
        try:
            # Store metrics for analysis
            self.search_metrics[query_id] = response.retrieval_metrics

            # Update model performance if model selection was used
            if response.model_selection and response.query_classification:
                # Calculate performance score based on results quality
                performance_score = min(len(response.results) / request.limit, 1.0)

                await self.model_selector.update_performance_history(
                    response.model_selection.primary_model,
                    response.query_classification,
                    performance_score,
                )

            logger.debug("Stored search data for learning: %s", query_id)

        except Exception:
            logger.exception("Failed to store search for learning: %s")

    async def update_with_user_feedback(
        self, query_id: str, _user_feedback: dict[str, Any]
    ) -> None:
        """Update ML components with user feedback."""
        try:
            # This would update the adaptive fusion tuner and other components
            # with user feedback for continuous improvement
            logger.debug("Processing user feedback for query %s", query_id)

        except Exception:
            logger.exception("Failed to process user feedback: %s")

    def get_performance_statistics(self) -> dict[str, Any]:
        """Get performance statistics for monitoring."""
        return {
            "total_searches": len(self.search_metrics),
            "average_search_time": (
                sum(m.total_time_ms for m in self.search_metrics.values())
                / max(len(self.search_metrics), 1)
            ),
            "fusion_tuner_stats": self.adaptive_fusion_tuner.get_performance_stats(),
            "splade_cache_stats": self.splade_provider.get_cache_stats(),
            "ab_test_assignments": dict(self.ab_test_assignments),
        }
