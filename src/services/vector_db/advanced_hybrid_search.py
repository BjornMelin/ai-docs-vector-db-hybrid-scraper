"""Advanced hybrid search service with ML-based optimization.

This module implements the main orchestrator for advanced hybrid search that combines
query classification, model selection, adaptive fusion, and SPLADE sparse vectors.
"""

import asyncio
import logging
import time
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient

from ...config import UnifiedConfig
from ...config.enums import ABTestVariant
from ...config.enums import OptimizationStrategy
from ...models.vector_search import AdvancedHybridSearchRequest
from ...models.vector_search import AdvancedSearchResponse
from ...models.vector_search import RetrievalMetrics
from ...models.vector_search import SearchResult
from ..errors import QdrantServiceError
from .adaptive_fusion_tuner import AdaptiveFusionTuner
from .model_selector import ModelSelector
from .query_classifier import QueryClassifier
from .search import QdrantSearch
from .splade_provider import SPLADEProvider

logger = logging.getLogger(__name__)


class AdvancedHybridSearchService:
    """Advanced hybrid search service with ML-based optimization."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        config: UnifiedConfig,
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

        # Initialize ML components
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
        except Exception as e:
            logger.error(
                f"Failed to initialize advanced search service: {e}", exc_info=True
            )
            if not self.enable_fallback:
                raise

    async def advanced_hybrid_search(
        self, request: AdvancedHybridSearchRequest
    ) -> AdvancedSearchResponse:
        """Perform advanced hybrid search with ML optimization.

        Args:
            request: Advanced hybrid search request

        Returns:
            AdvancedSearchResponse with results and optimization metadata
        """
        start_time = time.time()
        query_id = str(uuid.uuid4())

        # Initialize response with defaults
        response = AdvancedSearchResponse(
            results=[], retrieval_metrics=RetrievalMetrics()
        )

        try:
            # Step 1: Query Classification
            query_classification = None
            if request.enable_query_classification:
                query_classification = await self._classify_query_with_timeout(
                    request.query,
                    {"user_id": request.user_id, "session_id": request.session_id},
                )
                response.query_classification = query_classification

            # Step 2: A/B Test Assignment
            ab_test_variant = None
            if request.ab_test_config:
                ab_test_variant = self._assign_ab_test_variant(request)
                response.ab_test_variant = ab_test_variant

            # Step 3: Model Selection
            model_selection = None
            if request.enable_model_selection and query_classification:
                model_selection = await self._select_model_with_timeout(
                    query_classification, request
                )
                response.model_selection = model_selection

            # Step 4: Generate Dense Vector
            dense_vector = await self._generate_dense_vector(
                request.query, model_selection
            )

            # Step 5: Generate Sparse Vector (SPLADE)
            sparse_vector = None
            if request.enable_splade:
                sparse_vector = await self._generate_sparse_vector_with_timeout(
                    request.query, request.splade_config
                )

            # Step 6: Execute Search Strategies
            search_results = await self._execute_search_strategies(
                request, dense_vector, sparse_vector, ab_test_variant
            )

            # Step 7: Adaptive Fusion (if enabled)
            if request.enable_adaptive_fusion and query_classification:
                optimized_results = await self._apply_adaptive_fusion(
                    request, query_classification, search_results, query_id
                )
                response.fusion_weights = optimized_results.get("fusion_weights")
                response.effectiveness_score = optimized_results.get(
                    "effectiveness_score"
                )
                response.optimization_applied = True
                final_results = optimized_results["results"]
            else:
                final_results = search_results

            # Step 8: Format Results
            response.results = self._format_search_results(final_results)

            # Step 9: Calculate Metrics
            end_time = time.time()
            response.retrieval_metrics = self._calculate_retrieval_metrics(
                start_time, end_time, len(response.results), request
            )

            # Step 10: Store for Learning
            await self._store_search_for_learning(query_id, request, response)

            return response

        except Exception as e:
            logger.error(f"Advanced hybrid search failed: {e}", exc_info=True)

            # Fallback to basic hybrid search
            if self.enable_fallback:
                response.fallback_reason = f"Advanced search failed: {e!s}"
                fallback_results = await self._fallback_search(request)
                response.results = self._format_search_results(fallback_results)

                end_time = time.time()
                response.retrieval_metrics = self._calculate_retrieval_metrics(
                    start_time, end_time, len(response.results), request
                )

                return response
            else:
                raise QdrantServiceError(f"Advanced hybrid search failed: {e}") from e

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
        except Exception as e:
            logger.warning(f"Query classification failed: {e}")
            return None

    async def _select_model_with_timeout(
        self, query_classification: Any, request: AdvancedHybridSearchRequest
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
        except Exception as e:
            logger.warning(f"Model selection failed: {e}")
            return None

    async def _generate_sparse_vector_with_timeout(
        self, query: str, splade_config: Any
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
        except Exception as e:
            logger.warning(f"SPLADE generation failed: {e}")
            return None

    async def _generate_dense_vector(
        self, query: str, model_selection: Any
    ) -> list[float]:
        """Generate dense vector using selected model."""
        # This would integrate with the embedding service
        # For now, return a placeholder - in real implementation,
        # this would call the appropriate embedding model

        # Placeholder implementation
        # In real implementation, this would:
        # 1. Use the selected model from model_selection
        # 2. Call the embedding service with the optimal model
        # 3. Return the actual embedding vector

        logger.debug(f"Generating dense vector for query: {query[:50]}...")

        # Return placeholder vector (in real implementation, use actual embeddings)
        import numpy as np

        return np.random.random(1536).tolist()  # OpenAI embedding dimension

    async def _execute_search_strategies(
        self,
        request: AdvancedHybridSearchRequest,
        dense_vector: list[float],
        sparse_vector: dict[int, float] | None,
        ab_test_variant: ABTestVariant | None,
    ) -> dict[str, Any]:
        """Execute different search strategies based on configuration."""
        search_results = {}

        # Dense vector search
        if dense_vector:
            dense_results = await self.qdrant_search.filtered_search(
                collection_name=request.collection_name,
                query_vector=dense_vector,
                filters={},
                limit=request.limit * 2,  # Get more for fusion
                search_accuracy=request.search_params.accuracy_level.value,
            )
            search_results["dense"] = dense_results

        # Sparse vector search (if available)
        if sparse_vector:
            # Convert sparse vector to format expected by hybrid search
            sparse_results = await self.qdrant_search.hybrid_search(
                collection_name=request.collection_name,
                query_vector=dense_vector,
                sparse_vector=sparse_vector,
                limit=request.limit * 2,
                score_threshold=request.score_threshold,
                fusion_type=request.fusion_config.algorithm.value,
                search_accuracy=request.search_params.accuracy_level.value,
            )
            search_results["sparse"] = sparse_results

        # Hybrid search (baseline)
        hybrid_results = await self.qdrant_search.hybrid_search(
            collection_name=request.collection_name,
            query_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=request.limit,
            score_threshold=request.score_threshold,
            fusion_type=request.fusion_config.algorithm.value,
            search_accuracy=request.search_params.accuracy_level.value,
        )
        search_results["hybrid"] = hybrid_results

        return search_results

    async def _apply_adaptive_fusion(
        self,
        request: AdvancedHybridSearchRequest,
        query_classification: Any,
        search_results: dict[str, Any],
        query_id: str,
    ) -> dict[str, Any]:
        """Apply adaptive fusion to optimize results."""
        try:
            # Get adaptive weights
            fusion_weights = await self.adaptive_fusion_tuner.compute_adaptive_weights(
                query_classification=query_classification,
                query_id=query_id,
                dense_results=search_results.get("dense"),
                sparse_results=search_results.get("sparse"),
            )

            # Apply fusion with optimized weights
            fused_results = self._apply_weighted_fusion(
                search_results, fusion_weights, request.limit
            )

            return {
                "results": fused_results,
                "fusion_weights": fusion_weights,
                "effectiveness_score": fusion_weights.effectiveness_score,
            }

        except Exception as e:
            logger.error(f"Adaptive fusion failed: {e}", exc_info=True)
            # Fallback to baseline hybrid results
            return {
                "results": search_results.get("hybrid", []),
                "fusion_weights": None,
                "effectiveness_score": None,
            }

    def _apply_weighted_fusion(
        self, search_results: dict[str, Any], fusion_weights: Any, limit: int
    ) -> list[dict[str, Any]]:
        """Apply weighted fusion to search results."""
        # Simple weighted combination of results
        # In production, this would be more sophisticated

        dense_results = search_results.get("dense", [])
        sparse_results = search_results.get("sparse", [])

        if not dense_results and not sparse_results:
            return search_results.get("hybrid", [])

        # Create a combined ranking using fusion weights
        combined_scores = {}

        # Add dense results with weight
        dense_weight = fusion_weights.dense_weight if fusion_weights else 0.7
        for i, result in enumerate(dense_results[: limit * 2]):
            doc_id = result["id"]
            # Weight by position and fusion weight
            position_score = 1.0 / (i + 1)
            combined_scores[doc_id] = (
                combined_scores.get(doc_id, 0) + dense_weight * position_score
            )

        # Add sparse results with weight
        sparse_weight = fusion_weights.sparse_weight if fusion_weights else 0.3
        for i, result in enumerate(sparse_results[: limit * 2]):
            doc_id = result["id"]
            position_score = 1.0 / (i + 1)
            combined_scores[doc_id] = (
                combined_scores.get(doc_id, 0) + sparse_weight * position_score
            )

        # Sort by combined score and return top results
        sorted_results = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Reconstruct result objects (simplified)
        final_results = []
        all_results = {r["id"]: r for r in dense_results + sparse_results}

        for doc_id, score in sorted_results[:limit]:
            if doc_id in all_results:
                result = all_results[doc_id].copy()
                result["score"] = float(score)  # Update with fusion score
                final_results.append(result)

        return final_results

    def _assign_ab_test_variant(
        self, request: AdvancedHybridSearchRequest
    ) -> ABTestVariant:
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

    async def _fallback_search(
        self, request: AdvancedHybridSearchRequest
    ) -> list[dict[str, Any]]:
        """Perform fallback search when advanced features fail."""
        try:
            # Generate basic dense vector (placeholder)
            dense_vector = await self._generate_dense_vector(request.query, None)

            # Perform basic hybrid search
            return await self.qdrant_search.hybrid_search(
                collection_name=request.collection_name,
                query_vector=dense_vector,
                sparse_vector=None,
                limit=request.limit,
                score_threshold=request.score_threshold,
                fusion_type="rrf",  # Default fusion
                search_accuracy="balanced",
            )

        except Exception as e:
            logger.error(f"Fallback search also failed: {e}", exc_info=True)
            return []

    def _format_search_results(
        self, results: list[dict[str, Any]]
    ) -> list[SearchResult]:
        """Format search results to standard format."""
        formatted_results = []

        for result in results:
            search_result = SearchResult(
                id=str(result.get("id", "")),
                score=float(result.get("score", 0.0)),
                payload=result.get("payload", {}),
                vector=result.get("vector"),
            )
            formatted_results.append(search_result)

        return formatted_results

    def _calculate_retrieval_metrics(
        self,
        start_time: float,
        end_time: float,
        results_count: int,
        request: AdvancedHybridSearchRequest,
    ) -> RetrievalMetrics:
        """Calculate retrieval performance metrics."""
        total_time_ms = (end_time - start_time) * 1000

        return RetrievalMetrics(
            query_vector_time_ms=50.0,  # Placeholder
            search_time_ms=total_time_ms - 50.0,
            total_time_ms=total_time_ms,
            results_count=results_count,
            filtered_count=results_count,
            cache_hit=False,  # Would be determined by actual cache usage
            hnsw_ef_used=request.search_params.hnsw_ef,
        )

    async def _store_search_for_learning(
        self,
        query_id: str,
        request: AdvancedHybridSearchRequest,
        response: AdvancedSearchResponse,
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

            logger.debug(f"Stored search data for learning: {query_id}")

        except Exception as e:
            logger.error(f"Failed to store search for learning: {e}", exc_info=True)

    async def update_with_user_feedback(
        self, query_id: str, user_feedback: dict[str, Any]
    ) -> None:
        """Update ML components with user feedback."""
        try:
            # This would update the adaptive fusion tuner and other components
            # with user feedback for continuous improvement

            # Extract any stored weights for this query
            # In production, this would retrieve from persistent storage

            logger.debug(f"Processing user feedback for query {query_id}")

        except Exception as e:
            logger.error(f"Failed to process user feedback: {e}", exc_info=True)

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
