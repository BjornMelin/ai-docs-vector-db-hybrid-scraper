"""Modernized QdrantSearch module demonstrating FastAPI-native error handling.

This module shows how the vector database search functionality has been
updated to use FastAPI's native exception handling while preserving all
critical functionality including circuit breakers and monitoring.
"""

import logging
from typing import Any

import numpy as np
from qdrant_client import AsyncQdrantClient, models

from src.api.exceptions import VectorDBException, handle_service_error
from src.config import Config, SearchAccuracy, VectorType
from src.services.adapters.error_adapter import legacy_error_handler

from ...models.vector_search import PrefetchConfig
from ..monitoring.metrics import get_metrics_registry
from .utils import build_filter


logger = logging.getLogger(__name__)


class QdrantSearchModernized:
    """Modernized search operations for Qdrant with FastAPI-native error handling.

    This demonstrates the updated error handling approach while maintaining
    full backward compatibility with existing circuit breakers and monitoring.
    """

    def __init__(self, client: AsyncQdrantClient, config: Config):
        """Initialize search service.

        Args:
            client: Initialized Qdrant client
            config: Unified configuration
        """
        self.client = client
        self.config = config
        self.prefetch_config = PrefetchConfig()

        # Initialize metrics if monitoring is enabled
        try:
            if config.monitoring.enabled:
                self.metrics_registry = get_metrics_registry()
            else:
                self.metrics_registry = None
        except Exception:
            logger.warning(
                "Metrics registry not available - search monitoring disabled"
            )
            self.metrics_registry = None

    @legacy_error_handler(operation="hybrid_search")
    async def hybrid_search(
        self,
        collection_name: str,
        query_vector: list[float],
        sparse_vector: dict[int, float] | None = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        fusion_type: str = "rrf",
        search_accuracy: str = "balanced",
    ) -> list[dict[str, Any]]:
        """Perform hybrid search combining dense and sparse vectors.

        Uses modern FastAPI error handling while maintaining all existing
        functionality including monitoring and circuit breaker integration.

        Args:
            collection_name: Collection to search
            query_vector: Dense embedding vector
            sparse_vector: Optional sparse vector (token_id -> weight)
            limit: Maximum results to return
            score_threshold: Minimum score threshold
            fusion_type: Fusion method ("rrf" or "dbsf")
            search_accuracy: Accuracy level ("fast", "balanced", "accurate", "exact")

        Returns:
            List of results with id, score, and payload

        Raises:
            VectorDBException: If search fails (FastAPI-compatible)
        """
        # Validate inputs with modern error handling
        if not collection_name or not isinstance(collection_name, str):
            raise VectorDBException(
                "Collection name must be a non-empty string",
                context={"collection_name": collection_name},
            )

        if not query_vector or not isinstance(query_vector, list):
            raise VectorDBException(
                "Query vector must be a non-empty list",
                context={"vector_length": len(query_vector) if query_vector else 0},
            )

        # Monitor search performance with existing infrastructure
        if self.metrics_registry:
            decorator = self.metrics_registry.monitor_search_performance(
                collection=collection_name, query_type="hybrid"
            )

            async def _monitored_search():
                return await self._execute_hybrid_search(
                    collection_name,
                    query_vector,
                    sparse_vector,
                    limit,
                    score_threshold,
                    fusion_type,
                    search_accuracy,
                )

            return await decorator(_monitored_search)()
        else:
            return await self._execute_hybrid_search(
                collection_name,
                query_vector,
                sparse_vector,
                limit,
                score_threshold,
                fusion_type,
                search_accuracy,
            )

    async def _execute_hybrid_search(
        self,
        collection_name: str,
        query_vector: list[float],
        sparse_vector: dict[int, float] | None = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        fusion_type: str = "rrf",
        search_accuracy: str = "balanced",
    ) -> list[dict[str, Any]]:
        """Execute the actual hybrid search operation with modern error handling."""
        try:
            # Build prefetch queries with optimized limits
            prefetch_queries = []

            # Dense vector prefetch with optimized limit
            dense_limit = self._calculate_prefetch_limit(VectorType.DENSE, limit)
            prefetch_queries.append(
                models.Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=dense_limit,
                    score_threshold=score_threshold,
                )
            )

            # Sparse vector prefetch if provided
            if sparse_vector:
                sparse_limit = self._calculate_prefetch_limit(VectorType.SPARSE, limit)
                prefetch_queries.append(
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=list(sparse_vector.keys()),
                            values=list(sparse_vector.values()),
                        ),
                        using="sparse",
                        limit=sparse_limit,
                        score_threshold=score_threshold,
                    )
                )

            # Configure search parameters based on accuracy
            search_params = self._get_search_params(search_accuracy)

            # Execute hybrid search using Query API
            query_request = models.QueryRequest(
                prefetch=prefetch_queries,
                query=query_vector,
                using="dense",
                limit=limit,
                score_threshold=score_threshold,
                params=search_params,
            )

            # Execute search with modern error context
            search_results = await self.client.query_points(
                collection_name=collection_name,
                query=query_request,
            )

            # Process and return results
            results = []
            for point in search_results.points:
                result = {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload or {},
                }
                results.append(result)

            logger.info(
                f"Hybrid search completed successfully: {len(results)} results "
                f"from collection '{collection_name}'"
            )
            return results

        except Exception as e:
            # Modern error handling with detailed context
            error_context = {
                "collection": collection_name,
                "vector_length": len(query_vector),
                "sparse_provided": sparse_vector is not None,
                "limit": limit,
                "fusion_type": fusion_type,
                "search_accuracy": search_accuracy,
            }

            # Handle specific error types with appropriate exceptions
            error_msg = str(e).lower()

            if "not found" in error_msg or "does not exist" in error_msg:
                raise VectorDBException(
                    f"Collection '{collection_name}' not found. Please create it first.",
                    context=error_context,
                ) from e
            elif "wrong vector size" in error_msg or "dimension" in error_msg:
                raise VectorDBException(
                    f"Vector dimension mismatch for collection '{collection_name}'. "
                    f"Check vector dimensions.",
                    context=error_context,
                ) from e
            elif "timeout" in error_msg:
                raise VectorDBException(
                    "Search request timed out. Try reducing the limit or simplifying the query.",
                    context=error_context,
                ) from e
            elif "connection" in error_msg:
                raise VectorDBException(
                    "Vector database connection failed. Service may be temporarily unavailable.",
                    context=error_context,
                ) from e
            else:
                # Generic service error with full context
                raise handle_service_error(
                    operation="hybrid_search",
                    error=e,
                    context=error_context,
                ) from e

    @legacy_error_handler(operation="multi_stage_search")
    async def multi_stage_search(
        self,
        collection_name: str,
        stages: list[dict[str, Any]],
        limit: int = 10,
        fusion_algorithm: str = "rrf",
        search_accuracy: str = "balanced",
    ) -> list[dict[str, Any]]:
        """Perform multi-stage retrieval with modern error handling.

        Args:
            collection_name: Collection to search
            stages: List of search stages with query_vector, vector_name, limit, etc.
            limit: Final number of results to return
            fusion_algorithm: Fusion method ("rrf" or "dbsf")
            search_accuracy: Accuracy level

        Returns:
            List of results with id, score, and payload

        Raises:
            VectorDBException: If search fails (FastAPI-compatible)
        """
        # Input validation with detailed error messages
        if not stages:
            raise VectorDBException(
                "Stages list cannot be empty",
                context={"collection": collection_name, "stages_count": 0},
            )

        if not isinstance(stages, list):
            raise VectorDBException(
                "Stages must be a list",
                context={
                    "collection": collection_name,
                    "stages_type": type(stages).__name__,
                },
            )

        # Validate each stage
        for i, stage in enumerate(stages):
            if not isinstance(stage, dict):
                raise VectorDBException(
                    f"Stage {i} must be a dictionary",
                    context={
                        "collection": collection_name,
                        "stage_index": i,
                        "stage_type": type(stage).__name__,
                    },
                )

            required_fields = ["query_vector", "vector_name", "limit"]
            for field in required_fields:
                if field not in stage:
                    raise VectorDBException(
                        f"Stage {i} must contain '{field}'",
                        context={
                            "collection": collection_name,
                            "stage_index": i,
                            "missing_field": field,
                        },
                    )

        # Monitor search performance
        if self.metrics_registry:
            decorator = self.metrics_registry.monitor_search_performance(
                collection=collection_name, query_type="multi_stage"
            )

            async def _monitored_search():
                return await self._execute_multi_stage_search(
                    collection_name, stages, limit, fusion_algorithm, search_accuracy
                )

            return await decorator(_monitored_search)()
        else:
            return await self._execute_multi_stage_search(
                collection_name, stages, limit, fusion_algorithm, search_accuracy
            )

    async def _execute_multi_stage_search(
        self,
        collection_name: str,
        stages: list[dict[str, Any]],
        limit: int = 10,
        fusion_algorithm: str = "rrf",
        search_accuracy: str = "balanced",
    ) -> list[dict[str, Any]]:
        """Execute multi-stage search with enhanced error context."""
        try:
            prefetch_queries = []

            # Build prefetch queries from all but the final stage
            for i, stage in enumerate(stages[:-1]):
                try:
                    vector_type = VectorType(stage.get("vector_type", "dense"))
                    stage_limit = self._calculate_prefetch_limit(
                        vector_type, stage["limit"]
                    )

                    prefetch_query = models.Prefetch(
                        query=stage["query_vector"],
                        using=stage["vector_name"],
                        limit=stage_limit,
                        filter=build_filter(stage.get("filter")),
                    )
                    prefetch_queries.append(prefetch_query)

                except Exception as stage_error:
                    raise VectorDBException(
                        f"Failed to build stage {i} query: {stage_error}",
                        context={
                            "collection": collection_name,
                            "stage_index": i,
                            "stage_data": stage,
                        },
                    ) from stage_error

            # Build final query from the last stage
            final_stage = stages[-1]
            search_params = self._get_search_params(search_accuracy)

            query_request = models.QueryRequest(
                prefetch=prefetch_queries,
                query=final_stage["query_vector"],
                using=final_stage["vector_name"],
                limit=limit,
                filter=build_filter(final_stage.get("filter")),
                params=search_params,
            )

            # Execute search
            search_results = await self.client.query_points(
                collection_name=collection_name,
                query=query_request,
            )

            # Process results
            results = []
            for point in search_results.points:
                result = {
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload or {},
                }
                results.append(result)

            logger.info(
                f"Multi-stage search completed: {len(results)} results "
                f"from {len(stages)} stages in collection '{collection_name}'"
            )
            return results

        except VectorDBException:
            # Re-raise our own exceptions
            raise
        except Exception as e:
            # Convert other exceptions with full context
            context = {
                "collection": collection_name,
                "stages_count": len(stages),
                "fusion_algorithm": fusion_algorithm,
                "search_accuracy": search_accuracy,
                "final_limit": limit,
            }

            raise handle_service_error(
                operation="multi_stage_search",
                error=e,
                context=context,
            ) from e

    def _calculate_prefetch_limit(
        self, vector_type: VectorType, target_limit: int
    ) -> int:
        """Calculate optimal prefetch limit based on vector type and research.

        This preserves the existing logic while working with the new error system.
        """
        multipliers = {
            VectorType.DENSE: self.prefetch_config.dense_multiplier,
            VectorType.SPARSE: self.prefetch_config.sparse_multiplier,
            VectorType.MATRYOSHKA: self.prefetch_config.matryoshka_multiplier,
        }

        multiplier = multipliers.get(vector_type, 2.0)
        calculated_limit = int(target_limit * multiplier)

        return min(calculated_limit, self.prefetch_config.max_prefetch_limit)

    def _get_search_params(self, search_accuracy: str) -> models.SearchParams:
        """Get search parameters based on accuracy level.

        This preserves existing functionality while working with new error handling.
        """
        try:
            accuracy_enum = SearchAccuracy(search_accuracy)
        except ValueError:
            # Use new error handling for invalid enum values
            raise VectorDBException(
                f"Invalid search accuracy: {search_accuracy}. "
                f"Valid options: {[a.value for a in SearchAccuracy]}",
                context={"provided_accuracy": search_accuracy},
            )

        accuracy_params = {
            SearchAccuracy.FAST: {"hnsw_ef": 32, "exact": False},
            SearchAccuracy.BALANCED: {"hnsw_ef": 64, "exact": False},
            SearchAccuracy.ACCURATE: {"hnsw_ef": 128, "exact": False},
            SearchAccuracy.EXACT: {"exact": True},
        }

        params = accuracy_params[accuracy_enum]
        return models.SearchParams(**params)


# Compatibility function for gradual migration
def create_modernized_search_service(
    client: AsyncQdrantClient, config: Config
) -> QdrantSearchModernized:
    """Create a modernized search service instance.

    This function provides a migration path from the legacy QdrantSearch
    to the modernized version with FastAPI-native error handling.
    """
    return QdrantSearchModernized(client, config)
