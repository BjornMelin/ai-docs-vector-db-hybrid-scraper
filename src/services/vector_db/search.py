"""Focused QdrantSearch module for vector search operations.

This module provides a clean, focused implementation of search functionality
extracted from QdrantService, focusing specifically on search operations.
"""

import logging
from typing import Any

import numpy as np
from qdrant_client import AsyncQdrantClient, models

from src.config import Config, SearchAccuracy, VectorType
from src.models.vector_search import PrefetchConfig
from src.services.errors import QdrantServiceError
from src.services.monitoring.metrics import get_metrics_registry

from .utils import build_filter


logger = logging.getLogger(__name__)


class QdrantSearch:
    """Focused search operations for Qdrant with advanced Query API support."""

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
        except (AttributeError, ImportError, ValueError):
            logger.warning(
                "Metrics registry not available - search monitoring disabled"
            )
            self.metrics_registry = None

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

        Uses Qdrant's Query API with prefetch and fusion for optimal results.
        Enhanced with research-backed prefetch limits and search parameters.

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
            QdrantServiceError: If search fails

        """
        # Monitor search performance
        if self.metrics_registry:
            # Use the monitoring decorator manually
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
        """Execute the actual hybrid search operation."""
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
                    params=self._get_search_params(SearchAccuracy(search_accuracy)),
                )
            )

            # Sparse vector prefetch if available with optimized limit
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
                        params=self._get_search_params(SearchAccuracy(search_accuracy)),
                    )
                )

            # Execute hybrid query with fusion
            if len(prefetch_queries) > 1:
                # Use fusion for multiple queries
                fusion_method = (
                    models.Fusion.RRF
                    if fusion_type.lower() == "rrf"
                    else models.Fusion.DBSF
                )
                results = await self.client.query_points(
                    collection_name=collection_name,
                    prefetch=prefetch_queries,
                    query=models.FusionQuery(fusion=fusion_method),
                    limit=limit,
                    score_threshold=score_threshold,
                    params=self._get_search_params(SearchAccuracy(search_accuracy)),
                    with_payload=True,
                    with_vectors=False,
                )
            else:
                # Single query without fusion
                results = await self.client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    using="dense",
                    limit=limit,
                    score_threshold=score_threshold,
                    params=self._get_search_params(SearchAccuracy(search_accuracy)),
                    with_payload=True,
                    with_vectors=False,
                )

            # Format results
            return [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": point.payload or {},
                }
                for point in results.points
            ]

        except Exception as e:
            logger.exception(
                "Hybrid search failed for collection %s",
                collection_name,
            )

            error_msg = str(e).lower()
            if "collection not found" in error_msg:
                msg = (
                    f"Collection '{collection_name}' not found. Please create it first."
                )
                raise QdrantServiceError(msg) from e
            if "wrong vector size" in error_msg:
                msg = (
                    f"Vector dimension mismatch. Expected size for "
                    f"collection '{collection_name}'."
                )
                raise QdrantServiceError(msg) from e
            if "timeout" in error_msg:
                msg = (
                    "Search request timed out. Try reducing the limit or "
                    "simplifying the query."
                )
                raise QdrantServiceError(msg) from e
            msg = f"Hybrid search failed: {e}"
            raise QdrantServiceError(msg) from e

    async def multi_stage_search(
        self,
        collection_name: str,
        stages: list[dict[str, Any]],
        limit: int = 10,
        fusion_algorithm: str = "rrf",
        search_accuracy: str = "balanced",
    ) -> list[dict[str, Any]]:
        """Perform multi-stage retrieval with different strategies.

        Implements advanced Query API patterns for Matryoshka embeddings
        and complex retrieval strategies.

        Args:
            collection_name: Collection to search
            stages: List of search stages with query_vector, vector_name, limit, etc.
            limit: Final number of results to return
            fusion_algorithm: Fusion method ("rrf" or "dbsf")
            search_accuracy: Accuracy level ("fast", "balanced", "accurate", "exact")

        Returns:
            List of results with id, score, and payload

        Raises:
            QdrantServiceError: If search fails

        """
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
        """Execute the actual multi-stage search operation."""
        # Validate input parameters
        if not stages:
            msg = "Stages list cannot be empty"
            raise ValueError(msg)

        if not isinstance(stages, list):
            msg = "Stages must be a list"
            raise TypeError(msg)

        for i, stage in enumerate(stages):
            if not isinstance(stage, dict):
                msg = f"Stage {i} must be a dictionary"
                raise TypeError(msg)
            if "query_vector" not in stage:
                msg = f"Stage {i} must contain 'query_vector'"
                raise ValueError(msg)
            if "vector_name" not in stage:
                msg = f"Stage {i} must contain 'vector_name'"
                raise ValueError(msg)
            if "limit" not in stage:
                msg = f"Stage {i} must contain 'limit'"
                raise ValueError(msg)

        try:
            prefetch_queries = []

            # Build prefetch queries from all but the final stage
            for stage in stages[:-1]:
                # Calculate optimal prefetch limit based on vector type
                vector_type = VectorType(stage.get("vector_type", "dense"))
                stage_limit = self._calculate_prefetch_limit(
                    vector_type, stage["limit"]
                )

                # Build prefetch query
                prefetch_query = models.Prefetch(
                    query=stage["query_vector"],
                    using=stage["vector_name"],
                    limit=stage_limit,
                    filter=build_filter(stage.get("filter")),
                    params=self._get_search_params(SearchAccuracy(search_accuracy)),
                )
                prefetch_queries.append(prefetch_query)

            # Final stage query with fusion
            fusion_method = (
                models.Fusion.RRF
                if fusion_algorithm.lower() == "rrf"
                else models.Fusion.DBSF
            )

            # Execute multi-stage query
            results = await self.client.query_points(
                collection_name=collection_name,
                query=models.FusionQuery(fusion=fusion_method),
                prefetch=prefetch_queries,
                limit=limit,
                params=self._get_search_params(SearchAccuracy(search_accuracy)),
                with_payload=True,
                with_vectors=False,
            )

            # Format results
            return [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": point.payload or {},
                }
                for point in results.points
            ]

        except Exception as e:
            logger.exception(
                "Multi-stage search failed for collection %s",
                collection_name,
            )
            msg = f"Multi-stage search failed: {e}"
            raise QdrantServiceError(msg) from e

    async def hyde_search(
        self,
        collection_name: str,
        query: str,
        query_embedding: list[float],
        hypothetical_embeddings: list[list[float]],
        limit: int = 10,
        fusion_algorithm: str = "rrf",
        search_accuracy: str = "balanced",
    ) -> list[dict[str, Any]]:
        """Search using HyDE (Hypothetical Document Embeddings) with Query API prefetch.

        Args:
            collection_name: Collection to search
            query: Original query text
            query_embedding: Original query embedding
            hypothetical_embeddings: List of hypothetical document embeddings
            limit: Number of results to return
            fusion_algorithm: Fusion method ("rrf" or "dbsf")
            search_accuracy: Accuracy level

        Returns:
            List of search results

        Raises:
            QdrantServiceError: If search fails

        """
        # Monitor search performance
        if self.metrics_registry:
            decorator = self.metrics_registry.monitor_search_performance(
                collection=collection_name, query_type="hyde"
            )

            async def _monitored_search():
                return await self._execute_hyde_search(
                    collection_name,
                    query,
                    query_embedding,
                    hypothetical_embeddings,
                    limit,
                    fusion_algorithm,
                    search_accuracy,
                )

            return await decorator(_monitored_search)()
        return await self._execute_hyde_search(
            collection_name,
            query,
            query_embedding,
            hypothetical_embeddings,
            limit,
            fusion_algorithm,
            search_accuracy,
        )

    async def _execute_hyde_search(
        self,
        collection_name: str,
        _query: str,
        query_embedding: list[float],
        hypothetical_embeddings: list[list[float]],
        limit: int = 10,
        fusion_algorithm: str = "rrf",
        search_accuracy: str = "balanced",
    ) -> list[dict[str, Any]]:
        """Execute the actual HyDE search operation."""
        try:
            # Average hypothetical embeddings for wider retrieval
            hypothetical_vector = np.mean(hypothetical_embeddings, axis=0).tolist()

            # Calculate optimal prefetch limits
            hyde_limit = self._calculate_prefetch_limit(VectorType.HYDE, limit)
            dense_limit = self._calculate_prefetch_limit(VectorType.DENSE, limit)

            # Build prefetch queries
            prefetch_queries = [
                # HyDE embedding - cast wider net for discovery
                models.Prefetch(
                    query=hypothetical_vector,
                    using="dense",
                    limit=hyde_limit,
                    params=self._get_search_params(SearchAccuracy(search_accuracy)),
                ),
                # Original query - for precision
                models.Prefetch(
                    query=query_embedding,
                    using="dense",
                    limit=dense_limit,
                    params=self._get_search_params(SearchAccuracy(search_accuracy)),
                ),
            ]

            # Execute HyDE search with fusion
            fusion_method = (
                models.Fusion.RRF
                if fusion_algorithm.lower() == "rrf"
                else models.Fusion.DBSF
            )

            results = await self.client.query_points(
                collection_name=collection_name,
                query=query_embedding,  # Final fusion uses original query
                using="dense",
                prefetch=prefetch_queries,
                fusion=models.FusionQuery(fusion=fusion_method),
                limit=limit,
                params=self._get_search_params(SearchAccuracy(search_accuracy)),
                with_payload=True,
                with_vectors=False,
            )

            # Format results
            return [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": point.payload or {},
                }
                for point in results.points
            ]

        except Exception as e:
            logger.exception(
                "HyDE search failed for collection %s",
                collection_name,
            )
            msg = f"HyDE search failed: {e}"
            raise QdrantServiceError(msg) from e

    async def filtered_search(
        self,
        collection_name: str,
        query_vector: list[float],
        filters: dict[str, Any],
        limit: int = 10,
        search_accuracy: str = "balanced",
    ) -> list[dict[str, Any]]:
        """Optimized filtered search using indexed payload fields.

        Args:
            collection_name: Collection to search
            query_vector: Query vector
            filters: Filters to apply (doc_type, language, created_after, etc.)
            limit: Number of results to return
            search_accuracy: Accuracy level

        Returns:
            List of search results

        Raises:
            QdrantServiceError: If search fails

        """
        # Monitor search performance
        if self.metrics_registry:
            decorator = self.metrics_registry.monitor_search_performance(
                collection=collection_name, query_type="filtered"
            )

            async def _monitored_search():
                return await self._execute_filtered_search(
                    collection_name, query_vector, filters, limit, search_accuracy
                )

            return await decorator(_monitored_search)()
        return await self._execute_filtered_search(
            collection_name, query_vector, filters, limit, search_accuracy
        )

    async def _execute_filtered_search(
        self,
        collection_name: str,
        query_vector: list[float],
        filters: dict[str, Any],
        limit: int = 10,
        search_accuracy: str = "balanced",
    ) -> list[dict[str, Any]]:
        """Execute the actual filtered search operation."""
        # Validate input parameters
        if not isinstance(query_vector, list):
            msg = "query_vector must be a list"
            raise TypeError(msg)

        if not query_vector:
            msg = "query_vector cannot be empty"
            raise ValueError(msg)

        if not isinstance(filters, dict):
            msg = "filters must be a dictionary"
            raise TypeError(msg)

        # Validate vector dimensions (assuming 1536 for OpenAI embeddings)
        expected_dim = 1536
        if len(query_vector) != expected_dim:
            msg = (
                f"query_vector dimension {len(query_vector)} does not match "
                f"expected {expected_dim}"
            )
            raise ValueError(msg)

        try:
            # Build optimized Qdrant filter
            filter_obj = build_filter(filters)

            # Execute filtered search with Query API
            results = await self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using="dense",
                filter=filter_obj,
                limit=limit,
                params=self._get_search_params(SearchAccuracy(search_accuracy)),
                with_payload=True,
                with_vectors=False,
            )

            # Format results
            return [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": point.payload or {},
                }
                for point in results.points
            ]

        except Exception as e:
            logger.exception(
                "Filtered search failed for collection %s",
                collection_name,
            )
            msg = f"Filtered search failed: {e}"
            raise QdrantServiceError(msg) from e

    def _calculate_prefetch_limit(
        self, vector_type: VectorType, final_limit: int
    ) -> int:
        """Calculate optimal prefetch limit based on research findings."""
        return self.prefetch_config.calculate_prefetch_limit(vector_type, final_limit)

    def _get_search_params(self, accuracy_level: SearchAccuracy) -> models.SearchParams:
        """Get optimized search parameters for different accuracy levels."""
        params_map = {
            SearchAccuracy.FAST: models.SearchParams(hnsw_ef=50, exact=False),
            SearchAccuracy.BALANCED: models.SearchParams(hnsw_ef=100, exact=False),
            SearchAccuracy.ACCURATE: models.SearchParams(hnsw_ef=200, exact=False),
            SearchAccuracy.EXACT: models.SearchParams(exact=True),
        }

        return params_map.get(accuracy_level, params_map[SearchAccuracy.BALANCED])
