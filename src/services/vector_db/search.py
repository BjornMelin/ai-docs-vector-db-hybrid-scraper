"""Focused QdrantSearch module for vector search operations.

This module provides a clean, focused implementation of search functionality
extracted from QdrantService, focusing specifically on search operations.
"""

import logging  # noqa: PLC0415
from typing import Any

import numpy as np
from qdrant_client import AsyncQdrantClient, models

from src.config import Config, SearchAccuracy, VectorType

from ...models.vector_search import PrefetchConfig
from ..errors import QdrantServiceError
from ..monitoring.metrics import get_metrics_registry
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
        except Exception:
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
            logger.error(
                f"Hybrid search failed in collection {collection_name}: {e}",
                exc_info=True,
            )

            error_msg = str(e).lower()
            if "collection not found" in error_msg:
                raise QdrantServiceError(
                    f"Collection '{collection_name}' not found. Please create it first."
                ) from e
            elif "wrong vector size" in error_msg:
                raise QdrantServiceError(
                    f"Vector dimension mismatch. Expected size for collection '{collection_name}'."
                ) from e
            elif "timeout" in error_msg:
                raise QdrantServiceError(
                    "Search request timed out. Try reducing the limit or simplifying the query."
                ) from e
            else:
                raise QdrantServiceError(f"Hybrid search failed: {e}") from e

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
        """Execute the actual multi-stage search operation."""
        # Validate input parameters
        if not stages:
            raise ValueError("Stages list cannot be empty")

        if not isinstance(stages, list):
            raise ValueError("Stages must be a list")

        for i, stage in enumerate(stages):
            if not isinstance(stage, dict):
                raise ValueError(f"Stage {i} must be a dictionary")
            if "query_vector" not in stage:
                raise ValueError(f"Stage {i} must contain 'query_vector'")
            if "vector_name" not in stage:
                raise ValueError(f"Stage {i} must contain 'vector_name'")
            if "limit" not in stage:
                raise ValueError(f"Stage {i} must contain 'limit'")

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
            logger.error(
                f"Multi-stage search failed in collection {collection_name}: {e}",
                exc_info=True,
            )
            raise QdrantServiceError(f"Multi-stage search failed: {e}") from e

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
        else:
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
            logger.error(
                f"HyDE search failed in collection {collection_name}: {e}",
                exc_info=True,
            )
            raise QdrantServiceError(f"HyDE search failed: {e}") from e

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
        else:
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
            raise ValueError("query_vector must be a list")

        if not query_vector:
            raise ValueError("query_vector cannot be empty")

        if not isinstance(filters, dict):
            raise ValueError("filters must be a dictionary")

        # Validate vector dimensions (assuming 1536 for OpenAI embeddings)
        expected_dim = 1536
        if len(query_vector) != expected_dim:
            raise ValueError(
                f"query_vector dimension {len(query_vector)} does not match expected {expected_dim}"
            )

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
            logger.error(
                f"Filtered search failed in collection {collection_name}: {e}",
                exc_info=True,
            )
            raise QdrantServiceError(f"Filtered search failed: {e}") from e

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
