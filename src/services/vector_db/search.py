"""Focused QdrantSearch module for vector search operations.

This module provides a clean, focused implementation of search functionality
extracted from QdrantService, focusing specifically on search operations.
"""

import logging
from typing import Any

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client import models

from ...config import UnifiedConfig
from ...config.enums import SearchAccuracy
from ...config.enums import VectorType
from ...models.vector_search import PrefetchConfig
from ..errors import QdrantServiceError

logger = logging.getLogger(__name__)


class QdrantSearch:
    """Focused search operations for Qdrant with advanced Query API support."""

    def __init__(self, client: AsyncQdrantClient, config: UnifiedConfig):
        """Initialize search service.

        Args:
            client: Initialized Qdrant client
            config: Unified configuration
        """
        self.client = client
        self.config = config
        self.prefetch_config = PrefetchConfig()

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
                    filter=self._build_filter(stage.get("filter")),
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
            filter_obj = self._build_filter(filters)

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

    def _build_filter(self, filters: dict[str, Any] | None) -> models.Filter | None:
        """Build optimized Qdrant filter from filter dictionary using indexed fields."""
        if not filters:
            return None

        conditions = []

        # Keyword filters for exact matching
        keyword_fields = [
            "doc_type",
            "language",
            "framework",
            "version",
            "crawl_source",
            "site_name",
            "embedding_model",
            "embedding_provider",
            "url",
        ]
        for field in keyword_fields:
            if field in filters:
                if not isinstance(filters[field], str | int | float | bool):
                    raise ValueError(f"Filter value for {field} must be a simple type")
                conditions.append(
                    models.FieldCondition(
                        key=field, match=models.MatchValue(value=filters[field])
                    )
                )

        # Text filters for partial matching
        for field in ["title", "content_preview"]:
            if field in filters:
                if not isinstance(filters[field], str):
                    raise ValueError(f"Text filter value for {field} must be a string")
                conditions.append(
                    models.FieldCondition(
                        key=field, match=models.MatchText(text=filters[field])
                    )
                )

        # Range filters for timestamps and metrics
        range_mappings = [
            ("created_after", "created_at", "gte"),
            ("created_before", "created_at", "lte"),
            ("updated_after", "last_updated", "gte"),
            ("updated_before", "last_updated", "lte"),
            ("scraped_after", "scraped_at", "gte"),
            ("scraped_before", "scraped_at", "lte"),
            ("min_word_count", "word_count", "gte"),
            ("max_word_count", "word_count", "lte"),
            ("min_char_count", "char_count", "gte"),
            ("max_char_count", "char_count", "lte"),
            ("min_quality_score", "quality_score", "gte"),
            ("max_quality_score", "quality_score", "lte"),
            ("min_score", "score", "gte"),
            ("min_total_chunks", "total_chunks", "gte"),
            ("max_total_chunks", "total_chunks", "lte"),
            ("min_links_count", "links_count", "gte"),
            ("max_links_count", "links_count", "lte"),
        ]

        for filter_key, field_key, operator in range_mappings:
            if filter_key in filters:
                range_params = {operator: filters[filter_key]}
                conditions.append(
                    models.FieldCondition(
                        key=field_key, range=models.Range(**range_params)
                    )
                )

        # Exact match filters for structural fields
        for field in ["chunk_index", "depth"]:
            if field in filters:
                conditions.append(
                    models.FieldCondition(
                        key=field, match=models.MatchValue(value=filters[field])
                    )
                )

        return models.Filter(must=conditions) if conditions else None
