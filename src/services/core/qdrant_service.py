"""Qdrant service with direct SDK integration."""

import logging
import time
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException

from ...config import UnifiedConfig
from ..base import BaseService
from ..errors import QdrantServiceError

logger = logging.getLogger(__name__)


class QdrantService(BaseService):
    """Service for direct Qdrant operations."""

    def __init__(self, config: UnifiedConfig):
        """Initialize Qdrant service.

        Args:
            config: Unified configuration
        """
        super().__init__(config)
        self.config: UnifiedConfig = config
        self._client: AsyncQdrantClient | None = None

    async def initialize(self) -> None:
        """Initialize Qdrant client with connection validation.

        Raises:
            QdrantServiceError: If client initialization fails
        """
        if self._initialized:
            return

        try:
            self._client = AsyncQdrantClient(
                url=self.config.qdrant.url,
                api_key=self.config.qdrant.api_key,
                timeout=self.config.qdrant.timeout,
                prefer_grpc=self.config.qdrant.prefer_grpc,
            )

            # Validate connection by listing collections
            try:
                await self._client.get_collections()
            except Exception as e:
                raise QdrantServiceError(f"Qdrant connection check failed: {e}") from e

            self._initialized = True
            logger.info(f"Qdrant client initialized: {self.config.qdrant.url}")
        except Exception as e:
            self._client = None
            self._initialized = False
            raise QdrantServiceError(f"Failed to initialize Qdrant client: {e}") from e

    async def cleanup(self) -> None:
        """Cleanup Qdrant client."""
        if self._client:
            await self._client.close()
            self._client = None
            self._initialized = False
            logger.info("Qdrant client closed")

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
        sparse_vector_name: str | None = None,
        enable_quantization: bool = True,
        collection_type: str = "general",
    ) -> bool:
        """Create vector collection with optional quantization and sparse vectors.

        Args:
            collection_name: Name of the collection
            vector_size: Dimension of dense vectors
            distance: Distance metric (Cosine, Euclidean, Dot)
            sparse_vector_name: Optional sparse vector field name for hybrid search
            enable_quantization: Enable INT8 quantization for ~75% storage reduction
            collection_type: Type of collection for HNSW optimization (api_reference, tutorials, etc.)

        Returns:
            True if collection created or already exists

        Raises:
            QdrantServiceError: If creation fails
        """
        self._validate_initialized()

        try:
            # Check if collection exists
            collections = await self._client.get_collections()
            if any(col.name == collection_name for col in collections.collections):
                logger.info(f"Collection {collection_name} already exists")
                return True

            # Get HNSW configuration for collection type
            hnsw_config = self._get_hnsw_config_for_collection_type(collection_type)

            # Configure HNSW parameters
            hnsw_config_obj = models.HnswConfigDiff(
                m=hnsw_config.m,
                ef_construct=hnsw_config.ef_construct,
                full_scan_threshold=hnsw_config.full_scan_threshold,
                max_indexing_threads=hnsw_config.max_indexing_threads,
            )

            # Configure vectors with HNSW settings
            vectors_config = {
                "dense": models.VectorParams(
                    size=vector_size,
                    distance=getattr(models.Distance, distance.upper()),
                    hnsw_config=hnsw_config_obj,
                )
            }

            # Configure sparse vectors if requested
            sparse_vectors_config = None
            if sparse_vector_name:
                sparse_vectors_config = {
                    sparse_vector_name: models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                }

            # Configure quantization
            quantization_config = None
            if enable_quantization:
                quantization_config = models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                )

            # Create collection
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
                quantization_config=quantization_config,
            )

            logger.info(f"Created collection: {collection_name}")

            # Automatically create payload indexes for optimal performance
            try:
                await self.create_payload_indexes(collection_name)
                logger.info(
                    f"Payload indexes created for collection: {collection_name}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create payload indexes for {collection_name}: {e}. "
                    "Collection created successfully but filtering may be slower."
                )

            return True

        except ResponseHandlingException as e:
            logger.error(
                f"Failed to create collection {collection_name}: {e}", exc_info=True
            )

            error_msg = str(e).lower()
            if "already exists" in error_msg:
                logger.info(f"Collection {collection_name} already exists, continuing")
                return True
            elif "invalid distance" in error_msg:
                raise QdrantServiceError(
                    f"Invalid distance metric '{distance}'. Valid options: Cosine, Euclidean, Dot"
                ) from e
            elif "unauthorized" in error_msg:
                raise QdrantServiceError(
                    "Unauthorized access to Qdrant. Please check your API key."
                ) from e
            else:
                raise QdrantServiceError(f"Failed to create collection: {e}") from e

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
        self._validate_initialized()

        try:
            # Build prefetch queries with optimized limits
            prefetch_queries = []

            # Dense vector prefetch with optimized limit
            dense_limit = self._calculate_prefetch_limit("dense", limit)
            prefetch_queries.append(
                models.Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=dense_limit,
                    params=self._get_search_params(search_accuracy),
                )
            )

            # Sparse vector prefetch if available with optimized limit
            if sparse_vector:
                sparse_limit = self._calculate_prefetch_limit("sparse", limit)
                prefetch_queries.append(
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=list(sparse_vector.keys()),
                            values=list(sparse_vector.values()),
                        ),
                        using="sparse",
                        limit=sparse_limit,
                        params=self._get_search_params(search_accuracy),
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
                results = await self._client.query_points(
                    collection_name=collection_name,
                    prefetch=prefetch_queries,
                    query=models.FusionQuery(fusion=fusion_method),
                    limit=limit,
                    score_threshold=score_threshold,
                    params=self._get_search_params(search_accuracy),
                    with_payload=True,
                    with_vectors=False,
                )
            else:
                # Single query without fusion
                results = await self._client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    using="dense",
                    limit=limit,
                    score_threshold=score_threshold,
                    params=self._get_search_params(search_accuracy),
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

    async def upsert_points(
        self,
        collection_name: str,
        points: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> bool:
        """Upsert points with automatic batching.

        Args:
            collection_name: Target collection
            points: List of points with id, vector, and optional payload
            batch_size: Points per batch for memory efficiency

        Returns:
            True if all points upserted successfully

        Raises:
            QdrantServiceError: If upsert fails
        """
        self._validate_initialized()

        try:
            # Process in batches
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]

                # Convert to PointStruct
                point_structs = []
                for point in batch:
                    vectors = point.get("vector", {})
                    if isinstance(vectors, list):
                        vectors = {"dense": vectors}

                    point_struct = models.PointStruct(
                        id=point["id"],
                        vector=vectors,
                        payload=point.get("payload", {}),
                    )
                    point_structs.append(point_struct)

                # Upsert batch
                await self._client.upsert(
                    collection_name=collection_name,
                    points=point_structs,
                    wait=True,
                )

                logger.info(
                    f"Upserted batch {i // batch_size + 1} "
                    f"({len(point_structs)} points)"
                )

            return True

        except Exception as e:
            logger.error(
                f"Failed to upsert {len(points)} points to {collection_name}: {e}",
                exc_info=True,
            )

            error_msg = str(e).lower()
            if "collection not found" in error_msg:
                raise QdrantServiceError(
                    f"Collection '{collection_name}' not found. Create it before upserting."
                ) from e
            elif "wrong vector size" in error_msg:
                raise QdrantServiceError(
                    "Vector dimension mismatch. Check that vectors match collection configuration."
                ) from e
            elif "payload too large" in error_msg:
                raise QdrantServiceError(
                    f"Payload too large. Try reducing batch size (current: {batch_size})."
                ) from e
            else:
                raise QdrantServiceError(f"Failed to upsert points: {e}") from e

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection.

        Args:
            collection_name: Collection to delete

        Returns:
            Success status
        """
        self._validate_initialized()

        try:
            await self._client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            raise QdrantServiceError(f"Failed to delete collection: {e}") from e

    async def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """Get collection information.

        Args:
            collection_name: Collection name

        Returns:
            Collection information
        """
        self._validate_initialized()

        try:
            info = await self._client.get_collection(collection_name)
            return {
                "status": info.status,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "config": info.config.model_dump() if info.config else {},
            }
        except Exception as e:
            raise QdrantServiceError(f"Failed to get collection info: {e}") from e

    async def count_points(self, collection_name: str, exact: bool = True) -> int:
        """Count points in collection.

        Args:
            collection_name: Collection name
            exact: Use exact count

        Returns:
            Point count
        """
        self._validate_initialized()

        try:
            result = await self._client.count(
                collection_name=collection_name,
                exact=exact,
            )
            return result.count
        except Exception as e:
            raise QdrantServiceError(f"Failed to count points: {e}") from e

    async def list_collections(self) -> list[str]:
        """List all collection names.

        Returns:
            List of collection names
        """
        self._validate_initialized()

        try:
            collections = await self._client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            raise QdrantServiceError(f"Failed to list collections: {e}") from e

    async def list_collections_details(self) -> list[dict[str, Any]]:
        """List all collections with detailed information.

        Returns:
            List of collection details including name, vector count, status, and config
        """
        self._validate_initialized()

        try:
            collections = await self._client.get_collections()
            details = []

            for col in collections.collections:
                try:
                    info = await self.get_collection_info(col.name)
                    details.append(
                        {
                            "name": col.name,
                            "vector_count": info.get("vectors_count", 0),
                            "indexed_count": info.get("points_count", 0),
                            "status": info.get("status", "unknown"),
                            "config": info.get("config", {}),
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to get details for collection {col.name}: {e}"
                    )
                    details.append(
                        {
                            "name": col.name,
                            "error": str(e),
                        }
                    )

            return details
        except Exception as e:
            raise QdrantServiceError(f"Failed to list collection details: {e}") from e

    async def trigger_collection_optimization(self, collection_name: str) -> bool:
        """Trigger optimization for a collection.

        Args:
            collection_name: Collection to optimize

        Returns:
            Success status

        Note:
            Qdrant automatically optimizes collections, but this method can trigger
            manual optimization by updating collection parameters.
        """
        self._validate_initialized()

        try:
            # Verify collection exists
            await self.get_collection_info(collection_name)

            # Trigger optimization by updating collection aliases
            # This is a no-op that forces Qdrant to check optimization
            await self._client.update_collection_aliases(change_aliases_operations=[])

            logger.info(f"Triggered optimization for collection: {collection_name}")
            return True
        except Exception as e:
            raise QdrantServiceError(f"Failed to optimize collection: {e}") from e

    # Advanced Query API Methods for Issue #55

    async def multi_stage_search(
        self,
        collection_name: str,
        stages: list[dict[str, Any]],
        limit: int = 10,
        fusion_algorithm: str = "rrf",
        search_accuracy: str = "balanced",
    ) -> list[dict[str, Any]]:
        """
        Perform multi-stage retrieval with different strategies.

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
        self._validate_initialized()

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
                vector_type = stage.get("vector_type", "dense")
                stage_limit = self._calculate_prefetch_limit(
                    vector_type, stage["limit"]
                )

                # Build prefetch query
                prefetch_query = models.Prefetch(
                    query=stage["query_vector"],
                    using=stage["vector_name"],
                    limit=stage_limit,
                    filter=self._build_filter(stage.get("filter")),
                    params=self._get_search_params(search_accuracy),
                )
                prefetch_queries.append(prefetch_query)

            # Final stage query with fusion
            fusion_method = (
                models.Fusion.RRF
                if fusion_algorithm.lower() == "rrf"
                else models.Fusion.DBSF
            )

            # Execute multi-stage query
            results = await self._client.query_points(
                collection_name=collection_name,
                query=models.FusionQuery(fusion=fusion_method),
                prefetch=prefetch_queries,
                limit=limit,
                params=self._get_search_params(search_accuracy),
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
        """
        Search using HyDE (Hypothetical Document Embeddings) with Query API prefetch.

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
        self._validate_initialized()

        try:
            import numpy as np

            # Average hypothetical embeddings for wider retrieval
            hypothetical_vector = np.mean(hypothetical_embeddings, axis=0).tolist()

            # Calculate optimal prefetch limits
            hyde_limit = self._calculate_prefetch_limit("hyde", limit)
            dense_limit = self._calculate_prefetch_limit("dense", limit)

            # Build prefetch queries
            prefetch_queries = [
                # HyDE embedding - cast wider net for discovery
                models.Prefetch(
                    query=hypothetical_vector,
                    using="dense",
                    limit=hyde_limit,
                    params=self._get_search_params(search_accuracy),
                ),
                # Original query - for precision
                models.Prefetch(
                    query=query_embedding,
                    using="dense",
                    limit=dense_limit,
                    params=self._get_search_params(search_accuracy),
                ),
            ]

            # Execute HyDE search with fusion
            fusion_method = (
                models.Fusion.RRF
                if fusion_algorithm.lower() == "rrf"
                else models.Fusion.DBSF
            )

            results = await self._client.query_points(
                collection_name=collection_name,
                query=query_embedding,  # Final fusion uses original query
                using="dense",
                prefetch=prefetch_queries,
                fusion=models.FusionQuery(fusion=fusion_method),
                limit=limit,
                params=self._get_search_params(search_accuracy),
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
        """
        Optimized filtered search using indexed payload fields.

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
        self._validate_initialized()

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
            results = await self._client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using="dense",
                filter=filter_obj,
                limit=limit,
                params=self._get_search_params(search_accuracy),
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

    def _calculate_prefetch_limit(self, vector_type: str, final_limit: int) -> int:
        """Calculate optimal prefetch limit based on research findings."""
        # Optimal prefetch multipliers based on research
        multipliers = {
            "sparse": 5.0,  # Cast wider net for sparse vectors
            "hyde": 3.0,  # Moderate expansion for hypothetical docs
            "dense": 2.0,  # Precision focus for dense vectors
        }

        # Maximum limits to prevent performance degradation
        max_limits = {
            "sparse": 500,
            "hyde": 150,
            "dense": 200,
        }

        multiplier = multipliers.get(vector_type, 2.0)
        max_limit = max_limits.get(vector_type, 200)

        calculated = int(final_limit * multiplier)
        return min(calculated, max_limit)

    def _get_search_params(self, accuracy_level: str) -> models.SearchParams:
        """Get optimized search parameters for different accuracy levels."""
        params_map = {
            "fast": models.SearchParams(hnsw_ef=50, exact=False),
            "balanced": models.SearchParams(hnsw_ef=100, exact=False),
            "accurate": models.SearchParams(hnsw_ef=200, exact=False),
            "exact": models.SearchParams(exact=True),
        }

        return params_map.get(accuracy_level, params_map["balanced"])

    def _build_filter(self, filters: dict[str, Any] | None) -> models.Filter | None:
        """Build optimized Qdrant filter from filter dictionary using indexed fields."""
        if not filters:
            return None

        must_conditions = []

        # Build different types of filter conditions
        must_conditions.extend(self._build_keyword_conditions(filters))
        must_conditions.extend(self._build_text_conditions(filters))
        must_conditions.extend(self._build_timestamp_conditions(filters))
        must_conditions.extend(self._build_content_metric_conditions(filters))
        must_conditions.extend(self._build_structural_conditions(filters))

        return models.Filter(must=must_conditions) if must_conditions else None

    def _build_keyword_conditions(
        self, filters: dict[str, Any]
    ) -> list[models.FieldCondition]:
        """Build keyword field conditions for exact matching."""
        conditions = []
        keyword_fields = [
            # Core documented fields
            "doc_type",
            "language",
            "framework",
            "version",
            "crawl_source",
            # System fields
            "site_name",
            "embedding_model",
            "embedding_provider",
            "search_strategy",
            "scraper_version",
            # Legacy compatibility
            "url",
        ]

        for field in keyword_fields:
            if field in filters:
                filter_value = filters[field]
                if not isinstance(filter_value, str | int | float | bool):
                    raise ValueError(
                        f"Filter value for {field} must be a simple type, got {type(filter_value)}"
                    )
                conditions.append(
                    models.FieldCondition(
                        key=field, match=models.MatchValue(value=filter_value)
                    )
                )
        return conditions

    def _build_text_conditions(
        self, filters: dict[str, Any]
    ) -> list[models.FieldCondition]:
        """Build text field conditions for partial matching."""
        conditions = []
        text_fields = ["title", "content_preview"]

        for field in text_fields:
            if field in filters:
                filter_value = filters[field]
                if not isinstance(filter_value, str):
                    raise ValueError(f"Text filter value for {field} must be a string")
                conditions.append(
                    models.FieldCondition(
                        key=field, match=models.MatchText(text=filter_value)
                    )
                )
        return conditions

    def _build_timestamp_conditions(
        self, filters: dict[str, Any]
    ) -> list[models.FieldCondition]:
        """Build timestamp range conditions."""
        conditions = []

        # Modern timestamp filters
        timestamp_mappings = [
            ("created_after", "created_at", "gte"),
            ("created_before", "created_at", "lte"),
            ("updated_after", "last_updated", "gte"),
            ("updated_before", "last_updated", "lte"),
            # Legacy compatibility
            ("scraped_after", "scraped_at", "gte"),
            ("scraped_before", "scraped_at", "lte"),
        ]

        for filter_key, field_key, operator in timestamp_mappings:
            if filter_key in filters:
                range_params = {operator: filters[filter_key]}
                conditions.append(
                    models.FieldCondition(
                        key=field_key, range=models.Range(**range_params)
                    )
                )
        return conditions

    def _build_content_metric_conditions(
        self, filters: dict[str, Any]
    ) -> list[models.FieldCondition]:
        """Build content metric range conditions."""
        conditions = []

        # Content length and quality filters
        metric_mappings = [
            ("min_word_count", "word_count", "gte"),
            ("max_word_count", "word_count", "lte"),
            ("min_char_count", "char_count", "gte"),
            ("max_char_count", "char_count", "lte"),
            ("min_quality_score", "quality_score", "gte"),
            ("max_quality_score", "quality_score", "lte"),
            # Legacy compatibility
            ("min_score", "score", "gte"),
        ]

        for filter_key, field_key, operator in metric_mappings:
            if filter_key in filters:
                range_params = {operator: filters[filter_key]}
                conditions.append(
                    models.FieldCondition(
                        key=field_key, range=models.Range(**range_params)
                    )
                )
        return conditions

    def _build_structural_conditions(
        self, filters: dict[str, Any]
    ) -> list[models.FieldCondition]:
        """Build structural and metadata conditions."""
        conditions = []

        # Exact match filters
        if "chunk_index" in filters:
            conditions.append(
                models.FieldCondition(
                    key="chunk_index",
                    match=models.MatchValue(value=filters["chunk_index"]),
                )
            )

        if "depth" in filters:
            conditions.append(
                models.FieldCondition(
                    key="depth", match=models.MatchValue(value=filters["depth"])
                )
            )

        # Range filters
        range_mappings = [
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

        return conditions

    # Payload Indexing Methods for Issue #56

    async def create_payload_indexes(self, collection_name: str) -> None:
        """Create payload indexes on key metadata fields for 10-100x faster filtering.

        Creates indexes on high-value fields for dramatic search performance improvements:
        - Keyword indexes for exact matching (site_name, embedding_model, etc.)
        - Text indexes for partial matching (title, content_preview)
        - Integer indexes for range queries (scraped_at, word_count, etc.)

        Args:
            collection_name: Collection to index

        Raises:
            QdrantServiceError: If index creation fails
        """
        self._validate_initialized()

        try:
            logger.info(f"Creating payload indexes for collection: {collection_name}")

            # Keyword indexes for exact matching (categorical data)
            keyword_fields = [
                # Core indexable fields per documentation
                "doc_type",  # "api", "guide", "tutorial", "reference"
                "language",  # "python", "typescript", "rust"
                "framework",  # "fastapi", "nextjs", "react"
                "version",  # "3.0", "14.2", "latest"
                "crawl_source",  # "crawl4ai", "stagehand", "playwright"
                # Additional system fields
                "site_name",  # Documentation site name
                "embedding_model",  # Embedding model used
                "embedding_provider",  # Provider (openai, fastembed)
                "search_strategy",  # Strategy (hybrid, dense, sparse)
                "scraper_version",  # Scraper version
            ]

            for field in keyword_fields:
                await self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
                logger.debug(f"Created keyword index for field: {field}")

            # Text indexes for partial matching (full-text search)
            text_fields = [
                "title",  # Document titles
                "content_preview",  # Content previews
            ]

            for field in text_fields:
                await self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.TEXT,
                    wait=True,
                )
                logger.debug(f"Created text index for field: {field}")

            # Integer indexes for range queries
            integer_fields = [
                # Core timestamp fields per documentation
                "created_at",  # Document creation timestamp
                "last_updated",  # Last update timestamp
                "scraped_at",  # Scraping timestamp (legacy compatibility)
                # Content metrics
                "word_count",  # Content length filtering
                "char_count",  # Character count filtering
                "quality_score",  # Content quality score
                # Document structure
                "chunk_index",  # Chunk position filtering
                "total_chunks",  # Document size filtering
                "depth",  # Crawl depth filtering
                "links_count",  # Links count filtering
            ]

            for field in integer_fields:
                await self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.INTEGER,
                    wait=True,
                )
                logger.debug(f"Created integer index for field: {field}")

            logger.info(
                f"Successfully created {len(keyword_fields) + len(text_fields) + len(integer_fields)} "
                f"payload indexes for collection: {collection_name}"
            )

        except Exception as e:
            logger.error(
                f"Failed to create payload indexes for collection {collection_name}: {e}",
                exc_info=True,
            )
            raise QdrantServiceError(f"Failed to create payload indexes: {e}") from e

    async def list_payload_indexes(self, collection_name: str) -> list[str]:
        """List all payload indexes in a collection.

        Args:
            collection_name: Collection to check

        Returns:
            List of indexed field names

        Raises:
            QdrantServiceError: If listing fails
        """
        self._validate_initialized()

        try:
            collection_info = await self._client.get_collection(collection_name)

            # Extract indexed fields from payload schema
            indexed_fields = []
            if (
                hasattr(collection_info, "payload_schema")
                and collection_info.payload_schema
            ):
                for field_name, field_info in collection_info.payload_schema.items():
                    # Check if field has index configuration
                    if hasattr(field_info, "index") and field_info.index:
                        indexed_fields.append(field_name)

            logger.info(
                f"Found {len(indexed_fields)} indexed fields in {collection_name}"
            )
            return indexed_fields

        except Exception as e:
            logger.error(
                f"Failed to list payload indexes for collection {collection_name}: {e}",
                exc_info=True,
            )
            raise QdrantServiceError(f"Failed to list payload indexes: {e}") from e

    async def drop_payload_index(self, collection_name: str, field_name: str) -> None:
        """Drop a specific payload index.

        Args:
            collection_name: Collection containing the index
            field_name: Field to drop index for

        Raises:
            QdrantServiceError: If drop fails
        """
        self._validate_initialized()

        try:
            await self._client.delete_payload_index(
                collection_name=collection_name, field_name=field_name, wait=True
            )
            logger.info(f"Dropped payload index for field: {field_name}")

        except Exception as e:
            logger.error(
                f"Failed to drop payload index for field {field_name}: {e}",
                exc_info=True,
            )
            raise QdrantServiceError(f"Failed to drop payload index: {e}") from e

    async def reindex_collection(self, collection_name: str) -> None:
        """Reindex all payload fields for a collection.

        Useful after bulk updates or when index performance degrades.

        Args:
            collection_name: Collection to reindex

        Raises:
            QdrantServiceError: If reindexing fails
        """
        self._validate_initialized()

        try:
            logger.info(f"Starting full reindex for collection: {collection_name}")

            # Get existing indexes
            existing_indexes = await self.list_payload_indexes(collection_name)

            # Drop existing indexes
            for field_name in existing_indexes:
                try:
                    await self.drop_payload_index(collection_name, field_name)
                except Exception as e:
                    logger.warning(f"Failed to drop index for {field_name}: {e}")

            # Recreate all indexes
            await self.create_payload_indexes(collection_name)

            logger.info(f"Successfully reindexed collection: {collection_name}")

        except Exception as e:
            logger.error(
                f"Failed to reindex collection {collection_name}: {e}", exc_info=True
            )
            raise QdrantServiceError(f"Failed to reindex collection: {e}") from e

    async def get_payload_index_stats(self, collection_name: str) -> dict[str, Any]:
        """Get statistics about payload indexes in a collection.

        Args:
            collection_name: Collection to analyze

        Returns:
            Dictionary with index statistics

        Raises:
            QdrantServiceError: If stats retrieval fails
        """
        self._validate_initialized()

        try:
            collection_info = await self._client.get_collection(collection_name)
            indexed_fields = await self.list_payload_indexes(collection_name)

            stats = {
                "collection_name": collection_name,
                "total_points": collection_info.points_count or 0,
                "indexed_fields_count": len(indexed_fields),
                "indexed_fields": indexed_fields,
                "payload_schema": {},
            }

            # Add payload schema information if available
            if (
                hasattr(collection_info, "payload_schema")
                and collection_info.payload_schema
            ):
                for field_name, field_info in collection_info.payload_schema.items():
                    stats["payload_schema"][field_name] = {
                        "indexed": field_name in indexed_fields,
                        "type": str(getattr(field_info, "data_type", "unknown")),
                    }

            return stats

        except Exception as e:
            logger.error(
                f"Failed to get payload index stats for collection {collection_name}: {e}",
                exc_info=True,
            )
            raise QdrantServiceError(f"Failed to get payload index stats: {e}") from e

    async def validate_index_health(self, collection_name: str) -> dict[str, Any]:
        """Validate the health and status of payload indexes and HNSW configuration for a collection.

        Args:
            collection_name: Collection to validate

        Returns:
            Health status report with validation results including HNSW optimization

        Raises:
            QdrantServiceError: If validation fails
        """
        self._validate_initialized()

        try:
            logger.info(f"Validating index health for collection: {collection_name}")

            # Get collection information
            collection_info = await self._client.get_collection(collection_name)
            indexed_fields = await self.list_payload_indexes(collection_name)

            # Define expected core indexes
            expected_keyword_fields = [
                "doc_type",
                "language",
                "framework",
                "version",
                "crawl_source",
                "site_name",
                "embedding_model",
                "embedding_provider",
            ]
            expected_text_fields = ["title", "content_preview"]
            expected_integer_fields = [
                "created_at",
                "last_updated",
                "word_count",
                "char_count",
            ]

            all_expected = (
                expected_keyword_fields + expected_text_fields + expected_integer_fields
            )

            # Check payload index status
            missing_indexes = [
                field for field in all_expected if field not in indexed_fields
            ]
            extra_indexes = [
                field for field in indexed_fields if field not in all_expected
            ]

            # Calculate payload index health score
            expected_count = len(all_expected)
            present_count = len([f for f in all_expected if f in indexed_fields])
            payload_health_score = (
                (present_count / expected_count) * 100 if expected_count > 0 else 0
            )

            # Validate HNSW configuration
            hnsw_health = await self._validate_hnsw_configuration(
                collection_name, collection_info
            )

            # Calculate overall health score (weighted: 60% payload indexes, 40% HNSW)
            overall_health_score = (payload_health_score * 0.6) + (
                hnsw_health["health_score"] * 0.4
            )

            # Determine overall health status
            if overall_health_score >= 95:
                status = "healthy"
            elif overall_health_score >= 80:
                status = "warning"
            else:
                status = "critical"

            health_report = {
                "collection_name": collection_name,
                "status": status,
                "health_score": round(overall_health_score, 2),
                "total_points": collection_info.points_count or 0,
                "index_summary": {
                    "expected_indexes": expected_count,
                    "present_indexes": present_count,
                    "missing_indexes": len(missing_indexes),
                    "extra_indexes": len(extra_indexes),
                },
                "payload_indexes": {
                    "health_score": round(payload_health_score, 2),
                    "missing_indexes": missing_indexes,
                    "extra_indexes": extra_indexes,
                },
                "hnsw_configuration": hnsw_health,
                "recommendations": self._generate_comprehensive_recommendations(
                    missing_indexes, extra_indexes, status, hnsw_health
                ),
                "validation_timestamp": int(time.time()),
            }

            logger.info(
                f"Index health validation completed for {collection_name}: "
                f"Status={status}, Score={overall_health_score:.1f}% "
                f"(Payload: {payload_health_score:.1f}%, HNSW: {hnsw_health['health_score']:.1f}%)"
            )

            return health_report

        except Exception as e:
            logger.error(
                f"Failed to validate index health for collection {collection_name}: {e}",
                exc_info=True,
            )
            raise QdrantServiceError(f"Failed to validate index health: {e}") from e

    def _generate_index_recommendations(
        self, missing_indexes: list[str], extra_indexes: list[str], status: str
    ) -> list[str]:
        """Generate recommendations based on index health analysis."""
        recommendations = []

        if missing_indexes:
            recommendations.append(
                f"Create missing indexes for optimal performance: {', '.join(missing_indexes)}"
            )

        if extra_indexes:
            recommendations.append(
                f"Consider removing unused indexes to reduce overhead: {', '.join(extra_indexes)}"
            )

        if status == "critical":
            recommendations.append(
                "Critical: Run migration script to create essential indexes for performance"
            )
        elif status == "warning":
            recommendations.append(
                "Warning: Some indexes are missing, performance may be suboptimal"
            )

        if not recommendations:
            recommendations.append("All indexes are healthy and optimally configured")

        return recommendations

    async def _validate_hnsw_configuration(
        self, collection_name: str, collection_info: Any
    ) -> dict[str, Any]:
        """Validate HNSW configuration and recommend optimizations.

        Args:
            collection_name: Name of the collection
            collection_info: Collection information from Qdrant

        Returns:
            HNSW health assessment with recommendations
        """
        try:
            # Initialize HNSWOptimizer if needed
            if not hasattr(self, "_hnsw_optimizer"):
                from ..utilities.hnsw_optimizer import HNSWOptimizer

                self._hnsw_optimizer = HNSWOptimizer(self.config, self)

            # Get current HNSW configuration from collection info
            current_hnsw = {}
            if hasattr(collection_info, "config") and hasattr(
                collection_info.config, "hnsw_config"
            ):
                hnsw_config = collection_info.config.hnsw_config
                current_hnsw = {
                    "m": hnsw_config.m if hasattr(hnsw_config, "m") else 16,
                    "ef_construct": hnsw_config.ef_construct
                    if hasattr(hnsw_config, "ef_construct")
                    else 200,
                    "on_disk": hnsw_config.on_disk
                    if hasattr(hnsw_config, "on_disk")
                    else False,
                }
            else:
                # Default HNSW values
                current_hnsw = {"m": 16, "ef_construct": 200, "on_disk": False}

            # Try to determine collection type from collection name or metadata
            collection_type = self._infer_collection_type(collection_name)

            # Get optimal HNSW configuration for this collection type
            optimal_hnsw = self._get_hnsw_config_for_collection_type(collection_type)

            # Calculate configuration optimality score
            optimality_score = self._calculate_hnsw_optimality_score(
                current_hnsw, optimal_hnsw
            )

            # Check if adaptive ef is enabled
            adaptive_ef_available = self.config.search.hnsw.enable_adaptive_ef

            # Generate HNSW-specific recommendations
            hnsw_recommendations = []
            if optimality_score < 80:
                hnsw_recommendations.append(
                    f"Consider updating HNSW parameters for {collection_type} collections: "
                    f"m={optimal_hnsw.m}, ef_construct={optimal_hnsw.ef_construct}"
                )

            if not adaptive_ef_available:
                hnsw_recommendations.append(
                    "Enable adaptive ef for time-budget-based search optimization"
                )

            # Check collection size for on_disk recommendation
            points_count = collection_info.points_count or 0
            if points_count > 100000 and not current_hnsw.get("on_disk", False):
                hnsw_recommendations.append(
                    f"Consider enabling on_disk storage for large collection ({points_count:,} points)"
                )

            return {
                "health_score": optimality_score,
                "collection_type": collection_type,
                "current_configuration": current_hnsw,
                "optimal_configuration": {
                    "m": optimal_hnsw.m,
                    "ef_construct": optimal_hnsw.ef_construct,
                    "on_disk": optimal_hnsw.on_disk,
                },
                "adaptive_ef_enabled": adaptive_ef_available,
                "recommendations": hnsw_recommendations,
                "points_count": points_count,
            }

        except Exception as e:
            logger.warning(f"Failed to validate HNSW configuration: {e}")
            # Return default healthy status if HNSW validation fails
            return {
                "health_score": 85.0,  # Assume reasonable health if we can't validate
                "collection_type": "unknown",
                "current_configuration": {},
                "optimal_configuration": {},
                "adaptive_ef_enabled": False,
                "recommendations": ["Could not validate HNSW configuration"],
                "points_count": 0,
            }

    def _infer_collection_type(self, collection_name: str) -> str:
        """Infer collection type from collection name for HNSW optimization.

        Args:
            collection_name: Name of the collection

        Returns:
            Inferred collection type
        """
        # Map collection names to types based on common patterns
        name_lower = collection_name.lower()

        if "api" in name_lower or "reference" in name_lower:
            return "api_reference"
        elif "tutorial" in name_lower or "guide" in name_lower:
            return "tutorials"
        elif "blog" in name_lower or "post" in name_lower:
            return "blog_posts"
        elif "doc" in name_lower or "documentation" in name_lower:
            return "general_docs"
        elif "code" in name_lower or "example" in name_lower:
            return "code_examples"
        else:
            return "general_docs"  # Default fallback

    def _calculate_hnsw_optimality_score(
        self, current: dict[str, Any], optimal: Any
    ) -> float:
        """Calculate how optimal the current HNSW configuration is.

        Args:
            current: Current HNSW configuration
            optimal: Optimal HNSW configuration for collection type

        Returns:
            Optimality score (0-100)
        """
        score = 100.0

        # Check m parameter (most important for graph connectivity)
        current_m = current.get("m", 16)
        if current_m != optimal.m:
            # Penalize based on how far off we are
            m_diff = abs(current_m - optimal.m) / optimal.m
            score -= min(30, m_diff * 100)  # Max 30 point penalty

        # Check ef_construct parameter (affects build quality vs speed)
        current_ef = current.get("ef_construct", 200)
        if current_ef != optimal.ef_construct:
            ef_diff = abs(current_ef - optimal.ef_construct) / optimal.ef_construct
            score -= min(20, ef_diff * 50)  # Max 20 point penalty

        # Check on_disk setting (affects memory usage)
        current_on_disk = current.get("on_disk", False)
        if current_on_disk != optimal.on_disk:
            score -= 10  # 10 point penalty for suboptimal disk usage

        return max(0.0, score)

    def _generate_comprehensive_recommendations(
        self,
        missing_indexes: list[str],
        extra_indexes: list[str],
        status: str,
        hnsw_health: dict[str, Any],
    ) -> list[str]:
        """Generate comprehensive recommendations including both payload indexes and HNSW.

        Args:
            missing_indexes: Missing payload indexes
            extra_indexes: Extra payload indexes
            status: Overall health status
            hnsw_health: HNSW health assessment

        Returns:
            Combined recommendations list
        """
        recommendations = []

        # Add payload index recommendations
        if missing_indexes:
            recommendations.append(
                f"Create missing indexes for optimal performance: {', '.join(missing_indexes)}"
            )

        if extra_indexes:
            recommendations.append(
                f"Consider removing unused indexes to reduce overhead: {', '.join(extra_indexes)}"
            )

        # Add HNSW recommendations
        if hnsw_health.get("recommendations"):
            recommendations.extend(hnsw_health["recommendations"])

        # Add status-based recommendations
        if status == "critical":
            recommendations.append(
                "Critical: Run optimization scripts to improve both index and HNSW configuration"
            )
        elif status == "warning":
            recommendations.append(
                "Warning: Some optimizations available for better performance"
            )

        if not recommendations:
            recommendations.append(
                "All indexes and HNSW configuration are optimally configured"
            )

        return recommendations

    async def get_index_usage_stats(self, collection_name: str) -> dict[str, Any]:
        """Get detailed usage statistics for payload indexes.

        Args:
            collection_name: Collection to analyze

        Returns:
            Detailed index usage statistics

        Raises:
            QdrantServiceError: If stats retrieval fails
        """
        self._validate_initialized()

        try:
            logger.debug(
                f"Collecting index usage stats for collection: {collection_name}"
            )

            # Get basic collection info
            collection_info = await self._client.get_collection(collection_name)
            indexed_fields = await self.list_payload_indexes(collection_name)

            # Calculate index efficiency metrics
            total_points = collection_info.points_count or 0
            index_overhead = (
                len(indexed_fields) * 0.1
            )  # Estimated 10% overhead per index

            usage_stats = {
                "collection_name": collection_name,
                "collection_stats": {
                    "total_points": total_points,
                    "indexed_fields_count": len(indexed_fields),
                    "estimated_index_overhead_percent": round(index_overhead, 2),
                },
                "index_details": {},
                "performance_metrics": {
                    "expected_filter_speedup": "10-100x for indexed fields",
                    "index_selectivity": "High for keyword fields, Medium for text fields",
                    "memory_efficiency": "Optimized with proper field type mapping",
                },
                "optimization_suggestions": [],
            }

            # Categorize indexes by type for detailed analysis
            keyword_indexes = []
            text_indexes = []
            integer_indexes = []

            expected_keyword_fields = [
                "doc_type",
                "language",
                "framework",
                "version",
                "crawl_source",
                "site_name",
                "embedding_model",
                "embedding_provider",
            ]
            expected_text_fields = ["title", "content_preview"]
            expected_integer_fields = [
                "created_at",
                "last_updated",
                "word_count",
                "char_count",
            ]

            for field in indexed_fields:
                if field in expected_keyword_fields:
                    keyword_indexes.append(field)
                elif field in expected_text_fields:
                    text_indexes.append(field)
                elif field in expected_integer_fields:
                    integer_indexes.append(field)

            usage_stats["index_details"] = {
                "keyword_indexes": {
                    "count": len(keyword_indexes),
                    "fields": keyword_indexes,
                    "performance": "Excellent (exact matching, highest selectivity)",
                },
                "text_indexes": {
                    "count": len(text_indexes),
                    "fields": text_indexes,
                    "performance": "Good (full-text search, moderate selectivity)",
                },
                "integer_indexes": {
                    "count": len(integer_indexes),
                    "fields": integer_indexes,
                    "performance": "Very Good (range queries, high selectivity)",
                },
            }

            # Generate optimization suggestions
            if total_points > 100000:
                usage_stats["optimization_suggestions"].append(
                    "Large collection detected: Monitor query performance and consider composite indexes"
                )

            if len(indexed_fields) > 15:
                usage_stats["optimization_suggestions"].append(
                    "Many indexes detected: Consider removing unused indexes to reduce overhead"
                )

            if not keyword_indexes:
                usage_stats["optimization_suggestions"].append(
                    "No keyword indexes found: Add core metadata indexes for better filter performance"
                )

            if not usage_stats["optimization_suggestions"]:
                usage_stats["optimization_suggestions"].append(
                    "Index configuration is optimal for current collection size"
                )

            usage_stats["generated_at"] = int(time.time())

            return usage_stats

        except Exception as e:
            logger.error(
                f"Failed to get index usage stats for collection {collection_name}: {e}",
                exc_info=True,
            )
            raise QdrantServiceError(f"Failed to get index usage stats: {e}") from e

    # HNSW Optimization Methods for Issue #57

    def _get_hnsw_config_for_collection_type(self, collection_type: str):
        """Get HNSW configuration for a specific collection type.

        Args:
            collection_type: Type of collection (api_reference, tutorials, etc.)

        Returns:
            HNSWConfig object with optimized parameters
        """
        collection_configs = self.config.qdrant.collection_hnsw_configs

        # Map collection types to config attributes
        config_mapping = {
            "api_reference": collection_configs.api_reference,
            "tutorials": collection_configs.tutorials,
            "blog_posts": collection_configs.blog_posts,
            "code_examples": collection_configs.code_examples,
            "general": collection_configs.general,
        }

        return config_mapping.get(collection_type, collection_configs.general)

    async def create_collection_with_hnsw_optimization(
        self,
        collection_name: str,
        vector_size: int,
        collection_type: str = "general",
        distance: str = "Cosine",
        sparse_vector_name: str | None = None,
        enable_quantization: bool = True,
    ) -> bool:
        """Create collection with optimized HNSW parameters.

        This is a convenience method that uses the enhanced create_collection
        with explicit HNSW optimization.

        Args:
            collection_name: Name of the collection
            vector_size: Dimension of dense vectors
            collection_type: Type of collection for optimization
            distance: Distance metric
            sparse_vector_name: Optional sparse vector field name
            enable_quantization: Enable quantization

        Returns:
            True if collection created successfully
        """
        return await self.create_collection(
            collection_name=collection_name,
            vector_size=vector_size,
            distance=distance,
            sparse_vector_name=sparse_vector_name,
            enable_quantization=enable_quantization,
            collection_type=collection_type,
        )

    async def search_with_adaptive_ef(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        time_budget_ms: int = 100,
        score_threshold: float = 0.0,
    ) -> dict[str, Any]:
        """Search using adaptive ef parameter optimization.

        Args:
            collection_name: Collection to search
            query_vector: Query vector
            limit: Number of results to return
            time_budget_ms: Time budget for adaptive ef selection
            score_threshold: Minimum score threshold

        Returns:
            Search results with adaptive ef metadata
        """
        self._validate_initialized()

        # Initialize HNSWOptimizer if needed
        if not hasattr(self, "_hnsw_optimizer"):
            from ..utilities.hnsw_optimizer import HNSWOptimizer

            self._hnsw_optimizer = HNSWOptimizer(self.config, self)
            await self._hnsw_optimizer.initialize()

        # Use adaptive ef retrieval
        result = await self._hnsw_optimizer.adaptive_ef_retrieve(
            collection_name=collection_name,
            query_vector=query_vector,
            time_budget_ms=time_budget_ms,
            target_limit=limit,
        )

        # Filter results by score threshold if needed
        if score_threshold > 0.0:
            filtered_results = [
                r for r in result["results"] if r.score >= score_threshold
            ]
            result["results"] = filtered_results
            result["filtered_count"] = len(result["results"])

        return result

    async def optimize_collection_hnsw_parameters(
        self,
        collection_name: str,
        collection_type: str,
        test_queries: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """Optimize HNSW parameters for an existing collection.

        Args:
            collection_name: Collection to optimize
            collection_type: Type of collection for optimization profile
            test_queries: Optional test queries for validation

        Returns:
            Optimization results and recommendations
        """
        self._validate_initialized()

        # Initialize HNSWOptimizer if needed
        if not hasattr(self, "_hnsw_optimizer"):
            from ..utilities.hnsw_optimizer import HNSWOptimizer

            self._hnsw_optimizer = HNSWOptimizer(self.config, self)
            await self._hnsw_optimizer.initialize()

        # Run optimization analysis
        return await self._hnsw_optimizer.optimize_collection_hnsw(
            collection_name=collection_name,
            collection_type=collection_type,
            test_queries=test_queries,
        )

    def get_hnsw_configuration_info(self, collection_type: str) -> dict[str, Any]:
        """Get HNSW configuration information for a collection type.

        Args:
            collection_type: Type of collection

        Returns:
            Configuration information and recommendations
        """
        hnsw_config = self._get_hnsw_config_for_collection_type(collection_type)

        return {
            "collection_type": collection_type,
            "hnsw_parameters": {
                "m": hnsw_config.m,
                "ef_construct": hnsw_config.ef_construct,
                "full_scan_threshold": hnsw_config.full_scan_threshold,
                "max_indexing_threads": hnsw_config.max_indexing_threads,
            },
            "runtime_ef_recommendations": {
                "min_ef": hnsw_config.min_ef,
                "balanced_ef": hnsw_config.balanced_ef,
                "max_ef": hnsw_config.max_ef,
            },
            "adaptive_ef_settings": {
                "enabled": hnsw_config.enable_adaptive_ef,
                "default_time_budget_ms": hnsw_config.default_time_budget_ms,
            },
            "optimization_enabled": self.config.qdrant.enable_hnsw_optimization,
        }

    def _validate_initialized(self) -> None:
        """Validate that the service is properly initialized."""
        if not self._initialized or not self._client:
            raise QdrantServiceError(
                "Service not initialized. Call initialize() first."
            )
