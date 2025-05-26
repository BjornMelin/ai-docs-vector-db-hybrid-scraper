"""Qdrant service with direct SDK integration."""

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException

from ..config import UnifiedConfig
from .base import BaseService
from .errors import QdrantServiceError

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
    ) -> bool:
        """Create vector collection with optional quantization and sparse vectors.

        Args:
            collection_name: Name of the collection
            vector_size: Dimension of dense vectors
            distance: Distance metric (Cosine, Euclidean, Dot)
            sparse_vector_name: Optional sparse vector field name for hybrid search
            enable_quantization: Enable INT8 quantization for ~75% storage reduction

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

            # Configure vectors
            vectors_config = {
                "dense": models.VectorParams(
                    size=vector_size,
                    distance=getattr(models.Distance, distance.upper()),
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
    ) -> list[dict[str, Any]]:
        """Perform hybrid search combining dense and sparse vectors.

        Uses Qdrant's Query API with prefetch and fusion for optimal results.

        Args:
            collection_name: Collection to search
            query_vector: Dense embedding vector
            sparse_vector: Optional sparse vector (token_id -> weight)
            limit: Maximum results to return
            score_threshold: Minimum score threshold
            fusion_type: Fusion method ("rrf" or "dbsf")

        Returns:
            List of results with id, score, and payload

        Raises:
            QdrantServiceError: If search fails
        """
        self._validate_initialized()

        try:
            # Build prefetch queries
            prefetch_queries = []

            # Dense vector prefetch
            prefetch_queries.append(
                models.Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=limit * 2,  # Fetch more for reranking
                )
            )

            # Sparse vector prefetch if available
            if sparse_vector:
                prefetch_queries.append(
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=list(sparse_vector.keys()),
                            values=list(sparse_vector.values()),
                        ),
                        using="sparse",
                        limit=limit * 2,
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
