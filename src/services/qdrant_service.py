"""Qdrant service with direct SDK integration."""

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException

from .base import BaseService
from .config import APIConfig
from .errors import QdrantServiceError

logger = logging.getLogger(__name__)


class QdrantService(BaseService):
    """Service for direct Qdrant operations."""

    def __init__(self, config: APIConfig):
        """Initialize Qdrant service.

        Args:
            config: API configuration
        """
        super().__init__(config)
        self.config: APIConfig = config
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
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
                timeout=self.config.qdrant_timeout,
                prefer_grpc=self.config.qdrant_prefer_grpc,
            )
            
            # Validate connection by checking health
            health = await self._client.get_health()
            if not health:
                raise QdrantServiceError("Qdrant health check failed")
                
            self._initialized = True
            logger.info(f"Qdrant client initialized: {self.config.qdrant_url}")
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
        """Create vector collection with quantization.

        Args:
            collection_name: Name of the collection
            vector_size: Size of the dense vectors
            distance: Distance metric (Cosine, Euclidean, Dot)
            sparse_vector_name: Optional name for sparse vectors
            enable_quantization: Enable scalar quantization

        Returns:
            Success status
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
            raise QdrantServiceError(f"Failed to create collection: {e}")

    async def hybrid_search(
        self,
        collection_name: str,
        query_vector: list[float],
        sparse_vector: dict[int, float] | None = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        fusion_type: str = "rrf",
    ) -> list[dict[str, Any]]:
        """Hybrid search using Qdrant Query API.

        Args:
            collection_name: Collection to search
            query_vector: Dense query vector
            sparse_vector: Optional sparse vector (indices -> values)
            limit: Number of results
            score_threshold: Minimum score threshold
            fusion_type: Fusion method ('rrf' or 'dbsf')

        Returns:
            List of search results
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
            raise QdrantServiceError(f"Hybrid search failed: {e}")

    async def upsert_points(
        self,
        collection_name: str,
        points: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> bool:
        """Upsert points with batching.

        Args:
            collection_name: Collection name
            points: List of point data with id, vector, payload
            batch_size: Batch size for upsert

        Returns:
            Success status
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
            raise QdrantServiceError(f"Failed to upsert points: {e}")

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
            raise QdrantServiceError(f"Failed to delete collection: {e}")

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
            raise QdrantServiceError(f"Failed to get collection info: {e}")

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
            raise QdrantServiceError(f"Failed to count points: {e}")
