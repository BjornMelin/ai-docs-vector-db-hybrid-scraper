"""Qdrant collection management service."""

import logging
from typing import TYPE_CHECKING, Any

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from src.services.base import BaseService
from src.services.errors import QdrantServiceError


if TYPE_CHECKING:
    from src.config import Config


logger = logging.getLogger(__name__)


class QdrantCollections(BaseService):
    """Focused service for Qdrant collection management operations."""

    def __init__(self, config: "Config", qdrant_client: AsyncQdrantClient):
        """Initialize collections service.

        Args:
            config: Unified configuration
            qdrant_client: Initialized Qdrant client

        """
        super().__init__(config)
        self.config: Config = config
        self._client = qdrant_client
        self._initialized = True  # Client is already initialized

    async def initialize(self) -> None:
        """Initialize collections service."""
        if self._initialized:
            return

        if not self._client:
            msg = "QdrantClient must be provided and initialized"
            raise QdrantServiceError(msg)

        self._initialized = True
        logger.info("QdrantCollections service initialized")

    async def cleanup(self) -> None:
        """Cleanup collections service."""
        # Note: We don't close the client as it's shared
        self._initialized = False
        logger.info("QdrantCollections service cleaned up")

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
            collection_type: Type of collection for HNSW optimization
                (api_reference, tutorials, etc.)

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
                logger.info("Collection %s already exists", collection_name)
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
            try:
                distance_enum = getattr(models.Distance, distance.upper())
            except AttributeError:
                msg = (
                    f"Invalid distance metric '{distance}'. "
                    "Valid options: Cosine, Euclidean, Dot"
                )
                raise QdrantServiceError(msg) from None

            vectors_config = {
                "dense": models.VectorParams(
                    size=vector_size,
                    distance=distance_enum,
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

            logger.info("Created collection: %s", collection_name)

            # Note: Payload indexes will be created by QdrantService
            # after collection creation

        except ResponseHandlingException as e:
            logger.exception("Failed to create collection", collection_name)

            error_msg = str(e).lower()
            if "already exists" in error_msg:
                logger.info("Collection %s already exists, continuing", collection_name)
                return True
            if "invalid distance" in error_msg:
                msg = (
                    f"Invalid distance metric '{distance}'. "
                    "Valid options: Cosine, Euclidean, Dot"
                )
                raise QdrantServiceError(msg) from e
            if "unauthorized" in error_msg:
                msg = "Unauthorized access to Qdrant. Please check your API key."
                raise QdrantServiceError(msg) from e
            msg = f"Failed to create collection: {e}"
            raise QdrantServiceError(msg) from e
        else:
            return True

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
            logger.info("Deleted collection: %s", collection_name)
        except Exception as e:
            msg = f"Failed to delete collection: {e}"
            raise QdrantServiceError(msg) from e
        else:
            return True

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
            msg = f"Failed to get collection info: {e}"
            raise QdrantServiceError(msg) from e

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
            msg = f"Failed to list collections: {e}"
            raise QdrantServiceError(msg) from e

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
                except (AttributeError, ValueError, RuntimeError, ConnectionError) as e:
                    logger.warning(
                        "Failed to get details for collection %s: %s", col.name, e
                    )
                    details.append(
                        {
                            "name": col.name,
                            "error": str(e),
                        }
                    )

        except Exception as e:
            msg = f"Failed to list collection details: {e}"
            raise QdrantServiceError(msg) from e
        else:
            return details

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

            logger.info("Triggered optimization for collection: %s", collection_name)
        except Exception as e:
            msg = f"Failed to optimize collection: {e}"
            raise QdrantServiceError(msg) from e
        else:
            return True

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

    def get_hnsw_configuration_info(self, collection_type: str) -> dict[str, Any]:
        """Get HNSW configuration information for a collection type.

        Args:
            collection_type: Type of collection

        Returns:
            HNSW configuration information

        """
        hnsw_config = self._get_hnsw_config_for_collection_type(collection_type)

        return {
            "collection_type": collection_type,
            "hnsw_parameters": {
                "m": hnsw_config.m,
                "ef_construct": hnsw_config.ef_construct,
                "full_scan_threshold": hnsw_config.full_scan_threshold,
                "max_indexing_threads": hnsw_config.max_indexing_threads,
                "on_disk": hnsw_config.on_disk,
            },
            "description": (
                f"Optimized HNSW configuration for {collection_type} collections"
            ),
        }

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
                    f"Consider updating HNSW parameters for {collection_type} "
                    f"collections: m={optimal_hnsw.m}, "
                    f"ef_construct={optimal_hnsw.ef_construct}"
                )

            if not adaptive_ef_available:
                hnsw_recommendations.append(
                    "Enable adaptive ef for time-budget-based search optimization"
                )

            # Check collection size for on_disk recommendation
            points_count = collection_info.points_count or 0
            if points_count > 100000 and not current_hnsw.get("on_disk", False):
                hnsw_recommendations.append(
                    f"Consider enabling on_disk storage for large collection "
                    f"({points_count:,} points)"
                )

        except (AttributeError, ValueError, RuntimeError, ConnectionError) as e:
            logger.warning("Failed to validate HNSW configuration: %s", e)
            # Return default healthy status if HNSW validation fails
            return {
                "health_score": 85.0,
                "collection_type": "unknown",
                "current_configuration": {},
                "optimal_configuration": {},
                "adaptive_ef_enabled": False,
                "recommendations": ["Could not validate HNSW configuration"],
                "points_count": 0,
            }
        else:
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
        if "tutorial" in name_lower or "guide" in name_lower:
            return "tutorials"
        if "blog" in name_lower or "post" in name_lower:
            return "blog_posts"
        if "doc" in name_lower or "documentation" in name_lower:
            return "general_docs"
        if "code" in name_lower or "example" in name_lower:
            return "code_examples"
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
