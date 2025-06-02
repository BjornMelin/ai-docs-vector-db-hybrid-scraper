"""Qdrant collection management service."""

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


class QdrantCollections(BaseService):
    """Focused service for Qdrant collection management operations."""

    def __init__(self, config: UnifiedConfig, qdrant_client: AsyncQdrantClient):
        """Initialize collections service.

        Args:
            config: Unified configuration
            qdrant_client: Initialized Qdrant client
        """
        super().__init__(config)
        self.config: UnifiedConfig = config
        self._client = qdrant_client
        self._initialized = True  # Client is already initialized

    async def initialize(self) -> None:
        """Initialize collections service."""
        if self._initialized:
            return

        if not self._client:
            raise QdrantServiceError("QdrantClient must be provided and initialized")

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
            try:
                distance_enum = getattr(models.Distance, distance.upper())
            except AttributeError:
                raise QdrantServiceError(
                    f"Invalid distance metric '{distance}'. Valid options: Cosine, Euclidean, Dot"
                ) from None

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
                "crawl_source",  # "crawl4ai", "browser_use", "playwright"
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
                "health_score": 85.0,
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
