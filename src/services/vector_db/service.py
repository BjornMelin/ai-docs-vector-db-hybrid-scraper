"""Unified QdrantService facade that delegates to focused modules.

This module provides a clean facade over the modularized Qdrant functionality,
using the centralized ClientManager for all client operations.
"""

import logging
from typing import TYPE_CHECKING

from src.config import Config
from src.services.base import BaseService
from src.services.errors import QdrantServiceError

from .collections import QdrantCollections
from .documents import QdrantDocuments
from .indexing import QdrantIndexing
from .search import QdrantSearch


# Removed search interceptor (over-engineered deployment infrastructure)

if TYPE_CHECKING:
    from src.infrastructure.client_manager import ClientManager

logger = logging.getLogger(__name__)


class QdrantService(BaseService):
    """Unified Qdrant service facade delegating to focused modules."""

    def __init__(self, config: Config, client_manager: "ClientManager"):
        """Initialize Qdrant service with modular components.

        Args:
            config: Unified configuration
            client_manager: ClientManager instance for dependency injection

        """
        super().__init__(config)
        self.config: Config = config
        self._client_manager = client_manager

        # Initialize focused modules (will be set after client initialization)
        self._collections: QdrantCollections | None = None
        self._search: QdrantSearch | None = None
        self._indexing: QdrantIndexing | None = None
        self._documents: QdrantDocuments | None = None

        # Enterprise deployment infrastructure components (feature flag controlled)
        self._feature_flag_manager = None
        self._ab_testing_manager = None
        self._blue_green_deployment = None
        self._canary_deployment = None

    async def initialize(self) -> None:
        """Initialize all Qdrant modules with connection validation.

        Raises:
            QdrantServiceError: If initialization fails

        """
        if self._initialized:
            return

        try:
            # Get the Qdrant client from ClientManager
            client = await self._client_manager.get_qdrant_client()

            # Initialize all focused modules with the shared client
            self._collections = QdrantCollections(self.config, client)
            self._search = QdrantSearch(client, self.config)
            self._indexing = QdrantIndexing(client, self.config)
            self._documents = QdrantDocuments(client, self.config)

            # Initialize each module
            await self._collections.initialize()

            # Initialize deployment infrastructure if enabled
            await self._initialize_deployment_services()

            self._initialized = True
            logger.info("QdrantService initialized with modular architecture")

        except Exception as e:
            self._initialized = False
            msg = f"Failed to initialize QdrantService: {e}"
            raise QdrantServiceError(msg) from e

    async def _initialize_deployment_services(self) -> None:
        """Initialize deployment services based on configuration and feature flags.

        Conditionally initializes enterprise deployment services:
        - Feature flag management
        - A/B testing manager
        - Blue-green deployment
        - Canary deployment

        Services are only initialized if enabled in configuration.
        """
        try:
            # Initialize feature flag manager if deployment services are enabled
            if self.config.deployment.enable_feature_flags:
                self._feature_flag_manager = (
                    await self._client_manager.get_feature_flag_manager()
                )
                logger.info("Initialized FeatureFlagManager for QdrantService")

            # Initialize A/B testing manager if enabled
            if self.config.deployment.enable_ab_testing:
                self._ab_testing_manager = (
                    await self._client_manager.get_ab_testing_manager()
                )
                if self._ab_testing_manager:
                    logger.info("Initialized ABTestingManager for QdrantService")

            # Store deployment services for search routing
            if (
                self.config.deployment.enable_blue_green
                or self.config.deployment.enable_canary
                or self.config.deployment.enable_ab_testing
            ):
                # Get deployment services and store references
                blue_green = await self._client_manager.get_blue_green_deployment()
                canary = await self._client_manager.get_canary_deployment()

                # Store deployment services for routing decisions
                self._blue_green_deployment = blue_green
                self._canary_deployment = canary

                logger.info("Initialized deployment services for QdrantService routing")

        except Exception as e:
            logger.warning(
                f"Failed to initialize deployment services: {e}. Continuing with standard mode."
            )
            # Don't raise exception - deployment services are optional

    async def cleanup(self) -> None:
        """Cleanup all Qdrant modules (delegated to ClientManager)."""
        if self._collections:
            await self._collections.cleanup()

        # Note: ClientManager handles client cleanup, we just reset our references
        self._collections = None
        self._search = None
        self._indexing = None
        self._documents = None

        # Reset deployment service references
        self._feature_flag_manager = None
        self._ab_testing_manager = None
        self._blue_green_deployment = None
        self._canary_deployment = None

        self._initialized = False
        logger.info("QdrantService cleanup completed")

    # Collection Management API (delegates to QdrantCollections)

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
            collection_name: Name of the collection to create
            vector_size: Dimension of the vectors to be stored
            distance: Distance metric for similarity search (Cosine, Euclid, Dot)
            sparse_vector_name: Optional name for sparse vector field
            enable_quantization: Whether to enable vector quantization for storage efficiency
            collection_type: Type of collection for specialized configurations

        Returns:
            bool: True if collection created successfully

        Raises:
            QdrantServiceError: If collection creation fails

        """
        self._validate_initialized()

        # Create the collection first
        result = await self._collections.create_collection(
            collection_name=collection_name,
            vector_size=vector_size,
            distance=distance,
            sparse_vector_name=sparse_vector_name,
            enable_quantization=enable_quantization,
            collection_type=collection_type,
        )

        # Create payload indexes for optimal performance
        if result:
            try:
                await self._indexing.create_payload_indexes(collection_name)
                logger.info(
                    f"Payload indexes created for collection: {collection_name}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create payload indexes for {collection_name}: {e}. "
                    "Collection created successfully but filtering may be slower."
                )

        return result

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            bool: True if collection deleted successfully

        Raises:
            QdrantServiceError: If collection deletion fails

        """
        self._validate_initialized()
        return await self._collections.delete_collection(collection_name)

    async def list_collections(self) -> list[str]:
        """List all collection names.

        Returns:
            list[str]: List of collection names in the database

        Raises:
            QdrantServiceError: If listing collections fails

        """
        self._validate_initialized()
        return await self._collections.list_collections()

    async def list_collections_details(self) -> list[dict[str, object]]:
        """List all collections with detailed information.

        Returns:
            list[dict[str, object]]: List of collection details including:
                - name: Collection name
                - vectors_count: Number of vectors
                - indexed_vectors_count: Number of indexed vectors
                - config: Collection configuration

        Raises:
            QdrantServiceError: If listing collection details fails

        """
        self._validate_initialized()
        return await self._collections.list_collections_details()

    async def get_collection_info(self, collection_name: str) -> dict[str, object]:
        """Get collection information.

        Args:
            collection_name: Name of the collection to inspect

        Returns:
            dict[str, object]: Collection information including:
                - status: Collection status
                - vectors_count: Number of vectors
                - indexed_vectors_count: Number of indexed vectors
                - config: Collection configuration
                - optimizer_status: HNSW optimizer status

        Raises:
            QdrantServiceError: If getting collection info fails

        """
        self._validate_initialized()
        return await self._collections.get_collection_info(collection_name)

    async def trigger_collection_optimization(self, collection_name: str) -> bool:
        """Trigger optimization for a collection.

        Args:
            collection_name: Name of the collection to optimize

        Returns:
            bool: True if optimization triggered successfully

        Raises:
            QdrantServiceError: If optimization trigger fails

        """
        self._validate_initialized()
        return await self._collections.trigger_collection_optimization(collection_name)

    # Search API (delegates to QdrantSearch)

    async def hybrid_search(
        self,
        collection_name: str,
        query_vector: list[float],
        sparse_vector: dict[int, float] | None = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        fusion_type: str = "rrf",
        search_accuracy: str = "balanced",
        user_id: str | None = None,
        request_id: str | None = None,
    ) -> list[dict[str, object]]:
        """Perform hybrid search combining dense and sparse vectors.

        Args:
            collection_name: Name of the collection to search
            query_vector: Dense query vector for similarity search
            sparse_vector: Optional sparse vector for keyword matching
            limit: Maximum number of results to return
            score_threshold: Minimum score threshold for results
            fusion_type: Fusion algorithm ("rrf" or "dbsf")
            search_accuracy: Accuracy level ("balanced", "fast", "accurate")
            user_id: Optional user ID for canary routing consistency
            request_id: Optional request ID for canary metrics tracking

        Returns:
            list[dict[str, object]]: Search results with score and payload

        Raises:
            QdrantServiceError: If search fails

        """
        self._validate_initialized()

        # Route search through deployment infrastructure if enabled
        if self._ab_testing_manager and user_id:
            # Check for active A/B tests and route accordingly
            variant = await self._ab_testing_manager.assign_user_to_variant(
                test_id="hybrid_search_test", user_id=user_id
            )
            if variant and variant == "variant":
                # Use alternative search parameters for A/B testing
                search_accuracy = "accurate"  # Example: use higher accuracy for variant

        # Check canary deployment routing
        if self._canary_deployment and user_id:
            canary_assignment = await self._canary_deployment.should_route_to_canary(
                user_id
            )
            if canary_assignment:
                # Route to canary version with enhanced monitoring
                logger.debug(
                    f"Routing user {user_id} to canary deployment for hybrid search"
                )

        # Execute search with deployment tracking
        result = await self._search.hybrid_search(
            collection_name=collection_name,
            query_vector=query_vector,
            sparse_vector=sparse_vector,
            limit=limit,
            score_threshold=score_threshold,
            fusion_type=fusion_type,
            search_accuracy=search_accuracy,
        )

        # Track metrics for deployment monitoring
        if self._ab_testing_manager and request_id:
            await self._ab_testing_manager.track_conversion(
                test_id="hybrid_search_test",
                user_id=user_id or "anonymous",
                event_type="search_executed",
                request_id=request_id,
            )

        return result

    async def multi_stage_search(
        self,
        collection_name: str,
        stages: list[dict[str, object]],
        limit: int = 10,
        fusion_algorithm: str = "rrf",
        search_accuracy: str = "balanced",
        user_id: str | None = None,
        request_id: str | None = None,
    ) -> list[dict[str, object]]:
        """Perform multi-stage retrieval with different strategies.

        Args:
            collection_name: Name of the collection to search
            stages: List of search stage configurations, each containing:
                - query_vector: Vector for this stage
                - filters: Optional filters for this stage
                - weight: Weight for this stage in fusion
            limit: Maximum number of results to return
            fusion_algorithm: Algorithm for combining results ("rrf" or "dbsf")
            search_accuracy: Accuracy level ("balanced", "fast", "accurate")
            user_id: Optional user ID for canary routing consistency
            request_id: Optional request ID for canary metrics tracking

        Returns:
            list[dict[str, object]]: Fused search results

        Raises:
            QdrantServiceError: If multi-stage search fails

        """
        self._validate_initialized()

        # Route search through deployment infrastructure if enabled
        if self._ab_testing_manager and user_id:
            # Check for active A/B tests
            variant = await self._ab_testing_manager.assign_user_to_variant(
                test_id="multi_stage_search_test", user_id=user_id
            )
            if variant and variant == "variant":
                # Modify stages for A/B testing (e.g., add additional stage)
                search_accuracy = "accurate"

        # Check canary deployment routing
        if self._canary_deployment and user_id:
            canary_assignment = await self._canary_deployment.should_route_to_canary(
                user_id
            )
            if canary_assignment:
                logger.debug(
                    f"Routing user {user_id} to canary deployment for multi-stage search"
                )

        # Execute search with deployment tracking
        result = await self._search.multi_stage_search(
            collection_name=collection_name,
            stages=stages,
            limit=limit,
            fusion_algorithm=fusion_algorithm,
            search_accuracy=search_accuracy,
        )

        # Track metrics for deployment monitoring
        if self._ab_testing_manager and request_id:
            await self._ab_testing_manager.track_conversion(
                test_id="multi_stage_search_test",
                user_id=user_id or "anonymous",
                event_type="search_executed",
                request_id=request_id,
            )

        return result

    async def hyde_search(
        self,
        collection_name: str,
        query: str,
        query_embedding: list[float],
        hypothetical_embeddings: list[list[float]],
        limit: int = 10,
        fusion_algorithm: str = "rrf",
        search_accuracy: str = "balanced",
        user_id: str | None = None,
        request_id: str | None = None,
    ) -> list[dict[str, object]]:
        """Search using HyDE (Hypothetical Document Embeddings).

        Args:
            collection_name: Name of the collection to search
            query: Original search query text
            query_embedding: Embedding of the original query
            hypothetical_embeddings: List of embeddings from hypothetical documents
            limit: Maximum number of results to return
            fusion_algorithm: Algorithm for combining results ("rrf" or "dbsf")
            search_accuracy: Accuracy level ("balanced", "fast", "accurate")
            user_id: Optional user ID for canary routing consistency
            request_id: Optional request ID for canary metrics tracking

        Returns:
            list[dict[str, object]]: Search results combining query and hypothetical matches

        Raises:
            QdrantServiceError: If HyDE search fails

        """
        self._validate_initialized()

        # Route search through deployment infrastructure if enabled
        if self._ab_testing_manager and user_id:
            # Check for active A/B tests
            variant = await self._ab_testing_manager.assign_user_to_variant(
                test_id="hyde_search_test", user_id=user_id
            )
            if variant and variant == "variant":
                # Modify HyDE parameters for A/B testing
                search_accuracy = "accurate"

        # Check canary deployment routing
        if self._canary_deployment and user_id:
            canary_assignment = await self._canary_deployment.should_route_to_canary(
                user_id
            )
            if canary_assignment:
                logger.debug(
                    f"Routing user {user_id} to canary deployment for HyDE search"
                )

        # Execute search with deployment tracking
        result = await self._search.hyde_search(
            collection_name=collection_name,
            query=query,
            query_embedding=query_embedding,
            hypothetical_embeddings=hypothetical_embeddings,
            limit=limit,
            fusion_algorithm=fusion_algorithm,
            search_accuracy=search_accuracy,
        )

        # Track metrics for deployment monitoring
        if self._ab_testing_manager and request_id:
            await self._ab_testing_manager.track_conversion(
                test_id="hyde_search_test",
                user_id=user_id or "anonymous",
                event_type="search_executed",
                request_id=request_id,
            )

        return result

    async def filtered_search(
        self,
        collection_name: str,
        query_vector: list[float],
        filters: dict[str, object],
        limit: int = 10,
        search_accuracy: str = "balanced",
        user_id: str | None = None,
        request_id: str | None = None,
    ) -> list[dict[str, object]]:
        """Optimized filtered search using indexed payload fields.

        Args:
            collection_name: Name of the collection to search
            query_vector: Query vector for similarity search
            filters: Filter conditions for payload fields (e.g., {"key": "value"})
            limit: Maximum number of results to return
            search_accuracy: Accuracy level ("balanced", "fast", "accurate")
            user_id: Optional user ID for canary routing consistency
            request_id: Optional request ID for canary metrics tracking

        Returns:
            list[dict[str, object]]: Filtered search results

        Raises:
            QdrantServiceError: If filtered search fails

        """
        self._validate_initialized()

        # Route search through deployment infrastructure if enabled
        if self._ab_testing_manager and user_id:
            # Check for active A/B tests
            variant = await self._ab_testing_manager.assign_user_to_variant(
                test_id="filtered_search_test", user_id=user_id
            )
            if variant and variant == "variant":
                # Modify filtering for A/B testing
                search_accuracy = "accurate"

        # Check canary deployment routing
        if self._canary_deployment and user_id:
            canary_assignment = await self._canary_deployment.should_route_to_canary(
                user_id
            )
            if canary_assignment:
                logger.debug(
                    f"Routing user {user_id} to canary deployment for filtered search"
                )

        # Execute search with deployment tracking
        result = await self._search.filtered_search(
            collection_name=collection_name,
            query_vector=query_vector,
            filters=filters,
            limit=limit,
            search_accuracy=search_accuracy,
        )

        # Track metrics for deployment monitoring
        if self._ab_testing_manager and request_id:
            await self._ab_testing_manager.track_conversion(
                test_id="filtered_search_test",
                user_id=user_id or "anonymous",
                event_type="search_executed",
                request_id=request_id,
            )

        return result

    # Indexing API (delegates to QdrantIndexing)

    async def create_payload_indexes(self, collection_name: str) -> None:
        """Create payload indexes on key metadata fields.

        Args:
            collection_name: Name of the collection to index

        Raises:
            QdrantServiceError: If index creation fails

        """
        self._validate_initialized()
        await self._indexing.create_payload_indexes(collection_name)

    async def list_payload_indexes(self, collection_name: str) -> list[str]:
        """List all payload indexes in a collection.

        Args:
            collection_name: Name of the collection to inspect

        Returns:
            list[str]: List of indexed field names

        Raises:
            QdrantServiceError: If listing indexes fails

        """
        self._validate_initialized()
        return await self._indexing.list_payload_indexes(collection_name)

    async def drop_payload_index(self, collection_name: str, field_name: str) -> None:
        """Drop a specific payload index.

        Args:
            collection_name: Name of the collection containing the index
            field_name: Name of the field to drop index from

        Raises:
            QdrantServiceError: If dropping index fails

        """
        self._validate_initialized()
        await self._indexing.drop_payload_index(collection_name, field_name)

    async def reindex_collection(self, collection_name: str) -> None:
        """Reindex all payload fields for a collection.

        Args:
            collection_name: Name of the collection to reindex

        Raises:
            QdrantServiceError: If reindexing fails

        """
        self._validate_initialized()
        await self._indexing.reindex_collection(collection_name)

    async def get_payload_index_stats(self, collection_name: str) -> dict[str, object]:
        """Get statistics about payload indexes in a collection.

        Args:
            collection_name: Name of the collection to analyze

        Returns:
            dict[str, object]: Index statistics including:
                - indexed_fields: List of indexed fields
                - total_indexes: Total number of indexes
                - index_sizes: Size of each index

        Raises:
            QdrantServiceError: If getting stats fails

        """
        self._validate_initialized()
        return await self._indexing.get_payload_index_stats(collection_name)

    async def validate_index_health(self, collection_name: str) -> dict[str, object]:
        """Validate the health and status of payload indexes.

        Args:
            collection_name: Name of the collection to validate

        Returns:
            dict[str, object]: Health report including:
                - healthy: Overall health status
                - issues: List of any detected issues
                - recommendations: Optimization recommendations

        Raises:
            QdrantServiceError: If validation fails

        """
        self._validate_initialized()
        return await self._indexing.validate_index_health(collection_name)

    async def get_index_usage_stats(self, collection_name: str) -> dict[str, object]:
        """Get detailed usage statistics for payload indexes.

        Args:
            collection_name: Name of the collection to analyze

        Returns:
            dict[str, object]: Usage statistics including:
                - query_count: Number of queries using indexes
                - hit_rate: Index hit rate percentage
                - performance_metrics: Query performance data

        Raises:
            QdrantServiceError: If getting stats fails

        """
        self._validate_initialized()
        return await self._indexing.get_index_usage_stats(collection_name)

    # Document API (delegates to QdrantDocuments)

    async def upsert_points(
        self,
        collection_name: str,
        points: list[dict[str, object]],
        batch_size: int = 100,
    ) -> bool:
        """Upsert points with automatic batching.

        Args:
            collection_name: Name of the collection to upsert into
            points: List of point dictionaries containing:
                - id: Point ID (str or int)
                - vector: Vector data (list[float])
                - payload: Optional metadata (dict)
            batch_size: Number of points to process per batch

        Returns:
            bool: True if all points upserted successfully

        Raises:
            QdrantServiceError: If upsert fails

        """
        self._validate_initialized()
        return await self._documents.upsert_points(
            collection_name=collection_name,
            points=points,
            batch_size=batch_size,
        )

    async def get_points(
        self,
        collection_name: str,
        point_ids: list[str | int],
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[dict[str, object]]:
        """Retrieve specific points by their IDs.

        Args:
            collection_name: Name of the collection to retrieve from
            point_ids: List of point IDs to retrieve
            with_payload: Whether to include payload data
            with_vectors: Whether to include vector data

        Returns:
            list[dict[str, object]]: Retrieved points with requested data

        Raises:
            QdrantServiceError: If retrieval fails

        """
        self._validate_initialized()
        return await self._documents.get_points(
            collection_name=collection_name,
            point_ids=point_ids,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

    async def delete_points(
        self,
        collection_name: str,
        point_ids: list[str | int] | None = None,
        filter_condition: dict[str, object] | None = None,
    ) -> bool:
        """Delete points by IDs or filter condition.

        Args:
            collection_name: Name of the collection to delete from
            point_ids: Optional list of specific point IDs to delete
            filter_condition: Optional filter to delete matching points

        Returns:
            bool: True if deletion successful

        Raises:
            QdrantServiceError: If deletion fails

        """
        self._validate_initialized()
        return await self._documents.delete_points(
            collection_name=collection_name,
            point_ids=point_ids,
            filter_condition=filter_condition,
        )

    async def update_point_payload(
        self,
        collection_name: str,
        point_id: str | int,
        payload: dict[str, object],
        replace: bool = False,
    ) -> bool:
        """Update payload for a specific point.

        Args:
            collection_name: Name of the collection containing the point
            point_id: ID of the point to update
            payload: New payload data to set or merge
            replace: If True, replace entire payload; if False, merge with existing

        Returns:
            bool: True if update successful

        Raises:
            QdrantServiceError: If update fails

        """
        self._validate_initialized()
        return await self._documents.update_point_payload(
            collection_name=collection_name,
            point_id=point_id,
            payload=payload,
            replace=replace,
        )

    async def count_points(
        self,
        collection_name: str,
        filter_condition: dict[str, object] | None = None,
        exact: bool = True,
    ) -> int:
        """Count points in collection with optional filtering.

        Args:
            collection_name: Name of the collection to count
            filter_condition: Optional filter to count matching points only
            exact: If True, return exact count; if False, return approximate

        Returns:
            int: Number of points matching the criteria

        Raises:
            QdrantServiceError: If counting fails

        """
        self._validate_initialized()
        return await self._documents.count_points(
            collection_name=collection_name,
            filter_condition=filter_condition,
            exact=exact,
        )

    async def scroll_points(
        self,
        collection_name: str,
        limit: int = 100,
        offset: str | int | None = None,
        filter_condition: dict[str, object] | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> dict[str, object]:
        """Scroll through points in a collection with pagination.

        Args:
            collection_name: Name of the collection to scroll through
            limit: Maximum number of points per page
            offset: Pagination offset (point ID or index)
            filter_condition: Optional filter to scroll matching points only
            with_payload: Whether to include payload data
            with_vectors: Whether to include vector data

        Returns:
            dict[str, object]: Scroll results containing:
                - points: List of points in current page
                - next_page_offset: Offset for next page (if available)

        Raises:
            QdrantServiceError: If scrolling fails

        """
        self._validate_initialized()
        return await self._documents.scroll_points(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            filter_condition=filter_condition,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

    async def clear_collection(self, collection_name: str) -> bool:
        """Clear all points from a collection without deleting the collection.

        Args:
            collection_name: Name of the collection to clear

        Returns:
            bool: True if collection cleared successfully

        Raises:
            QdrantServiceError: If clearing fails

        """
        self._validate_initialized()
        return await self._documents.clear_collection(collection_name)

    # HNSW Optimization API (delegates to QdrantCollections for now)

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

        Args:
            collection_name: Name of the collection to create
            vector_size: Dimension of the vectors to be stored
            collection_type: Type for optimized HNSW settings ("general", "code", "scientific")
            distance: Distance metric for similarity search (Cosine, Euclid, Dot)
            sparse_vector_name: Optional name for sparse vector field
            enable_quantization: Whether to enable vector quantization

        Returns:
            bool: True if collection created successfully

        Raises:
            QdrantServiceError: If collection creation fails

        """
        self._validate_initialized()
        return await self._collections.create_collection(
            collection_name=collection_name,
            vector_size=vector_size,
            distance=distance,
            sparse_vector_name=sparse_vector_name,
            enable_quantization=enable_quantization,
            collection_type=collection_type,
        )

    def get_hnsw_configuration_info(self, collection_type: str) -> dict[str, object]:
        """Get HNSW configuration information for a collection type.

        Args:
            collection_type: Type of collection ("general", "code", "scientific")

        Returns:
            dict[str, object]: HNSW configuration including:
                - m: Number of bi-directional links
                - ef_construct: Size of dynamic list for construction
                - full_scan_threshold: Threshold for full scan vs index

        Raises:
            QdrantServiceError: If getting configuration fails

        """
        self._validate_initialized()
        return self._collections.get_hnsw_configuration_info(collection_type)

    def _validate_initialized(self) -> None:
        """Validate that the service is properly initialized.

        Raises:
            QdrantServiceError: If service is not initialized

        """
        if not self._initialized or not self._collections:
            msg = "Service not initialized. Call initialize() first."
            raise QdrantServiceError(msg)
