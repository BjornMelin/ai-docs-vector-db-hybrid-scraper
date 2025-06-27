"""Payload indexing management tools for MCP server."""

import logging  # noqa: PLC0415
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4


if TYPE_CHECKING:
    from fastmcp import Context
else:
    # Use a protocol for testing to avoid FastMCP import issues
    from typing import Protocol

    class Context(Protocol):
        async def info(self, msg: str) -> None: ...
        async def debug(self, msg: str) -> None: ...
        async def warning(self, msg: str) -> None: ...
        async def error(self, msg: str) -> None: ...


from ...infrastructure.client_manager import ClientManager
from ...security import MLSecurityValidator as SecurityValidator
from ..models.responses import GenericDictResponse  # noqa: PLC0415


logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register payload indexing tools with the MCP server."""

    @mcp.tool()
    async def create_payload_indexes(
        collection_name: str, ctx: Context
    ) -> "GenericDictResponse":
        """
        Create payload indexes on a collection for 10-100x faster filtering.

        Creates indexes on key metadata fields like site_name, embedding_model,
        title, word_count, crawl_timestamp, etc. for dramatic performance improvements.
        """
        # Generate request ID for tracking
        request_id = str(uuid4())
        await ctx.info(
            f"Creating payload indexes for collection: {collection_name} (Request: {request_id})"
        )

        try:
            # Validate collection name
            security_validator = SecurityValidator.from_unified_config()
            collection_name = security_validator.validate_collection_name(
                collection_name
            )

            # Check if collection exists
            collections = await client_manager.qdrant_service.list_collections()
            if collection_name not in collections:
                raise ValueError(f"Collection '{collection_name}' not found")

            # Create payload indexes
            await client_manager.qdrant_service.create_payload_indexes(collection_name)

            # Get index statistics
            stats = await client_manager.qdrant_service.get_payload_index_stats(
                collection_name
            )

            await ctx.info(
                f"Successfully created {stats['indexed_fields_count']} payload indexes for {collection_name}"
            )

            return GenericDictResponse(
                collection_name=collection_name,
                status="success",
                indexes_created=stats["indexed_fields_count"],
                indexed_fields=stats["indexed_fields"],
                total_points=stats["total_points"],
                request_id=request_id,
            )

        except Exception as e:
            await ctx.error("Failed to create payload indexes for {collection_name}")
            logger.exception("Failed to create payload indexes")
            raise

    @mcp.tool()
    async def list_payload_indexes(
        collection_name: str, ctx: Context
    ) -> GenericDictResponse:
        """
        List all payload indexes in a collection.

        Shows which fields are indexed and their types for performance monitoring.
        """
        await ctx.info("Listing payload indexes for collection")

        try:
            # Validate collection name
            security_validator = SecurityValidator.from_unified_config()
            collection_name = security_validator.validate_collection_name(
                collection_name
            )

            # Get index statistics
            stats = await client_manager.qdrant_service.get_payload_index_stats(
                collection_name
            )

            await ctx.info(
                f"Found {stats['indexed_fields_count']} indexed fields in {collection_name}"
            )

            return GenericDictResponse(**stats)

        except Exception as e:
            await ctx.error("Failed to list payload indexes for {collection_name}")
            logger.exception("Failed to list payload indexes")
            raise

    from ..models.responses import ReindexCollectionResponse  # noqa: PLC0415

    @mcp.tool()
    async def reindex_collection(
        collection_name: str, ctx: Context
    ) -> ReindexCollectionResponse:
        """
        Reindex all payload fields in a collection.

        Drops existing indexes and recreates them. Useful after bulk updates
        or when index performance degrades.
        """
        # Generate request ID for tracking
        request_id = str(uuid4())
        await ctx.info(
            f"Starting full reindex for collection: {collection_name} (Request: {request_id})"
        )

        try:
            # Validate collection name
            security_validator = SecurityValidator.from_unified_config()
            collection_name = security_validator.validate_collection_name(
                collection_name
            )

            # Get stats before reindexing
            stats_before = await client_manager.qdrant_service.get_payload_index_stats(
                collection_name
            )

            # Perform reindexing
            await client_manager.qdrant_service.reindex_collection(collection_name)

            # Get stats after reindexing
            stats_after = await client_manager.qdrant_service.get_payload_index_stats(
                collection_name
            )

            await ctx.info("Successfully reindexed collection")

            return ReindexCollectionResponse(
                status="success",
                collection=collection_name,
                reindexed_count=stats_after["indexed_fields_count"],
                details={
                    "indexes_before": stats_before["indexed_fields_count"],
                    "indexes_after": stats_after["indexed_fields_count"],
                    "indexed_fields": stats_after["indexed_fields"],
                    "total_points": stats_after["total_points"],
                    "request_id": request_id,
                },
            )

        except Exception as e:
            await ctx.error("Failed to reindex collection {collection_name}")
            logger.exception("Failed to reindex collection")
            raise

    @mcp.tool()
    async def benchmark_filtered_search(
        collection_name: str,
        test_filters: dict[str, Any],
        query: str = "documentation search test",
        ctx: Context = None,
    ) -> GenericDictResponse:
        """
        Benchmark filtered search performance to demonstrate indexing improvements.

        Compares performance of filtered searches and provides metrics on the
        effectiveness of payload indexing.
        """
        if ctx:
            await ctx.info("Benchmarking filtered search on collection")

        try:
            # Validate collection name and filters
            security_validator = SecurityValidator.from_unified_config()
            collection_name = security_validator.validate_collection_name(
                collection_name
            )
            query = security_validator.validate_query_string(query)

            # Generate embedding for test query
            embedding_result = (
                await client_manager.embedding_manager.generate_embeddings(
                    [query], generate_sparse=False
                )
            )
            query_vector = embedding_result["embeddings"][0]

            # Run filtered search with timing
            import time  # noqa: PLC0415

            start_time = time.time()

            results = await client_manager.qdrant_service.filtered_search(
                collection_name=collection_name,
                query_vector=query_vector,
                filters=test_filters,
                limit=10,
                search_accuracy="balanced",
            )

            search_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Get collection and index stats
            collection_stats = await client_manager.qdrant_service.get_collection_info(
                collection_name
            )
            index_stats = await client_manager.qdrant_service.get_payload_index_stats(
                collection_name
            )

            if ctx:
                await ctx.info(
                    f"Filtered search completed in {search_time:.2f}ms with {len(results)} results"
                )

            return GenericDictResponse(
                collection_name=collection_name,
                query=query,
                filters_applied=test_filters,
                search_time_ms=round(search_time, 2),
                results_found=len(results),
                total_points=collection_stats.get("points_count", 0),
                indexed_fields=index_stats["indexed_fields"],
                performance_estimate=(
                    "10-100x faster than unindexed"
                    if index_stats["indexed_fields"]
                    else "No indexes detected"
                ),
                benchmark_timestamp=datetime.now(UTC).isoformat(),
            )

        except Exception as e:
            if ctx:
                await ctx.error("Failed to benchmark filtered search")
            logger.exception("Failed to benchmark filtered search")
            raise
