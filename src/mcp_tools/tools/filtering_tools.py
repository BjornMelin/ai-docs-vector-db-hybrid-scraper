import typing

"""Advanced filtering tools for MCP server.

This module exposes advanced filtering capabilities through the Model Context Protocol,
including temporal filtering, content type filtering, metadata filtering, and filter
composition with boolean logic.
"""

import logging
from typing import TYPE_CHECKING
from typing import Any
from uuid import uuid4

if TYPE_CHECKING:
    from fastmcp import Context
else:
    from typing import Protocol

    class Context(Protocol):
        async def info(self, msg: str) -> None: ...
        async def debug(self, msg: str) -> None: ...
        async def warning(self, msg: str) -> None: ...
        async def error(self, msg: str) -> None: ...


from pydantic import BaseModel
from pydantic import Field

from ...infrastructure.client_manager import ClientManager
from ...services.query_processing import AdvancedSearchOrchestrator
from ...services.query_processing import AdvancedSearchRequest
from ...services.query_processing import SearchMode
from ...services.query_processing import SearchPipeline
from ..models.responses import SearchResult

logger = logging.getLogger(__name__)


# Request models for MCP tools
class TemporalFilterRequest(BaseModel):
    """Request for temporal filtering."""

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Search query")
    start_date: str | None = Field(None, description="Start date (ISO format)")
    end_date: str | None = Field(None, description="End date (ISO format)")
    time_window: str | None = Field(
        None, description="Relative time window (e.g., '7d', '1M')"
    )
    freshness_weight: float = Field(
        0.1, ge=0.0, le=1.0, description="Weight for content freshness"
    )
    limit: int = Field(10, ge=1, le=100, description="Number of results")


class ContentTypeFilterRequest(BaseModel):
    """Request for content type filtering."""

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Search query")
    allowed_types: list[str] = Field(
        ..., description="Allowed content types (e.g., 'documentation', 'code')"
    )
    exclude_types: list[str] = Field(
        default_factory=list, description="Content types to exclude"
    )
    priority_types: list[str] = Field(
        default_factory=list, description="Priority content types"
    )
    limit: int = Field(10, ge=1, le=100, description="Number of results")


class MetadataFilterRequest(BaseModel):
    """Request for metadata filtering."""

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Search query")
    metadata_filters: dict[str, Any] = Field(
        ..., description="Key-value pairs for metadata filtering"
    )
    filter_operator: str = Field("AND", description="Filter operator: 'AND' or 'OR'")
    exact_match: bool = Field(True, description="Use exact matching")
    case_sensitive: bool = Field(False, description="Case-sensitive matching")
    limit: int = Field(10, ge=1, le=100, description="Number of results")


class SimilarityFilterRequest(BaseModel):
    """Request for similarity-based filtering."""

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Search query")
    min_similarity: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum similarity score"
    )
    max_similarity: float = Field(
        1.0, ge=0.0, le=1.0, description="Maximum similarity score"
    )
    similarity_metric: str = Field("cosine", description="Similarity metric")
    adaptive_threshold: bool = Field(False, description="Use adaptive threshold")
    boost_recent: bool = Field(False, description="Boost recent content")
    limit: int = Field(10, ge=1, le=100, description="Number of results")


class CompositeFilterRequest(BaseModel):
    """Request for composite filtering with boolean logic."""

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Search query")
    temporal_config: dict[str, Any] | None = Field(
        None, description="Temporal filtering configuration"
    )
    content_type_config: dict[str, Any] | None = Field(
        None, description="Content type filtering configuration"
    )
    metadata_config: dict[str, Any] | None = Field(
        None, description="Metadata filtering configuration"
    )
    similarity_config: dict[str, Any] | None = Field(
        None, description="Similarity filtering configuration"
    )
    operator: str = Field("AND", description="Logical operator: 'AND', 'OR'")
    nested_logic: bool = Field(False, description="Enable nested boolean expressions")
    optimize_order: bool = Field(True, description="Optimize filter execution order")
    limit: int = Field(10, ge=1, le=100, description="Number of results")


# Helper function to create the orchestrator
def create_orchestrator() -> AdvancedSearchOrchestrator:
    """Create and configure the search orchestrator."""
    return AdvancedSearchOrchestrator(
        cache_size=1000, enable_performance_optimization=True
    )


# Individual tool implementations
async def temporal_filter_tool(
    request: TemporalFilterRequest,
    ctx: Context,
    orchestrator: AdvancedSearchOrchestrator,
) -> list[SearchResult]:
    """Implementation for temporal filtering."""
    request_id = str(uuid4())
    await ctx.info(f"Starting temporal filtered search {request_id}")

    try:
        # Build temporal criteria
        temporal_criteria = {
            "start_date": request.start_date,
            "end_date": request.end_date,
            "time_window": request.time_window,
            "freshness_weight": request.freshness_weight,
        }

        # Create advanced search request
        search_request = AdvancedSearchRequest(
            query=request.query,
            collection_name=request.collection_name,
            limit=request.limit,
            search_mode=SearchMode.ENHANCED,
            pipeline=SearchPipeline.BALANCED,
            temporal_criteria=temporal_criteria,
            enable_caching=True,
        )

        # Execute search
        result = await orchestrator.search(search_request)

        # Convert results
        converted_results = []
        for r in result.results:
            search_result = SearchResult(
                id=r.get("id", "unknown"),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                title=r.get("title"),
                metadata=r.get("metadata", {}),
            )
            converted_results.append(search_result)

        await ctx.info(
            f"Temporal search completed: {len(converted_results)} results "
            f"(processing time: {result.processing_time_ms:.1f}ms)"
        )

        return converted_results

    except Exception as e:
        await ctx.error(f"Temporal filter search failed: {e!s}")
        logger.error(f"Temporal filter error: {e}", exc_info=True)
        raise


async def content_type_filter_tool(
    request: ContentTypeFilterRequest,
    ctx: Context,
    orchestrator: AdvancedSearchOrchestrator,
) -> list[SearchResult]:
    """Implementation for content type filtering."""
    request_id = str(uuid4())
    await ctx.info(f"Starting content type filtered search {request_id}")

    try:
        # Build content type criteria
        type_criteria = {
            "allowed_types": request.allowed_types,
            "exclude_types": request.exclude_types,
            "priority_types": request.priority_types,
        }

        # Create advanced search request
        search_request = AdvancedSearchRequest(
            query=request.query,
            collection_name=request.collection_name,
            limit=request.limit,
            search_mode=SearchMode.ENHANCED,
            pipeline=SearchPipeline.BALANCED,
            content_type_criteria=type_criteria,
            enable_caching=True,
        )

        # Execute search
        result = await orchestrator.search(search_request)

        # Convert results
        converted_results = []
        for r in result.results:
            search_result = SearchResult(
                id=r.get("id", "unknown"),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                title=r.get("title"),
                metadata=r.get("metadata", {}),
            )
            converted_results.append(search_result)

        await ctx.info(
            f"Content type search completed: {len(converted_results)} results "
            f"(processing time: {result.processing_time_ms:.1f}ms)"
        )

        return converted_results

    except Exception as e:
        await ctx.error(f"Content type filter search failed: {e!s}")
        logger.error(f"Content type filter error: {e}", exc_info=True)
        raise


async def metadata_filter_tool(
    request: MetadataFilterRequest,
    ctx: Context,
    orchestrator: AdvancedSearchOrchestrator,
) -> list[SearchResult]:
    """Implementation for metadata filtering."""
    request_id = str(uuid4())
    await ctx.info(f"Starting metadata filtered search {request_id}")

    try:
        # Build metadata criteria
        metadata_criteria = {
            "filters": request.metadata_filters,
            "filter_operator": request.filter_operator,
            "exact_match": request.exact_match,
            "case_sensitive": request.case_sensitive,
        }

        # Create advanced search request
        search_request = AdvancedSearchRequest(
            query=request.query,
            collection_name=request.collection_name,
            limit=request.limit,
            search_mode=SearchMode.ENHANCED,
            pipeline=SearchPipeline.BALANCED,
            metadata_criteria=metadata_criteria,
            enable_caching=True,
        )

        # Execute search
        result = await orchestrator.search(search_request)

        # Convert results
        converted_results = []
        for r in result.results:
            search_result = SearchResult(
                id=r.get("id", "unknown"),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                title=r.get("title"),
                metadata=r.get("metadata", {}),
            )
            converted_results.append(search_result)

        await ctx.info(
            f"Metadata search completed: {len(converted_results)} results "
            f"(processing time: {result.processing_time_ms:.1f}ms)"
        )

        return converted_results

    except Exception as e:
        await ctx.error(f"Metadata filter search failed: {e!s}")
        logger.error(f"Metadata filter error: {e}", exc_info=True)
        raise


async def similarity_filter_tool(
    request: SimilarityFilterRequest,
    ctx: Context,
    orchestrator: AdvancedSearchOrchestrator,
) -> list[SearchResult]:
    """Implementation for similarity filtering."""
    request_id = str(uuid4())
    await ctx.info(f"Starting similarity filtered search {request_id}")

    try:
        # Build similarity criteria
        similarity_criteria = {
            "min_similarity": request.min_similarity,
            "max_similarity": request.max_similarity,
            "similarity_metric": request.similarity_metric,
            "adaptive_threshold": request.adaptive_threshold,
            "boost_recent": request.boost_recent,
        }

        # Create advanced search request
        search_request = AdvancedSearchRequest(
            query=request.query,
            collection_name=request.collection_name,
            limit=request.limit,
            search_mode=SearchMode.ENHANCED,
            pipeline=SearchPipeline.BALANCED,
            similarity_threshold_criteria=similarity_criteria,
            enable_caching=True,
        )

        # Execute search
        result = await orchestrator.search(search_request)

        # Convert results
        converted_results = []
        for r in result.results:
            search_result = SearchResult(
                id=r.get("id", "unknown"),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                title=r.get("title"),
                metadata=r.get("metadata", {}),
            )
            converted_results.append(search_result)

        await ctx.info(
            f"Similarity search completed: {len(converted_results)} results "
            f"(processing time: {result.processing_time_ms:.1f}ms)"
        )

        return converted_results

    except Exception as e:
        await ctx.error(f"Similarity filter search failed: {e!s}")
        logger.error(f"Similarity filter error: {e}", exc_info=True)
        raise


async def composite_filter_tool(
    request: CompositeFilterRequest,
    ctx: Context,
    orchestrator: AdvancedSearchOrchestrator,
) -> list[SearchResult]:
    """Implementation for composite filtering."""
    request_id = str(uuid4())
    await ctx.info(f"Starting composite filtered search {request_id}")

    try:
        # Build composite criteria from individual configs
        filters = {}
        if request.temporal_config:
            filters["temporal"] = request.temporal_config
        if request.content_type_config:
            filters["content_type"] = request.content_type_config
        if request.metadata_config:
            filters["metadata"] = request.metadata_config
        if request.similarity_config:
            filters["similarity"] = request.similarity_config

        composite_criteria = {
            "filters": filters,
            "operator": request.operator,
            "nested_logic": request.nested_logic,
            "optimize_order": request.optimize_order,
        }

        # Create advanced search request
        search_request = AdvancedSearchRequest(
            query=request.query,
            collection_name=request.collection_name,
            limit=request.limit,
            search_mode=SearchMode.ENHANCED,
            pipeline=SearchPipeline.BALANCED,
            filter_composition=composite_criteria,
            enable_caching=True,
        )

        # Execute search
        result = await orchestrator.search(search_request)

        # Convert results
        converted_results = []
        for r in result.results:
            search_result = SearchResult(
                id=r.get("id", "unknown"),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                title=r.get("title"),
                metadata=r.get("metadata", {}),
            )
            converted_results.append(search_result)

        await ctx.info(
            f"Composite search completed: {len(converted_results)} results "
            f"(processing time: {result.processing_time_ms:.1f}ms)"
        )

        return converted_results

    except Exception as e:
        await ctx.error(f"Composite filter search failed: {e!s}")
        logger.error(f"Composite filter error: {e}", exc_info=True)
        raise


def register_filtering_tools(mcp, client_manager: ClientManager):
    """Register advanced filtering tools with the MCP server."""

    # Initialize the orchestrator
    orchestrator = create_orchestrator()

    @mcp.tool()
    async def search_with_temporal_filter(
        request: TemporalFilterRequest, ctx: Context
    ) -> list[SearchResult]:
        """
        Search with temporal filtering for date-based content.

        Apply temporal filters to find content within specific date ranges,
        with support for relative time windows and content freshness scoring.
        """
        return await temporal_filter_tool(request, ctx, orchestrator)

    @mcp.tool()
    async def search_with_content_type_filter(
        request: ContentTypeFilterRequest, ctx: Context
    ) -> list[SearchResult]:
        """
        Search with content type filtering.

        Filter results by content type (documentation, code, tutorials, etc.)
        with configurable type classification confidence thresholds.
        """
        return await content_type_filter_tool(request, ctx, orchestrator)

    @mcp.tool()
    async def search_with_metadata_filter(
        request: MetadataFilterRequest, ctx: Context
    ) -> list[SearchResult]:
        """
        Search with metadata filtering.

        Apply metadata-based filters using key-value pairs with flexible
        matching modes (all/any) and partial matching support.
        """
        return await metadata_filter_tool(request, ctx, orchestrator)

    @mcp.tool()
    async def search_with_similarity_filter(
        request: SimilarityFilterRequest, ctx: Context
    ) -> list[SearchResult]:
        """
        Search with similarity-based filtering.

        Filter results based on semantic similarity thresholds with support
        for synonym expansion and fuzzy matching.
        """
        return await similarity_filter_tool(request, ctx, orchestrator)

    @mcp.tool()
    async def search_with_composite_filter(
        request: CompositeFilterRequest, ctx: Context
    ) -> list[SearchResult]:
        """
        Search with composite filtering using boolean logic.

        Combine multiple filters using AND/OR/NOT operators with support
        for nested expressions and optimized execution order.
        """
        return await composite_filter_tool(request, ctx, orchestrator)

    logger.info("Advanced filtering tools registered successfully")
