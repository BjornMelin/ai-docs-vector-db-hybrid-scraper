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
    time_window: str | None = Field(None, description="Relative time window (e.g., '7d', '1M')")
    freshness_weight: float = Field(0.1, ge=0.0, le=1.0, description="Weight for content freshness")
    enable_decay: bool = Field(True, description="Enable time decay scoring")
    limit: int = Field(10, ge=1, le=100, description="Number of results")


class ContentTypeFilterRequest(BaseModel):
    """Request for content type filtering."""

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Search query")
    allowed_types: list[str] = Field(default_factory=list, description="Allowed content types")
    excluded_types: list[str] = Field(default_factory=list, description="Excluded content types")
    semantic_classification: bool = Field(False, description="Enable semantic content classification")
    confidence_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Classification confidence threshold")
    limit: int = Field(10, ge=1, le=100, description="Number of results")


class MetadataFilterRequest(BaseModel):
    """Request for metadata filtering."""

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Search query")
    filters: dict[str, Any] = Field(..., description="Key-value metadata filters")
    boolean_operator: str = Field("AND", description="Boolean operator (AND, OR)")
    enable_fuzzy_match: bool = Field(False, description="Enable fuzzy matching")
    fuzzy_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Fuzzy match threshold")
    limit: int = Field(10, ge=1, le=100, description="Number of results")


class CompositeFilterRequest(BaseModel):
    """Request for composite filtering with boolean logic."""

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Search query")
    filters: list[dict[str, Any]] = Field(..., description="List of filter conditions")
    composition_logic: dict[str, Any] = Field(..., description="Boolean composition logic")
    enable_optimization: bool = Field(True, description="Enable filter optimization")
    limit: int = Field(10, ge=1, le=100, description="Number of results")


class AdaptiveThresholdRequest(BaseModel):
    """Request for adaptive similarity threshold search."""

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Search query")
    base_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Base similarity threshold")
    adaptive_mode: str = Field("dynamic", description="Adaptive mode (static, dynamic, auto)")
    quality_target: float = Field(0.8, ge=0.0, le=1.0, description="Target quality score")
    min_results: int = Field(5, ge=1, description="Minimum results to return")
    max_threshold_reduction: float = Field(0.3, ge=0.0, le=0.5, description="Maximum threshold reduction")
    limit: int = Field(10, ge=1, le=100, description="Number of results")


def register_filtering_tools(mcp, client_manager: ClientManager):
    """Register advanced filtering tools with the MCP server."""

    # Initialize the orchestrator
    orchestrator = AdvancedSearchOrchestrator(
        enable_all_features=True,
        enable_performance_optimization=True
    )

    @mcp.tool()
    async def search_with_temporal_filter(
        request: TemporalFilterRequest,
        ctx: Context
    ) -> list[SearchResult]:
        """
        Search with temporal filtering for date-based content.
        
        Apply temporal filters to find content within specific date ranges,
        with support for relative time windows and content freshness scoring.
        """
        request_id = str(uuid4())
        await ctx.info(f"Starting temporal filtered search {request_id}")

        try:
            # Build temporal criteria
            temporal_criteria = {
                "start_date": request.start_date,
                "end_date": request.end_date,
                "time_window": request.time_window,
                "freshness_weight": request.freshness_weight,
                "enable_decay": request.enable_decay
            }

            # Create advanced search request
            search_request = AdvancedSearchRequest(
                query=request.query,
                collection_name=request.collection_name,
                limit=request.limit,
                search_mode=SearchMode.ENHANCED,
                pipeline=SearchPipeline.BALANCED,
                temporal_criteria=temporal_criteria,
                enable_expansion=True,
                enable_caching=True
            )

            # Execute search
            result = await orchestrator.search(search_request)

            # Convert to SearchResult format
            search_results = []
            for res in result.results:
                search_results.append(SearchResult(
                    id=res.get("id", str(uuid4())),
                    content=res.get("content", ""),
                    score=res.get("score", 0.0),
                    url=res.get("url"),
                    title=res.get("title"),
                    metadata={
                        **res.get("metadata", {}),
                        "temporal_relevance": res.get("temporal_relevance", 1.0),
                        "published_date": res.get("published_date")
                    }
                ))

            await ctx.info(
                f"Temporal filtered search {request_id} completed: "
                f"{len(search_results)} results found"
            )
            return search_results

        except Exception as e:
            await ctx.error(f"Temporal filtered search {request_id} failed: {e}")
            logger.error(f"Temporal filtered search failed: {e}")
            raise

    @mcp.tool()
    async def search_with_content_type_filter(
        request: ContentTypeFilterRequest,
        ctx: Context
    ) -> list[SearchResult]:
        """
        Search with content type filtering.
        
        Filter search results by content type, with support for semantic
        content classification and type-specific ranking.
        """
        request_id = str(uuid4())
        await ctx.info(f"Starting content type filtered search {request_id}")

        try:
            # Build content type criteria
            content_type_criteria = {
                "allowed_types": request.allowed_types,
                "excluded_types": request.excluded_types,
                "semantic_classification": request.semantic_classification,
                "confidence_threshold": request.confidence_threshold
            }

            # Create advanced search request
            search_request = AdvancedSearchRequest(
                query=request.query,
                collection_name=request.collection_name,
                limit=request.limit,
                search_mode=SearchMode.ENHANCED,
                pipeline=SearchPipeline.BALANCED,
                content_type_criteria=content_type_criteria,
                enable_expansion=True,
                enable_caching=True
            )

            # Execute search
            result = await orchestrator.search(search_request)

            # Convert to SearchResult format
            search_results = []
            for res in result.results:
                search_results.append(SearchResult(
                    id=res.get("id", str(uuid4())),
                    content=res.get("content", ""),
                    score=res.get("score", 0.0),
                    url=res.get("url"),
                    title=res.get("title"),
                    metadata={
                        **res.get("metadata", {}),
                        "content_type": res.get("content_type"),
                        "content_type_match": res.get("content_type_match", 1.0),
                        "classification_confidence": res.get("classification_confidence")
                    }
                ))

            await ctx.info(
                f"Content type filtered search {request_id} completed: "
                f"{len(search_results)} results found"
            )
            return search_results

        except Exception as e:
            await ctx.error(f"Content type filtered search {request_id} failed: {e}")
            logger.error(f"Content type filtered search failed: {e}")
            raise

    @mcp.tool()
    async def search_with_metadata_filter(
        request: MetadataFilterRequest,
        ctx: Context
    ) -> list[SearchResult]:
        """
        Search with custom metadata filtering.
        
        Apply flexible metadata filters with boolean logic, fuzzy matching,
        and field-specific boost values.
        """
        request_id = str(uuid4())
        await ctx.info(f"Starting metadata filtered search {request_id}")

        try:
            # Build metadata criteria
            metadata_criteria = {
                "filters": request.filters,
                "boolean_operator": request.boolean_operator,
                "enable_fuzzy_match": request.enable_fuzzy_match,
                "fuzzy_threshold": request.fuzzy_threshold
            }

            # Create advanced search request
            search_request = AdvancedSearchRequest(
                query=request.query,
                collection_name=request.collection_name,
                limit=request.limit,
                search_mode=SearchMode.ENHANCED,
                pipeline=SearchPipeline.BALANCED,
                metadata_criteria=metadata_criteria,
                enable_expansion=True,
                enable_caching=True
            )

            # Execute search
            result = await orchestrator.search(search_request)

            # Convert to SearchResult format
            search_results = []
            for res in result.results:
                search_results.append(SearchResult(
                    id=res.get("id", str(uuid4())),
                    content=res.get("content", ""),
                    score=res.get("score", 0.0),
                    url=res.get("url"),
                    title=res.get("title"),
                    metadata={
                        **res.get("metadata", {}),
                        "metadata_match_score": res.get("metadata_match_score", 1.0),
                        "matched_fields": res.get("matched_fields", [])
                    }
                ))

            await ctx.info(
                f"Metadata filtered search {request_id} completed: "
                f"{len(search_results)} results found"
            )
            return search_results

        except Exception as e:
            await ctx.error(f"Metadata filtered search {request_id} failed: {e}")
            logger.error(f"Metadata filtered search failed: {e}")
            raise

    @mcp.tool()
    async def search_with_composite_filters(
        request: CompositeFilterRequest,
        ctx: Context
    ) -> list[SearchResult]:
        """
        Search with composite filters using boolean logic.
        
        Combine multiple filter types with AND, OR, and NOT operators
        for complex filtering scenarios.
        """
        request_id = str(uuid4())
        await ctx.info(f"Starting composite filtered search {request_id}")

        try:
            # Create advanced search request with filter composition
            search_request = AdvancedSearchRequest(
                query=request.query,
                collection_name=request.collection_name,
                limit=request.limit,
                search_mode=SearchMode.COMPREHENSIVE,
                pipeline=SearchPipeline.COMPREHENSIVE,
                filter_composition=request.composition_logic,
                enable_expansion=True,
                enable_clustering=True,
                enable_caching=True
            )

            # Process individual filters
            for filter_def in request.filters:
                filter_type = filter_def.get("type", "metadata")

                if filter_type == "temporal":
                    search_request.temporal_criteria = filter_def.get("criteria", {})
                elif filter_type == "content_type":
                    search_request.content_type_criteria = filter_def.get("criteria", {})
                elif filter_type == "metadata":
                    search_request.metadata_criteria = filter_def.get("criteria", {})
                elif filter_type == "similarity":
                    search_request.similarity_threshold_criteria = filter_def.get("criteria", {})

            # Execute search
            result = await orchestrator.search(search_request)

            # Convert to SearchResult format
            search_results = []
            for res in result.results:
                search_results.append(SearchResult(
                    id=res.get("id", str(uuid4())),
                    content=res.get("content", ""),
                    score=res.get("score", 0.0),
                    url=res.get("url"),
                    title=res.get("title"),
                    metadata={
                        **res.get("metadata", {}),
                        "filter_scores": res.get("filter_scores", {}),
                        "composite_score": res.get("composite_score", 1.0)
                    }
                ))

            await ctx.info(
                f"Composite filtered search {request_id} completed: "
                f"{len(search_results)} results found with "
                f"{len(request.filters)} filters applied"
            )
            return search_results

        except Exception as e:
            await ctx.error(f"Composite filtered search {request_id} failed: {e}")
            logger.error(f"Composite filtered search failed: {e}")
            raise

    @mcp.tool()
    async def search_with_adaptive_threshold(
        request: AdaptiveThresholdRequest,
        ctx: Context
    ) -> list[SearchResult]:
        """
        Search with adaptive similarity threshold management.
        
        Dynamically adjust similarity thresholds based on result quality
        and quantity to ensure optimal retrieval performance.
        """
        request_id = str(uuid4())
        await ctx.info(f"Starting adaptive threshold search {request_id}")

        try:
            # Build similarity threshold criteria
            similarity_criteria = {
                "base_threshold": request.base_threshold,
                "adaptive_mode": request.adaptive_mode,
                "quality_target": request.quality_target,
                "min_results": request.min_results,
                "max_threshold_reduction": request.max_threshold_reduction,
                "enable_feedback_learning": True
            }

            # Create advanced search request
            search_request = AdvancedSearchRequest(
                query=request.query,
                collection_name=request.collection_name,
                limit=request.limit,
                search_mode=SearchMode.INTELLIGENT,
                pipeline=SearchPipeline.PRECISION,
                similarity_threshold_criteria=similarity_criteria,
                enable_expansion=True,
                enable_personalization=True,
                enable_caching=True,
                quality_threshold=request.quality_target
            )

            # Execute search
            result = await orchestrator.search(search_request)

            # Convert to SearchResult format
            search_results = []
            for res in result.results:
                search_results.append(SearchResult(
                    id=res.get("id", str(uuid4())),
                    content=res.get("content", ""),
                    score=res.get("score", 0.0),
                    url=res.get("url"),
                    title=res.get("title"),
                    metadata={
                        **res.get("metadata", {}),
                        "similarity_score": res.get("score", 0.0),
                        "adjusted_threshold": res.get("adjusted_threshold"),
                        "quality_score": result.quality_score
                    }
                ))

            await ctx.info(
                f"Adaptive threshold search {request_id} completed: "
                f"{len(search_results)} results found with quality score "
                f"{result.quality_score:.2f}"
            )
            return search_results

        except Exception as e:
            await ctx.error(f"Adaptive threshold search {request_id} failed: {e}")
            logger.error(f"Adaptive threshold search failed: {e}")
            raise

