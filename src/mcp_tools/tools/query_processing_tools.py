"""Advanced query processing tools for MCP server.

This module exposes advanced query processing capabilities through the Model Context Protocol,
including query expansion, clustering, federated search, personalization, and orchestration.
"""

import logging
from typing import TYPE_CHECKING
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
from ...services.query_processing import ProcessingStage
from ...services.query_processing import SearchMode
from ...services.query_processing import SearchOrchestrator
from ...services.query_processing import SearchPipeline
from ...services.query_processing import SearchRequest
from ..models.responses import SearchResult

logger = logging.getLogger(__name__)


# Request models for MCP tools
class QueryExpansionRequest(BaseModel):
    """Request for query expansion."""

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Search query to expand")
    expansion_methods: list[str] = Field(
        ["synonyms", "related_terms"], description="Expansion methods to use"
    )
    expansion_depth: int = Field(2, ge=1, le=5, description="Depth of expansion")
    context_window: int = Field(100, ge=50, le=500, description="Context window size")
    limit: int = Field(10, ge=1, le=100, description="Number of results")


class ClusteredSearchRequest(BaseModel):
    """Request for clustered search."""

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Search query")
    num_clusters: int = Field(
        5, ge=2, le=20, description="Number of clusters to create"
    )
    clustering_algorithm: str = Field(
        "kmeans", description="Clustering algorithm (kmeans, hierarchical, dbscan)"
    )
    min_cluster_size: int = Field(2, ge=1, description="Minimum cluster size")
    limit: int = Field(10, ge=1, le=100, description="Number of results")


class FederatedSearchRequest(BaseModel):
    """Request for federated search."""

    collections: list[str] = Field(..., description="Collections to search across")
    query: str = Field(..., description="Search query")
    federation_strategy: str = Field(
        "parallel", description="Federation strategy (parallel, sequential, adaptive)"
    )
    result_merging: str = Field(
        "score", description="Result merging strategy (score, relevance, round_robin)"
    )
    per_collection_limit: int = Field(
        5, ge=1, le=50, description="Results per collection"
    )
    total_limit: int = Field(10, ge=1, le=100, description="Total results")


class PersonalizedSearchRequest(BaseModel):
    """Request for personalized search."""

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Search query")
    user_id: str = Field(..., description="User ID for personalization")
    personalization_strength: float = Field(
        0.5, ge=0.0, le=1.0, description="Personalization strength"
    )
    use_history: bool = Field(True, description="Use search history")
    use_preferences: bool = Field(True, description="Use user preferences")
    limit: int = Field(10, ge=1, le=100, description="Number of results")


class OrchestrationRequest(BaseModel):
    """Request for orchestrated search."""

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Search query")
    stages: list[str] = Field(
        None, description="Stages to include (if None, automatic selection)"
    )
    time_budget_ms: float = Field(
        3000.0, ge=100.0, description="Time budget in milliseconds"
    )
    quality_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Quality threshold"
    )
    limit: int = Field(10, ge=1, le=100, description="Number of results")


# Helper function to create the orchestrator
def create_orchestrator() -> SearchOrchestrator:
    """Create and configure the search orchestrator."""
    return SearchOrchestrator(enable_performance_optimization=True)


# Individual tool implementations
async def query_expansion_tool(
    request: QueryExpansionRequest,
    ctx: Context,
    orchestrator: SearchOrchestrator,
) -> list[SearchResult]:
    """Implementation for query expansion."""
    request_id = str(uuid4())
    await ctx.info(f"Starting query expansion search {request_id}")

    try:
        # Build expansion configuration
        expansion_config = {
            "methods": request.expansion_methods,
            "depth": request.expansion_depth,
            "context_window": request.context_window,
        }

        # Create search request
        search_request = SearchRequest(
            query=request.query,
            collection_name=request.collection_name,
            limit=request.limit,
            mode=SearchMode.ENHANCED,
            pipeline=SearchPipeline.BALANCED,
            enable_expansion=True,
            enable_caching=True,
        )

        # Execute search
        result = await orchestrator.search(search_request)

        # Convert results
        converted_results = []
        for r in result.results:
            search_result = SearchResult(
                id=r.get("id", ""),
                content=r.get("content", ""),
                metadata=r.get("metadata", {}),
                score=r.get("score", 0.0),
                source=r.get("metadata", {}).get("source", "unknown"),
                relevance_explanation=f"Query expanded: {result.expanded_query}"
                if result.expanded_query
                else "Query processed",
            )
            converted_results.append(search_result)

        await ctx.info(
            f"Query expansion search completed: {len(converted_results)} results "
            f"(expanded query: {result.query_processed})"
        )

        return converted_results

    except Exception as e:
        await ctx.error(f"Query expansion search failed: {e!s}")
        logger.error(f"Query expansion error: {e}", exc_info=True)
        raise


async def clustered_search_tool(
    request: ClusteredSearchRequest,
    ctx: Context,
    orchestrator: SearchOrchestrator,
) -> list[SearchResult]:
    """Implementation for clustered search."""
    request_id = str(uuid4())
    await ctx.info(f"Starting clustered search {request_id}")

    try:
        # Build clustering configuration
        clustering_config = {
            "num_clusters": request.num_clusters,
            "algorithm": request.clustering_algorithm,
            "min_cluster_size": request.min_cluster_size,
        }

        # Create search request
        search_request = SearchRequest(
            query=request.query,
            collection_name=request.collection_name,
            limit=request.limit * 2,  # Get more results for clustering
            mode=SearchMode.ENHANCED,
            pipeline=SearchPipeline.BALANCED,
            enable_clustering=True,
            enable_caching=True,
        )

        # Execute search
        result = await orchestrator.search(search_request)

        # Convert results
        converted_results = []
        for r in result.results[: request.limit]:
            cluster_info = r.get("cluster_id", "unclustered")
            search_result = SearchResult(
                id=r.get("id", ""),
                content=r.get("content", ""),
                metadata=r.get("metadata", {}),
                score=r.get("score", 0.0),
                source=r.get("metadata", {}).get("source", "unknown"),
                relevance_explanation=f"Cluster: {cluster_info}",
            )
            converted_results.append(search_result)

        await ctx.info(
            f"Clustered search completed: {len(converted_results)} results "
            f"in {request.num_clusters} clusters"
        )

        return converted_results

    except Exception as e:
        await ctx.error(f"Clustered search failed: {e!s}")
        logger.error(f"Clustered search error: {e}", exc_info=True)
        raise


async def federated_search_tool(
    request: FederatedSearchRequest,
    ctx: Context,
    orchestrator: SearchOrchestrator,
) -> list[SearchResult]:
    """Implementation for federated search."""
    request_id = str(uuid4())
    await ctx.info(f"Starting federated search {request_id}")

    try:
        # Build federation configuration
        federation_config = {
            "collections": request.collections,
            "strategy": request.federation_strategy,
            "merging": request.result_merging,
            "per_collection_limit": request.per_collection_limit,
        }

        # Create search request
        search_request = SearchRequest(
            query=request.query,
            collection_name=request.collections[0],  # Primary collection
            limit=request.total_limit,
            mode=SearchMode.ENHANCED,
            pipeline=SearchPipeline.BALANCED,
            enable_federation=True,
            enable_caching=True,
        )

        # Execute search
        result = await orchestrator.search(search_request)

        # Convert results
        converted_results = []
        for r in result.results:
            collection = r.get("metadata", {}).get("collection", "unknown")
            search_result = SearchResult(
                id=r.get("id", ""),
                content=r.get("content", ""),
                metadata=r.get("metadata", {}),
                score=r.get("score", 0.0),
                source=r.get("metadata", {}).get("source", "unknown"),
                relevance_explanation=f"From: {collection}",
            )
            converted_results.append(search_result)

        await ctx.info(
            f"Federated search completed: {len(converted_results)} results "
            f"from {len(request.collections)} collections"
        )

        return converted_results

    except Exception as e:
        await ctx.error(f"Federated search failed: {e!s}")
        logger.error(f"Federated search error: {e}", exc_info=True)
        raise


async def personalized_search_tool(
    request: PersonalizedSearchRequest,
    ctx: Context,
    orchestrator: SearchOrchestrator,
) -> list[SearchResult]:
    """Implementation for personalized search."""
    request_id = str(uuid4())
    await ctx.info(f"Starting personalized search {request_id}")

    try:
        # Build personalization configuration
        personalization_config = {
            "strength": request.personalization_strength,
            "use_history": request.use_history,
            "use_preferences": request.use_preferences,
        }

        # Create search request
        search_request = SearchRequest(
            query=request.query,
            collection_name=request.collection_name,
            limit=request.limit,
            mode=SearchMode.ENHANCED,
            pipeline=SearchPipeline.BALANCED,
            user_id=request.user_id,
            enable_personalization=True,
            enable_caching=True,
        )

        # Execute search
        result = await orchestrator.search(search_request)

        # Convert results
        converted_results = []
        for r in result.results:
            personalization_score = r.get("personalized_score", 0.0)
            search_result = SearchResult(
                id=r.get("id", ""),
                content=r.get("content", ""),
                metadata=r.get("metadata", {}),
                score=r.get("score", 0.0),
                source=r.get("metadata", {}).get("source", "unknown"),
                relevance_explanation=f"Personalized (score: {personalization_score:.2f})",
            )
            converted_results.append(search_result)

        await ctx.info(
            f"Personalized search completed: {len(converted_results)} results "
            f"for user {request.user_id}"
        )

        return converted_results

    except Exception as e:
        await ctx.error(f"Personalized search failed: {e!s}")
        logger.error(f"Personalized search error: {e}", exc_info=True)
        raise


async def orchestrated_search_tool(
    request: OrchestrationRequest,
    ctx: Context,
    orchestrator: SearchOrchestrator,
) -> list[SearchResult]:
    """Implementation for orchestrated search."""
    request_id = str(uuid4())
    await ctx.info(f"Starting orchestrated search {request_id}")

    try:
        # Build orchestration configuration
        orchestration_config = {
            "time_budget_ms": request.time_budget_ms,
            "quality_threshold": request.quality_threshold,
        }

        # Determine stages to skip
        skip_stages = []
        if request.stages:
            all_stages = [s.value for s in ProcessingStage]
            skip_stages = [s for s in all_stages if s not in request.stages]

        # Create search request
        search_request = SearchRequest(
            query=request.query,
            collection_name=request.collection_name,
            limit=request.limit,
            mode=SearchMode.ENHANCED,
            pipeline=SearchPipeline.BALANCED,
            enable_caching=True,
            max_processing_time_ms=request.time_budget_ms,
        )

        # Execute search
        result = await orchestrator.search(search_request)

        # Convert results
        converted_results = []
        for r in result.results:
            pipeline_info = f"Features: {', '.join(result.features_used) if result.features_used else 'basic'}"
            search_result = SearchResult(
                id=r.get("id", ""),
                content=r.get("content", ""),
                metadata=r.get("metadata", {}),
                score=r.get("score", 0.0),
                source=r.get("metadata", {}).get("source", "unknown"),
                relevance_explanation=pipeline_info,
            )
            converted_results.append(search_result)

        await ctx.info(
            f"Orchestrated search completed: {len(converted_results)} results\n"
            f"Processing time: {result.processing_time_ms:.1f}ms"
        )

        return converted_results

    except Exception as e:
        await ctx.error(f"Orchestrated search failed: {e!s}")
        logger.error(f"Orchestrated search error: {e}", exc_info=True)
        raise


def register_query_processing_tools(mcp, client_manager: ClientManager):
    """Register advanced query processing tools with the MCP server."""

    # Initialize the orchestrator
    orchestrator = create_orchestrator()

    @mcp.tool()
    async def search_with_query_expansion(
        request: QueryExpansionRequest, ctx: Context
    ) -> list[SearchResult]:
        """
        Search with automatic query expansion.

        Enhance queries by adding synonyms, related terms, and contextual
        expansions to improve recall and find more relevant results.
        """
        return await query_expansion_tool(request, ctx, orchestrator)

    @mcp.tool()
    async def search_with_clustering(
        request: ClusteredSearchRequest, ctx: Context
    ) -> list[SearchResult]:
        """
        Search with result clustering.

        Group search results into meaningful clusters based on content
        similarity, topics, or other features for better organization.
        """
        return await clustered_search_tool(request, ctx, orchestrator)

    @mcp.tool()
    async def search_federated(
        request: FederatedSearchRequest, ctx: Context
    ) -> list[SearchResult]:
        """
        Search across multiple collections simultaneously.

        Execute federated search across different data sources with
        intelligent result merging and ranking strategies.
        """
        return await federated_search_tool(request, ctx, orchestrator)

    @mcp.tool()
    async def search_personalized(
        request: PersonalizedSearchRequest, ctx: Context
    ) -> list[SearchResult]:
        """
        Search with user personalization.

        Tailor search results based on user history, preferences, and
        behavior patterns for more relevant results.
        """
        return await personalized_search_tool(request, ctx, orchestrator)

    @mcp.tool()
    async def search_orchestrated(
        request: OrchestrationRequest, ctx: Context
    ) -> list[SearchResult]:
        """
        Search with intelligent orchestration.

        Automatically select and coordinate multiple processing stages
        based on query characteristics and time/quality constraints.
        """
        return await orchestrated_search_tool(request, ctx, orchestrator)

    logger.info("Advanced query processing tools registered successfully")
