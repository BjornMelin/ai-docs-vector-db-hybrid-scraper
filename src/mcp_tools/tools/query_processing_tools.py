"""Advanced query processing tools for MCP server.

This module exposes advanced query processing capabilities through the Model Context Protocol,
including query expansion, result clustering, personalized ranking, and federated search.
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
class QueryExpansionRequest(BaseModel):
    """Request for query expansion search."""

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Original search query")
    expansion_strategy: str = Field("balanced", description="Expansion strategy (synonym, semantic, hybrid)")
    max_expansions: int = Field(5, ge=1, le=20, description="Maximum expansion terms")
    confidence_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Expansion confidence threshold")
    include_related: bool = Field(True, description="Include related concepts")
    limit: int = Field(10, ge=1, le=100, description="Number of results")


class ClusteredSearchRequest(BaseModel):
    """Request for search with result clustering."""

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Search query")
    clustering_method: str = Field("hdbscan", description="Clustering method (hdbscan, kmeans, hierarchical)")
    num_clusters: int | None = Field(None, description="Number of clusters (auto if None)")
    min_cluster_size: int = Field(5, ge=2, description="Minimum cluster size")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Cluster similarity threshold")
    limit: int = Field(50, ge=10, le=200, description="Number of results to cluster")


class PersonalizedSearchRequest(BaseModel):
    """Request for personalized search with user-based ranking."""

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Search query")
    user_id: str = Field(..., description="User identifier")
    session_id: str | None = Field(None, description="Session identifier")
    user_preferences: dict[str, Any] = Field(default_factory=dict, description="User preferences")
    ranking_strategy: str = Field("collaborative", description="Ranking strategy")
    personalization_weight: float = Field(0.3, ge=0.0, le=1.0, description="Personalization weight")
    limit: int = Field(10, ge=1, le=100, description="Number of results")


class FederatedSearchRequest(BaseModel):
    """Request for federated search across collections."""

    query: str = Field(..., description="Search query")
    collections: list[str] = Field(..., description="Collections to search")
    collection_weights: dict[str, float] | None = Field(None, description="Collection-specific weights")
    merge_strategy: str = Field("rrf", description="Result merging strategy (rrf, score, weighted)")
    enable_deduplication: bool = Field(True, description="Enable result deduplication")
    limit_per_collection: int = Field(20, ge=5, le=50, description="Results per collection")
    final_limit: int = Field(10, ge=1, le=100, description="Final number of results")


class PipelineSearchRequest(BaseModel):
    """Request for search with custom processing pipeline."""

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Search query")
    pipeline: str = Field("balanced", description="Pipeline configuration")
    stages: list[str] = Field(
        default_factory=lambda: ["expansion", "filtering", "ranking"],
        description="Processing stages to enable"
    )
    stage_configs: dict[str, Any] = Field(default_factory=dict, description="Stage-specific configurations")
    time_budget_ms: float = Field(3000.0, ge=100.0, description="Time budget in milliseconds")
    quality_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Quality threshold")
    limit: int = Field(10, ge=1, le=100, description="Number of results")


def register_query_processing_tools(mcp, client_manager: ClientManager):
    """Register advanced query processing tools with the MCP server."""

    # Initialize the orchestrator
    orchestrator = AdvancedSearchOrchestrator(
        enable_all_features=True,
        enable_performance_optimization=True
    )

    @mcp.tool()
    async def search_with_query_expansion(
        request: QueryExpansionRequest,
        ctx: Context
    ) -> list[SearchResult]:
        """
        Search with intelligent query expansion.
        
        Expand the original query with synonyms, related terms, and semantic
        variations to improve recall and find more relevant results.
        """
        request_id = str(uuid4())
        await ctx.info(f"Starting query expansion search {request_id}")

        try:
            # Configure expansion settings
            expansion_config = {
                "expansion_strategy": request.expansion_strategy,
                "max_expansions": request.max_expansions,
                "confidence_threshold": request.confidence_threshold,
                "include_related": request.include_related
            }

            # Create advanced search request
            search_request = AdvancedSearchRequest(
                query=request.query,
                collection_name=request.collection_name,
                limit=request.limit,
                search_mode=SearchMode.ENHANCED,
                pipeline=SearchPipeline.DISCOVERY,
                enable_expansion=True,
                expansion_config=expansion_config,
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
                        "expansion_terms": res.get("expansion_terms", []),
                        "original_query": request.query,
                        "expanded_query": result.query_processed
                    }
                ))

            await ctx.info(
                f"Query expansion search {request_id} completed: "
                f"{len(search_results)} results found with "
                f"{len(result.search_metadata.get('expanded_terms', []))} expansion terms"
            )
            return search_results

        except Exception as e:
            await ctx.error(f"Query expansion search {request_id} failed: {e}")
            logger.error(f"Query expansion search failed: {e}")
            raise

    @mcp.tool()
    async def search_with_clustering(
        request: ClusteredSearchRequest,
        ctx: Context
    ) -> dict[str, Any]:
        """
        Search with result clustering for topic discovery.
        
        Group search results into meaningful clusters to identify topics,
        themes, and related content groups within the results.
        """
        request_id = str(uuid4())
        await ctx.info(f"Starting clustered search {request_id}")

        try:
            # Configure clustering settings
            clustering_config = {
                "clustering_method": request.clustering_method,
                "num_clusters": request.num_clusters,
                "min_cluster_size": request.min_cluster_size,
                "similarity_threshold": request.similarity_threshold
            }

            # Create advanced search request
            search_request = AdvancedSearchRequest(
                query=request.query,
                collection_name=request.collection_name,
                limit=request.limit,
                search_mode=SearchMode.INTELLIGENT,
                pipeline=SearchPipeline.DISCOVERY,
                enable_expansion=True,
                enable_clustering=True,
                clustering_config=clustering_config,
                enable_caching=True
            )

            # Execute search
            result = await orchestrator.search(search_request)

            # Organize results by cluster
            clusters = {}
            unclustered_results = []

            for res in result.results:
                cluster_id = res.get("cluster_id")
                if cluster_id:
                    if cluster_id not in clusters:
                        clusters[cluster_id] = {
                            "cluster_id": cluster_id,
                            "cluster_label": res.get("cluster_label", f"Cluster {cluster_id}"),
                            "cluster_score": res.get("cluster_score", 0.0),
                            "results": []
                        }

                    clusters[cluster_id]["results"].append(SearchResult(
                        id=res.get("id", str(uuid4())),
                        content=res.get("content", ""),
                        score=res.get("score", 0.0),
                        url=res.get("url"),
                        title=res.get("title"),
                        metadata=res.get("metadata", {})
                    ))
                else:
                    unclustered_results.append(SearchResult(
                        id=res.get("id", str(uuid4())),
                        content=res.get("content", ""),
                        score=res.get("score", 0.0),
                        url=res.get("url"),
                        title=res.get("title"),
                        metadata=res.get("metadata", {})
                    ))

            response = {
                "request_id": request_id,
                "total_results": len(result.results),
                "clusters_found": result.clusters_found,
                "clusters": list(clusters.values()),
                "unclustered_results": unclustered_results,
                "clustering_metadata": {
                    "method": request.clustering_method,
                    "cluster_distribution": result.cluster_distribution,
                    "quality_score": result.quality_score
                }
            }

            await ctx.info(
                f"Clustered search {request_id} completed: "
                f"{len(result.results)} results in {result.clusters_found} clusters"
            )
            return response

        except Exception as e:
            await ctx.error(f"Clustered search {request_id} failed: {e}")
            logger.error(f"Clustered search failed: {e}")
            raise

    @mcp.tool()
    async def search_with_personalization(
        request: PersonalizedSearchRequest,
        ctx: Context
    ) -> list[SearchResult]:
        """
        Search with personalized ranking based on user preferences.
        
        Tailor search results to individual users based on their preferences,
        interaction history, and collaborative filtering signals.
        """
        request_id = str(uuid4())
        await ctx.info(f"Starting personalized search {request_id} for user {request.user_id}")

        try:
            # Configure ranking settings
            ranking_config = {
                "ranking_strategy": request.ranking_strategy,
                "personalization_weight": request.personalization_weight,
                "user_preferences": request.user_preferences
            }

            # Create advanced search request
            search_request = AdvancedSearchRequest(
                query=request.query,
                collection_name=request.collection_name,
                limit=request.limit,
                search_mode=SearchMode.PERSONALIZED,
                pipeline=SearchPipeline.PERSONALIZED,
                enable_expansion=True,
                enable_personalization=True,
                ranking_config=ranking_config,
                user_id=request.user_id,
                session_id=request.session_id,
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
                        "personalization_boost": res.get("personalization_boost", 0.0),
                        "ranking_factors": res.get("ranking_factors", {}),
                        "user_relevance": res.get("user_relevance", 1.0)
                    }
                ))

            await ctx.info(
                f"Personalized search {request_id} completed: "
                f"{len(search_results)} results with average personalization "
                f"boost {sum(r.metadata.get('personalization_boost', 0) for r in search_results) / len(search_results):.2f}"
            )
            return search_results

        except Exception as e:
            await ctx.error(f"Personalized search {request_id} failed: {e}")
            logger.error(f"Personalized search failed: {e}")
            raise

    @mcp.tool()
    async def federated_search(
        request: FederatedSearchRequest,
        ctx: Context
    ) -> dict[str, Any]:
        """
        Search across multiple collections simultaneously.
        
        Perform federated search to retrieve and merge results from multiple
        collections, with intelligent deduplication and ranking.
        """
        request_id = str(uuid4())
        await ctx.info(
            f"Starting federated search {request_id} across "
            f"{len(request.collections)} collections"
        )

        try:
            # Configure federation settings
            federation_config = {
                "collections": request.collections,
                "collection_weights": request.collection_weights or {},
                "merge_strategy": request.merge_strategy,
                "enable_deduplication": request.enable_deduplication,
                "limit_per_collection": request.limit_per_collection
            }

            # Create advanced search request
            search_request = AdvancedSearchRequest(
                query=request.query,
                limit=request.final_limit,
                search_mode=SearchMode.FEDERATED,
                pipeline=SearchPipeline.COMPREHENSIVE,
                enable_expansion=True,
                enable_federation=True,
                federation_config=federation_config,
                enable_caching=True
            )

            # Execute search
            result = await orchestrator.search(search_request)

            # Organize results by collection
            results_by_collection = {}
            for res in result.results:
                collection = res.get("metadata", {}).get("source_collection", "unknown")
                if collection not in results_by_collection:
                    results_by_collection[collection] = []

                results_by_collection[collection].append(SearchResult(
                    id=res.get("id", str(uuid4())),
                    content=res.get("content", ""),
                    score=res.get("score", 0.0),
                    url=res.get("url"),
                    title=res.get("title"),
                    metadata=res.get("metadata", {})
                ))

            response = {
                "request_id": request_id,
                "total_results": len(result.results),
                "collections_searched": result.search_metadata.get("collections_searched", []),
                "collections_failed": result.search_metadata.get("collections_failed", []),
                "results_by_collection": results_by_collection,
                "merged_results": [
                    SearchResult(
                        id=res.get("id", str(uuid4())),
                        content=res.get("content", ""),
                        score=res.get("score", 0.0),
                        url=res.get("url"),
                        title=res.get("title"),
                        metadata=res.get("metadata", {})
                    )
                    for res in result.results
                ],
                "federation_metadata": {
                    "merge_strategy": request.merge_strategy,
                    "deduplication_applied": request.enable_deduplication,
                    "quality_score": result.quality_score
                }
            }

            await ctx.info(
                f"Federated search {request_id} completed: "
                f"{len(result.results)} total results from "
                f"{len(results_by_collection)} collections"
            )
            return response

        except Exception as e:
            await ctx.error(f"Federated search {request_id} failed: {e}")
            logger.error(f"Federated search failed: {e}")
            raise

    @mcp.tool()
    async def search_with_custom_pipeline(
        request: PipelineSearchRequest,
        ctx: Context
    ) -> dict[str, Any]:
        """
        Search with custom processing pipeline configuration.
        
        Configure and execute a custom search pipeline with specific stages,
        parameters, and performance constraints.
        """
        request_id = str(uuid4())
        await ctx.info(
            f"Starting custom pipeline search {request_id} with "
            f"{len(request.stages)} stages"
        )

        try:
            # Map pipeline name to enum
            pipeline_map = {
                "fast": SearchPipeline.FAST,
                "balanced": SearchPipeline.BALANCED,
                "comprehensive": SearchPipeline.COMPREHENSIVE,
                "discovery": SearchPipeline.DISCOVERY,
                "precision": SearchPipeline.PRECISION,
                "personalized": SearchPipeline.PERSONALIZED
            }

            # Create advanced search request
            search_request = AdvancedSearchRequest(
                query=request.query,
                collection_name=request.collection_name,
                limit=request.limit,
                search_mode=SearchMode.INTELLIGENT,
                pipeline=pipeline_map.get(request.pipeline, SearchPipeline.BALANCED),
                max_processing_time_ms=request.time_budget_ms,
                quality_threshold=request.quality_threshold,
                enable_caching=True
            )

            # Configure individual stages
            if "expansion" in request.stages:
                search_request.enable_expansion = True
                if "expansion" in request.stage_configs:
                    search_request.expansion_config = request.stage_configs["expansion"]

            if "clustering" in request.stages:
                search_request.enable_clustering = True
                if "clustering" in request.stage_configs:
                    search_request.clustering_config = request.stage_configs["clustering"]

            if "ranking" in request.stages:
                search_request.enable_personalization = True
                if "ranking" in request.stage_configs:
                    search_request.ranking_config = request.stage_configs["ranking"]

            if "federation" in request.stages:
                search_request.enable_federation = True
                if "federation" in request.stage_configs:
                    search_request.federation_config = request.stage_configs["federation"]

            # Execute search
            result = await orchestrator.search(search_request)

            # Convert to response format
            response = {
                "request_id": request_id,
                "results": [
                    SearchResult(
                        id=res.get("id", str(uuid4())),
                        content=res.get("content", ""),
                        score=res.get("score", 0.0),
                        url=res.get("url"),
                        title=res.get("title"),
                        metadata=res.get("metadata", {})
                    ).__dict__
                    for res in result.results
                ],
                "pipeline_execution": {
                    "pipeline": request.pipeline,
                    "stages_executed": [
                        stage.stage.value for stage in result.stage_results
                        if stage.success
                    ],
                    "stages_failed": [
                        stage.stage.value for stage in result.stage_results
                        if not stage.success
                    ],
                    "total_processing_time_ms": result.total_processing_time_ms,
                    "stage_timings": {
                        stage.stage.value: stage.processing_time_ms
                        for stage in result.stage_results
                    }
                },
                "quality_metrics": {
                    "quality_score": result.quality_score,
                    "diversity_score": result.diversity_score,
                    "relevance_score": result.relevance_score
                },
                "features_used": result.features_used,
                "optimizations_applied": result.optimizations_applied
            }

            await ctx.info(
                f"Custom pipeline search {request_id} completed in "
                f"{result.total_processing_time_ms:.1f}ms with quality score "
                f"{result.quality_score:.2f}"
            )
            return response

        except Exception as e:
            await ctx.error(f"Custom pipeline search {request_id} failed: {e}")
            logger.error(f"Custom pipeline search failed: {e}")
            raise

