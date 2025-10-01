"""Model Context Protocol query processing tools.

The module exposes typed wrappers around the search orchestrator. Each helper
prepares a request, invokes the orchestrator, and converts raw responses into
MCP-friendly result models.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from fastmcp import Context
from pydantic import BaseModel, Field

from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.models.responses import SearchResult
from src.services.query_processing import (
    SearchMode,
    SearchOrchestrator,
    SearchPipeline,
    SearchRequest,
)


ResultPayload = Mapping[str, Any]
ExplanationFactory = Callable[[ResultPayload], str | None]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request models used by MCP tools
# ---------------------------------------------------------------------------


class QueryExpansionRequest(BaseModel):
    """Request payload for the query expansion tool."""

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Search query to expand")
    expansion_methods: list[str] = Field(
        default_factory=lambda: ["synonyms", "related_terms"],
        description="Expansion methods to apply",
    )
    expansion_depth: int = Field(2, ge=1, le=5, description="Expansion breadth")
    context_window: int = Field(100, ge=50, le=500, description="Context window size")
    limit: int = Field(10, ge=1, le=100, description="Number of results")


class ClusteredSearchRequest(BaseModel):
    """Request payload for clustered search."""

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Search query")
    num_clusters: int = Field(
        5, ge=2, le=20, description="Target number of clusters for presentation"
    )
    clustering_algorithm: str = Field(
        "kmeans",
        description="Clustering algorithm (kmeans, hierarchical, dbscan)",
    )
    min_cluster_size: int = Field(2, ge=1, description="Minimum cluster size")
    limit: int = Field(10, ge=1, le=100, description="Number of results to return")


class FederatedSearchRequest(BaseModel):
    """Request payload for federated search."""

    collections: list[str] = Field(..., description="Collections to search across")
    query: str = Field(..., description="Search query")
    federation_strategy: str = Field(
        "parallel",
        description="Strategy (parallel, sequential, adaptive)",
    )
    result_merging: str = Field(
        "score",
        description="Merge strategy (score, relevance, round_robin)",
    )
    per_collection_limit: int = Field(
        5, ge=1, le=50, description="Results per collection"
    )
    total_limit: int = Field(10, ge=1, le=100, description="Total results")


class PersonalizedSearchRequest(BaseModel):
    """Request payload for personalized search."""

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Search query")
    user_id: str = Field(..., description="User identifier")
    personalization_strength: float = Field(
        0.5, ge=0.0, le=1.0, description="Strength of personalization"
    )
    use_history: bool = Field(True, description="Use search history")
    use_preferences: bool = Field(True, description="Use preference signals")
    limit: int = Field(10, ge=1, le=100, description="Number of results")


class OrchestrationRequest(BaseModel):
    """Request payload for orchestrated search."""

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Search query")
    stages: list[str] | None = Field(
        default=None,
        description="Explicit pipeline stages (None => automatic selection)",
    )
    time_budget_ms: float = Field(
        3000.0, ge=100.0, description="Processing time budget in milliseconds"
    )
    quality_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Quality threshold for orchestration"
    )
    limit: int = Field(10, ge=1, le=100, description="Number of results")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def create_orchestrator() -> SearchOrchestrator:
    """Instantiate the shared search orchestrator used by MCP tools."""

    return SearchOrchestrator(enable_performance_optimization=True)


def _build_search_request(
    *,
    query: str,
    collection_name: str | None,
    limit: int,
    overrides: dict[str, Any] | None = None,
) -> SearchRequest:
    """Create a SearchRequest with consistent defaults."""

    params: dict[str, Any] = {
        "query": query,
        "collection_name": collection_name,
        "limit": limit,
        "offset": 0,
        "mode": SearchMode.ENHANCED,
        "pipeline": SearchPipeline.BALANCED,
        "enable_expansion": False,
        "enable_clustering": False,
        "enable_personalization": False,
        "enable_federation": False,
        "enable_rag": False,
        "rag_max_tokens": None,
        "rag_temperature": None,
        "rag_top_k": None,
        "require_high_confidence": False,
        "user_id": None,
        "session_id": None,
        "enable_caching": True,
        "max_processing_time_ms": 3000.0,
    }

    if overrides:
        params.update(overrides)

    return SearchRequest(**params)


def _safe_metadata(payload: ResultPayload) -> dict[str, Any]:
    """Return payload metadata as a mutable dictionary."""

    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        return dict(metadata)
    return {}


def _best_effort_score(value: Any) -> float:
    """Convert arbitrary score values to floats safely."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _convert_results(
    payloads: Iterable[ResultPayload],
    *,
    limit: int,
    explanation_factory: ExplanationFactory | None = None,
    global_metadata: Mapping[str, Any] | None = None,
) -> list[SearchResult]:
    """Convert orchestrator payloads into MCP SearchResult models."""

    converted: list[SearchResult] = []
    for payload in payloads:
        metadata = _safe_metadata(payload)
        if global_metadata:
            metadata.update(global_metadata)

        explanation = explanation_factory(payload) if explanation_factory else None
        if explanation:
            metadata.setdefault("explanation", explanation)

        url = payload.get("url") or metadata.get("url")
        title = payload.get("title")

        converted.append(
            SearchResult(
                id=str(payload.get("id", "")),
                content=str(payload.get("content", "")),
                score=_best_effort_score(payload.get("score")),
                title=title if isinstance(title, str) else None,
                url=url if isinstance(url, str) else None,
                metadata=metadata or None,
            )
        )

        if len(converted) >= limit:
            break

    return converted


async def _run_search(
    orchestrator: SearchOrchestrator,
    search_request: SearchRequest,
) -> Any:
    """Execute the orchestrator search call."""

    return await orchestrator.search(search_request)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def query_expansion_tool(
    request: QueryExpansionRequest,
    ctx: Context,
    orchestrator: SearchOrchestrator,
) -> list[SearchResult]:
    """Execute a search with query expansion enabled."""

    search_request = _build_search_request(
        query=request.query,
        collection_name=request.collection_name,
        limit=request.limit,
        overrides={
            "enable_expansion": True,
            "enable_caching": True,
            "max_processing_time_ms": 3000.0,
        },
    )

    await ctx.info("Starting query expansion search")

    try:
        result = await _run_search(orchestrator, search_request)
    except Exception:  # pragma: no cover - surfaced via ctx.error
        await ctx.error("Query expansion search failed")
        logger.exception("Query expansion search failed")
        raise

    expanded_query = (
        result.expanded_query if hasattr(result, "expanded_query") else None
    )

    def explanation(_: ResultPayload) -> str | None:
        if expanded_query:
            return f"Expanded query: {expanded_query}"
        return None

    converted = _convert_results(
        result.results,
        limit=request.limit,
        explanation_factory=explanation,
        global_metadata={"expanded_query": expanded_query} if expanded_query else None,
    )

    processed_query = getattr(result, "query_processed", request.query)
    await ctx.info(
        "Query expansion search completed: "
        f"{len(converted)} results (processed query: {processed_query})"
    )

    return converted


async def clustered_search_tool(
    request: ClusteredSearchRequest,
    ctx: Context,
    orchestrator: SearchOrchestrator,
) -> list[SearchResult]:
    """Execute a clustered search."""

    search_request = _build_search_request(
        query=request.query,
        collection_name=request.collection_name,
        limit=request.limit * 2,
        overrides={
            "enable_clustering": True,
            "enable_expansion": True,
            "max_processing_time_ms": 4000.0,
        },
    )

    await ctx.info("Starting clustered search")

    try:
        result = await _run_search(orchestrator, search_request)
    except Exception:  # pragma: no cover - surfaced via ctx.error
        await ctx.error("Clustered search failed")
        logger.exception("Clustered search failed")
        raise

    def explanation(payload: ResultPayload) -> str | None:
        cluster_id = payload.get("cluster_id")
        if cluster_id is None:
            return None
        cluster_label = payload.get("cluster_label")
        if isinstance(cluster_label, str) and cluster_label:
            return f"Cluster {cluster_id}: {cluster_label}"
        return f"Cluster {cluster_id}"

    converted = _convert_results(
        result.results,
        limit=request.limit,
        explanation_factory=explanation,
    )

    await ctx.info(
        "Clustered search completed: "
        f"{len(converted)} results (target clusters: {request.num_clusters})"
    )

    return converted


async def federated_search_tool(
    request: FederatedSearchRequest,
    ctx: Context,
    orchestrator: SearchOrchestrator,
) -> list[SearchResult]:
    """Execute a federated search across collections."""

    primary_collection = request.collections[0] if request.collections else None

    search_request = _build_search_request(
        query=request.query,
        collection_name=primary_collection,
        limit=request.total_limit,
        overrides={
            "enable_federation": True,
            "enable_expansion": True,
            "max_processing_time_ms": 4000.0,
        },
    )

    await ctx.info("Starting federated search")

    try:
        result = await _run_search(orchestrator, search_request)
    except Exception:  # pragma: no cover - surfaced via ctx.error
        await ctx.error("Federated search failed")
        logger.exception("Federated search failed")
        raise

    def explanation(payload: ResultPayload) -> str | None:
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            collection = metadata.get("collection")
            if isinstance(collection, str) and collection:
                return f"From collection: {collection}"
        return None

    converted = _convert_results(
        result.results,
        limit=request.total_limit,
        explanation_factory=explanation,
    )

    await ctx.info(
        "Federated search completed: "
        f"{len(converted)} results from {len(request.collections)} collections"
    )

    return converted


async def personalized_search_tool(
    request: PersonalizedSearchRequest,
    ctx: Context,
    orchestrator: SearchOrchestrator,
) -> list[SearchResult]:
    """Execute a personalized search for the supplied user."""

    search_request = _build_search_request(
        query=request.query,
        collection_name=request.collection_name,
        limit=request.limit,
        overrides={
            "enable_personalization": True,
            "user_id": request.user_id,
            "enable_expansion": True,
            "max_processing_time_ms": 3500.0,
        },
    )

    await ctx.info("Starting personalized search")

    try:
        result = await _run_search(orchestrator, search_request)
    except Exception:  # pragma: no cover - surfaced via ctx.error
        await ctx.error("Personalized search failed")
        logger.exception("Personalized search failed")
        raise

    def explanation(payload: ResultPayload) -> str | None:
        personalized_score = payload.get("personalized_score")
        if personalized_score is None:
            return None
        return f"Personalized score: {_best_effort_score(personalized_score):.2f}"

    converted = _convert_results(
        result.results,
        limit=request.limit,
        explanation_factory=explanation,
    )

    await ctx.info(
        "Personalized search completed: "
        f"{len(converted)} results for user {request.user_id}"
    )

    return converted


async def orchestrated_search_tool(
    request: OrchestrationRequest,
    ctx: Context,
    orchestrator: SearchOrchestrator,
) -> list[SearchResult]:
    """Execute a fully orchestrated search pipeline."""

    search_request = _build_search_request(
        query=request.query,
        collection_name=request.collection_name,
        limit=request.limit,
        overrides={
            "enable_expansion": True,
            "enable_clustering": True,
            "enable_personalization": True,
            "max_processing_time_ms": request.time_budget_ms,
        },
    )

    await ctx.info("Starting orchestrated search")

    try:
        result = await _run_search(orchestrator, search_request)
    except Exception:  # pragma: no cover - surfaced via ctx.error
        await ctx.error("Orchestrated search failed")
        logger.exception("Orchestrated search failed")
        raise

    features_used = getattr(result, "features_used", [])

    def explanation(_: ResultPayload) -> str | None:
        if not features_used:
            return "Features: basic"
        return f"Features: {', '.join(features_used)}"

    converted = _convert_results(
        result.results,
        limit=request.limit,
        explanation_factory=explanation,
    )

    processing_time = getattr(result, "processing_time_ms", 0.0)
    await ctx.info(
        "Orchestrated search completed: "
        f"{len(converted)} results ({processing_time:.1f} ms)"
    )

    return converted


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------


def register_query_processing_tools(mcp, _client_manager: ClientManager) -> None:
    """Register query processing tools with the MCP server."""

    orchestrator = create_orchestrator()

    @mcp.tool()
    async def search_with_query_expansion(
        request: QueryExpansionRequest, ctx: Context
    ) -> list[SearchResult]:
        return await query_expansion_tool(request, ctx, orchestrator)

    @mcp.tool()
    async def search_with_clustering(
        request: ClusteredSearchRequest, ctx: Context
    ) -> list[SearchResult]:
        return await clustered_search_tool(request, ctx, orchestrator)

    @mcp.tool()
    async def search_federated(
        request: FederatedSearchRequest, ctx: Context
    ) -> list[SearchResult]:
        return await federated_search_tool(request, ctx, orchestrator)

    @mcp.tool()
    async def search_personalized(
        request: PersonalizedSearchRequest, ctx: Context
    ) -> list[SearchResult]:
        return await personalized_search_tool(request, ctx, orchestrator)

    @mcp.tool()
    async def search_orchestrated(
        request: OrchestrationRequest, ctx: Context
    ) -> list[SearchResult]:
        return await orchestrated_search_tool(request, ctx, orchestrator)

    logger.info("Query processing tools registered successfully")


def register_tools(mcp, client_manager: ClientManager) -> None:
    """Match other tool modules' registration API for consistency."""

    register_query_processing_tools(mcp, client_manager)


__all__ = [
    "ClusteredSearchRequest",
    "FederatedSearchRequest",
    "OrchestrationRequest",
    "PersonalizedSearchRequest",
    "QueryExpansionRequest",
    "register_query_processing_tools",
    "register_tools",
]
