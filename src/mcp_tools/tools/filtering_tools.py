"""Advanced filtering tools for MCP server.

This module exposes advanced filtering capabilities through the Model Context Protocol,
including temporal filtering, content type filtering, metadata filtering, and filter
composition with boolean logic.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from fastmcp import Context
from pydantic import BaseModel, Field

from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.models.responses import SearchResult
from src.services.query_processing import (
    AdvancedSearchOrchestrator,
    AdvancedSearchRequest,
    AdvancedSearchResult,
    SearchMode,
    SearchPipeline,
)


logger = logging.getLogger(__name__)

_EPOCH = datetime.fromtimestamp(0, tz=UTC)
DEFAULT_MAX_PROCESSING_TIME_MS = 3000.0


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


def _build_search_request(
    query: str,
    collection_name: str,
    limit: int,
    *,
    mode: SearchMode = SearchMode.ENHANCED,
    pipeline: SearchPipeline = SearchPipeline.BALANCED,
) -> AdvancedSearchRequest:
    """Create an orchestrator search request with explicit defaults."""

    return AdvancedSearchRequest(
        query=query,
        collection_name=collection_name,
        limit=limit,
        offset=0,
        mode=mode,
        pipeline=pipeline,
        enable_expansion=True,
        enable_clustering=False,
        enable_personalization=False,
        enable_federation=False,
        enable_rag=False,
        rag_max_tokens=None,
        rag_temperature=None,
        require_high_confidence=False,
        user_id=None,
        session_id=None,
        enable_caching=True,
        max_processing_time_ms=DEFAULT_MAX_PROCESSING_TIME_MS,
    )


def _prepare_entries(result: AdvancedSearchResult) -> list[dict[str, Any]]:
    """Return a mutable copy of orchestrator results."""

    entries: list[dict[str, Any]] = []
    for item in result.results:
        entry = dict(item)
        metadata = entry.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = dict(metadata)
        entry["metadata"] = metadata
        entries.append(entry)
    return entries


def _convert_entries(
    entries: Iterable[dict[str, Any]], *, limit: int
) -> list[SearchResult]:
    """Convert raw orchestrator entries to MCP search results."""

    converted: list[SearchResult] = []
    for entry in entries:
        payload = dict(entry)
        payload.setdefault("id", "unknown")
        payload.setdefault("content", "")
        payload.setdefault("score", 0.0)
        if payload.get("metadata") is None:
            payload["metadata"] = {}
        converted.append(SearchResult.model_validate(payload))
        if len(converted) >= limit:
            break
    return converted


def _parse_datetime(value: Any) -> datetime | None:
    """Parse ISO8601 strings or timestamps into timezone-aware datetimes."""

    if value is None:
        return None

    parsed: datetime | None = None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(float(value), tz=UTC)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped:
            normalized = (
                stripped[:-1] + "+00:00" if stripped.endswith("Z") else stripped
            )
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                parsed = None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _parse_time_window(window: str | None) -> timedelta | None:
    """Parse simple relative windows such as '7d' or '12h'."""

    if not window:
        return None
    normalized = window.strip().lower()
    if len(normalized) < 2:
        return None
    unit = normalized[-1]
    try:
        value = int(normalized[:-1])
    except ValueError:
        return None
    seconds_multiplier = {
        "h": 3600,
        "d": 86400,
        "w": 604800,
        "m": 2592000,
    }.get(unit)
    if seconds_multiplier is None:
        return None
    return timedelta(seconds=value * seconds_multiplier)


def _extract_timestamp(entry: dict[str, Any]) -> datetime | None:
    """Extract the best-effort timestamp from a result entry."""

    metadata = entry.get("metadata") or {}
    candidates = [
        entry.get("timestamp"),
        metadata.get("timestamp"),
        metadata.get("created_at"),
        metadata.get("updated_at"),
        metadata.get("crawled_at"),
        metadata.get("published_at"),
        metadata.get("datetime"),
        metadata.get("date"),
    ]
    for candidate in candidates:
        parsed = _parse_datetime(candidate)
        if parsed is not None:
            return parsed
    return None


def _entry_key(entry: dict[str, Any]) -> str:
    """Return a stable key for result deduplication."""

    identifier = entry.get("id")
    if isinstance(identifier, str) and identifier:
        return identifier
    if identifier is not None:
        return str(identifier)
    content = entry.get("content")
    if isinstance(content, str) and content:
        return f"content:{content}"
    return f"object:{id(entry)}"


def _filter_temporal(
    entries: list[dict[str, Any]], request: TemporalFilterRequest
) -> list[dict[str, Any]]:
    """Apply temporal filtering and freshness weighting."""

    start = _parse_datetime(request.start_date)
    end = _parse_datetime(request.end_date)
    window_delta = _parse_time_window(request.time_window)
    if window_delta:
        window_start = datetime.now(tz=UTC) - window_delta
        start = window_start if start is None or window_start > start else start

    filtered_pairs: list[tuple[dict[str, Any], datetime | None]] = []
    for entry in entries:
        timestamp = _extract_timestamp(entry)
        if start and (timestamp is None or timestamp < start):
            continue
        if end and (timestamp is None or timestamp > end):
            continue
        filtered_pairs.append((entry, timestamp))

    if request.freshness_weight <= 0:
        return [entry for entry, _ in filtered_pairs]

    weight = min(max(request.freshness_weight, 0.0), 1.0)
    now = datetime.now(tz=UTC)
    scored_pairs: list[tuple[dict[str, Any], datetime]] = []
    for entry, timestamp in filtered_pairs:
        base_score = float(entry.get("score", 0.0))
        if timestamp is not None:
            age_seconds = max((now - timestamp).total_seconds(), 0.0)
            freshness = 1.0 / (1.0 + age_seconds / 86400.0)
            scored_pairs.append((entry, timestamp))
        else:
            freshness = 0.0
            scored_pairs.append((entry, _EPOCH))
        entry["score"] = (1.0 - weight) * base_score + weight * freshness

    scored_pairs.sort(key=lambda pair: pair[1], reverse=True)
    return [entry for entry, _ in scored_pairs]


def _extract_content_type(entry: dict[str, Any]) -> str | None:
    """Extract normalized content type from entry metadata."""

    metadata = entry.get("metadata") or {}
    candidates = [
        entry.get("content_type"),
        metadata.get("content_type"),
        metadata.get("document_type"),
        metadata.get("type"),
        metadata.get("category"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().lower()
    return None


def _filter_by_content_type(
    entries: list[dict[str, Any]], request: ContentTypeFilterRequest
) -> list[dict[str, Any]]:
    """Filter entries according to allowed, excluded, and priority types."""

    allowed = {value.lower() for value in request.allowed_types}
    exclude = {value.lower() for value in request.exclude_types}
    priority_order = {
        value.lower(): index for index, value in enumerate(request.priority_types)
    }

    prioritized: list[tuple[int, dict[str, Any]]] = []
    regular: list[dict[str, Any]] = []

    for entry in entries:
        content_type = _extract_content_type(entry)
        if content_type is None:
            if allowed:
                continue
        else:
            if allowed and content_type not in allowed:
                continue
            if content_type in exclude:
                continue

        if content_type is not None and content_type in priority_order:
            prioritized.append((priority_order[content_type], entry))
        else:
            regular.append(entry)

    prioritized.sort(key=lambda item: item[0])
    prioritized_entries = [item[1] for item in prioritized]
    return prioritized_entries + regular


def _matches_metadata_field(
    actual: Any, expected: Any, *, exact: bool, case_sensitive: bool
) -> bool:
    """Return True when a metadata value matches the expected constraint."""

    if isinstance(actual, (list, tuple, set)):
        return any(
            _matches_metadata_field(
                item, expected, exact=exact, case_sensitive=case_sensitive
            )
            for item in actual
        )
    if isinstance(expected, (list, tuple, set)):
        return any(
            _matches_metadata_field(
                actual, item, exact=exact, case_sensitive=case_sensitive
            )
            for item in expected
        )

    if isinstance(actual, str) and isinstance(expected, str):
        actual_cmp = actual if case_sensitive else actual.casefold()
        expected_cmp = expected if case_sensitive else expected.casefold()
        return actual_cmp == expected_cmp if exact else expected_cmp in actual_cmp

    if isinstance(actual, str) and not case_sensitive:
        actual = actual.casefold()
    if isinstance(expected, str) and not case_sensitive:
        expected = expected.casefold()
    return actual == expected


def _filter_by_metadata(
    entries: list[dict[str, Any]], request: MetadataFilterRequest
) -> list[dict[str, Any]]:
    """Filter entries using metadata key/value pairs."""

    operator = request.filter_operator.upper()
    exact = request.exact_match
    case_sensitive = request.case_sensitive

    filtered: list[dict[str, Any]] = []
    for entry in entries:
        metadata = entry.get("metadata") or {}
        outcomes = [
            _matches_metadata_field(
                metadata.get(key), expected, exact=exact, case_sensitive=case_sensitive
            )
            for key, expected in request.metadata_filters.items()
        ]
        if not outcomes:
            filtered.append(entry)
            continue
        if operator == "OR":
            if any(outcomes):
                filtered.append(entry)
        else:
            if all(outcomes):
                filtered.append(entry)
    return filtered


def _filter_by_similarity(
    entries: list[dict[str, Any]], request: SimilarityFilterRequest
) -> list[dict[str, Any]]:
    """Filter entries based on similarity score bounds."""

    filtered = [
        entry
        for entry in entries
        if request.min_similarity
        <= float(entry.get("score", 0.0))
        <= request.max_similarity
    ]

    if not (request.adaptive_threshold or request.boost_recent):
        return filtered

    filtered.sort(
        key=lambda entry: (
            float(entry.get("score", 0.0)) if request.adaptive_threshold else 0.0,
            _extract_timestamp(entry) or _EPOCH if request.boost_recent else _EPOCH,
        ),
        reverse=True,
    )
    return filtered


def _combine_filter_results(
    base_entries: list[dict[str, Any]],
    filtered_groups: list[list[dict[str, Any]]],
    operator: str,
) -> list[dict[str, Any]]:
    """Combine filtered result sets according to the requested operator."""

    if not filtered_groups:
        return base_entries

    if operator.upper() == "OR":
        seen: set[str] = set()
        combined: list[dict[str, Any]] = []
        for group in filtered_groups:
            for entry in group:
                key = _entry_key(entry)
                if key not in seen:
                    seen.add(key)
                    combined.append(entry)
        return combined

    intersection_keys: set[str] | None = None
    for group in filtered_groups:
        group_keys = {_entry_key(entry) for entry in group}
        if intersection_keys is None:
            intersection_keys = group_keys
        else:
            intersection_keys &= group_keys
        if not intersection_keys:
            return []

    assert intersection_keys is not None  # Narrow type for mypy/pyright
    ordered: list[dict[str, Any]] = []
    for entry in base_entries:
        if _entry_key(entry) in intersection_keys:
            ordered.append(entry)
    return ordered


async def temporal_filter_tool(
    request: TemporalFilterRequest,
    ctx: Context,
    orchestrator: AdvancedSearchOrchestrator,
) -> list[SearchResult]:
    """Implementation for temporal filtering."""

    request_id = str(uuid4())
    await ctx.info(f"Starting temporal filtered search {request_id}")

    try:
        search_request = _build_search_request(
            request.query, request.collection_name, request.limit
        )
        orchestrator_result = await orchestrator.search(search_request)
        entries = _prepare_entries(orchestrator_result)
        filtered_entries = _filter_temporal(entries, request)
        converted_results = _convert_entries(filtered_entries, limit=request.limit)
        await ctx.info(
            f"Temporal search completed: {len(converted_results)} results "
            f"(processing time: {orchestrator_result.processing_time_ms:.1f}ms)"
        )
        return converted_results
    except Exception as exc:  # noqa: BLE001
        await ctx.error(f"Temporal filter search failed: {exc!s}")
        logger.exception("Temporal filter search failed: %s", exc)
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
        search_request = _build_search_request(
            request.query, request.collection_name, request.limit
        )
        orchestrator_result = await orchestrator.search(search_request)
        entries = _prepare_entries(orchestrator_result)
        filtered_entries = _filter_by_content_type(entries, request)
        converted_results = _convert_entries(filtered_entries, limit=request.limit)
        await ctx.info(
            f"Content type search completed: {len(converted_results)} results "
            f"(processing time: {orchestrator_result.processing_time_ms:.1f}ms)"
        )
        return converted_results
    except Exception as exc:  # noqa: BLE001
        await ctx.error(f"Content type filter search failed: {exc!s}")
        logger.exception("Content type filter search failed: %s", exc)
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
        search_request = _build_search_request(
            request.query, request.collection_name, request.limit
        )
        orchestrator_result = await orchestrator.search(search_request)
        entries = _prepare_entries(orchestrator_result)
        filtered_entries = _filter_by_metadata(entries, request)
        converted_results = _convert_entries(filtered_entries, limit=request.limit)
        await ctx.info(
            f"Metadata search completed: {len(converted_results)} results "
            f"(processing time: {orchestrator_result.processing_time_ms:.1f}ms)"
        )
        return converted_results
    except Exception as exc:  # noqa: BLE001
        await ctx.error(f"Metadata filter search failed: {exc!s}")
        logger.exception("Metadata filter search failed: %s", exc)
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
        search_request = _build_search_request(
            request.query, request.collection_name, request.limit
        )
        orchestrator_result = await orchestrator.search(search_request)
        entries = _prepare_entries(orchestrator_result)
        filtered_entries = _filter_by_similarity(entries, request)
        converted_results = _convert_entries(filtered_entries, limit=request.limit)
        await ctx.info(
            f"Similarity search completed: {len(converted_results)} results "
            f"(processing time: {orchestrator_result.processing_time_ms:.1f}ms)"
        )
        return converted_results
    except Exception as exc:  # noqa: BLE001
        await ctx.error(f"Similarity filter search failed: {exc!s}")
        logger.exception("Similarity filter search failed: %s", exc)
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
        search_request = _build_search_request(
            request.query, request.collection_name, request.limit
        )
        orchestrator_result = await orchestrator.search(search_request)
        base_entries = _prepare_entries(orchestrator_result)

        filtered_groups: list[list[dict[str, Any]]] = []
        if request.temporal_config:
            temporal_request = TemporalFilterRequest.model_validate(
                {
                    **request.temporal_config,
                    "collection_name": request.collection_name,
                    "query": request.query,
                    "limit": request.limit,
                }
            )
            filtered_groups.append(_filter_temporal(base_entries, temporal_request))

        if request.content_type_config:
            content_type_request = ContentTypeFilterRequest.model_validate(
                {
                    **request.content_type_config,
                    "collection_name": request.collection_name,
                    "query": request.query,
                    "limit": request.limit,
                }
            )
            filtered_groups.append(
                _filter_by_content_type(base_entries, content_type_request)
            )

        if request.metadata_config:
            metadata_request = MetadataFilterRequest.model_validate(
                {
                    **request.metadata_config,
                    "collection_name": request.collection_name,
                    "query": request.query,
                    "limit": request.limit,
                }
            )
            filtered_groups.append(_filter_by_metadata(base_entries, metadata_request))

        if request.similarity_config:
            similarity_request = SimilarityFilterRequest.model_validate(
                {
                    **request.similarity_config,
                    "collection_name": request.collection_name,
                    "query": request.query,
                    "limit": request.limit,
                }
            )
            filtered_groups.append(
                _filter_by_similarity(base_entries, similarity_request)
            )

        if request.optimize_order and filtered_groups:
            filtered_groups.sort(key=len)

        combined_entries = _combine_filter_results(
            base_entries, filtered_groups, request.operator
        )
        converted_results = _convert_entries(combined_entries, limit=request.limit)
        await ctx.info(
            f"Composite search completed: {len(converted_results)} results "
            f"(processing time: {orchestrator_result.processing_time_ms:.1f}ms)"
        )
        return converted_results
    except Exception as exc:  # noqa: BLE001
        await ctx.error(f"Composite filter search failed: {exc!s}")
        logger.exception("Composite filter search failed: %s", exc)
        raise


def register_filtering_tools(mcp, _client_manager: ClientManager):
    """Register advanced filtering tools with the MCP server."""
    orchestrator = create_orchestrator()

    @mcp.tool()
    async def search_with_temporal_filter(
        request: TemporalFilterRequest, ctx: Context
    ) -> list[SearchResult]:
        """Search with temporal filtering for date-based content."""

        return await temporal_filter_tool(request, ctx, orchestrator)

    @mcp.tool()
    async def search_with_content_type_filter(
        request: ContentTypeFilterRequest, ctx: Context
    ) -> list[SearchResult]:
        """Search with content type filtering."""

        return await content_type_filter_tool(request, ctx, orchestrator)

    @mcp.tool()
    async def search_with_metadata_filter(
        request: MetadataFilterRequest, ctx: Context
    ) -> list[SearchResult]:
        """Search with metadata filtering."""

        return await metadata_filter_tool(request, ctx, orchestrator)

    @mcp.tool()
    async def search_with_similarity_filter(
        request: SimilarityFilterRequest, ctx: Context
    ) -> list[SearchResult]:
        """Search with similarity-based filtering."""

        return await similarity_filter_tool(request, ctx, orchestrator)

    @mcp.tool()
    async def search_with_composite_filter(
        request: CompositeFilterRequest, ctx: Context
    ) -> list[SearchResult]:
        """Search with composite filtering using boolean logic."""

        return await composite_filter_tool(request, ctx, orchestrator)

    logger.info("Advanced filtering tools registered successfully")


# Helper function to create the orchestrator


def create_orchestrator() -> AdvancedSearchOrchestrator:
    """Create and configure the search orchestrator."""
    return AdvancedSearchOrchestrator(
        cache_size=1000, enable_performance_optimization=True
    )
