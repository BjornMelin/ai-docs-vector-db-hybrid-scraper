"""Payload indexing management tools for MCP server."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Protocol, cast, runtime_checkable
from uuid import uuid4

from fastmcp import Context
from qdrant_client import models

from src.contracts.retrieval import SearchRecord
from src.mcp_tools.models.responses import (
    GenericDictResponse,
    ReindexCollectionResponse,
)
from src.security.ml_security import MLSecurityValidator
from src.services.vector_db.service import VectorStoreService


logger = logging.getLogger(__name__)

# Default payload index definitions expressed with Qdrant field schemas.
_INDEX_DEFINITIONS: dict[str, models.PayloadSchemaType] = {
    "site_name": models.PayloadSchemaType.KEYWORD,
    "embedding_model": models.PayloadSchemaType.KEYWORD,
    "title": models.PayloadSchemaType.TEXT,
    "word_count": models.PayloadSchemaType.INTEGER,
    "crawl_timestamp": models.PayloadSchemaType.DATETIME,
}


@runtime_checkable
class VectorStoreServiceProvider(Protocol):
    """Protocol for objects that can provide a VectorStoreService instance."""

    async def get_vector_store_service(self) -> VectorStoreService:
        """Return the VectorStoreService instance."""
        ...


def _record_to_dict(record: SearchRecord) -> dict[str, Any]:
    """Serialize a search record into a plain dictionary."""
    payload: dict[str, Any] = {
        "id": record.id,
        "score": record.score,
    }

    if record.content:
        payload["content"] = record.content
    if record.title is not None:
        payload["title"] = record.title
    if record.url is not None:
        payload["url"] = record.url
    if record.collection is not None:
        payload["collection"] = record.collection
    if record.normalized_score is not None:
        payload["normalized_score"] = record.normalized_score
    if record.raw_score is not None:
        payload["raw_score"] = record.raw_score
    if record.group_id is not None:
        payload["group_id"] = record.group_id
    if record.group_rank is not None:
        payload["group_rank"] = record.group_rank
    if record.grouping_applied is not None:
        payload["grouping_applied"] = record.grouping_applied
    if record.content_type is not None:
        payload["content_type"] = record.content_type

    payload["metadata"] = dict(record.metadata or {})
    return payload


def _normalise_summary(
    collection: str,
    summary: Mapping[str, Any],
    *,
    status: str = "success",
    request_id: str | None = None,
) -> GenericDictResponse:
    """Shape summary metadata into a stable GenericDictResponse."""
    indexed_fields = summary.get("indexed_fields", [])
    points = summary.get("points_count", 0)
    payload_schema = summary.get("payload_schema", {})
    payload: dict[str, Any] = {
        "collection_name": collection,
        "status": status,
        "indexes_created": len(indexed_fields),
        "indexed_fields_count": len(indexed_fields),
        "indexed_fields": indexed_fields,
        "total_points": points,
        "payload_schema": payload_schema,
    }
    if request_id is not None:
        payload["request_id"] = request_id
    return GenericDictResponse.model_validate(payload)


def register_tools(
    mcp,
    *,
    vector_service: VectorStoreService | VectorStoreServiceProvider,
) -> None:
    """Register payload indexing helpers with the MCP server."""
    validator = MLSecurityValidator.from_unified_config()
    cached_service: VectorStoreService | None = None

    async def _resolve_service() -> VectorStoreService:
        """Resolve the vector store service from the provided vector_service."""
        nonlocal cached_service
        if cached_service is not None:
            return cached_service

        if isinstance(vector_service, VectorStoreService):
            cached_service = vector_service
            return cached_service

        if not isinstance(vector_service, VectorStoreServiceProvider):
            msg = (
                "vector_service must be a VectorStoreService or implement "
                "VectorStoreServiceProvider"
            )
            raise TypeError(msg)

        resolved = await vector_service.get_vector_store_service()
        if resolved is None:
            msg = "get_vector_store_service() returned None"
            raise ValueError(msg)
        if isinstance(resolved, VectorStoreService):
            cached_service = resolved
            return cached_service

        required = (
            "list_collections",
            "ensure_payload_indexes",
            "get_payload_index_summary",
            "drop_payload_indexes",
            "collection_stats",
            "search_documents",
        )
        missing = [name for name in required if not hasattr(resolved, name)]
        if missing:
            msg = (
                "get_vector_store_service() must return a VectorStoreService-like "
                f"object; missing {missing} on {type(resolved).__name__}"
            )
            raise TypeError(msg)
        cached_service = cast(VectorStoreService, resolved)
        return cached_service

    @mcp.tool()
    async def create_payload_indexes(
        collection_name: str,
        ctx: Context,
    ) -> GenericDictResponse:
        """Create payload indexes for a collection."""
        request_id = str(uuid4())
        await ctx.info(
            f"Creating payload indexes for collection: {collection_name} "
            f"(request {request_id})"
        )

        safe_name = validator.validate_collection_name(collection_name)
        service = await _resolve_service()
        collections = await service.list_collections()
        if safe_name not in collections:
            msg = f"Collection '{safe_name}' not found"
            raise ValueError(msg)

        summary = await service.ensure_payload_indexes(safe_name, _INDEX_DEFINITIONS)
        await ctx.info(
            f"Ensured {summary['indexed_fields_count']} payload indexes for {safe_name}"
        )
        return _normalise_summary(safe_name, summary, request_id=request_id)

    @mcp.tool()
    async def list_payload_indexes(
        collection_name: str,
        ctx: Context,
    ) -> GenericDictResponse:
        """List existing payload indexes for a collection."""
        safe_name = validator.validate_collection_name(collection_name)
        service = await _resolve_service()
        summary = await service.get_payload_index_summary(safe_name)
        count = summary["indexed_fields_count"]
        await ctx.info(f"Collection {safe_name} exposes {count} payload indexes")
        return _normalise_summary(safe_name, summary)

    @mcp.tool()
    async def reindex_collection(
        collection_name: str,
        ctx: Context,
    ) -> ReindexCollectionResponse:
        """Reindex payload fields for an existing collection."""
        request_id = str(uuid4())
        await ctx.info(
            "Reindexing payload fields for collection: "
            f"{collection_name} (request {request_id})"
        )

        safe_name = validator.validate_collection_name(collection_name)
        service = await _resolve_service()

        before = await service.get_payload_index_summary(safe_name)
        await service.drop_payload_indexes(safe_name, _INDEX_DEFINITIONS.keys())
        after = await service.ensure_payload_indexes(safe_name, _INDEX_DEFINITIONS)

        await ctx.info(f"Successfully reindexed payload fields for {safe_name}")

        return ReindexCollectionResponse(
            status="success",
            collection=safe_name,
            reindexed_count=after["indexed_fields_count"],
            details={
                "indexes_before": before["indexed_fields_count"],
                "indexes_after": after["indexed_fields_count"],
                "indexed_fields": after["indexed_fields"],
                "total_points": after.get("points_count", 0),
                "request_id": request_id,
            },
        )

    @mcp.tool()
    async def benchmark_filtered_search(
        collection_name: str,
        test_filters: dict[str, Any],
        query: str = "documentation search test",
        ctx: Context | None = None,
    ) -> GenericDictResponse:
        """Benchmark filtered search performance on a collection."""
        if ctx:
            await ctx.info(f"Benchmarking filtered search on {collection_name}")

        safe_name = validator.validate_collection_name(collection_name)
        clean_query = validator.validate_query_string(query)

        service = await _resolve_service()

        start = perf_counter()
        matches = await service.search_documents(
            safe_name,
            clean_query,
            limit=10,
            filters=test_filters,
        )
        elapsed_ms = (perf_counter() - start) * 1000

        results = [_record_to_dict(match) for match in matches]

        stats = await service.collection_stats(safe_name)
        summary = await service.get_payload_index_summary(safe_name)

        performance_estimate = (
            "10-100x faster than unindexed"
            if summary["indexed_fields_count"]
            else "No indexes detected"
        )

        payload = {
            "collection_name": safe_name,
            "query": clean_query,
            "filters_applied": test_filters,
            "results_found": len(results),
            "search_time_ms": round(elapsed_ms, 2),
            "benchmark_timestamp": datetime.now(tz=UTC).isoformat(),
            "performance_estimate": performance_estimate,
            "results": results,
            "total_points": stats.get("points_count", 0),
            "indexed_fields": summary["indexed_fields"],
        }
        if ctx:
            await ctx.info(
                f"Filtered search completed in {payload['search_time_ms']:.2f}ms with "
                f"{payload['results_found']} results"
            )
        return GenericDictResponse.model_validate(payload)
