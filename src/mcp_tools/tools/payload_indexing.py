"""Payload indexing management tools for MCP server."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastmcp import Context
from qdrant_client import models

from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.models.responses import (
    GenericDictResponse,
    ReindexCollectionResponse,
)
from src.security.ml_security import MLSecurityValidator


logger = logging.getLogger(__name__)

# Default payload index definitions expressed with Qdrant field schemas.
_INDEX_DEFINITIONS: dict[str, models.PayloadSchemaType] = {
    "site_name": models.PayloadSchemaType.KEYWORD,
    "embedding_model": models.PayloadSchemaType.KEYWORD,
    "title": models.PayloadSchemaType.TEXT,
    "word_count": models.PayloadSchemaType.INTEGER,
    "crawl_timestamp": models.PayloadSchemaType.DATETIME,
}

_VECTOR_SERVICE_INIT_LOCK = asyncio.Lock()


async def _get_vector_service(client_manager: ClientManager):
    """Return an initialized VectorStoreService instance."""

    service = await client_manager.get_vector_store_service()
    if not service.is_initialized():
        async with _VECTOR_SERVICE_INIT_LOCK:
            if not service.is_initialized():
                await service.initialize()
    return service


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


def register_tools(mcp, client_manager: ClientManager):
    """Register payload indexing helpers with the MCP server."""

    validator = MLSecurityValidator.from_unified_config()

    @mcp.tool()
    async def create_payload_indexes(
        collection_name: str,
        ctx: Context,
    ) -> GenericDictResponse:
        request_id = str(uuid4())
        await ctx.info(
            f"Creating payload indexes for collection: {collection_name} "
            f"(request {request_id})"
        )

        safe_name = validator.validate_collection_name(collection_name)
        service = await _get_vector_service(client_manager)
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
        safe_name = validator.validate_collection_name(collection_name)
        service = await _get_vector_service(client_manager)
        summary = await service.get_payload_index_summary(safe_name)
        count = summary["indexed_fields_count"]
        await ctx.info(f"Collection {safe_name} exposes {count} payload indexes")
        return _normalise_summary(safe_name, summary)

    @mcp.tool()
    async def reindex_collection(
        collection_name: str,
        ctx: Context,
    ) -> ReindexCollectionResponse:
        request_id = str(uuid4())
        await ctx.info(
            "Reindexing payload fields for collection: "
            f"{collection_name} (request {request_id})"
        )

        safe_name = validator.validate_collection_name(collection_name)
        service = await _get_vector_service(client_manager)

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
        if ctx:
            await ctx.info(
                "Benchmarking filtered search on %s",  # type: ignore[arg-type]
                collection_name,
            )

        safe_name = validator.validate_collection_name(collection_name)
        clean_query = validator.validate_query_string(query)

        service = await _get_vector_service(client_manager)

        start = perf_counter()
        matches = await service.search_documents(
            safe_name,
            clean_query,
            limit=10,
            filters=test_filters,
        )
        elapsed_ms = (perf_counter() - start) * 1000

        results = [
            {
                "id": match.id,
                "score": match.score,
                "payload": match.payload or {},
            }
            for match in matches
        ]

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
