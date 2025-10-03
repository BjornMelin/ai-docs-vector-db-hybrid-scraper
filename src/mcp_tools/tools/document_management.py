"""Document management MCP tools backed by VectorStoreService.

The tools expose collection lifecycle helpers without the
legacy caching, batching, or bespoke orchestration code paths.
"""

# pylint: disable=too-many-statements  # tool registration defines several closures sharing client state.

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from fastmcp import Context

from src.infrastructure.client_manager import ClientManager
from src.services.vector_db.service import VectorStoreService
from src.services.vector_db.types import CollectionSchema


logger = logging.getLogger(__name__)

_DEFAULT_VECTOR_SIZE = 1536
_DEFAULT_DISTANCE = "cosine"
_VECTOR_SERVICE_INIT_LOCK = asyncio.Lock()


async def _get_vector_service(client_manager: ClientManager) -> VectorStoreService:
    """Return the shared VectorStoreService instance, initialising if required."""

    service = await client_manager.get_vector_store_service()
    if not service.is_initialized():
        async with _VECTOR_SERVICE_INIT_LOCK:
            if not service.is_initialized():
                await service.initialize()
    return service


def _timestamp() -> str:
    """Return an ISO-8601 timestamp in UTC."""

    ts = datetime.now(tz=timezone.utc).isoformat()  # noqa: UP017 (Python 3.10 support)
    return ts.replace("+00:00", "Z")


def _normalize_stats(stats: Any) -> dict[str, Any]:
    """Convert service stats payloads to plain dictionaries."""

    if isinstance(stats, dict):
        return stats
    if hasattr(stats, "model_dump"):
        return stats.model_dump()  # type: ignore[no-untyped-call]
    if hasattr(stats, "dict"):
        return stats.dict()  # type: ignore[no-untyped-call]
    if hasattr(stats, "__dict__"):
        return dict(vars(stats))
    msg = f"Unsupported stats payload type: {type(stats)!r}"
    raise TypeError(msg)


def register_tools(mcp, client_manager: ClientManager) -> None:
    """Register document management helpers with the MCP server."""

    @mcp.tool()
    async def create_document_workspace(
        workspace_name: str,
        collections: list[str],
        configuration: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Provision collections for a logical workspace.

        Each collection name is prefixed with the workspace identifier to keep the
        storage layout consistent. The tool delegates to ``VectorStoreService`` to
        guarantee the collections exist with the requested schema.
        """

        service = await _get_vector_service(client_manager)
        config = configuration or {}
        vector_size = int(config.get("vector_size", _DEFAULT_VECTOR_SIZE))
        distance = str(config.get("distance", _DEFAULT_DISTANCE))
        requires_sparse = bool(config.get("requires_sparse", False))

        created: list[str] = []
        for suffix in collections:
            collection_name = f"{workspace_name}_{suffix}" if suffix else workspace_name
            schema = CollectionSchema(
                name=collection_name,
                vector_size=vector_size,
                distance=distance,
                requires_sparse=requires_sparse,
            )
            await service.ensure_collection(schema)
            created.append(collection_name)
            if ctx:
                await ctx.debug(f"Ensured collection '{collection_name}' exists")

        response = {
            "workspace": workspace_name,
            "collections": created,
            "vector_size": vector_size,
            "distance": distance,
            "requires_sparse": requires_sparse,
            "created_at": _timestamp(),
        }
        if ctx:
            await ctx.info(
                f"Workspace '{workspace_name}' initialised with "
                f"{len(created)} collections"
            )
        return response

    @mcp.tool()
    async def manage_document_lifecycle(
        collection_name: str,
        lifecycle_action: str,
        filters: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Apply lifecycle actions to a collection using VectorStoreService."""

        service = await _get_vector_service(client_manager)
        action = lifecycle_action.lower()

        if action == "analyze":
            stats = await service.collection_stats(collection_name)
            documents, _ = await service.list_documents(collection_name, limit=25)
            if ctx:
                await ctx.info(f"Collected lifecycle analytics for '{collection_name}'")
            return {
                "collection": collection_name,
                "action": action,
                "stats": _normalize_stats(stats),
                "sample_documents": documents,
            }

        if action in {"cleanup", "archive"}:
            if not filters:
                msg = (
                    "Lifecycle cleanup/archive requires filters to "
                    "avoid deleting all documents"
                )
                raise ValueError(msg)
            await service.delete(collection_name, filters=filters)
            if ctx:
                await ctx.info(
                    f"Applied {action} action to '{collection_name}' "
                    f"using filters {filters}"
                )
            return {
                "collection": collection_name,
                "action": action,
                "filters": filters,
                "status": "deleted",
            }

        if action == "optimize":
            stats = await service.collection_stats(collection_name)
            params = stats.get("config", {}).get("params", {})
            vector_params = params.get("vectors", {})
            vector_size = int(vector_params.get("size") or service.embedding_dimension)
            distance = str(vector_params.get("distance") or _DEFAULT_DISTANCE)
            requires_sparse = bool(params.get("sparse_vectors"))
            await service.drop_collection(collection_name)
            schema = CollectionSchema(
                name=collection_name,
                vector_size=vector_size,
                distance=distance,
                requires_sparse=requires_sparse,
            )
            await service.ensure_collection(schema)
            if ctx:
                await ctx.info(
                    f"Recreated collection '{collection_name}' "
                    f"with vector size {vector_size}"
                )
            return {
                "collection": collection_name,
                "action": action,
                "vector_size": vector_size,
                "distance": distance,
                "requires_sparse": requires_sparse,
                "status": "recreated",
            }

        msg = f"Unknown lifecycle action: {lifecycle_action}"
        raise ValueError(msg)

    @mcp.tool()
    async def list_documents(
        collection_name: str,
        limit: int = 50,
        offset: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """List documents from a collection with pagination support."""

        service = await _get_vector_service(client_manager)
        documents, next_offset = await service.list_documents(
            collection_name,
            limit=limit,
            offset=offset,
        )
        if ctx:
            await ctx.debug(
                f"Fetched {len(documents)} documents from '{collection_name}' "
                f"(limit={limit})"
            )
        return {"documents": documents, "next_offset": next_offset}

    @mcp.tool()
    async def get_workspace_analytics(
        workspace_name: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Return lightweight analytics for collections grouped by workspace."""

        service = await _get_vector_service(client_manager)
        collections = await service.list_collections()

        def _matches_workspace(name: str) -> bool:
            if workspace_name is None:
                return True
            return name.startswith(f"{workspace_name}_") or name == workspace_name

        analytics: dict[str, Any] = {
            "generated_at": _timestamp(),
            "workspaces": {},
        }

        for collection in collections:
            if not _matches_workspace(collection):
                continue
            workspace = workspace_name or collection.split("_")[0]
            workspace_entry = analytics["workspaces"].setdefault(
                workspace,
                {"collections": []},
            )
            stats = await service.collection_stats(collection)
            workspace_entry["collections"].append(
                {
                    "name": collection,
                    "stats": _normalize_stats(stats),
                }
            )

        if ctx:
            await ctx.info(
                f"Workspace analytics generated for "
                f"{len(analytics['workspaces'])} workspaces"
            )
        return analytics
