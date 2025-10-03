"""Project management MCP tools backed by the consolidated vector service."""

# pylint: disable=too-many-statements  # tool registration defines closures to reuse the shared client manager.

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastmcp import Context

from src.config import SearchStrategy
from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.models.requests import ProjectRequest
from src.mcp_tools.models.responses import OperationStatus, ProjectInfo, SearchResult
from src.mcp_tools.tools._shared import ensure_vector_service, match_to_result
from src.services.vector_db.types import CollectionSchema


logger = logging.getLogger(__name__)

_TIER_VECTOR_SIZE = {
    "economy": 384,
    "balanced": 768,
    "premium": 1536,
}


def _timestamp() -> str:
    return datetime.now(tz=UTC).isoformat()


def _collection_schema(collection: str, tier: str) -> CollectionSchema:
    vector_size = _TIER_VECTOR_SIZE.get(tier, _TIER_VECTOR_SIZE["balanced"])
    return CollectionSchema(name=collection, vector_size=vector_size, distance="cosine")


def register_tools(mcp, client_manager: ClientManager) -> None:
    """Register project management tools with the MCP server."""

    @mcp.tool()
    async def create_project(
        request: ProjectRequest, ctx: Context | None = None
    ) -> ProjectInfo:
        """Create a new project with a dedicated vector collection."""

        project_storage = await client_manager.get_project_storage()  # type: ignore[attr-defined]
        vector_service = await ensure_vector_service(client_manager)

        project_id = str(uuid4())
        collection_name = f"project_{project_id}"
        schema = _collection_schema(collection_name, request.quality_tier)
        await vector_service.ensure_collection(schema)

        project_record: dict[str, Any] = {
            "id": project_id,
            "name": request.name,
            "description": request.description,
            "quality_tier": request.quality_tier,
            "collection": collection_name,
            "urls": request.urls or [],
            "created_at": _timestamp(),
        }
        await project_storage.save_project(project_id, project_record)

        if ctx:
            await ctx.info(
                f"Created project '{request.name}' with collection '{collection_name}'"
            )

        return ProjectInfo(**project_record)

    @mcp.tool()
    async def list_projects(ctx: Context | None = None) -> list[ProjectInfo]:
        """Return all projects with lightweight collection statistics."""

        project_storage = await client_manager.get_project_storage()  # type: ignore[attr-defined]
        vector_service = await ensure_vector_service(client_manager)

        projects = await project_storage.list_projects()
        enriched: list[ProjectInfo] = []
        for project in projects:
            collection = project.get("collection")
            if collection:
                try:
                    stats = await vector_service.collection_stats(collection)
                    project.setdefault("stats", dict(stats))
                except Exception as exc:  # pragma: no cover - service errors
                    logger.warning(
                        "Failed to fetch stats for collection %s: %s", collection, exc
                    )
                    if ctx:
                        await ctx.warning(
                            f"Collection stats unavailable for '{collection}': {exc}"
                        )
            enriched.append(ProjectInfo(**project))
        if ctx:
            await ctx.info(f"Loaded {len(enriched)} projects from storage")
        return enriched

    @mcp.tool()
    async def update_project(
        project_id: str,
        name: str | None = None,
        description: str | None = None,
        ctx: Context | None = None,
    ) -> ProjectInfo:
        """Update basic project metadata."""

        project_storage = await client_manager.get_project_storage()  # type: ignore[attr-defined]
        project = await project_storage.get_project(project_id)
        if not project:
            msg = f"Project {project_id} not found"
            raise ValueError(msg)

        updates: dict[str, Any] = {}
        if name is not None:
            project["name"] = name
            updates["name"] = name
        if description is not None:
            project["description"] = description
            updates["description"] = description
        if updates:
            await project_storage.update_project(project_id, updates)
            if ctx:
                changed_fields = ", ".join(updates.keys())
                await ctx.info(
                    f"Updated project {project_id} with fields: {changed_fields}"
                )
        return ProjectInfo(**project)

    @mcp.tool()
    async def delete_project(
        project_id: str,
        delete_collection: bool = True,
        ctx: Context | None = None,
    ) -> OperationStatus:
        """Delete a project record and optionally drop its vector collection."""

        project_storage = await client_manager.get_project_storage()  # type: ignore[attr-defined]
        project = await project_storage.get_project(project_id)
        if not project:
            msg = f"Project {project_id} not found"
            raise ValueError(msg)

        collection_name = project.get("collection")
        if delete_collection and collection_name:
            vector_service = await ensure_vector_service(client_manager)
            await vector_service.drop_collection(collection_name)
            if ctx:
                message = (
                    f"Dropped vector collection '{collection_name}' for project "
                    f"{project_id}"
                )
                await ctx.debug(message)

        await project_storage.delete_project(project_id)
        if ctx:
            await ctx.info(f"Deleted project {project_id}")

        return OperationStatus(
            status="deleted",
            details={"project_id": project_id, "collection_deleted": delete_collection},
        )

    @mcp.tool()
    async def search_project(
        project_id: str,
        query: str,
        limit: int = 10,
        strategy: SearchStrategy = SearchStrategy.HYBRID,
        ctx: Context | None = None,
    ) -> list[SearchResult]:
        """Search within a project's dedicated collection."""

        project_storage = await client_manager.get_project_storage()  # type: ignore[attr-defined]
        project = await project_storage.get_project(project_id)
        if not project:
            msg = f"Project {project_id} not found"
            raise ValueError(msg)

        collection = project.get("collection")
        if not collection:
            msg = f"Project {project_id} is missing a collection reference"
            raise ValueError(msg)

        vector_service = await ensure_vector_service(client_manager)
        matches = await vector_service.search_documents(
            collection,
            query,
            limit=limit,
        )

        results = [
            match_to_result(match, include_metadata=True) for match in matches[:limit]
        ]
        if ctx:
            summary = (
                f"Project search returned {len(results)} results for project "
                f"{project_id}"
            )
            await ctx.info(summary)
        return results
