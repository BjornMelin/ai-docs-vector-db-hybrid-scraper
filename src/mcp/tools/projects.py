"""Project management tools for MCP server."""

import logging
from datetime import UTC
from datetime import datetime
from typing import Any
from uuid import uuid4

from ...config.enums import SearchStrategy
from ...infrastructure.client_manager import ClientManager
from ..models.requests import ProjectRequest
from ..models.responses import SearchResult

logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):  # noqa: PLR0915
    """Register project management tools with the MCP server."""

    @mcp.tool()
    async def create_project(request: ProjectRequest) -> dict[str, Any]:
        """
        Create a new documentation project.

        Projects allow grouping related documents with shared configuration
        and quality settings.
        """
        project_id = str(uuid4())
        project = {
            "id": project_id,
            "name": request.name,
            "description": request.description,
            "created_at": datetime.now(UTC).isoformat(),
            "quality_tier": request.quality_tier,
            "collection": f"project_{project_id}",
            "document_count": 0,
            "urls": [],
        }

        # Store project in both memory and persistent storage
        client_manager.projects[project_id] = project
        await client_manager.project_storage.save_project(project_id, project)

        # Create collection with quality-based config
        vector_size = 1536 if request.quality_tier == "premium" else 384
        enable_hybrid = request.quality_tier in ["balanced", "premium"]

        await client_manager.qdrant_service.create_collection(
            collection_name=project["collection"],
            vector_size=vector_size,
            distance="Cosine",
            sparse_vector_name="sparse" if enable_hybrid else None,
            enable_quantization=request.quality_tier != "premium",
        )

        # Add initial URLs if provided
        if request.urls:
            # Process URLs directly without calling add_documents_batch
            successful_count = 0
            for url in request.urls:
                try:
                    # Process each URL
                    crawl_result = await client_manager.crawl_manager.crawl_single(url)
                    if crawl_result and crawl_result.markdown:
                        successful_count += 1
                except Exception as e:
                    logger.warning(f"Failed to process URL {url}: {e}")

            project["urls"] = request.urls
            project["document_count"] = successful_count

            # Update persistent storage
            await client_manager.project_storage.update_project(
                project_id,
                {"urls": project["urls"], "document_count": project["document_count"]},
            )

        return project

    @mcp.tool()
    async def list_projects() -> list[dict[str, Any]]:
        """
        List all documentation projects.

        Returns all projects with their metadata and statistics.
        """
        projects = []
        for project in client_manager.projects.values():
            # Get collection stats
            try:
                info = await client_manager.qdrant_service.get_collection_info(
                    project["collection"]
                )
                project["vector_count"] = info.vectors_count
                project["indexed_count"] = info.indexed_vectors_count
            except Exception:
                project["vector_count"] = 0
                project["indexed_count"] = 0

            projects.append(project)

        return projects

    @mcp.tool()
    async def search_project(
        project_id: str,
        query: str,
        limit: int = 10,
        strategy: SearchStrategy = SearchStrategy.HYBRID,
    ) -> list[SearchResult]:
        """
        Search within a specific project.

        Uses project-specific quality settings and collection.
        """
        project = client_manager.projects.get(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        # Generate embedding for query
        generate_sparse = strategy == SearchStrategy.HYBRID
        embedding_result = await client_manager.embedding_manager.generate_embeddings(
            [query], generate_sparse=generate_sparse
        )

        query_vector = embedding_result["embeddings"][0]
        sparse_vector = None
        if embedding_result.get("sparse_embeddings"):
            sparse_vector = embedding_result["sparse_embeddings"][0]

        # Perform search
        results = await client_manager.qdrant_service.hybrid_search(
            collection_name=project["collection"],
            query_vector=query_vector,
            sparse_vector=sparse_vector if strategy == SearchStrategy.HYBRID else None,
            limit=limit,
            score_threshold=0.0,
            fusion_type="rrf" if strategy == SearchStrategy.HYBRID else None,
        )

        # Convert to search results
        search_results = []
        for point in results:
            result = SearchResult(
                id=str(point.id),
                content=point.payload.get("content", ""),
                score=point.score,
                url=point.payload.get("url"),
                title=point.payload.get("title"),
                metadata=point.payload,
            )
            search_results.append(result)

        return search_results

    @mcp.tool()
    async def update_project(
        project_id: str,
        name: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """
        Update project metadata.

        Updates the name and/or description of an existing project.
        """
        project = client_manager.projects.get(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        updates = {}
        if name is not None:
            project["name"] = name
            updates["name"] = name
        if description is not None:
            project["description"] = description
            updates["description"] = description

        if updates:
            await client_manager.project_storage.update_project(project_id, updates)

        return project

    @mcp.tool()
    async def delete_project(
        project_id: str, delete_collection: bool = True
    ) -> dict[str, str]:
        """
        Delete a project and optionally its collection.

        Args:
            project_id: Project ID to delete
            delete_collection: Whether to delete the associated Qdrant collection

        Returns:
            Status message
        """
        project = client_manager.projects.get(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        # Delete collection if requested
        if delete_collection:
            try:
                await client_manager.qdrant_service.delete_collection(
                    project["collection"]
                )
            except Exception as e:
                logger.warning(
                    f"Failed to delete collection {project['collection']}: {e}"
                )

        # Remove from in-memory storage
        del client_manager.projects[project_id]

        # Remove from persistent storage
        await client_manager.project_storage.delete_project(project_id)

        return {
            "status": "deleted",
            "project_id": project_id,
            "collection_deleted": str(delete_collection),
        }
