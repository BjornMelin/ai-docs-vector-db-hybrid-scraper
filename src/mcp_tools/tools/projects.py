"""Project management tools for MCP server."""

import contextlib
import logging
from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from fastmcp import Context
else:
    # Use a protocol for testing to avoid FastMCP import issues
    from typing import Protocol

    class Context(Protocol):
        async def info(self, msg: str) -> None: ...
        async def debug(self, msg: str) -> None: ...
        async def warning(self, msg: str) -> None: ...
        async def error(self, msg: str) -> None: ...


from ...config.enums import SearchStrategy
from ...infrastructure.client_manager import ClientManager
from ..models.requests import ProjectRequest
from ..models.responses import SearchResult

logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):  # noqa: PLR0915
    """Register project management tools with the MCP server."""

    from ..models.responses import OperationStatus
    from ..models.responses import ProjectInfo

    @mcp.tool()
    async def create_project(
        request: ProjectRequest, ctx: Context = None
    ) -> ProjectInfo:
        """
        Create a new documentation project.

        Projects allow grouping related documents with shared configuration
        and quality settings.
        """
        project_id = str(uuid4())
        if ctx:
            await ctx.info(f"Creating project: {request.name} with ID: {project_id}")

        try:
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

            # Store project in persistent storage (single source of truth)
            project_storage = await client_manager.get_project_storage()
            await project_storage.save_project(project_id, project)

            if ctx:
                await ctx.debug(f"Project {project_id} stored in persistent storage")

            # Create collection with quality-based config
            vector_size = 1536 if request.quality_tier == "premium" else 384
            enable_hybrid = request.quality_tier in ["balanced", "premium"]

            if ctx:
                await ctx.debug(
                    f"Creating collection with vector size {vector_size}, hybrid: {enable_hybrid}"
                )

            try:
                await client_manager.qdrant_service.create_collection(
                    collection_name=project["collection"],
                    vector_size=vector_size,
                    distance="Cosine",
                    sparse_vector_name="sparse" if enable_hybrid else None,
                    enable_quantization=request.quality_tier != "premium",
                )
            except Exception as e:
                # Clean up the project if collection creation fails
                with contextlib.suppress(Exception):
                    await project_storage.delete_project(project_id)
                raise e

            # Add initial URLs if provided
            if request.urls:
                if ctx:
                    await ctx.debug(f"Processing {len(request.urls)} initial URLs")

                # Process URLs using 5-tier UnifiedBrowserManager via CrawlManager
                successful_count = 0
                crawl_manager = await client_manager.get_crawl_manager()
                for url in request.urls:
                    try:
                        # Scrape each URL using intelligent tier routing
                        crawl_result = await crawl_manager.scrape_url(url)
                        if (
                            crawl_result
                            and crawl_result.get("success")
                            and crawl_result.get("content")
                        ):
                            successful_count += 1
                    except Exception as e:
                        if ctx:
                            await ctx.warning(f"Failed to process URL {url}: {e}")
                        logger.warning(f"Failed to process URL {url}: {e}")

                project["urls"] = request.urls
                project["document_count"] = successful_count

                # Update persistent storage
                await project_storage.update_project(
                    project_id,
                    {
                        "urls": project["urls"],
                        "document_count": project["document_count"],
                    },
                )

                if ctx:
                    await ctx.debug(
                        f"Processed {successful_count}/{len(request.urls)} URLs successfully"
                    )

            if ctx:
                await ctx.info(
                    f"Project {request.name} created successfully with collection {project['collection']}"
                )

            return ProjectInfo(**project)

        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to create project {request.name}: {e}")
            logger.error(f"Failed to create project: {e}")
            raise

    @mcp.tool()
    async def list_projects(ctx: Context = None) -> list[ProjectInfo]:
        """
        List all documentation projects.

        Returns all projects with their metadata and statistics.
        """
        if ctx:
            await ctx.info("Retrieving list of all projects")

        try:
            # Get project storage instance
            project_storage = await client_manager.get_project_storage()
            projects_list = await project_storage.list_projects()

            if ctx:
                await ctx.debug(f"Found {len(projects_list)} projects in storage")

            projects = []
            for project in projects_list:
                # Get collection stats
                try:
                    info = await client_manager.qdrant_service.get_collection_info(
                        project["collection"]
                    )
                    project["vector_count"] = info.vectors_count
                    project["indexed_count"] = info.indexed_vectors_count
                    if ctx:
                        await ctx.debug(
                            f"Project {project['name']}: {info.vectors_count} vectors"
                        )
                except Exception as e:
                    project["vector_count"] = 0
                    project["indexed_count"] = 0
                    if ctx:
                        await ctx.warning(
                            f"Failed to get stats for project {project['name']}: {e}"
                        )

                projects.append(project)

            if ctx:
                await ctx.info(f"Successfully retrieved {len(projects)} projects")

            return [ProjectInfo(**p) for p in projects]

        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to list projects: {e}")
            logger.error(f"Failed to list projects: {e}")
            raise

    @mcp.tool()
    async def search_project(
        project_id: str,
        query: str,
        limit: int = 10,
        strategy: SearchStrategy = SearchStrategy.HYBRID,
        ctx: Context = None,
    ) -> list[SearchResult]:
        """
        Search within a specific project.

        Uses project-specific quality settings and collection.
        """
        if ctx:
            await ctx.info(
                f"Searching project {project_id} with query: {query[:50]}..."
            )

        try:
            project_storage = await client_manager.get_project_storage()
            project = await project_storage.get_project(project_id)
            if not project:
                if ctx:
                    await ctx.error(f"Project {project_id} not found")
                raise ValueError(f"Project {project_id} not found")

            if ctx:
                await ctx.debug(
                    f"Using collection: {project['collection']}, strategy: {strategy}"
                )

            # Generate embedding for query
            generate_sparse = strategy == SearchStrategy.HYBRID
            embedding_result = (
                await client_manager.embedding_manager.generate_embeddings(
                    [query], generate_sparse=generate_sparse
                )
            )

            query_vector = embedding_result["embeddings"][0]
            sparse_vector = None
            if embedding_result.get("sparse_embeddings"):
                sparse_vector = embedding_result["sparse_embeddings"][0]

            if ctx:
                await ctx.debug(
                    f"Generated embeddings: dense={len(query_vector)} dims, sparse={sparse_vector is not None}"
                )

            # Perform search
            results = await client_manager.qdrant_service.hybrid_search(
                collection_name=project["collection"],
                query_vector=query_vector,
                sparse_vector=sparse_vector
                if strategy == SearchStrategy.HYBRID
                else None,
                limit=limit,
                score_threshold=0.0,
                fusion_type="rrf" if strategy == SearchStrategy.HYBRID else None,
            )

            if ctx:
                await ctx.debug(f"Search returned {len(results)} results")

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

            if ctx:
                await ctx.info(
                    f"Project search completed: {len(search_results)} results for project {project_id}"
                )

            return search_results

        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to search project {project_id}: {e}")
            logger.error(f"Failed to search project: {e}")
            raise

    @mcp.tool()
    async def update_project(
        project_id: str,
        name: str | None = None,
        description: str | None = None,
        ctx: Context = None,
    ) -> ProjectInfo:
        """
        Update project metadata.

        Updates the name and/or description of an existing project.
        """
        if ctx:
            await ctx.info(f"Updating project {project_id}")

        try:
            project_storage = await client_manager.get_project_storage()
            project = await project_storage.get_project(project_id)
            if not project:
                if ctx:
                    await ctx.error(f"Project {project_id} not found")
                raise ValueError(f"Project {project_id} not found")

            updates = {}
            if name is not None:
                project["name"] = name
                updates["name"] = name
                if ctx:
                    await ctx.debug(f"Updated project name to: {name}")
            if description is not None:
                project["description"] = description
                updates["description"] = description
                if ctx:
                    await ctx.debug("Updated project description")

            if updates:
                await project_storage.update_project(project_id, updates)
                if ctx:
                    await ctx.debug(f"Persisted {len(updates)} updates to storage")

            if ctx:
                await ctx.info(f"Project {project_id} updated successfully")

            return ProjectInfo(**project)

        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to update project {project_id}: {e}")
            logger.error(f"Failed to update project: {e}")
            raise

    @mcp.tool()
    async def delete_project(
        project_id: str, delete_collection: bool = True, ctx: Context = None
    ) -> OperationStatus:
        """
        Delete a project and optionally its collection.

        Args:
            project_id: Project ID to delete
            delete_collection: Whether to delete the associated Qdrant collection

        Returns:
            Status message
        """
        if ctx:
            await ctx.info(
                f"Deleting project {project_id}, delete_collection={delete_collection}"
            )

        try:
            project_storage = await client_manager.get_project_storage()
            project = await project_storage.get_project(project_id)
            if not project:
                if ctx:
                    await ctx.error(f"Project {project_id} not found")
                raise ValueError(f"Project {project_id} not found")

            # Delete collection if requested
            if delete_collection:
                try:
                    await client_manager.qdrant_service.delete_collection(
                        project["collection"]
                    )
                    if ctx:
                        await ctx.debug(f"Deleted collection: {project['collection']}")
                except Exception as e:
                    if ctx:
                        await ctx.warning(
                            f"Failed to delete collection {project['collection']}: {e}"
                        )
                    logger.warning(
                        f"Failed to delete collection {project['collection']}: {e}"
                    )

            # Remove from persistent storage (single source of truth)
            await project_storage.delete_project(project_id)
            if ctx:
                await ctx.debug(f"Removed project {project_id} from persistent storage")

            if ctx:
                await ctx.info(f"Project {project_id} deleted successfully")

            return OperationStatus(
                status="deleted",
                details={
                    "project_id": project_id,
                    "collection_deleted": str(delete_collection),
                },
            )

        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to delete project {project_id}: {e}")
            logger.error(f"Failed to delete project: {e}")
            raise
