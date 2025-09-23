"""Project management tools for MCP server."""

import contextlib
import json
import logging
import subprocess
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import yaml

from src.mcp_tools.models.responses import OperationStatus, ProjectInfo


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


from src.config import SearchStrategy
from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.models.requests import ProjectRequest
from src.mcp_tools.models.responses import SearchResult


logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register project management tools with the MCP server."""

    # Helper functions for error handling
    async def _raise_project_not_found(project_id: str, ctx: Context = None) -> None:
        """Raise ValueError for project not found."""
        if ctx:
            await ctx.error(f"Project {project_id} not found")
        msg = f"Project {project_id} not found"
        raise ValueError(msg)

    async def _raise_unsupported_format(format_type: str, ctx: Context = None) -> None:
        """Raise ValueError for unsupported format."""
        if ctx:
            await ctx.error(f"Unsupported format: {format_type}")
        msg = f"Unsupported format: {format_type}. Use 'json' or 'yaml'"
        raise ValueError(msg)

    @mcp.tool()
    async def create_project(
        request: ProjectRequest, ctx: Context = None
    ) -> ProjectInfo:
        """Create a new documentation project.

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
                    f"Creating collection with vector size {vector_size}, "
                    f"hybrid: {enable_hybrid}"
                )

            try:
                await client_manager.qdrant_service.create_collection(
                    collection_name=project["collection"],
                    vector_size=vector_size,
                    distance="Cosine",
                    sparse_vector_name="sparse" if enable_hybrid else None,
                    enable_quantization=request.quality_tier != "premium",
                )
            except (AttributeError, ConnectionError, OSError):
                # Clean up the project if collection creation fails
                with contextlib.suppress(Exception):
                    await project_storage.delete_project(project_id)
                raise

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
                    except (subprocess.SubprocessError, OSError, TimeoutError) as e:
                        if ctx:
                            await ctx.warning(f"Failed to process URL {url}: {e}")
                        logger.warning(
                            f"Failed to process URL {url}: {e}"
                        )  # TODO: Convert f-string to logging format

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
                        f"Processed {successful_count}/{len(request.urls)} "
                        f"URLs successfully"
                    )

            if ctx:
                await ctx.info(
                    f"Project {request.name} created successfully with "
                    f"collection {project['collection']}"
                )

            return ProjectInfo(**project)

        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to create project {request.name}: {e}")
            logger.exception("Failed to create project")
            raise

    @mcp.tool()
    async def list_projects(ctx: Context = None) -> list[ProjectInfo]:
        """List all documentation projects.

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
                except (ValueError, ConnectionError, TimeoutError, RuntimeError) as e:
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
            logger.exception("Failed to list projects")
            raise

    @mcp.tool()
    async def search_project(
        project_id: str,
        query: str,
        limit: int = 10,
        strategy: SearchStrategy = SearchStrategy.HYBRID,
        ctx: Context = None,
    ) -> list[SearchResult]:
        """Search within a specific project.

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
                await _raise_project_not_found(project_id, ctx)

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
                    f"Generated embeddings: dense={len(query_vector)} dims, "
                    f"sparse={sparse_vector is not None}"
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
                    f"Project search completed: {len(search_results)} results "
                    f"for project {project_id}"
                )

        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to search project {project_id}: {e}")
            logger.exception("Failed to search project")
            raise
        else:
            return search_results

    @mcp.tool()
    async def update_project(
        project_id: str,
        name: str | None = None,
        description: str | None = None,
        ctx: Context = None,
    ) -> ProjectInfo:
        """Update project metadata.

        Updates the name and/or description of an existing project.
        """
        if ctx:
            await ctx.info(f"Updating project {project_id}")

        try:
            project_storage = await client_manager.get_project_storage()
            project = await project_storage.get_project(project_id)
            if not project:
                await _raise_project_not_found(project_id, ctx)

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
            logger.exception("Failed to update project")
            raise

    @mcp.tool()
    async def delete_project(
        project_id: str, delete_collection: bool = True, ctx: Context = None
    ) -> OperationStatus:
        """Delete a project and optionally its collection.

        Args:
            project_id: Project ID to delete
            delete_collection: Whether to delete the associated Qdrant collection
            ctx: MCP context for status updates

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
                await _raise_project_not_found(project_id, ctx)

            # Delete collection if requested
            if delete_collection:
                try:
                    await client_manager.qdrant_service.delete_collection(
                        project["collection"]
                    )
                    if ctx:
                        await ctx.debug(f"Deleted collection: {project['collection']}")
                except (ValueError, ConnectionError, TimeoutError, RuntimeError) as e:
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
            logger.exception("Failed to delete project")
            raise

    @mcp.tool()
    async def export_project(
        project_id: str, format_type: str = "json", ctx: Context = None
    ) -> dict:
        """Export project data in the specified format.

        Args:
            project_id: Project ID to export
            format_type: Export format ('json' or 'yaml')
            ctx: MCP context for status updates

        Returns:
            Dictionary containing status, format, and exported data

        """
        if ctx:
            await ctx.info(f"Exporting project {project_id} in {format_type} format")

        try:
            # Validate format
            if format_type not in ["json", "yaml"]:
                await _raise_unsupported_format(format_type, ctx)

            # Get project data
            project_storage = await client_manager.get_project_storage()
            project = await project_storage.get_project(project_id)
            if not project:
                await _raise_project_not_found(project_id, ctx)

            # Prepare project data for export
            export_data = {
                "id": project.get("id"),
                "name": project.get("name"),
                "description": project.get("description"),
                "quality_tier": project.get("quality_tier"),
                "collection_name": project.get("collection"),
                "created_at": project.get("created_at"),
                "urls": project.get("urls", []),
                "document_count": project.get("document_count", 0),
            }

            # Format the data
            if format_type == "json":
                formatted_data = json.dumps(export_data, indent=2)
                if ctx:
                    await ctx.debug("Exported project data as JSON")
            else:  # yaml
                formatted_data = yaml.dump(export_data, default_flow_style=False)
                if ctx:
                    await ctx.debug("Exported project data as YAML")

            if ctx:
                await ctx.info(f"Project {project_id} exported successfully")

        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to export project {project_id}: {e}")
            logger.exception("Failed to export project")
            raise
        else:
            return {
                "status": "exported",
                "format": format_type,
                "data": formatted_data,
            }

    @mcp.tool()
    async def add_project_urls(
        project_id: str, urls: list[str], ctx: Context = None
    ) -> dict:
        """Add new URLs to an existing project.

        Args:
            project_id: Project ID to add URLs to
            urls: List of URLs to add
            ctx: MCP context for logging

        Returns:
            Status dict with urls_added and total_urls

        """
        if ctx:
            await ctx.info(f"Adding {len(urls)} URLs to project {project_id}")

        try:
            project_storage = await client_manager.get_project_storage()
            project = await project_storage.get_project(project_id)
            if not project:
                await _raise_project_not_found(project_id, ctx)

            # Get existing URLs to avoid duplicates
            existing_urls = set(project.get("urls", []))
            new_urls = [url for url in urls if url not in existing_urls]

            if ctx:
                await ctx.debug(
                    f"Found {len(new_urls)} new URLs out of {len(urls)} provided"
                )

            # Process new URLs
            urls_added = 0
            crawling_service = client_manager.get_crawling_service()
            document_service = client_manager.get_document_service()

            for url in new_urls:
                try:
                    # Crawl the URL
                    crawl_result = await crawling_service.crawl_url(url)
                    if crawl_result and crawl_result.get("content"):
                        # Add document to collection
                        await document_service.add_document(
                            collection_name=project["collection"],
                            content=crawl_result["content"],
                            metadata={
                                "url": url,
                                "title": crawl_result.get("title", ""),
                                "project_id": project_id,
                            },
                        )
                        urls_added += 1
                        if ctx:
                            await ctx.debug(f"Successfully added URL: {url}")
                except (subprocess.SubprocessError, OSError, TimeoutError) as e:
                    if ctx:
                        await ctx.warning(f"Failed to process URL {url}: {e}")
                    logger.warning(
                        f"Failed to process URL {url}: {e}"
                    )  # TODO: Convert f-string to logging format

            # Update project URLs
            updated_urls = list(existing_urls) + new_urls
            await project_storage.update_project(project_id, {"urls": updated_urls})

            if ctx:
                await ctx.info(
                    f"Added {urls_added} URLs to project {project_id}. "
                    f"Total URLs: {len(updated_urls)}"
                )

            return {
                "status": "urls_added",
                "urls_added": urls_added,
                "total_urls": len(updated_urls),
            }

        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to add URLs to project {project_id}: {e}")
            logger.exception("Failed to add URLs to project")
            raise

    @mcp.tool()
    async def get_project_details(project_id: str, ctx: Context = None) -> dict:
        """Get detailed information about a specific project.

        Args:
            project_id: The ID of the project to retrieve
            ctx: Optional context for logging

        Returns:
            Dictionary containing project details

        Raises:
            ValueError: If project not found

        """
        if ctx:
            await ctx.info(f"Getting details for project {project_id}")

        try:
            project_storage = await client_manager.get_project_storage()
            project = await project_storage.get_project(project_id)

            if not project:
                await _raise_project_not_found(project_id, ctx)

            # Get collection stats if available
            try:
                info = await client_manager.qdrant_service.get_collection_info(
                    project.get("collection", f"project_{project_id}")
                )
                project["vector_count"] = info.vectors_count
                project["indexed_count"] = info.indexed_vectors_count
                if ctx:
                    await ctx.debug(
                        f"Project {project.get('name', project_id)}: "
                        f"{info.vectors_count} vectors"
                    )
            except (ValueError, ConnectionError, TimeoutError, RuntimeError) as e:
                project["vector_count"] = 0
                project["indexed_count"] = 0
                if ctx:
                    await ctx.warning(
                        f"Failed to get collection stats for project {project_id}: {e}"
                    )

            if ctx:
                await ctx.info(
                    f"Successfully retrieved details for project {project_id}"
                )

        except ValueError:
            # Re-raise ValueError as is
            raise
        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to get project details for {project_id}: {e}")
            logger.exception("Failed to get project details")
            raise
        else:
            return {"project": project}
