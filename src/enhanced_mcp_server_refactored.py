#!/usr/bin/env python3
"""Enhanced AI Documentation Vector DB MCP Server - Refactored with Service Layer.

This enhanced version provides a unified interface with additional capabilities
for documentation management, now using the new service layer architecture.

Built with FastMCP 2.0 and leveraging MCP composition patterns.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel
from pydantic import Field
from rich.console import Console

from .chunking import CodeAwareChunker
from .chunking import get_chunker
from .error_handling import handle_mcp_errors
from .security import SecurityValidator
from .security import validate_startup_security

# New service layer imports
from .services import APIConfig
from .services import CrawlManager
from .services import EmbeddingManager
from .services import QdrantService
from .services import QualityTier

# Load environment variables
load_dotenv()

# Initialize console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global configuration and services
config: APIConfig | None = None
qdrant_service: QdrantService | None = None
embedding_manager: EmbeddingManager | None = None
crawl_manager: CrawlManager | None = None
security_validator: SecurityValidator | None = None
chunker: CodeAwareChunker | None = None

# Validate security requirements at startup
try:
    ENV_VARS = validate_startup_security()
    logger.info("✅ Enhanced MCP Server security validation complete")
except Exception as e:
    logger.error(f"❌ Enhanced MCP Server startup failed: {e}")
    raise


# ========== Configuration Models ==========


class ProjectConfig(BaseModel):
    """Configuration for a documentation project."""

    name: str = Field(..., description="Project name")
    urls: list[str] = Field(..., description="List of URLs to scrape")
    collection_name: str = Field(..., description="Qdrant collection name")
    chunk_size: int = Field(default=1600, description="Target chunk size")
    chunk_overlap: int = Field(default=200, description="Chunk overlap")
    quality_tier: str = Field(default="balanced", description="Embedding quality tier")
    tags: list[str] = Field(default_factory=list, description="Project tags")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation time"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last update time"
    )


class SearchConfig(BaseModel):
    """Advanced search configuration."""

    query: str = Field(..., description="Search query")
    projects: list[str] | None = Field(None, description="Filter by projects")
    tags: list[str] | None = Field(None, description="Filter by tags")
    limit: int = Field(default=10, ge=1, le=100, description="Number of results")
    rerank: bool = Field(default=True, description="Enable reranking")
    include_metadata: bool = Field(default=True, description="Include metadata")


# ========== Server Initialization ==========

# Create the enhanced FastMCP server
mcp = FastMCP(
    name="Enhanced AI Docs Vector DB",
    instructions="""
    Enhanced documentation management server with project-based organization.
    
    Key features:
    - Project-based documentation management
    - Advanced chunking strategies (code-aware, AST-based)
    - Multi-project search with tag filtering
    - Automatic content updates and versioning
    - Cost estimation and optimization
    
    Tools:
    - create_project: Create a new documentation project
    - add_urls_to_project: Add URLs to an existing project
    - search_projects: Search across multiple projects
    - update_project: Update project configuration
    - list_projects: List all projects
    - get_project_info: Get detailed project information
    - estimate_costs: Estimate costs for operations
    - optimize_project: Optimize project for cost/performance
    """,
)

# Project storage
PROJECTS_FILE = Path(".projects.json")
projects: dict[str, ProjectConfig] = {}


# ========== Service Initialization ==========


async def initialize_services():
    """Initialize all services with proper configuration."""
    global \
        config, \
        qdrant_service, \
        embedding_manager, \
        crawl_manager, \
        security_validator, \
        chunker

    if config is None:
        # Create configuration
        config = APIConfig(
            openai_api_key=ENV_VARS.get("OPENAI_API_KEY"),
            firecrawl_api_key=ENV_VARS.get("FIRECRAWL_API_KEY"),
            qdrant_url=ENV_VARS.get("QDRANT_URL", "http://localhost:6333"),
            enable_local_embeddings=True,
            preferred_crawl_provider="firecrawl"
            if ENV_VARS.get("FIRECRAWL_API_KEY")
            else "crawl4ai",
        )

        # Initialize services
        qdrant_service = QdrantService(config)
        embedding_manager = EmbeddingManager(config)
        crawl_manager = CrawlManager(config)
        security_validator = SecurityValidator()

        # Initialize all services
        await qdrant_service.initialize()
        await embedding_manager.initialize()
        await crawl_manager.initialize()

        # Initialize chunker
        chunker = get_chunker("enhanced")

        logger.info("✅ All services initialized successfully")


async def cleanup_services():
    """Cleanup all services."""
    if qdrant_service:
        await qdrant_service.cleanup()
    if embedding_manager:
        await embedding_manager.cleanup()
    if crawl_manager:
        await crawl_manager.cleanup()
    logger.info("✅ All services cleaned up")


# ========== Helper Functions ==========


def load_projects():
    """Load projects from disk."""
    global projects
    if PROJECTS_FILE.exists():
        with open(PROJECTS_FILE) as f:
            data = json.load(f)
            projects = {name: ProjectConfig(**config) for name, config in data.items()}
    logger.info(f"📁 Loaded {len(projects)} projects")


def save_projects():
    """Save projects to disk."""
    with open(PROJECTS_FILE, "w") as f:
        data = {
            name: config.model_dump(mode="json") for name, config in projects.items()
        }
        json.dump(data, f, indent=2, default=str)
    logger.info(f"💾 Saved {len(projects)} projects")


async def index_urls(urls: list[str], project: ProjectConfig) -> dict[str, Any]:
    """Index multiple URLs for a project."""
    results = {
        "successful": 0,
        "failed": 0,
        "errors": [],
        "chunks_indexed": 0,
    }

    for url in urls:
        try:
            # Validate URL
            if not security_validator.validate_url(url):
                results["failed"] += 1
                results["errors"].append(f"Invalid URL: {url}")
                continue

            # Scrape URL
            scraped = await crawl_manager.scrape_url(url)
            if not scraped["success"]:
                results["failed"] += 1
                results["errors"].append(
                    f"Failed to scrape {url}: {scraped.get('error')}"
                )
                continue

            # Chunk content
            content = scraped.get("content", "")
            chunks = chunker.chunk_text(
                content,
                chunk_size=project.chunk_size,
                chunk_overlap=project.chunk_overlap,
            )

            # Generate embeddings
            quality_tier = QualityTier[project.quality_tier.upper()]
            embeddings = await embedding_manager.generate_embeddings(
                chunks,
                quality_tier=quality_tier,
            )

            # Prepare points
            points = []
            for i, (chunk, embedding) in enumerate(
                zip(chunks, embeddings, strict=False)
            ):
                points.append(
                    {
                        "id": f"{hash(url)}_{i}",
                        "vector": embedding,
                        "payload": {
                            "project": project.name,
                            "url": url,
                            "content": chunk,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "tags": project.tags,
                            "indexed_at": datetime.now().isoformat(),
                        },
                    }
                )

            # Index in Qdrant
            success = await qdrant_service.upsert_points(
                collection_name=project.collection_name,
                points=points,
            )

            if success:
                results["successful"] += 1
                results["chunks_indexed"] += len(points)
            else:
                results["failed"] += 1
                results["errors"].append(f"Failed to index {url}")

        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Error processing {url}: {e!s}")
            logger.error(f"Error indexing {url}: {e}")

    return results


# ========== MCP Tool Implementations ==========


@mcp.tool()
@handle_mcp_errors
async def create_project(config: ProjectConfig) -> dict[str, Any]:
    """Create a new documentation project.

    Args:
        config: Project configuration

    Returns:
        Project creation status
    """
    await initialize_services()

    # Check if project exists
    if config.name in projects:
        return {
            "success": False,
            "error": f"Project '{config.name}' already exists",
        }

    # Validate URLs
    for url in config.urls:
        if not security_validator.validate_url(url):
            return {
                "success": False,
                "error": f"Invalid URL: {url}",
            }

    # Create collection
    embedding_size = 1536 if config.quality_tier != "fast" else 384
    await qdrant_service.create_collection(
        collection_name=config.collection_name,
        vector_size=embedding_size,
        enable_quantization=True,  # Enable by default for storage efficiency
    )

    # Save project
    projects[config.name] = config
    save_projects()

    # Index URLs
    results = await index_urls(config.urls, config)

    return {
        "success": True,
        "project": config.name,
        "collection": config.collection_name,
        "indexing_results": results,
    }


@mcp.tool()
@handle_mcp_errors
async def add_urls_to_project(
    project_name: str,
    urls: list[str],
) -> dict[str, Any]:
    """Add URLs to an existing project.

    Args:
        project_name: Name of the project
        urls: List of URLs to add

    Returns:
        URL addition status
    """
    await initialize_services()

    if project_name not in projects:
        return {
            "success": False,
            "error": f"Project '{project_name}' not found",
        }

    project = projects[project_name]

    # Filter new URLs
    new_urls = [url for url in urls if url not in project.urls]
    if not new_urls:
        return {
            "success": True,
            "message": "All URLs already in project",
        }

    # Index new URLs
    results = await index_urls(new_urls, project)

    # Update project
    project.urls.extend(new_urls)
    project.updated_at = datetime.now()
    save_projects()

    return {
        "success": True,
        "project": project_name,
        "new_urls": len(new_urls),
        "indexing_results": results,
    }


@mcp.tool()
@handle_mcp_errors
async def search_projects(config: SearchConfig) -> dict[str, Any]:
    """Search across multiple projects with advanced filtering.

    Args:
        config: Search configuration

    Returns:
        Search results from multiple projects
    """
    await initialize_services()

    # Generate query embedding
    embeddings = await embedding_manager.generate_embeddings(
        [config.query],
        quality_tier=QualityTier.BALANCED,
    )
    query_vector = embeddings[0]

    # Determine which collections to search
    collections_to_search = []
    if config.projects:
        # Search specific projects
        for project_name in config.projects:
            if project_name in projects:
                collections_to_search.append(projects[project_name].collection_name)
    else:
        # Search all projects
        collections_to_search = [p.collection_name for p in projects.values()]

    # Search each collection
    all_results = []
    for collection in collections_to_search:
        try:
            results = await qdrant_service.hybrid_search(
                collection_name=collection,
                query_vector=query_vector,
                limit=config.limit * 2,  # Get extra for filtering
            )

            # Apply tag filtering if specified
            if config.tags:
                results = [
                    r
                    for r in results
                    if any(tag in r["payload"].get("tags", []) for tag in config.tags)
                ]

            all_results.extend(results)
        except Exception as e:
            logger.warning(f"Failed to search collection {collection}: {e}")

    # Sort by score and limit
    all_results.sort(key=lambda x: x["score"], reverse=True)
    all_results = all_results[: config.limit]

    # Optionally rerank results
    if config.rerank and len(all_results) > 1:
        # TODO: Implement reranking with BGE or similar
        pass

    # Format results
    formatted_results = []
    for result in all_results:
        formatted = {
            "score": result["score"],
            "content": result["payload"]["content"],
            "url": result["payload"]["url"],
            "project": result["payload"]["project"],
        }

        if config.include_metadata:
            formatted["metadata"] = {
                "tags": result["payload"].get("tags", []),
                "chunk_index": result["payload"]["chunk_index"],
                "total_chunks": result["payload"]["total_chunks"],
                "indexed_at": result["payload"]["indexed_at"],
            }

        formatted_results.append(formatted)

    return {
        "results": formatted_results,
        "total": len(formatted_results),
        "query": config.query,
        "searched_collections": len(collections_to_search),
    }


@mcp.tool()
@handle_mcp_errors
async def list_projects() -> dict[str, Any]:
    """List all documentation projects.

    Returns:
        List of projects with metadata
    """
    await initialize_services()

    project_list = []
    for name, project in projects.items():
        # Get collection info
        try:
            info = await qdrant_service.get_collection_info(project.collection_name)
            points_count = info.get("points_count", 0)
        except Exception:
            points_count = 0

        project_list.append(
            {
                "name": name,
                "urls": len(project.urls),
                "collection": project.collection_name,
                "points": points_count,
                "tags": project.tags,
                "quality_tier": project.quality_tier,
                "created": project.created_at.isoformat(),
                "updated": project.updated_at.isoformat(),
            }
        )

    return {
        "projects": project_list,
        "total": len(project_list),
    }


@mcp.tool()
@handle_mcp_errors
async def get_project_info(name: str) -> dict[str, Any]:
    """Get detailed information about a project.

    Args:
        name: Project name

    Returns:
        Detailed project information
    """
    await initialize_services()

    if name not in projects:
        return {
            "success": False,
            "error": f"Project '{name}' not found",
        }

    project = projects[name]

    # Get collection info
    collection_info = await qdrant_service.get_collection_info(project.collection_name)

    # Estimate costs
    costs = embedding_manager.estimate_cost(
        ["sample"] * collection_info.get("points_count", 0)
    )

    return {
        "project": project.model_dump(mode="json"),
        "collection_info": collection_info,
        "estimated_costs": costs,
    }


@mcp.tool()
@handle_mcp_errors
async def estimate_costs(
    texts: list[str],
    quality_tier: str = "balanced",
) -> dict[str, Any]:
    """Estimate costs for embedding generation.

    Args:
        texts: List of texts to estimate
        quality_tier: Quality tier to use

    Returns:
        Cost estimation breakdown
    """
    await initialize_services()

    costs = embedding_manager.estimate_cost(texts)

    # Add recommendations
    recommendations = []
    total_chars = sum(len(text) for text in texts)

    if total_chars > 100000 and quality_tier != "fast":
        recommendations.append(
            "Consider using 'fast' quality tier for large datasets to reduce costs"
        )

    if len(texts) < 100 and quality_tier != "best":
        recommendations.append(
            "For small datasets, 'best' quality tier provides superior results at minimal extra cost"
        )

    return {
        "costs": costs,
        "quality_tier": quality_tier,
        "total_texts": len(texts),
        "total_characters": total_chars,
        "recommendations": recommendations,
    }


@mcp.tool()
@handle_mcp_errors
async def optimize_project(name: str) -> dict[str, Any]:
    """Optimize a project for cost and performance.

    Args:
        name: Project name

    Returns:
        Optimization results and recommendations
    """
    await initialize_services()

    if name not in projects:
        return {
            "success": False,
            "error": f"Project '{name}' not found",
        }

    project = projects[name]
    recommendations = []
    actions_taken = []

    # Get collection info
    info = await qdrant_service.get_collection_info(project.collection_name)

    # Check if quantization is enabled
    if not info.get("config", {}).get("quantization_config"):
        recommendations.append(
            "Enable scalar quantization to reduce storage by 75% with minimal accuracy loss"
        )

    # Check chunk size optimization
    if project.chunk_size > 2000:
        recommendations.append(
            f"Consider reducing chunk size from {project.chunk_size} to 1600 for better search granularity"
        )

    # Check quality tier vs size
    points_count = info.get("points_count", 0)
    if points_count > 10000 and project.quality_tier == "best":
        recommendations.append(
            "For large collections, consider 'balanced' quality tier to reduce costs by 50%"
        )

    # Estimate potential savings
    current_costs = embedding_manager.estimate_cost(["sample"] * points_count)

    return {
        "project": name,
        "current_points": points_count,
        "current_costs": current_costs,
        "recommendations": recommendations,
        "actions_taken": actions_taken,
    }


# ========== Server Lifecycle ==========


@mcp.before_startup()
async def before_startup():
    """Initialize server and load projects."""
    load_projects()
    await initialize_services()
    logger.info("🚀 Enhanced AI Docs Vector DB Server initialized")


@mcp.after_startup()
async def after_startup():
    """Log startup information."""
    logger.info(f"📊 Loaded {len(projects)} projects")
    logger.info(f"🔧 Providers: {list(embedding_manager.providers.keys())}")
    logger.info("✨ Enhanced server ready!")


@mcp.before_shutdown()
async def before_shutdown():
    """Cleanup before shutdown."""
    save_projects()
    await cleanup_services()
    logger.info("👋 Enhanced server shutdown complete")


# ========== Main Entry Point ==========

if __name__ == "__main__":
    import sys

    try:
        asyncio.run(mcp.run())
    except KeyboardInterrupt:
        logger.info("⌨️ Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Server crashed: {e}")
        sys.exit(1)
