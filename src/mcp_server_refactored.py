#!/usr/bin/env python3
"""AI Documentation Vector DB MCP Server - Refactored with Service Layer.

A unified Model Context Protocol server that combines web scraping (Crawl4AI + Firecrawl),
vector database management (Qdrant), and intelligent search capabilities.

Built with FastMCP 2.0 for efficient, scalable MCP server implementation.
Now uses the new service layer for better abstraction and testability.
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel
from pydantic import Field
from qdrant_client.models import Distance
from rich.console import Console

from .error_handling import ConfigurationError
from .error_handling import ExternalServiceError
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

# Initialize console for rich output
console = Console()

# Configure logging
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

# Environment variables
ENV_VARS = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY"),
    "QDRANT_URL": os.getenv("QDRANT_URL", "http://localhost:6333"),
    "CACHE_DIR": os.getenv("CACHE_DIR", ".cache/scraping"),
}

# ========== Pydantic Models ==========


class SearchParams(BaseModel):
    """Parameters for search operations."""

    query: str = Field(..., description="Search query")
    collection: str = Field(default="documentation", description="Collection to search")
    limit: int = Field(default=5, ge=1, le=100, description="Number of results")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Score threshold")


class CollectionParams(BaseModel):
    """Parameters for collection operations."""

    name: str = Field(..., description="Collection name")
    vector_size: int = Field(default=1536, ge=1, description="Vector dimension size")
    distance: str = Field(default="cosine", description="Distance metric")


# ========== Main MCP Server ==========

# Create the FastMCP server instance
mcp = FastMCP(
    name="AI Docs Vector DB",
    instructions="""
    This MCP server provides comprehensive documentation scraping and vector search capabilities.

    Available tools:
    - scrape_url: Scrape and index documentation from any URL
    - search: Search indexed documentation using semantic search
    - list_collections: List all vector database collections
    - create_collection: Create a new collection
    - delete_collection: Delete a collection
    - get_collection_info: Get detailed information about a collection
    - clear_cache: Clear scraping cache

    The server supports both Crawl4AI (bulk scraping) and Firecrawl (on-demand) backends,
    with intelligent chunking strategies and hybrid search capabilities.
    """,
)


async def initialize_services():
    """Initialize all services with proper configuration."""
    global config, qdrant_service, embedding_manager, crawl_manager, security_validator

    if config is None:
        # Validate environment
        if not ENV_VARS["OPENAI_API_KEY"]:
            raise ConfigurationError("OPENAI_API_KEY environment variable is required")

        # Create configuration
        config = APIConfig(
            openai_api_key=ENV_VARS["OPENAI_API_KEY"],
            firecrawl_api_key=ENV_VARS.get("FIRECRAWL_API_KEY"),
            qdrant_url=ENV_VARS["QDRANT_URL"],
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


# ========== MCP Tool Implementations ==========


@mcp.tool()
@handle_mcp_errors
async def search(params: SearchParams) -> dict[str, Any]:
    """Search indexed documentation using semantic search with reranking.

    Args:
        params: Search parameters including query and collection

    Returns:
        Search results with relevant documentation chunks
    """
    # Initialize services if needed
    await initialize_services()

    # Generate query embedding
    embeddings = await embedding_manager.generate_embeddings(
        [params.query],
        quality_tier=QualityTier.BALANCED,
    )
    query_vector = embeddings[0]

    # Perform hybrid search
    results = await qdrant_service.hybrid_search(
        collection_name=params.collection,
        query_vector=query_vector,
        limit=params.limit,
    )

    # Filter by threshold
    filtered_results = [r for r in results if r["score"] >= params.threshold]

    return {
        "results": filtered_results,
        "total": len(filtered_results),
        "query": params.query,
        "collection": params.collection,
    }


@mcp.tool()
@handle_mcp_errors
async def scrape_url(url: str, collection: str = "documentation") -> dict[str, Any]:
    """Scrape and index documentation from a URL.

    Args:
        url: URL to scrape
        collection: Collection to store documents in

    Returns:
        Scraping results with indexed document count
    """
    # Initialize services if needed
    await initialize_services()

    # Validate URL
    if not security_validator.validate_url(url):
        raise ValueError(f"Invalid or disallowed URL: {url}")

    # Check cache
    cache_dir = Path(ENV_VARS["CACHE_DIR"])
    cache_file = cache_dir / f"{hash(url)}.json"

    if cache_file.exists():
        logger.info(f"📁 Using cached result for {url}")
        with open(cache_file) as f:
            scraped_data = json.load(f)
    else:
        # Scrape URL
        scraped_data = await crawl_manager.scrape_url(url)

        if not scraped_data["success"]:
            raise ExternalServiceError(
                f"Failed to scrape {url}: {scraped_data.get('error', 'Unknown error')}"
            )

        # Cache result
        cache_dir.mkdir(exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(scraped_data, f)

    # Process content
    content = scraped_data.get("content", "")
    if not content:
        return {
            "success": False,
            "error": "No content extracted from URL",
            "url": url,
        }

    # TODO: Add chunking logic here
    chunks = [content]  # Simplified for now

    # Generate embeddings
    embeddings = await embedding_manager.generate_embeddings(
        chunks,
        quality_tier=QualityTier.FAST,  # Use fast tier for bulk
    )

    # Ensure collection exists
    await qdrant_service.create_collection(
        collection_name=collection,
        vector_size=len(embeddings[0]) if embeddings else 1536,
        distance=Distance.COSINE,
    )

    # Index documents
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
        points.append(
            {
                "id": f"{hash(url)}_{i}",
                "vector": embedding,
                "payload": {
                    "url": url,
                    "content": chunk[:1000],  # Store preview
                    "full_content": chunk,
                    "timestamp": time.time(),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            }
        )

    success = await qdrant_service.upsert_points(
        collection_name=collection,
        points=points,
    )

    return {
        "success": success,
        "url": url,
        "chunks_indexed": len(points),
        "collection": collection,
    }


@mcp.tool()
@handle_mcp_errors
async def list_collections() -> dict[str, Any]:
    """List all vector database collections.

    Returns:
        List of collections with metadata
    """
    # Initialize services if needed
    await initialize_services()

    collections = await qdrant_service.list_collections()

    result = []
    for collection in collections:
        info = await qdrant_service.get_collection_info(collection["name"])
        result.append(
            {
                "name": collection["name"],
                "vectors_count": info.get("vectors_count", 0),
                "points_count": info.get("points_count", 0),
                "status": info.get("status", "unknown"),
            }
        )

    return {
        "collections": result,
        "total": len(result),
    }


@mcp.tool()
@handle_mcp_errors
async def create_collection(params: CollectionParams) -> dict[str, Any]:
    """Create a new collection in the vector database.

    Args:
        params: Collection parameters

    Returns:
        Creation status
    """
    # Initialize services if needed
    await initialize_services()

    # Validate collection name
    if not security_validator.validate_collection_name(params.name):
        raise ValueError(f"Invalid collection name: {params.name}")

    success = await qdrant_service.create_collection(
        collection_name=params.name,
        vector_size=params.vector_size,
        distance=params.distance.upper(),
    )

    return {
        "success": success,
        "collection": params.name,
        "vector_size": params.vector_size,
        "distance": params.distance,
    }


@mcp.tool()
@handle_mcp_errors
async def delete_collection(name: str) -> dict[str, Any]:
    """Delete a collection from the vector database.

    Args:
        name: Collection name to delete

    Returns:
        Deletion status
    """
    # Initialize services if needed
    await initialize_services()

    success = await qdrant_service.delete_collection(name)

    return {
        "success": success,
        "collection": name,
        "message": "Collection deleted successfully"
        if success
        else "Failed to delete collection",
    }


@mcp.tool()
@handle_mcp_errors
async def get_collection_info(name: str) -> dict[str, Any]:
    """Get detailed information about a collection.

    Args:
        name: Collection name

    Returns:
        Collection information
    """
    # Initialize services if needed
    await initialize_services()

    info = await qdrant_service.get_collection_info(name)

    return {
        "name": name,
        "info": info,
    }


@mcp.tool()
@handle_mcp_errors
async def clear_cache() -> dict[str, Any]:
    """Clear the scraping cache.

    Returns:
        Cache clearing status
    """
    cache_dir = Path(ENV_VARS["CACHE_DIR"])

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.info("🗑️ Cache cleared successfully")
        return {
            "success": True,
            "message": "Cache cleared successfully",
        }
    else:
        return {
            "success": True,
            "message": "Cache directory does not exist",
        }


# ========== Server Lifecycle ==========


@mcp.before_startup()
async def before_startup():
    """Run security validations and setup before server starts."""
    validate_startup_security()
    logger.info("🔒 Security validation passed")

    # Pre-initialize services
    await initialize_services()


@mcp.after_startup()
async def after_startup():
    """Log startup completion."""
    logger.info("🚀 AI Docs Vector DB MCP Server started successfully!")
    logger.info(f"📊 Qdrant URL: {ENV_VARS['QDRANT_URL']}")
    logger.info(f"🔑 OpenAI: {'✅' if ENV_VARS['OPENAI_API_KEY'] else '❌'}")
    logger.info(f"🔥 Firecrawl: {'✅' if ENV_VARS.get('FIRECRAWL_API_KEY') else '❌'}")


@mcp.before_shutdown()
async def before_shutdown():
    """Cleanup resources before shutdown."""
    logger.info("🛑 Shutting down AI Docs Vector DB MCP Server...")
    await cleanup_services()


# ========== Main Entry Point ==========

if __name__ == "__main__":
    import asyncio
    import sys

    try:
        # Run the server
        asyncio.run(mcp.run())
    except KeyboardInterrupt:
        logger.info("⌨️ Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Server crashed: {e}")
        sys.exit(1)
