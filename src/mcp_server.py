#!/usr/bin/env python3
"""AI Documentation Vector DB MCP Server.

A unified Model Context Protocol server that combines web scraping (Crawl4AI + Firecrawl),
vector database management (Qdrant), and intelligent search capabilities.

Built with FastMCP 2.0 for efficient, scalable MCP server implementation.
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
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic import Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance
from qdrant_client.models import VectorParams
from rich.console import Console

from .error_handling import (
    ConfigurationError,
    ExternalServiceError,
    NetworkError,
    RateLimitError,
    ValidationError,
    circuit_breaker,
    firecrawl_rate_limiter,
    handle_mcp_errors,
    openai_rate_limiter,
    qdrant_rate_limiter,
    retry_async,
    validate_input,
)
from .security import SecurityValidator, validate_startup_security

# Try importing optional dependencies
try:
    from firecrawl import FirecrawlApp
except ImportError:
    FirecrawlApp = None

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

# Validate security requirements at startup
try:
    ENV_VARS = validate_startup_security()
    logger.info("✅ MCP Server security validation complete")
except Exception as e:
    logger.error(f"❌ MCP Server startup failed: {e}")
    raise


# ========== Pydantic Models for MCP Tools ==========


class ScrapeConfig(BaseModel):
    """Configuration for scraping operations"""

    url: str = Field(description="URL to scrape")
    max_depth: int = Field(default=3, description="Maximum crawl depth")
    chunk_size: int = Field(default=1600, description="Chunk size for text splitting")
    chunk_overlap: int = Field(default=320, description="Overlap between chunks")
    use_firecrawl: bool = Field(
        default=False, description="Use Firecrawl instead of Crawl4AI"
    )


class SearchQuery(BaseModel):
    """Search query configuration"""

    query: str = Field(description="Search query text")
    collection_name: str = Field(
        default="documentation", description="Collection to search in"
    )
    limit: int = Field(default=5, description="Number of results to return")
    min_score: float = Field(default=0.5, description="Minimum similarity score")


class CollectionConfig(BaseModel):
    """Configuration for collection operations"""

    collection_name: str = Field(description="Name of the collection")
    vector_size: int = Field(default=1536, description="Vector dimension size")
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

# Global clients (initialized on demand)
qdrant_client = None
openai_client = None


@retry_async(max_attempts=3, exceptions=(NetworkError, ConnectionError))
async def get_qdrant_client():
    """Get or create Qdrant client with error handling."""
    global qdrant_client
    if qdrant_client is None:
        try:
            qdrant_url = ENV_VARS.get("QDRANT_URL", "http://localhost:6333")
            qdrant_client = AsyncQdrantClient(url=qdrant_url)
            logger.info(f"✅ Qdrant client initialized: {qdrant_url}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Qdrant client: {e}")
            raise NetworkError(f"Cannot connect to Qdrant at {qdrant_url}: {e}")
    return qdrant_client


@retry_async(max_attempts=3, exceptions=(NetworkError, ConnectionError))
async def get_openai_client():
    """Get or create OpenAI client with error handling."""
    global openai_client
    if openai_client is None:
        try:
            api_key = ENV_VARS["OPENAI_API_KEY"]
            openai_client = AsyncOpenAI(api_key=api_key)
            logger.info("✅ OpenAI client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise ExternalServiceError(f"Cannot initialize OpenAI client: {e}")
    return openai_client


# ========== Scraping Tools ==========


@mcp.tool()
@handle_mcp_errors
@validate_input(
    url=SecurityValidator.validate_url,
    max_depth=lambda x: max(1, min(int(x), 10)),  # Limit depth 1-10
    chunk_size=lambda x: max(100, min(int(x), 8000)),  # Limit chunk size 100-8000
)
async def scrape_url(
    url: str, max_depth: int = 3, chunk_size: int = 1600
) -> dict[str, Any]:
    """Scrape and index documentation from a URL using Crawl4AI.

    Args:
        url: The URL to scrape (will be validated for security)
        max_depth: Maximum crawl depth (1-10, default: 3)
        chunk_size: Size of text chunks for indexing (100-8000, default: 1600)

    Returns:
        Dictionary with scraping results
    """
    logger.info(f"Starting scrape operation for URL: {url}")

    try:
        # Import and use the existing scraper
        from .crawl4ai_bulk_embedder import main as run_scraper

        # Create temporary config file with safe filename
        safe_filename = SecurityValidator.sanitize_filename(
            f"temp_scrape_{int(time.time())}.json"
        )
        config_path = Path(safe_filename)

        config = {
            "sites": [
                {
                    "name": "scraped_docs",
                    "url": url,
                    "max_depth": max_depth,
                    "patterns": [f"{url}/**"],
                    "exclude_patterns": [],
                }
            ]
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Running scraper with config: {config_path}")

        # Run the scraper
        await run_scraper(["--config", str(config_path)])

        # Clean up
        if config_path.exists():
            config_path.unlink()

        logger.info(f"✅ Successfully scraped {url}")

        return {
            "success": True,
            "message": f"Successfully scraped {url}",
            "url": url,
            "max_depth": max_depth,
            "chunk_size": chunk_size,
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"❌ Error scraping URL {url}: {e}")
        # Clean up on error
        if "config_path" in locals() and config_path.exists():
            config_path.unlink()
        raise ExternalServiceError(f"Scraping failed for {url}: {e}")


@mcp.tool()
@handle_mcp_errors
@validate_input(
    url=SecurityValidator.validate_url,
    formats=lambda x: [
        f.strip().lower()
        for f in x
        if f.strip().lower() in ["markdown", "html", "links", "screenshot"]
    ],
)
async def scrape_with_firecrawl(
    url: str, formats: list[str] = ["markdown"]
) -> dict[str, Any]:
    """Scrape a URL using Firecrawl API for high-quality extraction.

    Args:
        url: The URL to scrape
        formats: Output formats (markdown, html, links, etc.)

    Returns:
        Dictionary with scraped content
    """
    logger.info(f"Starting Firecrawl scrape for URL: {url}")

    if not FirecrawlApp:
        raise ExternalServiceError(
            "Firecrawl not installed. Run: pip install firecrawl-py"
        )

    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ConfigurationError("FIRECRAWL_API_KEY environment variable not set")

    # Apply rate limiting
    await firecrawl_rate_limiter.acquire()

    try:
        app = FirecrawlApp(api_key=api_key)
        result = app.scrape_url(url, params={"formats": formats})

        logger.info(f"✅ Successfully scraped {url} with Firecrawl")
        return {
            "success": True,
            "url": url,
            "content": result.get("markdown", result.get("content", "")),
            "metadata": result.get("metadata", {}),
            "formats": formats,
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"❌ Error with Firecrawl for {url}: {e}")
        raise ExternalServiceError(f"Firecrawl scraping failed for {url}: {e}")


# ========== Search Tools ==========


@mcp.tool()
@handle_mcp_errors
@validate_input(
    query=SecurityValidator.validate_query_string,
    collection=SecurityValidator.validate_collection_name,
    limit=lambda x: max(1, min(int(x), 50)),  # Limit results 1-50
)
@circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
async def search(
    query: str, collection: str = "documentation", limit: int = 5
) -> dict[str, Any]:
    """Search indexed documentation using semantic search.

    Args:
        query: The search query (will be validated and sanitized)
        collection: Collection name to search in (default: documentation)
        limit: Number of results to return (1-50, default: 5)

    Returns:
        Dictionary with search results
    """
    logger.info(
        f"Search request: query='{query[:100]}...', collection='{collection}', limit={limit}"
    )

    # Apply rate limiting
    await openai_rate_limiter.acquire()
    await qdrant_rate_limiter.acquire()

    try:
        qdrant = await get_qdrant_client()
        openai = await get_openai_client()

        # Get query embedding with retry
        response = await openai.embeddings.create(
            model="text-embedding-3-small",
            input=[query],  # Wrap in list for consistency
            dimensions=1536,
        )
        query_vector = response.data[0].embedding

        # Search in Qdrant with retry
        results = await qdrant.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
            score_threshold=0.1,  # Filter very low relevance results
        )

        # Format results safely
        formatted_results = []
        for result in results:
            # Sanitize content for output
            content = result.payload.get("content", "")
            if len(content) > 1000:
                content = content[:1000] + "..."

            formatted_results.append(
                {
                    "score": round(float(result.score), 4),
                    "content": content,
                    "url": result.payload.get("url", ""),
                    "title": result.payload.get("title", ""),
                    "chunk_index": result.payload.get("chunk_index", 0),
                    "id": str(result.id) if result.id else "",
                }
            )

        logger.info(f"✅ Search completed: {len(formatted_results)} results")

        return {
            "success": True,
            "query": query,
            "collection": collection,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"❌ Search failed for query '{query}': {e}")
        raise ExternalServiceError(f"Search operation failed: {e}")


# ========== Collection Management Tools ==========


@mcp.tool()
@handle_mcp_errors
@circuit_breaker(failure_threshold=3, recovery_timeout=30.0)
async def list_collections() -> dict[str, Any]:
    """List all vector database collections.

    Returns:
        Dictionary with collection information
    """
    logger.info("Listing vector database collections")

    # Apply rate limiting
    await qdrant_rate_limiter.acquire()

    qdrant = await get_qdrant_client()
    response = await qdrant.get_collections()

    collections = []
    for collection in response.collections:
        try:
            info = await qdrant.get_collection(collection.name)
            collections.append(
                {
                    "name": collection.name,
                    "vectors_count": info.vectors_count or 0,
                    "points_count": info.points_count or 0,
                    "indexed_vectors_count": info.indexed_vectors_count or 0,
                    "status": getattr(info, "status", "unknown"),
                }
            )
        except Exception as e:
            logger.warning(f"Could not get info for collection {collection.name}: {e}")
            collections.append(
                {
                    "name": collection.name,
                    "vectors_count": 0,
                    "points_count": 0,
                    "indexed_vectors_count": 0,
                    "status": "error",
                    "error": str(e),
                }
            )

    logger.info(f"✅ Listed {len(collections)} collections")

    return {
        "success": True,
        "collections": collections,
        "total": len(collections),
        "timestamp": time.time(),
    }


@mcp.tool()
@handle_mcp_errors
@validate_input(
    name=SecurityValidator.validate_collection_name,
    vector_size=lambda x: max(1, min(int(x), 4096)),  # Limit vector size 1-4096
    distance=lambda x: x.lower()
    if x.lower() in ["cosine", "euclidean", "dot"]
    else "cosine",
)
async def create_collection(
    name: str, vector_size: int = 1536, distance: str = "cosine"
) -> dict[str, Any]:
    """Create a new vector database collection.

    Args:
        name: Collection name (validated for security)
        vector_size: Vector dimension size (1-4096, default: 1536 for OpenAI)
        distance: Distance metric (cosine, euclidean, dot)

    Returns:
        Dictionary with creation status
    """
    logger.info(
        f"Creating collection: name='{name}', vector_size={vector_size}, distance='{distance}'"
    )

    # Apply rate limiting
    await qdrant_rate_limiter.acquire()

    qdrant = await get_qdrant_client()

    # Map distance metric
    distance_map = {
        "cosine": Distance.COSINE,
        "euclidean": Distance.EUCLID,
        "dot": Distance.DOT,
    }
    distance_metric = distance_map.get(distance.lower(), Distance.COSINE)

    # Create collection
    await qdrant.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=distance_metric),
    )

    logger.info(f"✅ Collection '{name}' created successfully")

    return {
        "success": True,
        "name": name,
        "vector_size": vector_size,
        "distance": distance,
        "timestamp": time.time(),
    }


@mcp.tool()
@handle_mcp_errors
@validate_input(name=SecurityValidator.validate_collection_name)
async def delete_collection(name: str) -> dict[str, Any]:
    """Delete a vector database collection.

    Args:
        name: Collection name to delete (validated for security)

    Returns:
        Dictionary with deletion status
    """
    logger.warning(f"Deleting collection: '{name}' - THIS ACTION CANNOT BE UNDONE")

    # Apply rate limiting
    await qdrant_rate_limiter.acquire()

    qdrant = await get_qdrant_client()
    await qdrant.delete_collection(collection_name=name)

    logger.info(f"✅ Collection '{name}' deleted successfully")

    return {
        "success": True,
        "message": f"Collection '{name}' deleted successfully",
        "name": name,
        "timestamp": time.time(),
    }


@mcp.tool()
@handle_mcp_errors
@validate_input(name=SecurityValidator.validate_collection_name)
async def get_collection_info(name: str) -> dict[str, Any]:
    """Get detailed information about a collection.

    Args:
        name: Collection name (validated for security)

    Returns:
        Dictionary with collection details
    """
    logger.info(f"Getting collection info: '{name}'")

    # Apply rate limiting
    await qdrant_rate_limiter.acquire()

    qdrant = await get_qdrant_client()
    info = await qdrant.get_collection(collection_name=name)

    result = {
        "success": True,
        "name": name,
        "status": str(info.status) if info.status else "unknown",
        "vectors_count": info.vectors_count or 0,
        "points_count": info.points_count or 0,
        "indexed_vectors_count": info.indexed_vectors_count or 0,
        "segments_count": getattr(info, "segments_count", 0),
        "timestamp": time.time(),
    }

    # Safely extract config information
    try:
        if hasattr(info, "config") and hasattr(info.config, "params"):
            if hasattr(info.config.params, "vectors"):
                result["config"] = {
                    "vector_size": getattr(
                        info.config.params.vectors, "size", "unknown"
                    ),
                    "distance": str(
                        getattr(info.config.params.vectors, "distance", "unknown")
                    ),
                }
    except Exception as e:
        logger.warning(f"Could not extract config for collection {name}: {e}")
        result["config"] = {"error": "config not available"}

    logger.info(f"✅ Retrieved info for collection '{name}'")

    return result


# ========== Utility Tools ==========


@mcp.tool()
@handle_mcp_errors
async def clear_cache() -> dict[str, Any]:
    """Clear the scraping cache safely.

    Returns:
        Dictionary with cache clearing status
    """
    logger.info("Clearing scraping cache")

    # Clear Crawl4AI cache safely
    cache_dir = Path.home() / ".crawl4ai" / "cache"
    cleared_paths = []

    if cache_dir.exists() and cache_dir.is_dir():
        try:
            # List what we're about to clear for logging
            cache_size = sum(
                f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
            )
            file_count = len(list(cache_dir.rglob("*")))

            logger.info(
                f"Clearing cache: {file_count} files, {cache_size / 1024 / 1024:.2f} MB"
            )

            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cleared_paths.append(str(cache_dir))

        except Exception as e:
            logger.warning(f"Could not clear cache directory {cache_dir}: {e}")

    # Also clear any temporary scraping files
    temp_pattern = Path.cwd().glob("temp_scrape_*.json")
    for temp_file in temp_pattern:
        try:
            temp_file.unlink()
            cleared_paths.append(str(temp_file))
        except Exception as e:
            logger.warning(f"Could not remove temp file {temp_file}: {e}")

    logger.info(f"✅ Cache cleared: {len(cleared_paths)} locations")

    return {
        "success": True,
        "message": "Cache cleared successfully",
        "cleared_paths": cleared_paths,
        "timestamp": time.time(),
    }


# ========== Resources ==========


@mcp.resource("config://environment")
async def get_environment_config() -> dict[str, Any]:
    """Get current environment configuration"""
    return {
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "openai_api_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "firecrawl_api_key_set": bool(os.getenv("FIRECRAWL_API_KEY")),
        "environment": os.getenv("ENVIRONMENT", "development"),
    }


@mcp.resource("stats://database")
async def get_database_stats() -> dict[str, Any]:
    """Get database statistics"""
    try:
        qdrant = await get_qdrant_client()
        response = await qdrant.get_collections()

        total_vectors = 0
        for collection in response.collections:
            info = await qdrant.get_collection(collection.name)
            total_vectors += info.vectors_count

        return {
            "total_collections": len(response.collections),
            "total_vectors": total_vectors,
            "status": "connected",
        }
    except Exception as e:
        return {
            "total_collections": 0,
            "total_vectors": 0,
            "status": "error",
            "error": str(e),
        }


# ========== Main Entry Point ==========


def main():
    """Main entry point for the MCP server"""
    mcp.run()


if __name__ == "__main__":
    main()
