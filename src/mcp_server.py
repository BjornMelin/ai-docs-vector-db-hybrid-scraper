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


async def get_qdrant_client():
    """Get or create Qdrant client"""
    global qdrant_client
    if qdrant_client is None:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_client = AsyncQdrantClient(url=qdrant_url)
    return qdrant_client


async def get_openai_client():
    """Get or create OpenAI client"""
    global openai_client
    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        openai_client = AsyncOpenAI(api_key=api_key)
    return openai_client


# ========== Scraping Tools ==========


@mcp.tool()
async def scrape_url(
    url: str, max_depth: int = 3, chunk_size: int = 1600
) -> dict[str, Any]:
    """Scrape and index documentation from a URL using Crawl4AI.

    Args:
        url: The URL to scrape
        max_depth: Maximum crawl depth (default: 3)
        chunk_size: Size of text chunks for indexing (default: 1600)

    Returns:
        Dictionary with scraping results
    """
    try:
        # Import and use the existing scraper
        from .crawl4ai_bulk_embedder import main as run_scraper

        # Create temporary config file
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

        config_path = Path("temp_scrape_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Run the scraper
        await run_scraper(["--config", str(config_path)])

        # Clean up
        config_path.unlink()

        return {
            "success": True,
            "message": f"Successfully scraped {url}",
            "url": url,
            "max_depth": max_depth,
            "chunk_size": chunk_size,
        }

    except Exception as e:
        logger.error(f"Error scraping URL: {e}")
        return {"success": False, "error": str(e), "url": url}


@mcp.tool()
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
    if not FirecrawlApp:
        return {
            "success": False,
            "error": "Firecrawl not installed. Run: pip install firecrawl-py",
        }

    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        return {
            "success": False,
            "error": "FIRECRAWL_API_KEY environment variable not set",
        }

    try:
        app = FirecrawlApp(api_key=api_key)
        result = app.scrape_url(url, params={"formats": formats})

        return {
            "success": True,
            "url": url,
            "content": result.get("markdown", result.get("content", "")),
            "metadata": result.get("metadata", {}),
            "formats": formats,
        }

    except Exception as e:
        logger.error(f"Error with Firecrawl: {e}")
        return {"success": False, "error": str(e), "url": url}


# ========== Search Tools ==========


@mcp.tool()
async def search(
    query: str, collection: str = "documentation", limit: int = 5
) -> dict[str, Any]:
    """Search indexed documentation using semantic search.

    Args:
        query: The search query
        collection: Collection name to search in (default: documentation)
        limit: Number of results to return (default: 5)

    Returns:
        Dictionary with search results
    """
    try:
        qdrant = await get_qdrant_client()
        openai = await get_openai_client()

        # Get query embedding
        response = await openai.embeddings.create(
            model="text-embedding-3-small", input=query
        )
        query_vector = response.data[0].embedding

        # Search in Qdrant
        results = await qdrant.search(
            collection_name=collection, query_vector=query_vector, limit=limit
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "score": result.score,
                    "content": result.payload.get("content", ""),
                    "url": result.payload.get("url", ""),
                    "title": result.payload.get("title", ""),
                    "chunk_index": result.payload.get("chunk_index", 0),
                }
            )

        return {
            "success": True,
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results),
        }

    except Exception as e:
        logger.error(f"Error searching: {e}")
        return {"success": False, "error": str(e), "query": query}


# ========== Collection Management Tools ==========


@mcp.tool()
async def list_collections() -> dict[str, Any]:
    """List all vector database collections.

    Returns:
        Dictionary with collection information
    """
    try:
        qdrant = await get_qdrant_client()
        response = await qdrant.get_collections()

        collections = []
        for collection in response.collections:
            info = await qdrant.get_collection(collection.name)
            collections.append(
                {
                    "name": collection.name,
                    "vectors_count": info.vectors_count,
                    "points_count": info.points_count,
                    "indexed_vectors_count": info.indexed_vectors_count,
                }
            )

        return {"success": True, "collections": collections, "total": len(collections)}

    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def create_collection(
    name: str, vector_size: int = 1536, distance: str = "cosine"
) -> dict[str, Any]:
    """Create a new vector database collection.

    Args:
        name: Collection name
        vector_size: Vector dimension size (default: 1536 for OpenAI)
        distance: Distance metric (cosine, euclidean, dot)

    Returns:
        Dictionary with creation status
    """
    try:
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

        return {
            "success": True,
            "name": name,
            "vector_size": vector_size,
            "distance": distance,
        }

    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def delete_collection(name: str) -> dict[str, Any]:
    """Delete a vector database collection.

    Args:
        name: Collection name to delete

    Returns:
        Dictionary with deletion status
    """
    try:
        qdrant = await get_qdrant_client()
        await qdrant.delete_collection(collection_name=name)

        return {"success": True, "message": f"Collection '{name}' deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_collection_info(name: str) -> dict[str, Any]:
    """Get detailed information about a collection.

    Args:
        name: Collection name

    Returns:
        Dictionary with collection details
    """
    try:
        qdrant = await get_qdrant_client()
        info = await qdrant.get_collection(collection_name=name)

        return {
            "success": True,
            "name": name,
            "status": info.status,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "segments_count": info.segments_count,
            "config": {
                "vector_size": info.config.params.vectors.size,
                "distance": str(info.config.params.vectors.distance),
            },
        }

    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        return {"success": False, "error": str(e)}


# ========== Utility Tools ==========


@mcp.tool()
async def clear_cache() -> dict[str, Any]:
    """Clear the scraping cache.

    Returns:
        Dictionary with cache clearing status
    """
    try:
        # Clear Crawl4AI cache
        cache_dir = Path.home() / ".crawl4ai" / "cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

        return {
            "success": True,
            "message": "Cache cleared successfully",
            "cache_path": str(cache_dir),
        }

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return {"success": False, "error": str(e)}


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
