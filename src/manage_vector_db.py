#!/usr/bin/env python3
"""Vector Database Management Utility
Modern Python 3.13 implementation with async patterns for Qdrant operations

Provides comprehensive database management, search, and maintenance utilities
"""

import asyncio
import logging
import os
from typing import Any

import click
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic import Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance
from qdrant_client.models import VectorParams
from rich.console import Console
from rich.table import Table

# Import unified configuration
try:
    from config import get_config
except ImportError:
    from .config import get_config

console = Console()


class SearchResult(BaseModel):
    """Search result from vector database"""

    id: int
    score: float
    url: str
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionInfo(BaseModel):
    """Collection information"""

    name: str
    vector_count: int
    vector_size: int


class DatabaseStats(BaseModel):
    """Database statistics"""

    total_collections: int
    total_vectors: int
    collections: list[CollectionInfo] = Field(default_factory=list)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


class VectorDBManager:
    """Comprehensive vector database management with modern async patterns"""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        openai_api_key: str | None = None,
    ) -> None:
        self.url = url
        self.client: AsyncQdrantClient | None = None
        self.openai_client: AsyncOpenAI | None = None

        if openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=openai_api_key)

    async def connect(self) -> None:
        """Connect to Qdrant client"""
        try:
            self.client = AsyncQdrantClient(url=self.url)
        except Exception as e:
            console.print(f"❌ Error connecting to Qdrant: {e}", style="red")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Qdrant client"""
        if self.client:
            await self.client.close()
            self.client = None

    async def list_collections(self) -> list[str]:
        """List all collections"""
        try:
            if not self.client:
                raise RuntimeError("Client not connected")

            collections_response = await self.client.get_collections()
            return [collection.name for collection in collections_response.collections]
        except Exception as e:
            console.print(f"❌ Error listing collections: {e}", style="red")
            return []

    async def create_collection(
        self, collection_name: str, vector_size: int = 1536
    ) -> bool:
        """Create a new collection"""
        try:
            if not self.client:
                raise RuntimeError("Client not connected")

            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            console.print(
                f"✅ Successfully created collection: {collection_name}", style="green"
            )
            return True
        except Exception as e:
            console.print(
                f"❌ Error creating collection {collection_name}: {e}", style="red"
            )
            return False

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            if not self.client:
                raise RuntimeError("Client not connected")

            await self.client.delete_collection(collection_name)
            console.print(
                f"✅ Successfully deleted collection: {collection_name}", style="green"
            )
            return True
        except Exception as e:
            console.print(
                f"❌ Error deleting collection {collection_name}: {e}", style="red"
            )
            return False

    async def get_collection_info(self, collection_name: str) -> CollectionInfo | None:
        """Get information about a specific collection"""
        try:
            if not self.client:
                raise RuntimeError("Client not connected")

            collection_info = await self.client.get_collection(collection_name)
            count_result = await self.client.count(collection_name)

            return CollectionInfo(
                name=collection_name,
                vector_count=count_result.count,
                vector_size=collection_info.config.params.vectors.size,
            )
        except Exception as e:
            console.print(
                f"❌ Error getting collection info for {collection_name}: {e}",
                style="red",
            )
            return None

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Search for similar vectors"""
        try:
            if not self.client:
                raise RuntimeError("Client not connected")

            search_results = await self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )

            results = []
            for result in search_results:
                payload = result.payload or {}
                results.append(
                    SearchResult(
                        id=result.id,
                        score=result.score,
                        url=payload.get("url", ""),
                        title=payload.get("title", ""),
                        content=payload.get("content", ""),
                        metadata=payload,
                    )
                )

            return results
        except Exception as e:
            console.print(f"❌ Error searching vectors: {e}", style="red")
            return []

    async def get_database_stats(self) -> DatabaseStats | None:
        """Get comprehensive database statistics"""
        try:
            if not self.client:
                raise RuntimeError("Client not connected")

            collections_response = await self.client.get_collections()
            collections = []
            total_vectors = 0

            for collection in collections_response.collections:
                count_result = await self.client.count(collection.name)
                vector_count = count_result.count
                total_vectors += vector_count

                collection_info = await self.client.get_collection(collection.name)
                collections.append(
                    CollectionInfo(
                        name=collection.name,
                        vector_count=vector_count,
                        vector_size=collection_info.config.params.vectors.size,
                    )
                )

            return DatabaseStats(
                total_collections=len(collections),
                total_vectors=total_vectors,
                collections=collections,
            )
        except Exception as e:
            console.print(f"❌ Error getting database stats: {e}", style="red")
            return None

    async def clear_collection(self, collection_name: str) -> bool:
        """Clear all vectors from a collection"""
        try:
            if not self.client:
                raise RuntimeError("Client not connected")

            # Get vector size before deletion
            collection_info = await self.client.get_collection(collection_name)
            vector_size = collection_info.config.params.vectors.size

            # Delete and recreate collection
            await self.client.delete_collection(collection_name)
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

            console.print(
                f"✅ Successfully cleared collection: {collection_name}", style="green"
            )
            return True
        except Exception as e:
            console.print(
                f"❌ Error clearing collection {collection_name}: {e}", style="red"
            )
            return False


async def create_embeddings(text: str, openai_client: AsyncOpenAI) -> list[float]:
    """Create embeddings for text using OpenAI"""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
        )
        return response.data[0].embedding
    except Exception as e:
        console.print(f"❌ Error creating embeddings: {e}", style="red")
        return []


# CLI Commands
@click.group()
@click.option("--url", help="Qdrant server URL (overrides config)")
@click.option("--log-level", default="INFO", help="Logging level")
@click.pass_context
def cli(ctx, url, log_level):
    """Vector Database Management CLI"""
    # Get configuration from unified config
    unified_config = get_config()
    
    # Use URL from command line or config
    if not url:
        url = unified_config.qdrant.url
    ctx.ensure_object(dict)
    ctx.obj["url"] = url
    ctx.obj["log_level"] = log_level
    setup_logging(log_level)


@cli.command()
@click.pass_context
async def list(ctx):
    """List all collections"""
    manager = VectorDBManager(ctx.obj["url"])
    await manager.connect()
    try:
        collections = await manager.list_collections()
        if collections:
            console.print("📋 Collections:", style="bold yellow")
            for collection in collections:
                console.print(f"  • {collection}", style="cyan")
        else:
            console.print("No collections found", style="yellow")
    finally:
        await manager.disconnect()


@cli.command()
@click.argument("collection_name")
@click.option("--vector-size", default=1536, help="Vector size")
@click.pass_context
async def create(ctx, collection_name, vector_size):
    """Create a new collection"""
    manager = VectorDBManager(ctx.obj["url"])
    await manager.connect()
    try:
        await manager.create_collection(collection_name, vector_size)
    finally:
        await manager.disconnect()


@cli.command()
@click.argument("collection_name")
@click.pass_context
async def delete(ctx, collection_name):
    """Delete a collection"""
    manager = VectorDBManager(ctx.obj["url"])
    await manager.connect()
    try:
        await manager.delete_collection(collection_name)
    finally:
        await manager.disconnect()


@cli.command()
@click.pass_context
async def stats(ctx):
    """Show database statistics"""
    manager = VectorDBManager(ctx.obj["url"])
    await manager.connect()
    try:
        stats = await manager.get_database_stats()
        if stats:
            console.print("📊 Database Statistics", style="bold blue")
            console.print(f"Total Collections: {stats.total_collections}", style="cyan")
            console.print(f"Total Vectors: {stats.total_vectors}", style="cyan")

            if stats.collections:
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Collection", style="cyan")
                table.add_column("Vectors", justify="right", style="green")
                table.add_column("Vector Size", justify="right", style="yellow")

                for collection in stats.collections:
                    table.add_row(
                        collection.name,
                        str(collection.vector_count),
                        str(collection.vector_size),
                    )

                console.print(table)
        else:
            console.print("Could not retrieve database stats", style="red")
    finally:
        await manager.disconnect()


@cli.command()
@click.argument("collection_name")
@click.pass_context
async def info(ctx, collection_name):
    """Show collection information"""
    manager = VectorDBManager(ctx.obj["url"])
    await manager.connect()
    try:
        info = await manager.get_collection_info(collection_name)
        if info:
            console.print(f"📁 Collection: {info.name}", style="bold blue")
            console.print(f"Vector Count: {info.vector_count}", style="cyan")
            console.print(f"Vector Size: {info.vector_size}", style="cyan")
        else:
            console.print(f"Collection '{collection_name}' not found", style="red")
    finally:
        await manager.disconnect()


@cli.command()
@click.argument("collection_name")
@click.pass_context
async def clear(ctx, collection_name):
    """Clear all vectors from a collection"""
    manager = VectorDBManager(ctx.obj["url"])
    await manager.connect()
    try:
        await manager.clear_collection(collection_name)
    finally:
        await manager.disconnect()


@cli.command()
@click.argument("collection_name")
@click.argument("query")
@click.option("--limit", default=5, help="Number of results")
@click.pass_context
async def search(ctx, collection_name, query, limit):
    """Search for similar documents"""
    # Get configuration from unified config
    unified_config = get_config()
    
    if not unified_config.openai.api_key:
        console.print(
            "❌ OpenAI API key not configured. Please set AI_DOCS__OPENAI__API_KEY", style="red"
        )
        return

    manager = VectorDBManager(ctx.obj["url"], unified_config.openai.api_key)
    await manager.connect()
    try:
        # Create query embedding
        query_vector = await create_embeddings(query, manager.openai_client)
        if not query_vector:
            console.print("Failed to create query embedding", style="red")
            return

        # Search
        results = await manager.search_vectors(collection_name, query_vector, limit)

        if results:
            console.print(f"🔍 Search Results for: '{query}'", style="bold yellow")
            for i, result in enumerate(results, 1):
                console.print(f"\n{i}. {result.title}", style="bold cyan")
                console.print(f"   URL: {result.url}", style="blue")
                console.print(f"   Score: {result.score:.4f}", style="green")
                preview = (
                    result.content[:200] + "..."
                    if len(result.content) > 200
                    else result.content
                )
                console.print(f"   Preview: {preview}", style="white")
        else:
            console.print("No results found", style="yellow")
    finally:
        await manager.disconnect()


def main():
    """Main entry point"""

    # Make the CLI async-compatible
    def sync_cli():
        import asyncio

        return asyncio.run(cli())

    # Create sync version of cli
    for command in cli.commands.values():
        if asyncio.iscoroutinefunction(command.callback):
            original_callback = command.callback

            def make_sync_callback(func):
                def sync_callback(*args, **kwargs):
                    return asyncio.run(func(*args, **kwargs))

                return sync_callback

            command.callback = make_sync_callback(original_callback)

    cli()


if __name__ == "__main__":
    main()
