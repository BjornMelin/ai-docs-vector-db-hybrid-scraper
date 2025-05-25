#!/usr/bin/env python3
"""Vector Database Management Utility
Modern Python 3.13 implementation with async patterns for Qdrant operations

Provides comprehensive database management, search, and maintenance utilities
"""

import asyncio
import logging
from typing import Any

import click
from pydantic import BaseModel
from pydantic import Field
from rich.console import Console
from rich.table import Table

# Import unified configuration and service layer
from .config import get_config
from .services.config import APIConfig
from .services.embeddings.manager import EmbeddingManager
from .services.qdrant_service import QdrantService

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
    log_level = getattr(logging, level.upper())
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    return logger


class VectorDBManager:
    """Comprehensive vector database management using service layer"""

    def __init__(
        self,
        qdrant_service: QdrantService | None = None,
        embedding_manager: EmbeddingManager | None = None,
    ) -> None:
        """Initialize with service layer components.
        
        Args:
            qdrant_service: QdrantService instance for database operations
            embedding_manager: EmbeddingManager for generating embeddings
        """
        self.qdrant_service = qdrant_service
        self.embedding_manager = embedding_manager
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize services if not already initialized"""
        if self._initialized:
            return

        # If services not provided, create them from unified config
        if not self.qdrant_service or not self.embedding_manager:
            api_config = APIConfig.from_unified_config()

            if not self.qdrant_service:
                self.qdrant_service = QdrantService(api_config)

            if not self.embedding_manager:
                self.embedding_manager = EmbeddingManager(api_config)

        # Initialize services
        await self.qdrant_service.initialize()
        await self.embedding_manager.initialize()
        self._initialized = True

    async def cleanup(self) -> None:
        """Cleanup services"""
        if self.qdrant_service:
            await self.qdrant_service.cleanup()
        if self.embedding_manager:
            await self.embedding_manager.cleanup()
        self._initialized = False

    async def list_collections(self) -> list[str]:
        """List all collections"""
        try:
            await self.initialize()
            return await self.qdrant_service.list_collections()
        except Exception as e:
            console.print(f"❌ Error listing collections: {e}", style="red")
            return []

    async def create_collection(
        self, collection_name: str, vector_size: int = 1536
    ) -> bool:
        """Create a new collection"""
        try:
            await self.initialize()
            await self.qdrant_service.create_collection(
                collection_name=collection_name,
                vector_size=vector_size,
                distance="Cosine"
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
            await self.initialize()
            await self.qdrant_service.delete_collection(collection_name)
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
            await self.initialize()
            collection_info = await self.qdrant_service.get_collection_info(collection_name)

            if not collection_info:
                return None

            return CollectionInfo(
                name=collection_name,
                vector_count=collection_info.vector_count,
                vector_size=collection_info.vector_size,
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
            await self.initialize()
            search_results = await self.qdrant_service.search_vectors(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
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
            await self.initialize()
            collection_names = await self.qdrant_service.list_collections()
            collections = []
            total_vectors = 0

            for collection_name in collection_names:
                collection_info = await self.qdrant_service.get_collection_info(collection_name)
                if collection_info:
                    total_vectors += collection_info.vector_count
                    collections.append(
                        CollectionInfo(
                            name=collection_name,
                            vector_count=collection_info.vector_count,
                            vector_size=collection_info.vector_size,
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
            await self.initialize()

            # Get vector size before deletion
            collection_info = await self.qdrant_service.get_collection_info(collection_name)
            if not collection_info:
                console.print(f"❌ Collection {collection_name} not found", style="red")
                return False

            vector_size = collection_info.vector_size

            # Delete and recreate collection
            await self.qdrant_service.delete_collection(collection_name)
            await self.qdrant_service.create_collection(
                collection_name=collection_name,
                vector_size=vector_size,
                distance="Cosine",
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

    async def get_stats(self) -> DatabaseStats | None:
        """Alias for get_database_stats for backward compatibility"""
        return await self.get_database_stats()


async def create_embeddings(text: str, embedding_manager: EmbeddingManager) -> list[float]:
    """Create embeddings for text using EmbeddingManager"""
    try:
        embeddings = await embedding_manager.create_embeddings([text])
        return embeddings[0] if embeddings else []
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
    manager = VectorDBManager()
    try:
        collections = await manager.list_collections()
        if collections:
            console.print("📋 Collections:", style="bold yellow")
            for collection in collections:
                console.print(f"  • {collection}", style="cyan")
        else:
            console.print("No collections found", style="yellow")
    finally:
        await manager.cleanup()


@cli.command()
@click.argument("collection_name")
@click.option("--vector-size", default=1536, help="Vector size")
@click.pass_context
async def create(ctx, collection_name, vector_size):
    """Create a new collection"""
    manager = VectorDBManager()
    try:
        await manager.create_collection(collection_name, vector_size)
    finally:
        await manager.cleanup()


@cli.command()
@click.argument("collection_name")
@click.pass_context
async def delete(ctx, collection_name):
    """Delete a collection"""
    manager = VectorDBManager()
    try:
        await manager.delete_collection(collection_name)
    finally:
        await manager.cleanup()


@cli.command()
@click.pass_context
async def stats(ctx):
    """Show database statistics"""
    manager = VectorDBManager()
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
        await manager.cleanup()


@cli.command()
@click.argument("collection_name")
@click.pass_context
async def info(ctx, collection_name):
    """Show collection information"""
    manager = VectorDBManager()
    try:
        info = await manager.get_collection_info(collection_name)
        if info:
            console.print(f"📁 Collection: {info.name}", style="bold blue")
            console.print(f"Vector Count: {info.vector_count}", style="cyan")
            console.print(f"Vector Size: {info.vector_size}", style="cyan")
        else:
            console.print(f"Collection '{collection_name}' not found", style="red")
    finally:
        await manager.cleanup()


@cli.command()
@click.argument("collection_name")
@click.pass_context
async def clear(ctx, collection_name):
    """Clear all vectors from a collection"""
    manager = VectorDBManager()
    try:
        await manager.clear_collection(collection_name)
    finally:
        await manager.cleanup()


@cli.command()
@click.argument("collection_name")
@click.argument("query")
@click.option("--limit", default=5, help="Number of results")
@click.pass_context
async def search(ctx, collection_name, query, limit):
    """Search for similar documents"""
    manager = VectorDBManager()
    try:
        # Initialize manager to ensure embedding_manager is available
        await manager.initialize()

        # Create query embedding
        query_vector = await create_embeddings(query, manager.embedding_manager)
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
        await manager.cleanup()


def main():
    """Main entry point"""
    import asyncio

    # Avoid double-wrapping if already processed
    if hasattr(cli, '_commands_wrapped'):
        cli()
        return

    # Create sync version of cli
    for command in cli.commands.values():
        if asyncio.iscoroutinefunction(command.callback):
            original_callback = command.callback

            def make_sync_callback(func):
                def sync_callback(*args, **kwargs):
                    return asyncio.run(func(*args, **kwargs))
                # Copy function metadata
                sync_callback.__name__ = func.__name__
                sync_callback.__doc__ = func.__doc__
                return sync_callback

            command.callback = make_sync_callback(original_callback)
    
    # Mark as wrapped
    cli._commands_wrapped = True
    cli()


if __name__ == "__main__":
    main()
