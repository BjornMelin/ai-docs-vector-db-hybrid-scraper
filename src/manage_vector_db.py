#!/usr/bin/env python3
"""Vector Database Management Utility
Modern Python 3.13 implementation with async patterns for Qdrant operations.

Provides comprehensive database management, search, and maintenance utilities
"""

import logging
from typing import Any

import click
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

# Import unified configuration and service layer
from src.config import get_config
from src.infrastructure.client_manager import ClientManager
from src.utils import async_command


console = Console()


class SearchResult(BaseModel):
    """Search result from vector database."""

    id: int
    score: float
    url: str
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionInfo(BaseModel):
    """Collection information."""

    name: str
    vector_count: int
    vector_size: int


class DatabaseStats(BaseModel):
    """Database statistics."""

    total_collections: int
    total_vectors: int
    collections: list[CollectionInfo] = Field(default_factory=list)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper())
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    return logger


class VectorDBManager:
    """Comprehensive vector database management using ClientManager."""

    def __init__(
        self,
        client_manager: ClientManager,
        qdrant_url: str | None = None,
    ) -> None:
        """Initialize with ClientManager.

        Args:
            client_manager: ClientManager instance for all services
            qdrant_url: Optional Qdrant URL override

        """
        self.client_manager = client_manager
        self.qdrant_url = qdrant_url
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize services using ClientManager."""
        if self._initialized:
            return

        # Override Qdrant URL if provided
        if self.qdrant_url:
            # Get the current config and override URL
            config = self.client_manager.config
            config.qdrant.url = self.qdrant_url

        # Ensure ClientManager is initialized
        if not self.client_manager._initialized:
            await self.client_manager.initialize()

        self._initialized = True

    async def get_qdrant_service(self):
        """Get QdrantService from ClientManager."""
        if not self._initialized:
            await self.initialize()
        return await self.client_manager.get_qdrant_service()

    async def get_embedding_manager(self):
        """Get EmbeddingManager from ClientManager."""
        if not self._initialized:
            await self.initialize()
        return await self.client_manager.get_embedding_manager()

    async def cleanup(self) -> None:
        """Cleanup services (delegated to ClientManager)."""
        if self.client_manager:
            await self.client_manager.cleanup()
        self._initialized = False

    async def list_collections(self) -> list[str]:
        """List all collections."""
        try:
            await self.initialize()
            qdrant_service = await self.get_qdrant_service()
            return await qdrant_service.list_collections()
        except Exception as e:
            console.print(f"❌ Error listing collections: {e}", style="red")
            return []

    async def create_collection(
        self, collection_name: str, vector_size: int = 1536
    ) -> bool:
        """Create a new collection."""
        try:
            await self.initialize()
            qdrant_service = await self.get_qdrant_service()
            await qdrant_service.create_collection(
                collection_name=collection_name,
                vector_size=vector_size,
                distance="Cosine",
            )
            console.print(
                f"✅ Successfully created collection: {collection_name}", style="green"
            )
            return True  # noqa: TRY300
        except Exception as e:
            console.print(
                f"❌ Error creating collection {collection_name}: {e}", style="red"
            )
            return False

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            await self.initialize()
            qdrant_service = await self.get_qdrant_service()
            await qdrant_service.delete_collection(collection_name)
            console.print(
                f"✅ Successfully deleted collection: {collection_name}", style="green"
            )
            return True  # noqa: TRY300
        except Exception as e:
            console.print(
                f"❌ Error deleting collection {collection_name}: {e}", style="red"
            )
            return False

    async def get_collection_info(self, collection_name: str) -> CollectionInfo | None:
        """Get information about a specific collection."""
        try:
            await self.initialize()
            qdrant_service = await self.get_qdrant_service()
            collection_info = await qdrant_service.get_collection_info(collection_name)

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
        """Search for similar vectors."""
        try:
            await self.initialize()
            qdrant_service = await self.get_qdrant_service()
            search_results = await qdrant_service.search_vectors(
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
        """Get comprehensive database statistics."""
        try:
            await self.initialize()
            qdrant_service = await self.get_qdrant_service()
            collection_names = await qdrant_service.list_collections()
            collections = []
            total_vectors = 0

            for collection_name in collection_names:
                collection_info = await qdrant_service.get_collection_info(
                    collection_name
                )
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
        """Clear all vectors from a collection."""
        try:
            await self.initialize()

            # Get vector size before deletion
            qdrant_service = await self.get_qdrant_service()
            collection_info = await qdrant_service.get_collection_info(collection_name)
            if not collection_info:
                console.print(f"❌ Collection {collection_name} not found", style="red")
                return False

            vector_size = collection_info.vector_size

            # Delete and recreate collection
            await qdrant_service.delete_collection(collection_name)
            await qdrant_service.create_collection(
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
        """Alias for get_database_stats for backward compatibility."""
        return await self.get_database_stats()


async def create_embeddings(text: str, embedding_manager) -> list[float]:
    """Create embeddings for text using EmbeddingManager."""
    try:
        embeddings = await embedding_manager.generate_embeddings([text])
        return embeddings[0] if embeddings else []
    except Exception as e:
        console.print(f"❌ Error creating embeddings: {e}", style="red")
        return []


def _create_manager_from_context(ctx) -> VectorDBManager:
    """Create VectorDBManager with ClientManager from CLI context."""
    config = get_config()
    if ctx.obj.get("url"):
        config.qdrant.url = ctx.obj.get("url")

    client_manager = ClientManager(config)
    return VectorDBManager(client_manager=client_manager)


# CLI Commands
@click.group()
@click.option("--url", help="Qdrant server URL (overrides config)")
@click.option("--log-level", default="INFO", help="Logging level")
@click.pass_context
def cli(ctx, url, log_level):
    """Vector Database Management CLI."""
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
@async_command
async def list(ctx):
    """List all collections."""
    manager = _create_manager_from_context(ctx)
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
@async_command
async def create(ctx, collection_name, vector_size):
    """Create a new collection."""
    manager = _create_manager_from_context(ctx)
    try:
        await manager.create_collection(collection_name, vector_size)
    finally:
        await manager.cleanup()


@cli.command()
@click.argument("collection_name")
@click.pass_context
@async_command
async def delete(ctx, collection_name):
    """Delete a collection."""
    manager = _create_manager_from_context(ctx)
    try:
        await manager.delete_collection(collection_name)
    finally:
        await manager.cleanup()


@cli.command()
@click.pass_context
@async_command
async def stats(ctx):
    """Show database statistics."""
    manager = _create_manager_from_context(ctx)
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
@async_command
async def info(ctx, collection_name):
    """Show collection information."""
    manager = _create_manager_from_context(ctx)
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
@async_command
async def clear(ctx, collection_name):
    """Clear all vectors from a collection."""
    manager = _create_manager_from_context(ctx)
    try:
        await manager.clear_collection(collection_name)
    finally:
        await manager.cleanup()


@cli.command()
@click.argument("collection_name")
@click.argument("query")
@click.option("--limit", default=5, help="Number of results")
@click.pass_context
@async_command
async def search(ctx, collection_name, query, limit):
    """Search for similar documents."""
    manager = _create_manager_from_context(ctx)
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
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
