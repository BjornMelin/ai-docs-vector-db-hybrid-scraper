#!/usr/bin/env python3
"""Crawl4AI Bulk Embedder - High-performance bulk web scraping and embedding pipeline.

This tool provides a CLI for bulk scraping URLs using Crawl4AI and generating embeddings
for storage in the unified vector store service.
It supports concurrent processing, resumability,
and various input formats.
"""

import asyncio
import csv
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import aiofiles  # type: ignore[import]
import click  # type: ignore[import]
import httpx  # type: ignore[import]
from defusedxml import ElementTree  # type: ignore[import]
from pydantic import BaseModel, Field  # type: ignore[import]
from rich.console import Console  # type: ignore[import]
from rich.progress import (  # type: ignore[import]
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table  # type: ignore[import]

from .chunking import DocumentChunker
from .config.loader import Settings, get_settings
from .infrastructure.container import (
    ApplicationContainer,
    get_container,
    initialize_container,
    shutdown_container,
)
from .services.embeddings.manager import QualityTier
from .services.errors import ServiceError
from .services.logging_config import configure_logging
from .services.vector_db.service import VectorStoreService
from .services.vector_db.types import CollectionSchema, VectorRecord


class ScrapingError(ServiceError):
    """Exception raised when web scraping fails."""


class ContentExtractionError(ServiceError):
    """Exception raised when content extraction fails."""


class ChunkGenerationError(ServiceError):
    """Exception raised when chunk generation fails."""


def _raise_scraping_error(error_msg: str) -> None:
    """Helper function to raise scraping error."""
    raise ScrapingError(error_msg)


def _raise_content_extraction_error() -> None:
    """Helper function to raise content extraction error."""
    msg = "No content extracted"
    raise ContentExtractionError(msg)


def _raise_chunk_generation_error() -> None:
    """Helper function to raise chunk generation error."""
    msg = "No chunks generated"
    raise ChunkGenerationError(msg)


logger = logging.getLogger(__name__)
console = Console()


class ProcessingState(BaseModel):
    """State tracking for resumable processing."""

    urls_to_process: list[str] = Field(default_factory=list)
    completed_urls: list[str] = Field(default_factory=list)
    failed_urls: dict[str, str] = Field(default_factory=dict)
    total_chunks_processed: int = 0
    total_embeddings_generated: int = 0
    start_time: datetime = Field(default_factory=datetime.now)
    last_checkpoint: datetime = Field(default_factory=datetime.now)
    collection_name: str = "bulk_embeddings"


class BulkEmbedder:  # pylint: disable=too-many-instance-attributes
    """Bulk embedder for crawling and embedding web content."""

    def __init__(
        self,
        config: Settings,
        collection_name: str = "bulk_embeddings",
        state_file: Path | None = None,
        *,
        container: ApplicationContainer | None = None,
    ):
        """Initialize bulk embedder.

        Args:
            config: Unified configuration
            collection_name: Target vector collection name
            state_file: Optional state file for resumability
            container: Optional DI container override (useful for testing)
        """

        self.config = config
        self.collection_name = collection_name
        self.state_file = state_file or Path(".crawl4ai_state.json")
        self.state = self._load_state()
        self._container_override = container

        # Services (initialized in async context)
        self.crawl_manager: Any | None = None
        self.embedding_manager: Any | None = None
        self.vector_service: VectorStoreService | None = None

    def _load_state(self) -> ProcessingState:
        """Load processing state from file if exists."""

        if self.state_file.exists():
            try:
                data = self._load_state_data()
            except (ImportError, OSError, PermissionError):
                logger.warning("Failed to load state")
                return ProcessingState(collection_name=self.collection_name)

            try:
                return self._validate_state_data(data)
            except (ValueError, KeyError, TypeError) as e:
                logger.warning("Invalid state data, starting fresh: %s", e)
                return ProcessingState(collection_name=self.collection_name)
        return ProcessingState(collection_name=self.collection_name)

    def _load_state_data(self) -> dict:
        """Load state data from file."""

        with Path(self.state_file).open(encoding="utf-8") as f:
            return json.load(f)

    def _validate_state_data(self, data: dict) -> ProcessingState:
        """Validate and create state from data."""

        state = ProcessingState.model_validate(data)
        logger.info("Resumed from state: %d completed", len(state.completed_urls))
        return state

    def _save_state(self) -> None:
        """Save current processing state."""

        self.state.last_checkpoint = datetime.now(tz=UTC)
        with Path(self.state_file).open("w", encoding="utf-8") as f:
            json.dump(self.state.model_dump(mode="json"), f, indent=2, default=str)

    async def initialize_services(self) -> None:
        """Initialize all required services."""

        container = self._container_override or get_container()
        if container is None:
            container = await initialize_container(self.config)
        elif self._container_override is not None:
            # When an override is supplied, rely on it directly without lifecycle calls.
            container = self._container_override

        if self.crawl_manager is None:
            self.crawl_manager = container.browser_manager()
        if self.crawl_manager is None:
            raise RuntimeError("Crawl manager unavailable")

        if self.embedding_manager is None:
            self.embedding_manager = container.embedding_manager()
        if self.embedding_manager is None:
            raise RuntimeError("Embedding manager unavailable")

        self.vector_service = container.vector_store_service()
        if self.vector_service is None:
            raise RuntimeError("Vector store service unavailable")

        if (
            hasattr(self.vector_service, "is_initialized")
            and not self.vector_service.is_initialized()
        ):
            await self.vector_service.initialize()

        # Create collection if it doesn't exist
        collections = await self.vector_service.list_collections()
        if self.collection_name not in collections:
            generate_sparse_flag = bool(
                getattr(self.config.fastembed, "generate_sparse", False)
            )
            schema = CollectionSchema(
                name=self.collection_name,
                vector_size=self.vector_service.embedding_dimension,
                distance="cosine",
                requires_sparse=generate_sparse_flag,
            )
            await self.vector_service.ensure_collection(schema)
            logger.info("Created vector collection '%s'", self.collection_name)

    async def load_urls_from_file(self, file_path: Path) -> list[str]:
        """Load URLs from various file formats."""

        urls = []

        if file_path.suffix == ".csv":
            async with aiofiles.open(file_path) as f:
                content = await f.read()
                reader = csv.DictReader(content.splitlines())
                for row in reader:
                    # Try common column names
                    url = (
                        row.get("url")
                        or row.get("URL")
                        or row.get("link")
                        or row.get("website")
                    )
                    if url:
                        urls.append(url.strip())
        elif file_path.suffix == ".json":
            async with aiofiles.open(file_path) as f:
                content = await f.read()
                data = json.loads(content)
                if isinstance(data, list):
                    urls = [
                        item if isinstance(item, str) else item.get("url")
                        for item in data
                    ]
                    urls = [url for url in urls if url]
                elif isinstance(data, dict) and "urls" in data:
                    urls = data["urls"]
        elif file_path.suffix == ".txt":
            async with aiofiles.open(file_path) as f:
                content = await f.read()
                urls = [
                    line.strip()
                    for line in content.splitlines()
                    if line.strip() and not line.startswith("#")
                ]
        else:
            # Try as plain text
            async with aiofiles.open(file_path) as f:
                content = await f.read()
                urls = [
                    line.strip()
                    for line in content.splitlines()
                    if line.strip() and not line.startswith("#")
                ]

        return urls

    async def load_urls_from_sitemap(self, sitemap_url: str) -> list[str]:
        """Load URLs from a sitemap."""

        urls = []

        async with httpx.AsyncClient() as client:
            response = await client.get(sitemap_url)
            response.raise_for_status()

            root = ElementTree.fromstring(response.content)
            # Handle both regular sitemaps and sitemap indexes
            for url_elem in root.findall(
                ".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"
            ):
                loc = url_elem.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
                if loc is not None and loc.text:
                    urls.append(loc.text.strip())

            # Check for sitemap index
            for sitemap_elem in root.findall(
                ".//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap"
            ):
                loc = sitemap_elem.find(
                    "{http://www.sitemaps.org/schemas/sitemap/0.9}loc"
                )
                if loc is not None and loc.text:
                    # Recursively load sub-sitemaps
                    sub_urls = await self.load_urls_from_sitemap(loc.text.strip())
                    urls.extend(sub_urls)

        return urls

    async def process_url(self, url: str) -> dict[str, Any]:
        """Process a single URL: scrape, chunk, embed, and store."""

        result = {
            "url": url,
            "success": False,
            "chunks": 0,
            "error": None,
        }

        try:
            result = await self._execute_processing_pipeline(url, result)

        except Exception as e:
            result["error"] = str(e)
            logger.exception("Failed to process %s", url)

        return result

    async def _execute_processing_pipeline(
        self, url: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute the complete URL processing pipeline."""
        # Step 1: Scrape and extract content
        scrape_result, content_to_chunk, _ = await self._scrape_and_extract(url)

        # Step 2: Chunk the content
        chunks = await self._chunk_content(content_to_chunk)

        # Step 3: Generate embeddings
        dense_embeddings, sparse_embeddings = await self._generate_embeddings(chunks)

        # Step 4: Prepare and store points
        await self._store_points(
            url=url,
            chunks=chunks,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
            scrape_result=scrape_result,
        )

        result["success"] = True
        result["chunks"] = len(chunks)
        return result

    async def _scrape_and_extract(
        self, url: str
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """Scrape URL and extract content for processing."""
        scrape_result: dict[str, Any]
        try:
            if self.crawl_manager is None:
                raise RuntimeError("Crawl manager not initialized")
            crawl_manager = cast(Any, self.crawl_manager)
            scrape_result = await crawl_manager.scrape_url(url=url)
        except (httpx.HTTPError, ValueError, ConnectionError, TimeoutError) as exc:
            msg = f"Scraping failed: {exc}"
            raise ScrapingError(msg) from exc

        if not scrape_result.get("success"):
            msg = scrape_result.get("error", "Scraping failed")
            raise ScrapingError(msg)

        content = scrape_result.get("content", {})
        markdown_content = content.get("markdown", "")
        text_content = content.get("text", "")
        metadata = scrape_result.get("metadata", {})

        # Use markdown if available, fallback to text
        if not (content_to_chunk := markdown_content or text_content):
            _raise_content_extraction_error()

        return scrape_result, content_to_chunk, metadata

    async def _chunk_content(self, content_to_chunk: str) -> list[dict[str, Any]]:
        """Chunk content using DocumentChunker."""
        try:
            chunker = DocumentChunker(self.config.chunking)
        except Exception as e:
            error_msg = f"Chunker initialization failed: {e}"
            raise ChunkGenerationError(error_msg) from e

        try:
            chunks = chunker.chunk_content(content_to_chunk)
        except Exception as e:
            error_msg = f"Chunking failed: {e}"
            raise ChunkGenerationError(error_msg) from e

        if not chunks:
            _raise_chunk_generation_error()

        return chunks

    async def _generate_embeddings(
        self, chunks: list[dict[str, Any]]
    ) -> tuple[list[Any], list[Any]]:
        """Generate embeddings for chunks."""
        try:
            texts = [chunk["content"] for chunk in chunks]
        except (KeyError, TypeError, AttributeError) as e:
            error_msg = f"Text extraction failed: {e}"
            raise RuntimeError(error_msg) from e

        try:
            if self.embedding_manager is None:
                raise RuntimeError("Embedding manager not initialized")
            embedding_manager = cast(Any, self.embedding_manager)
            generate_sparse_flag = bool(
                getattr(self.config.fastembed, "generate_sparse", False)
            )
            embedding_result = await embedding_manager.generate_embeddings(
                texts=texts,
                quality_tier=QualityTier.BALANCED,
                auto_select=True,
                generate_sparse=generate_sparse_flag,
            )
        except (ValueError, ConnectionError, RuntimeError) as e:
            error_msg = f"Embedding generation failed: {e}"
            raise RuntimeError(error_msg) from e

        dense_embeddings = embedding_result.get("embeddings", [])
        sparse_embeddings = embedding_result.get("sparse_embeddings", [])

        return dense_embeddings, sparse_embeddings

    async def _store_points(  # pylint: disable=too-many-arguments
        self,
        *,
        url: str,
        chunks: list[dict[str, Any]],
        dense_embeddings: list[Any],
        sparse_embeddings: list[Any],
        scrape_result: dict[str, Any],
    ) -> None:
        """Prepare and store vector records in the unified vector service."""
        try:
            records = self._prepare_records(
                url=url,
                chunks=chunks,
                dense_embeddings=dense_embeddings,
                sparse_embeddings=sparse_embeddings,
                scrape_result=scrape_result,
            )
        except (ValueError, TypeError, KeyError) as e:
            error_msg = f"Point preparation failed: {e}"
            raise RuntimeError(error_msg) from e

        if not self.vector_service:
            raise RuntimeError("Vector store service not initialized")

        try:
            await self.vector_service.upsert_vectors(
                self.collection_name,
                records,
                batch_size=100,
            )
        except (ConnectionError, ValueError, RuntimeError) as e:
            error_msg = f"Point storage failed: {e}"
            raise RuntimeError(error_msg) from e

    def _prepare_records(  # pylint: disable=too-many-arguments
        self,
        *,
        url: str,
        chunks: list[dict[str, Any]],
        dense_embeddings: list[Any],
        sparse_embeddings: list[Any],
        scrape_result: dict[str, Any],
    ) -> list[VectorRecord]:
        """Prepare records for vector store ingestion."""

        metadata = scrape_result.get("metadata", {})
        records: list[VectorRecord] = []
        for i, (chunk, embedding) in enumerate(
            zip(chunks, dense_embeddings, strict=False)
        ):
            point_id = f"{urlparse(url).netloc}_{datetime.now(tz=UTC).timestamp()}_{i}"

            payload = {
                "url": url,
                "title": metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "content": chunk["content"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "start_char": chunk.get("start_pos", 0),
                "end_char": chunk.get("end_pos", 0),
                "chunk_type": chunk.get("chunk_type", "text"),
                "has_code": chunk.get("has_code", False),
                "scraped_at": datetime.now(tz=UTC).isoformat(),
                "provider": scrape_result.get("provider", "unknown"),
            }

            sparse_vector = None
            if sparse_embeddings and i < len(sparse_embeddings):
                sparse_vector = sparse_embeddings[i]

            records.append(
                VectorRecord(
                    id=point_id,
                    vector=list(embedding),
                    payload=payload,
                    sparse_vector=sparse_vector,
                )
            )

        return records

    async def process_urls_batch(
        self,
        urls: list[str],
        max_concurrent: int = 5,
        progress: Progress | None = None,
    ) -> dict[str, Any]:
        """Process URLs in batches with concurrency control."""

        semaphore = asyncio.Semaphore(max_concurrent)

        task_id: int | None = None
        if progress:
            task_id = progress.add_task(
                "[green]Processing URLs...",
                total=len(urls),
            )

        async def process_with_semaphore(url: str) -> dict[str, Any]:
            """Process a single URL with semaphore control."""

            async with semaphore:
                result = await self.process_url(url)

                if result["success"]:
                    self.state.completed_urls.append(url)
                    self.state.total_chunks_processed += result["chunks"]
                    self.state.total_embeddings_generated += result["chunks"]
                else:
                    self.state.failed_urls[url] = result["error"]

                if progress and task_id is not None:
                    progress.update(task_id, advance=1)

                if len(self.state.completed_urls) % 10 == 0:
                    self._save_state()

                return result

        # Create tasks
        tasks = []

        for url in urls:
            task = asyncio.create_task(process_with_semaphore(url))
            tasks.append(task)

        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        failed = len(results) - successful

        return {
            "total": len(urls),
            "successful": successful,
            "failed": failed,
            "results": results,
        }

    async def run(
        self,
        urls: list[str],
        max_concurrent: int = 5,
        resume: bool = True,
    ) -> None:
        """Run the bulk embedding pipeline."""

        # Filter out already completed URLs if resuming
        if resume and self.state.completed_urls:
            urls = [url for url in urls if url not in self.state.completed_urls]
            console.print(f"[yellow]Resuming: {len(urls)} URLs remaining[/yellow]")

        if not urls:
            console.print("[green]All URLs already processed![/green]")
            return

        # Update state
        self.state.urls_to_process = urls

        # Initialize services
        console.print("[cyan]Initializing services...[/cyan]")
        await self.initialize_services()

        # Process URLs with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            results = await self.process_urls_batch(
                urls=urls,
                max_concurrent=max_concurrent,
                progress=progress,
            )

        # Save final state
        self._save_state()

        # Display summary
        self._display_summary(results)

    def _display_summary(self, results: dict[str, Any]) -> None:
        """Display processing summary."""

        table = Table(title="Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total URLs", str(results["total"]))
        table.add_row("Successful", str(results["successful"]))
        table.add_row("Failed", str(results["failed"]))
        table.add_row("Total Chunks", str(self.state.total_chunks_processed))
        table.add_row("Total Embeddings", str(self.state.total_embeddings_generated))

        duration = datetime.now(tz=UTC) - self.state.start_time
        table.add_row("Duration", str(duration).split(".", maxsplit=1)[0])

        console.print(table)

        # Show failed URLs if any
        if self.state.failed_urls:
            console.print("\n[red]Failed URLs:[/red]")
            for _url, _error in self.state.failed_urls.items():
                console.print("  • {url}")


@click.command()
@click.option(
    "--urls",
    "-u",
    multiple=True,
    help="Individual URLs to process",
)
@click.option(
    "--file",
    "-f",
    type=click.Path(exists=True, path_type=Path),
    help="File containing URLs (txt, csv, json)",
)
@click.option(
    "--sitemap",
    "-s",
    help="Sitemap URL to crawl",
)
@click.option(
    "--collection",
    "-c",
    default="bulk_embeddings",
    help="Qdrant collection name",
)
@click.option(
    "--concurrent",
    "-n",
    default=5,
    type=int,
    help="Maximum concurrent requests",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option(
    "--state-file",
    type=click.Path(path_type=Path),
    default=".crawl4ai_state.json",
    help="State file for resumability",
)
@click.option(
    "--no-resume",
    is_flag=True,
    help="Start fresh, ignore previous state",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def main(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    urls: tuple[str, ...],
    file: Path | None = None,
    sitemap: str | None = None,
    collection: str = "bulk_embeddings",
    concurrent: int = 5,
    config_path: Path | None = None,
    state_file: Path | None = None,
    no_resume: bool = False,
    verbose: bool = False,
) -> None:
    """Crawl4AI Bulk Embedder - High-performance web scraping and embedding pipeline.

    This tool crawls URLs, chunks content, generates embeddings, and stores them
    in a Qdrant vector database for semantic search.

    Examples:
        # Process individual URLs
        crawl4ai-bulk-embedder -u https://example.com -u https://docs.example.com

        # Process URLs from file
        crawl4ai-bulk-embedder -f urls.txt

        # Crawl from sitemap
        crawl4ai-bulk-embedder -s https://example.com/sitemap.xml

        # Custom configuration
        crawl4ai-bulk-embedder -f urls.csv --config config.json --concurrent 10

    """
    # Load configuration
    config = get_settings()

    # Setup logging
    configure_logging(
        level="DEBUG" if verbose else "INFO",
        enable_color=True,
        settings=config,
    )
    if config_path is not None:
        logger.debug("CLI config path provided: %s", config_path)

    # Validate inputs
    if not any([urls, file, sitemap]):
        console.print("[red]Error: Must provide URLs via -u, -f, or -s[/red]")
        sys.exit(1)

    # Run async main
    asyncio.run(
        _async_main(
            urls=list(urls),
            file=file,
            sitemap=sitemap,
            collection=collection,
            concurrent=concurrent,
            config=config,
            state_file=state_file or Path(".crawl4ai_state.json"),
            resume=not no_resume,
        )
    )


async def _async_main(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    urls: list[str],
    file: Path | None = None,
    sitemap: str | None = None,
    collection: str = "bulk_embeddings",
    concurrent: int = 5,
    config: Settings,
    state_file: Path,
    resume: bool = True,
) -> None:
    """Async main function."""
    # Initialize container
    await initialize_container(config)

    # Create embedder
    embedder = BulkEmbedder(
        config=config,
        collection_name=collection,
        state_file=state_file,
    )

    # Collect all URLs
    all_urls = list(urls)

    # Load from file
    if file:
        console.print(f"[cyan]Loading URLs from {file}...[/cyan]")
        file_urls = await embedder.load_urls_from_file(file)
        all_urls.extend(file_urls)
        console.print(f"[green]Loaded {len(file_urls)} URLs from file[/green]")

    # Load from sitemap
    if sitemap:
        console.print(f"[cyan]Loading URLs from sitemap {sitemap}...[/cyan]")
        sitemap_urls = await embedder.load_urls_from_sitemap(sitemap)
        all_urls.extend(sitemap_urls)
        console.print(f"[green]Loaded {len(sitemap_urls)} URLs from sitemap[/green]")

    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in all_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    console.print(f"\n[bold]Total unique URLs to process: {len(unique_urls)}[/bold]")

    try:
        # Run the pipeline
        await embedder.run(
            urls=unique_urls,
            max_concurrent=concurrent,
            resume=resume,
        )
    finally:
        # Cleanup
        await shutdown_container()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter  # type: ignore[call-arg]  # Click handles CLI arguments
