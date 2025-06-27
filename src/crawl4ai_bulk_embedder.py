#!/usr/bin/env python3
"""Crawl4AI Bulk Embedder - High-performance bulk web scraping and embedding pipeline.

This tool provides a CLI for bulk scraping URLs using Crawl4AI and generating embeddings
for storage in Qdrant vector database. It supports concurrent processing, resumability,
and various input formats.
"""

import asyncio
import csv
import json
import logging
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import click
import httpx
from defusedxml import ElementTree
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .chunking import DocumentChunker
from .config import Config, get_config
from .infrastructure.client_manager import ClientManager
from .services.embeddings.manager import QualityTier


class ScrapingError(Exception):
    """Exception raised when web scraping fails."""


class ContentExtractionError(Exception):
    """Exception raised when content extraction fails."""


class ChunkGenerationError(Exception):
    """Exception raised when chunk generation fails."""


from .services.logging_config import configure_logging


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


class BulkEmbedder:
    """Bulk embedder for crawling and embedding web content."""

    def __init__(
        self,
        config: Config,
        client_manager: ClientManager,
        collection_name: str = "bulk_embeddings",
        state_file: Path | None = None,
    ):
        """Initialize bulk embedder.

        Args:
            config: Unified configuration
            client_manager: Client manager for services
            collection_name: Name of Qdrant collection
            state_file: Optional state file for resumability
        """
        self.config = config
        self.client_manager = client_manager
        self.collection_name = collection_name
        self.state_file = state_file or Path(".crawl4ai_state.json")
        self.state = self._load_state()

        # Services (initialized in async context)
        self.crawl_manager = None
        self.embedding_manager = None
        self.qdrant_service = None

    def _load_state(self) -> ProcessingState:
        """Load processing state from file if exists."""
        if self.state_file.exists():
            try:
                with Path(self.state_file).open() as f:
                    data = json.load(f)
                    state = ProcessingState.model_validate(data)
                    logger.info(
                        f"Resumed from state: {len(state.completed_urls)} completed"
                    )
                    return state
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
        return ProcessingState(collection_name=self.collection_name)

    def _save_state(self) -> None:
        """Save current processing state."""
        self.state.last_checkpoint = datetime.now(tz=UTC)
        with Path(self.state_file).open("w") as f:
            json.dump(self.state.model_dump(mode="json"), f, indent=2, default=str)

    async def initialize_services(self) -> None:
        """Initialize all required services."""
        # Get services from client manager
        self.crawl_manager = await self.client_manager.get_crawl_manager()
        self.embedding_manager = await self.client_manager.get_embedding_manager()
        self.qdrant_service = await self.client_manager.get_qdrant_service()

        # Create collection if it doesn't exist
        collections = await self.qdrant_service.list_collections()
        if self.collection_name not in collections:
            await self.qdrant_service.create_collection(
                collection_name=self.collection_name,
                vector_size=self.config.openai.dimensions or 1536,
                distance="Cosine",
                sparse_vector_name="sparse"
                if self.config.fastembed.generate_sparse
                else None,
                enable_quantization=True,
                collection_type="general",
            )
            logger.info(f"Created collection: {self.collection_name}")

    async def load_urls_from_file(self, file_path: Path) -> list[str]:
        """Load URLs from various file formats."""
        urls = []

        if file_path.suffix == ".csv":
            with Path(file_path).open() as f:
                reader = csv.DictReader(f)
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
            with Path(file_path).open() as f:
                data = json.load(f)
                if isinstance(data, list):
                    urls = [
                        item if isinstance(item, str) else item.get("url")
                        for item in data
                    ]
                    urls = [url for url in urls if url]
                elif isinstance(data, dict) and "urls" in data:
                    urls = data["urls"]
        elif file_path.suffix == ".txt":
            with Path(file_path).open() as f:
                urls = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
        else:
            # Try as plain text
            with Path(file_path).open() as f:
                urls = [
                    line.strip()
                    for line in f
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
            # Scrape the URL
            scrape_result = await self.crawl_manager.scrape_url(url=url)

            if not scrape_result.get("success"):
                raise ScrapingError(scrape_result.get("error", "Scraping failed"))

            content = scrape_result.get("content", {})
            markdown_content = content.get("markdown", "")
            text_content = content.get("text", "")
            metadata = scrape_result.get("metadata", {})

            # Use markdown if available, fallback to text
            content_to_chunk = markdown_content or text_content

            if not content_to_chunk:
                raise ContentExtractionError("No content extracted")

            # Chunk the content using DocumentChunker
            chunker = DocumentChunker(self.config.chunking)
            chunk_results = chunker.chunk_text(content_to_chunk)
            chunks = chunk_results.chunks

            if not chunks:
                raise ChunkGenerationError("No chunks generated")

            # Generate embeddings for all chunks
            texts = [chunk.content for chunk in chunks]
            embedding_result = await self.embedding_manager.generate_embeddings(
                texts=texts,
                quality_tier=QualityTier.BALANCED,
                auto_select=True,
                generate_sparse=self.config.fastembed.generate_sparse,
            )

            dense_embeddings = embedding_result.get("embeddings", [])
            sparse_embeddings = embedding_result.get("sparse_embeddings", [])

            # Prepare points for Qdrant
            points = []
            for i, (chunk, embedding) in enumerate(
                zip(chunks, dense_embeddings, strict=False)
            ):
                point_id = (
                    f"{urlparse(url).netloc}_{datetime.now(tz=UTC).timestamp()}_{i}"
                )

                payload = {
                    "url": url,
                    "title": metadata.get("title", ""),
                    "description": metadata.get("description", ""),
                    "content": chunk.content,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "start_char": chunk.start_pos,
                    "end_char": chunk.end_pos,
                    "chunk_type": chunk.chunk_type,
                    "has_code": chunk.has_code,
                    "scraped_at": datetime.now(tz=UTC).isoformat(),
                    "provider": scrape_result.get("provider", "unknown"),
                }

                point = {
                    "id": point_id,
                    "vector": embedding,
                    "payload": payload,
                }

                # Add sparse vector if available
                if sparse_embeddings and i < len(sparse_embeddings):
                    point["sparse_vector"] = sparse_embeddings[i]

                points.append(point)

            # Store in Qdrant
            await self.qdrant_service.upsert_points(
                collection_name=self.collection_name,
                points=points,
                batch_size=100,
            )

            result["success"] = True
            result["chunks"] = len(chunks)

        except Exception as e:
            result["error"] = str(e)
            logger.exception(f"Failed to process {url}: {e}")

        return result

    async def process_urls_batch(
        self,
        urls: list[str],
        max_concurrent: int = 5,
        progress: Progress | None = None,
    ) -> dict[str, Any]:
        """Process URLs in batches with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(url: str) -> dict[str, Any]:
            async with semaphore:
                result = await self.process_url(url)

                # Update state
                if result["success"]:
                    self.state.completed_urls.append(url)
                    self.state.total_chunks_processed += result["chunks"]
                    self.state.total_embeddings_generated += result["chunks"]
                else:
                    self.state.failed_urls[url] = result["error"]

                # Update progress
                if progress:
                    task_id = getattr(process_with_semaphore, "task_id", None)
                    if task_id:
                        progress.update(task_id, advance=1)

                # Save state periodically
                if len(self.state.completed_urls) % 10 == 0:
                    self._save_state()

                return result

        # Create tasks
        tasks = []
        if progress:
            task_id = progress.add_task(
                "[green]Processing URLs...",
                total=len(urls),
            )
            process_with_semaphore.task_id = task_id

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
        table.add_row("Duration", str(duration).split(".")[0])

        console.print(table)

        # Show failed URLs if any
        if self.state.failed_urls:
            console.print("\n[red]Failed URLs:[/red]")
            for url, error in self.state.failed_urls.items():
                console.print(f"  • {url}: {error}")


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
def main(
    urls: tuple[str, ...],
    file: Path | None,
    sitemap: str | None,
    collection: str,
    concurrent: int,
    _config_path: Path | None,
    state_file: Path,
    no_resume: bool,
    verbose: bool,
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
    # Setup logging
    configure_logging(
        level="DEBUG" if verbose else "INFO",
        enable_color=True,
    )

    # Load configuration
    config = get_config()

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
            state_file=state_file,
            resume=not no_resume,
        )
    )


async def _async_main(
    urls: list[str],
    file: Path | None,
    sitemap: str | None,
    collection: str,
    concurrent: int,
    config: Config,
    state_file: Path,
    resume: bool,
) -> None:
    """Async main function."""
    # Initialize client manager
    client_manager = ClientManager(config)

    # Create embedder
    embedder = BulkEmbedder(
        config=config,
        client_manager=client_manager,
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
        await client_manager.cleanup()


if __name__ == "__main__":
    main()
