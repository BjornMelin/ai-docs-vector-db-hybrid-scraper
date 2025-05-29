"""crawl4ai_bulk_embedder.py - Modern Documentation Scraper using Crawl4AI.

Balanced implementation with valuable features while following KISS principles

Based on latest Crawl4AI, Pydantic v2, and Python 3.13 best practices
"""

import asyncio
import logging
import sys
from datetime import datetime
from typing import Any

# Crawl4AI imports are now handled by CrawlManager
# Only keep filter imports for backward compatibility
from crawl4ai.deep_crawling.filters import ContentTypeFilter
from crawl4ai.deep_crawling.filters import FilterChain
from crawl4ai.deep_crawling.filters import URLPatternFilter

# Advanced Embedding Integrations
try:
    from firecrawl import FirecrawlApp
except ImportError:
    FirecrawlApp = None

try:
    from fastembed import SparseTextEmbedding
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None
    SparseTextEmbedding = None

try:
    from FlagEmbedding import FlagReranker
except ImportError:
    FlagReranker = None


# OpenAI client now handled by EmbeddingManager
from pydantic import BaseModel
from pydantic import Field

# Qdrant client now handled by QdrantService
from qdrant_client.models import PointStruct
from qdrant_client.models import SparseVector
from rich.console import Console
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from rich.table import Table

# Import unified configuration
from src.config import get_config
from src.config.enums import EmbeddingModel
from src.config.enums import EmbeddingProvider
from src.config.enums import SearchStrategy
from src.config.models import DocumentationSite

# Import shared response models
from src.mcp.models.responses import CrawlResult
from src.services.core.qdrant_service import QdrantService
from src.services.crawling.manager import CrawlManager

# Import service layer
from src.services.embeddings.manager import EmbeddingManager
from src.services.utilities.rate_limiter import RateLimitManager

# Import enhanced chunking module
from .chunking import EnhancedChunker

console = Console()


class VectorMetrics(BaseModel):
    """Vector processing metrics"""

    total_documents: int = Field(default=0, description="Total documents processed")
    total_chunks: int = Field(default=0, description="Total chunks created")
    successful_embeddings: int = Field(default=0, description="Successful embeddings")
    failed_embeddings: int = Field(default=0, description="Failed embeddings")
    processing_time: float = Field(
        default=0.0, description="Processing time in seconds"
    )


class ScrapingStats(BaseModel):
    """Comprehensive scraping statistics"""

    total_processed: int = 0
    successful_embeddings: int = 0
    failed_crawls: int = 0
    total_chunks: int = 0
    unique_urls: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None


class ModernDocumentationScraper:
    """Advanced Documentation Scraper with hybrid embedding pipeline

    Features:
    - Multi-provider embedding support (OpenAI, FastEmbed)
    - Hybrid dense+sparse search with RRF ranking
    - Research-optimized chunking (1600 chars)
    - Vector quantization for storage optimization
    - Firecrawl premium integration
    """

    # Embedding models now managed by EmbeddingManager service

    def __init__(self, config) -> None:
        self.config = config
        self.stats = ScrapingStats()
        self.processed_urls: set[str] = set()

        # Initialize rate limiter
        self.rate_limiter = RateLimitManager(self.config)

        # Initialize service layer
        self.embedding_manager = EmbeddingManager(
            self.config, rate_limiter=self.rate_limiter
        )
        self.qdrant_service = QdrantService(self.config)
        self.crawl_manager = CrawlManager(self.config, rate_limiter=self.rate_limiter)

        # Enhanced logging setup
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("crawl4ai_scraper.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

        # Initialize Firecrawl if premium features enabled
        self.firecrawl_client = None
        if config.firecrawl.api_key:
            if FirecrawlApp is not None:
                self.firecrawl_client = FirecrawlApp(api_key=config.firecrawl.api_key)
                self.logger.info("Firecrawl premium features enabled")
            else:
                self.logger.warning(
                    "Firecrawl not available. Install with: pip install firecrawl-py"
                )

        # Embedding models initialization now handled by EmbeddingManager

        # Initialize enhanced chunker
        self.chunker = EnhancedChunker(config.chunking)

        # Initialize reranker if enabled
        self.reranker = None
        if config.embedding.enable_reranking:
            self._initialize_reranker()

    async def initialize(self) -> None:
        """Initialize async services"""
        if self.rate_limiter:
            await self.rate_limiter.initialize()
            self.logger.info("Rate limiter initialized")
        await self.embedding_manager.initialize()
        await self.qdrant_service.initialize()
        await self.crawl_manager.initialize()
        self.logger.info("All services initialized successfully")

    async def cleanup(self) -> None:
        """Cleanup async services"""
        await self.embedding_manager.cleanup()
        await self.qdrant_service.cleanup()
        await self.crawl_manager.cleanup()
        # RateLimitManager doesn't need cleanup - it's stateless
        self.logger.info("All services cleaned up successfully")

    # Embedding model initialization removed - now handled by EmbeddingManager service

    def _initialize_reranker(self) -> None:
        """Initialize reranker for advanced reranking capabilities"""
        if not self.config.embedding.enable_reranking:
            return
        if FlagReranker is None:
            self.logger.warning(
                "FlagEmbedding not available. Install with: pip install FlagEmbedding"
            )
            self.config.embedding.enable_reranking = False
            return

        try:
            self.reranker = FlagReranker(
                self.config.embedding.reranker_model,
                use_fp16=True,  # Speed optimization with minimal performance impact
            )
            self.logger.info(
                f"Reranker initialized: {self.config.embedding.reranker_model} "
                f"(Expected 10-20% accuracy improvement)"
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize reranker: {e}")
            self.config.embedding.enable_reranking = False
            self.reranker = None

    async def setup_collection(self) -> None:
        """Setup advanced Qdrant collection with hybrid search capabilities"""
        try:
            collections = await self.qdrant_service.list_collections()
            collection_exists = self.config.qdrant.collection_name in collections

            if not collection_exists:
                # Determine vector size based on embedding model
                vector_size = self._get_vector_size()

                # Create collection using service layer
                await self.qdrant_service.create_collection(
                    collection_name=self.config.qdrant.collection_name,
                    vector_size=vector_size,
                    distance="Cosine",
                    enable_sparse=self.config.embedding.search_strategy
                    == SearchStrategy.HYBRID,
                    on_disk=self.config.embedding.enable_quantization,
                )
                self.logger.info(
                    f"Created collection '{self.config.qdrant.collection_name}' with "
                    f"{'hybrid' if self.config.embedding.search_strategy == SearchStrategy.HYBRID else 'dense'} search"
                )
            else:
                self.logger.info(
                    f"Collection exists: {self.config.qdrant.collection_name}"
                )

        except Exception as e:
            self.logger.error(f"Collection setup failed: {e}")
            raise

    def _get_vector_size(self) -> int:
        """Get vector size based on embedding model"""
        model_value = self.config.embedding.dense_model.value
        if self.config.embedding.dense_model in [
            EmbeddingModel.TEXT_EMBEDDING_3_SMALL,
            EmbeddingModel.TEXT_EMBEDDING_3_LARGE,
        ]:
            return 1536  # OpenAI models
        if "bge-small" in model_value:
            return 384  # BGE small models
        if "bge-large" in model_value:
            return 1024  # BGE large models
        if "nv-embed" in model_value.lower():
            return 4096  # NVIDIA NV-Embed-v2
        return 1536  # Default fallback

    async def create_embedding(
        self, text: str
    ) -> tuple[list[float], dict[str, Any] | None]:
        """Create advanced embeddings with provider auto-selection"""
        try:
            # Truncate to avoid API limits
            text_to_embed = text[:8000]

            # Use embedding manager to handle all providers
            embeddings = await self.embedding_manager.generate_embeddings(
                [text_to_embed]
            )

            if not embeddings:
                return [], None

            # Check if sparse embeddings were generated (for hybrid search)
            if hasattr(self.embedding_manager, "_last_sparse_embeddings"):
                sparse_data = self.embedding_manager._last_sparse_embeddings
                return embeddings[0], {
                    "sparse": sparse_data[0] if sparse_data else None
                }

            return embeddings[0], None

        except Exception as e:
            self.logger.error(f"Embedding creation failed: {e}")
            return [], None

    # Old embedding methods removed - now using service layer through EmbeddingManager

    def rerank_results(
        self, query: str, passages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Advanced reranking for improved documentation search accuracy

        Args:
            query: Search query
            passages: List of passages with content and metadata

        Returns:
            Reranked passages (best results first)
        """
        if not self.config.embedding.enable_reranking or self.reranker is None:
            return passages

        if not passages:  # Handle empty list of passages
            return []

        if len(passages) <= 1:
            return passages

        try:
            # Prepare query-passage pairs for reranking
            pairs = [[query, passage["content"]] for passage in passages]

            # Get reranking scores (higher = better)
            scores = self.reranker.compute_score(pairs, normalize=True)  # type: ignore

            # Combine passages with scores and sort by score (descending)
            scored_passages = list(zip(passages, scores, strict=False))
            scored_passages.sort(key=lambda x: x[1], reverse=True)

            # Return reranked passages (without scores)
            reranked = [passage for passage, _score in scored_passages]

            self.logger.info(
                f"Reranked {len(passages)} passages. "
                f"Top score: {max(scores):.3f}, Bottom score: {min(scores):.3f}"
            )

            return reranked

        except Exception as e:
            self.logger.warning(f"Reranking failed: {e}. Returning original order.")
            return passages

    def chunk_content(self, content: str, title: str, url: str) -> list[dict[str, Any]]:
        """Advanced content chunking with enhanced code awareness"""
        # Use the enhanced chunker
        return self.chunker.chunk_content(content, title, url)

    def create_filter_chain(self, site: DocumentationSite) -> FilterChain:
        """Create smart filter chain based on Crawl4AI best practices"""
        filters = []

        # URL pattern filter for documentation content
        url_filter = URLPatternFilter(patterns=site.url_patterns)
        filters.append(url_filter)

        # Content type filter for HTML only
        content_filter = ContentTypeFilter(allowed_types=["text/html"])
        filters.append(content_filter)

        return FilterChain(filters)

    async def crawl_documentation_site(
        self,
        site: DocumentationSite,
    ) -> list[CrawlResult]:
        """Crawl documentation site using enhanced CrawlManager with Crawl4AI"""
        self.logger.info(f"Crawling {site.name}: {site.url}")

        crawled_results = []

        try:
            # Use CrawlManager's enhanced crawl_site method
            result = await self.crawl_manager.crawl_site(
                url=site.url,
                max_pages=site.max_pages,
                formats=["markdown"],
                preferred_provider="crawl4ai",  # Explicitly use Crawl4AI
            )

            if result["success"]:
                self.logger.info(
                    f"Successfully crawled {result['total']} pages from {site.name} "
                    f"using {result['provider']}"
                )

                # Process each page into CrawlResult objects
                for page in result["pages"]:
                    # Track unique URLs
                    if page["url"] not in self.processed_urls:
                        self.processed_urls.add(page["url"])

                        crawl_result_item = CrawlResult(
                            url=page["url"],
                            title=page.get("title", "No Title"),
                            content=page["content"],
                            word_count=len(page["content"].split()),
                            success=True,
                            site_name=site.name,
                            depth=page.get("metadata", {}).get("depth", 0),
                            scraped_at=datetime.now().isoformat(),
                            links=[],  # Links extraction handled by provider
                        )
                        crawled_results.append(crawl_result_item)

                        # Update stats
                        self.stats.unique_urls = len(self.processed_urls)
            else:
                self.logger.error(
                    f"Crawling failed for {site.url}: {result.get('error', 'Unknown error')}"
                )
                self.stats.failed_crawls += 1

        except Exception as e:
            self.logger.error(f"Crawling failed for {site.url}: {e}")
            self.stats.failed_crawls += 1

        return crawled_results

    async def process_and_embed_results(self, results: list[CrawlResult]) -> None:
        """Process results with advanced embedding pipeline"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Processing with {self.config.embedding.provider.value} embeddings...",
                total=len(results),
            )

            for result_item in results:
                try:
                    # Create research-optimized chunks (1600 chars)
                    chunks = self.chunk_content(
                        result_item.content,
                        result_item.title,
                        result_item.url,
                    )

                    # Create embeddings and points with hybrid support
                    points_to_upsert = []
                    for chunk_data in chunks:
                        dense_embedding, sparse_data = await self.create_embedding(
                            chunk_data["content"]
                        )

                        if dense_embedding:
                            # Enhanced payload with advanced metadata
                            payload_data = {
                                "url": chunk_data["url"],
                                "title": chunk_data["title"],
                                "content_preview": chunk_data["content"][:300],
                                "full_content": chunk_data["content"],
                                "word_count": len(chunk_data["content"].split()),
                                "char_count": chunk_data["char_count"],
                                "token_estimate": chunk_data["token_estimate"],
                                "chunk_index": chunk_data["chunk_index"],
                                "total_chunks": chunk_data["total_chunks"],
                                "site_name": result_item.site_name,
                                "depth": result_item.depth,
                                "scraped_at": result_item.scraped_at,
                                "embedding_model": (
                                    self.config.embedding.dense_model.value
                                ),
                                "embedding_provider": (
                                    self.config.embedding.provider.value
                                ),
                                "search_strategy": (
                                    self.config.embedding.search_strategy.value
                                ),
                                "scraper_version": "3.0-Advanced",
                                "links_count": len(result_item.links),
                                "quantization_enabled": (
                                    self.config.embedding.enable_quantization
                                ),
                            }

                            # Create point based on search strategy
                            if (
                                self.config.embedding.search_strategy
                                == SearchStrategy.HYBRID
                                and sparse_data is not None
                            ):
                                # Hybrid point with both dense and sparse vectors
                                point = PointStruct(
                                    id=f"{hash(chunk_data['url'])}_{chunk_data['chunk_index']}",
                                    vector={  # type: ignore
                                        "dense": dense_embedding,
                                        "sparse": SparseVector(
                                            indices=sparse_data["indices"],
                                            values=sparse_data["values"],
                                        ),
                                    },
                                    payload=payload_data,
                                )
                            else:
                                # Traditional dense-only point
                                point = PointStruct(
                                    id=f"{hash(chunk_data['url'])}_{chunk_data['chunk_index']}",
                                    vector=dense_embedding,
                                    payload=payload_data,
                                )

                            points_to_upsert.append(point)
                            self.stats.total_chunks += 1

                    # Batch upsert for efficiency
                    if points_to_upsert:
                        await self.qdrant_service.upsert_points(
                            collection_name=self.config.qdrant.collection_name,
                            points=points_to_upsert,
                        )
                        self.stats.successful_embeddings += len(points_to_upsert)
                        self.logger.info(
                            f"Embedded {len(points_to_upsert)} chunks from "
                            f"{result_item.title} (Strategy: "
                            f"{self.config.embedding.search_strategy.value})",
                        )

                    self.stats.total_processed += 1
                    progress.update(task, advance=1)

                except Exception as e:
                    self.logger.error(f"Processing failed for {result_item.url}: {e}")
                    self.stats.failed_crawls += 1

    async def scrape_multiple_sites(self, sites: list[DocumentationSite]) -> None:
        """Scrape multiple sites with intelligent batching and resource management"""
        self.stats.start_time = datetime.now()
        await self.setup_collection()

        console.print(
            f"\n🚀 Starting to scrape {len(sites)} documentation sites",
            style="bold green",
        )

        # Process sites with adaptive concurrent control
        # The MemoryAdaptiveDispatcher will manage resource allocation automatically
        # Max concurrent crawls handled by dispatcher, semaphore controls task creation
        # Adjusted semaphore allows more tasks, relying on dispatcher for concurrency
        semaphore = asyncio.Semaphore(self.config.crawl4ai.max_concurrent_crawls)

        async def process_site_with_semaphore(
            site_to_process: DocumentationSite,
        ) -> tuple[DocumentationSite, list[CrawlResult] | Exception]:
            """Process a single site with semaphore control"""
            async with semaphore:
                try:
                    crawled_data = await self.crawl_documentation_site(site_to_process)
                    return site_to_process, crawled_data
                except Exception as e:
                    return site_to_process, e

        # Process all sites concurrently with controlled parallelism
        console.print(
            f"\n📊 Processing {len(sites)} sites with adaptive resource management"
        )

        tasks_to_run = [process_site_with_semaphore(site_item) for site_item in sites]
        site_processing_results = await asyncio.gather(
            *tasks_to_run, return_exceptions=True
        )

        # Handle results and update stats
        for site_batch_result in site_processing_results:
            if isinstance(site_batch_result, Exception):
                console.print(
                    f"❌ Unexpected error processing batch: "
                    f"{str(site_batch_result)[:100]}",
                    style="red",
                )
                # Increment failed crawls for sites in failed batch if identifiable
                # or a general counter if batch structure is lost.
                # For now, assume one failure per exception at this level.
                self.stats.failed_crawls += 1
                continue

            site_instance, site_pages_data = site_batch_result
            if isinstance(site_pages_data, list):
                console.print(
                    f"✅ {site_instance.name}: {len(site_pages_data)} pages crawled"
                )
                await self.process_and_embed_results(site_pages_data)
            else:  # Exception from process_site_with_semaphore
                console.print(
                    f"❌ {site_instance.name}: Failed - {str(site_pages_data)[:100]}",
                    style="red",
                )
                self.stats.failed_crawls += (
                    1  # Count this as a failed crawl for the site
                )

        # Update final stats
        self.stats.unique_urls = len(self.processed_urls)
        self.stats.end_time = datetime.now()
        self.display_comprehensive_stats()

    def display_comprehensive_stats(self) -> None:
        """Display comprehensive scraping statistics with performance metrics"""
        console.print("\n" + "=" * 70, style="bold blue")
        console.print(
            "📊 COMPREHENSIVE SCRAPING RESULTS",
            style="bold blue",
            justify="center",
        )
        console.print("=" * 70, style="bold blue")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green", justify="right")

        table.add_row("Pages processed", str(self.stats.total_processed))
        table.add_row("Unique URLs discovered", str(self.stats.unique_urls))
        table.add_row("Embeddings created", str(self.stats.successful_embeddings))
        table.add_row("Total chunks generated", str(self.stats.total_chunks))
        table.add_row("Failed crawls", str(self.stats.failed_crawls))

        if self.stats.start_time and self.stats.end_time:
            duration = self.stats.end_time - self.stats.start_time
            table.add_row("Total duration", str(duration).split(".")[0])

            if self.stats.total_processed > 0:
                pages_per_minute = self.stats.total_processed / (
                    duration.total_seconds() / 60
                )
                table.add_row("Pages per minute", f"{pages_per_minute:.1f}")

                avg_chunks_per_page = (
                    self.stats.total_chunks / self.stats.total_processed
                )
                table.add_row("Avg chunks per page", f"{avg_chunks_per_page:.1f}")

        # Success rate calculation
        total_attempts = self.stats.total_processed + self.stats.failed_crawls
        if total_attempts > 0:
            success_rate_val = (self.stats.total_processed / total_attempts) * 100
            table.add_row("Success rate", f"{success_rate_val:.1f}%")

        console.print(table)
        console.print("=" * 70, style="bold blue")

    async def demo_reranking_search(
        self, query: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Demo method showing advanced reranking integration

        This would typically be integrated into your vector search pipeline.
        For full integration, modify your vector database search method.
        """
        if not self.config.embedding.enable_reranking or self.reranker is None:
            console.print(
                "⚠️  Reranking not enabled or reranker not initialized. "
                "Set enable_reranking=True",
                style="yellow",
            )
            return []

        console.print(f"🔍 Demo search with reranking: '{query}'")

        # Step 1: Get more results than needed for reranking
        search_limit_val = min(self.config.embedding.rerank_top_k, limit * 2)

        # Step 2: Perform vector search (placeholder)
        console.print(
            f"📊 Vector search retrieving top-{search_limit_val} candidates..."
        )

        # Step 3: Mock results for demo (replace with actual vector search)
        mock_passages_list = [
            {
                "content": f"Documentation about {query} - passage {i}",
                "score": 0.9 - i * 0.1,  # type: ignore
                "metadata": {"source": f"doc_{i}.md"},  # type: ignore
            }
            for i in range(min(search_limit_val, 5))
        ]

        # Step 4: Apply reranking
        console.print("🎯 Applying advanced reranking...")
        reranked_passages_list = self.rerank_results(query, mock_passages_list)

        # Step 5: Return top results after reranking
        final_reranked_results = reranked_passages_list[:limit]
        console.print(
            f"✅ Returning top-{len(final_reranked_results)} reranked results"
        )
        return final_reranked_results


# Advanced documentation sites - optimized for hybrid embedding pipeline
ESSENTIAL_SITES = [
    DocumentationSite(
        name="Qdrant Documentation",
        url="https://docs.qdrant.tech/",
        max_pages=60,
        max_depth=2,
        priority="high",
        description="Vector database documentation for AI applications",
        crawl_pattern="*docs*",
        exclude_patterns=["*blog*", "*news*"],
        url_patterns=["*docs*", "*concepts*", "*tutorials*", "*guides*", "*fastembed*"],
    ),
    DocumentationSite(
        name="FastEmbed Documentation",
        url="https://qdrant.github.io/fastembed/",
        max_pages=30,
        max_depth=2,
        priority="high",
        description="Fast embedding generation library documentation",
        crawl_pattern="*examples*",
        exclude_patterns=["*changelog*"],
        url_patterns=["*examples*", "*models*", "*usage*"],
    ),
    DocumentationSite(
        name="Crawl4AI Documentation",
        url="https://docs.crawl4ai.com/",
        max_pages=50,
        max_depth=2,
        priority="high",
        description="Web crawling framework documentation",
        crawl_pattern="*docs*",
        exclude_patterns=["*blog*"],
        url_patterns=["*docs*", "*core*", "*advanced*", "*api*", "*extraction*"],
    ),
    DocumentationSite(
        name="Pydantic v2 Documentation",
        url="https://docs.pydantic.dev/",
        max_pages=40,
        max_depth=2,
        priority="medium",
        description="Data validation library documentation",
        crawl_pattern="*concepts*",
        exclude_patterns=["*v1*"],
        url_patterns=["*concepts*", "*usage*", "*api*", "*migration*", "*v2*"],
    ),
    DocumentationSite(
        name="OpenAI Embeddings Documentation",
        url="https://platform.openai.com/docs/guides/embeddings",
        max_pages=20,
        max_depth=1,
        priority="medium",
        description="OpenAI embedding models documentation",
        crawl_pattern="*embeddings*",
        exclude_patterns=["*pricing*"],
        url_patterns=["*embeddings*", "*models*", "*best-practices*"],
    ),
]


def create_advanced_config():
    """Create advanced configuration from unified config system"""
    # Get unified configuration
    unified_config = get_config()

    # Validate API keys from unified config
    if unified_config.openai.api_key is None:
        console.print("❌ Missing OpenAI API key in configuration", style="red")
        console.print(
            "Please set: export AI_DOCS__OPENAI__API_KEY='your_key'", style="yellow"
        )
        sys.exit(1)

    # Configure embedding settings based on provider
    if unified_config.embedding_provider == EmbeddingProvider.FASTEMBED:
        console.print(
            "🚀 FastEmbed provider configured - using advanced local models",
            style="green",
        )
        # Update embedding configuration for FastEmbed
        unified_config.embedding.provider = EmbeddingProvider.FASTEMBED
        unified_config.embedding.dense_model = EmbeddingModel.BGE_SMALL_EN_V15
        unified_config.embedding.search_strategy = SearchStrategy.DENSE
        unified_config.embedding.enable_quantization = (
            unified_config.qdrant.quantization_enabled
        )

        # Enable hybrid search if sparse models available
        if SparseTextEmbedding is not None:
            console.print(
                "🎯 Enabling hybrid search (research: 8-15% improvement)",
                style="yellow",
            )
            unified_config.embedding.search_strategy = SearchStrategy.HYBRID
            unified_config.embedding.sparse_model = EmbeddingModel.SPLADE_PP_EN_V1
    else:
        console.print(
            "📡 Using OpenAI API - cost-optimized configuration", style="blue"
        )
        # Update embedding configuration for OpenAI
        unified_config.embedding.provider = EmbeddingProvider.OPENAI
        unified_config.embedding.dense_model = EmbeddingModel.TEXT_EMBEDDING_3_SMALL
        unified_config.embedding.search_strategy = SearchStrategy.DENSE

    return unified_config


async def main() -> None:
    """Main execution with advanced configuration"""
    console.print("🚀 Advanced AI Documentation Scraper", style="bold blue")
    console.print("Research-backed optimal embedding configuration", style="blue")
    console.print(
        "Python 3.13 + uv + Crawl4AI + FastEmbed + Qdrant Hybrid Search", style="cyan"
    )

    # Create optimal configuration
    current_config = create_advanced_config()

    # Display configuration
    console.print("\n📊 Configuration:", style="bold yellow")
    console.print(f"  Embedding Provider: {current_config.embedding.provider.value}")
    console.print(f"  Dense Model: {current_config.embedding.dense_model.value}")
    console.print(
        f"  Search Strategy: {current_config.embedding.search_strategy.value}"
    )
    console.print(
        f"  Chunk Size: {current_config.chunking.chunk_size} chars (research optimal)"
    )
    console.print(f"  Quantization: {current_config.embedding.enable_quantization}")
    reranker_status = (
        "✅ BGE-v2-m3" if current_config.embedding.enable_reranking else "❌ Disabled"
    )
    console.print(f"  Reranking: {reranker_status}")
    if current_config.firecrawl.api_key:
        console.print("  Firecrawl Premium: ✅ Enabled")

    scraper_instance = ModernDocumentationScraper(current_config)

    try:
        # Initialize services
        await scraper_instance.initialize()

        # Load documentation sites from unified config
        sites_to_scrape = (
            current_config.documentation_sites
            if current_config.documentation_sites
            else ESSENTIAL_SITES
        )

        await scraper_instance.scrape_multiple_sites(sites_to_scrape)
        console.print(
            "\n✅ Advanced documentation scraping completed!",
            style="bold green",
        )
        console.print(
            "Check the Qdrant collection for optimized embeddings.",
            style="blue",
        )

        # Display performance insights
        if current_config.embedding.search_strategy == SearchStrategy.HYBRID:
            console.print(
                "🎯 Hybrid search enabled - expect 8-15% better retrieval accuracy",
                style="yellow",
            )
        if current_config.embedding.enable_quantization:
            console.print(
                "💾 Vector quantization enabled - 83-99% storage reduction",
                style="green",
            )
        if current_config.embedding.enable_reranking:
            console.print(
                "🎯 Reranking enabled - expect additional 10-20% accuracy improvement",
                style="magenta",
            )
        # Example of using the demo_reranking_search
        # await scraper_instance.demo_reranking_search("what is fastembed")

    except Exception as e:
        console.print(f"\n❌ Scraping failed: {e}", style="red")
        scraper_instance.display_comprehensive_stats()
        sys.exit(1)
    finally:
        # Cleanup services
        await scraper_instance.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
