"""Simple API layer with immediate gratification patterns.

This module provides the one-line setup experience while maintaining
access to sophisticated features through progressive disclosure.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from src.config import get_settings
from src.infrastructure.client_manager import ClientManager
from src.services.cache.manager import CacheManager
from src.services.crawling.manager import CrawlingManager
from src.services.embeddings.manager import EmbeddingManager
from src.services.vector_db.search import VectorSearchService

from .protocols import EmbeddingProtocol, SearchProtocol
from .types import ProgressiveResponse, SearchOptions


logger = logging.getLogger(__name__)


@dataclass
class SimpleSearchResult:
    """Simple search result with progressive feature discovery."""

    content: str
    title: str = ""
    url: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Progressive disclosure
    _embedding: List[float] | None = field(default=None, repr=False)
    _analysis: Dict[str, Any] | None = field(default=None, repr=False)
    _suggestions: List[str] | None = field(default=None, repr=False)

    def get_embedding(self) -> List[float] | None:
        """Get the embedding vector (progressive feature)."""
        return self._embedding

    def get_analysis(self) -> Dict[str, Any]:
        """Get content analysis (progressive feature)."""
        if self._analysis is None:
            # Lazy loading of analysis
            self._analysis = {
                "length": len(self.content),
                "complexity": self._calculate_complexity(),
                "type": self._detect_content_type(),
            }
        return self._analysis

    def get_suggestions(self) -> List[str]:
        """Get related search suggestions (progressive feature)."""
        if self._suggestions is None:
            # Generate suggestions based on content
            self._suggestions = self._generate_suggestions()
        return self._suggestions

    def _calculate_complexity(self) -> float:
        """Calculate content complexity score."""
        words = self.content.split()
        unique_words = set(words)
        return len(unique_words) / len(words) if words else 0.0

    def _detect_content_type(self) -> str:
        """Detect content type."""
        content_lower = self.content.lower()
        if any(
            keyword in content_lower
            for keyword in ["def ", "class ", "import ", "function"]
        ):
            return "code"
        elif any(
            keyword in content_lower for keyword in ["## ", "### ", "documentation"]
        ):
            return "documentation"
        else:
            return "text"

    def _generate_suggestions(self) -> List[str]:
        """Generate search suggestions."""
        # Simple keyword extraction for suggestions
        words = self.content.split()
        important_words = [w for w in words if len(w) > 4 and w.isalpha()]
        return important_words[:3]


class AIDocSystem:
    """Main AI Documentation System with progressive API design.

    This class provides the primary interface for all AI documentation
    operations, designed for immediate success with one-line setup
    while revealing sophisticated features through progressive disclosure.

    Examples:
        Quick start (30 seconds to value):
            >>> system = AIDocSystem.quick_start()
            >>> results = await system.search("machine learning")

        With configuration:
            >>> system = AIDocSystem(
            ...     embedding_provider="openai", enable_cache=True, quality_tier="best"
            ... )

        Builder pattern (progressive disclosure):
            >>> system = AIDocSystem.builder().build()
    """

    def __init__(
        self,
        *,
        config_path: Union[str, Path] | None = None,
        embedding_provider: str = "fastembed",
        quality_tier: str = "balanced",
        enable_cache: bool = True,
        enable_monitoring: bool = True,
        workspace_dir: Union[str, Path] | None = None,
    ):
        """Initialize AI Documentation System.

        Args:
            config_path: Path to configuration file (optional)
            embedding_provider: Provider for embeddings ("openai", "fastembed")
            quality_tier: Quality tier ("fast", "balanced", "best")
            enable_cache: Enable caching for performance
            enable_monitoring: Enable metrics and monitoring
            workspace_dir: Working directory for data storage
        """
        self.config = get_settings()
        self.embedding_provider = embedding_provider
        self.quality_tier = quality_tier
        self.enable_cache = enable_cache
        self.enable_monitoring = enable_monitoring
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()

        # Core components (lazy initialization)
        self._client_manager: ClientManager | None = None
        self._embedding_manager: EmbeddingManager | None = None
        self._search_service: VectorSearchService | None = None
        self._crawling_manager: CrawlingManager | None = None
        self._cache_manager: CacheManager | None = None

        self._initialized = False
        self._context_manager = None

    @classmethod
    def quick_start(
        cls,
        *,
        provider: str = "fastembed",
        workspace: Union[str, Path] | None = None,
    ) -> "AIDocSystem":
        """Create a system with sensible defaults for immediate use.

        This is the 30-second success pattern - everything works out of the box.

        Args:
            provider: Embedding provider ("fastembed" for local, "openai" for cloud)
            workspace: Optional workspace directory

        Returns:
            Configured AIDocSystem ready for use

        Examples:
            >>> system = AIDocSystem.quick_start()
            >>> # System is ready to use immediately
            >>> results = await system.search("your query")
        """
        return cls(
            embedding_provider=provider,
            quality_tier="balanced",
            enable_cache=True,
            enable_monitoring=False,  # Minimal for quick start
            workspace_dir=workspace,
        )

    @classmethod
    def builder(cls) -> "AIDocSystemBuilder":
        """Create a builder for progressive configuration.

        This reveals the builder pattern for users who want more control.

        Returns:
            Builder instance for progressive configuration

        Examples:
            >>> system = (
            ...     AIDocSystem.builder()
            ...     .with_embedding_provider("openai")
            ...     .with_quality_tier("best")
            ...     .with_cache(enabled=True, ttl=3600)
            ...     .with_monitoring(enabled=True)
            ...     .build()
            ... )
        """
        from .builders import AIDocSystemBuilder

        return AIDocSystemBuilder()

    async def __aenter__(self) -> "AIDocSystem":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.cleanup()

    async def initialize(self) -> None:
        """Initialize the system components.

        This method handles all the complex initialization behind the scenes.
        """
        if self._initialized:
            return

        logger.info("Initializing AI Documentation System...")

        # Initialize client manager first
        self._client_manager = ClientManager()

        # Initialize embedding manager
        self._embedding_manager = EmbeddingManager(
            config=self.config,
            client_manager=self._client_manager,
        )

        # Initialize search service
        self._search_service = VectorSearchService(
            config=self.config,
            embedding_manager=self._embedding_manager,
        )

        # Initialize crawling manager
        self._crawling_manager = CrawlingManager(
            config=self.config,
            client_manager=self._client_manager,
        )

        # Initialize cache if enabled
        if self.enable_cache:
            self._cache_manager = CacheManager(
                config=self.config,
                client_manager=self._client_manager,
            )
            await self._cache_manager.initialize()

        # Initialize services
        await self._embedding_manager.initialize()
        await self._search_service.initialize()
        await self._crawling_manager.initialize()

        self._initialized = True
        logger.info("AI Documentation System initialized successfully")

    async def cleanup(self) -> None:
        """Cleanup system resources."""
        if not self._initialized:
            return

        logger.info("Cleaning up AI Documentation System...")

        # Cleanup in reverse order
        if self._crawling_manager:
            await self._crawling_manager.cleanup()
        if self._search_service:
            await self._search_service.cleanup()
        if self._embedding_manager:
            await self._embedding_manager.cleanup()
        if self._cache_manager:
            await self._cache_manager.cleanup()
        if self._client_manager:
            await self._client_manager.cleanup()

        self._initialized = False
        logger.info("AI Documentation System cleaned up")

    def _ensure_initialized(self) -> None:
        """Ensure system is initialized."""
        if not self._initialized:
            raise RuntimeError(
                "AIDocSystem not initialized. "
                "Use 'async with system:' or call 'await system.initialize()'"
            )

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        options: SearchOptions | None = None,
    ) -> List[SimpleSearchResult]:
        """Search for documents using hybrid vector search.

        This is the main search interface that provides immediate value.

        Args:
            query: Search query
            limit: Maximum number of results
            options: Advanced search options (progressive feature)

        Returns:
            List of search results with progressive features

        Examples:
            Basic search:
                >>> results = await system.search("machine learning")

            With options (progressive disclosure):
                >>> options = SearchOptions(rerank=True, include_embeddings=True)
                >>> results = await system.search("ML", limit=5, options=options)
        """
        self._ensure_initialized()

        # Use simple search by default, with progressive options
        search_params = {
            "query": query,
            "limit": limit,
            "include_metadata": True,
        }

        if options:
            search_params.update(options.model_dump(exclude_none=True))

        # Perform search through service layer
        raw_results = await self._search_service.search(**search_params)

        # Convert to simple results with progressive features
        results = []
        for result in raw_results:
            simple_result = SimpleSearchResult(
                content=result.get("content", ""),
                title=result.get("title", ""),
                url=result.get("url", ""),
                score=result.get("score", 0.0),
                metadata=result.get("metadata", {}),
                _embedding=result.get("embedding")
                if options and options.include_embeddings
                else None,
            )
            results.append(simple_result)

        return results

    async def add_document(
        self,
        content: str,
        *,
        title: str = "",
        url: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        """Add a document to the system.

        Args:
            content: Document content
            title: Document title
            url: Document URL
            metadata: Additional metadata

        Returns:
            Document ID
        """
        self._ensure_initialized()

        # Process document through service layer
        doc_data = {
            "content": content,
            "title": title,
            "url": url,
            "metadata": metadata or {},
        }

        return await self._search_service.add_document(doc_data)

    async def add_documents_from_urls(
        self,
        urls: List[str],
        *,
        batch_size: int = 10,
        progress_callback: callable | None = None,
    ) -> List[str]:
        """Add documents by crawling URLs.

        Args:
            urls: List of URLs to crawl
            batch_size: Number of URLs to process concurrently
            progress_callback: Optional progress callback

        Returns:
            List of document IDs
        """
        self._ensure_initialized()

        document_ids = []

        for i in range(0, len(urls), batch_size):
            batch = urls[i : i + batch_size]

            # Crawl batch of URLs
            tasks = [self._crawling_manager.crawl_url(url) for url in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process successful results
            for url, result in zip(batch, results, strict=False):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to crawl {url}: {result}")
                    continue

                # Add document to search service
                doc_id = await self.add_document(
                    content=result.get("content", ""),
                    title=result.get("title", ""),
                    url=url,
                    metadata=result.get("metadata", {}),
                )
                document_ids.append(doc_id)

            # Progress callback
            if progress_callback:
                progress = (i + len(batch)) / len(urls)
                progress_callback(
                    progress, f"Processed {i + len(batch)}/{len(urls)} URLs"
                )

        return document_ids

    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics and health information.

        Returns:
            Dictionary with system statistics
        """
        self._ensure_initialized()

        stats = {
            "initialized": self._initialized,
            "embedding_provider": self.embedding_provider,
            "quality_tier": self.quality_tier,
            "cache_enabled": self.enable_cache,
            "monitoring_enabled": self.enable_monitoring,
        }

        # Add service-specific stats
        if self._search_service:
            stats["search"] = await self._search_service.get_stats()

        if self._embedding_manager:
            stats["embeddings"] = self._embedding_manager.usage_stats

        if self._cache_manager:
            stats["cache"] = await self._cache_manager.get_stats()

        return stats

    def discover_features(self) -> Dict[str, Any]:
        """Discover available features and capabilities.

        This method helps users understand what advanced features
        are available as they become more sophisticated.

        Returns:
            Dictionary describing available features and how to access them
        """
        return {
            "basic_features": {
                "search": "await system.search('query')",
                "add_document": "await system.add_document(content='...')",
                "crawl_urls": "await system.add_documents_from_urls(['url1', 'url2'])",
            },
            "progressive_features": {
                "advanced_search": {
                    "description": "Advanced search with reranking and filters",
                    "example": "options = SearchOptions(rerank=True); await system.search('query', options=options)",
                },
                "builder_pattern": {
                    "description": "Configure system with builder pattern",
                    "example": "system = AIDocSystem.builder().with_cache(ttl=3600).build()",
                },
                "embeddings": {
                    "description": "Access raw embeddings and analysis",
                    "example": "result.get_embedding(); result.get_analysis()",
                },
                "batch_processing": {
                    "description": "Process multiple documents efficiently",
                    "example": "await system.add_documents_from_urls(urls, batch_size=20)",
                },
            },
            "expert_features": {
                "custom_protocols": {
                    "description": "Implement custom search and embedding protocols",
                    "example": "Implement SearchProtocol or EmbeddingProtocol interfaces",
                },
                "advanced_config": {
                    "description": "Fine-tune system configuration",
                    "example": "config = AdvancedConfigBuilder().with_custom_settings(...).build()",
                },
                "monitoring": {
                    "description": "Detailed metrics and performance monitoring",
                    "example": "stats = await system.get_stats()",
                },
            },
            "next_steps": [
                "Try the builder pattern: AIDocSystem.builder()",
                "Explore SearchOptions for advanced search features",
                "Check out the protocol interfaces for custom implementations",
                "Enable monitoring for production deployments",
            ],
        }
