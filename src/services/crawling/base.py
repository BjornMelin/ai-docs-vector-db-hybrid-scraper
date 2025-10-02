"""Base crawl provider interface."""

from abc import ABC, abstractmethod
from typing import Any


class CrawlProvider(ABC):
    """Abstract base class for crawl providers."""

    @abstractmethod
    async def scrape_url(
        self, url: str, formats: list[str] | None = None
    ) -> dict[str, Any]:
        """Scrape a single URL.

        Args:
            url: URL to scrape
            formats: Output formats (e.g., ['markdown', 'html'])

        Returns:
            Scrape result with content and metadata
        """

    @abstractmethod
    async def crawl_site(
        self,
        url: str,
        max_pages: int = 50,
        formats: list[str] | None = None,
    ) -> dict[str, Any]:
        """Crawl an entire site.

        Args:
            url: Starting URL
            max_pages: Maximum pages to crawl
            formats: Output formats

        Returns:
            Crawl result with pages and metadata
        """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup provider resources."""
