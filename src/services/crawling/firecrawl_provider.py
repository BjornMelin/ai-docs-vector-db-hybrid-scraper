"""Firecrawl provider using direct SDK."""

import logging
from collections.abc import Iterable
from typing import Any, cast

from firecrawl import AsyncFirecrawl
from firecrawl.client import AsyncV2Proxy
from firecrawl.v2.types import (
    CrawlJob,
    CrawlResponse,
    FormatOption,
    MapData,
    ScrapeOptions,
    ScrapeResponse,
)

from src.config import FirecrawlConfig
from src.services.base import BaseService
from src.services.errors import CrawlServiceError
from src.services.utilities.rate_limiter import RateLimitManager

from .base import CrawlProvider


logger = logging.getLogger(__name__)


def _raise_no_crawl_id_returned() -> None:
    """Raise CrawlServiceError for missing crawl ID."""

    msg = "No crawl ID returned"
    raise CrawlServiceError(msg)


class FirecrawlProvider(BaseService, CrawlProvider):
    """Firecrawl provider for web crawling."""

    def __init__(
        self, config: FirecrawlConfig, rate_limiter: RateLimitManager | None = None
    ):
        """Initialize Firecrawl provider.

        Args:
            config: Firecrawl configuration model
            rate_limiter: Optional rate limiter
        """

        super().__init__()
        self.config = config
        self._client: object | None = None
        self._firecrawl: AsyncFirecrawl | None = None
        self._initialized = False
        self.rate_limiter = rate_limiter

    async def initialize(self) -> None:
        """Initialize Firecrawl client."""
        if self._initialized:
            return

        try:
            api_key = self.config.api_key
            if not api_key:
                msg = "Firecrawl API key is required"
                raise CrawlServiceError(msg)

            self._firecrawl = AsyncFirecrawl(
                api_key=api_key, api_url=self.config.api_url
            )
            self._client = self._firecrawl
            self._initialized = True
            logger.info("Firecrawl client initialized")
        except Exception as e:
            msg = "Failed to initialize Firecrawl"
            raise CrawlServiceError(msg) from e

    async def cleanup(self) -> None:
        """Cleanup Firecrawl resources."""

        self._client = None
        self._firecrawl = None
        self._initialized = False
        logger.info("Firecrawl resources cleaned up")

    async def scrape_url(
        self, url: str, formats: list[str] | None = None
    ) -> dict[str, Any]:
        """Scrape single URL.

        Args:
            url: URL to scrape
            formats: Output formats (default: ['markdown'])

        Returns:
            Scrape result
        """

        if not self._initialized:
            msg = "Provider not initialized"
            raise CrawlServiceError(msg)

        format_options = self._normalize_formats(formats)

        try:
            response = await self._scrape_url_with_rate_limit(url, format_options)

            if response.success and response.data:
                document = response.data
                metadata = document.metadata_dict
                return {
                    "success": True,
                    "content": document.markdown or "",
                    "html": document.html or "",
                    "metadata": metadata,
                    "url": url,
                }
            return {
                "success": False,
                "error": response.error or "Scraping failed",
                "content": "",
                "metadata": {},
                "url": url,
            }

        except Exception as e:
            logger.exception("Failed to scrape %s", url)

            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                logger.warning("Firecrawl rate limit hit for %s", url)
                error_detail = "Rate limit exceeded. Please try again later."
            elif "invalid api key" in error_msg or "unauthorized" in error_msg:
                logger.exception("Invalid Firecrawl API key")
                error_detail = (
                    "Invalid API key. Please check your Firecrawl configuration."
                )
            elif "timeout" in error_msg:
                logger.warning("Timeout while scraping %s", url)
                error_detail = (
                    "Request timed out. The page may be too large or slow to load."
                )
            elif "not found" in error_msg or "404" in error_msg:
                logger.info("Page not found for %s", url)
                error_detail = "Page not found (404)."
            else:
                error_detail = "Scraping failed"

            return {
                "success": False,
                "error": error_detail,
                "content": "",
                "metadata": {},
                "url": url,
            }

    async def crawl_site(
        self,
        url: str,
        max_pages: int = 50,
        formats: list[str] | None = None,
    ) -> dict[str, Any]:
        """Crawl entire site.

        Args:
            url: Starting URL
            max_pages: Maximum pages to crawl
            formats: Output formats

        Returns:
            Crawl result

        """
        if not self._initialized:
            msg = "Provider not initialized"
            raise CrawlServiceError(msg)

        format_options = self._normalize_formats(formats)

        try:
            crawl_response = await self._start_crawl_with_rate_limit(
                url, max_pages, format_options
            )

            crawl_id = crawl_response.id
            if not crawl_id:
                _raise_no_crawl_id_returned()

            logger.info("Started crawl job %s for %s", crawl_id, url)

            v2_client = self._get_v2_client()
            crawl_job = await v2_client.wait_crawl(
                crawl_id, poll_interval=5, timeout=600
            )

            if crawl_job.status == "completed":
                return self._build_crawl_success(crawl_job, crawl_id)
            if crawl_job.status == "failed":
                return {
                    "success": False,
                    "error": "Crawl failed",
                    "pages": [],
                    "total": 0,
                    "crawl_id": crawl_id,
                }

            return {
                "success": False,
                "error": "Crawl ended with unknown status",
                "pages": [],
                "total": 0,
                "crawl_id": crawl_id,
            }

        except Exception as e:
            logger.exception("Failed to crawl %s", url)
            return {
                "success": False,
                "error": str(e),
                "pages": [],
                "total": 0,
            }

    async def cancel_crawl(self, crawl_id: str) -> bool:
        """Cancel a crawl job.

        Args:
            crawl_id: Crawl job ID

        Returns:
            Success status
        """

        if not self._initialized:
            msg = "Provider not initialized"
            raise CrawlServiceError(msg)

        try:
            v2_client = self._get_v2_client()
            return await v2_client.cancel_crawl(crawl_id)
        except (ConnectionError, OSError, PermissionError):
            logger.exception("Failed to cancel crawl %s", crawl_id)
            return False

    async def map_url(
        self, url: str, include_subdomains: bool = False
    ) -> dict[str, Any]:
        """Map a website to get list of URLs.

        Args:
            url: URL to map
            include_subdomains: Include subdomains

        Returns:
            Map result with URLs
        """

        if not self._initialized:
            msg = "Provider not initialized"
            raise CrawlServiceError(msg)

        try:
            v2_client = self._get_v2_client()
            map_result = await v2_client.map(url, include_subdomains=include_subdomains)

            return self._build_map_response(map_result)

        except Exception as e:
            logger.exception("Failed to map %s", url)
            return {
                "success": False,
                "error": str(e),
                "urls": [],
                "total": 0,
            }

    async def _scrape_url_with_rate_limit(
        self, url: str, formats: list[FormatOption]
    ) -> ScrapeResponse:
        """Scrape URL with rate limiting."""

        if self.rate_limiter:
            await self.rate_limiter.acquire("firecrawl")

        v2_client = self._get_v2_client()
        result = await v2_client.scrape(url=url, formats=formats)
        return cast(ScrapeResponse, result)

    async def _start_crawl_with_rate_limit(
        self, url: str, max_pages: int, formats: list[FormatOption]
    ) -> CrawlResponse:
        """Start crawl with rate limiting."""

        if self.rate_limiter:
            await self.rate_limiter.acquire("firecrawl")

        v2_client = self._get_v2_client()
        scrape_options = ScrapeOptions(formats=formats)
        return await v2_client.start_crawl(
            url=url,
            limit=max_pages,
            scrape_options=scrape_options,
        )

    def _get_client(self) -> AsyncFirecrawl:
        """Return initialized client or raise."""

        if not self._initialized or not isinstance(self._firecrawl, AsyncFirecrawl):
            msg = "Provider not initialized"
            raise CrawlServiceError(msg)
        return self._firecrawl

    def _get_v2_client(self) -> AsyncV2Proxy:
        """Return Firecrawl v2 proxy client."""

        client = self._get_client()
        v2_client = getattr(client, "v2", None)
        if not isinstance(v2_client, AsyncV2Proxy):
            msg = "Firecrawl v2 client unavailable"
            raise CrawlServiceError(msg)
        return v2_client

    @staticmethod
    def _normalize_formats(formats: Iterable[str] | None) -> list[FormatOption]:
        """Normalize format list for Firecrawl SDK."""

        normalized = list(formats) if formats else ["markdown"]
        return [cast(FormatOption, fmt) for fmt in normalized]

    @staticmethod
    def _build_crawl_success(crawl_job: CrawlJob, crawl_id: str) -> dict[str, Any]:
        """Convert crawl job into response payload."""

        pages: list[dict[str, Any]] = []
        for document in crawl_job.data:
            metadata = document.metadata_dict
            page_url = metadata.get("url") or metadata.get("sourceUrl") or ""
            pages.append(
                {
                    "url": page_url,
                    "content": document.markdown or "",
                    "html": document.html or "",
                    "metadata": metadata,
                }
            )

        return {
            "success": True,
            "pages": pages,
            "total": len(pages),
            "crawl_id": crawl_id,
        }

    @staticmethod
    def _build_map_response(map_result: MapData) -> dict[str, Any]:
        """Convert map data into response payload."""

        links = [link.model_dump(exclude_none=True) for link in map_result.links]
        return {
            "success": True,
            "urls": links,
            "total": len(links),
        }
