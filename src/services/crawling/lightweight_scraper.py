import typing
"""Lightweight HTTP scraper using httpx and BeautifulSoup for simple static pages."""

import logging
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from src.config import Config  # LightweightScraperConfig not in simplified config

from ..errors import CrawlServiceError
from .base import CrawlProvider

logger = logging.getLogger(__name__)


class TierRecommendation:
    """Recommendation for which tier to use."""

    LIGHTWEIGHT_OK = "lightweight"
    BROWSER_REQUIRED = "browser"
    STREAMING_REQUIRED = "streaming"


class LightweightScraper(CrawlProvider):
    """Ultra-fast scraper for simple static content using httpx + BeautifulSoup."""

    def __init__(
        self,
        config: Config,
        rate_limiter: object | None = None,
    ):
        """Initialize lightweight scraper.

        Args:
            config: Configuration for the lightweight scraper
            rate_limiter: Optional rate limiter instance
        """
        self.config = config
        self.rate_limiter = rate_limiter
        self._http_client: httpx.AsyncClient | None = None
        self._initialized = False

        # Compile URL patterns for efficiency
        self._simple_patterns = [
            re.compile(pattern) for pattern in config.simple_url_patterns
        ]

    async def initialize(self) -> None:
        """Initialize the HTTP client."""
        if self._initialized:
            return

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout),
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; LightweightScraper/1.0)"},
        )
        self._initialized = True
        logger.info("Lightweight scraper initialized")

    async def cleanup(self) -> None:
        """Cleanup HTTP client resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._initialized = False
        logger.info("Lightweight scraper cleaned up")

    async def can_handle(self, url: str) -> bool:
        """Determine if URL can be handled by lightweight tier.

        Performs quick checks to determine if the URL is suitable for
        lightweight scraping without making a full request.

        Args:
            url: URL to analyze

        Returns:
            True if URL can be handled by lightweight tier
        """
        if not self.config.enable_lightweight_tier:
            return False

        # Check URL patterns first (fastest)
        if any(pattern.search(url) for pattern in self._simple_patterns):
            logger.debug(f"URL {url} matches simple pattern")
            return True

        # Check known simple sites
        domain = urlparse(url).netloc
        if domain in self.config.known_simple_sites:
            logger.debug(f"URL {url} is from known simple site")
            return True

        # Perform HEAD request analysis if enabled
        if self.config.use_head_analysis:
            try:
                recommendation = await self._analyze_url(url)
                return recommendation == TierRecommendation.LIGHTWEIGHT_OK
            except Exception as e:
                logger.debug(f"HEAD analysis failed for {url}: {e}")
                return False

        return False

    async def _analyze_url(self, url: str) -> str:
        """Analyze URL with HEAD request to determine tier recommendation.

        Args:
            url: URL to analyze

        Returns:
            TierRecommendation constant
        """
        if not self._http_client:
            raise CrawlServiceError("Scraper not initialized")

        try:
            response = await self._http_client.head(
                url, timeout=self.config.head_timeout
            )

            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                # Non-HTML content (JSON, XML, etc.) is usually simple
                return TierRecommendation.LIGHTWEIGHT_OK

            # Check for SPA indicators
            spa_headers = [
                "x-powered-by",
                "x-framework",
                "x-react",
                "x-vue",
                "x-angular",
            ]
            if any(header in response.headers for header in spa_headers):
                logger.debug(f"SPA indicators found for {url}")
                return TierRecommendation.BROWSER_REQUIRED

            # Check response size
            content_length = int(response.headers.get("content-length", 0))
            if content_length > self.config.max_lightweight_size:
                logger.debug(
                    f"Content too large for lightweight tier: {content_length}"
                )
                return TierRecommendation.STREAMING_REQUIRED

            # Check for JavaScript requirements in headers
            csp = response.headers.get("content-security-policy", "")
            if "script-src" in csp and "'unsafe-inline'" in csp:
                logger.debug("Heavy JavaScript usage indicated by CSP")
                return TierRecommendation.BROWSER_REQUIRED

            return TierRecommendation.LIGHTWEIGHT_OK

        except httpx.TimeoutException:
            logger.debug(f"HEAD request timeout for {url}")
            return TierRecommendation.BROWSER_REQUIRED
        except Exception as e:
            logger.debug(f"HEAD request failed for {url}: {e}")
            return TierRecommendation.BROWSER_REQUIRED

    async def scrape_url(
        self, url: str, formats: list[str] | None = None
    ) -> dict[str, Any]:
        """Scrape a single URL using lightweight HTTP approach.

        Args:
            url: URL to scrape
            formats: Output formats (default: ["markdown"])

        Returns:
            Scrape result with content and metadata, or None to escalate
        """
        if not self._initialized:
            raise CrawlServiceError("Scraper not initialized")

        formats = formats or ["markdown"]

        # Apply rate limiting if available
        if self.rate_limiter:
            await self.rate_limiter.acquire(url)

        try:
            # Perform GET request
            response = await self._http_client.get(url)
            response.raise_for_status()

            # Parse HTML
            html_content = response.text
            soup = BeautifulSoup(html_content, "lxml")

            # Extract content
            extracted = await self._extract_content(soup, url)

            # Check if content is sufficient
            if (
                not extracted
                or len(extracted.get("text", "")) < self.config.content_threshold
            ):
                logger.info(
                    f"Insufficient content extracted from {url}, escalating to next tier"
                )
                return {
                    "success": False,
                    "error": "Insufficient content for lightweight tier",
                    "should_escalate": True,
                }

            # Format content based on requested formats
            result = {
                "success": True,
                "url": url,
                "content": {},
                "metadata": {
                    "title": extracted.get("title", ""),
                    "description": extracted.get("description", ""),
                    "language": extracted.get("language", "en"),
                    "word_count": len(extracted.get("text", "").split()),
                    "tier": "lightweight",
                },
            }

            if "markdown" in formats:
                result["content"]["markdown"] = self._convert_to_markdown(
                    extracted["content"], extracted.get("title", "")
                )
            if "html" in formats:
                result["content"]["html"] = str(extracted["content"])
            if "text" in formats:
                result["content"]["text"] = extracted["text"]

            return result

        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error scraping {url}: {e}")
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}",
                "should_escalate": e.response.status_code not in [404, 403, 401],
            }
        except Exception as e:
            logger.exception(f"Error scraping {url} with lightweight tier: {e}")
            return {
                "success": False,
                "error": str(e),
                "should_escalate": True,
            }

    async def _extract_content(
        self, soup: BeautifulSoup, url: str
    ) -> dict[str, Any] | None:
        """Extract content from parsed HTML.

        Args:
            soup: BeautifulSoup parsed HTML
            url: Original URL for domain-specific extraction

        Returns:
            Extracted content dictionary or None
        """
        # Remove script and style elements
        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        # Try known selectors first
        domain = urlparse(url).netloc
        content_element = None

        if domain in self.config.known_simple_sites:
            selector = self.config.known_simple_sites[domain].get("selector")
            if selector:
                content_element = soup.select_one(selector)

        # Fallback to intelligent content detection
        if not content_element:
            # Try common content selectors
            content_selectors = [
                "main",
                "article",
                "[role='main']",
                ".content",
                "#content",
                ".documentation",
                ".doc-content",
                ".markdown-body",
                ".post-content",
            ]

            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    break

            # If still no content, try to find the largest text block
            if not content_element:
                content_element = self._find_main_content(soup)

        if not content_element:
            return None

        # Extract metadata
        title = ""
        if soup.title:
            title = soup.title.string or ""
        elif h1 := soup.find("h1"):
            title = h1.get_text(strip=True)

        description = ""
        if meta_desc := soup.find("meta", attrs={"name": "description"}):
            description = meta_desc.get("content", "")

        language = "en"
        if html_tag := soup.find("html"):
            language = html_tag.get("lang", "en")[:2]

        return {
            "content": content_element,
            "text": content_element.get_text(separator=" ", strip=True),
            "title": title,
            "description": description,
            "language": language,
        }

    def _find_main_content(self, soup: BeautifulSoup) -> Any | None:
        """Find main content using readability-like algorithm.

        Args:
            soup: BeautifulSoup parsed HTML

        Returns:
            Main content element or None
        """
        # Find all text-containing elements
        candidates = []

        for element in soup.find_all(["div", "section", "article"]):
            # Skip elements with too many links (likely navigation)
            links = element.find_all("a")
            text_length = len(element.get_text(strip=True))

            if text_length > 100 and len(links) < text_length / 50:
                # Simple scoring based on text length and structure
                score = text_length
                # Bonus for semantic elements
                if element.name in ["article", "main"]:
                    score *= 2
                # Bonus for class/id indicators
                if any(
                    indicator
                    in str(element.get("class", [])) + str(element.get("id", ""))
                    for indicator in ["content", "article", "post", "main", "text"]
                ):
                    score *= 1.5

                candidates.append((score, element))

        if candidates:
            # Return highest scoring element
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        return None

    def _convert_to_markdown(self, content_element: Any, title: str = "") -> str:
        """Convert HTML content to markdown.

        Args:
            content_element: BeautifulSoup element
            title: Optional title to prepend

        Returns:
            Markdown formatted content
        """
        markdown_lines = []

        if title:
            markdown_lines.append(f"# {title}\n")

        # Simple HTML to Markdown conversion
        for element in content_element.descendants:
            if element.name == "h1":
                markdown_lines.append(f"\n# {element.get_text(strip=True)}\n")
            elif element.name == "h2":
                markdown_lines.append(f"\n## {element.get_text(strip=True)}\n")
            elif element.name == "h3":
                markdown_lines.append(f"\n### {element.get_text(strip=True)}\n")
            elif element.name == "h4":
                markdown_lines.append(f"\n#### {element.get_text(strip=True)}\n")
            elif element.name == "p":
                text = element.get_text(strip=True)
                if text:
                    markdown_lines.append(f"\n{text}\n")
            elif element.name == "ul":
                markdown_lines.append("\n")
            elif element.name == "li":
                text = element.get_text(strip=True)
                if text:
                    # Check if it's in an ordered list
                    if element.parent.name == "ol":
                        # Count only li elements, not all children
                        li_elements = [
                            e for e in element.parent.children if e.name == "li"
                        ]
                        index = li_elements.index(element) + 1
                        markdown_lines.append(f"{index}. {text}\n")
                    else:
                        markdown_lines.append(f"- {text}\n")
            elif element.name == "code":
                text = element.get_text(strip=True)
                if text and element.parent.name != "pre":
                    markdown_lines.append(f"`{text}`")
            elif element.name == "pre":
                code = element.get_text()
                markdown_lines.append(f"\n```\n{code}\n```\n")
            elif element.name == "a":
                text = element.get_text(strip=True)
                href = element.get("href", "")
                if text and href:
                    markdown_lines.append(f"[{text}]({href})")
            elif element.name in {"strong", "b"}:
                text = element.get_text(strip=True)
                if text:
                    markdown_lines.append(f"**{text}**")
            elif element.name in {"em", "i"}:
                text = element.get_text(strip=True)
                if text:
                    markdown_lines.append(f"*{text}*")
            elif element.name == "blockquote":
                text = element.get_text(strip=True)
                if text:
                    markdown_lines.append(f"\n> {text}\n")

        # Clean up and join
        markdown = "".join(markdown_lines)
        # Remove excessive newlines
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        return markdown.strip()

    async def crawl_site(
        self,
        url: str,
        max_pages: int = 50,
        formats: list[str] | None = None,
    ) -> dict[str, Any]:
        """Crawl an entire site (not implemented for lightweight tier).

        The lightweight tier is designed for single-page scraping only.
        Site crawling should use the standard Crawl4AI tier.

        Args:
            url: Starting URL
            max_pages: Maximum pages to crawl
            formats: Output formats

        Returns:
            Error result indicating this tier doesn't support crawling
        """
        return {
            "success": False,
            "error": "Lightweight tier does not support site crawling",
            "pages": [],
            "total": 0,
            "should_escalate": True,
        }
