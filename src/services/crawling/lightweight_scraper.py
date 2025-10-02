"""Lightweight HTTP scraper using httpx and BeautifulSoup for simple static pages."""

import logging
import re
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup, Tag

from src.config import Config
from src.services.errors import CrawlServiceError

from .base import CrawlProvider


logger = logging.getLogger(__name__)


class RateLimiterProtocol(Protocol):
    """Protocol for rate limiter interface."""

    async def acquire(self, url: str) -> None:
        """Acquire rate limit token for URL."""


class TierRecommendation:
    """Recommendation for which tier to use."""

    LIGHTWEIGHT_OK = "lightweight"
    BROWSER_REQUIRED = "browser"
    STREAMING_REQUIRED = "streaming"


class LightweightScraper(CrawlProvider):
    """Ultra-fast scraper for simple static content using httpx + BeautifulSoup."""

    # pylint: disable=too-many-instance-attributes
    # Justified: Configuration values and patterns need separate attributes for clarity
    def __init__(
        self,
        config: Config,
        rate_limiter: RateLimiterProtocol | None = None,
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

        # Extract config values with sensible defaults
        browser_config = getattr(config, "browser_automation", None)
        self._timeout = (
            getattr(browser_config, "lightweight_timeout", 10.0)
            if browser_config
            else 10.0
        )
        self._head_timeout = (
            getattr(browser_config, "head_timeout", 2.0) if browser_config else 2.0
        )
        self._content_threshold = (
            getattr(browser_config, "content_threshold", 100) if browser_config else 100
        )
        self._max_lightweight_size = (
            getattr(browser_config, "max_lightweight_size", 1_000_000)
            if browser_config
            else 1_000_000
        )
        self._enable_lightweight_tier = (
            getattr(browser_config, "enable_lightweight_tier", True)
            if browser_config
            else True
        )
        self._use_head_analysis = (
            getattr(browser_config, "use_head_analysis", False)
            if browser_config
            else False
        )

        # Simple URL patterns for static content
        self._simple_patterns = [
            re.compile(r".*\.md$"),
            re.compile(r".*/raw/.*"),
            re.compile(r".*\.(txt|json|xml|csv)$"),
            re.compile(r".*\.github\.io/.*"),
            re.compile(r".*readthedocs\.io/.*"),
        ]

        # Known simple sites with content selectors
        self._known_simple_sites: dict[str, dict[str, str]] = {
            "docs.python.org": {"selector": ".document"},
            "golang.org": {"selector": "#page"},
            "developer.mozilla.org": {"selector": "main"},
        }

    async def initialize(self) -> None:
        """Initialize the HTTP client."""
        if self._initialized:
            return

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout),
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
        if not self._enable_lightweight_tier:
            return False

        # Check URL patterns first (fastest)
        if any(pattern.search(url) for pattern in self._simple_patterns):
            logger.debug("URL %s matches simple pattern", url)
            return True

        # Check known simple sites
        domain = urlparse(url).netloc
        if domain in self._known_simple_sites:
            logger.debug("URL %s is from known simple site", url)
            return True

        # Perform HEAD request analysis if enabled
        if self._use_head_analysis:
            try:
                recommendation = await self._analyze_url(url)
            except (ConnectionError, OSError, PermissionError):
                logger.debug("HEAD analysis failed for %s", url)
                return False

            return recommendation == TierRecommendation.LIGHTWEIGHT_OK
        return False

    async def _analyze_url(self, url: str) -> str:
        """Analyze URL with HEAD request to determine tier recommendation.

        Args:
            url: URL to analyze

        Returns:
            TierRecommendation constant
        """

        if not self._http_client:
            msg = "Scraper not initialized"
            raise CrawlServiceError(msg)

        try:
            response = await self._http_client.head(url, timeout=self._head_timeout)
        except (httpx.TimeoutException, ConnectionError, OSError, PermissionError):
            logger.debug("HEAD request failed for %s", url)
            return TierRecommendation.BROWSER_REQUIRED

        # Check content type - non-HTML is usually simple
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type:
            return TierRecommendation.LIGHTWEIGHT_OK

        # Check for SPA indicators in headers
        spa_headers = ["x-powered-by", "x-framework", "x-react", "x-vue", "x-angular"]
        if any(header in response.headers for header in spa_headers):
            logger.debug("SPA indicators found for %s", url)
            return TierRecommendation.BROWSER_REQUIRED

        # Check response size - very large content needs streaming
        content_length = int(response.headers.get("content-length", 0))
        if content_length > self._max_lightweight_size:
            logger.debug("Content too large for lightweight tier")
            return TierRecommendation.STREAMING_REQUIRED

        # Check for heavy JavaScript usage via CSP
        csp = response.headers.get("content-security-policy", "")
        if "script-src" in csp and "'unsafe-inline'" in csp:
            logger.debug("Heavy JavaScript usage indicated by CSP")
            return TierRecommendation.BROWSER_REQUIRED

        return TierRecommendation.LIGHTWEIGHT_OK

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
        if not self._initialized or not self._http_client:
            msg = "Scraper not initialized"
            raise CrawlServiceError(msg)

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

        except httpx.HTTPStatusError as e:
            logger.warning("HTTP error scraping %s", url)
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}",
                "should_escalate": e.response.status_code not in [404, 403, 401],
            }
        except Exception as e:
            logger.exception("Error scraping %s with lightweight tier", url)
            return {
                "success": False,
                "error": str(e),
                "should_escalate": True,
            }

        # Check if content is sufficient
        if not extracted or len(extracted.get("text", "")) < self._content_threshold:
            logger.info(
                "Insufficient content extracted from %s, escalating to next tier",
                url,
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

    def _find_content_element(self, soup: BeautifulSoup, domain: str) -> Tag | None:
        """Find the main content element using various strategies.

        Args:
            soup: BeautifulSoup parsed HTML
            domain: Domain name for site-specific selection

        Returns:
            Main content element or None
        """
        # Try known site-specific selector first
        if domain in self._known_simple_sites:
            selector = self._known_simple_sites[domain].get("selector")
            if selector and (element := soup.select_one(selector)):
                return element

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
            if element := soup.select_one(selector):
                return element

        # Fallback to intelligent detection
        return self._find_main_content(soup)

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

        # Find main content element
        domain = urlparse(url).netloc
        content_element = self._find_content_element(soup, domain)

        if not content_element:
            return None

        # Extract metadata
        title = ""
        if soup.title:
            title = soup.title.string or ""
        elif h1 := soup.find("h1"):
            title = h1.get_text(strip=True)

        description = ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and isinstance(meta_desc, Tag):
            description = meta_desc.get("content", "")

        language = "en"
        html_tag = soup.find("html")
        if html_tag and isinstance(html_tag, Tag):
            lang_attr = html_tag.get("lang", "en")
            if isinstance(lang_attr, str):
                language = lang_attr[:2]

        return {
            "content": content_element,
            "text": content_element.get_text(separator=" ", strip=True),
            "title": title,
            "description": description,
            "language": language,
        }

    def _find_main_content(self, soup: BeautifulSoup) -> Tag | None:
        """Find main content using readability-like algorithm.

        Args:
            soup: BeautifulSoup parsed HTML

        Returns:
            Main content element or None
        """

        # Find all text-containing elements
        candidates: list[tuple[float, Tag]] = []

        for element in soup.find_all(["div", "section", "article"]):
            if not isinstance(element, Tag):
                continue

            # Skip elements with too many links (likely navigation)
            links = element.find_all("a")
            text_length = len(element.get_text(strip=True))

            if text_length > 100 and len(links) < text_length / 50:
                # Simple scoring based on text length and structure
                score = float(text_length)
                # Bonus for semantic elements
                if element.name in ["article", "main"]:
                    score *= 2
                # Bonus for class/id indicators
                class_attr = element.get("class", "")
                id_attr = element.get("id", "")
                combined_attrs = f"{class_attr} {id_attr}"
                if any(
                    indicator in combined_attrs
                    for indicator in ["content", "article", "post", "main", "text"]
                ):
                    score *= 1.5

                candidates.append((score, element))

        if candidates:
            # Return highest scoring element
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        return None

    def _convert_heading(self, element: Tag, level: int) -> str:
        """Convert heading element to markdown.

        Args:
            element: Heading element
            level: Heading level (1-4)

        Returns:
            Markdown heading string
        """

        prefix = "#" * level
        text = element.get_text(strip=True)
        return f"\n{prefix} {text}\n"

    def _convert_list_item(self, element: Tag) -> str:
        """Convert list item to markdown.

        Args:
            element: List item element

        Returns:
            Markdown list item string
        """

        text = element.get_text(strip=True)
        if not text or not element.parent:
            return ""

        if element.parent.name == "ol":
            # Count only li elements for numbering
            li_elements = [
                e
                for e in element.parent.children
                if isinstance(e, Tag) and e.name == "li"
            ]
            index = li_elements.index(element) + 1
            return f"{index}. {text}\n"
        return f"- {text}\n"

    def _convert_inline_element(self, element: Tag) -> str:
        """Convert inline element to markdown.

        Args:
            element: Inline element (code, a, strong, em, etc.)

        Returns:
            Markdown inline string
        """

        text = element.get_text(strip=True)
        if not text:
            return ""

        if element.name == "code" and element.parent and element.parent.name != "pre":
            return f"`{text}`"
        if element.name == "a":
            href = element.get("href", "")
            if href:
                return f"[{text}]({href})"
        if element.name in {"strong", "b"}:
            return f"**{text}**"
        if element.name in {"em", "i"}:
            return f"*{text}*"
        return ""

    def _convert_to_markdown(self, content_element, title="") -> str:
        """Convert HTML content to markdown.

        Args:
            content_element: BeautifulSoup element
            title: Optional title to prepend

        Returns:
            Markdown formatted content
        """
        # pylint: disable=too-many-branches
        # Justified: Comprehensive HTML element handling requires multiple branches
        markdown_lines: list[str] = []

        if title:
            markdown_lines.append(f"# {title}\n")

        # Simple HTML to Markdown conversion
        for element in content_element.descendants:
            if not hasattr(element, "name"):
                continue

            # Headings
            if element.name == "h1":
                markdown_lines.append(self._convert_heading(element, 1))
            elif element.name == "h2":
                markdown_lines.append(self._convert_heading(element, 2))
            elif element.name == "h3":
                markdown_lines.append(self._convert_heading(element, 3))
            elif element.name == "h4":
                markdown_lines.append(self._convert_heading(element, 4))
            # Block elements
            elif element.name == "p":
                text = element.get_text(strip=True)
                if text:
                    markdown_lines.append(f"\n{text}\n")
            elif element.name == "ul":
                markdown_lines.append("\n")
            elif element.name == "li":
                markdown_lines.append(self._convert_list_item(element))
            elif element.name == "pre":
                code = element.get_text()
                markdown_lines.append(f"\n```\n{code}\n```\n")
            elif element.name == "blockquote":
                text = element.get_text(strip=True)
                if text:
                    markdown_lines.append(f"\n> {text}\n")
            # Inline elements
            else:
                inline_md = self._convert_inline_element(element)
                if inline_md:
                    markdown_lines.append(inline_md)

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

        # Parameters intentionally unused - method not supported
        _ = (url, max_pages, formats)

        return {
            "success": False,
            "error": "Lightweight tier does not support site crawling",
            "pages": [],
            "total": 0,
            "should_escalate": True,
        }
