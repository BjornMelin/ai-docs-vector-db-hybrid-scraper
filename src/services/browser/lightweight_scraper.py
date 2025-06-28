"""Lightweight HTTP scraper for static content using httpx + BeautifulSoup.

This module implements Tier 0 of the 5-tier browser automation system,
providing 5-10x performance improvement for static content by avoiding
browser overhead for simple HTML pages.
"""

import logging
import re
import time
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from src.config import Config
from src.services.base import BaseService


logger = logging.getLogger(__name__)


class ContentAnalysis(BaseModel):
    """Analysis of content to determine if lightweight scraping is viable."""

    can_handle: bool = Field(
        description="Whether content can be handled by lightweight scraper"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0-1)")
    reasons: list[str] = Field(default_factory=list, description="Reasons for decision")
    content_type: str | None = Field(default=None, description="Detected content type")
    size_estimate: int | None = Field(default=None, description="Content size in bytes")


class ScrapedContent(BaseModel):
    """Structured content extracted by lightweight scraper."""

    url: str = Field(description="Source URL")
    title: str = Field(default="", description="Page title")
    text: str = Field(description="Extracted text content")
    headings: list[dict[str, Any]] = Field(
        default_factory=list, description="Heading structure"
    )
    links: list[dict[str, str]] = Field(
        default_factory=list, description="Extracted links"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    extraction_time_ms: float = Field(
        description="Time taken for extraction in milliseconds"
    )
    tier: int = Field(default=0, description="Scraping tier used")
    success: bool = Field(default=True, description="Whether extraction was successful")


class LightweightScraper(BaseService):
    """High-performance HTTP scraper for static content.

    Uses httpx for async HTTP requests and BeautifulSoup with lxml parser
    for fast HTML processing. Designed for 5-10x performance improvement
    over browser-based scraping for static documentation and simple pages.
    """

    def __init__(self, config: Config):
        """Initialize lightweight scraper.

        Args:
            config: Unified configuration instance

        """
        super().__init__(config)
        self._client: httpx.AsyncClient | None = None
        self._content_threshold = getattr(
            config.browser_automation, "content_threshold", 100
        )
        self._timeout = getattr(config.browser_automation, "lightweight_timeout", 10.0)
        self._max_retries = getattr(config.browser_automation, "max_retries", 2)

        # URL patterns for static content detection
        self._static_patterns = [
            r".*\.md$",
            r".*\.txt$",
            r".*\.json$",
            r".*\.xml$",
            r".*\.csv$",
            r".*/raw/.*",  # GitHub raw files
            r".*\.github\.io/.*",  # GitHub Pages
            r".*readthedocs\.io/.*",  # ReadTheDocs
            r".*gitbook\.io/.*",  # GitBook
        ]

        # Known simple site selectors
        self._site_selectors = {
            "docs.python.org": {"content": ".document", "title": "h1"},
            "golang.org": {"content": "#page", "title": "h1"},
            "developer.mozilla.org": {"content": "main", "title": "h1"},
            "stackoverflow.com": {"content": ".post-text", "title": "h1"},
            "github.com": {"content": ".markdown-body", "title": "h1"},
        }

    async def initialize(self) -> None:
        """Initialize the lightweight scraper."""
        if self._client is None:
            # Configure httpx client for optimal performance
            limits = httpx.Limits(
                max_keepalive_connections=20, max_connections=100, keepalive_expiry=30.0
            )

            timeout = httpx.Timeout(
                connect=5.0, read=self._timeout, write=5.0, pool=2.0
            )

            self._client = httpx.AsyncClient(
                limits=limits,
                timeout=timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; LightweightScraper/1.0; +https://github.com/ai-docs-vector-db)",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
                    "Accept-Encoding": "gzip, deflate",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
                http2=True,
                follow_redirects=True,
            )

        logger.info("LightweightScraper initialized")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def can_handle(self, url: str) -> ContentAnalysis:
        """Determine if URL can be handled by lightweight scraper.

        Uses HEAD request and pattern analysis to quickly assess
        whether the content is suitable for lightweight scraping.

        Args:
            url: URL to analyze

        Returns:
            ContentAnalysis with decision and confidence score

        """
        start_time = time.time()
        analysis = ContentAnalysis(can_handle=False, confidence=0.0)

        try:
            # Pattern-based pre-filtering
            url_confidence = self._analyze_url_patterns(url)
            if url_confidence > 0.7:
                analysis.can_handle = True
                analysis.confidence = url_confidence
                analysis.reasons.append(
                    f"URL pattern indicates static content (confidence: {url_confidence:.2f})"
                )
                return analysis

            # HEAD request analysis for more detailed assessment
            head_analysis = await self._analyze_head_request(url)
            if head_analysis:
                analysis.can_handle = head_analysis["can_handle"]
                analysis.confidence = head_analysis["confidence"]
                analysis.reasons.extend(head_analysis["reasons"])
                analysis.content_type = head_analysis.get("content_type")
                analysis.size_estimate = head_analysis.get("size_estimate")

        except Exception as e:
            logger.warning(f"Error analyzing URL {url}: {e}")
            analysis.reasons.append(f"Analysis error: {e!s}")

        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"Content analysis for {url} completed in {elapsed_ms:.1f}ms: {analysis.can_handle}"
        )

        return analysis

    def _analyze_url_patterns(self, url: str) -> float:
        """Analyze URL patterns to estimate static content probability.

        Args:
            url: URL to analyze

        Returns:
            Confidence score (0.0 to 1.0)

        """
        parsed = urlparse(url)
        path = parsed.path.lower()
        domain = parsed.netloc.lower()

        # Check static file extensions
        for pattern in self._static_patterns:
            if re.match(pattern, url, re.IGNORECASE):
                return 0.9

        # Check known documentation domains
        doc_domains = [
            "docs.python.org",
            "golang.org/doc",
            "developer.mozilla.org",
            "readthedocs.io",
            "gitbook.io",
            "github.io",
            "sphinx-doc.org",
        ]

        for doc_domain in doc_domains:
            if doc_domain in domain:
                return 0.8

        # Check path patterns that indicate documentation
        if any(
            path.startswith(prefix)
            for prefix in ["/docs/", "/doc/", "/documentation/", "/api/"]
        ):
            return 0.7

        # Check for common static indicators
        if any(
            indicator in path
            for indicator in [".html", ".htm", "/readme", "/changelog"]
        ):
            return 0.6

        return 0.3  # Default low confidence

    async def _analyze_head_request(self, url: str) -> dict[str, Any] | None:
        """Perform HEAD request analysis for content assessment.

        Args:
            url: URL to analyze

        Returns:
            Dictionary with analysis results or None if failed

        """
        if not self._client:
            await self.initialize()

        try:
            response = await self._client.head(url, timeout=5.0)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            content_length = response.headers.get("content-length")

            analysis = {
                "can_handle": False,
                "confidence": 0.0,
                "reasons": [],
                "content_type": content_type,
                "size_estimate": int(content_length) if content_length else None,
            }

            # Content type analysis
            if "text/html" in content_type:
                analysis["confidence"] += 0.4
                analysis["reasons"].append("HTML content type detected")
            elif "text/plain" in content_type:
                analysis["confidence"] += 0.6
                analysis["reasons"].append("Plain text content")
            elif "application/json" in content_type:
                analysis["confidence"] += 0.5
                analysis["reasons"].append("JSON content")
            elif "application/xml" in content_type or "text/xml" in content_type:
                analysis["confidence"] += 0.5
                analysis["reasons"].append("XML content")
            else:
                analysis["reasons"].append(f"Unknown content type: {content_type}")
                return analysis

            # Size analysis
            if content_length:
                size = int(content_length)
                if size < 100_000:  # 100KB
                    analysis["confidence"] += 0.3
                    analysis["reasons"].append(f"Small content size: {size} bytes")
                elif size < 1_000_000:  # 1MB
                    analysis["confidence"] += 0.1
                    analysis["reasons"].append(f"Medium content size: {size} bytes")
                else:
                    analysis["confidence"] -= 0.2
                    analysis["reasons"].append(f"Large content size: {size} bytes")

            # Server header analysis
            server = response.headers.get("server", "").lower()
            if "nginx" in server or "apache" in server:
                analysis["confidence"] += 0.1
                analysis["reasons"].append(f"Static server detected: {server}")

            # Check for SPA indicators
            if any(
                header in response.headers for header in ["x-powered-by", "x-framework"]
            ):
                analysis["confidence"] -= 0.3
                analysis["reasons"].append("Framework headers suggest dynamic content")

            analysis["can_handle"] = analysis["confidence"] > 0.5

            return analysis

        except Exception as e:
            logger.debug(f"HEAD request failed for {url}: {e}")
            return None

    async def scrape(self, url: str) -> ScrapedContent | None:
        """Scrape content using lightweight HTTP + BeautifulSoup approach.

        Args:
            url: URL to scrape

        Returns:
            ScrapedContent if successful, None if content should escalate to higher tier

        """
        start_time = time.time()

        if not self._client:
            await self.initialize()

        try:
            # Attempt lightweight scraping
            response = await self._client.get(url)
            response.raise_for_status()

            # Parse content with BeautifulSoup
            content = self._extract_content(response.text, url)

            # Validate content quality
            if not self._is_sufficient_content(content):
                logger.debug(
                    f"Insufficient content quality for {url}, escalating to higher tier"
                )
                return None

            extraction_time = (time.time() - start_time) * 1000

            return ScrapedContent(
                url=url,
                title=content["title"],
                text=content["text"],
                headings=content["headings"],
                links=content["links"],
                metadata=content["metadata"],
                extraction_time_ms=extraction_time,
                tier=0,
                success=True,
            )

        except httpx.TimeoutException:
            logger.debug(f"Timeout for {url}, may need browser automation")
            return None
        except httpx.HTTPStatusError as e:
            if e.response.status_code in [403, 429, 503]:
                logger.debug(f"Anti-bot protection detected for {url}, escalating")
                return None
            raise
        except Exception as e:
            logger.warning(f"Error scraping {url}: {e}")
            return None

    def _extract_content(self, html: str, url: str) -> dict[str, Any]:
        """Extract structured content from HTML.

        Args:
            html: Raw HTML content
            url: Source URL for context

        Returns:
            Dictionary with extracted content

        """
        # Use lxml parser for maximum performance
        soup = BeautifulSoup(html, "lxml")

        # Get domain for site-specific extraction
        domain = urlparse(url).netloc.lower()

        # Extract title
        title = ""
        if soup.title:
            title = soup.title.string.strip() if soup.title.string else ""

        # Try site-specific selectors first
        main_content = None
        if domain in self._site_selectors:
            selector = self._site_selectors[domain]["content"]
            main_content = soup.select_one(selector)

        # Fallback to intelligent content detection
        if not main_content:
            main_content = self._find_main_content(soup)

        # Extract structured data
        headings = self._extract_headings(main_content)
        links = self._extract_links(main_content, url)
        text = self._extract_clean_text(main_content)

        # Extract metadata
        metadata = self._extract_metadata(soup)

        return {
            "title": title,
            "text": text,
            "headings": headings,
            "links": links,
            "metadata": metadata,
        }

    def _find_main_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Find main content area using content heuristics.

        Args:
            soup: BeautifulSoup object

        Returns:
            BeautifulSoup object with main content

        """
        # Try common content selectors in order of specificity
        selectors = [
            "main",
            '[role="main"]',
            "article",
            ".content",
            ".main-content",
            ".documentation",
            ".docs-content",
            ".markdown-body",
            "#content",
            ".post-content",
            ".entry-content",
        ]

        for selector in selectors:
            content = soup.select_one(selector)
            if content:
                return content

        # Fallback: remove navigation/footer and return body
        unwanted = soup.find_all(
            ["nav", "header", "footer", "aside", "script", "style"]
        )
        for element in unwanted:
            element.decompose()

        return soup.body or soup

    def _extract_headings(self, content: BeautifulSoup) -> list[dict[str, Any]]:
        """Extract heading structure from content.

        Args:
            content: BeautifulSoup content object

        Returns:
            List of heading dictionaries

        """
        headings = []
        for heading in content.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            level = int(heading.name[1])
            text = heading.get_text(strip=True)
            if text:  # Only include non-empty headings
                headings.append(
                    {
                        "level": level,
                        "text": text,
                        "id": heading.get("id"),
                        "tag": heading.name,
                    }
                )
        return headings

    def _extract_links(
        self, content: BeautifulSoup, base_url: str
    ) -> list[dict[str, str]]:
        """Extract and categorize links from content.

        Args:
            content: BeautifulSoup content object
            base_url: Base URL for resolving relative links

        Returns:
            List of link dictionaries

        """
        links = []
        for link in content.find_all("a", href=True):
            href = link["href"].strip()
            text = link.get_text(strip=True)

            if not href or href.startswith("#"):
                continue  # Skip empty links and anchors

            # Resolve relative URLs
            if not href.startswith(("http://", "https://", "mailto:")):
                href = urljoin(base_url, href)

            # Categorize link type
            link_type = "external"
            if href.startswith("mailto:"):
                link_type = "email"
            elif urlparse(href).netloc == urlparse(base_url).netloc:
                link_type = "internal"

            links.append(
                {
                    "url": href,
                    "text": text,
                    "type": link_type,
                    "title": link.get("title", ""),
                }
            )

        return links

    def _extract_clean_text(self, content: BeautifulSoup) -> str:
        """Extract clean text content.

        Args:
            content: BeautifulSoup content object

        Returns:
            Cleaned text string

        """
        # Remove script and style elements
        for script in content(["script", "style", "noscript"]):
            script.decompose()

        # Get text with space separation
        text = content.get_text(separator=" ", strip=True)

        # Clean and normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(
            r"[\u00a0\u2000-\u200f\u2028-\u202f]", " ", text
        )  # Unicode spaces

        return text.strip()

    def _extract_metadata(self, soup: BeautifulSoup) -> dict[str, Any]:
        """Extract metadata from page.

        Args:
            soup: BeautifulSoup object

        Returns:
            Dictionary with metadata

        """
        metadata = {}

        # Meta tags
        for meta in soup.find_all("meta"):
            name = meta.get("name") or meta.get("property")
            content = meta.get("content")
            if name and content:
                metadata[name] = content

        # Language
        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            metadata["language"] = html_tag["lang"]

        return metadata

    def _is_sufficient_content(self, content: dict[str, Any]) -> bool:
        """Check if extracted content meets quality thresholds.

        Args:
            content: Extracted content dictionary

        Returns:
            True if content is sufficient, False to escalate

        """
        text_length = len(content["text"])

        # Minimum content length check
        if text_length < self._content_threshold:
            logger.debug(
                f"Content too short: {text_length} < {self._content_threshold}"
            )
            return False

        # Check for meaningful structure
        if not content["headings"] and text_length < 500:
            logger.debug("No headings and short content, may need browser rendering")
            return False

        # Check for JavaScript-heavy indicators
        text = content["text"].lower()
        js_indicators = [
            "please enable javascript",
            "loading...",
            "javascript required",
        ]
        if any(indicator in text for indicator in js_indicators):
            logger.debug("JavaScript requirement detected in content")
            return False

        return True
