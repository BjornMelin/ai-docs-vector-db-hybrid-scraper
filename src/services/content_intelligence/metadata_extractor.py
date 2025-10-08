"""Metadata extraction and enrichment for content intelligence.

This module provides automatic metadata enrichment by extracting structured
metadata from page elements, generating semantic tags and categories,
parsing timestamps and freshness indicators, and detecting content hierarchy.
"""

import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from .models import ContentMetadata


logger = logging.getLogger(__name__)

# Metadata extraction constants
MAX_RELATED_URLS = 10
MAX_TITLE_LENGTH = 100
MIN_DESCRIPTION_LENGTH = 50
MAX_DESCRIPTION_LENGTH = 300


class MetadataExtractor:
    """Automatic metadata enrichment for web content."""

    def __init__(self):
        """Initialize metadata extractor."""
        self._initialized = False

        # Common metadata patterns
        self._meta_patterns = {
            "title": [
                r"<title[^>]*>([^<]+)</title>",
                r'<meta\s+property=["\']og:title["\']\s+content=["\']([^"\']+)["\']',
                r'<meta\s+name=["\']title["\']\s+content=["\']([^"\']+)["\']',
                r"<h1[^>]*>([^<]+)</h1>",
            ],
            "description": [
                r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']+)["\']',
                r'<meta\s+property=["\']og:description["\']\s+content=["\']([^"\']+)["\']',
                r'<meta\s+name=["\']twitter:description["\']\s+content=["\']([^"\']+)["\']',
            ],
            "author": [
                r'<meta\s+name=["\']author["\']\s+content=["\']([^"\']+)["\']',
                r'<meta\s+property=["\']article:author["\']\s+content=["\']([^"\']+)["\']',
                r'<span[^>]*class=["\'][^"\']*author[^"\']*["\'][^>]*>([^<]+)</span>',
                r"by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            ],
            "published_date": [
                r'<meta\s+property=["\']article:published_time["\']\s+content=["\']([^"\']+)["\']',
                r'<meta\s+name=["\']publish[^"\']*["\']\s+content=["\']([^"\']+)["\']',
                r'<time[^>]*datetime=["\']([^"\']+)["\']',
                r"published[^:]*:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
            ],
            "last_modified": [
                r'<meta\s+property=["\']article:modified_time["\']\s+content=["\']([^"\']+)["\']',
                r'<meta\s+name=["\']last-modified["\']\s+content=["\']([^"\']+)["\']',
                r"last\s+updated[^:]*:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
            ],
            "language": [
                r'<html[^>]*lang=["\']([^"\']+)["\']',
                r'<meta\s+name=["\']language["\']\s+content=["\']([^"\']+)["\']',
                r'<meta\s+http-equiv=["\']content-language["\']\s+content=["\']([^"\']+)["\']',
            ],
            "charset": [
                r'<meta\s+charset=["\']([^"\']+)["\']',
                r'<meta\s+http-equiv=["\']content-type["\']\s+content=["\'][^;]*charset=([^"\';\s]+)',
            ],
            "keywords": [
                r'<meta\s+name=["\']keywords["\']\s+content=["\']([^"\']+)["\']',
                r'<meta\s+property=["\']article:tag["\']\s+content=["\']([^"\']+)["\']',
            ],
        }

        # Schema.org patterns
        self._schema_patterns = {
            "article": r'<[^>]*itemtype=["\'][^"\']*schema\.org/Article["\']',
            "blogposting": r'<[^>]*itemtype=["\'][^"\']*schema\.org/BlogPosting["\']',
            "webpage": r'<[^>]*itemtype=["\'][^"\']*schema\.org/WebPage["\']',
            "person": r'<[^>]*itemtype=["\'][^"\']*schema\.org/Person["\']',
            "organization": r'<[^>]*itemtype=["\'][^"\']*schema\.org/Organization["\']',
            "product": r'<[^>]*itemtype=["\'][^"\']*schema\.org/Product["\']',
        }

        # Entity extraction patterns
        self._entity_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "url": r'https?://[^\s<>"\']+',
            "phone": r"(\+?1[-.\s]?)?(\()?[0-9]{3}(\))?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}",
            "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
            "version": r"\bv?\d+\.\d+(\.\d+)?(-[a-zA-Z0-9]+)?\b",
        }

    async def initialize(self) -> None:
        """Initialize the metadata extractor."""
        self._initialized = True
        logger.info("MetadataExtractor initialized")

    async def extract_metadata(
        self,
        content: str,
        url: str,
        raw_html: str | None = None,
        extraction_metadata: dict[str, Any] | None = None,
    ) -> ContentMetadata:
        """Extract and enrich metadata from content and HTML.

        Args:
            content: Processed text content
            url: Source URL
            raw_html: Optional raw HTML for metadata extraction
            extraction_metadata: Optional metadata from extraction process

        Returns:
            ContentMetadata: Enriched metadata

        """
        if not self._initialized:
            await self.initialize()

        # Start with basic analysis
        metadata = ContentMetadata(url=url)

        # Extract basic content characteristics
        words = content.split()
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        metadata.word_count = len(words)
        metadata.char_count = len(content)
        metadata.paragraph_count = len(paragraphs)

        # Count structural elements
        metadata.heading_count = len(re.findall(r"^#{1,6}\s", content, re.MULTILINE))
        metadata.link_count = len(re.findall(r"\[([^\]]+)\]\([^)]+\)", content))
        metadata.image_count = len(re.findall(r"!\[([^\]]*)\]\([^)]+\)", content))

        # Extract metadata from HTML if available
        if raw_html:
            await self._extract_html_metadata(metadata, raw_html)

        # Extract content-based metadata
        await self._extract_content_metadata(metadata, content, url)

        # Extract semantic information
        await self._extract_semantic_metadata(metadata, content)

        # Extract technical metadata
        await self._extract_technical_metadata(
            metadata, content, url, extraction_metadata
        )

        # Extract hierarchy and relationships
        await self._extract_hierarchy_metadata(metadata, content, url)

        # Extract structured data
        if raw_html:
            await self._extract_structured_data(metadata, raw_html)

        return metadata

    async def _extract_html_metadata(
        self, metadata: ContentMetadata, html: str
    ) -> None:
        """Extract metadata from HTML meta tags and elements.

        Args:
            metadata: ContentMetadata to update
            html: Raw HTML content

        """
        # Extract basic meta information
        for field, patterns in self._meta_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    if value and not getattr(metadata, field, None):
                        if field in ["published_date", "last_modified"]:
                            parsed_date = self._parse_date(value)
                            if parsed_date:
                                setattr(metadata, field, parsed_date)
                        else:
                            setattr(metadata, field, value)
                        break

        # Extract keywords and tags
        keywords_meta = re.findall(
            r'<meta\s+name=["\']keywords["\']\s+content=["\']([^"\']+)["\']',
            html,
            re.IGNORECASE,
        )
        if keywords_meta:
            keywords = [kw.strip() for kw in keywords_meta[0].split(",")]
            metadata.keywords.extend(keywords)

        # Extract article tags
        tags_meta = re.findall(
            r'<meta\s+property=["\']article:tag["\']\s+content=["\']([^"\']+)["\']',
            html,
            re.IGNORECASE,
        )
        metadata.tags.extend(tags_meta)

    async def _extract_content_metadata(
        self, metadata: ContentMetadata, content: str, _url: str
    ) -> None:
        """Extract metadata from content analysis.

        Args:
            metadata: ContentMetadata to update
            content: Text content
            url: Source URL

        """
        # Extract title from content if not already set
        if not metadata.title:
            # Look for markdown headers
            h1_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            if h1_match:
                metadata.title = h1_match.group(1).strip()
            else:
                # Use first line if it looks like a title
                first_line = content.split("\n")[0].strip()
                if (
                    first_line
                    and len(first_line) < MAX_TITLE_LENGTH
                    and not first_line.endswith(".")
                ):
                    metadata.title = first_line

        # Extract description from content if not set
        if not metadata.description:
            # Use first paragraph as description
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            if paragraphs:
                first_para = paragraphs[0]
                if (
                    len(first_para) > MIN_DESCRIPTION_LENGTH
                    and len(first_para) < MAX_DESCRIPTION_LENGTH
                ):
                    metadata.description = first_para

        # Extract entities
        entities = []
        for entity_type, pattern in self._entity_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            entities.extend(
                [
                    {
                        "type": entity_type,
                        "value": match if isinstance(match, str) else match[0],
                        "text": match if isinstance(match, str) else "".join(match),
                    }
                    for match in matches
                ]
            )
        metadata.entities = entities

        # Extract breadcrumbs
        breadcrumb_patterns = [
            r"([^>]+)\s*>\s*([^>]+)\s*>\s*([^>]+)",  # Basic breadcrumb
            r"Home\s*[>|/]\s*([^>|/]+)\s*[>|/]\s*([^>|/]+)",  # Home-based breadcrumb
        ]

        for pattern in breadcrumb_patterns:
            match = re.search(pattern, content)
            if match:
                metadata.breadcrumbs = [
                    part.strip() for part in match.groups() if part.strip()
                ]
                break

    async def _extract_semantic_metadata(
        self, metadata: ContentMetadata, content: str
    ) -> None:
        """Extract semantic metadata from content.

        Args:
            metadata: ContentMetadata to update
            content: Text content

        """
        # Extract topics based on content analysis
        content_lower = content.lower()

        # Technical topics
        technical_topics = {
            "programming": [
                "code",
                "function",
                "class",
                "method",
                "programming",
                "software",
            ],
            "api": ["api", "endpoint", "request", "response", "json", "rest"],
            "database": ["database", "sql", "query", "table", "record"],
            "web": ["html", "css", "javascript", "web", "browser", "frontend"],
            "security": ["security", "auth", "password", "encryption", "vulnerability"],
            "machine learning": [
                "ml",
                "ai",
                "model",
                "training",
                "algorithm",
                "neural",
            ],
            "devops": ["docker", "kubernetes", "deployment", "ci/cd", "pipeline"],
        }

        for topic, keywords in technical_topics.items():
            if any(keyword in content_lower for keyword in keywords):
                metadata.topics.append(topic)

        # Extract semantic tags
        tag_patterns = [
            r"#(\w+)",  # Hashtags
            r"tags?\s*:\s*([^,\n]+(?:,\s*[^,\n]+)*)",  # Tag lists
            r"categories?\s*:\s*([^,\n]+(?:,\s*[^,\n]+)*)",  # Category lists
        ]

        for pattern in tag_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, str):
                    if "," in match:
                        tags = [tag.strip() for tag in match.split(",")]
                        metadata.tags.extend(tags)
                    else:
                        metadata.tags.append(match.strip())

        # Remove duplicates and clean tags
        metadata.tags = list({tag for tag in metadata.tags if tag and len(tag) > 1})
        metadata.topics = list(set(metadata.topics))

    async def _extract_technical_metadata(
        self,
        metadata: ContentMetadata,
        content: str,
        url: str,
        extraction_metadata: dict[str, Any] | None,
    ) -> None:
        """Extract technical metadata.

        Args:
            metadata: ContentMetadata to update
            content: Text content
            url: Source URL
            extraction_metadata: Optional extraction metadata

        """
        # Generate content hash
        content_bytes = content.encode("utf-8")
        metadata.content_hash = hashlib.sha256(content_bytes).hexdigest()

        # Set extraction method from metadata
        if extraction_metadata:
            metadata.extraction_method = extraction_metadata.get("tier_used", "unknown")
            metadata.page_load_time_ms = extraction_metadata.get(
                "automation_time_ms", 0.0
            )

        # Parse URL for additional context
        parsed_url = urlparse(url)
        if parsed_url.path:
            path_parts = [part for part in parsed_url.path.split("/") if part]
            if len(path_parts) > 1 and len(path_parts) >= 2:
                # Use path structure to infer relationships
                metadata.parent_url = (
                    f"{parsed_url.scheme}://{parsed_url.netloc}/{path_parts[0]}/"
                )

    async def _extract_hierarchy_metadata(
        self, metadata: ContentMetadata, content: str, url: str
    ) -> None:
        """Extract content hierarchy and relationship metadata.

        Args:
            metadata: ContentMetadata to update
            content: Text content
            url: Source URL

        """
        # Extract related URLs from content
        url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+|\[[^\]]+\]\([^)]+\)'
        found_urls = re.findall(url_pattern, content)

        # Filter and process URLs
        related_urls = []
        for found_url in found_urls:
            # Handle markdown links
            processed_url = found_url
            if found_url.startswith("[") and "](" in found_url:
                url_match = re.search(r"\]\(([^)]+)\)", found_url)
                if url_match:
                    processed_url = url_match.group(1)

            # Add protocol if missing
            if processed_url.startswith("www."):
                processed_url = "https://" + processed_url

            # Filter out same domain if different page
            parsed_current = urlparse(url)
            parsed_found = urlparse(processed_url)

            if (
                parsed_found.netloc
                and parsed_found.netloc != parsed_current.netloc
                and processed_url not in related_urls
            ):
                related_urls.append(processed_url)

        metadata.related_urls = related_urls[:MAX_RELATED_URLS]

    async def _extract_structured_data(
        self, metadata: ContentMetadata, html: str
    ) -> None:
        """Extract structured data and schema information.

        Args:
            metadata: ContentMetadata to update
            html: Raw HTML content

        """
        # Detect Schema.org types
        schema_types = []
        for schema_type, pattern in self._schema_patterns.items():
            if re.search(pattern, html, re.IGNORECASE):
                schema_types.append(schema_type)
        metadata.schema_types = schema_types

        # Extract JSON-LD structured data
        json_ld_pattern = (
            r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
        )
        json_ld_matches = re.findall(json_ld_pattern, html, re.DOTALL | re.IGNORECASE)

        structured_data = {}
        for match in json_ld_matches:
            try:
                data = json.loads(match.strip())
                if isinstance(data, dict):
                    if "@type" in data:
                        structured_data[data["@type"]] = data
                    elif "type" in data:
                        structured_data[data["type"]] = data
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "@type" in item:
                            structured_data[item["@type"]] = item
            except json.JSONDecodeError:
                continue

        metadata.structured_data = structured_data

        # Extract microdata
        microdata_pattern = r'<[^>]*itemprop=["\']([^"\']+)["\'][^>]*>([^<]*)</[^>]*>'
        microdata_matches = re.findall(microdata_pattern, html, re.IGNORECASE)

        microdata = {}
        for prop, value in microdata_matches:
            if prop and value.strip():
                microdata[prop] = value.strip()

        if microdata:
            metadata.structured_data["microdata"] = microdata

    def _parse_date(self, date_str: str) -> datetime | None:
        """Parse date string into datetime object.

        Args:
            date_str: Date string to parse

        Returns:
            datetime | None: Parsed datetime or None if parsing fails

        """
        # Common date formats
        date_formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%d %H:%M:%S",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%B %d, %Y",
            "%d %B %Y",
        ]

        # Clean the date string
        date_str = date_str.strip()

        # Remove timezone info for simple parsing
        date_str = re.sub(r"[+-]\d{2}:?\d{2}$", "", date_str)

        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue

        # Try to extract just the date part if full parsing fails
        date_match = re.search(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})", date_str)
        if date_match:
            try:
                return datetime.strptime(
                    date_match.group(1).replace("/", "-"), "%Y-%m-%d"
                ).replace(tzinfo=UTC)
            except ValueError:
                pass

        return None

    async def cleanup(self) -> None:
        """Cleanup metadata extractor resources."""
        self._initialized = False
        logger.info("MetadataExtractor cleaned up")
