"""Response models for MCP server tools."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class SearchResult(BaseModel):
    """Search result with metadata"""

    id: str
    content: str
    score: float
    url: str | None = None
    title: str | None = None
    metadata: dict[str, Any] | None = None


class CrawlResult(BaseModel):
    """Result from crawling a single page"""

    url: str = Field(..., description="Page URL")
    title: str = Field(default="", description="Page title")
    content: str = Field(default="", description="Page content")
    word_count: int = Field(default=0, description="Word count")
    success: bool = Field(default=False, description="Success status")
    site_name: str = Field(default="", description="Site name")
    depth: int = Field(default=0, description="Crawl depth")
    scraped_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Crawl timestamp",
    )
    links: list[str] = Field(default_factory=list, description="Extracted links")
    metadata: dict = Field(default_factory=dict, description="Page metadata")
    error: str | None = Field(default=None, description="Error message if failed")
