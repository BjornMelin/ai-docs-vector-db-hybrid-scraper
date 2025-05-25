"""Response models for MCP server tools."""

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