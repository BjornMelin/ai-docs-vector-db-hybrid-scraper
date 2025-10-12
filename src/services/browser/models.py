"""Shared data models for browser orchestration."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProviderKind(str, Enum):
    """Supported browser automation providers."""

    LIGHTWEIGHT = "lightweight"
    CRAWL4AI = "crawl4ai"
    PLAYWRIGHT = "playwright"
    BROWSER_USE = "browser_use"
    FIRECRAWL = "firecrawl"


@dataclass(slots=True)
class ScrapeRequest:
    """Normalized request issued to the router."""

    url: str
    provider: ProviderKind | None = None
    timeout_ms: int | None = None
    require_interaction: bool = False
    actions: Sequence[Mapping[str, Any]] | None = None
    instructions: Sequence[Mapping[str, Any]] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BrowserResult:
    """Standardized scrape result returned by providers."""

    success: bool
    url: str
    title: str
    content: str
    html: str
    metadata: MutableMapping[str, Any]
    provider: ProviderKind
    links: MutableMapping[str, Any] | None = None
    assets: MutableMapping[str, Any] | None = None
    elapsed_ms: int | None = None

    @classmethod
    def failure(
        cls,
        *,
        url: str,
        provider: ProviderKind,
        error: str,
        metadata: Mapping[str, Any] | None = None,
        elapsed_ms: int | None = None,
    ) -> BrowserResult:
        """Factory helper for failed attempts."""

        meta: MutableMapping[str, Any] = {
            "error": error,
            **({} if metadata is None else dict(metadata)),
        }
        return cls(
            success=False,
            url=url,
            title="",
            content="",
            html="",
            metadata=meta,
            provider=provider,
            links=None,
            assets=None,
            elapsed_ms=elapsed_ms,
        )
