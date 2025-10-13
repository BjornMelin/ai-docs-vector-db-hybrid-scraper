"""Lightweight HTTP scraping provider."""

from __future__ import annotations

import asyncio

import httpx
import trafilatura

from src.config.browser import LightweightSettings
from src.services.browser.models import BrowserResult, ProviderKind, ScrapeRequest
from src.services.browser.runtime import execute_with_retry

from .base import BrowserProvider, ProviderContext


class LightweightProvider(BrowserProvider):
    """Fetches pages via httpx + Trafilatura."""

    kind = ProviderKind.LIGHTWEIGHT

    def __init__(self, context: ProviderContext, settings: LightweightSettings) -> None:
        super().__init__(context)
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        """Create HTTP client."""

        self._client = httpx.AsyncClient(
            follow_redirects=self._settings.allow_redirects,
            timeout=self._settings.timeout_seconds,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                )
            },
        )

    async def close(self) -> None:
        """Dispose HTTP client."""

        if self._client:
            await self._client.aclose()
            self._client = None

    async def scrape(self, request: ScrapeRequest) -> BrowserResult:
        """Fetch and extract markdown content."""

        if self._client is None:  # pragma: no cover - guarded by lifecycle
            raise RuntimeError("Provider not initialized")

        async def _call() -> httpx.Response:
            assert self._client is not None
            return await self._client.get(request.url)

        response = await execute_with_retry(
            provider=self.kind,
            operation="fetch",
            func=_call,
        )
        response.raise_for_status()
        html = response.text
        loop = asyncio.get_running_loop()
        extracted = await loop.run_in_executor(
            None,
            lambda: trafilatura.extract(html, include_formatting=True),
        )
        content = extracted or ""
        return BrowserResult(
            success=True,
            url=str(response.url),
            title="",
            content=content,
            html=html,
            metadata={
                "status_code": response.status_code,
                "headers": dict(response.headers),
            },
            provider=self.kind,
            links=None,
            assets=None,
            elapsed_ms=None,
        )
