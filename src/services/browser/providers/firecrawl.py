"""Firecrawl provider integration leveraging the official SDK."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, cast


try:
    from firecrawl import AsyncFirecrawlApp  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - compat fallback
    from firecrawl import (  # type: ignore[attr-defined]
        AsyncFirecrawl as AsyncFirecrawlApp,
    )

from src.config.browser import FirecrawlSettings
from src.services.browser.errors import BrowserProviderError
from src.services.browser.models import BrowserResult, ProviderKind, ScrapeRequest
from src.services.browser.runtime import execute_with_retry

from .base import BrowserProvider, ProviderContext


MIN_TIMEOUT_SECONDS = 0.001


class FirecrawlProvider(BrowserProvider):
    """Thin wrapper around the Firecrawl v2 Python SDK."""

    kind = ProviderKind.FIRECRAWL

    def __init__(self, context: ProviderContext, settings: FirecrawlSettings) -> None:
        super().__init__(context)
        self._settings = settings
        self._client: AsyncFirecrawlApp | None = None

    async def initialize(self) -> None:
        """Instantiate the AsyncFirecrawl client."""

        api_key = self._settings.api_key or os.getenv(
            "AI_DOCS__BROWSER__FIRECRAWL__API_KEY"
        )
        if not api_key:
            raise BrowserProviderError(
                "Firecrawl API key missing",
                provider=self.kind.value,
            )
        kwargs: dict[str, Any] = {"api_key": api_key}
        if self._settings.api_url:
            kwargs["api_url"] = str(self._settings.api_url)
        self._client = AsyncFirecrawlApp(**kwargs)

    async def close(self) -> None:
        """Firecrawl client does not expose a close hook."""

        self._client = None

    def _coerce_timeout(self, raw_timeout: Any) -> float:
        """Normalize Firecrawl timeout override to a positive float."""

        try:
            timeout = float(raw_timeout)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise BrowserProviderError(
                "Firecrawl timeout override must be numeric",
                provider=self.kind.value,
            ) from exc
        if timeout <= 0:
            raise BrowserProviderError(
                "Firecrawl timeout override must be greater than zero",
                provider=self.kind.value,
            )
        return timeout

    def _effective_formats(
        self, metadata: Mapping[str, Any] | None, request_formats: Sequence[str] | None
    ) -> list[str]:
        if request_formats:
            return list(request_formats)
        meta_formats = []
        if metadata:
            meta = metadata.get("firecrawl")
            if isinstance(meta, dict):
                formats = meta.get("formats")
                if isinstance(formats, list | tuple):
                    meta_formats = [str(fmt) for fmt in formats]
        return meta_formats or list(self._settings.default_formats)

    def _compose_scrape_options(
        self, request: ScrapeRequest
    ) -> tuple[list[str], dict[str, Any], float]:
        """Build format list, provider options, and effective timeout."""

        metadata = request.metadata or {}
        provider_meta = metadata.get("firecrawl")
        custom_formats: Sequence[str] | None = None
        options: dict[str, Any] = {}
        metadata_timeout: float | None = None
        if isinstance(provider_meta, dict):
            meta_formats = provider_meta.get("formats")
            if isinstance(meta_formats, list | tuple):
                custom_formats = [str(fmt) for fmt in meta_formats]
            for key, value in provider_meta.items():
                if key == "formats":
                    continue
                if key == "timeout":
                    metadata_timeout = self._coerce_timeout(value)
                    continue
                options[key] = value

        request_timeout = None
        if request.timeout_ms is not None:
            request_timeout = max(request.timeout_ms / 1000, 0.0)

        formats = self._effective_formats(metadata, custom_formats)
        effective_timeout = float(self._settings.timeout_seconds)
        if metadata_timeout is not None:
            effective_timeout = metadata_timeout
        if request_timeout is not None:
            effective_timeout = min(effective_timeout, request_timeout)
        effective_timeout = max(effective_timeout, MIN_TIMEOUT_SECONDS)
        options["timeout"] = effective_timeout

        return formats, options, effective_timeout

    def _build_result(
        self, document: dict[str, Any], request: ScrapeRequest
    ) -> BrowserResult:
        data = document.get("data") if isinstance(document, dict) else None
        root = data if isinstance(data, dict) else document
        metadata_out = dict(root.get("metadata") or {})
        content = root.get("markdown") or root.get("summary") or ""
        html = root.get("html") or root.get("raw_html") or ""
        resulting_url = (
            metadata_out.get("source_url")
            or metadata_out.get("url")
            or document.get("url")
            or request.url
        )
        links_obj = root.get("links")
        links = links_obj if isinstance(links_obj, dict) else None
        return BrowserResult(
            success=True,
            url=str(resulting_url),
            title=metadata_out.get("title", ""),
            content=content,
            html=html,
            metadata=root,
            provider=self.kind,
            links=links,
            assets=None,
            elapsed_ms=None,
        )

    async def scrape(self, request: ScrapeRequest) -> BrowserResult:
        """Call the Firecrawl scrape endpoint."""

        if self._client is None:  # pragma: no cover - guarded by lifecycle
            raise RuntimeError("Provider not initialized")

        formats, options, _ = self._compose_scrape_options(request)

        client = self._client
        assert client is not None  # narrow for typing

        async def _call() -> Any:
            return await client.scrape(  # type: ignore[no-any-return]
                url=request.url,
                formats=formats,
                **options,
            )

        document = cast(
            dict[str, Any],
            await execute_with_retry(
                provider=self.kind,
                operation="scrape",
                func=_call,
            ),
        )

        return self._build_result(document, request)

    async def crawl_site(
        self, url: str, *, limit: int | None = None, **overrides: Any
    ) -> dict[str, Any]:
        """Invoke Firecrawl crawl endpoint with retry + metrics."""

        if self._client is None:  # pragma: no cover - lifecycle guard
            raise RuntimeError("Provider not initialized")

        options = dict(overrides)
        if limit is not None:
            options.setdefault("limit", limit)
        options.setdefault(
            "scrape_options", {"formats": self._settings.default_formats}
        )
        options.setdefault("timeout", self._settings.timeout_seconds)

        client = self._client
        assert client is not None

        async def _call() -> Any:
            return await client.crawl(url=url, **options)

        return cast(
            dict[str, Any],
            await execute_with_retry(
                provider=self.kind,
                operation="crawl",
                func=_call,
            ),
        )

    async def search(
        self, query: str, *, limit: int = 3, timeout: int | None = None
    ) -> dict[str, Any]:
        """Call Firecrawl search endpoint and return raw result dict."""

        if self._client is None:
            raise RuntimeError("Provider not initialized")

        kw: dict[str, Any] = {"limit": limit}
        if timeout is not None:
            kw["timeout"] = timeout

        client = self._client
        assert client is not None

        async def _call() -> Any:
            return await client.search(query, **kw)

        return cast(
            dict[str, Any],
            await execute_with_retry(
                provider=self.kind,
                operation="search",
                func=_call,
            ),
        )

    async def start_crawl(self, url: str, *, limit: int = 10) -> Any:
        """Start an async crawl and return job descriptor (SDK model/dict)."""

        if self._client is None:
            raise RuntimeError("Provider not initialized")

        client = self._client
        assert client is not None

        async def _call() -> Any:
            return await client.start_crawl(url, limit=limit)

        return await execute_with_retry(
            provider=self.kind,
            operation="start_crawl",
            func=_call,
        )

    async def get_crawl_status(self, job_id: str) -> Any:
        """Fetch crawl job status from Firecrawl."""

        if self._client is None:
            raise RuntimeError("Provider not initialized")

        client = self._client
        assert client is not None

        async def _call() -> Any:
            return await client.get_crawl_status(job_id)

        return await execute_with_retry(
            provider=self.kind,
            operation="crawl_status",
            func=_call,
        )

    async def batch_scrape(
        self,
        urls: Sequence[str],
        *,
        formats: Sequence[str] | None = None,
        poll_interval: int = 2,
        timeout: int | None = None,
    ) -> Any:
        """Submit a batch scrape job and wait for results via SDK convenience."""

        if self._client is None:
            raise RuntimeError("Provider not initialized")

        fmts = list(formats or self._settings.default_formats)
        kw: dict[str, Any] = {"formats": fmts, "poll_interval": poll_interval}
        if timeout is not None:
            kw["timeout"] = timeout

        client = self._client
        assert client is not None

        async def _call() -> Any:
            return await client.batch_scrape(list(urls), **kw)

        return await execute_with_retry(
            provider=self.kind,
            operation="batch_scrape",
            func=_call,
        )
