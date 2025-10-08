"""
Firecrawl v2 adapter.

Implements a thin, async wrapper around the official Firecrawl v2 Python SDK
to provide scrape, crawl, and search capabilities with normalized outputs.

Docs:
- Quickstart and features: https://docs.firecrawl.dev/introduction
- Python SDK + AsyncFirecrawl usage: https://docs.firecrawl.dev/sdks/python
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from numbers import Integral
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError


try:
    # Official SDK (v2+). Async class mirrors the sync API.
    # https://docs.firecrawl.dev/sdks/python
    from firecrawl import AsyncFirecrawl
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        "firecrawl-py is required for FirecrawlAdapter. "
        "Install with `pip install firecrawl-py`."
    ) from exc


logger = logging.getLogger(__name__)


class FirecrawlAdapterConfig(BaseModel):
    """Configuration for FirecrawlAdapter.

    Attributes:
        api_key: Firecrawl API key. Uses env if not provided.
        api_url: Optional custom endpoint.
        default_formats: Preferred output formats.
        timeout_seconds: Overall adapter timeout for waiter methods.
    """

    api_key: str = Field(default="")
    api_url: str | None = Field(default=None)
    default_formats: list[str] = Field(default_factory=lambda: ["markdown", "html"])
    timeout_seconds: int = Field(default=120)


@dataclass(slots=True)
class FirecrawlCrawlOptions:
    """Optional overrides for crawl operations."""

    limit: int = 50
    formats: Sequence[str] | None = None
    poll_interval: int = 2
    timeout: int | None = None


class FirecrawlAdapter:
    """Async adapter over Firecrawl v2.

    Methods mirror the app's provider surface and return normalized dictionaries.
    """

    def __init__(self, config: FirecrawlAdapterConfig) -> None:
        self._config = config
        self._client: AsyncFirecrawl | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the AsyncFirecrawl client."""
        if self._initialized:
            return

        api_key = self._config.api_key or os.getenv("AI_DOCS__FIRECRAWL__API_KEY", "")
        if not api_key:
            msg = "Firecrawl API key missing (AI_DOCS__FIRECRAWL__API_KEY)"
            raise ValueError(msg)

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if self._config.api_url:
            client_kwargs["api_url"] = self._config.api_url

        self._client = AsyncFirecrawl(**client_kwargs)
        self._initialized = True
        logger.info("FirecrawlAdapter initialized")

    async def cleanup(self) -> None:
        """Reset client state."""

        self._client = None
        self._initialized = False
        logger.info("FirecrawlAdapter cleaned up")

    # --- Public API -----------------------------------------------------

    async def scrape(
        self,
        url: str,
        *,
        formats: Sequence[str] | None = None,
        only_main_content: bool | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Scrape a single URL.

        Args:
            url: Target URL.
            formats: Output formats, e.g. ["markdown","html","screenshot"].
            only_main_content: If True, reduce boilerplate.
            timeout: Per-request timeout override in seconds.

        Returns:
            Normalized dict with keys: success, url, content, html, metadata, provider.
        """

        self._ensure_ready()

        try:
            fmts = self._normalize_formats(formats, self._config.default_formats)
        except TypeError as err:
            logger.warning("Firecrawl scrape invalid formats for %s: %s", url, err)
            return {
                "success": False,
                "url": url,
                "error": str(err),
                "provider": "firecrawl",
            }

        to = timeout if timeout is not None else self._config.timeout_seconds
        scrape_kwargs: dict[str, Any] = {"formats": fmts}
        if only_main_content is not None:
            scrape_kwargs["only_main_content"] = only_main_content

        try:
            document = await asyncio.wait_for(
                self._client.scrape(  # type: ignore[attr-defined]
                    url, **scrape_kwargs
                ),
                timeout=to,
            )
            return self._normalize_single(url, document)
        except TimeoutError as err:
            logger.warning("Firecrawl scrape timed out for %s: %s", url, err)
            return {
                "success": False,
                "url": url,
                "error": "scrape timeout",
                "provider": "firecrawl",
            }
        except (ValidationError, ValueError, httpx.HTTPError) as err:
            logger.warning("Firecrawl scrape failed for %s: %s", url, err)
            return {
                "success": False,
                "url": url,
                "error": str(err),
                "provider": "firecrawl",
            }
        except Exception as err:  # pragma: no cover - unexpected SDK failure
            logger.exception("Unexpected Firecrawl scrape failure for %s", url)
            return {
                "success": False,
                "url": url,
                "error": str(err),
                "provider": "firecrawl",
            }

    async def crawl(
        self,
        url: str,
        *,
        options: FirecrawlCrawlOptions | None = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Crawl a site starting at ``url`` and return normalized pages.

        Uses waiter `crawl` for simplicity. Pagination is handled by the SDK.
        See: docs on crawl + pagination.  # noqa: E501

        Args:
            url: Seed URL for the crawl.
            options: Baseline crawl options applied before overrides.
            **overrides: Per-call overrides for ``limit``, ``formats``,
                ``poll_interval``, and ``timeout``.

        Returns:
            Normalized crawl result or an error payload when parameters are invalid.
        """

        self._ensure_ready()

        base_options = options or FirecrawlCrawlOptions()
        try:
            self._validate_overrides(overrides)
            effective_limit = self._coerce_int(
                overrides.get("limit"), base_options.limit
            )
            fmts = self._normalize_formats(
                overrides.get("formats", base_options.formats),
                base_options.formats or self._config.default_formats,
            )
            poll_interval = self._coerce_int(
                overrides.get("poll_interval"), base_options.poll_interval
            )
            to = self._coerce_int(
                overrides.get("timeout"),
                base_options.timeout
                if base_options.timeout is not None
                else self._config.timeout_seconds,
            )
        except TypeError as err:
            logger.warning("Firecrawl crawl parameter error for %s: %s", url, err)
            return {
                "success": False,
                "url": url,
                "error": str(err),
                "provider": "firecrawl",
            }

        try:
            job = await asyncio.wait_for(
                self._client.crawl(  # type: ignore[attr-defined]
                    url=url,
                    limit=effective_limit,
                    scrape_options={"formats": fmts},
                    poll_interval=poll_interval,
                    timeout=to,
                ),
                timeout=to + poll_interval + 5,
            )
            return self._normalize_crawl_job(job, seed_url=url)
        except TimeoutError as err:
            logger.warning("Firecrawl crawl timed out for %s: %s", url, err)
            return {
                "success": False,
                "url": url,
                "error": "crawl timeout",
                "provider": "firecrawl",
            }
        except (ValidationError, ValueError, httpx.HTTPError) as err:
            logger.warning("Firecrawl crawl failed for %s: %s", url, err)
            return {
                "success": False,
                "url": url,
                "error": str(err),
                "provider": "firecrawl",
            }
        except Exception as err:  # pragma: no cover - unexpected SDK failure
            logger.exception("Unexpected Firecrawl crawl failure for %s", url)
            return {
                "success": False,
                "url": url,
                "error": str(err),
                "provider": "firecrawl",
            }

    async def search(
        self, query: str, *, limit: int = 3, sources: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """Search the web and optionally scrape results.

        Firecrawl v2 supports web/news/images and returns full content when
        requested. Example usage in SDK docs.  # noqa: E501
        """

        self._ensure_ready()

        kwargs: dict[str, Any] = {"limit": limit}
        if sources:
            kwargs["sources"] = list(sources)

        to = self._config.timeout_seconds

        try:
            result = await asyncio.wait_for(
                self._client.search(  # type: ignore[attr-defined]
                    query, **kwargs
                ),
                timeout=to,
            )
            normalized_results = self._normalize_search_results(result)
            return {
                "success": True,
                "query": query,
                "results": normalized_results,
                "sources": kwargs.get("sources", []),
                "provider": "firecrawl",
            }
        except TimeoutError as err:
            logger.warning("Firecrawl search timed out for %s: %s", query, err)
            return {
                "success": False,
                "query": query,
                "error": "search timeout",
                "provider": "firecrawl",
            }
        except (ValidationError, ValueError, httpx.HTTPError) as err:
            logger.warning("Firecrawl search failed for %s: %s", query, err)
            return {
                "success": False,
                "query": query,
                "error": str(err),
                "provider": "firecrawl",
            }
        except Exception as err:  # pragma: no cover - unexpected SDK failure
            logger.exception("Unexpected Firecrawl search failure for %s", query)
            return {
                "success": False,
                "query": query,
                "error": str(err),
                "provider": "firecrawl",
            }

    # --- Internals ------------------------------------------------------

    def _ensure_ready(self) -> None:
        """Ensure FirecrawlAdapter is initialized and ready."""

        if not self._initialized or self._client is None:
            raise RuntimeError("FirecrawlAdapter not initialized")

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        """Return ``value`` as an ``int`` or fall back to ``default``.

        Accepts integral numeric types and digit-only strings while rejecting
        booleans and non-numeric inputs.
        """

        if value is None:
            return default
        if isinstance(value, bool):
            msg = "Expected integer override, received bool"
            raise TypeError(msg)
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError as exc:  # pragma: no cover - invalid str handled below
                msg = f"Expected integer override, received {value!r}"
                raise TypeError(msg) from exc
        msg = f"Expected integer override, received {value!r}"
        raise TypeError(msg)

    @staticmethod
    def _validate_overrides(overrides: Mapping[str, Any]) -> None:
        """Ensure only supported override keys are provided."""

        extra_keys = set(overrides) - {"limit", "formats", "poll_interval", "timeout"}
        if extra_keys:
            unexpected = ", ".join(sorted(extra_keys))
            msg = f"Unsupported crawl overrides: {unexpected}"
            raise TypeError(msg)

    @staticmethod
    def _normalize_formats(value: Any, fallback: Sequence[str]) -> list[str]:
        """Normalize format inputs to a concrete list of strings."""

        if value is None:
            result = list(fallback)
        elif isinstance(value, str):
            result = [value]
        elif isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            result = [str(item) for item in value]
        else:
            msg = "formats must be a string or a sequence of strings"
            raise TypeError(msg)

        return result or list(fallback)

    @staticmethod
    def _ensure_dict(data: Any) -> dict[str, Any]:
        if data is None:
            return {}
        if isinstance(data, BaseModel):
            return data.model_dump(mode="python", exclude_none=True)
        if isinstance(data, Mapping):
            return dict(data)
        return {}

    @classmethod
    def _normalize_single(cls, url: str | None, data: Any) -> dict[str, Any]:
        """Normalize SDK result shape to app schema."""

        payload = cls._ensure_dict(data)
        nested = payload.get("data")
        if isinstance(nested, BaseModel | Mapping):
            root = cls._ensure_dict(nested)
        else:
            root = payload
        metadata = cls._ensure_dict(root.get("metadata"))

        resolved_url = (
            metadata.get("source_url")
            or metadata.get("sourceURL")
            or metadata.get("url")
            or url
            or ""
        )
        title = metadata.get("title") or metadata.get("og_title") or ""
        content = root.get("markdown") or root.get("summary") or ""
        html = root.get("html") or root.get("raw_html") or ""

        normalized: dict[str, Any] = {
            "success": True,
            "url": resolved_url,
            "title": title,
            "content": content,
            "html": html,
            "metadata": metadata,
            "provider": "firecrawl",
        }

        if warning := root.get("warning"):
            normalized["warning"] = warning
        if links := root.get("links"):
            normalized["links"] = links
        if actions := root.get("actions"):
            normalized["actions"] = actions

        return normalized

    @classmethod
    def _normalize_crawl_job(cls, job: Any, *, seed_url: str) -> dict[str, Any]:
        """Normalize a CrawlJob to the app schema."""

        job_dict = cls._ensure_dict(job)
        raw_pages = job_dict.pop("data", [])
        pages = [cls._normalize_single(None, page) for page in raw_pages]

        status = job_dict.get("status")
        completed = int(job_dict.get("completed", len(pages)))
        total = int(job_dict.get("total", completed))
        credits_used = job_dict.get("credits_used")
        expires_at = cls._format_datetime(job_dict.get("expires_at"))

        normalized = {
            "success": status == "completed",
            "status": status,
            "seed_url": seed_url,
            "completed": completed,
            "total": total,
            "total_pages": len(pages),
            "credits_used": credits_used,
            "expires_at": expires_at,
            "next": job_dict.get("next"),
            "pages": pages,
            "provider": "firecrawl",
        }

        return normalized

    @staticmethod
    def _format_datetime(value: Any) -> str | None:
        """Format datetime to ISO 8601 string or return None."""

        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, str):
            return value
        return None

    @classmethod
    def _normalize_search_results(cls, result: Any) -> dict[str, Any]:
        """Normalize search results with potential nested items."""

        data = cls._ensure_dict(result)
        normalized: dict[str, Any] = {}
        for key, raw_items in data.items():
            if isinstance(raw_items, str | bytes) or not isinstance(
                raw_items, Sequence
            ):
                continue
            items: list[Any] = []
            for item in raw_items:
                if isinstance(item, BaseModel):
                    if hasattr(item, "markdown") or hasattr(item, "metadata"):
                        normalized_item = cls._normalize_single(None, item)
                        normalized_item.pop("provider", None)
                        normalized_item.pop("success", None)
                        items.append(normalized_item)
                    else:
                        items.append(item.model_dump(mode="python", exclude_none=True))
                elif isinstance(item, Mapping):
                    items.append(dict(item))
                else:
                    items.append(item)
            if items:
                normalized[key] = items
        return normalized
