"""Service health checks implemented with async-first primitives."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

import httpx
from openai import AsyncOpenAI, OpenAIError
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from redis import asyncio as redis_async

from src.config import Settings
from src.config.models import CrawlProvider, EmbeddingProvider


__all__ = ["HealthCheckResult", "perform_health_checks", "summarize_results"]


_STATUS = Literal["healthy", "unhealthy", "skipped"]


@dataclass(slots=True)
class HealthCheckResult:
    """Structured outcome for a single service health check."""

    service: str
    status: _STATUS
    latency_ms: float | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def connected(self) -> bool:
        """Return True when the service reported a healthy status."""

        return self.status == "healthy"


async def _check_qdrant(config: Settings) -> HealthCheckResult:
    """Perform health check for Qdrant vector database."""

    start = perf_counter()
    client = AsyncQdrantClient(
        url=config.qdrant.url,
        api_key=config.qdrant.api_key,
        timeout=int(config.qdrant.timeout),
    )
    try:
        collections = await client.get_collections()
    except (
        UnexpectedResponse,
        ValueError,
        ConnectionError,
        TimeoutError,
        RuntimeError,
    ) as exc:
        return HealthCheckResult(
            service="qdrant",
            status="unhealthy",
            error=str(exc),
        )
    finally:
        await client.close()
    latency = (perf_counter() - start) * 1000
    collection_count = len(collections.collections)
    return HealthCheckResult(
        service="qdrant",
        status="healthy",
        latency_ms=latency,
        details={"collections": collection_count, "url": config.qdrant.url},
    )


async def _check_redis(config: Settings) -> HealthCheckResult:
    """Perform health check for Redis cache service."""

    if not config.cache.enable_redis_cache:
        return HealthCheckResult(
            service="redis",
            status="skipped",
            details={"reason": "disabled"},
        )
    start = perf_counter()
    client = redis_async.from_url(config.cache.redis_url)
    try:
        await client.ping()
    except (redis_async.RedisError, ConnectionError, TimeoutError, ValueError) as exc:
        return HealthCheckResult(
            service="redis",
            status="unhealthy",
            error=str(exc),
        )
    finally:
        await client.aclose()
    latency = (perf_counter() - start) * 1000
    return HealthCheckResult(
        service="redis",
        status="healthy",
        latency_ms=latency,
        details={"url": config.cache.redis_url},
    )


async def _check_openai(config: Settings) -> HealthCheckResult:
    """Perform health check for OpenAI API service."""

    if config.embedding_provider != EmbeddingProvider.OPENAI:
        return HealthCheckResult(
            service="openai",
            status="skipped",
            details={"reason": "provider_disabled"},
        )
    api_key = config.openai.api_key or config.openai.api_key
    if not api_key:
        return HealthCheckResult(
            service="openai",
            status="unhealthy",
            error="API key missing",
        )
    client = AsyncOpenAI(api_key=api_key)
    start = perf_counter()
    try:
        models = await client.models.list()
    except (
        ConnectionError,
        TimeoutError,
        RuntimeError,
        httpx.HTTPError,
        OpenAIError,
    ) as exc:
        return HealthCheckResult(
            service="openai",
            status="unhealthy",
            error=str(exc),
        )
    latency = (perf_counter() - start) * 1000
    return HealthCheckResult(
        service="openai",
        status="healthy",
        latency_ms=latency,
        details={"model_count": len(models.data), "model": config.openai.model},
    )


async def _check_firecrawl(config: Settings) -> HealthCheckResult:
    """Perform health check for Firecrawl API service."""

    if config.crawl_provider != CrawlProvider.FIRECRAWL:
        return HealthCheckResult(
            service="firecrawl",
            status="skipped",
            details={"reason": "provider_disabled"},
        )
    api_key = config.firecrawl.api_key or config.firecrawl.api_key
    if not api_key:
        return HealthCheckResult(
            service="firecrawl",
            status="unhealthy",
            error="API key missing",
        )
    headers = {"Authorization": f"Bearer {api_key}"}
    start = perf_counter()
    async with httpx.AsyncClient(timeout=config.firecrawl.timeout) as client:
        try:
            response = await client.get(
                f"{config.firecrawl.api_url}/health", headers=headers
            )
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - unexpected errors reported
            return HealthCheckResult(
                service="firecrawl",
                status="unhealthy",
                error=str(exc),
            )
    latency = (perf_counter() - start) * 1000
    return HealthCheckResult(
        service="firecrawl",
        status="healthy",
        latency_ms=latency,
        details={"status_code": response.status_code, "url": config.firecrawl.api_url},
    )


async def perform_health_checks(config: Settings) -> list[HealthCheckResult]:
    """Run all configured health checks concurrently."""

    results = await asyncio.gather(
        _check_qdrant(config),
        _check_redis(config),
        _check_openai(config),
        _check_firecrawl(config),
    )
    return list(results)


def summarize_results(results: list[HealthCheckResult]) -> dict[str, Any]:
    """Return a summary structure for CLI and API consumers."""

    summary = {
        "overall_status": "healthy",
        "services": {},
    }
    for result in results:
        summary["services"][result.service] = {
            "status": result.status,
            "connected": result.connected,
            "latency_ms": result.latency_ms,
            "error": result.error,
            "details": result.details,
        }
        if result.status == "unhealthy":
            summary["overall_status"] = "unhealthy"
        elif result.status == "skipped" and summary["overall_status"] != "unhealthy":
            summary["overall_status"] = "degraded"
    return summary
