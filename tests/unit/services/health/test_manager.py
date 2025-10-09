"""Tests for the centralized health check manager."""

from __future__ import annotations

import asyncio

import pytest

from src.config.models import CrawlProvider, EmbeddingProvider
from src.services.health.manager import (
    HAS_QDRANT_CLIENT,
    HAS_REDIS,
    HealthCheck,
    HealthCheckConfig,
    HealthCheckManager,
    HealthCheckResult,
    HealthStatus,
    build_health_manager,
)


@pytest.fixture()
def configured_settings(config_factory):
    """Provide settings with all optional providers enabled."""

    return config_factory(
        qdrant={"url": "http://localhost:6333", "api_key": "test", "timeout": 5.0},
        cache={"enable_redis_cache": True, "redis_url": "redis://localhost:6379/0"},
        embedding_provider=EmbeddingProvider.OPENAI,
        crawl_provider=CrawlProvider.FIRECRAWL,
        openai={"api_key": "sk-test"},
        firecrawl={"api_key": "fc-test"},
    )


class _StubHealthCheck(HealthCheck):
    """Deterministic health check used for manager tests."""

    def __init__(self, name: str, result: HealthCheckResult):
        """Create a stubbed health check."""

        super().__init__(name)
        self._result = result

    async def check(self) -> HealthCheckResult:
        """Return the pre-configured result."""

        await asyncio.sleep(0)
        return self._result


@pytest.mark.asyncio()
async def test_health_manager_aggregates_status() -> None:
    """Health manager should aggregate results and compute overall status."""

    config = HealthCheckConfig()
    manager = HealthCheckManager(config)
    manager.add_health_check(
        _StubHealthCheck(
            "healthy",
            HealthCheckResult(
                name="healthy",
                status=HealthStatus.HEALTHY,
                message="ok",
                duration_ms=1.0,
            ),
        )
    )
    manager.add_health_check(
        _StubHealthCheck(
            "unhealthy",
            HealthCheckResult(
                name="unhealthy",
                status=HealthStatus.UNHEALTHY,
                message="failed",
                duration_ms=1.0,
            ),
        )
    )

    await manager.check_all()
    summary = manager.get_health_summary()

    assert summary["overall_status"] == HealthStatus.UNHEALTHY.value
    assert summary["healthy_count"] == 1
    assert summary["total_count"] == 2


def test_build_health_manager_includes_expected_checks(configured_settings) -> None:
    """Building the manager wires up configured dependency checks."""

    manager = build_health_manager(configured_settings)
    check_names = {check.name for check in manager._health_checks}

    expected_checks = {"openai", "firecrawl"}
    if HAS_QDRANT_CLIENT:
        expected_checks.add("qdrant")
    if HAS_REDIS:
        expected_checks.add("redis")

    assert expected_checks.issubset(check_names)


def test_build_health_manager_skips_openai_when_disabled(config_factory) -> None:
    """OpenAI checks should be omitted when another embedding provider is used."""

    settings = config_factory(
        embedding_provider=EmbeddingProvider.FASTEMBED,
        crawl_provider=CrawlProvider.CRAWL4AI,
        cache={"enable_redis_cache": False},
        qdrant={"url": "http://localhost:6333", "timeout": 5.0},
    )
    manager = build_health_manager(settings)
    check_names = {check.name for check in manager._health_checks}

    assert "openai" not in check_names
    assert "firecrawl" not in check_names
