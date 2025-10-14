"""Tests for the centralized health check manager."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fakeredis import aioredis as fakeredis_aioredis
from pytest_mock import MockerFixture

from src.config.models import CrawlProvider, EmbeddingProvider
from src.services.observability.health_manager import (
    ApplicationMetadataHealthCheck,
    DragonflyHealthCheck,
    HealthCheck,
    HealthCheckConfig,
    HealthCheckManager,
    HealthCheckResult,
    HealthStatus,
    RAGConfigurationHealthCheck,
    build_health_manager,
)


@pytest.fixture()
def configured_settings(config_factory):
    """Provide settings with all optional providers enabled."""
    return config_factory(
        qdrant={"url": "http://localhost:6333", "api_key": "test", "timeout": 5.0},
        cache={
            "enable_dragonfly_cache": True,
            "dragonfly_url": "redis://localhost:6379/0",
        },
        embedding_provider=EmbeddingProvider.OPENAI,
        crawl_provider=CrawlProvider.FIRECRAWL,
        openai={"api_key": "sk-test"},
        browser={
            "firecrawl": {
                "api_key": "fc-test",
                "api_url": "https://api.firecrawl.dev",
            }
        },
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


def test_build_health_manager_includes_expected_checks(
    configured_settings, mocker: MockerFixture
) -> None:
    """Building the manager wires up configured dependency checks."""
    mocker.patch(
        "src.services.observability.health_manager.AsyncQdrantClient", autospec=True
    )
    mocker.patch(
        "src.services.observability.health_manager.QdrantHealthCheck",
        return_value=SimpleNamespace(name="qdrant"),
    )
    mocker.patch(
        "src.services.observability.health_manager.DragonflyHealthCheck",
        return_value=SimpleNamespace(name="dragonfly"),
    )
    mocker.patch(
        "src.services.observability.health_manager.OpenAIHealthCheck",
        return_value=SimpleNamespace(name="openai"),
    )
    mocker.patch(
        "src.services.observability.health_manager.FirecrawlHealthCheck",
        return_value=SimpleNamespace(name="firecrawl"),
    )
    mocker.patch("src.services.observability.health_manager.AsyncOpenAI", autospec=True)

    manager = build_health_manager(configured_settings)
    check_names = set(manager.list_checks())

    expected_checks = {
        "openai",
        "firecrawl",
        "qdrant",
        "dragonfly",
        "application_metadata",
        "rag_configuration",
    }

    assert expected_checks.issubset(check_names)


def test_build_health_manager_skips_openai_when_disabled(config_factory) -> None:
    """OpenAI checks should be omitted when another embedding provider is used."""
    settings = config_factory(
        embedding_provider=EmbeddingProvider.FASTEMBED,
        crawl_provider=CrawlProvider.CRAWL4AI,
        cache={"enable_dragonfly_cache": False},
        qdrant={"url": "http://localhost:6333", "timeout": 5.0},
        browser={"firecrawl": {}},
    )
    manager = build_health_manager(settings)
    check_names = set(manager.list_checks())

    assert "openai" not in check_names
    assert "firecrawl" not in check_names


@pytest.mark.asyncio()
async def test_rag_health_check_skips_when_disabled(config_factory) -> None:
    """RAG configuration check should skip when the feature is disabled."""
    settings = config_factory(rag={"enable_rag": False})
    check = RAGConfigurationHealthCheck(settings)

    result = await check.check()

    assert result.status is HealthStatus.SKIPPED
    assert not result.metadata["enabled"]


@pytest.mark.asyncio()
async def test_rag_health_check_flags_missing_model(config_factory) -> None:
    """RAG configuration check should report missing required fields."""
    settings = config_factory(rag={"enable_rag": True, "model": ""})
    check = RAGConfigurationHealthCheck(settings)

    result = await check.check()

    assert result.status is HealthStatus.UNHEALTHY
    assert "Missing required RAG configuration fields" in result.message


@pytest.mark.asyncio()
async def test_rag_health_check_reports_healthy(config_factory) -> None:
    """RAG configuration check should report healthy when configured."""
    settings = config_factory(rag={"enable_rag": True, "model": "gpt-4o"})
    check = RAGConfigurationHealthCheck(settings)

    result = await check.check()

    assert result.status is HealthStatus.HEALTHY
    assert result.metadata["enabled"]


@pytest.mark.asyncio()
async def test_application_metadata_health_check_degraded_without_name(
    config_factory,
) -> None:
    """Application metadata check should degrade when required metadata is empty."""
    settings = config_factory(app_name="")
    check = ApplicationMetadataHealthCheck(settings)

    result = await check.check()

    assert result.status is HealthStatus.DEGRADED
    assert "app_name" in result.message


@pytest.mark.asyncio()
async def test_application_metadata_health_check_reports_metadata(
    config_factory,
) -> None:
    """Application metadata check should expose configured metadata."""
    settings = config_factory(app_name="Docs App", version="1.2.3")
    check = ApplicationMetadataHealthCheck(settings)

    result = await check.check()

    assert result.status is HealthStatus.HEALTHY
    assert result.metadata["app_name"] == "Docs App"
    assert result.metadata["version"] == "1.2.3"


@pytest.mark.asyncio()
async def test_dragonfly_health_check_reports_metadata(
    mocker: MockerFixture,
) -> None:
    """Dragonfly health check should report version and client metrics."""
    fake_client = fakeredis_aioredis.FakeRedis(decode_responses=True)
    mocker.patch(
        "src.services.observability.health_manager.redis.from_url",
        return_value=fake_client,
    )
    ping_future: asyncio.Future[bool] = asyncio.Future()
    ping_future.set_result(True)
    mocker.patch.object(fake_client, "ping", return_value=ping_future)
    info_future: asyncio.Future[dict[str, int | str]] = asyncio.Future()
    info_future.set_result(
        {
            "redis_version": "7.2",
            "connected_clients": 5,
            "used_memory_human": "2M",
        }
    )
    mocker.patch.object(fake_client, "info", return_value=info_future)

    check = DragonflyHealthCheck("redis://fakeredis")
    result = await check.check()

    assert result.status is HealthStatus.HEALTHY
    assert result.metadata["engine_version"] == "7.2"
    await fake_client.aclose()


@pytest.mark.asyncio()
async def test_dragonfly_health_check_handles_ping_failure(
    mocker: MockerFixture,
) -> None:
    """Dragonfly health check should mark the service unhealthy when ping fails."""
    fake_client = fakeredis_aioredis.FakeRedis(decode_responses=True)
    mocker.patch(
        "src.services.observability.health_manager.redis.from_url",
        return_value=fake_client,
    )
    ping_future: asyncio.Future[bool] = asyncio.Future()
    ping_future.set_result(False)
    mocker.patch.object(fake_client, "ping", return_value=ping_future)

    check = DragonflyHealthCheck("redis://fakeredis")
    result = await check.check()

    assert result.status is HealthStatus.UNHEALTHY
    await fake_client.aclose()


@pytest.mark.asyncio()
async def test_health_manager_check_single_returns_result() -> None:
    """Health manager should run a named check and track the latest result."""
    config = HealthCheckConfig()
    expected = HealthCheckResult(
        name="single",
        status=HealthStatus.HEALTHY,
        message="ok",
        duration_ms=1.0,
    )
    manager = HealthCheckManager(config)
    manager.add_health_check(_StubHealthCheck("single", expected))

    result = await manager.check_single("single")

    assert result == expected
    assert manager.get_last_results()["single"] == expected


@pytest.mark.asyncio()
async def test_health_manager_check_all_handles_exceptions() -> None:
    """Manager should coerce unexpected exceptions into unhealthy results."""

    class _FailingHealthCheck(HealthCheck):
        """Failing health check for exception handling verification."""

        def __init__(self, name: str) -> None:
            """Initialize the failing check with a name."""
            super().__init__(name)

        async def check(self) -> HealthCheckResult:
            """Raise an error to simulate probe failure."""
            raise RuntimeError("boom")

    config = HealthCheckConfig()
    manager = HealthCheckManager(config)
    manager.add_health_check(_FailingHealthCheck("failing"))

    results = await manager.check_all()

    failing_result = results["failing"]
    assert failing_result.status is HealthStatus.UNHEALTHY
    assert "boom" in failing_result.message
