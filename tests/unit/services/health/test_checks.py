"""Tests for service health checks."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import Config
from src.config.models import CrawlProvider, EmbeddingProvider
from src.services.health import HealthCheckResult, checks


@pytest.fixture()
def config() -> Config:
    """Return a configuration object with defaults."""

    cfg = Config()
    cfg.embedding_provider = EmbeddingProvider.OPENAI
    cfg.crawl_provider = CrawlProvider.FIRECRAWL
    cfg.openai.api_key = "sk-test"
    cfg.firecrawl.api_key = "fc-test"
    return cfg


@pytest.mark.asyncio()
async def test_perform_health_checks_aggregates(monkeypatch, config):
    """perform_health_checks combines individual results."""

    async def fake(service):
        await asyncio.sleep(0)
        return HealthCheckResult(service=service, status="healthy")

    monkeypatch.setattr(checks, "_check_qdrant", lambda cfg: fake("qdrant"))
    monkeypatch.setattr(checks, "_check_redis", lambda cfg: fake("redis"))
    monkeypatch.setattr(checks, "_check_openai", lambda cfg: fake("openai"))
    monkeypatch.setattr(checks, "_check_firecrawl", lambda cfg: fake("firecrawl"))

    results = await checks.perform_health_checks(config)

    assert {result.service for result in results} == {
        "qdrant",
        "redis",
        "openai",
        "firecrawl",
    }
    assert all(result.connected for result in results)


@pytest.mark.asyncio()
async def test_check_openai_skips_when_disabled(monkeypatch, config):
    """OpenAI check is skipped when provider is not OpenAI."""

    config.embedding_provider = EmbeddingProvider.FASTEMBED
    result = await checks._check_openai(config)

    assert result.status == "skipped"
    assert not result.connected


@pytest.mark.asyncio()
async def test_check_firecrawl_reports_error(monkeypatch, config):
    """Firecrawl health check surfaces HTTP errors."""

    mock_client = MagicMock()
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_client
    mock_ctx.__aexit__.return_value = False
    mock_client.get.side_effect = RuntimeError("boom")
    monkeypatch.setattr(checks.httpx, "AsyncClient", lambda timeout: mock_ctx)

    result = await checks._check_firecrawl(config)

    assert result.status == "unhealthy"
    assert result.error is not None
    assert "boom" in result.error
