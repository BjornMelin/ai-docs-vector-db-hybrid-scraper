"""Regression tests for the consolidated Pydantic settings models."""

from __future__ import annotations

from typing import cast

import pytest

from src.config.settings import (
    ApplicationMode,
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingModel,
    Environment,
    SearchStrategy,
    Settings,
)
from tests._helpers.config import make_test_settings


def test_settings_effective_search_strategy(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Enterprise mode should enable hybrid search by default."""
    simple = make_test_settings(tmp_path_factory.mktemp("simple"))
    assert simple.get_effective_search_strategy() is SearchStrategy.DENSE

    enterprise = make_test_settings(
        tmp_path_factory.mktemp("enterprise"), mode=ApplicationMode.ENTERPRISE
    )
    assert enterprise.get_effective_search_strategy() is SearchStrategy.HYBRID


def test_settings_sync_service_endpoints(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Top-level endpoints should propagate to nested config sections."""
    redis_url = "redis://localhost:6380/1"
    qdrant_url = "http://qdrant:6333"

    settings = cast(
        Settings,
        make_test_settings(
            tmp_path_factory.mktemp("sync"),
            redis_url=redis_url,
            qdrant_url=qdrant_url,
        ),
    )
    data = settings.model_dump()
    assert data["cache"]["redis_url"] == redis_url
    assert data["task_queue"]["redis_url"] == redis_url
    assert data["qdrant"]["url"] == qdrant_url


def test_embedding_config_allows_hybrid_strategy() -> None:
    """Embedding configuration should support hybrid vector strategies."""
    config = EmbeddingConfig(
        provider=EmbeddingProvider.FASTEMBED,
        dense_model=EmbeddingModel.TEXT_EMBEDDING_3_SMALL,
        search_strategy=SearchStrategy.HYBRID,
    )
    assert config.search_strategy is SearchStrategy.HYBRID
    assert config.enable_quantization is True


def test_settings_require_api_key_when_using_openai(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Production deployments must provide an OpenAI API key."""
    with pytest.raises(ValueError):
        make_test_settings(
            tmp_path_factory.mktemp("prod"),
            environment=Environment.PRODUCTION,
            embedding_provider=EmbeddingProvider.OPENAI,
            openai_api_key=None,
        )


def test_performance_profile_scales_with_mode(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Enterprise profiles expose larger concurrency budgets."""
    simple = cast(Settings, make_test_settings(tmp_path_factory.mktemp("perf-simple")))
    enterprise = cast(
        Settings,
        make_test_settings(
            tmp_path_factory.mktemp("perf-enterprise"),
            mode=ApplicationMode.ENTERPRISE,
        ),
    )
    simple_data = simple.model_dump()
    enterprise_data = enterprise.model_dump()

    assert (
        simple_data["performance"]["max_concurrent_requests"]
        <= enterprise_data["performance"]["max_concurrent_requests"]
    )
    assert (
        simple_data["performance"]["max_concurrent_embeddings"]
        <= enterprise_data["performance"]["max_concurrent_embeddings"]
    )
    assert enterprise_data["performance"]["max_concurrent_embeddings"] <= 100


def test_hyde_and_security_defaults(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """HyDE and security defaults remain enabled for final surfaces."""
    settings = cast(Settings, make_test_settings(tmp_path_factory.mktemp("defaults")))
    config_data = settings.model_dump()

    assert config_data["hyde"]["enable_hyde"] is True
    assert config_data["security"]["enable_rate_limiting"] is True
    assert config_data["security"]["rate_limit_requests"] == 100
