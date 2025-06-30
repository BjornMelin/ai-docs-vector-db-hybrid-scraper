"""Tests for modern configuration system.

Tests the new Pydantic Settings-based configuration system that replaces
the complex 18-file configuration with a clean, modern approach.
"""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config.settings import (
    ApplicationMode,
    ChunkingStrategy,
    CrawlProvider,
    EmbeddingProvider,
    Environment,
    SearchStrategy,
    Settings as Config,
    create_enterprise_config,
    create_simple_config,
    get_settings as get_config,
    reset_settings as reset_config,
    set_settings as set_config,
)


class TestModernConfig:
    """Test suite for modern configuration system."""

    def setup_method(self):
        """Set up test environment."""
        reset_config()

        # Clear any environment variables that might interfere
        env_vars_to_clear = [
            "AI_DOCS__MODE",
            "AI_DOCS__ENVIRONMENT",
            "AI_DOCS__OPENAI_API_KEY",
            "AI_DOCS__FIRECRAWL_API_KEY",
            "AI_DOCS__EMBEDDING_PROVIDER",
            "AI_DOCS__CRAWL_PROVIDER",
        ]

        for var in env_vars_to_clear:
            os.environ.pop(var, None)

    def teardown_method(self):
        """Clean up after test."""
        reset_config()


class TestConfigDefaults:
    """Test default configuration values."""

    def test_default_configuration(self):
        """Test that default configuration values are set correctly."""
        config = Config()

        # Application defaults
        assert config.mode == ApplicationMode.SIMPLE
        assert config.environment == Environment.DEVELOPMENT
        assert config.debug is False
        assert config.log_level.value == "INFO"

        # Provider defaults
        assert config.embedding_provider == EmbeddingProvider.FASTEMBED
        assert config.crawl_provider == CrawlProvider.CRAWL4AI

        # Service URLs
        assert config.qdrant_url == "http://localhost:6333"
        assert config.redis_url == "redis://localhost:6379"

        # API keys should be None by default
        assert config.openai_api_key is None
        assert config.firecrawl_api_key is None
        assert config.qdrant_api_key is None

    def test_nested_config_defaults(self):
        """Test that nested configuration sections have correct defaults."""
        config = Config()

        # Performance defaults
        assert config.performance.max_concurrent_crawls == 10
        assert config.performance.max_concurrent_embeddings == 32
        assert config.performance.request_timeout == 30.0

        # Cache defaults
        assert config.cache.enable_caching is True
        assert config.cache.enable_local_cache is True
        assert config.cache.ttl_embeddings == 86400  # 24 hours
        assert config.cache.ttl_crawl == 3600  # 1 hour

        # Security defaults
        assert config.security.max_query_length == 1000
        assert config.security.rate_limit_requests_per_minute == 60
        assert config.security.require_api_keys is True

        # Chunking defaults
        assert config.chunking.strategy == ChunkingStrategy.ENHANCED
        assert config.chunking.max_chunk_size == 1600
        assert config.chunking.overlap == 200


class TestEnvironmentVariableLoading:
    """Test configuration loading from environment variables."""

    def test_basic_environment_loading(self):
        """Test loading basic configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "AI_DOCS__MODE": "enterprise",
                "AI_DOCS__ENVIRONMENT": "production",
                "AI_DOCS__DEBUG": "true",
                "AI_DOCS__LOG_LEVEL": "WARNING",
            },
        ):
            config = Config()

            assert config.mode == ApplicationMode.ENTERPRISE
            assert config.environment == Environment.PRODUCTION
            assert config.debug is True
            assert config.log_level.value == "WARNING"

    def test_nested_environment_loading(self):
        """Test loading nested configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "AI_DOCS__PERFORMANCE__MAX_CONCURRENT_CRAWLS": "25",
                "AI_DOCS__PERFORMANCE__REQUEST_TIMEOUT": "60.0",
                "AI_DOCS__CACHE__TTL_EMBEDDINGS": "172800",  # 48 hours
                "AI_DOCS__SECURITY__MAX_QUERY_LENGTH": "2000",
                "AI_DOCS__CHUNKING__MAX_CHUNK_SIZE": "2000",
            },
        ):
            config = Config()

            assert config.performance.max_concurrent_crawls == 25
            assert config.performance.request_timeout == 60.0
            assert config.cache.ttl_embeddings == 172800
            assert config.security.max_query_length == 2000
            assert config.chunking.max_chunk_size == 2000

    def test_api_key_loading(self):
        """Test loading API keys from environment variables."""
        with patch.dict(
            os.environ,
            {
                "AI_DOCS__OPENAI_API_KEY": "sk-test-key-123",
                "AI_DOCS__FIRECRAWL_API_KEY": "fc-test-key-456",
                "AI_DOCS__QDRANT_API_KEY": "qdrant-test-key-789",
            },
        ):
            config = Config()

            assert config.openai_api_key == "sk-test-key-123"
            assert config.firecrawl_api_key == "fc-test-key-456"
            assert config.qdrant_api_key == "qdrant-test-key-789"

    def test_provider_selection_loading(self):
        """Test loading provider selection from environment variables."""
        with patch.dict(
            os.environ,
            {
                "AI_DOCS__EMBEDDING_PROVIDER": "openai",
                "AI_DOCS__CRAWL_PROVIDER": "firecrawl",
            },
        ):
            config = Config()

            assert config.embedding_provider == EmbeddingProvider.OPENAI
            assert config.crawl_provider == CrawlProvider.FIRECRAWL


class TestConfigValidation:
    """Test configuration validation rules."""

    def test_openai_api_key_validation(self):
        """Test OpenAI API key validation."""
        # Valid key
        config = Config(openai_api_key="sk-valid-key")
        assert config.openai_api_key == "sk-valid-key"

        # Invalid key should raise error
        with pytest.raises(
            ValidationError, match="OpenAI API key must start with 'sk-'"
        ):
            Config(openai_api_key="invalid-key")

    def test_firecrawl_api_key_validation(self):
        """Test Firecrawl API key validation."""
        # Valid key
        config = Config(firecrawl_api_key="fc-valid-key")
        assert config.firecrawl_api_key == "fc-valid-key"

        # Invalid key should raise error
        with pytest.raises(
            ValidationError, match="Firecrawl API key must start with 'fc-'"
        ):
            Config(firecrawl_api_key="invalid-key")

    def test_provider_validation(self):
        """Test provider validation and API key requirements."""
        # OpenAI provider requires API key
        with pytest.raises(ValidationError, match="OpenAI API key is required"):
            Config(embedding_provider=EmbeddingProvider.OPENAI)

        # Firecrawl provider requires API key
        with pytest.raises(ValidationError, match="Firecrawl API key is required"):
            Config(crawl_provider=CrawlProvider.FIRECRAWL)

        # Valid configuration with API keys
        config = Config(
            embedding_provider=EmbeddingProvider.OPENAI,
            openai_api_key="sk-test-key",
            crawl_provider=CrawlProvider.FIRECRAWL,
            firecrawl_api_key="fc-test-key",
        )
        assert config.embedding_provider == EmbeddingProvider.OPENAI
        assert config.crawl_provider == CrawlProvider.FIRECRAWL

    def test_numeric_constraints(self):
        """Test numeric field constraints."""
        # Valid values
        config = Config()
        config.performance.max_concurrent_crawls = 25
        config.cache.ttl_embeddings = 86400

        # Invalid values should raise errors
        with pytest.raises(ValidationError):
            Config(performance={"max_concurrent_crawls": 0})  # Must be > 0

        with pytest.raises(ValidationError):
            Config(performance={"max_concurrent_crawls": 100})  # Must be <= 50


class TestApplicationModes:
    """Test application mode behavior."""

    def test_simple_mode_optimizations(self):
        """Test that simple mode applies performance optimizations."""
        config = Config(
            mode=ApplicationMode.SIMPLE,
            performance={"max_concurrent_crawls": 50},  # Try to set high value
        )

        # Should be capped at 10 for simple mode
        assert config.performance.max_concurrent_crawls == 10
        assert config.reranking.enabled is False  # Disabled in simple mode

    def test_enterprise_mode_features(self):
        """Test that enterprise mode allows full feature set."""
        config = Config(
            mode=ApplicationMode.ENTERPRISE, performance={"max_concurrent_crawls": 50}
        )

        # Should allow higher values in enterprise mode
        assert config.performance.max_concurrent_crawls == 50
        # Reranking can be enabled in enterprise mode
        assert config.reranking.enabled is False  # Default is False, but can be enabled

    def test_mode_specific_strategies(self):
        """Test mode-specific strategy selection."""
        simple_config = Config(mode=ApplicationMode.SIMPLE)
        enterprise_config = Config(mode=ApplicationMode.ENTERPRISE)

        # Simple mode uses basic strategies
        assert simple_config.get_effective_chunking_strategy() == ChunkingStrategy.BASIC
        assert simple_config.get_effective_search_strategy() == SearchStrategy.DENSE

        # Enterprise mode uses advanced strategies
        assert (
            enterprise_config.get_effective_chunking_strategy()
            == ChunkingStrategy.ENHANCED
        )
        assert (
            enterprise_config.get_effective_search_strategy() == SearchStrategy.HYBRID
        )


class TestConfigFactories:
    """Test configuration factory functions."""

    def test_create_simple_config(self):
        """Test simple configuration factory."""
        config = create_simple_config()

        assert config.mode == ApplicationMode.SIMPLE
        assert config.performance.max_concurrent_crawls <= 10
        assert config.reranking.enabled is False

    def test_create_enterprise_config(self):
        """Test enterprise configuration factory."""
        config = create_enterprise_config()

        assert config.mode == ApplicationMode.ENTERPRISE
        assert config.performance.max_concurrent_crawls <= 50

    def test_global_config_management(self):
        """Test global configuration instance management."""
        # Initially no config
        reset_config()

        # Get default config
        config1 = get_config()
        config2 = get_config()

        # Should be the same instance
        assert config1 is config2

        # Set new config
        new_config = Config(debug=True)
        set_config(new_config)

        config3 = get_config()
        assert config3 is new_config
        assert config3.debug is True

        # Reset config
        reset_config()
        config4 = get_config()
        assert config4 is not new_config


class TestConfigSerialization:
    """Test configuration serialization and deserialization."""

    def test_model_dump(self):
        """Test configuration serialization."""
        config = Config(
            mode=ApplicationMode.ENTERPRISE, debug=True, openai_api_key="sk-test-key"
        )

        data = config.model_dump()

        assert isinstance(data, dict)
        assert data["mode"] == "enterprise"
        assert data["debug"] is True
        assert data["openai_api_key"] == "sk-test-key"
        assert "performance" in data
        assert "cache" in data

    def test_model_dump_exclude_none(self):
        """Test configuration serialization excluding None values."""
        config = Config()

        data = config.model_dump(exclude_none=True)

        # None values should be excluded
        assert "openai_api_key" not in data
        assert "firecrawl_api_key" not in data
        assert "qdrant_api_key" not in data

    def test_json_serialization(self):
        """Test JSON serialization."""
        config = Config(mode=ApplicationMode.SIMPLE)

        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        assert "simple" in json_str


class TestConfigUtilityMethods:
    """Test configuration utility methods."""

    def test_is_enterprise_mode(self):
        """Test enterprise mode detection."""
        simple_config = Config(mode=ApplicationMode.SIMPLE)
        enterprise_config = Config(mode=ApplicationMode.ENTERPRISE)

        assert simple_config.is_enterprise_mode() is False
        assert enterprise_config.is_enterprise_mode() is True

    def test_is_development(self):
        """Test development environment detection."""
        dev_config = Config(environment=Environment.DEVELOPMENT)
        prod_config = Config(environment=Environment.PRODUCTION)

        assert dev_config.is_development() is True
        assert prod_config.is_development() is False

    def test_effective_strategies(self):
        """Test effective strategy methods."""
        simple_config = Config(mode=ApplicationMode.SIMPLE)
        enterprise_config = Config(mode=ApplicationMode.ENTERPRISE)

        # Simple mode
        assert simple_config.get_effective_chunking_strategy() == ChunkingStrategy.BASIC
        assert simple_config.get_effective_search_strategy() == SearchStrategy.DENSE

        # Enterprise mode
        assert (
            enterprise_config.get_effective_chunking_strategy()
            == ChunkingStrategy.ENHANCED
        )
        assert (
            enterprise_config.get_effective_search_strategy() == SearchStrategy.HYBRID
        )


class TestConfigSyncBehavior:
    """Test configuration synchronization behavior."""

    def test_service_url_sync(self):
        """Test that service URLs are synced to nested configs."""
        config = Config(
            qdrant_url="http://custom-qdrant:6333",
            redis_url="redis://custom-redis:6379",
            qdrant_api_key="test-key",
        )

        # URLs should be synced to nested configs
        assert config.qdrant.url == "http://custom-qdrant:6333"
        assert config.cache.redis_url == "redis://custom-redis:6379"
        assert config.qdrant.api_key == "test-key"

    def test_api_key_sync(self):
        """Test that API keys are synced to nested configs."""
        config = Config(
            openai_api_key="sk-test-key",
            firecrawl_api_key="fc-test-key",
            embedding_provider=EmbeddingProvider.OPENAI,
            crawl_provider=CrawlProvider.FIRECRAWL,
        )

        # API keys should be synced to nested configs
        assert config.openai.api_key == "sk-test-key"
        assert config.firecrawl.api_key == "fc-test-key"


@pytest.mark.parametrize(
    ("mode", "expected_crawls"),
    [
        (ApplicationMode.SIMPLE, 10),
        (ApplicationMode.ENTERPRISE, 25),
    ],
)
def test_mode_performance_limits(mode, expected_crawls):
    """Test performance limits based on application mode."""
    config = Config(mode=mode, performance={"max_concurrent_crawls": 25})

    if mode == ApplicationMode.SIMPLE:
        # Simple mode caps at 10
        assert config.performance.max_concurrent_crawls == 10
    else:
        # Enterprise mode allows higher values
        assert config.performance.max_concurrent_crawls == expected_crawls


@pytest.mark.parametrize(
    ("provider", "api_key", "should_pass"),
    [
        (EmbeddingProvider.FASTEMBED, None, True),  # FastEmbed doesn't need key
        (EmbeddingProvider.OPENAI, None, False),  # OpenAI needs key
        (EmbeddingProvider.OPENAI, "sk-test", True),  # OpenAI with valid key
    ],
)
def test_provider_api_key_requirements(provider, api_key, should_pass):
    """Test API key requirements for different providers."""
    if should_pass:
        config = Config(embedding_provider=provider, openai_api_key=api_key)
        assert config.embedding_provider == provider
    else:
        with pytest.raises(ValidationError):
            Config(embedding_provider=provider, openai_api_key=api_key)
