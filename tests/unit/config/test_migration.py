"""Tests for configuration settings system.

Tests the unified Pydantic Settings system that replaced
the legacy 18-file configuration system.
"""

import pytest

from src.config.settings import (
    ApplicationMode,
    CrawlProvider,
    EmbeddingProvider,
    Environment,
    Settings,
)


class TestSettings:
    """Test suite for configuration settings."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()

        assert settings.mode == ApplicationMode.SIMPLE
        assert settings.environment == Environment.DEVELOPMENT
        assert settings.embedding_provider == EmbeddingProvider.FASTEMBED
        assert settings.crawl_provider == CrawlProvider.CRAWL4AI
        assert settings.debug is False

    def test_simple_mode_configuration(self):
        """Test simple mode configuration."""
        settings = Settings(mode=ApplicationMode.SIMPLE)

        assert settings.is_enterprise_mode() is False
        assert settings.performance.max_concurrent_crawls <= 10
        assert settings.reranking.enabled is False
        assert settings.observability.enabled is False

    def test_enterprise_mode_configuration(self):
        """Test enterprise mode configuration."""
        settings = Settings(mode=ApplicationMode.ENTERPRISE)

        assert settings.is_enterprise_mode() is True
        assert settings.performance.max_concurrent_crawls <= 50

    def test_environment_detection(self):
        """Test environment detection methods."""
        dev_settings = Settings(environment=Environment.DEVELOPMENT)
        prod_settings = Settings(environment=Environment.PRODUCTION)

        assert dev_settings.is_development() is True
        assert dev_settings.is_production() is False
        assert prod_settings.is_development() is False
        assert prod_settings.is_production() is True

    def test_effective_strategies(self):
        """Test effective strategy selection based on mode."""
        simple_settings = Settings(mode=ApplicationMode.SIMPLE)
        enterprise_settings = Settings(mode=ApplicationMode.ENTERPRISE)

        # Simple mode should use basic strategies
        assert simple_settings.get_effective_chunking_strategy().value == "basic"
        assert simple_settings.get_effective_search_strategy().value == "dense"

        # Enterprise mode should use enhanced strategies
        assert enterprise_settings.get_effective_search_strategy().value == "hybrid"

    def test_api_key_validation(self):
        """Test API key validation."""
        # Valid OpenAI key
        settings = Settings(openai_api_key="sk-test123")
        assert settings.openai_api_key == "sk-test123"

        # Invalid OpenAI key should raise error
        with pytest.raises(ValueError, match="OpenAI API key must start with 'sk-'"):
            Settings(openai_api_key="invalid-key")

    def test_provider_key_requirements(self):
        """Test that required API keys are validated for selected providers."""
        # Should raise error when OpenAI provider selected without key
        with pytest.raises(ValueError, match="OpenAI API key required"):
            Settings(embedding_provider=EmbeddingProvider.OPENAI)

        # Should raise error when Firecrawl provider selected without key
        with pytest.raises(ValueError, match="Firecrawl API key required"):
            Settings(crawl_provider=CrawlProvider.FIRECRAWL)

    def test_api_key_sync(self):
        """Test that top-level API keys sync with nested configs."""
        settings = Settings(
            openai_api_key="sk-test123",
            firecrawl_api_key="fc-test456",
            qdrant_api_key="qdrant-test789",
        )

        assert settings.openai.api_key == "sk-test123"
        assert settings.firecrawl.api_key == "fc-test456"
        assert settings.qdrant.api_key == "qdrant-test789"

    def test_service_url_sync(self):
        """Test that service URLs sync with nested configs."""
        settings = Settings(
            qdrant_url="http://test-qdrant:6333", redis_url="redis://test-redis:6379"
        )

        assert settings.qdrant.url == "http://test-qdrant:6333"
        assert settings.cache.redis_url == "redis://test-redis:6379"
        assert settings.task_queue.redis_url == "redis://test-redis:6379"

    def test_directory_creation(self):
        """Test that required directories are created."""
        settings = Settings()

        # Directories should exist after settings creation
        assert settings.data_dir.exists()
        assert settings.cache_dir.exists()
        assert settings.logs_dir.exists()


class TestConfigurationSections:
    """Test configuration section defaults and validation."""

    def test_cache_config_defaults(self):
        """Test cache configuration defaults."""
        settings = Settings()

        assert settings.cache.enable_caching is True
        assert settings.cache.enable_local_cache is True
        assert settings.cache.enable_redis_cache is True
        assert settings.cache.ttl_embeddings == 86400
        assert settings.cache.local_max_size == 1000

    def test_performance_config_defaults(self):
        """Test performance configuration defaults."""
        settings = Settings()

        assert settings.performance.max_concurrent_requests == 10
        assert settings.performance.max_concurrent_crawls == 10
        assert settings.performance.request_timeout == 30.0
        assert settings.performance.max_retries == 3

    def test_security_config_defaults(self):
        """Test security configuration defaults."""
        settings = Settings()

        assert settings.security.require_api_keys is True
        assert settings.security.enable_rate_limiting is True
        assert settings.security.rate_limit_requests == 100
        assert settings.security.max_query_length == 1000

    def test_chunking_config_validation(self):
        """Test chunking configuration validation."""
        # Valid configuration
        settings = Settings()
        settings.chunking.chunk_size = 1000
        settings.chunking.chunk_overlap = 200
        settings.chunking.min_chunk_size = 100
        settings.chunking.max_chunk_size = 2000

        # This should validate successfully
        validated = settings.chunking.model_validate(settings.chunking.model_dump())
        assert validated.chunk_size == 1000

        # Invalid: overlap >= chunk_size should fail
        with pytest.raises(
            ValueError, match="chunk_overlap must be less than chunk_size"
        ):
            from src.config.settings import ChunkingConfig

            ChunkingConfig(chunk_size=1000, chunk_overlap=1000)


class TestGlobalConfiguration:
    """Test global configuration management."""

    def test_get_settings_singleton(self):
        """Test that get_settings returns the same instance."""
        from src.config.settings import get_settings, reset_settings

        # Reset to ensure clean state
        reset_settings()

        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same instance
        assert settings1 is settings2

    def test_set_and_reset_settings(self):
        """Test setting and resetting global settings."""
        from src.config.settings import get_settings, reset_settings, set_settings

        # Create custom settings
        custom_settings = Settings(debug=True, app_name="Test App")
        set_settings(custom_settings)

        # Should return the custom settings
        retrieved = get_settings()
        assert retrieved.debug is True
        assert retrieved.app_name == "Test App"

        # Reset should clear the global instance
        reset_settings()
        new_settings = get_settings()
        assert new_settings is not custom_settings

    def test_convenience_config_functions(self):
        """Test convenience functions for accessing config sections."""
        from src.config.settings import (
            get_cache_config,
            get_embedding_config,
            get_openai_config,
            get_performance_config,
            get_qdrant_config,
            get_security_config,
            reset_settings,
        )

        # Reset to ensure clean state
        reset_settings()

        # All should return the appropriate config sections
        qdrant_config = get_qdrant_config()
        embedding_config = get_embedding_config()
        cache_config = get_cache_config()
        performance_config = get_performance_config()
        openai_config = get_openai_config()
        security_config = get_security_config()

        assert qdrant_config.url == "http://localhost:6333"
        assert embedding_config.provider == EmbeddingProvider.FASTEMBED
        assert cache_config.enable_caching is True
        assert performance_config.max_concurrent_requests == 10
        assert openai_config.model == "text-embedding-3-small"
        assert security_config.require_api_keys is True
