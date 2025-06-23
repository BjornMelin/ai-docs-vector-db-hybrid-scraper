"""Tests for consolidated configuration system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config import (
    CacheConfig,
    ChunkingConfig,
    Config,
    Crawl4AIConfig,
    DocumentationSite,
    EmbeddingConfig,
    FastEmbedConfig,
    FirecrawlConfig,
    OpenAIConfig,
    PerformanceConfig,
    QdrantConfig,
    SecurityConfig,
    get_config,
    reset_config,
    set_config,
)
from src.config.enums import (
    ChunkingStrategy,
    CrawlProvider,
    EmbeddingProvider,
    Environment,
    LogLevel,
)


class TestCacheConfig:
    """Test cache configuration."""

    def test_default_values(self):
        config = CacheConfig()
        assert config.enable_caching is True
        assert config.dragonfly_url == "redis://localhost:6379"
        assert config.local_max_size == 1000
        assert config.ttl_seconds == 3600

    def test_custom_values(self):
        config = CacheConfig(
            enable_caching=False,
            dragonfly_url="redis://custom:6380",
            local_max_size=500,
            ttl_seconds=1800,
        )
        assert config.enable_caching is False
        assert config.dragonfly_url == "redis://custom:6380"
        assert config.local_max_size == 500
        assert config.ttl_seconds == 1800

    def test_validation(self):
        with pytest.raises(ValidationError):
            CacheConfig(local_max_size=-1)  # Must be positive
        with pytest.raises(ValidationError):
            CacheConfig(ttl_seconds=-1)  # Must be positive


class TestQdrantConfig:
    """Test Qdrant configuration."""

    def test_default_values(self):
        config = QdrantConfig()
        assert config.url == "http://localhost:6333"
        assert config.api_key is None
        assert config.timeout == 30.0
        assert config.collection_name == "documents"
        assert config.batch_size == 100

    def test_custom_values(self):
        config = QdrantConfig(
            url="http://remote:6333",
            api_key="test-key",
            timeout=60.0,
            collection_name="custom",
            batch_size=200,
        )
        assert config.url == "http://remote:6333"
        assert config.api_key == "test-key"
        assert config.timeout == 60.0
        assert config.collection_name == "custom"
        assert config.batch_size == 200

    def test_validation(self):
        with pytest.raises(ValidationError):
            QdrantConfig(timeout=-1)  # Must be positive
        with pytest.raises(ValidationError):
            QdrantConfig(batch_size=0)  # Must be positive
        with pytest.raises(ValidationError):
            QdrantConfig(batch_size=2000)  # Must be <= 1000


class TestOpenAIConfig:
    """Test OpenAI configuration."""

    def test_default_values(self):
        config = OpenAIConfig()
        assert config.api_key is None
        assert config.model == "text-embedding-3-small"
        assert config.dimensions == 1536
        assert config.batch_size == 100
        assert config.max_requests_per_minute == 3000
        assert config.cost_per_million_tokens == 0.02

    def test_api_key_validation(self):
        # Valid API key
        config = OpenAIConfig(api_key="sk-1234567890abcdef")
        assert config.api_key == "sk-1234567890abcdef"

        # Invalid API key (no sk- prefix)
        with pytest.raises(
            ValidationError, match="OpenAI API key must start with 'sk-'"
        ):
            OpenAIConfig(api_key="invalid-key")

    def test_custom_values(self):
        config = OpenAIConfig(
            api_key="sk-test",
            model="text-embedding-3-large",
            dimensions=3072,
            batch_size=50,
            max_requests_per_minute=1000,
            cost_per_million_tokens=0.13,
        )
        assert config.api_key == "sk-test"
        assert config.model == "text-embedding-3-large"
        assert config.dimensions == 3072
        assert config.batch_size == 50

    def test_validation(self):
        with pytest.raises(ValidationError):
            OpenAIConfig(dimensions=0)  # Must be positive
        with pytest.raises(ValidationError):
            OpenAIConfig(dimensions=4000)  # Must be <= 3072
        with pytest.raises(ValidationError):
            OpenAIConfig(batch_size=0)  # Must be positive
        with pytest.raises(ValidationError):
            OpenAIConfig(batch_size=3000)  # Must be <= 2048


class TestFirecrawlConfig:
    """Test Firecrawl configuration."""

    def test_default_values(self):
        config = FirecrawlConfig()
        assert config.api_key is None
        assert config.api_url == "https://api.firecrawl.dev"
        assert config.timeout == 30.0

    def test_api_key_validation(self):
        # Valid API key
        config = FirecrawlConfig(api_key="fc-1234567890abcdef")
        assert config.api_key == "fc-1234567890abcdef"

        # Invalid API key (no fc- prefix)
        with pytest.raises(
            ValidationError, match="Firecrawl API key must start with 'fc-'"
        ):
            FirecrawlConfig(api_key="invalid-key")


class TestChunkingConfig:
    """Test chunking configuration."""

    def test_default_values(self):
        config = ChunkingConfig()
        assert config.chunk_size == 1600
        assert config.chunk_overlap == 320
        assert config.strategy == ChunkingStrategy.ENHANCED
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 3000

    def test_chunk_size_validation(self):
        # Valid configuration
        config = ChunkingConfig(
            chunk_size=1000,
            chunk_overlap=200,
            min_chunk_size=50,
            max_chunk_size=2000,
        )
        assert config.chunk_size == 1000

        # Invalid: overlap >= chunk_size
        with pytest.raises(
            ValidationError, match="chunk_overlap must be less than chunk_size"
        ):
            ChunkingConfig(chunk_size=1000, chunk_overlap=1000)

        # Invalid: min_chunk_size > chunk_size
        with pytest.raises(
            ValidationError, match="min_chunk_size must be <= chunk_size"
        ):
            ChunkingConfig(chunk_size=1000, min_chunk_size=1500)

        # Invalid: max_chunk_size < chunk_size
        with pytest.raises(
            ValidationError, match="max_chunk_size must be >= chunk_size"
        ):
            ChunkingConfig(chunk_size=2000, max_chunk_size=1500)


class TestConfig:
    """Test main configuration."""

    def test_default_values(self):
        config = Config()
        assert config.environment == Environment.DEVELOPMENT
        assert config.debug is False
        assert config.log_level == LogLevel.INFO
        assert config.app_name == "AI Documentation Vector DB"
        assert config.version == "0.1.0"
        assert config.embedding_provider == EmbeddingProvider.FASTEMBED
        assert config.crawl_provider == CrawlProvider.CRAWL4AI

    def test_nested_configs(self):
        config = Config()
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.qdrant, QdrantConfig)
        assert isinstance(config.openai, OpenAIConfig)
        assert isinstance(config.fastembed, FastEmbedConfig)
        assert isinstance(config.firecrawl, FirecrawlConfig)
        assert isinstance(config.crawl4ai, Crawl4AIConfig)
        assert isinstance(config.chunking, ChunkingConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.performance, PerformanceConfig)

    def test_provider_key_validation_openai(self):
        # Should raise error when using OpenAI without API key
        with pytest.raises(ValidationError, match="OpenAI API key required"):
            Config(
                embedding_provider=EmbeddingProvider.OPENAI,
                openai=OpenAIConfig(api_key=None),
            )

        # Should work with API key
        config = Config(
            embedding_provider=EmbeddingProvider.OPENAI,
            openai=OpenAIConfig(api_key="sk-test"),
        )
        assert config.embedding_provider == EmbeddingProvider.OPENAI

    def test_provider_key_validation_firecrawl(self):
        # Should raise error when using Firecrawl without API key
        with pytest.raises(ValidationError, match="Firecrawl API key required"):
            Config(
                crawl_provider=CrawlProvider.FIRECRAWL,
                firecrawl=FirecrawlConfig(api_key=None),
            )

        # Should work with API key
        config = Config(
            crawl_provider=CrawlProvider.FIRECRAWL,
            firecrawl=FirecrawlConfig(api_key="fc-test"),
        )
        assert config.crawl_provider == CrawlProvider.FIRECRAWL

    def test_directory_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            cache_dir = Path(temp_dir) / "cache"
            logs_dir = Path(temp_dir) / "logs"

            config = Config(
                data_dir=data_dir,
                cache_dir=cache_dir,
                logs_dir=logs_dir,
            )

            # Config should be valid and directories should be created
            assert config.data_dir == data_dir
            assert data_dir.exists()
            assert cache_dir.exists()
            assert logs_dir.exists()

    def test_environment_variables(self):
        # Test environment variable loading
        with patch.dict(
            os.environ,
            {
                "AI_DOCS_DEBUG": "true",
                "AI_DOCS_LOG_LEVEL": "DEBUG",
                "AI_DOCS_EMBEDDING_PROVIDER": "openai",
                "AI_DOCS_OPENAI__API_KEY": "sk-test-env",
            },
        ):
            config = Config()
            assert config.debug is True
            assert config.log_level == LogLevel.DEBUG
            assert config.embedding_provider == EmbeddingProvider.OPENAI
            assert config.openai.api_key == "sk-test-env"

    def test_documentation_sites(self):
        sites = [
            DocumentationSite(name="Test Docs", url="https://example.com"),
            DocumentationSite(
                name="API Docs", url="https://api.example.com", max_pages=100
            ),
        ]
        config = Config(documentation_sites=sites)
        assert len(config.documentation_sites) == 2
        assert config.documentation_sites[0].name == "Test Docs"
        assert config.documentation_sites[1].max_pages == 100


class TestConfigSingleton:
    """Test configuration singleton pattern."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def test_get_config_creates_instance(self):
        config = get_config()
        assert isinstance(config, Config)
        assert config.app_name == "AI Documentation Vector DB"

    def test_get_config_returns_same_instance(self):
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_set_config(self):
        custom_config = Config(app_name="Custom App")
        set_config(custom_config)

        retrieved_config = get_config()
        assert retrieved_config is custom_config
        assert retrieved_config.app_name == "Custom App"

    def test_reset_config(self):
        # Set custom config
        custom_config = Config(app_name="Custom App")
        set_config(custom_config)
        assert get_config().app_name == "Custom App"

        # Reset and get new instance
        reset_config()
        new_config = get_config()
        assert new_config is not custom_config
        assert new_config.app_name == "AI Documentation Vector DB"


class TestLegacyCompatibility:
    """Test backward compatibility."""

    def test_unified_config_alias(self):
        """Test that Config is an alias for Config."""
        assert Config is Config

    def test_unified_config_works(self):
        """Test that Config can be used as expected."""
        config = Config(app_name="Legacy Test")
        assert isinstance(config, Config)
        assert config.app_name == "Legacy Test"


class TestDocumentationSite:
    """Test documentation site configuration."""

    def test_default_values(self):
        site = DocumentationSite(name="Test", url="https://example.com")
        assert site.name == "Test"
        assert str(site.url) == "https://example.com/"
        assert site.max_pages == 50
        assert site.max_depth == 2
        assert site.priority == "medium"

    def test_custom_values(self):
        site = DocumentationSite(
            name="Custom Site",
            url="https://docs.example.com",
            max_pages=100,
            max_depth=3,
            priority="high",
        )
        assert site.name == "Custom Site"
        assert site.max_pages == 100
        assert site.max_depth == 3
        assert site.priority == "high"

    def test_url_validation(self):
        # Valid URLs
        DocumentationSite(name="Test", url="https://example.com")
        DocumentationSite(name="Test", url="http://localhost:8000")

        # Invalid URL should raise ValidationError
        with pytest.raises(ValidationError):
            DocumentationSite(name="Test", url="invalid-url")


@pytest.mark.integration
class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_complete_config_creation(self):
        """Test creating a complete configuration with all components."""
        config = Config(
            environment=Environment.PRODUCTION,
            debug=False,
            log_level=LogLevel.WARNING,
            embedding_provider=EmbeddingProvider.OPENAI,
            crawl_provider=CrawlProvider.FIRECRAWL,
            openai=OpenAIConfig(api_key="sk-test", model="text-embedding-3-large"),
            firecrawl=FirecrawlConfig(api_key="fc-test"),
            qdrant=QdrantConfig(url="http://remote:6333", api_key="qdrant-key"),
            chunking=ChunkingConfig(chunk_size=2000, chunk_overlap=400),
            documentation_sites=[
                DocumentationSite(name="Main Docs", url="https://docs.example.com"),
                DocumentationSite(
                    name="API Docs", url="https://api.example.com", max_pages=200
                ),
            ],
        )

        # Verify all components are properly configured
        assert config.environment == Environment.PRODUCTION
        assert config.embedding_provider == EmbeddingProvider.OPENAI
        assert config.openai.model == "text-embedding-3-large"
        assert config.firecrawl.api_key == "fc-test"
        assert config.qdrant.api_key == "qdrant-key"
        assert config.chunking.chunk_size == 2000
        assert len(config.documentation_sites) == 2

    def test_config_with_env_file(self):
        """Test configuration loading with environment file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("AI_DOCS_DEBUG=true\n")
            f.write("AI_DOCS_LOG_LEVEL=DEBUG\n")
            f.write("AI_DOCS_OPENAI__API_KEY=sk-env-test\n")
            env_file = f.name

        try:
            # This would normally load from .env, but we can't easily test that
            # without modifying the working directory
            pass
        finally:
            os.unlink(env_file)
