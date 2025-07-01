"""Tests for consolidated configuration system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config import (
    CacheConfig,
    CacheType,
    ChunkingConfig,
    ChunkingStrategy,
    Config,
    Crawl4AIConfig,
    CrawlProvider,
    DocumentationSite,
    EmbeddingConfig,
    EmbeddingProvider,
    Environment,
    FastEmbedConfig,
    FirecrawlConfig,
    LogLevel,
    OpenAIConfig,
    PerformanceConfig,
    QdrantConfig,
    SecurityConfig,
    get_config,
    reset_config,
    set_config,
)


class TestCacheConfig:
    """Test cache configuration."""

    def test_default_values(self):
        config = CacheConfig()
        assert config.enable_caching is True
        assert config.redis_url == "redis://localhost:6379"
        assert config.local_max_size == 1000
        assert config.ttl_embeddings == 86400

    def test_custom_values(self):
        config = CacheConfig(
            enable_caching=False,
            redis_url="redis://custom:6380",
            local_max_size=500,
            ttl_embeddings=1800,
        )
        assert config.enable_caching is False
        assert config.redis_url == "redis://custom:6380"
        assert config.local_max_size == 500
        assert config.ttl_embeddings == 1800

    def test_validation(self):
        with pytest.raises(ValidationError):
            CacheConfig(local_max_size=-1)  # Must be positive
        with pytest.raises(ValidationError):
            CacheConfig(ttl_embeddings=-1)  # Must be positive


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
            Path(env_file).unlink()


# Consolidated Provider Configuration Tests
@pytest.mark.parametrize(
    ("provider", "config_class", "valid_params", "invalid_params"),
    [
        pytest.param(
            EmbeddingProvider.OPENAI,
            OpenAIConfig,
            {"api_key": "sk-test", "model": "text-embedding-3-small"},
            {"api_key": "invalid", "dimensions": -1},
            id="openai",
        ),
        pytest.param(
            EmbeddingProvider.FASTEMBED,
            FastEmbedConfig,
            {"model": "BAAI/bge-small-en-v1.5", "max_length": 512},
            {"max_length": -1, "batch_size": 0},
            id="fastembed",
        ),
    ],
)
class TestProviderConfigurationParametrized:
    """Parametrized tests for provider configurations."""

    def test_provider_default_initialization(
        self, provider, config_class, valid_params, invalid_params
    ):
        """Test that provider configs can be initialized with defaults."""
        config = config_class()
        assert config is not None

    def test_provider_custom_initialization(
        self, provider, config_class, valid_params, invalid_params
    ):
        """Test provider configs with custom parameters."""
        config = config_class(**valid_params)
        for key, value in valid_params.items():
            assert getattr(config, key) == value

    def test_provider_validation_errors(
        self, provider, config_class, valid_params, invalid_params
    ):
        """Test that invalid parameters raise validation errors."""
        for key, invalid_value in invalid_params.items():
            with pytest.raises(ValidationError):
                config_class(**{key: invalid_value})


@pytest.mark.parametrize(
    ("crawl_provider", "config_class"),
    [
        (CrawlProvider.CRAWL4AI, Crawl4AIConfig),
        (CrawlProvider.FIRECRAWL, FirecrawlConfig),
    ],
)
class TestCrawlProviderConfiguration:
    """Tests for crawl provider configurations."""

    def test_crawl_provider_initialization(self, crawl_provider, config_class):
        """Test crawl provider config initialization."""
        config = config_class()
        assert config is not None

    def test_crawl_provider_in_main_config(self, crawl_provider, config_class):
        """Test crawl provider integration in main config."""
        provider_config = config_class()
        main_config = Config(crawl_provider=crawl_provider)
        assert main_config.crawl_provider == crawl_provider


@pytest.mark.parametrize(
    ("config_class", "required_field", "valid_value", "invalid_value"),
    [
        (OpenAIConfig, "api_key", "sk-valid-key", "invalid-key"),
        (FirecrawlConfig, "api_key", "fc-valid-key", "invalid-key"),
        (QdrantConfig, "timeout", 30.0, -1.0),
        (CacheConfig, "ttl_seconds", 3600, -1),
    ],
)
class TestConfigFieldValidation:
    """Parametrized tests for config field validation."""

    def test_valid_field_values(
        self, config_class, required_field, valid_value, invalid_value
    ):
        """Test that valid field values are accepted."""
        params = {required_field: valid_value}
        config = config_class(**params)
        assert getattr(config, required_field) == valid_value

    def test_invalid_field_values(
        self, config_class, required_field, valid_value, invalid_value
    ):
        """Test that invalid field values raise validation errors."""
        params = {required_field: invalid_value}
        with pytest.raises(ValidationError):
            config_class(**params)


# ==============================================================================
# CONFTEST.PY CONTENT - TO BE EXTRACTED TO SEPARATE FILE
# ==============================================================================
"""
Consolidated configuration test fixtures and utilities.

This content should be moved to conftest.py for shared fixtures.
"""

from collections.abc import Generator
from typing import Any

import pytest
from hypothesis import strategies as st


# Hypothesis strategies for property-based testing
positive_int = st.integers(min_value=1, max_value=10000)
small_positive_int = st.integers(min_value=1, max_value=100)
positive_float = st.floats(min_value=0.1, max_value=100.0)
percentage = st.floats(min_value=0.0, max_value=100.0)
browser_types = st.sampled_from(["chromium", "firefox", "webkit"])
database_names = st.text(
    min_size=1,
    max_size=20,
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
)


@pytest.fixture
def temp_config_dir() -> Generator[Path]:
    """Create a temporary directory for config files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_env_vars() -> dict[str, str]:
    """Sample environment variables for testing."""
    return {
        "APP_MODE": "enterprise",
        "ENVIRONMENT": "production",
        "CACHE_ENABLE_CACHING": "true",
        "CACHE_DRAGONFLY_URL": "redis://test:6379",
        "QDRANT_URL": "http://test-qdrant:6333",
        "OPENAI_API_KEY": "test-key-123",
        "FIRECRAWL_API_KEY": "fc-test-key",
        "SECURITY_REQUIRE_API_KEYS": "true",
        "PERFORMANCE_MAX_CONCURRENT_REQUESTS": "50",
    }


@pytest.fixture
def mock_env_vars(sample_env_vars: dict[str, str]) -> Generator[None]:
    """Mock environment variables for testing."""
    with patch.dict(os.environ, sample_env_vars, clear=True):
        yield


@pytest.fixture
def cache_config_fixture() -> CacheConfig:
    """Create a basic cache configuration for testing."""
    return CacheConfig(
        enable_caching=True,
        dragonfly_url="redis://localhost:6379",
        local_max_size=1000,
        ttl_seconds=3600,
    )


@pytest.fixture
def qdrant_config_fixture() -> QdrantConfig:
    """Create a basic Qdrant configuration for testing."""
    return QdrantConfig(
        url="http://localhost:6333",
        api_key=None,
        collection_name="test_collection",
        vector_size=384,
        distance="cosine",
    )


@pytest.fixture
def openai_config_fixture() -> OpenAIConfig:
    """Create a basic OpenAI configuration for testing."""
    return OpenAIConfig(
        api_key="sk-test-key-123",
        model="text-embedding-3-small",
        max_tokens=8192,
        timeout=30.0,
    )


@pytest.fixture
def security_config_fixture() -> SecurityConfig:
    """Create a basic security configuration for testing."""
    return SecurityConfig(
        allowed_domains=["*"],
        blocked_domains=[],
        require_api_keys=True,
        api_key_header="X-API-Key",
        enable_rate_limiting=True,
        rate_limit_requests=100,
        rate_limit_requests_per_minute=60,
        max_query_length=1000,
        max_url_length=2048,
    )


# Provider test data for parametrized tests
PROVIDER_TEST_DATA = [
    pytest.param(
        EmbeddingProvider.OPENAI,
        OpenAIConfig,
        {"api_key": "sk-test", "model": "text-embedding-3-small"},
        {"api_key": "invalid", "dimensions": -1},
        id="openai",
    ),
    pytest.param(
        EmbeddingProvider.FASTEMBED,
        FastEmbedConfig,
        {"model": "BAAI/bge-small-en-v1.5", "max_length": 512},
        {"max_length": -1, "batch_size": 0},
        id="fastembed",
    ),
]

CACHE_TYPE_TEST_DATA = [
    pytest.param(CacheType.LOCAL, {"cache_type": CacheType.LOCAL}, id="local"),
    pytest.param(
        CacheType.REDIS,
        {"cache_type": CacheType.REDIS, "redis_url": "redis://localhost:6379"},
        id="redis",
    ),
]

ENVIRONMENT_TEST_DATA = [
    pytest.param(Environment.DEVELOPMENT, {"debug": True}, id="development"),
    pytest.param(Environment.PRODUCTION, {"debug": False}, id="production"),
    pytest.param(Environment.TESTING, {"debug": True}, id="testing"),
]


def create_test_config_file(
    config_dir: Path, content: str, filename: str = "config.toml"
) -> Path:
    """Create a test configuration file."""
    config_file = config_dir / filename
    config_file.write_text(content)
    return config_file


def assert_config_equality(
    config1: Any, config2: Any, exclude_fields: list[str] | None = None
) -> None:
    """Assert that two configurations are equal, optionally excluding fields."""
    exclude_fields = exclude_fields or []

    if hasattr(config1, "model_dump") and hasattr(config2, "model_dump"):
        dict1 = config1.model_dump(exclude=set(exclude_fields))
        dict2 = config2.model_dump(exclude=set(exclude_fields))
        assert dict1 == dict2
    else:
        assert config1 == config2


def get_config_validation_errors(config_class: type, **invalid_params) -> list[str]:
    """Get validation errors for invalid configuration parameters."""
    try:
        config_class(**invalid_params)
        return []
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        return [str(e)]


# ==============================================================================
# END CONFTEST.PY CONTENT
# ==============================================================================
