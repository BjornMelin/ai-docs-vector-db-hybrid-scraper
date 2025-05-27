"""Tests for service configuration validation.

UnifiedConfig Pattern Notes:
- For environment variables: Use double underscore syntax (e.g., 'qdrant__url')
- For programmatic creation: Use nested configuration objects (e.g., qdrant=QdrantConfig(url="..."))
- The double underscore syntax is automatically parsed by Pydantic when loading from environment
"""

import pytest
from pydantic import ValidationError
from src.config.enums import CrawlProvider
from src.config.enums import EmbeddingProvider
from src.config.models import CacheConfig
from src.config.models import FirecrawlConfig
from src.config.models import OpenAIConfig
from src.config.models import PerformanceConfig
from src.config.models import QdrantConfig
from src.config.models import UnifiedConfig


class TestUnifiedConfig:
    """Test unified configuration validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UnifiedConfig()

        # Check nested Qdrant config
        assert config.qdrant.url == "http://localhost:6333"
        assert config.qdrant.collection_name == "documents"

        # Check nested OpenAI config
        assert config.openai.model == "text-embedding-3-small"
        assert config.openai.dimensions == 1536

        # Check provider preferences
        assert config.embedding_provider == EmbeddingProvider.FASTEMBED
        assert config.crawl_provider == CrawlProvider.CRAWL4AI

        # Check performance settings
        assert config.performance.max_retries == 3

    def test_valid_config(self):
        """Test valid configuration."""
        config = UnifiedConfig(
            embedding_provider=EmbeddingProvider.OPENAI,
            qdrant=QdrantConfig(url="https://my-qdrant.com"),
            openai=OpenAIConfig(
                api_key="sk-fake1234567890abcdef",
                model="text-embedding-3-large",
                dimensions=3072,
            ),
        )

        assert config.qdrant.url == "https://my-qdrant.com"
        assert config.openai.api_key == "sk-fake1234567890abcdef"
        assert config.openai.model == "text-embedding-3-large"
        assert config.embedding_provider == EmbeddingProvider.OPENAI

    def test_url_validation(self):
        """Test URL validation."""
        # Valid URLs
        config = UnifiedConfig(qdrant=QdrantConfig(url="http://localhost:6333/"))
        assert config.qdrant.url == "http://localhost:6333"  # Trailing slash removed

        # Invalid URL
        with pytest.raises(ValidationError, match="must start with http"):
            UnifiedConfig(qdrant=QdrantConfig(url="localhost:6333"))

    def test_openai_key_validation(self):
        """Test OpenAI API key validation."""
        # Valid key
        config = UnifiedConfig(openai=OpenAIConfig(api_key="sk-fake1234567890abcdef"))
        assert config.openai.api_key == "sk-fake1234567890abcdef"

        # None is valid
        config = UnifiedConfig(openai=OpenAIConfig(api_key=None))
        assert config.openai.api_key is None

        # Invalid key format
        with pytest.raises(ValidationError, match="must start with 'sk-'"):
            UnifiedConfig(openai=OpenAIConfig(api_key="invalid-key"))

    def test_model_validation(self):
        """Test model name validation."""
        # Valid models
        for model in [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]:
            config = UnifiedConfig(openai=OpenAIConfig(model=model))
            assert config.openai.model == model

        # Invalid model
        with pytest.raises(ValidationError, match="Invalid OpenAI model"):
            UnifiedConfig(openai=OpenAIConfig(model="invalid-model"))

    def test_numeric_validation(self):
        """Test numeric field validation."""
        # Valid values
        config = UnifiedConfig(
            openai=OpenAIConfig(
                dimensions=2048,
                batch_size=50,
            ),
            performance=PerformanceConfig(
                max_concurrent_requests=20,
                request_timeout=60.0,
                max_retries=5,
                retry_base_delay=2.0,
            ),
        )
        assert config.openai.dimensions == 2048
        assert config.performance.max_concurrent_requests == 20

        # Invalid dimensions (too large)
        with pytest.raises(ValidationError):
            UnifiedConfig(openai=OpenAIConfig(dimensions=4096))

        # Invalid batch size (too large)
        with pytest.raises(ValidationError):
            UnifiedConfig(openai=OpenAIConfig(batch_size=3000))

        # Invalid timeout (negative)
        with pytest.raises(ValidationError):
            UnifiedConfig(performance=PerformanceConfig(request_timeout=-1))

        # Invalid retries (too many)
        with pytest.raises(ValidationError):
            UnifiedConfig(performance=PerformanceConfig(max_retries=20))

    def test_provider_validation(self):
        """Test provider validation with required API keys."""
        # Valid configuration with OpenAI provider and API key
        config = UnifiedConfig(
            embedding_provider=EmbeddingProvider.OPENAI,
            openai=OpenAIConfig(api_key="sk-fake1234567890abcdef"),
        )
        assert config.embedding_provider == EmbeddingProvider.OPENAI

        # Invalid - OpenAI provider without API key
        with pytest.raises(ValidationError, match="OpenAI API key required"):
            UnifiedConfig(
                embedding_provider=EmbeddingProvider.OPENAI,
                openai=OpenAIConfig(api_key=None),
            )

        # Valid - Firecrawl provider with API key
        from src.config.models import FirecrawlConfig

        config = UnifiedConfig(
            crawl_provider=CrawlProvider.FIRECRAWL,
            firecrawl=FirecrawlConfig(api_key="fc-test123"),
        )
        assert config.crawl_provider == CrawlProvider.FIRECRAWL

        # Invalid - Firecrawl provider without API key
        with pytest.raises(ValidationError, match="Firecrawl API key required"):
            UnifiedConfig(
                crawl_provider=CrawlProvider.FIRECRAWL,
                firecrawl=FirecrawlConfig(api_key=None),
            )

    def test_cache_config(self):
        """Test cache configuration."""
        config = UnifiedConfig(
            cache=CacheConfig(
                enable_caching=True,
                ttl_embeddings=7200,
                local_max_size=500,
            )
        )

        assert config.cache.enable_caching is True
        assert config.cache.ttl_embeddings == 7200
        assert config.cache.local_max_size == 500

    def test_nested_config_validation(self):
        """Test nested configuration validation."""
        # Valid nested config
        config = UnifiedConfig(
            qdrant=QdrantConfig(
                batch_size=50,
                max_retries=5,
            ),
            openai=OpenAIConfig(
                batch_size=100,
            ),
            cache=CacheConfig(
                redis_pool_size=20,
            ),
        )

        assert config.qdrant.batch_size == 50
        assert config.qdrant.max_retries == 5
        assert config.openai.batch_size == 100
        assert config.cache.redis_pool_size == 20

        # Invalid nested values
        with pytest.raises(ValidationError):
            UnifiedConfig(
                qdrant=QdrantConfig(batch_size=2000)  # Too large
            )

    def test_env_var_loading(self):
        """Test that env var format works (for documentation)."""
        # This test documents how environment variables would be used
        # In practice, these would be set as actual environment variables:
        # AI_DOCS__QDRANT__URL=https://my-qdrant.com
        # AI_DOCS__OPENAI__API_KEY=sk-test123
        # AI_DOCS__CACHE__TTL_EMBEDDINGS=7200
        pass


class TestOpenAIConfigValidation:
    """Test OpenAI configuration API key validation."""

    def test_valid_openai_api_keys(self):
        """Test valid OpenAI API key formats."""
        # Valid key format with minimum length (20+ chars)
        valid_key = "sk-" + "a" * 17  # 20 characters total
        config = OpenAIConfig(api_key=valid_key)
        assert config.api_key == valid_key

        # Valid longer key (traditional format)
        valid_key_long = "sk-" + "a" * 48  # 51 characters total
        config = OpenAIConfig(api_key=valid_key_long)
        assert config.api_key == valid_key_long

        # Valid key with mixed alphanumeric and hyphens
        valid_key_mixed = "sk-fake1234567890abcdef-test123"
        config = OpenAIConfig(api_key=valid_key_mixed)
        assert config.api_key == valid_key_mixed

        # Valid key is None (optional)
        config = OpenAIConfig(api_key=None)
        assert config.api_key is None

        # Valid empty string becomes None
        config = OpenAIConfig(api_key="")
        assert config.api_key is None

        # Valid key with whitespace gets stripped
        valid_key_whitespace = "  sk-" + "a" * 17 + "  "
        config = OpenAIConfig(api_key=valid_key_whitespace)
        assert config.api_key == "sk-" + "a" * 17

    def test_invalid_openai_api_keys(self):
        """Test invalid OpenAI API key formats."""
        # Missing sk- prefix
        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(api_key="invalid-prefix-" + "a" * 40)
        assert "must start with 'sk-'" in str(exc_info.value)

        # Too short
        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(api_key="sk-short")
        assert "too short" in str(exc_info.value)

        # Invalid characters (spaces) - make sure it's long enough to trigger character validation
        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(api_key="sk-invalid key with spaces123")
        assert "invalid characters" in str(exc_info.value)

        # Invalid characters (special symbols) - make sure it's long enough
        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(api_key="sk-invalid@key#with$symbols123")
        assert "invalid characters" in str(exc_info.value)

        # Unicode characters should be rejected
        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(api_key="sk-tëst" + "a" * 15)  # Unicode chars
        assert "non-ASCII characters" in str(exc_info.value)

        # Too long (DoS protection)
        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(api_key="sk-" + "a" * 500)  # Too long
        assert "too long" in str(exc_info.value)

        # Just the prefix without content
        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(api_key="sk-")
        assert "too short" in str(exc_info.value)


class TestFirecrawlConfigValidation:
    """Test Firecrawl configuration API key validation."""

    def test_valid_firecrawl_api_keys(self):
        """Test valid Firecrawl API key formats."""
        # Valid key format with minimum length
        valid_key = "fc-fake123456"
        config = FirecrawlConfig(api_key=valid_key)
        assert config.api_key == valid_key

        # Valid key with mixed alphanumeric, hyphens, and underscores
        valid_key_mixed = "fc-fake_123-456_789"
        config = FirecrawlConfig(api_key=valid_key_mixed)
        assert config.api_key == valid_key_mixed

        # Valid key is None (optional)
        config = FirecrawlConfig(api_key=None)
        assert config.api_key is None

        # Valid empty string becomes None
        config = FirecrawlConfig(api_key="")
        assert config.api_key is None

        # Valid key with whitespace gets stripped
        valid_key_whitespace = "  fc-fake123456  "
        config = FirecrawlConfig(api_key=valid_key_whitespace)
        assert config.api_key == "fc-fake123456"

    def test_invalid_firecrawl_api_keys(self):
        """Test invalid Firecrawl API key formats."""
        # Missing fc- prefix
        with pytest.raises(ValidationError) as exc_info:
            FirecrawlConfig(api_key="invalid-prefix-123456")
        assert "must start with 'fc-'" in str(exc_info.value)

        # Too short
        with pytest.raises(ValidationError) as exc_info:
            FirecrawlConfig(api_key="fc-short")
        assert "too short" in str(exc_info.value)

        # Invalid characters (spaces)
        with pytest.raises(ValidationError) as exc_info:
            FirecrawlConfig(api_key="fc-invalid key with spaces")
        assert "invalid characters" in str(exc_info.value)

        # Invalid characters (special symbols)
        with pytest.raises(ValidationError) as exc_info:
            FirecrawlConfig(api_key="fc-invalid@key#with$symbols")
        assert "invalid characters" in str(exc_info.value)

        # Just the prefix without content
        with pytest.raises(ValidationError) as exc_info:
            FirecrawlConfig(api_key="fc-")
        assert "too short" in str(exc_info.value)


class TestUnifiedConfigProviderValidation:
    """Test UnifiedConfig provider-based API key validation."""

    def test_openai_provider_requires_api_key(self):
        """Test that OpenAI provider requires valid API key."""
        # Should fail when OpenAI provider selected but no API key
        with pytest.raises(ValidationError) as exc_info:
            UnifiedConfig(
                embedding_provider=EmbeddingProvider.OPENAI,
                openai=OpenAIConfig(api_key=None),
            )
        assert "OpenAI API key required" in str(exc_info.value)

        # Should succeed when OpenAI provider and valid API key provided
        config = UnifiedConfig(
            embedding_provider=EmbeddingProvider.OPENAI,
            openai=OpenAIConfig(api_key="sk-" + "a" * 17),
        )
        assert config.embedding_provider == EmbeddingProvider.OPENAI
        assert config.openai.api_key == "sk-" + "a" * 17

        # Should succeed when FastEmbed provider (no API key needed)
        config = UnifiedConfig(
            embedding_provider=EmbeddingProvider.FASTEMBED,
            openai=OpenAIConfig(api_key=None),  # No key needed for FastEmbed
        )
        assert config.embedding_provider == EmbeddingProvider.FASTEMBED

    def test_firecrawl_provider_requires_api_key(self):
        """Test that Firecrawl provider requires valid API key."""
        # Should fail when Firecrawl provider selected but no API key
        with pytest.raises(ValidationError) as exc_info:
            UnifiedConfig(
                crawl_provider=CrawlProvider.FIRECRAWL,
                firecrawl=FirecrawlConfig(api_key=None),
            )
        assert "Firecrawl API key required" in str(exc_info.value)

        # Should succeed when Firecrawl provider and valid API key provided
        config = UnifiedConfig(
            crawl_provider=CrawlProvider.FIRECRAWL,
            firecrawl=FirecrawlConfig(api_key="fc-fake123456"),
        )
        assert config.crawl_provider == CrawlProvider.FIRECRAWL
        assert config.firecrawl.api_key == "fc-fake123456"

        # Should succeed when Crawl4AI provider (no API key needed)
        config = UnifiedConfig(
            crawl_provider=CrawlProvider.CRAWL4AI,
            firecrawl=FirecrawlConfig(api_key=None),  # No key needed for Crawl4AI
        )
        assert config.crawl_provider == CrawlProvider.CRAWL4AI

    def test_combined_provider_validation(self):
        """Test validation with both providers requiring API keys."""
        # Should fail when both providers selected but keys missing
        with pytest.raises(ValidationError) as exc_info:
            UnifiedConfig(
                embedding_provider=EmbeddingProvider.OPENAI,
                crawl_provider=CrawlProvider.FIRECRAWL,
                openai=OpenAIConfig(api_key=None),
                firecrawl=FirecrawlConfig(api_key=None),
            )
        # Should mention at least one of the missing keys
        error_msg = str(exc_info.value)
        assert (
            "OpenAI API key required" in error_msg
            or "Firecrawl API key required" in error_msg
        )

        # Should succeed when both providers and valid API keys provided
        config = UnifiedConfig(
            embedding_provider=EmbeddingProvider.OPENAI,
            crawl_provider=CrawlProvider.FIRECRAWL,
            openai=OpenAIConfig(api_key="sk-" + "a" * 17),
            firecrawl=FirecrawlConfig(api_key="fc-fake123456"),
        )
        assert config.embedding_provider == EmbeddingProvider.OPENAI
        assert config.crawl_provider == CrawlProvider.FIRECRAWL
        assert config.openai.api_key == "sk-" + "a" * 17
        assert config.firecrawl.api_key == "fc-fake123456"
