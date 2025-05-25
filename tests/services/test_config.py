"""Tests for service configuration validation."""

import pytest
from pydantic import ValidationError

from src.config.models import UnifiedConfig, QdrantConfig, OpenAIConfig, CacheConfig, PerformanceConfig
from src.config.enums import EmbeddingProvider, CrawlProvider


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
                api_key="sk-test123",
                model="text-embedding-3-large",
                dimensions=3072,
            ),
        )

        assert config.qdrant.url == "https://my-qdrant.com"
        assert config.openai.api_key == "sk-test123"
        assert config.openai.model == "text-embedding-3-large"
        assert config.embedding_provider == EmbeddingProvider.OPENAI

    def test_url_validation(self):
        """Test URL validation."""
        # Valid URLs
        config = UnifiedConfig(
            qdrant=QdrantConfig(url="http://localhost:6333/")
        )
        assert config.qdrant.url == "http://localhost:6333"  # Trailing slash removed

        # Invalid URL
        with pytest.raises(ValidationError, match="must start with http"):
            UnifiedConfig(
                qdrant=QdrantConfig(url="localhost:6333")
            )

    def test_openai_key_validation(self):
        """Test OpenAI API key validation."""
        # Valid key
        config = UnifiedConfig(
            openai=OpenAIConfig(api_key="sk-proj-test123")
        )
        assert config.openai.api_key == "sk-proj-test123"

        # None is valid
        config = UnifiedConfig(
            openai=OpenAIConfig(api_key=None)
        )
        assert config.openai.api_key is None

        # Invalid key format
        with pytest.raises(ValidationError, match="must start with 'sk-'"):
            UnifiedConfig(
                openai=OpenAIConfig(api_key="invalid-key")
            )

    def test_model_validation(self):
        """Test model name validation."""
        # Valid models
        for model in [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]:
            config = UnifiedConfig(
                openai=OpenAIConfig(model=model)
            )
            assert config.openai.model == model

        # Invalid model
        with pytest.raises(ValidationError, match="Invalid OpenAI model"):
            UnifiedConfig(
                openai=OpenAIConfig(model="invalid-model")
            )

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
            UnifiedConfig(
                openai=OpenAIConfig(dimensions=4096)
            )

        # Invalid batch size (too large)
        with pytest.raises(ValidationError):
            UnifiedConfig(
                openai=OpenAIConfig(batch_size=3000)
            )

        # Invalid timeout (negative)
        with pytest.raises(ValidationError):
            UnifiedConfig(
                performance=PerformanceConfig(request_timeout=-1)
            )

        # Invalid retries (too many)
        with pytest.raises(ValidationError):
            UnifiedConfig(
                performance=PerformanceConfig(max_retries=20)
            )

    def test_provider_validation(self):
        """Test provider validation with required API keys."""
        # Valid configuration with OpenAI provider and API key
        config = UnifiedConfig(
            embedding_provider=EmbeddingProvider.OPENAI,
            openai=OpenAIConfig(api_key="sk-test123"),
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