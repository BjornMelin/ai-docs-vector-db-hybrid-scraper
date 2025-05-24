"""Tests for service configuration validation."""

import pytest
from pydantic import ValidationError
from src.services.config import APIConfig


class TestAPIConfig:
    """Test API configuration validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = APIConfig()

        assert config.qdrant_url == "http://localhost:6333"
        assert config.openai_model == "text-embedding-3-small"
        assert config.openai_dimensions == 1536
        assert config.preferred_embedding_provider == "fastembed"
        assert config.max_retries == 3

    def test_valid_config(self):
        """Test valid configuration."""
        config = APIConfig(
            qdrant_url="https://my-qdrant.com",
            openai_api_key="sk-test123",
            openai_model="text-embedding-3-large",
            openai_dimensions=3072,
            preferred_embedding_provider="openai",
        )

        assert config.qdrant_url == "https://my-qdrant.com"
        assert config.openai_api_key == "sk-test123"
        assert config.openai_model == "text-embedding-3-large"

    def test_url_validation(self):
        """Test URL validation."""
        # Valid URLs
        config = APIConfig(qdrant_url="http://localhost:6333/")
        assert config.qdrant_url == "http://localhost:6333"  # Trailing slash removed

        # Invalid URL
        with pytest.raises(ValidationError, match="must start with http"):
            APIConfig(qdrant_url="localhost:6333")

    def test_openai_key_validation(self):
        """Test OpenAI API key validation."""
        # Valid key
        config = APIConfig(openai_api_key="sk-proj-test123")
        assert config.openai_api_key == "sk-proj-test123"

        # None is valid
        config = APIConfig(openai_api_key=None)
        assert config.openai_api_key is None

        # Invalid key format
        with pytest.raises(ValidationError, match="must start with 'sk-'"):
            APIConfig(openai_api_key="invalid-key")

    def test_model_validation(self):
        """Test model name validation."""
        # Valid models
        for model in [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]:
            config = APIConfig(openai_model=model)
            assert config.openai_model == model

        # Invalid model
        with pytest.raises(ValidationError, match="Invalid OpenAI model"):
            APIConfig(openai_model="invalid-model")

    def test_numeric_validation(self):
        """Test numeric field validation."""
        # Valid values
        config = APIConfig(
            openai_dimensions=2048,
            openai_batch_size=50,
            max_concurrent_requests=20,
            request_timeout=60.0,
            max_retries=5,
            retry_base_delay=2.0,
        )
        assert config.openai_dimensions == 2048
        assert config.max_concurrent_requests == 20

        # Invalid dimensions (too large)
        with pytest.raises(ValidationError):
            APIConfig(openai_dimensions=4096)

        # Invalid batch size (too large)
        with pytest.raises(ValidationError):
            APIConfig(openai_batch_size=3000)

        # Invalid timeout (negative)
        with pytest.raises(ValidationError):
            APIConfig(request_timeout=-1)

        # Invalid retries (too many)
        with pytest.raises(ValidationError):
            APIConfig(max_retries=20)

    def test_provider_validation(self):
        """Test provider name validation."""
        # Valid providers
        config = APIConfig(
            preferred_embedding_provider="openai",
            preferred_crawl_provider="firecrawl",
        )
        assert config.preferred_embedding_provider == "openai"
        assert config.preferred_crawl_provider == "firecrawl"

        # Invalid embedding provider
        with pytest.raises(ValidationError, match="Invalid embedding provider"):
            APIConfig(preferred_embedding_provider="invalid")

        # Invalid crawl provider
        with pytest.raises(ValidationError, match="Invalid crawl provider"):
            APIConfig(preferred_crawl_provider="invalid")

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            APIConfig(unknown_field="value")
