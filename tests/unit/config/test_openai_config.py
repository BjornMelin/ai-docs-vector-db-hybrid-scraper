"""Test OpenAIConfig Pydantic model."""

import pytest
from pydantic import ValidationError

from config.models import OpenAIConfig


class TestOpenAIConfig:
    """Test OpenAIConfig model validation and behavior."""

    def test_default_values(self):
        """Test OpenAIConfig with default values."""
        config = OpenAIConfig()

        assert config.api_key is None
        assert config.model == "text-embedding-3-small"
        assert config.dimensions == 1536
        assert config.batch_size == 100

        # Rate limiting
        assert config.max_requests_per_minute == 3000
        assert config.max_tokens_per_minute == 1000000

        # Cost tracking
        assert config.cost_per_million_tokens == 0.02
        assert config.budget_limit is None

    def test_api_key_validation_valid(self):
        """Test valid API key formats."""
        # Valid API key
        config1 = OpenAIConfig(api_key="sk-proj-abcdef123456")
        assert config1.api_key == "sk-proj-abcdef123456"

        config2 = OpenAIConfig(api_key="sk-1234567890abcdefghij")
        assert config2.api_key == "sk-1234567890abcdefghij"

        # None is valid (optional)
        config3 = OpenAIConfig(api_key=None)
        assert config3.api_key is None

        # Empty string becomes None
        config4 = OpenAIConfig(api_key="")
        assert config4.api_key is None

        # Whitespace-only becomes None
        config5 = OpenAIConfig(api_key="   ")
        assert config5.api_key is None

    def test_api_key_validation_invalid_prefix(self):
        """Test API key with invalid prefix."""
        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(api_key="pk-1234567890abcdefghij")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("api_key",)
        assert "must start with 'sk-'" in str(errors[0]["msg"])

    def test_api_key_validation_too_short(self):
        """Test API key that's too short."""
        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(api_key="sk-abc")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("api_key",)
        assert "too short" in str(errors[0]["msg"])

    def test_api_key_validation_too_long(self):
        """Test API key that's too long."""
        long_key = "sk-" + "a" * 200
        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(api_key=long_key)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("api_key",)
        assert "too long" in str(errors[0]["msg"])

    def test_api_key_validation_invalid_characters(self):
        """Test API key with invalid characters."""
        # Special characters not allowed
        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(api_key="sk-test@key#123")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("api_key",)
        assert "too short" in str(errors[0]["msg"])

        # Non-ASCII characters
        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(api_key="sk-test™key123")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("api_key",)
        assert "non-ASCII characters" in str(errors[0]["msg"])

    def test_model_validation(self):
        """Test model name validation."""
        # Valid models
        valid_models = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]

        for model in valid_models:
            config = OpenAIConfig(model=model)
            assert config.model == model

        # Invalid model
        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(model="text-embedding-invalid")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("model",)
        assert "Invalid OpenAI model" in str(errors[0]["msg"])

    def test_dimensions_constraints(self):
        """Test dimensions constraints (0 < dimensions <= 3072)."""
        # Valid dimensions
        config1 = OpenAIConfig(dimensions=512)
        assert config1.dimensions == 512

        config2 = OpenAIConfig(dimensions=3072)
        assert config2.dimensions == 3072

        # Invalid dimensions
        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(dimensions=0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("dimensions",)
        assert "greater than 0" in str(errors[0]["msg"])

        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(dimensions=3073)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("dimensions",)
        assert "less than or equal to 3072" in str(errors[0]["msg"])

    def test_batch_size_constraints(self):
        """Test batch size constraints (0 < batch_size <= 2048)."""
        # Valid batch sizes
        config1 = OpenAIConfig(batch_size=1)
        assert config1.batch_size == 1

        config2 = OpenAIConfig(batch_size=2048)
        assert config2.batch_size == 2048

        # Invalid batch sizes
        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(batch_size=0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("batch_size",)

        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(batch_size=2049)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("batch_size",)

    def test_rate_limiting_constraints(self):
        """Test rate limiting constraints."""
        # Valid values
        config = OpenAIConfig(
            max_requests_per_minute=5000, max_tokens_per_minute=2000000
        )
        assert config.max_requests_per_minute == 5000
        assert config.max_tokens_per_minute == 2000000

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(max_requests_per_minute=0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_requests_per_minute",)

        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(max_tokens_per_minute=-100)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_tokens_per_minute",)

    def test_cost_tracking_constraints(self):
        """Test cost tracking constraints."""
        # Valid values
        config1 = OpenAIConfig(cost_per_million_tokens=0.05, budget_limit=100.0)
        assert config1.cost_per_million_tokens == 0.05
        assert config1.budget_limit == 100.0

        # Budget limit can be None (no limit)
        config2 = OpenAIConfig(budget_limit=None)
        assert config2.budget_limit is None

        # Invalid values
        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(cost_per_million_tokens=0.0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("cost_per_million_tokens",)
        assert "greater than 0" in str(errors[0]["msg"])

        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(budget_limit=-10.0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("budget_limit",)
        assert "greater than or equal to 0" in str(errors[0]["msg"])

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            OpenAIConfig(api_key="sk-test123456789012345", unknown_field="value")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "extra_forbidden"

    def test_model_serialization(self):
        """Test model serialization."""
        config = OpenAIConfig(
            api_key="sk-test123456789012345",
            model="text-embedding-3-large",
            dimensions=3072,
            budget_limit=50.0,
        )

        # Test model_dump
        data = config.model_dump()
        assert data["api_key"] == "sk-test123456789012345"
        assert data["model"] == "text-embedding-3-large"
        assert data["dimensions"] == 3072
        assert data["budget_limit"] == 50.0

        # Test model_dump_json
        json_str = config.model_dump_json()
        assert '"api_key":"sk-test123456789012345"' in json_str
        assert '"model":"text-embedding-3-large"' in json_str
