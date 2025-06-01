"""Test PerformanceConfig Pydantic model."""

import pytest
from pydantic import ValidationError

from config.models import PerformanceConfig


class TestPerformanceConfig:
    """Test PerformanceConfig model validation and behavior."""

    def test_default_values(self):
        """Test PerformanceConfig with default values."""
        config = PerformanceConfig()

        # Request settings
        assert config.max_concurrent_requests == 10
        assert config.request_timeout == 30.0

        # Retry settings
        assert config.max_retries == 3
        assert config.retry_base_delay == 1.0
        assert config.retry_max_delay == 60.0

        # Memory management
        assert config.max_memory_usage_mb == 1000.0
        assert config.gc_threshold == 0.8

        # Rate limiting
        assert isinstance(config.default_rate_limits, dict)
        assert "openai" in config.default_rate_limits
        assert "firecrawl" in config.default_rate_limits
        assert "crawl4ai" in config.default_rate_limits
        assert "qdrant" in config.default_rate_limits

        # Check specific rate limits
        assert config.default_rate_limits["openai"]["max_calls"] == 500
        assert config.default_rate_limits["openai"]["time_window"] == 60
        assert config.default_rate_limits["firecrawl"]["max_calls"] == 100
        assert config.default_rate_limits["firecrawl"]["time_window"] == 60
        assert config.default_rate_limits["crawl4ai"]["max_calls"] == 50
        assert config.default_rate_limits["crawl4ai"]["time_window"] == 1
        assert config.default_rate_limits["qdrant"]["max_calls"] == 100
        assert config.default_rate_limits["qdrant"]["time_window"] == 1

    def test_custom_values(self):
        """Test PerformanceConfig with custom values."""
        custom_rate_limits = {
            "openai": {"max_calls": 1000, "time_window": 60},
            "custom_api": {"max_calls": 200, "time_window": 10},
        }

        config = PerformanceConfig(
            max_concurrent_requests=50,
            request_timeout=60.0,
            max_retries=5,
            retry_base_delay=2.0,
            retry_max_delay=120.0,
            max_memory_usage_mb=2000.0,
            gc_threshold=0.9,
            default_rate_limits=custom_rate_limits,
        )

        assert config.max_concurrent_requests == 50
        assert config.request_timeout == 60.0
        assert config.max_retries == 5
        assert config.retry_base_delay == 2.0
        assert config.retry_max_delay == 120.0
        assert config.max_memory_usage_mb == 2000.0
        assert config.gc_threshold == 0.9
        assert config.default_rate_limits == custom_rate_limits

    def test_max_concurrent_requests_constraints(self):
        """Test max_concurrent_requests constraints (0 < value <= 100)."""
        # Valid values
        config1 = PerformanceConfig(max_concurrent_requests=1)
        assert config1.max_concurrent_requests == 1

        config2 = PerformanceConfig(max_concurrent_requests=100)
        assert config2.max_concurrent_requests == 100

        # Invalid: zero
        with pytest.raises(ValidationError) as exc_info:
            PerformanceConfig(max_concurrent_requests=0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_concurrent_requests",)
        assert "greater than 0" in str(errors[0]["msg"])

        # Invalid: exceeds maximum
        with pytest.raises(ValidationError) as exc_info:
            PerformanceConfig(max_concurrent_requests=101)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_concurrent_requests",)
        assert "less than or equal to 100" in str(errors[0]["msg"])

    def test_timeout_constraints(self):
        """Test timeout must be positive."""
        # Valid timeout
        config = PerformanceConfig(request_timeout=120.0)
        assert config.request_timeout == 120.0

        # Invalid: zero timeout
        with pytest.raises(ValidationError) as exc_info:
            PerformanceConfig(request_timeout=0.0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("request_timeout",)
        assert "greater than 0" in str(errors[0]["msg"])

    def test_max_retries_constraints(self):
        """Test max_retries constraints (0 <= value <= 10)."""
        # Valid values
        config1 = PerformanceConfig(max_retries=0)  # No retries
        assert config1.max_retries == 0

        config2 = PerformanceConfig(max_retries=10)
        assert config2.max_retries == 10

        # Invalid: negative
        with pytest.raises(ValidationError) as exc_info:
            PerformanceConfig(max_retries=-1)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_retries",)
        assert "greater than or equal to 0" in str(errors[0]["msg"])

        # Invalid: exceeds maximum
        with pytest.raises(ValidationError) as exc_info:
            PerformanceConfig(max_retries=11)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_retries",)
        assert "less than or equal to 10" in str(errors[0]["msg"])

    def test_retry_delay_constraints(self):
        """Test retry delay constraints."""
        # Valid delays
        config = PerformanceConfig(retry_base_delay=0.5, retry_max_delay=300.0)
        assert config.retry_base_delay == 0.5
        assert config.retry_max_delay == 300.0

        # Invalid: zero base delay
        with pytest.raises(ValidationError) as exc_info:
            PerformanceConfig(retry_base_delay=0.0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("retry_base_delay",)
        assert "greater than 0" in str(errors[0]["msg"])

        # Invalid: zero max delay
        with pytest.raises(ValidationError) as exc_info:
            PerformanceConfig(retry_max_delay=0.0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("retry_max_delay",)
        assert "greater than 0" in str(errors[0]["msg"])

    def test_memory_constraints(self):
        """Test memory management constraints."""
        # Valid memory usage
        config = PerformanceConfig(max_memory_usage_mb=5000.0)
        assert config.max_memory_usage_mb == 5000.0

        # Invalid: zero memory
        with pytest.raises(ValidationError) as exc_info:
            PerformanceConfig(max_memory_usage_mb=0.0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_memory_usage_mb",)
        assert "greater than 0" in str(errors[0]["msg"])

    def test_gc_threshold_constraints(self):
        """Test gc_threshold constraints (0 < value <= 1)."""
        # Valid values
        config1 = PerformanceConfig(gc_threshold=0.1)
        assert config1.gc_threshold == 0.1

        config2 = PerformanceConfig(gc_threshold=1.0)
        assert config2.gc_threshold == 1.0

        # Invalid: zero
        with pytest.raises(ValidationError) as exc_info:
            PerformanceConfig(gc_threshold=0.0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("gc_threshold",)
        assert "greater than 0" in str(errors[0]["msg"])

        # Invalid: exceeds 1
        with pytest.raises(ValidationError) as exc_info:
            PerformanceConfig(gc_threshold=1.1)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("gc_threshold",)
        assert "less than or equal to 1" in str(errors[0]["msg"])

    def test_rate_limits_validation_structure(self):
        """Test rate limits structure validation."""
        # Valid structure
        valid_limits = {
            "api1": {"max_calls": 100, "time_window": 60},
            "api2": {"max_calls": 50, "time_window": 1},
        }
        config = PerformanceConfig(default_rate_limits=valid_limits)
        assert config.default_rate_limits == valid_limits

        # Invalid: not a dict
        with pytest.raises(ValidationError) as exc_info:
            PerformanceConfig(default_rate_limits={"api1": "invalid"})
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "Input should be a valid dictionary" in str(errors[0]["msg"])

    def test_rate_limits_validation_missing_keys(self):
        """Test rate limits validation for missing required keys."""
        # Missing max_calls
        with pytest.raises(ValidationError) as exc_info:
            PerformanceConfig(default_rate_limits={"api1": {"time_window": 60}})
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "must contain keys" in str(errors[0]["msg"])

        # Missing time_window
        with pytest.raises(ValidationError) as exc_info:
            PerformanceConfig(default_rate_limits={"api1": {"max_calls": 100}})
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "must contain keys" in str(errors[0]["msg"])

    def test_rate_limits_validation_values(self):
        """Test rate limits value validation."""
        # Invalid: zero max_calls
        with pytest.raises(ValidationError) as exc_info:
            PerformanceConfig(
                default_rate_limits={"api1": {"max_calls": 0, "time_window": 60}}
            )
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "max_calls for provider 'api1' must be positive" in str(errors[0]["msg"])

        # Invalid: negative time_window
        with pytest.raises(ValidationError) as exc_info:
            PerformanceConfig(
                default_rate_limits={"api1": {"max_calls": 100, "time_window": -1}}
            )
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "time_window for provider 'api1' must be positive" in str(
            errors[0]["msg"]
        )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            PerformanceConfig(max_concurrent_requests=10, unknown_field="value")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "extra_forbidden"

    def test_model_serialization(self):
        """Test model serialization."""
        config = PerformanceConfig(
            max_concurrent_requests=25,
            request_timeout=45.0,
            max_retries=4,
            default_rate_limits={"custom_api": {"max_calls": 200, "time_window": 30}},
        )

        # Test model_dump
        data = config.model_dump()
        assert data["max_concurrent_requests"] == 25
        assert data["request_timeout"] == 45.0
        assert data["max_retries"] == 4
        assert "custom_api" in data["default_rate_limits"]
        assert data["default_rate_limits"]["custom_api"]["max_calls"] == 200

        # Test model_dump_json
        json_str = config.model_dump_json()
        assert '"max_concurrent_requests":25' in json_str
        assert '"request_timeout":45.0' in json_str
        assert '"custom_api"' in json_str

    def test_model_copy(self):
        """Test model copying with updates."""
        original = PerformanceConfig(max_concurrent_requests=10, gc_threshold=0.8)

        updated = original.model_copy(
            update={
                "max_concurrent_requests": 20,
                "gc_threshold": 0.9,
                "max_memory_usage_mb": 1500.0,
            }
        )

        assert original.max_concurrent_requests == 10
        assert original.gc_threshold == 0.8
        assert original.max_memory_usage_mb == 1000.0  # Default
        assert updated.max_concurrent_requests == 20
        assert updated.gc_threshold == 0.9
        assert updated.max_memory_usage_mb == 1500.0

    def test_performance_profiles(self):
        """Test different performance profile configurations."""
        # Low resource profile
        low_resource = PerformanceConfig(
            max_concurrent_requests=5,
            request_timeout=15.0,
            max_memory_usage_mb=500.0,
            gc_threshold=0.6,
        )
        assert low_resource.max_concurrent_requests == 5
        assert low_resource.max_memory_usage_mb == 500.0

        # High performance profile
        high_perf = PerformanceConfig(
            max_concurrent_requests=100,
            request_timeout=120.0,
            max_memory_usage_mb=8000.0,
            gc_threshold=0.95,
            max_retries=1,  # Fewer retries for speed
        )
        assert high_perf.max_concurrent_requests == 100
        assert high_perf.max_retries == 1

        # Resilient profile
        resilient = PerformanceConfig(
            max_retries=10,
            retry_base_delay=2.0,
            retry_max_delay=300.0,
            request_timeout=60.0,
        )
        assert resilient.max_retries == 10
        assert resilient.retry_max_delay == 300.0
