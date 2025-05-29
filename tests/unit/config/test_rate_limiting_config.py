"""Tests for rate limiting configuration integration."""

import pytest
from src.config.models import PerformanceConfig
from src.config.models import UnifiedConfig
from src.services.utilities.rate_limiter import RateLimitManager


class TestRateLimitingConfiguration:
    """Test rate limiting configuration and integration."""

    def test_performance_config_default_rate_limits(self):
        """Test that PerformanceConfig has correct default rate limits."""
        config = PerformanceConfig()

        # Check that default rate limits are set correctly
        assert "openai" in config.default_rate_limits
        assert "firecrawl" in config.default_rate_limits
        assert "crawl4ai" in config.default_rate_limits
        assert "qdrant" in config.default_rate_limits

        # Check structure of each provider
        for _provider, limits in config.default_rate_limits.items():
            assert "max_calls" in limits
            assert "time_window" in limits
            assert limits["max_calls"] > 0
            assert limits["time_window"] > 0

    def test_performance_config_custom_rate_limits(self):
        """Test custom rate limits validation."""
        custom_limits = {
            "custom_provider": {"max_calls": 50, "time_window": 30},
            "another_provider": {"max_calls": 100, "time_window": 60},
        }

        config = PerformanceConfig(default_rate_limits=custom_limits)
        assert config.default_rate_limits == custom_limits

    def test_performance_config_validation_errors(self):
        """Test that invalid rate limits raise validation errors."""

        # Test missing required keys
        with pytest.raises(ValueError, match="must contain keys"):
            PerformanceConfig(
                default_rate_limits={
                    "invalid": {"max_calls": 10}  # Missing time_window
                }
            )

        # Test negative max_calls
        with pytest.raises(ValueError, match="max_calls.*must be positive"):
            PerformanceConfig(
                default_rate_limits={"invalid": {"max_calls": -1, "time_window": 60}}
            )

        # Test negative time_window
        with pytest.raises(ValueError, match="time_window.*must be positive"):
            PerformanceConfig(
                default_rate_limits={"invalid": {"max_calls": 10, "time_window": -1}}
            )

    def test_rate_limit_manager_with_config(self):
        """Test RateLimitManager initialization with UnifiedConfig."""
        # Create config with custom rate limits
        config = UnifiedConfig()
        config.performance.default_rate_limits = {
            "test_provider": {"max_calls": 123, "time_window": 456}
        }

        # Create manager with config
        manager = RateLimitManager(config)

        # Check that limits are correctly set
        assert manager.default_limits == config.performance.default_rate_limits

        # Test that limiter is created with correct settings
        limiter = manager.get_limiter("test_provider")
        assert limiter.max_calls == 123
        assert limiter.time_window == 456

    def test_rate_limit_manager_requires_config(self):
        """Test that RateLimitManager requires UnifiedConfig."""
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            RateLimitManager()  # Should fail without config

    def test_rate_limit_manager_unknown_provider_error(self):
        """Test that unknown providers raise clear error messages."""
        config = UnifiedConfig()
        config.performance.default_rate_limits = {
            "known_provider": {"max_calls": 100, "time_window": 60}
        }

        manager = RateLimitManager(config)

        # Should raise error for unknown provider
        with pytest.raises(
            ValueError, match="No rate limits configured for provider 'unknown'"
        ):
            manager.get_limiter("unknown")

    def test_unified_config_rate_limits_integration(self):
        """Test that UnifiedConfig properly includes rate limiting configuration."""
        config = UnifiedConfig()

        # Check that performance config is included and has rate limits
        assert hasattr(config.performance, "default_rate_limits")
        assert isinstance(config.performance.default_rate_limits, dict)

        # Check that rate limits can be overridden
        custom_limits = {"custom": {"max_calls": 42, "time_window": 24}}
        config_data = {"performance": {"default_rate_limits": custom_limits}}
        custom_config = UnifiedConfig(**config_data)
        assert custom_config.performance.default_rate_limits == custom_limits

    @pytest.mark.parametrize(
        "provider,expected_max_calls,expected_time_window",
        [
            ("openai", 500, 60),
            ("firecrawl", 100, 60),
            ("crawl4ai", 50, 1),
            ("qdrant", 100, 1),
        ],
    )
    def test_default_rate_limits_values(
        self, provider, expected_max_calls, expected_time_window
    ):
        """Test that default rate limits have expected values for each provider."""
        config = PerformanceConfig()

        assert provider in config.default_rate_limits
        limits = config.default_rate_limits[provider]
        assert limits["max_calls"] == expected_max_calls
        assert limits["time_window"] == expected_time_window

    def test_rate_limit_manager_provider_specific_limits(self):
        """Test that RateLimitManager properly applies provider-specific limits."""
        # Create config with different limits for each provider
        config = UnifiedConfig()
        config.performance.default_rate_limits = {
            "provider_a": {"max_calls": 10, "time_window": 1},
            "provider_b": {"max_calls": 20, "time_window": 2},
            "provider_c": {"max_calls": 30, "time_window": 3},
        }

        manager = RateLimitManager(config)

        # Test each provider gets its specific limits
        limiter_a = manager.get_limiter("provider_a")
        assert limiter_a.max_calls == 10
        assert limiter_a.time_window == 1

        limiter_b = manager.get_limiter("provider_b")
        assert limiter_b.max_calls == 20
        assert limiter_b.time_window == 2

        limiter_c = manager.get_limiter("provider_c")
        assert limiter_c.max_calls == 30
        assert limiter_c.time_window == 3

    def test_rate_limit_manager_configured_provider_works(self):
        """Test that configured providers work correctly."""
        config = UnifiedConfig()
        config.performance.default_rate_limits = {
            "test_provider": {"max_calls": 42, "time_window": 30}
        }

        manager = RateLimitManager(config)

        # Should work for configured provider
        limiter = manager.get_limiter("test_provider")
        assert limiter.max_calls == 42
        assert limiter.time_window == 30

    def test_complex_rate_limits_configuration(self):
        """Test complex rate limiting configuration scenarios."""
        complex_limits = {
            "high_frequency_api": {"max_calls": 1000, "time_window": 1},  # 1000/sec
            "medium_frequency_api": {"max_calls": 100, "time_window": 60},  # 100/min
            "low_frequency_api": {"max_calls": 10, "time_window": 3600},  # 10/hour
            "burst_api": {"max_calls": 5, "time_window": 1},  # 5/sec
        }

        config = PerformanceConfig(default_rate_limits=complex_limits)
        assert config.default_rate_limits == complex_limits

        # Test with RateLimitManager
        unified_config = UnifiedConfig()
        unified_config.performance.default_rate_limits = complex_limits

        manager = RateLimitManager(unified_config)

        # Verify each API gets correct limits
        high_freq_limiter = manager.get_limiter("high_frequency_api")
        assert high_freq_limiter.max_calls == 1000
        assert high_freq_limiter.time_window == 1

        low_freq_limiter = manager.get_limiter("low_frequency_api")
        assert low_freq_limiter.max_calls == 10
        assert low_freq_limiter.time_window == 3600
