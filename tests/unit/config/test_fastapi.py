"""Comprehensive tests for FastAPI production configuration models.

This test file covers the FastAPI-specific configuration models including
CORS, compression, security, tracing, timeout, and performance configurations.
"""

import pytest
from src.config.enums import Environment
from src.config.fastapi import CompressionConfig
from src.config.fastapi import CORSConfig
from src.config.fastapi import FastAPIProductionConfig
from src.config.fastapi import PerformanceConfig
from src.config.fastapi import SecurityConfig
from src.config.fastapi import TimeoutConfig
from src.config.fastapi import TracingConfig
from src.config.fastapi import get_fastapi_config


class TestCORSConfig:
    """Test the CORSConfig model."""

    def test_cors_config_defaults(self):
        """Test CORSConfig with default values."""
        cors = CORSConfig()

        assert cors.enabled is True
        assert "http://localhost:3000" in cors.allow_origins
        assert "http://localhost:8000" in cors.allow_origins
        assert "GET" in cors.allow_methods
        assert "POST" in cors.allow_methods
        assert cors.allow_credentials is True
        assert cors.max_age == 3600

    def test_cors_config_custom_values(self):
        """Test CORSConfig with custom values."""
        cors = CORSConfig(
            enabled=False,
            allow_origins=["https://example.com"],
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
            allow_credentials=False,
            max_age=7200,
        )

        assert cors.enabled is False
        assert cors.allow_origins == ["https://example.com"]
        assert cors.allow_methods == ["GET", "POST"]
        assert cors.allow_headers == ["Content-Type"]
        assert cors.allow_credentials is False
        assert cors.max_age == 7200

    def test_cors_config_serialization(self):
        """Test CORSConfig serialization."""
        cors = CORSConfig(enabled=False, max_age=1800)
        data = cors.model_dump()

        assert isinstance(data, dict)
        assert data["enabled"] is False
        assert data["max_age"] == 1800


class TestCompressionConfig:
    """Test the CompressionConfig model."""

    def test_compression_config_defaults(self):
        """Test CompressionConfig with default values."""
        compression = CompressionConfig()

        assert compression.enabled is True
        assert compression.minimum_size == 1000
        assert compression.compression_level == 6

    def test_compression_config_custom_values(self):
        """Test CompressionConfig with custom values."""
        compression = CompressionConfig(
            enabled=False, minimum_size=500, compression_level=9
        )

        assert compression.enabled is False
        assert compression.minimum_size == 500
        assert compression.compression_level == 9

    def test_compression_config_validation(self):
        """Test CompressionConfig validation."""
        # Valid compression level
        compression = CompressionConfig(compression_level=1)
        assert compression.compression_level == 1

        compression = CompressionConfig(compression_level=9)
        assert compression.compression_level == 9

        # Invalid minimum size
        with pytest.raises(ValueError):
            CompressionConfig(minimum_size=0)

        # Invalid compression level
        with pytest.raises(ValueError):
            CompressionConfig(compression_level=0)

        with pytest.raises(ValueError):
            CompressionConfig(compression_level=10)


class TestSecurityConfig:
    """Test the SecurityConfig model."""

    def test_security_config_defaults(self):
        """Test SecurityConfig with default values."""
        security = SecurityConfig()

        assert security.enabled is True
        assert security.x_frame_options == "DENY"
        assert security.x_content_type_options == "nosniff"
        assert security.x_xss_protection == "1; mode=block"
        assert "max-age=31536000" in security.strict_transport_security
        assert security.content_security_policy == "default-src 'self'"
        assert security.enable_rate_limiting is True
        assert security.rate_limit_requests == 100
        assert security.rate_limit_window == 60

    def test_security_config_custom_values(self):
        """Test SecurityConfig with custom values."""
        security = SecurityConfig(
            enabled=False,
            x_frame_options="SAMEORIGIN",
            strict_transport_security=None,
            enable_rate_limiting=False,
            rate_limit_requests=50,
        )

        assert security.enabled is False
        assert security.x_frame_options == "SAMEORIGIN"
        assert security.strict_transport_security is None
        assert security.enable_rate_limiting is False
        assert security.rate_limit_requests == 50

    def test_security_config_validation(self):
        """Test SecurityConfig validation."""
        # Valid rate limit settings
        security = SecurityConfig(rate_limit_requests=1, rate_limit_window=1)
        assert security.rate_limit_requests == 1
        assert security.rate_limit_window == 1

        # Invalid rate limit settings
        with pytest.raises(ValueError):
            SecurityConfig(rate_limit_requests=0)

        with pytest.raises(ValueError):
            SecurityConfig(rate_limit_window=0)


class TestTracingConfig:
    """Test the TracingConfig model."""

    def test_tracing_config_defaults(self):
        """Test TracingConfig with default values."""
        tracing = TracingConfig()

        assert tracing.enabled is True
        assert tracing.correlation_id_header == "X-Correlation-ID"
        assert tracing.generate_correlation_id is True
        assert tracing.log_requests is True
        assert tracing.log_responses is False

    def test_tracing_config_custom_values(self):
        """Test TracingConfig with custom values."""
        tracing = TracingConfig(
            enabled=False,
            correlation_id_header="X-Request-ID",
            generate_correlation_id=False,
            log_responses=True,
        )

        assert tracing.enabled is False
        assert tracing.correlation_id_header == "X-Request-ID"
        assert tracing.generate_correlation_id is False
        assert tracing.log_responses is True


class TestTimeoutConfig:
    """Test the TimeoutConfig model."""

    def test_timeout_config_defaults(self):
        """Test TimeoutConfig with default values."""
        timeout = TimeoutConfig()

        assert timeout.enabled is True
        assert timeout.request_timeout == 30.0
        assert timeout.enable_circuit_breaker is True
        assert timeout.failure_threshold == 5
        assert timeout.recovery_timeout == 60.0
        assert timeout.half_open_max_calls == 3

    def test_timeout_config_custom_values(self):
        """Test TimeoutConfig with custom values."""
        timeout = TimeoutConfig(
            enabled=False,
            request_timeout=10.0,
            enable_circuit_breaker=False,
            failure_threshold=3,
        )

        assert timeout.enabled is False
        assert timeout.request_timeout == 10.0
        assert timeout.enable_circuit_breaker is False
        assert timeout.failure_threshold == 3

    def test_timeout_config_validation(self):
        """Test TimeoutConfig validation."""
        # Valid timeout values
        timeout = TimeoutConfig(request_timeout=0.1, recovery_timeout=1.0)
        assert timeout.request_timeout == 0.1
        assert timeout.recovery_timeout == 1.0

        # Invalid timeout values
        with pytest.raises(ValueError):
            TimeoutConfig(request_timeout=0)

        with pytest.raises(ValueError):
            TimeoutConfig(recovery_timeout=0)

        with pytest.raises(ValueError):
            TimeoutConfig(failure_threshold=0)

        with pytest.raises(ValueError):
            TimeoutConfig(half_open_max_calls=0)


class TestPerformanceConfig:
    """Test the PerformanceConfig model."""

    def test_performance_config_defaults(self):
        """Test PerformanceConfig with default values."""
        performance = PerformanceConfig()

        assert performance.enabled is True
        assert performance.track_response_time is True
        assert performance.track_memory_usage is False
        assert performance.slow_request_threshold == 1.0

    def test_performance_config_custom_values(self):
        """Test PerformanceConfig with custom values."""
        performance = PerformanceConfig(
            enabled=False, track_memory_usage=True, slow_request_threshold=0.5
        )

        assert performance.enabled is False
        assert performance.track_memory_usage is True
        assert performance.slow_request_threshold == 0.5

    def test_performance_config_validation(self):
        """Test PerformanceConfig validation."""
        # Valid threshold
        performance = PerformanceConfig(slow_request_threshold=0.1)
        assert performance.slow_request_threshold == 0.1

        # Invalid threshold
        with pytest.raises(ValueError):
            PerformanceConfig(slow_request_threshold=0)


class TestFastAPIProductionConfig:
    """Test the FastAPIProductionConfig model."""

    def test_fastapi_config_defaults(self):
        """Test FastAPIProductionConfig with default values."""
        config = FastAPIProductionConfig()

        assert config.environment == Environment.DEVELOPMENT
        assert config.debug is False
        assert config.server_name == "AI Docs Vector DB"
        assert config.version == "1.0.0"
        assert config.workers == 4
        assert config.max_requests == 1000
        assert config.max_requests_jitter == 100

        # Test that sub-configs are properly instantiated
        assert isinstance(config.cors, CORSConfig)
        assert isinstance(config.compression, CompressionConfig)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.tracing, TracingConfig)
        assert isinstance(config.timeout, TimeoutConfig)
        assert isinstance(config.performance, PerformanceConfig)

    def test_fastapi_config_custom_values(self):
        """Test FastAPIProductionConfig with custom values."""
        config = FastAPIProductionConfig(
            environment=Environment.PRODUCTION,
            debug=True,
            server_name="Custom Server",
            workers=8,
            max_requests=2000,
        )

        assert config.environment == Environment.PRODUCTION
        assert config.debug is True
        assert config.server_name == "Custom Server"
        assert config.workers == 8
        assert config.max_requests == 2000

    def test_fastapi_config_environment_specific_cors(self):
        """Test environment-specific CORS configuration."""
        # Development environment
        dev_config = FastAPIProductionConfig(environment=Environment.DEVELOPMENT)
        dev_origins = dev_config.get_environment_specific_cors()
        assert "http://localhost:3000" in dev_origins
        assert "http://127.0.0.1:3000" in dev_origins

        # Testing environment
        test_config = FastAPIProductionConfig(environment=Environment.TESTING)
        test_origins = test_config.get_environment_specific_cors()
        assert "http://testserver" in test_origins

        # Production environment
        prod_config = FastAPIProductionConfig(environment=Environment.PRODUCTION)
        prod_origins = prod_config.get_environment_specific_cors()
        assert "https://yourdomain.com" in prod_origins

    def test_fastapi_config_is_production(self):
        """Test is_production method."""
        dev_config = FastAPIProductionConfig(environment=Environment.DEVELOPMENT)
        assert dev_config.is_production() is False

        prod_config = FastAPIProductionConfig(environment=Environment.PRODUCTION)
        assert prod_config.is_production() is True

    def test_fastapi_config_get_security_headers(self):
        """Test get_security_headers method."""
        # Production config with security enabled
        prod_config = FastAPIProductionConfig(environment=Environment.PRODUCTION)
        headers = prod_config.get_security_headers()

        assert "X-Frame-Options" in headers
        assert "X-Content-Type-Options" in headers
        assert "X-XSS-Protection" in headers
        assert "Strict-Transport-Security" in headers
        assert "Content-Security-Policy" in headers

        # Development config (no HSTS)
        dev_config = FastAPIProductionConfig(environment=Environment.DEVELOPMENT)
        dev_headers = dev_config.get_security_headers()
        assert "Strict-Transport-Security" not in dev_headers

        # Security disabled
        disabled_config = FastAPIProductionConfig()
        disabled_config.security.enabled = False
        disabled_headers = disabled_config.get_security_headers()
        assert len(disabled_headers) == 0

    def test_fastapi_config_workers_validation(self):
        """Test worker count validation."""
        import os

        cpu_count = os.cpu_count() or 1

        # Valid worker count
        config = FastAPIProductionConfig(workers=cpu_count)
        assert config.workers == cpu_count

        # Worker count at limit
        config = FastAPIProductionConfig(workers=cpu_count * 2)
        assert config.workers == cpu_count * 2

        # Worker count exceeding limit
        with pytest.raises(ValueError, match="Worker count"):
            FastAPIProductionConfig(workers=cpu_count * 2 + 1)

    def test_fastapi_config_with_nested_configs(self):
        """Test FastAPIProductionConfig with custom nested configurations."""
        config = FastAPIProductionConfig(
            cors=CORSConfig(enabled=False),
            security=SecurityConfig(enable_rate_limiting=False),
            performance=PerformanceConfig(track_memory_usage=True),
        )

        assert config.cors.enabled is False
        assert config.security.enable_rate_limiting is False
        assert config.performance.track_memory_usage is True

    def test_fastapi_config_serialization(self):
        """Test FastAPIProductionConfig serialization."""
        config = FastAPIProductionConfig(environment=Environment.PRODUCTION)
        data = config.model_dump()

        assert isinstance(data, dict)
        assert data["environment"] == "production"
        assert "cors" in data
        assert "security" in data
        assert "performance" in data

    def test_fastapi_config_security_headers_with_none_values(self):
        """Test security headers when some values are None."""
        config = FastAPIProductionConfig()
        config.security.strict_transport_security = None
        config.security.content_security_policy = None

        headers = config.get_security_headers()

        assert "X-Frame-Options" in headers
        assert "Strict-Transport-Security" not in headers
        assert "Content-Security-Policy" not in headers


class TestGetFastAPIConfig:
    """Test the get_fastapi_config function."""

    def test_get_fastapi_config(self):
        """Test get_fastapi_config function."""
        config = get_fastapi_config()

        assert isinstance(config, FastAPIProductionConfig)
        assert config.environment == Environment.DEVELOPMENT
        assert config.server_name == "AI Docs Vector DB"


class TestFastAPIConfigIntegration:
    """Integration tests for FastAPI configuration."""

    def test_production_ready_configuration(self):
        """Test that production configuration has security settings enabled."""
        config = FastAPIProductionConfig(environment=Environment.PRODUCTION)

        # Security should be enabled with appropriate settings
        assert config.security.enabled is True
        assert config.security.enable_rate_limiting is True

        # Performance monitoring should be enabled
        assert config.performance.enabled is True
        assert config.performance.track_response_time is True

        # Circuit breaker should be enabled
        assert config.timeout.enable_circuit_breaker is True

    def test_development_friendly_configuration(self):
        """Test that development configuration has debug-friendly settings."""
        config = FastAPIProductionConfig(environment=Environment.DEVELOPMENT)

        # Should allow CORS from localhost
        cors_origins = config.get_environment_specific_cors()
        assert any("localhost" in origin for origin in cors_origins)

        # Should not enforce HSTS in development
        headers = config.get_security_headers()
        assert "Strict-Transport-Security" not in headers

    def test_complete_configuration_coverage(self):
        """Test that all configuration sections are properly initialized."""
        config = FastAPIProductionConfig()

        # All sections should be accessible and have expected types
        assert hasattr(config, "cors")
        assert hasattr(config, "compression")
        assert hasattr(config, "security")
        assert hasattr(config, "tracing")
        assert hasattr(config, "timeout")
        assert hasattr(config, "performance")

        # Each section should be properly configured
        assert config.cors.enabled is True
        assert config.compression.enabled is True
        assert config.security.enabled is True
        assert config.tracing.enabled is True
        assert config.timeout.enabled is True
        assert config.performance.enabled is True
