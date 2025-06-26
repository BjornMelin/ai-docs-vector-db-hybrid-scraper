"""Tests for circuit breaker factory."""

from unittest.mock import Mock, patch

import pytest

from src.config.core import CircuitBreakerConfig, Config
from src.services.functional.circuit_breaker import CircuitBreaker
from src.services.functional.circuit_breaker_factory import (
    CircuitBreakerFactory,
    create_crawl4ai_circuit_breaker,
    create_firecrawl_circuit_breaker,
    create_openai_circuit_breaker,
    create_qdrant_circuit_breaker,
    create_redis_circuit_breaker,
    get_circuit_breaker_factory,
    reset_circuit_breaker_factory,
)
from src.services.functional.enhanced_circuit_breaker import EnhancedCircuitBreaker


class TestCircuitBreakerFactory:
    """Test circuit breaker factory functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock(spec=Config)

        # Mock circuit breaker config
        cb_config = Mock(spec=CircuitBreakerConfig)
        cb_config.use_enhanced_circuit_breaker = True
        cb_config.failure_threshold = 5
        cb_config.recovery_timeout = 60.0
        cb_config.enable_metrics_collection = True
        cb_config.enable_detailed_metrics = True
        cb_config.enable_fallback_mechanisms = True
        cb_config.service_overrides = {
            "openai": {
                "failure_threshold": 3,
                "recovery_timeout": 30.0,
                "enable_metrics": True,
                "enable_fallback": True,
            },
            "qdrant": {
                "failure_threshold": 3,
                "recovery_timeout": 15.0,
                "enable_metrics": True,
                "enable_fallback": False,
            },
        }

        config.circuit_breaker = cb_config
        return config

    @pytest.fixture
    def factory(self, mock_config):
        """Create a circuit breaker factory."""
        return CircuitBreakerFactory(mock_config)

    def test_factory_initialization(self, mock_config):
        """Test factory initialization."""
        factory = CircuitBreakerFactory(mock_config)

        assert factory.config == mock_config
        assert len(factory._breaker_registry) == 0

    def test_create_enhanced_circuit_breaker(self, factory):
        """Test creating enhanced circuit breaker."""
        breaker = factory.create_service_circuit_breaker("openai")

        assert isinstance(breaker, EnhancedCircuitBreaker)
        assert breaker.config.service_name == "openai"
        assert breaker.config.failure_threshold == 3
        assert breaker.config.recovery_timeout == 30.0

    def test_create_legacy_circuit_breaker(self, mock_config):
        """Test creating legacy circuit breaker."""
        # Disable enhanced circuit breaker
        mock_config.circuit_breaker.use_enhanced_circuit_breaker = False

        factory = CircuitBreakerFactory(mock_config)
        breaker = factory.create_service_circuit_breaker("openai")

        assert isinstance(breaker, CircuitBreaker)

    def test_service_circuit_breaker_registry(self, factory):
        """Test that created circuit breakers are registered."""
        breaker1 = factory.create_service_circuit_breaker("openai")
        breaker2 = factory.create_service_circuit_breaker("qdrant")

        # Check registry
        assert len(factory._breaker_registry) == 2
        assert factory._breaker_registry["openai"] is breaker1
        assert factory._breaker_registry["qdrant"] is breaker2

    def test_get_existing_circuit_breaker(self, factory):
        """Test getting existing circuit breaker."""
        breaker1 = factory.create_service_circuit_breaker("openai")
        breaker2 = factory.get_service_circuit_breaker("openai")

        # Should return the same instance
        assert breaker1 is breaker2

    def test_get_nonexistent_circuit_breaker(self, factory):
        """Test getting non-existent circuit breaker."""
        breaker = factory.get_service_circuit_breaker("nonexistent")
        assert breaker is None

    def test_service_config_overrides(self, factory):
        """Test service-specific configuration overrides."""
        # OpenAI should use service-specific config
        openai_breaker = factory.create_service_circuit_breaker("openai")
        assert openai_breaker.config.failure_threshold == 3
        assert openai_breaker.config.recovery_timeout == 30.0

        # Qdrant should use different service-specific config
        qdrant_breaker = factory.create_service_circuit_breaker("qdrant")
        assert qdrant_breaker.config.failure_threshold == 3
        assert qdrant_breaker.config.recovery_timeout == 15.0

    def test_unknown_service_uses_defaults(self, factory):
        """Test that unknown service uses default configuration."""
        breaker = factory.create_service_circuit_breaker("unknown_service")

        # Should use defaults from config
        assert breaker.config.failure_threshold == 5  # Default
        assert breaker.config.recovery_timeout == 60.0  # Default

    def test_config_overrides_parameter(self, factory):
        """Test additional config overrides parameter."""
        overrides = {"failure_threshold": 10, "recovery_timeout": 120.0}
        breaker = factory.create_service_circuit_breaker("test", overrides)

        assert breaker.config.failure_threshold == 10
        assert breaker.config.recovery_timeout == 120.0

    def test_get_all_circuit_breakers(self, factory):
        """Test getting all circuit breakers."""
        breaker1 = factory.create_service_circuit_breaker("service1")
        breaker2 = factory.create_service_circuit_breaker("service2")

        all_breakers = factory.get_all_circuit_breakers()

        assert len(all_breakers) == 2
        assert all_breakers["service1"] is breaker1
        assert all_breakers["service2"] is breaker2

    def test_get_circuit_breaker_metrics(self, factory):
        """Test getting metrics for all circuit breakers."""
        factory.create_service_circuit_breaker("service1")
        factory.create_service_circuit_breaker("service2")

        metrics = factory.get_circuit_breaker_metrics()

        assert len(metrics) == 2
        assert "service1" in metrics
        assert "service2" in metrics

    def test_get_circuit_breaker_metrics_with_error(self, factory):
        """Test getting metrics when a breaker fails."""
        breaker = factory.create_service_circuit_breaker("service1")

        # Mock the get_metrics method to raise an exception
        breaker.get_metrics = Mock(side_effect=Exception("Metrics error"))

        metrics = factory.get_circuit_breaker_metrics()

        assert "service1" in metrics
        assert "error" in metrics["service1"]
        assert metrics["service1"]["error"] == "Metrics error"

    def test_reset_all_circuit_breakers(self, factory):
        """Test resetting all circuit breakers."""
        breaker1 = factory.create_service_circuit_breaker("service1")
        breaker2 = factory.create_service_circuit_breaker("service2")

        # Mock reset methods
        breaker1.reset = Mock()
        breaker2.reset = Mock()

        factory.reset_all_circuit_breakers()

        breaker1.reset.assert_called_once()
        breaker2.reset.assert_called_once()

    def test_reset_all_circuit_breakers_with_error(self, factory):
        """Test resetting all circuit breakers when one fails."""
        breaker1 = factory.create_service_circuit_breaker("service1")
        breaker2 = factory.create_service_circuit_breaker("service2")

        # Mock reset methods, one fails
        breaker1.reset = Mock(side_effect=Exception("Reset error"))
        breaker2.reset = Mock()

        # Should not raise exception
        factory.reset_all_circuit_breakers()

        breaker1.reset.assert_called_once()
        breaker2.reset.assert_called_once()

    def test_create_circuit_breaker_for_external_service(self, factory):
        """Test creating circuit breaker for external service with URL."""
        # Test OpenAI URL pattern
        breaker = factory.create_circuit_breaker_for_external_service(
            "external_openai", "https://api.openai.com/v1/embeddings"
        )

        assert breaker.config.service_name == "external_openai"
        assert breaker.config.failure_threshold == 3  # OpenAI defaults
        assert breaker.config.recovery_timeout == 30.0

    def test_get_defaults_for_url_patterns(self, factory):
        """Test URL pattern detection for defaults."""
        # Test different URL patterns
        test_cases = [
            ("https://api.openai.com/v1/embeddings", 3, 30.0),
            ("https://qdrant.example.com:6333", 3, 15.0),
            ("https://api.firecrawl.dev/v1/scrape", 5, 60.0),
            ("redis://localhost:6379", 2, 10.0),
            ("https://unknown-service.com/api", 5, 60.0),  # Default
        ]

        for url, expected_threshold, expected_timeout in test_cases:
            defaults = factory._get_defaults_for_url(url)
            assert defaults["failure_threshold"] == expected_threshold
            assert defaults["recovery_timeout"] == expected_timeout


class TestCircuitBreakerFactoryGlobal:
    """Test global circuit breaker factory functions."""

    def teardown_method(self):
        """Reset factory after each test."""
        reset_circuit_breaker_factory()

    @patch("src.config.core.get_config")
    def test_get_circuit_breaker_factory_singleton(self, mock_get_config):
        """Test global factory singleton behavior."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config

        factory1 = get_circuit_breaker_factory()
        factory2 = get_circuit_breaker_factory()

        # Should return the same instance
        assert factory1 is factory2
        mock_get_config.assert_called_once()

    def test_get_circuit_breaker_factory_with_config(self):
        """Test getting factory with explicit config."""
        mock_config = Mock()
        mock_config.circuit_breaker = Mock()
        mock_config.circuit_breaker.use_enhanced_circuit_breaker = True

        factory = get_circuit_breaker_factory(mock_config)

        assert isinstance(factory, CircuitBreakerFactory)
        assert factory.config is mock_config

    def test_reset_circuit_breaker_factory(self):
        """Test resetting global factory."""
        mock_config = Mock()
        mock_config.circuit_breaker = Mock()
        mock_config.circuit_breaker.use_enhanced_circuit_breaker = True

        factory1 = get_circuit_breaker_factory(mock_config)
        reset_circuit_breaker_factory()
        factory2 = get_circuit_breaker_factory(mock_config)

        # Should be different instances after reset
        assert factory1 is not factory2

    @patch(
        "src.services.functional.circuit_breaker_factory.get_circuit_breaker_factory"
    )
    def test_convenience_functions(self, mock_get_factory):
        """Test convenience functions for specific services."""
        mock_factory = Mock()
        mock_breaker = Mock()
        mock_factory.create_service_circuit_breaker.return_value = mock_breaker
        mock_get_factory.return_value = mock_factory

        # Test all convenience functions
        services = [
            ("openai", create_openai_circuit_breaker),
            ("firecrawl", create_firecrawl_circuit_breaker),
            ("qdrant", create_qdrant_circuit_breaker),
            ("redis", create_redis_circuit_breaker),
            ("crawl4ai", create_crawl4ai_circuit_breaker),
        ]

        for service_name, create_func in services:
            result = create_func()
            assert result is mock_breaker
            mock_factory.create_service_circuit_breaker.assert_called_with(service_name)

    @patch(
        "src.services.functional.circuit_breaker_factory.get_circuit_breaker_factory"
    )
    def test_convenience_functions_with_config(self, mock_get_factory):
        """Test convenience functions with explicit config."""
        mock_factory = Mock()
        mock_breaker = Mock()
        mock_factory.create_service_circuit_breaker.return_value = mock_breaker
        mock_get_factory.return_value = mock_factory

        mock_config = Mock()

        result = create_openai_circuit_breaker(mock_config)

        assert result is mock_breaker
        mock_get_factory.assert_called_with(mock_config)
        mock_factory.create_service_circuit_breaker.assert_called_with("openai")


class TestCircuitBreakerFactoryIntegration:
    """Test circuit breaker factory integration scenarios."""

    @pytest.fixture
    def real_config(self):
        """Create a real configuration for integration tests."""
        from src.config.core import CircuitBreakerConfig as ConfigCB

        config = Mock()
        config.circuit_breaker = ConfigCB(
            use_enhanced_circuit_breaker=True,
            failure_threshold=5,
            recovery_timeout=60.0,
            enable_metrics_collection=True,
            enable_detailed_metrics=True,
            enable_fallback_mechanisms=True,
        )
        return config

    def test_factory_with_real_config(self, real_config):
        """Test factory with real configuration structure."""
        factory = CircuitBreakerFactory(real_config)

        breaker = factory.create_service_circuit_breaker("test_service")

        assert isinstance(breaker, EnhancedCircuitBreaker)
        assert breaker.config.service_name == "test_service"

    def test_multiple_services_isolation(self, real_config):
        """Test that multiple services created by factory are isolated."""
        factory = CircuitBreakerFactory(real_config)

        breaker1 = factory.create_service_circuit_breaker("service1")
        breaker2 = factory.create_service_circuit_breaker("service2")

        assert breaker1 is not breaker2
        assert breaker1.config.service_name == "service1"
        assert breaker2.config.service_name == "service2"

    def test_factory_metrics_aggregation(self, real_config):
        """Test factory metrics aggregation."""
        factory = CircuitBreakerFactory(real_config)

        # Create multiple breakers
        factory.create_service_circuit_breaker("service1")
        factory.create_service_circuit_breaker("service2")
        factory.create_service_circuit_breaker("service3")

        all_breakers = factory.get_all_circuit_breakers()
        metrics = factory.get_circuit_breaker_metrics()

        assert len(all_breakers) == 3
        assert len(metrics) == 3
        assert all(
            service in metrics for service in ["service1", "service2", "service3"]
        )
