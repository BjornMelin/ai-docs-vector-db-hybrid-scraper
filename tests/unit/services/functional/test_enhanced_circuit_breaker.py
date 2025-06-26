"""Tests for enhanced circuit breaker implementation."""

import asyncio
from unittest.mock import Mock

import pytest

from src.services.functional.circuit_breaker import CircuitBreakerError
from src.services.functional.enhanced_circuit_breaker import (
    EnhancedCircuitBreaker,
    EnhancedCircuitBreakerConfig,
    EnhancedCircuitBreakerState,
    create_enhanced_circuit_breaker,
    enhanced_circuit_breaker,
    get_all_circuit_breaker_metrics,
    get_circuit_breaker_registry,
    register_circuit_breaker,
)


class TestEnhancedCircuitBreakerConfig:
    """Test enhanced circuit breaker configuration."""

    def test_from_service_config_openai(self):
        """Test service configuration for OpenAI."""
        config = EnhancedCircuitBreakerConfig.from_service_config("openai")

        assert config.service_name == "openai"
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 30
        assert config.enable_metrics is True

    def test_from_service_config_unknown_service(self):
        """Test service configuration for unknown service."""
        config = EnhancedCircuitBreakerConfig.from_service_config("unknown")

        assert config.service_name == "unknown"
        # Should use default values
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60

    def test_from_service_config_with_overrides(self):
        """Test service configuration with overrides."""
        overrides = {"failure_threshold": 10, "recovery_timeout": 120}
        config = EnhancedCircuitBreakerConfig.from_service_config("openai", overrides)

        assert config.service_name == "openai"
        assert config.failure_threshold == 10  # Override applied
        assert config.recovery_timeout == 120  # Override applied

    def test_simple_mode_config(self):
        """Test simple mode configuration."""
        config = EnhancedCircuitBreakerConfig.simple_mode("test")

        assert config.service_name == "test"
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 30
        assert config.enable_metrics is False
        assert config.enable_fallback is False
        assert config.collect_detailed_metrics is False

    def test_enterprise_mode_config(self):
        """Test enterprise mode configuration."""
        config = EnhancedCircuitBreakerConfig.enterprise_mode("test")

        assert config.service_name == "test"
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60
        assert config.enable_metrics is True
        assert config.enable_fallback is True
        assert config.collect_detailed_metrics is True


class TestEnhancedCircuitBreaker:
    """Test enhanced circuit breaker functionality."""

    @pytest.fixture
    def simple_config(self):
        """Create a simple circuit breaker configuration."""
        return EnhancedCircuitBreakerConfig.simple_mode("test")

    @pytest.fixture
    def enterprise_config(self):
        """Create an enterprise circuit breaker configuration."""
        return EnhancedCircuitBreakerConfig.enterprise_mode("test")

    @pytest.fixture
    def simple_breaker(self, simple_config):
        """Create a simple enhanced circuit breaker."""
        return EnhancedCircuitBreaker(simple_config)

    @pytest.fixture
    def enterprise_breaker(self, enterprise_config):
        """Create an enterprise enhanced circuit breaker."""
        return EnhancedCircuitBreaker(enterprise_config)

    @pytest.mark.asyncio
    async def test_closed_state_success(self, simple_breaker):
        """Test successful operation in closed state."""

        async def success_func():
            return "success"

        result = await simple_breaker.call(success_func)
        assert result == "success"
        assert simple_breaker.state == EnhancedCircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_closed_state_single_failure(self, simple_breaker):
        """Test single failure in closed state."""

        async def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await simple_breaker.call(failing_func)

        assert simple_breaker.state == EnhancedCircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self, simple_breaker):
        """Test circuit opens after failure threshold."""

        async def failing_func():
            raise ValueError("Test error")

        # Fail until threshold is reached
        for _ in range(simple_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await simple_breaker.call(failing_func)

        # Circuit should now be open
        assert simple_breaker.state == EnhancedCircuitBreakerState.OPEN

        # Next call should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            await simple_breaker.call(failing_func)

    @pytest.mark.asyncio
    async def test_metrics_collection_disabled(self, simple_breaker):
        """Test that metrics are not collected when disabled."""

        async def success_func():
            return "success"

        await simple_breaker.call(success_func)

        # Metrics should be minimal when disabled
        metrics = simple_breaker.get_metrics()
        assert metrics["service_name"] == "test"
        # Should still have basic metrics
        assert "state" in metrics

    @pytest.mark.asyncio
    async def test_metrics_collection_enabled(self, enterprise_breaker):
        """Test that detailed metrics are collected when enabled."""

        async def success_func():
            return "success"

        await enterprise_breaker.call(success_func)

        metrics = enterprise_breaker.get_metrics()
        assert metrics["service_name"] == "test"
        assert metrics["total_requests"] == 1
        assert metrics["successful_requests"] == 1
        assert metrics["failed_requests"] == 0
        assert metrics["success_rate"] == 1.0
        assert metrics["failure_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_failure_metrics_collection(self, enterprise_breaker):
        """Test failure metrics collection."""

        async def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await enterprise_breaker.call(failing_func)

        metrics = enterprise_breaker.get_metrics()
        assert metrics["total_requests"] == 1
        assert metrics["successful_requests"] == 0
        assert metrics["failed_requests"] == 1
        assert metrics["success_rate"] == 0.0
        assert metrics["failure_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_circuit_recovery(self):
        """Test circuit recovery after timeout."""
        # Create a breaker with a very short recovery timeout for testing
        config = EnhancedCircuitBreakerConfig.simple_mode("test_recovery")
        config.recovery_timeout = 1  # 1 second for quick testing
        breaker = EnhancedCircuitBreaker(config)

        async def failing_func():
            raise ValueError("Test error")

        async def success_func():
            return "success"

        # Open the circuit
        for _ in range(config.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(failing_func)

        assert breaker.state == EnhancedCircuitBreakerState.OPEN

        # Wait for recovery timeout + small buffer
        import asyncio

        await asyncio.sleep(1.1)

        # After recovery timeout, should allow test calls
        result = await breaker.call(success_func)
        assert result == "success"

    def test_get_metrics_structure(self, enterprise_breaker):
        """Test metrics structure and content."""
        metrics = enterprise_breaker.get_metrics()

        required_fields = [
            "service_name",
            "state",
            "total_requests",
            "successful_requests",
            "failed_requests",
            "blocked_requests",
            "failure_rate",
            "success_rate",
            "average_response_time",
            "circuit_open_count",
            "circuit_close_count",
            "uptime_seconds",
            "last_failure_time",
            "last_success_time",
        ]

        for field in required_fields:
            assert field in metrics

    def test_reset_circuit_breaker(self, simple_breaker):
        """Test resetting circuit breaker."""
        # This test verifies the reset method works without errors
        simple_breaker.reset()
        assert simple_breaker.state == EnhancedCircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_non_async_function_support(self, simple_breaker):
        """Test support for non-async functions."""

        def sync_func():
            return "sync_result"

        # Should handle sync functions gracefully
        # Note: The actual circuitbreaker library handles this
        result = await simple_breaker.call(sync_func)
        assert result == "sync_result"


class TestEnhancedCircuitBreakerDecorator:
    """Test enhanced circuit breaker decorator."""

    @pytest.mark.asyncio
    async def test_decorator_basic_usage(self):
        """Test basic decorator usage."""
        config = EnhancedCircuitBreakerConfig.simple_mode("test")

        @enhanced_circuit_breaker(config)
        async def test_function():
            return "decorated_result"

        result = await test_function()
        assert result == "decorated_result"

    @pytest.mark.asyncio
    async def test_decorator_with_failures(self):
        """Test decorator with failing function."""
        config = EnhancedCircuitBreakerConfig.simple_mode("test")

        @enhanced_circuit_breaker(config)
        async def failing_function():
            raise ValueError("Decorator test error")

        with pytest.raises(ValueError):
            await failing_function()

    @pytest.mark.asyncio
    async def test_decorator_circuit_breaker_access(self):
        """Test accessing circuit breaker from decorated function."""
        config = EnhancedCircuitBreakerConfig.enterprise_mode("test")

        @enhanced_circuit_breaker(config)
        async def test_function():
            return "result"

        await test_function()

        # Check that circuit breaker is attached
        assert hasattr(test_function, "_circuit_breaker")
        breaker = test_function._circuit_breaker
        assert isinstance(breaker, EnhancedCircuitBreaker)

        metrics = breaker.get_metrics()
        assert metrics["total_requests"] == 1


class TestEnhancedCircuitBreakerFactory:
    """Test enhanced circuit breaker factory functions."""

    def test_create_enhanced_circuit_breaker_simple(self):
        """Test creating simple enhanced circuit breaker."""
        breaker = create_enhanced_circuit_breaker("test", "simple")

        assert isinstance(breaker, EnhancedCircuitBreaker)
        assert breaker.config.service_name == "test"
        assert breaker.config.enable_metrics is False

    def test_create_enhanced_circuit_breaker_enterprise(self):
        """Test creating enterprise enhanced circuit breaker."""
        breaker = create_enhanced_circuit_breaker("test", "enterprise")

        assert isinstance(breaker, EnhancedCircuitBreaker)
        assert breaker.config.service_name == "test"
        assert breaker.config.enable_metrics is True

    def test_create_enhanced_circuit_breaker_with_overrides(self):
        """Test creating enhanced circuit breaker with overrides."""
        overrides = {"failure_threshold": 10}
        breaker = create_enhanced_circuit_breaker("test", "simple", overrides)

        assert breaker.config.failure_threshold == 10


class TestEnhancedCircuitBreakerRegistry:
    """Test enhanced circuit breaker registry functionality."""

    def test_register_and_get_circuit_breaker(self):
        """Test registering and retrieving circuit breaker."""
        config = EnhancedCircuitBreakerConfig.simple_mode("registry_test")
        breaker = EnhancedCircuitBreaker(config)

        register_circuit_breaker("registry_test", breaker)

        registry = get_circuit_breaker_registry()
        assert "registry_test" in registry
        assert registry["registry_test"] is breaker

    def test_get_all_circuit_breaker_metrics(self):
        """Test getting metrics for all registered circuit breakers."""
        # Clear registry first
        registry = get_circuit_breaker_registry()
        registry.clear()

        # Register test breakers
        config1 = EnhancedCircuitBreakerConfig.enterprise_mode("test1")
        breaker1 = EnhancedCircuitBreaker(config1)
        register_circuit_breaker("test1", breaker1)

        config2 = EnhancedCircuitBreakerConfig.enterprise_mode("test2")
        breaker2 = EnhancedCircuitBreaker(config2)
        register_circuit_breaker("test2", breaker2)

        metrics = get_all_circuit_breaker_metrics()
        assert "test1" in metrics
        assert "test2" in metrics
        assert metrics["test1"]["service_name"] == "test1"
        assert metrics["test2"]["service_name"] == "test2"


class TestEnhancedCircuitBreakerIntegration:
    """Test enhanced circuit breaker integration scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_services_isolation(self):
        """Test that multiple services are isolated from each other."""
        config1 = EnhancedCircuitBreakerConfig.simple_mode("service1")
        config2 = EnhancedCircuitBreakerConfig.simple_mode("service2")

        breaker1 = EnhancedCircuitBreaker(config1)
        breaker2 = EnhancedCircuitBreaker(config2)

        async def failing_func():
            raise ValueError("Test error")

        async def success_func():
            return "success"

        # Fail service1 to open its circuit
        for _ in range(breaker1.config.failure_threshold):
            with pytest.raises(ValueError):
                await breaker1.call(failing_func)

        assert breaker1.state == EnhancedCircuitBreakerState.OPEN
        assert breaker2.state == EnhancedCircuitBreakerState.CLOSED

        # Service2 should still work
        result = await breaker2.call(success_func)
        assert result == "success"
        assert breaker2.state == EnhancedCircuitBreakerState.CLOSED

        # Service1 should be blocked
        with pytest.raises(CircuitBreakerError):
            await breaker1.call(success_func)

    @pytest.mark.asyncio
    async def test_concurrent_calls(self):
        """Test concurrent calls to circuit breaker."""
        config = EnhancedCircuitBreakerConfig.enterprise_mode("concurrent_test")
        breaker = EnhancedCircuitBreaker(config)

        async def slow_success_func():
            await asyncio.sleep(0.01)  # Small delay
            return "success"

        # Run multiple concurrent calls
        tasks = [breaker.call(slow_success_func) for _ in range(10)]
        results = await asyncio.gather(*tasks)

        assert all(result == "success" for result in results)

        metrics = breaker.get_metrics()
        assert metrics["total_requests"] == 10
        assert metrics["successful_requests"] == 10

    @pytest.mark.asyncio
    async def test_exception_type_filtering(self):
        """Test that only specified exception types trigger circuit breaker."""
        # Note: This test would require modifying the config to specify expected_exception
        # For now, we test the basic behavior
        config = EnhancedCircuitBreakerConfig.simple_mode("filter_test")
        breaker = EnhancedCircuitBreaker(config)

        async def value_error_func():
            raise ValueError("Value error")

        async def type_error_func():
            raise TypeError("Type error")

        # Both should trigger the circuit breaker by default
        with pytest.raises(ValueError):
            await breaker.call(value_error_func)

        with pytest.raises(TypeError):
            await breaker.call(type_error_func)

        # Circuit should still be closed (only 2 failures, threshold is 3)
        assert breaker.state == EnhancedCircuitBreakerState.CLOSED
