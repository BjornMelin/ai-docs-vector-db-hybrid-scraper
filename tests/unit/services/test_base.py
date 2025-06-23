"""Tests for services/base.py - BaseService abstract class and patterns.

This module tests the abstract base service class that provides common patterns
for service lifecycle management, error handling, and configuration.
"""

import asyncio
from unittest.mock import Mock

import pytest

from src.config import Config
from src.services.base import BaseService
from src.services.errors import APIError


class ConcreteService(BaseService):
    """Concrete implementation of BaseService for testing."""

    def __init__(self, config: Config | None = None):
        super().__init__(config)
        self.init_called = False
        self.cleanup_called = False
        self.init_error = None
        self.cleanup_error = None

    async def initialize(self) -> None:
        """Initialize service resources."""
        if self.init_error:
            raise self.init_error
        self.init_called = True
        self._initialized = True

    async def cleanup(self) -> None:
        """Cleanup service resources."""
        if self.cleanup_error:
            raise self.cleanup_error
        self.cleanup_called = True
        self._initialized = False


class TestBaseService:
    """Test cases for BaseService abstract class."""

    def test_base_service_init_with_config(self):
        """Test BaseService initialization with config."""
        config = Mock(spec=Config)
        service = ConcreteService(config)

        assert service.config is config
        assert service._client is None
        assert service._initialized is False

    def test_base_service_init_without_config(self):
        """Test BaseService initialization without config."""
        service = ConcreteService()

        assert service.config is None
        assert service._client is None
        assert service._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_method(self):
        """Test service initialization."""
        service = ConcreteService()

        await service.initialize()

        assert service.init_called is True
        assert service._initialized is True

    @pytest.mark.asyncio
    async def test_cleanup_method(self):
        """Test service cleanup."""
        service = ConcreteService()
        await service.initialize()

        await service.cleanup()

        assert service.cleanup_called is True
        assert service._initialized is False

    @pytest.mark.asyncio
    async def test_context_manager_success(self):
        """Test context manager with successful operations."""
        service = ConcreteService()

        async with service.context() as ctx_service:
            assert ctx_service is service
            assert service.init_called is True
            assert service._initialized is True

        assert service.cleanup_called is True
        assert service._initialized is False

    @pytest.mark.asyncio
    async def test_context_manager_init_error(self):
        """Test context manager when initialization fails."""
        service = ConcreteService()
        service.init_error = ValueError("Init failed")

        with pytest.raises(ValueError, match="Init failed"):
            async with service.context():
                pass

        # Cleanup should still be called
        assert service.cleanup_called is True

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_error(self):
        """Test context manager when cleanup fails."""
        service = ConcreteService()
        service.cleanup_error = ValueError("Cleanup failed")

        with pytest.raises(ValueError, match="Cleanup failed"):
            async with service.context():
                pass

        assert service.init_called is True

    @pytest.mark.asyncio
    async def test_context_manager_both_errors(self):
        """Test context manager when both init and cleanup fail."""
        service = ConcreteService()
        service.init_error = ValueError("Init failed")
        service.cleanup_error = ValueError("Cleanup failed")

        # When init fails, cleanup still runs and its error may override the init error
        # The specific behavior depends on the contextlib implementation
        with pytest.raises(ValueError):  # Either error could be raised
            async with service.context():
                pass

    @pytest.mark.asyncio
    async def test_async_context_manager_enter(self):
        """Test __aenter__ method."""
        service = ConcreteService()

        result = await service.__aenter__()

        assert result is service
        assert service.init_called is True
        assert service._initialized is True

    @pytest.mark.asyncio
    async def test_async_context_manager_exit_success(self):
        """Test __aexit__ method with successful cleanup."""
        service = ConcreteService()
        await service.initialize()

        result = await service.__aexit__(None, None, None)

        assert result is False
        assert service.cleanup_called is True
        assert service._initialized is False

    @pytest.mark.asyncio
    async def test_async_context_manager_exit_with_exception(self):
        """Test __aexit__ method when an exception occurred."""
        service = ConcreteService()
        await service.initialize()

        # Simulate exception context
        result = await service.__aexit__(ValueError, ValueError("test error"), None)

        assert result is False  # Don't suppress the exception
        assert service.cleanup_called is True

    @pytest.mark.asyncio
    async def test_async_context_manager_full_flow(self):
        """Test full async context manager flow."""
        service = ConcreteService()

        async with service as ctx_service:
            assert ctx_service is service
            assert service.init_called is True
            assert service._initialized is True

        assert service.cleanup_called is True
        assert service._initialized is False


# NOTE: TestRetryWithBackoff class has been removed as the _retry_with_backoff
# method has been replaced with the @retry_async decorator from src.services.errors.
# Retry functionality tests are now covered by error handling decorator tests.


class TestValidateInitialized:
    """Test cases for _validate_initialized method."""

    def test_validate_initialized_when_initialized(self):
        """Test validation when service is initialized."""
        service = ConcreteService()
        service._initialized = True

        # Should not raise an exception
        service._validate_initialized()

    def test_validate_initialized_when_not_initialized(self):
        """Test validation when service is not initialized."""
        service = ConcreteService()
        service._initialized = False

        with pytest.raises(APIError) as exc_info:
            service._validate_initialized()

        error_msg = str(exc_info.value)
        assert "ConcreteService not initialized" in error_msg
        assert "Call initialize() or use context manager" in error_msg

    def test_validate_initialized_default_state(self):
        """Test validation in default uninitialized state."""
        service = ConcreteService()

        with pytest.raises(APIError):
            service._validate_initialized()


class TestServiceIntegration:
    """Integration tests for BaseService functionality."""

    @pytest.mark.asyncio
    async def test_service_lifecycle_full_flow(self):
        """Test complete service lifecycle."""
        service = ConcreteService()

        # Initial state
        assert not service._initialized
        assert not service.init_called
        assert not service.cleanup_called

        # Initialize
        await service.initialize()
        assert service._initialized
        assert service.init_called

        # Use service (validate initialized)
        service._validate_initialized()  # Should not raise

        # Cleanup
        await service.cleanup()
        assert not service._initialized
        assert service.cleanup_called

    @pytest.mark.asyncio
    async def test_nested_context_managers(self):
        """Test nested context manager usage."""
        service1 = ConcreteService()
        service2 = ConcreteService()

        async with service1.context():
            assert service1.init_called
            async with service2.context():
                assert service2.init_called
                # Both services should be initialized
                service1._validate_initialized()
                service2._validate_initialized()
            # service2 should be cleaned up
            assert service2.cleanup_called
        # service1 should be cleaned up
        assert service1.cleanup_called

    @pytest.mark.asyncio
    async def test_concurrent_service_operations(self):
        """Test concurrent service operations."""
        services = [ConcreteService() for _ in range(3)]

        async def use_service(service):
            async with service.context():
                await asyncio.sleep(0.01)  # Simulate work
                return service.init_called

        results = await asyncio.gather(*[use_service(s) for s in services])

        assert all(results)
        assert all(s.cleanup_called for s in services)

    @pytest.mark.asyncio
    async def test_service_with_configuration(self):
        """Test service with actual configuration object."""
        config = Mock(spec=Config)
        config.some_setting = "test_value"

        service = ConcreteService(config)

        async with service.context():
            assert service.config is config
            assert service.config.some_setting == "test_value"

    # NOTE: Retry integration test removed as _retry_with_backoff method
    # has been replaced with @retry_async decorator from src.services.errors.

    @pytest.mark.asyncio
    async def test_service_cleanup_on_exception(self):
        """Test service cleanup when exception occurs during usage."""
        service = ConcreteService()

        with pytest.raises(ValueError, match="test exception"):
            async with service.context():
                assert service.init_called
                raise ValueError("test exception")

        # Cleanup should still occur
        assert service.cleanup_called

    @pytest.mark.asyncio
    async def test_multiple_initialization_calls(self):
        """Test behavior when initialize is called multiple times."""
        service = ConcreteService()

        await service.initialize()
        first_init_state = service.init_called

        # Call initialize again
        await service.initialize()

        # Should still work (implementation dependent)
        assert service._initialized
        assert first_init_state  # First call succeeded

    @pytest.mark.asyncio
    async def test_cleanup_without_initialization(self):
        """Test cleanup called without prior initialization."""
        service = ConcreteService()

        # Should not raise an exception
        await service.cleanup()
        assert service.cleanup_called

    def test_abstract_base_service_cannot_be_instantiated(self):
        """Test that BaseService cannot be directly instantiated."""
        # This should raise TypeError due to abstract methods
        with pytest.raises(TypeError):
            BaseService()

    def test_base_service_abstract_methods_defined(self):
        """Test that abstract methods are properly defined in base class."""
        # Test that the abstract methods exist and are decorated correctly
        assert hasattr(BaseService, "initialize")
        assert hasattr(BaseService, "cleanup")

        # Verify they are abstract methods
        assert getattr(BaseService.initialize, "__isabstractmethod__", False)
        assert getattr(BaseService.cleanup, "__isabstractmethod__", False)
