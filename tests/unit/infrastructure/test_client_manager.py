"""Modern unit tests for ClientManager with dependency injection container mocking."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import Config
from src.infrastructure.client_manager import ClientManager
from src.infrastructure.container import ApplicationContainer
from src.infrastructure.shared import CircuitBreaker, ClientHealth, ClientState
from src.services.errors import APIError


class TestError(Exception):
    """Custom exception for this module."""


class MockClientProvider:
    """Mock client provider for testing."""

    def __init__(self, client=None):
        self.client = client or AsyncMock()
        self._healthy = True

    async def health_check(self):
        return self._healthy


@pytest.fixture(autouse=True)
async def ensure_clean_singleton():
    """Ensure clean singleton state before and after each test."""
    ClientManager.reset_singleton()
    yield
    ClientManager.reset_singleton()


@pytest.fixture
def test_config():
    """Create test configuration."""
    config = Config()
    config.openai.api_key = "test-openai-key"
    config.firecrawl.api_key = "test-firecrawl-key"
    config.cache.enable_caching = False
    config.qdrant.url = "http://localhost:6333"
    config.qdrant.api_key = "test-qdrant-key"
    config.cache.redis_url = "redis://localhost:6379"
    return config


@pytest.fixture
async def mock_container(test_config):
    """Create mock dependency injection container."""
    container = ApplicationContainer()
    container.config.override(test_config)

    # Create and override providers with mocks
    mock_openai_provider = MockClientProvider(AsyncMock())
    mock_qdrant_provider = MockClientProvider(AsyncMock())
    mock_redis_provider = MockClientProvider(AsyncMock())
    mock_firecrawl_provider = MockClientProvider(AsyncMock())
    mock_http_provider = MockClientProvider(AsyncMock())

    container.openai_provider.override(mock_openai_provider)
    container.qdrant_provider.override(mock_qdrant_provider)
    container.redis_provider.override(mock_redis_provider)
    container.firecrawl_provider.override(mock_firecrawl_provider)
    container.http_provider.override(mock_http_provider)

    # Mock parallel processing system
    mock_parallel_system = AsyncMock()
    mock_parallel_system.get_system_status.return_value = {"status": "healthy"}
    container.parallel_processing_system.override(mock_parallel_system)

    return container


@pytest.fixture
async def client_manager_with_mocks(mock_container):
    """Create ClientManager with mocked dependency injection container."""
    ClientManager.reset_singleton()

    with patch(
        "src.infrastructure.client_manager.get_container", return_value=mock_container
    ):
        manager = ClientManager()
        await manager.initialize()

        yield manager

        await manager.cleanup()
        ClientManager.reset_singleton()


class TestClientState:
    """Test ClientState enum values."""

    def test_client_state_values(self):
        """Verify all expected states exist."""
        assert ClientState.UNINITIALIZED.value == "uninitialized"
        assert ClientState.HEALTHY.value == "healthy"
        assert ClientState.DEGRADED.value == "degraded"
        assert ClientState.FAILED.value == "failed"

    def test_client_state_enum_completeness(self):
        """Verify enum contains all expected members."""
        expected_states = {"UNINITIALIZED", "HEALTHY", "DEGRADED", "FAILED"}
        actual_states = {state.name for state in ClientState}
        assert actual_states == expected_states


class TestClientHealth:
    """Test ClientHealth dataclass."""

    def test_client_health_initialization(self):
        """Test basic initialization."""
        health = ClientHealth(
            state=ClientState.HEALTHY,
            last_check=time.time(),
            last_error=None,
            consecutive_failures=0,
        )
        assert health.state == ClientState.HEALTHY
        assert health.last_error is None
        assert health.consecutive_failures == 0

    def test_client_health_with_error(self):
        """Test health with error state."""
        error_msg = "Connection failed"
        health = ClientHealth(
            state=ClientState.FAILED,
            last_check=time.time(),
            last_error=error_msg,
            consecutive_failures=3,
        )
        assert health.state == ClientState.FAILED
        assert health.last_error == error_msg
        assert health.consecutive_failures == 3


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker with test parameters."""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,
            half_open_requests=1,
        )

    @pytest.mark.asyncio
    async def test_circuit_breaker_healthy_state(self, circuit_breaker):
        """Test circuit breaker in healthy state allows calls."""

        async def success_func():
            return "success"

        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == ClientState.HEALTHY

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, circuit_breaker):
        """Test circuit breaker opens after threshold failures."""

        async def failing_func():
            raise ConnectionError("Test connection failure")

        for _ in range(3):
            with pytest.raises(ConnectionError):
                await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == ClientState.FAILED

        async def success_func():
            return "success"

        with pytest.raises(APIError, match="Circuit breaker is open"):
            await circuit_breaker.call(success_func)

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, circuit_breaker):
        """Test circuit breaker recovery after timeout."""

        async def failing_func():
            raise ConnectionError("Test connection failure")

        async def success_func():
            return "success"

        for _ in range(3):
            with pytest.raises(ConnectionError):
                await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == ClientState.FAILED

        await asyncio.sleep(1.1)

        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == ClientState.HEALTHY


class TestClientManagerInitialization:
    """Test ClientManager initialization and singleton pattern."""

    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """Test that ClientManager follows singleton pattern."""
        ClientManager.reset_singleton()

        with patch(
            "src.infrastructure.client_manager.get_container"
        ) as mock_get_container:
            mock_container = MagicMock()
            mock_get_container.return_value = mock_container

            manager1 = ClientManager()
            manager2 = ClientManager()

            assert manager1 is manager2

        ClientManager.reset_singleton()

    @pytest.mark.asyncio
    async def test_initialization_with_dependency_injection(
        self, client_manager_with_mocks
    ):
        """Test initialization with dependency injection container."""
        assert client_manager_with_mocks.is_initialized
        assert client_manager_with_mocks._providers is not None
        assert len(client_manager_with_mocks._providers) == 5


class TestClientManagerClientAccess:
    """Test client access methods using dependency injection."""

    @pytest.mark.asyncio
    async def test_get_qdrant_client(self, client_manager_with_mocks):
        """Test Qdrant client access through dependency injection."""
        client = await client_manager_with_mocks.get_qdrant_client()
        assert client is not None

    @pytest.mark.asyncio
    async def test_get_openai_client(self, client_manager_with_mocks):
        """Test OpenAI client access through dependency injection."""
        client = await client_manager_with_mocks.get_openai_client()
        assert client is not None

    @pytest.mark.asyncio
    async def test_get_redis_client(self, client_manager_with_mocks):
        """Test Redis client access through dependency injection."""
        client = await client_manager_with_mocks.get_redis_client()
        assert client is not None

    @pytest.mark.asyncio
    async def test_get_firecrawl_client(self, client_manager_with_mocks):
        """Test Firecrawl client access through dependency injection."""
        client = await client_manager_with_mocks.get_firecrawl_client()
        assert client is not None

    @pytest.mark.asyncio
    async def test_get_http_client(self, client_manager_with_mocks):
        """Test HTTP client access through dependency injection."""
        client = await client_manager_with_mocks.get_http_client()
        assert client is not None


class TestClientManagerServiceIntegration:
    """Test service integration methods."""

    @pytest.mark.asyncio
    async def test_get_vector_store_service(self, client_manager_with_mocks):
        """Test vector store service creation and caching."""
        with (
            patch(
                "src.infrastructure.client_manager.FastEmbedProvider"
            ) as mock_embed_cls,
            patch(
                "src.infrastructure.client_manager.VectorStoreService"
            ) as mock_service_cls,
        ):
            mock_embed = MagicMock()
            mock_embed_cls.return_value = mock_embed

            mock_service = MagicMock()
            mock_service.initialize = AsyncMock()
            mock_service.cleanup = AsyncMock()
            mock_service_cls.return_value = mock_service

            service1 = await client_manager_with_mocks.get_vector_store_service()
            assert service1 is mock_service
            assert mock_service.initialize.await_count == 1

            service2 = await client_manager_with_mocks.get_vector_store_service()
            assert service2 is service1
            assert mock_service.initialize.await_count == 1

    @pytest.mark.asyncio
    async def test_get_parallel_processing_system(self, client_manager_with_mocks):
        """Test parallel processing system access."""
        system = await client_manager_with_mocks.get_parallel_processing_system()
        assert system is not None

    @pytest.mark.asyncio
    async def test_get_browser_automation_router(self, client_manager_with_mocks):
        """Test browser automation router creation and caching."""
        with patch(
            "src.infrastructure.client_manager.AutomationRouter"
        ) as mock_router_class:
            mock_router = MagicMock()
            mock_router.initialize = AsyncMock()
            mock_router_class.return_value = mock_router

            router1 = await client_manager_with_mocks.get_browser_automation_router()
            assert router1 is mock_router
            mock_router.initialize.assert_awaited_once()

            router2 = await client_manager_with_mocks.get_browser_automation_router()
            assert router2 is router1
            mock_router.initialize.assert_awaited_once()


class TestClientManagerHealthAndStatus:
    """Test health monitoring and status methods."""

    @pytest.mark.asyncio
    async def test_get_health_status(self, client_manager_with_mocks):
        """Test health status retrieval."""
        with patch(
            "src.infrastructure.client_manager.deps_get_health_status",
            new_callable=AsyncMock,
        ) as mock_health_func:
            mock_health_func.return_value = {"test": "healthy"}

            status = await client_manager_with_mocks.get_health_status()
            assert status == {"test": "healthy"}
            mock_health_func.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_overall_health(self, client_manager_with_mocks):
        """Test overall health retrieval."""
        with patch(
            "src.infrastructure.client_manager.deps_get_overall_health",
            new_callable=AsyncMock,
        ) as mock_overall_func:
            mock_overall_func.return_value = {"healthy": True}

            health = await client_manager_with_mocks.get_overall_health()
            assert health == {"healthy": True}
            mock_overall_func.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_service_status(self, client_manager_with_mocks):
        """Test service status information."""
        status = await client_manager_with_mocks.get_service_status()

        assert isinstance(status, dict)
        assert "initialized" in status
        assert "mode" in status
        assert "providers" in status
        assert status["mode"] == "function_based_dependencies"


class TestClientManagerContextManager:
    """Test context manager functionality."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, client_manager_with_mocks):
        """Test async context manager protocol."""
        async with client_manager_with_mocks as manager:
            assert manager.is_initialized

        assert not client_manager_with_mocks._initialized

    @pytest.mark.asyncio
    async def test_managed_client_context(self, client_manager_with_mocks):
        """Test managed client context manager."""
        async with client_manager_with_mocks.managed_client("qdrant") as client:
            assert client is not None

    @pytest.mark.asyncio
    async def test_managed_client_invalid_type(self, client_manager_with_mocks):
        """Test managed client with invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown client type"):
            async with client_manager_with_mocks.managed_client("invalid_type"):
                pass


class TestClientManagerCleanup:
    """Test cleanup and resource management."""

    @pytest.mark.asyncio
    async def test_cleanup_resets_state(self, client_manager_with_mocks):
        """Test that cleanup properly resets manager state."""
        assert client_manager_with_mocks.is_initialized

        await client_manager_with_mocks.cleanup()

        assert not client_manager_with_mocks._initialized
        assert client_manager_with_mocks._providers == {}
        assert client_manager_with_mocks._parallel_processing_system is None

    @pytest.mark.asyncio
    async def test_cleanup_with_vector_store_service(self, client_manager_with_mocks):
        """Test cleanup when vector store service exists."""
        with (
            patch(
                "src.infrastructure.client_manager.FastEmbedProvider"
            ) as mock_embed_cls,
            patch(
                "src.infrastructure.client_manager.VectorStoreService"
            ) as mock_service_cls,
        ):
            mock_embed_cls.return_value = MagicMock()

            mock_service = MagicMock()
            mock_service.initialize = AsyncMock()
            mock_service.cleanup = AsyncMock()
            mock_service_cls.return_value = mock_service

            await client_manager_with_mocks.get_vector_store_service()
            assert client_manager_with_mocks._vector_store_service is mock_service

            await client_manager_with_mocks.cleanup()
            mock_service.cleanup.assert_awaited_once()


class TestClientManagerErrorHandling:
    """Test error handling and resilience."""

    @pytest.mark.asyncio
    async def test_get_qdrant_client_provider_error(self, client_manager_with_mocks):
        """Test error when Qdrant provider is not available."""
        original_provider = client_manager_with_mocks._providers.get("qdrant")
        client_manager_with_mocks._providers["qdrant"] = None

        try:
            with pytest.raises(APIError, match="Qdrant client provider not available"):
                await client_manager_with_mocks.get_qdrant_client()
        finally:
            if original_provider:
                client_manager_with_mocks._providers["qdrant"] = original_provider

    @pytest.mark.asyncio
    async def test_get_redis_client_provider_error(self, client_manager_with_mocks):
        """Test error when Redis provider is not available."""
        original_provider = client_manager_with_mocks._providers.get("redis")
        client_manager_with_mocks._providers["redis"] = None

        try:
            with pytest.raises(APIError, match="Redis client provider not available"):
                await client_manager_with_mocks.get_redis_client()
        finally:
            if original_provider:
                client_manager_with_mocks._providers["redis"] = original_provider

    @pytest.mark.asyncio
    async def test_get_http_client_provider_error(self, client_manager_with_mocks):
        """Test error when HTTP provider is not available."""
        original_provider = client_manager_with_mocks._providers.get("http")
        client_manager_with_mocks._providers["http"] = None

        try:
            with pytest.raises(APIError, match="HTTP client provider not available"):
                await client_manager_with_mocks.get_http_client()
        finally:
            if original_provider:
                client_manager_with_mocks._providers["http"] = original_provider


class TestClientManagerFactoryMethods:
    """Test factory methods for ClientManager creation."""

    @pytest.mark.asyncio
    async def test_from_unified_config(self):
        """Test factory method for creating from unified config."""
        manager = ClientManager.from_unified_config()
        assert isinstance(manager, ClientManager)

    @pytest.mark.asyncio
    async def test_from_unified_config_with_auto_detection(self):
        """Test factory method with auto-detection."""
        with patch(
            "src.infrastructure.client_manager.get_container"
        ) as mock_get_container:
            mock_container = MagicMock()
            mock_get_container.return_value = mock_container

            manager = await ClientManager.from_unified_config_with_auto_detection()
            assert isinstance(manager, ClientManager)
            assert manager.is_initialized
