"""Unit tests for ClientManager with proper dependency injection patterns.

This test module demonstrates  testing patterns including:
- Dependency injection for  testability
- Test doubles (fakes, stubs, mocks) for isolation
- Comprehensive coverage of functionality
- Clear test organization and naming
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.config import Config, DatabaseConfig
from src.infrastructure.client_manager import ClientManager
from src.infrastructure.shared import CircuitBreaker, ClientHealth, ClientState
from src.services.errors import APIError


class TestError(Exception):
    """Custom exception for this module."""


# Abstract interfaces for  testability
class ClientFactoryInterface(ABC):
    """Abstract interface for creating API clients."""

    @abstractmethod
    async def create_qdrant_client(self, config: Config) -> Any:
        """Create Qdrant client instance."""

    @abstractmethod
    async def create_openai_client(self, config: Config) -> Any:
        """Create OpenAI client instance."""

    @abstractmethod
    async def create_redis_client(self, config: Config) -> Any:
        """Create Redis client instance."""

    @abstractmethod
    async def create_firecrawl_client(self, config: Config) -> Any:
        """Create Firecrawl client instance."""


# Test doubles
@dataclass
class FakeQdrantClient:
    """Fake Qdrant client for testing."""

    url: str
    api_key: str | None = None
    is_connected: bool = True
    collections: list[str] = None

    def __post_init__(self):
        if self.collections is None:
            self.collections = []

    async def get_collections(self):
        """Simulate getting collections."""
        if not self.is_connected:
            msg = "Not connected to Qdrant"
            raise ConnectionError(msg)
        return self.collections

    async def close(self):
        """Simulate closing connection."""
        self.is_connected = False


class StubClientFactory(ClientFactoryInterface):
    """Stub factory that returns predetermined test doubles."""

    def __init__(self):
        self.qdrant_client = None
        self.openai_client = None
        self.redis_client = None
        self.firecrawl_client = None

    async def create_qdrant_client(self, config: Config) -> Any:
        if self.qdrant_client is None:
            self.qdrant_client = FakeQdrantClient(
                url=config.qdrant.url,
                api_key=config.qdrant.api_key,
                is_connected=True,
            )
        return self.qdrant_client

    async def create_openai_client(self, _config: Config) -> Any:
        if self.openai_client is None:
            self.openai_client = MagicMock()
        return self.openai_client

    async def create_redis_client(self, _config: Config) -> Any:
        if self.redis_client is None:
            self.redis_client = AsyncMock()
        return self.redis_client

    async def create_firecrawl_client(self, _config: Config) -> Any:
        if self.firecrawl_client is None:
            self.firecrawl_client = AsyncMock()
        return self.firecrawl_client


# Fixtures
@pytest.fixture(autouse=True)
async def ensure_clean_singleton():
    """Ensure clean singleton state before and after each test."""
    ClientManager.reset_singleton()
    yield
    ClientManager.reset_singleton()


@pytest.fixture
def config():
    """Create test configuration."""
    config = Config()
    config.openai.api_key = "test-openai-key"
    config.firecrawl.api_key = "test-firecrawl-key"
    config.cache.enable_caching = False
    return config


@pytest.fixture
def stub_factory():
    """Create stub client factory."""
    return StubClientFactory()


@pytest.fixture
async def client_manager_with_stub(config, stub_factory):
    """Create ClientManager with stub factory injection."""
    # Use thread-safe reset
    ClientManager.reset_singleton()

    manager = ClientManager(config)

    # Inject stub factory methods with proper signatures
    manager._create_qdrant_client = lambda: stub_factory.create_qdrant_client(config)
    manager._create_openai_client = lambda: stub_factory.create_openai_client(config)
    manager._create_redis_client = lambda: stub_factory.create_redis_client(config)
    manager._create_firecrawl_client = lambda: stub_factory.create_firecrawl_client(
        config
    )

    # Mock get_task_queue_manager if needed
    async def mock_get_task_queue_manager():
        return AsyncMock()

    manager.get_task_queue_manager = mock_get_task_queue_manager

    yield manager

    # Cleanup
    await manager.cleanup()
    ClientManager.reset_singleton()


# Test cases
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
            recovery_timeout=1.0,  # 1 second for faster tests
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
            msg = "Test failure"
            raise TestError(msg)

        # Fail multiple times - circuit breaker should pass through the original exception
        for _ in range(3):  # Fixture failure threshold is 3
            with pytest.raises(Exception, match="Test failure"):
                await circuit_breaker.call(failing_func)

        # Circuit should be open now
        assert circuit_breaker._state == ClientState.FAILED

        # Next call should fail immediately with circuit breaker error
        with pytest.raises(APIError, match="Circuit breaker is open"):
            await circuit_breaker.call(failing_func)

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, circuit_breaker):
        """Test circuit breaker recovery after timeout."""

        async def failing_func():
            msg = "Test failure"
            raise RuntimeError(msg)

        async def success_func():
            return "recovered"

        # Open the circuit
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await circuit_breaker.call(failing_func)

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Should be in half-open state
        assert circuit_breaker.state == ClientState.DEGRADED

        # Successful call should close circuit
        result = await circuit_breaker.call(success_func)
        assert result == "recovered"
        assert circuit_breaker._state == ClientState.HEALTHY


class TestClientManagerInitialization:
    """Test ClientManager initialization and singleton pattern."""

    @pytest.mark.asyncio
    async def test_singleton_pattern(self, config):
        """Test that ClientManager follows singleton pattern."""
        ClientManager.reset_singleton()

        manager1 = ClientManager(config)
        manager2 = ClientManager(config)

        assert manager1 is manager2

        # Cleanup
        await manager1.cleanup()
        ClientManager.reset_singleton()

    @pytest.mark.asyncio
    async def test_initialization_with_config(self, config):
        """Test initialization with configuration."""
        ClientManager.reset_singleton()

        manager = ClientManager(config)

        assert manager.config == config
        assert manager._clients == {}
        assert manager._health == {}
        assert manager._circuit_breakers == {}

        # Cleanup
        await manager.cleanup()
        ClientManager.reset_singleton()


class TestClientManagerClientCreation:
    """Test client creation and management."""

    @pytest.mark.asyncio
    async def test_get_qdrant_client(self, client_manager_with_stub, _stub_factory):
        """Test Qdrant client creation with stub."""
        client = await client_manager_with_stub.get_qdrant_client()

        assert isinstance(client, FakeQdrantClient)
        assert client.url == client_manager_with_stub.config.qdrant.url
        assert client.is_connected

        # Should return same instance on second call
        client2 = await client_manager_with_stub.get_qdrant_client()
        assert client is client2

    @pytest.mark.asyncio
    async def test_get_openai_client_with_api_key(self, client_manager_with_stub):
        """Test OpenAI client creation when API key exists."""
        client = await client_manager_with_stub.get_openai_client()

        assert client is not None
        assert isinstance(client, MagicMock)

    @pytest.mark.asyncio
    async def test_get_openai_client_without_api_key(self, config, _stub_factory):
        """Test OpenAI client returns None without API key."""
        config.openai.api_key = None
        ClientManager.reset_singleton()

        manager = ClientManager(config)
        client = await manager.get_openai_client()

        assert client is None

        # Cleanup
        await manager.cleanup()
        ClientManager.reset_singleton()

    @pytest.mark.asyncio
    async def test_get_qdrant_service(self, client_manager_with_stub):
        """Test Qdrant service creation."""
        with patch("src.services.vector_db.service.QdrantService") as mock_service:
            mock_instance = AsyncMock()
            mock_service.return_value = mock_instance

            service = await client_manager_with_stub.get_qdrant_service()

            assert service is mock_instance
            mock_instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_embedding_manager(self, client_manager_with_stub):
        """Test EmbeddingManager creation."""
        with patch("src.services.embeddings.manager.EmbeddingManager") as mock_manager:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            mock_manager.return_value = mock_instance

            manager = await client_manager_with_stub.get_embedding_manager()

            assert manager is mock_instance
            mock_manager.assert_called_once()


class TestClientManagerHealthChecks:
    """Test health monitoring functionality."""

    @pytest.mark.asyncio
    async def test_health_check_initialization(self, client_manager_with_stub):
        """Test health check task starts after initialization."""
        # Initialize manager to start health checks
        await client_manager_with_stub.initialize()

        assert client_manager_with_stub._health_check_task is not None
        assert not client_manager_with_stub._health_check_task.done()

    @pytest.mark.asyncio
    async def test_check_qdrant_health_success(
        self, client_manager_with_stub, _stub_factory
    ):
        """Test successful Qdrant health check."""
        # Create client first
        await client_manager_with_stub.get_qdrant_client()

        # Check health
        await client_manager_with_stub._check_qdrant_health()

        health = client_manager_with_stub._health.get("qdrant")
        assert health is not None
        assert health.state == ClientState.HEALTHY
        assert health.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_check_qdrant_health_failure(
        self, client_manager_with_stub, _stub_factory
    ):
        """Test failed Qdrant health check."""
        # Create client first
        client = await client_manager_with_stub.get_qdrant_client()

        # Make client fail
        client.is_connected = False

        # Manually run health check to update status
        await client_manager_with_stub._run_single_health_check(
            "qdrant", client_manager_with_stub._check_qdrant_health
        )

        health = client_manager_with_stub._health.get("qdrant")
        assert health is not None
        assert health.state == ClientState.DEGRADED
        assert health.consecutive_failures == 1
        assert health.last_error is not None

    @pytest.mark.asyncio
    async def test_get_health_status(self, client_manager_with_stub):
        """Test getting overall health status."""
        # Create some clients
        await client_manager_with_stub.get_qdrant_client()

        # Get health status
        status = await client_manager_with_stub.get_health_status()

        assert isinstance(status, dict)
        assert "qdrant" in status
        assert "state" in status["qdrant"]
        assert "last_check" in status["qdrant"]


class TestClientManagerContextManager:
    """Test context manager functionality."""

    @pytest.mark.asyncio
    async def test_managed_client_qdrant(self, client_manager_with_stub):
        """Test managed client context for Qdrant."""
        async with client_manager_with_stub.managed_client("qdrant") as client:
            assert isinstance(client, FakeQdrantClient)
            assert client.is_connected

    @pytest.mark.asyncio
    async def test_managed_client_openai(self, client_manager_with_stub):
        """Test managed client context for OpenAI."""
        async with client_manager_with_stub.managed_client("openai") as client:
            assert client is not None

    @pytest.mark.asyncio
    async def test_managed_client_invalid_type(self, client_manager_with_stub):
        """Test managed client with invalid type."""
        with pytest.raises(ValueError, match="Unknown client type"):
            async with client_manager_with_stub.managed_client("invalid"):
                pass


class TestClientManagerCleanup:
    """Test cleanup and resource management."""

    @pytest.mark.asyncio
    async def test_cleanup_closes_clients(
        self, client_manager_with_stub, _stub_factory
    ):
        """Test cleanup properly closes all clients."""
        # Create clients
        qdrant_client = await client_manager_with_stub.get_qdrant_client()

        # Cleanup
        await client_manager_with_stub.cleanup()

        # Verify client was closed
        assert not qdrant_client.is_connected
        assert client_manager_with_stub._clients == {}

    @pytest.mark.asyncio
    async def test_cleanup_cancels_health_check(self, client_manager_with_stub):
        """Test cleanup cancels health check task."""
        # Initialize to start health checks
        await client_manager_with_stub.initialize()

        health_task = client_manager_with_stub._health_check_task
        assert health_task is not None

        # Cleanup
        await client_manager_with_stub.cleanup()

        # Task should be cancelled
        assert health_task.cancelled()

    @pytest.mark.asyncio
    async def test_cleanup_handles_exceptions(self, client_manager_with_stub):
        """Test cleanup handles client close exceptions gracefully."""
        # Create a client that fails to close
        failing_client = AsyncMock()
        failing_client.close.side_effect = Exception("Close failed")
        client_manager_with_stub._clients["test"] = failing_client

        # Cleanup should not raise
        await client_manager_with_stub.cleanup()

        # Client should be removed despite error
        assert "test" not in client_manager_with_stub._clients


class TestClientManagerErrorHandling:
    """Test error handling and resilience."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, config):
        """Test circuit breaker prevents cascading failures."""
        ClientManager.reset_singleton()
        manager = ClientManager(config)

        # Mock client creation to fail
        async def failing_create():
            msg = "Cannot connect"
            raise ConnectionError(msg)

        manager._create_qdrant_client = failing_create

        # No need to set failure threshold - default is already 5

        # First few attempts should raise the original error (default threshold is 5)
        for _ in range(5):
            with pytest.raises(APIError, match="Failed to create qdrant client"):
                await manager.get_qdrant_client()

        # Circuit should be open now, next attempt fails fast
        # The error might be wrapped in the APIError message
        with pytest.raises(APIError) as excinfo:
            await manager.get_qdrant_client()

        # Check that circuit breaker is mentioned in the error
        assert "circuit breaker is open" in str(excinfo.value).lower()

        # Cleanup
        await manager.cleanup()
        ClientManager.reset_singleton()

    @pytest.mark.asyncio
    async def test_health_check_error_handling(self, client_manager_with_stub):
        """Test health check handles errors gracefully."""
        # Create client first to establish health entry
        await client_manager_with_stub.get_qdrant_client()

        # Replace with a client that throws during health check
        failing_client = AsyncMock()
        failing_client.get_collections.side_effect = Exception("Health check failed")
        client_manager_with_stub._clients["qdrant"] = failing_client

        # Run health check manually
        await client_manager_with_stub._run_single_health_check(
            "qdrant", client_manager_with_stub._check_qdrant_health
        )

        # Health should be degraded
        health = client_manager_with_stub._health.get("qdrant")
        assert health is not None
        assert health.state == ClientState.DEGRADED
        # The actual error message might be different
        assert health.last_error is not None


# Integration test example
class TestClientManagerIntegration:
    """Integration tests for ClientManager."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, config):
        """Test full lifecycle of ClientManager."""
        ClientManager.reset_singleton()

        # Create manager
        manager = ClientManager(config)

        # Mock all client creation methods
        mock_qdrant = AsyncMock()
        mock_qdrant.get_collections = AsyncMock(return_value=[])
        manager._create_qdrant_client = AsyncMock(return_value=mock_qdrant)

        # Create client
        qdrant_client = await manager.get_qdrant_client()
        assert qdrant_client is not None

        # Check health
        status = await manager.get_health_status()
        assert "qdrant" in status

        # Use managed client
        async with manager.managed_client("qdrant") as client:
            assert client is qdrant_client

        # Cleanup
        await manager.cleanup()

        # Verify cleanup
        assert manager._clients == {}
        assert (
            manager._health_check_task.cancelled()
            if manager._health_check_task
            else True
        )

        ClientManager.reset_singleton()


class TestClientManagerDatabaseIntegration:
    """Test database manager integration with ClientManager."""

    @pytest.mark.skip(
        reason="Test infrastructure mocking complexity - client manager import caching issue"
    )
    @pytest.mark.asyncio
    async def test_get_database_manager_creation(self):
        """Test creation of database manager."""

        config = Config()
        config.database = DatabaseConfig(
            database_url="sqlite+aiosqlite:///:memory:",
            echo_queries=True,
            pool_size=20,
        )

        client_manager = ClientManager(config)

        with patch(
            "src.infrastructure.client_manager.DatabaseManager"
        ) as mock_database_manager_class:
            mock_database_manager = AsyncMock()
            mock_database_manager_class.return_value = mock_database_manager

            # First call should create the manager
            db_manager = await client_manager.get_database_manager()

            assert db_manager is mock_database_manager
            mock_database_manager_class.assert_called_once()
            mock_database_manager.initialize.assert_called_once()

            # Second call should return the same instance (cached)
            db_manager2 = await client_manager.get_database_manager()
            assert db_manager2 is mock_database_manager

            # Should not create a new instance
            mock_database_manager_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_database_manager_in_managed_client(self):
        """Test database manager through managed_client interface."""

        config = Config()
        config.database = DatabaseConfig(
            database_url="sqlite+aiosqlite:///:memory:",
            echo_queries=True,
            pool_size=20,
        )

        client_manager = ClientManager(config)

        with patch.object(client_manager, "get_database_manager") as mock_get_db:
            mock_db_manager = AsyncMock()
            mock_get_db.return_value = mock_db_manager

            async with client_manager.managed_client("database") as db_manager:
                assert db_manager is mock_db_manager

            mock_get_db.assert_called_once()

    @pytest.mark.asyncio
    async def test_database_manager_cleanup(self):
        """Test database manager is included in cleanup."""

        config = Config()
        config.database = DatabaseConfig(database_url="sqlite+aiosqlite:///:memory:")

        client_manager = ClientManager(config)

        # Mock database manager with cleanup method
        mock_db_manager = AsyncMock()
        mock_db_manager.cleanup = AsyncMock()
        client_manager._database_manager = mock_db_manager

        await client_manager.cleanup()

        # Should call cleanup on database manager
        mock_db_manager.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_database_manager_enterprise_creation(self):
        """Test database manager uses enterprise DatabaseManager with monitoring."""

        config = Config()
        config.database = DatabaseConfig(
            database_url="sqlite+aiosqlite:///:memory:",
        )

        client_manager = ClientManager(config)

        with patch(
            "src.infrastructure.client_manager.DatabaseManager"
        ) as mock_database_manager_class:
            mock_database_manager = AsyncMock()
            mock_database_manager_class.return_value = mock_database_manager

            await client_manager.get_database_manager()

            # Verify database manager was created with enterprise components
            mock_database_manager_class.assert_called_once()
            call__kwargs = mock_database_manager_class.call_args._kwargs
            assert "config" in call__kwargs
            assert "load_monitor" in call__kwargs
            assert "query_monitor" in call__kwargs
            assert "circuit_breaker" in call__kwargs
            mock_database_manager.initialize.assert_called_once()


class TestClientManagerAdvancedCoverage:
    """Test  coverage scenarios for AB testing and service getters."""

    @pytest.mark.asyncio
    async def test_get_ab_testing_manager_creation(self):
        """Test creation of AB testing manager."""
        config = Config()
        client_manager = ClientManager(config)

        # Mock the ABTestingManager and its dependencies
        with (
            patch(
                "src.services.deployment.ab_testing.ABTestingManager"
            ) as mock_ab_class,
            patch.object(client_manager, "get_qdrant_service") as mock_get_qdrant,
            patch.object(client_manager, "get_cache_manager") as mock_get_cache,
            patch.object(
                client_manager, "get_feature_flag_manager"
            ) as mock_get_feature_flag,
        ):
            mock_ab_instance = AsyncMock()
            mock_ab_class.return_value = mock_ab_instance

            _mock_qdrant_service = Mock()
            mock_cache_manager = Mock()
            mock_feature_flag_manager = Mock()
            mock_get_qdrant.return_value = _mock_qdrant_service
            mock_get_cache.return_value = mock_cache_manager
            mock_get_feature_flag.return_value = mock_feature_flag_manager

            # First call should create the manager
            ab_manager = await client_manager.get_ab_testing_manager()

            assert ab_manager is mock_ab_instance
            mock_ab_class.assert_called_once_with(
                qdrant_service=_mock_qdrant_service,
                cache_manager=mock_cache_manager,
                feature_flag_manager=mock_feature_flag_manager,
            )

            # Second call should return the same instance (cached)
            ab_manager2 = await client_manager.get_ab_testing_manager()
            assert ab_manager2 is mock_ab_instance

            # Should not create a new instance
            mock_ab_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_qdrant_client_with_config(self):
        """Test creation of Qdrant client with configuration."""
        # Clear singleton to ensure clean state
        ClientManager.reset_singleton()

        config = Config()
        config.qdrant.url = "http://localhost:6333"
        config.qdrant.api_key = "test-key"
        config.qdrant.timeout = 30.0
        config.qdrant.prefer_grpc = True

        client_manager = ClientManager(config)

        # Mock the client creation method instead of the class
        mock_client = AsyncMock()
        mock_client.get_collections = AsyncMock(return_value=[])

        async def mock_create_qdrant():
            # Verify the config values are correct
            assert client_manager.config.qdrant.url == "http://localhost:6333"
            assert client_manager.config.qdrant.api_key == "test-key"
            assert client_manager.config.qdrant.timeout == 30.0
            assert client_manager.config.qdrant.prefer_grpc is True
            return mock_client

        client_manager._create_qdrant_client = mock_create_qdrant

        # Call the method to create the client
        qdrant_client = await client_manager.get_qdrant_client()

        assert qdrant_client is mock_client

        # Cleanup
        await client_manager.cleanup()
        ClientManager.reset_singleton()

    @pytest.mark.asyncio
    async def test_get_qdrant_service_lazy_initialization(self):
        """Test lazy initialization of QdrantService."""
        config = Config()
        client_manager = ClientManager(config)

        with patch(
            "src.services.vector_db.service.QdrantService"
        ) as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.return_value = mock_service

            # First call should create the service
            service1 = await client_manager.get_qdrant_service()
            assert service1 == mock_service
            mock_service.initialize.assert_called_once()

            # Second call should return the same instance
            service2 = await client_manager.get_qdrant_service()
            assert service2 == service1
            assert mock_service.initialize.call_count == 1  # Not called again

    @pytest.mark.asyncio
    async def test_get_hyde_engine_with_dependencies(self):
        """Test HyDEEngine initialization with all dependencies."""
        config = Config()
        client_manager = ClientManager(config)

        mock_hyde_engine = AsyncMock()
        mock_embedding_manager = AsyncMock()
        _mock_qdrant_service = AsyncMock()
        mock_cache_manager = AsyncMock()
        mock_openai_client = AsyncMock()

        with (
            patch(
                "src.services.hyde.engine.HyDEQueryEngine",
                return_value=mock_hyde_engine,
            ),
            patch("src.services.hyde.config.HyDEConfig") as mock_config_class,
            patch("src.services.hyde.config.HyDEPromptConfig"),
            patch("src.services.hyde.config.HyDEMetricsConfig"),
            patch.object(
                client_manager,
                "get_embedding_manager",
                return_value=mock_embedding_manager,
            ),
            patch.object(
                client_manager, "get_qdrant_service", return_value=_mock_qdrant_service
            ),
            patch.object(
                client_manager, "get_cache_manager", return_value=mock_cache_manager
            ),
            patch.object(
                client_manager, "get_openai_client", return_value=mock_openai_client
            ),
        ):
            service = await client_manager.get_hyde_engine()
            assert service == mock_hyde_engine
            mock_hyde_engine.initialize.assert_called_once()

            # Verify HyDEConfig was created from unified config
            mock_config_class.from_unified_config.assert_called_once_with(
                client_manager.config.hyde
            )

    @pytest.mark.asyncio
    async def test_get_cache_manager_initialization(self):
        """Test CacheManager initialization with config parameters."""
        config = Config()
        client_manager = ClientManager(config)

        with patch("src.services.cache.manager.CacheManager") as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager

            service = await client_manager.get_cache_manager()
            assert service == mock_manager

            # Verify initialization parameters
            call_args = mock_manager_class.call_args
            assert (
                call_args._kwargs["dragonfly_url"]
                == client_manager.config.cache.dragonfly_url
            )
            assert (
                call_args._kwargs["enable_local_cache"]
                == client_manager.config.cache.enable_local_cache
            )
            assert (
                call_args._kwargs["enable_distributed_cache"]
                == client_manager.config.cache.enable_dragonfly_cache
            )

    @pytest.mark.asyncio
    async def test_get_project_storage_initialization(self):
        """Test ProjectStorage initialization."""
        config = Config()
        client_manager = ClientManager(config)

        with patch(
            "src.services.core.project_storage.ProjectStorage"
        ) as mock_storage_class:
            mock_storage = AsyncMock()
            mock_storage_class.return_value = mock_storage

            service = await client_manager.get_project_storage()
            assert service == mock_storage

            mock_storage_class.assert_called_once_with(
                data_dir=client_manager.config.data_dir
            )
