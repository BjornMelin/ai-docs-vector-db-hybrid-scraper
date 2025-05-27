"""Tests for centralized client manager."""

import asyncio
import time
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
import redis.asyncio as redis
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from src.infrastructure import ClientManager
from src.infrastructure.client_manager import CircuitBreaker
from src.infrastructure.client_manager import ClientHealth
from src.infrastructure.client_manager import ClientManagerConfig
from src.infrastructure.client_manager import ClientState
from src.services.errors import APIError


@pytest.fixture
def client_config():
    """Test client manager configuration."""
    return ClientManagerConfig(
        qdrant_url="http://localhost:6333",
        openai_api_key="test-key",
        firecrawl_api_key="test-key",
        redis_url="redis://localhost:6379",
        health_check_interval=1.0,
        health_check_timeout=0.5,
        circuit_breaker_failure_threshold=3,
        circuit_breaker_recovery_timeout=2.0,
    )


@pytest.fixture
async def client_manager(client_config):
    """Client manager fixture."""
    # Reset singleton for each test
    ClientManager._instance = None
    manager = ClientManager(client_config)
    yield manager
    await manager.cleanup()
    ClientManager._instance = None


class TestClientManagerConfig:
    """Test configuration validation."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = ClientManagerConfig(
            qdrant_url="http://localhost:6333",
            redis_url="redis://localhost:6379",
        )
        assert config.qdrant_url == "http://localhost:6333"
        assert config.redis_url == "redis://localhost:6379"

    def test_invalid_qdrant_url(self):
        """Test invalid Qdrant URL."""
        with pytest.raises(ValueError, match="Qdrant URL must start with"):
            ClientManagerConfig(qdrant_url="invalid-url")

    def test_invalid_redis_url(self):
        """Test invalid Redis URL."""
        with pytest.raises(ValueError, match="Redis URL must start with"):
            ClientManagerConfig(redis_url="invalid-url")

    def test_default_values(self):
        """Test default configuration values."""
        config = ClientManagerConfig()
        assert config.qdrant_timeout == 30.0
        assert config.health_check_interval == 30.0
        assert config.circuit_breaker_failure_threshold == 5


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self):
        """Test circuit breaker with successful calls."""
        breaker = CircuitBreaker(failure_threshold=3)

        async def success_func():
            return "success"

        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state == ClientState.HEALTHY
        assert breaker._failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_threshold(self):
        """Test circuit breaker opens after failure threshold."""
        breaker = CircuitBreaker(failure_threshold=3)

        async def failing_func():
            raise Exception("Test error")

        # Fail up to threshold
        for _ in range(3):
            with pytest.raises(Exception, match="Test error"):
                await breaker.call(failing_func)

        assert breaker.state == ClientState.FAILED
        assert breaker._failure_count == 3

        # Circuit should be open
        with pytest.raises(APIError, match="Circuit breaker is open"):
            await breaker.call(failing_func)

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            half_open_requests=1,
        )

        async def failing_func():
            raise Exception("Test error")

        async def success_func():
            return "success"

        # Fail to open circuit
        for _ in range(2):
            with pytest.raises(Exception, match="Test error"):
                await breaker.call(failing_func)

        assert breaker.state == ClientState.FAILED

        # Wait for recovery timeout
        await asyncio.sleep(0.2)

        # Circuit should be half-open (degraded)
        assert breaker.state == ClientState.DEGRADED

        # Successful call should close circuit
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state == ClientState.HEALTHY

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_failure(self):
        """Test circuit breaker fails in half-open state."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            half_open_requests=1,
        )

        async def failing_func():
            raise Exception("Test error")

        # Open circuit
        for _ in range(2):
            with pytest.raises(Exception, match="Test error"):
                await breaker.call(failing_func)

        # Wait for recovery
        await asyncio.sleep(0.2)
        assert breaker.state == ClientState.DEGRADED

        # Fail in half-open state
        with pytest.raises(Exception, match="Test error"):
            await breaker.call(failing_func)

        # Circuit should be open again
        with pytest.raises(APIError, match="Circuit breaker is open"):
            await breaker.call(failing_func)


class TestClientManager:
    """Test client manager functionality."""

    def test_singleton_pattern(self, client_config):
        """Test singleton pattern implementation."""
        # Reset singleton
        ClientManager._instance = None

        manager1 = ClientManager(client_config)
        manager2 = ClientManager(client_config)

        assert manager1 is manager2
        assert ClientManager._instance is manager1

    @pytest.mark.asyncio
    async def test_initialize_cleanup(self, client_manager):
        """Test initialization and cleanup."""
        assert not client_manager._initialized

        await client_manager.initialize()
        assert client_manager._initialized
        assert client_manager._health_check_task is not None

        await client_manager.cleanup()
        assert not client_manager._initialized
        assert len(client_manager._clients) == 0

    @pytest.mark.asyncio
    async def test_get_qdrant_client_success(self, client_manager):
        """Test successful Qdrant client creation."""
        mock_client = AsyncMock(spec=AsyncQdrantClient)
        mock_client.get_collections = AsyncMock(return_value=[])

        with patch(
            "src.infrastructure.client_manager.AsyncQdrantClient",
            return_value=mock_client,
        ):
            client = await client_manager.get_qdrant_client()
            assert client is mock_client
            assert "qdrant" in client_manager._clients
            assert client_manager._health["qdrant"].state == ClientState.HEALTHY

            # Second call should return same instance
            client2 = await client_manager.get_qdrant_client()
            assert client2 is client

    @pytest.mark.asyncio
    async def test_get_qdrant_client_failure(self, client_manager):
        """Test Qdrant client creation failure."""
        with (
            patch(
                "src.infrastructure.client_manager.AsyncQdrantClient",
                side_effect=Exception("Connection failed"),
            ),
            pytest.raises(APIError, match="Failed to create qdrant client"),
        ):
            await client_manager.get_qdrant_client()

    @pytest.mark.asyncio
    async def test_get_openai_client_no_key(self, client_manager):
        """Test OpenAI client returns None without API key."""
        client_manager.config.openai_api_key = None
        client = await client_manager.get_openai_client()
        assert client is None

    @pytest.mark.asyncio
    async def test_get_openai_client_success(self, client_manager):
        """Test successful OpenAI client creation."""
        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_client.models = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=[])

        with patch(
            "src.infrastructure.client_manager.AsyncOpenAI",
            return_value=mock_client,
        ):
            client = await client_manager.get_openai_client()
            assert client is mock_client
            assert "openai" in client_manager._clients

    @pytest.mark.asyncio
    async def test_get_redis_client_success(self, client_manager):
        """Test successful Redis client creation."""
        mock_client = AsyncMock(spec=redis.Redis)
        mock_client.ping = AsyncMock(return_value=True)

        with patch(
            "redis.asyncio.from_url",
            return_value=mock_client,
        ):
            client = await client_manager.get_redis_client()
            assert client is mock_client
            assert "redis" in client_manager._clients

    @pytest.mark.asyncio
    async def test_health_check_loop(self, client_manager):
        """Test health check loop functionality."""
        # Create mock clients
        mock_qdrant = AsyncMock()
        mock_qdrant.get_collections = AsyncMock(return_value=[])
        client_manager._clients["qdrant"] = mock_qdrant
        client_manager._health["qdrant"] = ClientHealth(
            state=ClientState.HEALTHY,
            last_check=time.time(),
        )

        # Initialize to start health check loop
        await client_manager.initialize()

        # Wait for health check
        await asyncio.sleep(1.5)

        # Verify health check was called
        assert mock_qdrant.get_collections.called

    @pytest.mark.asyncio
    async def test_health_check_failure_handling(self, client_manager):
        """Test health check failure handling."""
        # Create failing mock client
        mock_qdrant = AsyncMock()
        mock_qdrant.get_collections = AsyncMock(
            side_effect=Exception("Health check failed")
        )
        client_manager._clients["qdrant"] = mock_qdrant
        client_manager._health["qdrant"] = ClientHealth(
            state=ClientState.HEALTHY,
            last_check=time.time(),
        )

        # Initialize circuit breaker
        client_manager._circuit_breakers["qdrant"] = CircuitBreaker(failure_threshold=3)

        # Run health check
        is_healthy = await client_manager._check_qdrant_health()
        assert not is_healthy

    @pytest.mark.asyncio
    async def test_managed_client_context_manager(self, client_manager):
        """Test managed client context manager."""
        mock_client = AsyncMock()
        mock_client.get_collections = AsyncMock(return_value=[])

        with patch(
            "src.infrastructure.client_manager.AsyncQdrantClient",
            return_value=mock_client,
        ):
            async with client_manager.managed_client("qdrant") as client:
                assert client is mock_client

    @pytest.mark.asyncio
    async def test_managed_client_invalid_type(self, client_manager):
        """Test managed client with invalid type."""
        with pytest.raises(ValueError, match="Unknown client type"):
            async with client_manager.managed_client("invalid"):
                pass

    @pytest.mark.asyncio
    async def test_get_health_status(self, client_manager):
        """Test getting health status."""
        # Add some health data
        client_manager._health["qdrant"] = ClientHealth(
            state=ClientState.HEALTHY,
            last_check=time.time(),
            consecutive_failures=0,
        )
        client_manager._health["redis"] = ClientHealth(
            state=ClientState.DEGRADED,
            last_check=time.time(),
            last_error="Connection timeout",
            consecutive_failures=2,
        )

        status = await client_manager.get_health_status()

        assert "qdrant" in status
        assert status["qdrant"]["state"] == "healthy"
        assert status["qdrant"]["consecutive_failures"] == 0

        assert "redis" in status
        assert status["redis"]["state"] == "degraded"
        assert status["redis"]["last_error"] == "Connection timeout"
        assert status["redis"]["consecutive_failures"] == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, client_manager):
        """Test circuit breaker integration with client manager."""
        client_manager.config.circuit_breaker_failure_threshold = 2

        # Mock failing client
        with patch(
            "src.infrastructure.client_manager.AsyncQdrantClient",
            side_effect=Exception("Connection failed"),
        ):
            # First failure
            with pytest.raises(APIError):
                await client_manager.get_qdrant_client()

            # Second failure (should open circuit)
            with pytest.raises(APIError):
                await client_manager.get_qdrant_client()

            # Circuit should be open
            with pytest.raises(APIError, match="qdrant client circuit breaker is open"):
                await client_manager.get_qdrant_client()

    def test_config_change_validation(self, client_config):
        """Test that config changes are properly rejected."""
        # Reset singleton
        ClientManager._instance = None

        # Create manager with first config
        manager1 = ClientManager(client_config)
        manager1._initialized = True  # Simulate initialization

        # Try to create with different config
        different_config = ClientManagerConfig(
            qdrant_url="http://different:6333",
            redis_url="redis://different:6379",
        )

        with pytest.raises(
            ValueError, match="already initialized with different config"
        ):
            ClientManager(different_config)

    @pytest.mark.asyncio
    async def test_concurrent_client_creation(self, client_manager):
        """Test concurrent client creation uses singleton properly."""
        mock_client = AsyncMock()
        mock_client.get_collections = AsyncMock(return_value=[])
        call_count = 0

        def create_client(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_client

        with patch(
            "src.infrastructure.client_manager.AsyncQdrantClient",
            side_effect=create_client,
        ):
            # Create multiple concurrent requests
            tasks = [client_manager.get_qdrant_client() for _ in range(10)]
            clients = await asyncio.gather(*tasks)

            # All should be the same instance
            assert all(c is mock_client for c in clients)
            # Should only create once
            assert call_count == 1

    @pytest.mark.asyncio
    async def test_client_recreation_on_recovery(self, client_manager):
        """Test that clients are recreated when they recover from failure."""
        # Create a mock client that will be "recreated"
        original_client = AsyncMock()
        original_client.get_collections = AsyncMock(return_value=[])
        original_client.close = AsyncMock()

        # Add client to manager
        client_manager._clients["qdrant"] = original_client
        client_manager._health["qdrant"] = ClientHealth(
            state=ClientState.FAILED,
            last_check=time.time(),
            consecutive_failures=5,
        )
        client_manager._circuit_breakers["qdrant"] = CircuitBreaker(failure_threshold=3)
        client_manager._circuit_breakers["qdrant"]._state = ClientState.FAILED

        # Mock successful health check
        async def mock_health_check():
            return True

        # Run single health check to trigger recreation
        await client_manager._run_single_health_check("qdrant", mock_health_check)

        # Verify client was removed (forcing recreation on next access)
        assert "qdrant" not in client_manager._clients

        # Verify health status was updated
        health = client_manager._health["qdrant"]
        assert health.state == ClientState.HEALTHY
        assert health.consecutive_failures == 0

        # Verify circuit breaker was reset
        breaker = client_manager._circuit_breakers["qdrant"]
        assert breaker._state == ClientState.HEALTHY
        assert breaker._failure_count == 0
