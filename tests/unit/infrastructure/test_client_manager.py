"""Unit tests for infrastructure client_manager module."""

import asyncio
import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from src.infrastructure.client_manager import CircuitBreaker
from src.infrastructure.client_manager import ClientHealth
from src.infrastructure.client_manager import ClientManager
from src.infrastructure.client_manager import ClientManagerConfig
from src.infrastructure.client_manager import ClientState
from src.services.errors import APIError


class TestClientState:
    """Test cases for ClientState enum."""

    def test_client_state_values(self):
        """Test ClientState enum values."""
        assert ClientState.UNINITIALIZED.value == "uninitialized"
        assert ClientState.HEALTHY.value == "healthy"
        assert ClientState.DEGRADED.value == "degraded"
        assert ClientState.FAILED.value == "failed"

    def test_client_state_members(self):
        """Test ClientState enum members."""
        expected_states = {"UNINITIALIZED", "HEALTHY", "DEGRADED", "FAILED"}
        actual_states = {state.name for state in ClientState}
        assert actual_states == expected_states


class TestClientHealth:
    """Test cases for ClientHealth dataclass."""

    def test_client_health_initialization(self):
        """Test ClientHealth initialization."""
        health = ClientHealth(
            state=ClientState.HEALTHY,
            last_check=1234567.0,
            last_error="test error",
            consecutive_failures=2,
        )

        assert health.state == ClientState.HEALTHY
        assert health.last_check == 1234567.0
        assert health.last_error == "test error"
        assert health.consecutive_failures == 2

    def test_client_health_default_values(self):
        """Test ClientHealth default values."""
        health = ClientHealth(state=ClientState.DEGRADED, last_check=1234567.0)

        assert health.state == ClientState.DEGRADED
        assert health.last_check == 1234567.0
        assert health.last_error is None
        assert health.consecutive_failures == 0


class TestClientManagerConfig:
    """Test cases for ClientManagerConfig."""

    def test_config_default_values(self):
        """Test default configuration values."""
        config = ClientManagerConfig()

        # Qdrant defaults
        assert config.qdrant_url == "http://localhost:6333"
        assert config.qdrant_api_key is None
        assert config.qdrant_timeout == 30.0
        assert config.qdrant_prefer_grpc is False

        # OpenAI defaults
        assert config.openai_api_key is None
        assert config.openai_timeout == 30.0
        assert config.openai_max_retries == 3

        # Firecrawl defaults
        assert config.firecrawl_api_key is None
        assert config.firecrawl_timeout == 60.0

        # Redis defaults
        assert config.redis_url == "redis://localhost:6379"
        assert config.redis_max_connections == 10
        assert config.redis_decode_responses is True

        # Health check defaults
        assert config.health_check_interval == 30.0
        assert config.health_check_timeout == 5.0
        assert config.max_consecutive_failures == 3

        # Circuit breaker defaults
        assert config.circuit_breaker_failure_threshold == 5
        assert config.circuit_breaker_recovery_timeout == 60.0
        assert config.circuit_breaker_half_open_requests == 1

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = ClientManagerConfig(
            qdrant_url="https://custom-qdrant:6333",
            qdrant_timeout=45.0,
            openai_timeout=60.0,
            redis_max_connections=20,
            health_check_interval=60.0,
        )

        assert config.qdrant_url == "https://custom-qdrant:6333"
        assert config.qdrant_timeout == 45.0
        assert config.openai_timeout == 60.0
        assert config.redis_max_connections == 20
        assert config.health_check_interval == 60.0

    def test_qdrant_url_validation_valid(self):
        """Test valid Qdrant URL validation."""
        # HTTP URL
        config = ClientManagerConfig(qdrant_url="http://localhost:6333")
        assert config.qdrant_url == "http://localhost:6333"

        # HTTPS URL
        config = ClientManagerConfig(qdrant_url="https://qdrant.example.com")
        assert config.qdrant_url == "https://qdrant.example.com"

    def test_qdrant_url_validation_invalid(self):
        """Test invalid Qdrant URL validation."""
        with pytest.raises(ValidationError) as exc_info:
            ClientManagerConfig(qdrant_url="ftp://invalid.com")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "Qdrant URL must start with http:// or https://" in str(errors[0])

    def test_redis_url_validation_valid(self):
        """Test valid Redis URL validation."""
        # Redis URL
        config = ClientManagerConfig(redis_url="redis://localhost:6379")
        assert config.redis_url == "redis://localhost:6379"

        # Secure Redis URL
        config = ClientManagerConfig(redis_url="rediss://secure.redis.com:6380")
        assert config.redis_url == "rediss://secure.redis.com:6380"

    def test_redis_url_validation_invalid(self):
        """Test invalid Redis URL validation."""
        with pytest.raises(ValidationError) as exc_info:
            ClientManagerConfig(redis_url="http://invalid.com")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "Redis URL must start with redis:// or rediss://" in str(errors[0])

    def test_positive_number_validations(self):
        """Test positive number field validations."""
        # Test that positive values work
        config = ClientManagerConfig(
            qdrant_timeout=1.0,
            health_check_interval=1.0,
            circuit_breaker_failure_threshold=1,
        )
        assert config.qdrant_timeout == 1.0

        # Test that zero/negative values fail
        with pytest.raises(ValidationError):
            ClientManagerConfig(qdrant_timeout=0.0)

        with pytest.raises(ValidationError):
            ClientManagerConfig(health_check_interval=-1.0)

        with pytest.raises(ValidationError):
            ClientManagerConfig(circuit_breaker_failure_threshold=0)


class TestCircuitBreaker:
    """Test cases for CircuitBreaker class."""

    def test_circuit_breaker_initialization(self):
        """Test CircuitBreaker initialization."""
        breaker = CircuitBreaker(
            failure_threshold=3, recovery_timeout=30.0, half_open_requests=2
        )

        assert breaker.failure_threshold == 3
        assert breaker.recovery_timeout == 30.0
        assert breaker.half_open_requests == 2
        assert breaker._failure_count == 0
        assert breaker._last_failure_time is None
        assert breaker._state == ClientState.HEALTHY
        assert breaker._half_open_attempts == 0

    def test_circuit_breaker_state_healthy(self):
        """Test circuit breaker state when healthy."""
        breaker = CircuitBreaker()
        assert breaker.state == ClientState.HEALTHY

    def test_circuit_breaker_state_failed(self):
        """Test circuit breaker state when failed."""
        breaker = CircuitBreaker()
        breaker._state = ClientState.FAILED
        breaker._last_failure_time = time.time()
        assert breaker.state == ClientState.FAILED

    def test_circuit_breaker_state_half_open(self):
        """Test circuit breaker state transitions to half-open."""
        breaker = CircuitBreaker(recovery_timeout=0.1)
        breaker._state = ClientState.FAILED
        breaker._last_failure_time = time.time() - 0.2  # Past recovery timeout
        assert breaker.state == ClientState.DEGRADED

    @pytest.mark.asyncio
    async def test_circuit_breaker_call_success(self):
        """Test circuit breaker with successful function call."""
        breaker = CircuitBreaker()
        mock_func = AsyncMock(return_value="success")

        result = await breaker.call(mock_func, "arg1", keyword="arg2")

        assert result == "success"
        mock_func.assert_called_once_with("arg1", keyword="arg2")
        assert breaker._failure_count == 0
        assert breaker._state == ClientState.HEALTHY

    @pytest.mark.asyncio
    async def test_circuit_breaker_call_failure(self):
        """Test circuit breaker with function call failure."""
        breaker = CircuitBreaker(failure_threshold=2)
        mock_func = AsyncMock(side_effect=Exception("Test error"))

        # First failure
        with pytest.raises(Exception, match="Test error"):
            await breaker.call(mock_func)

        assert breaker._failure_count == 1
        assert breaker._state == ClientState.HEALTHY

        # Second failure - should open circuit
        with pytest.raises(Exception, match="Test error"):
            await breaker.call(mock_func)

        assert breaker._failure_count == 2
        assert breaker._state == ClientState.FAILED

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_blocks_calls(self):
        """Test that open circuit breaker blocks calls."""
        breaker = CircuitBreaker(failure_threshold=1)
        mock_func = AsyncMock(side_effect=Exception("Test error"))

        # Trigger circuit opening
        with pytest.raises(Exception, match="Test error"):
            await breaker.call(mock_func)

        assert breaker._state == ClientState.FAILED

        # Subsequent calls should be blocked
        with pytest.raises(APIError, match="Circuit breaker is open"):
            await breaker.call(mock_func)

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_success(self):
        """Test circuit breaker half-open state with successful call."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        mock_func = AsyncMock(side_effect=Exception("Test error"))

        # Open circuit
        with pytest.raises(Exception, match="Test error"):
            await breaker.call(mock_func)

        assert breaker._state == ClientState.FAILED

        # Wait for recovery
        await asyncio.sleep(0.15)
        assert breaker.state == ClientState.DEGRADED

        # Successful call should close circuit
        mock_func.side_effect = None
        mock_func.return_value = "success"

        result = await breaker.call(mock_func)

        assert result == "success"
        assert breaker._state == ClientState.HEALTHY
        assert breaker._failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_failure(self):
        """Test circuit breaker half-open state with failed call."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        mock_func = AsyncMock(side_effect=Exception("Test error"))

        # Open circuit
        with pytest.raises(Exception, match="Test error"):
            await breaker.call(mock_func)

        # Wait for recovery
        await asyncio.sleep(0.15)
        assert breaker.state == ClientState.DEGRADED

        # Failed call should re-open circuit
        with pytest.raises(Exception, match="Test error"):
            await breaker.call(mock_func)

        assert breaker._state == ClientState.FAILED

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_request_limit(self):
        """Test circuit breaker half-open request limit."""
        breaker = CircuitBreaker(
            failure_threshold=1, recovery_timeout=0.1, half_open_requests=1
        )
        mock_func = AsyncMock(side_effect=Exception("Test error"))

        # Open circuit
        with pytest.raises(Exception, match="Test error"):
            await breaker.call(mock_func)

        # Wait for recovery
        await asyncio.sleep(0.15)

        # First half-open attempt
        breaker._half_open_attempts = 1  # Simulate reaching limit

        with pytest.raises(APIError, match="half-open test failed"):
            await breaker.call(mock_func)


class TestClientManagerSingleton:
    """Test cases for ClientManager singleton behavior."""

    def test_singleton_pattern(self):
        """Test that ClientManager follows singleton pattern."""
        # Clear any existing instance
        ClientManager._instance = None

        config = ClientManagerConfig()
        manager1 = ClientManager(config)
        manager2 = ClientManager()

        assert manager1 is manager2
        assert ClientManager._instance is manager1

    def test_singleton_config_validation(self):
        """Test singleton config validation on re-initialization."""
        # Clear any existing instance
        ClientManager._instance = None

        config1 = ClientManagerConfig(qdrant_timeout=30.0)
        config2 = ClientManagerConfig(qdrant_timeout=60.0)

        manager1 = ClientManager(config1)
        manager1._initialized = True  # Mark as initialized

        # Should raise error with different config
        with pytest.raises(
            ValueError, match="already initialized with different config"
        ):
            ClientManager(config2)

    @pytest.mark.asyncio
    async def test_singleton_cleanup_and_reinitialize(self):
        """Test singleton cleanup and reinitialization."""
        # Clear any existing instance
        ClientManager._instance = None

        config = ClientManagerConfig()
        manager = ClientManager(config)
        await manager.initialize()

        assert manager._initialized is True

        await manager.cleanup()
        assert manager._initialized is False

        # Should be able to reinitialize
        await manager.initialize()
        assert manager._initialized is True


class TestClientManagerFactoryMethod:
    """Test cases for ClientManager factory method."""

    @patch("src.config.loader.ConfigLoader")
    @patch.dict(
        "os.environ",
        {
            "QDRANT_URL": "http://test-qdrant:6333",
            "OPENAI_API_KEY": "test-openai-key",
            "FIRECRAWL_API_KEY": "test-firecrawl-key",
            "REDIS_URL": "redis://test-redis:6379",
        },
    )
    def test_from_unified_config(self, mock_config_loader):
        """Test ClientManager creation from unified config."""
        # Clear singleton
        ClientManager._instance = None

        mock_unified_config = MagicMock()
        mock_config_loader.load_config.return_value = mock_unified_config

        manager = ClientManager.from_unified_config()

        assert manager.config.qdrant_url == "http://test-qdrant:6333"
        assert manager.config.openai_api_key == "test-openai-key"
        assert manager.config.firecrawl_api_key == "test-firecrawl-key"
        assert manager.config.redis_url == "redis://test-redis:6379"
        assert manager.unified_config is mock_unified_config
        mock_config_loader.load_config.assert_called_once()

    @patch("src.config.loader.ConfigLoader")
    def test_from_unified_config_defaults(self, mock_config_loader):
        """Test ClientManager creation with default environment values."""
        # Clear singleton
        ClientManager._instance = None

        mock_unified_config = MagicMock()
        mock_config_loader.load_config.return_value = mock_unified_config

        with patch.dict("os.environ", {}, clear=True):
            manager = ClientManager.from_unified_config()

        assert manager.config.qdrant_url == "http://localhost:6333"
        assert manager.config.openai_api_key is None
        assert manager.config.redis_url == "redis://localhost:6379"


class TestClientManagerInitialization:
    """Test cases for ClientManager initialization."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test client manager initialization."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig()
        manager = ClientManager(config)

        assert manager._initialized is False
        assert manager._health_check_task is None

        await manager.initialize()

        assert manager._initialized is True
        assert manager._health_check_task is not None
        assert not manager._health_check_task.done()

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_double_initialization(self):
        """Test that double initialization is safe."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig()
        manager = ClientManager(config)

        await manager.initialize()
        first_task = manager._health_check_task

        # Second initialization should be no-op
        await manager.initialize()
        assert manager._health_check_task is first_task

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test client manager cleanup."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig()
        manager = ClientManager(config)
        await manager.initialize()

        # Add mock clients to test cleanup
        mock_client1 = AsyncMock()
        mock_client2 = AsyncMock()
        mock_client2.close = AsyncMock()

        manager._clients["test1"] = mock_client1
        manager._clients["test2"] = mock_client2
        manager._health["test1"] = ClientHealth(ClientState.HEALTHY, time.time())

        await manager.cleanup()

        assert len(manager._clients) == 0
        assert len(manager._health) == 0
        assert len(manager._circuit_breakers) == 0
        assert manager._initialized is False
        mock_client2.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_with_client_close_error(self):
        """Test cleanup handles client close errors gracefully."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig()
        manager = ClientManager(config)

        # Add mock client that raises error on close
        mock_client = AsyncMock()
        mock_client.close = AsyncMock(side_effect=Exception("Close error"))
        manager._clients["test"] = mock_client

        # Should not raise exception
        await manager.cleanup()
        assert len(manager._clients) == 0


class TestClientManagerClientCreation:
    """Test cases for client creation and retrieval."""

    @pytest.mark.asyncio
    async def test_get_qdrant_client_success(self):
        """Test successful Qdrant client creation."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig()
        manager = ClientManager(config)

        with patch.object(manager, "_create_qdrant_client") as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client

            client = await manager.get_qdrant_client()

            assert client is mock_client
            assert "qdrant" in manager._clients
            assert "qdrant" in manager._health
            assert manager._health["qdrant"].state == ClientState.HEALTHY
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_openai_client_with_api_key(self):
        """Test OpenAI client creation with API key."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig(openai_api_key="test-key")
        manager = ClientManager(config)

        with patch.object(manager, "_create_openai_client") as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client

            client = await manager.get_openai_client()

            assert client is mock_client
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_openai_client_without_api_key(self):
        """Test OpenAI client returns None without API key."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig(openai_api_key=None)
        manager = ClientManager(config)

        client = await manager.get_openai_client()
        assert client is None

    @pytest.mark.asyncio
    async def test_get_firecrawl_client_with_api_key(self):
        """Test Firecrawl client creation with API key."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig(firecrawl_api_key="test-key")
        manager = ClientManager(config)

        with patch.object(manager, "_create_firecrawl_client") as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client

            client = await manager.get_firecrawl_client()

            assert client is mock_client
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_firecrawl_client_without_api_key(self):
        """Test Firecrawl client returns None without API key."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig(firecrawl_api_key=None)
        manager = ClientManager(config)

        client = await manager.get_firecrawl_client()
        assert client is None

    @pytest.mark.asyncio
    async def test_get_redis_client_success(self):
        """Test successful Redis client creation."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig()
        manager = ClientManager(config)

        with patch.object(manager, "_create_redis_client") as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client

            client = await manager.get_redis_client()

            assert client is mock_client
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_creation_with_circuit_breaker_failure(self):
        """Test client creation failure with circuit breaker."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig()
        manager = ClientManager(config)

        with patch.object(manager, "_create_qdrant_client") as mock_create:
            mock_create.side_effect = Exception("Connection failed")

            with pytest.raises(APIError, match="Failed to create qdrant client"):
                await manager.get_qdrant_client()

            assert "qdrant" not in manager._clients

    @pytest.mark.asyncio
    async def test_client_reuse(self):
        """Test that clients are reused after creation."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig()
        manager = ClientManager(config)

        with patch.object(manager, "_create_qdrant_client") as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client

            client1 = await manager.get_qdrant_client()
            client2 = await manager.get_qdrant_client()

            assert client1 is client2
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_unhealthy_client_error(self):
        """Test error when client is unhealthy."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig(max_consecutive_failures=2)
        manager = ClientManager(config)

        # Create client first
        with patch.object(manager, "_create_qdrant_client") as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client
            await manager.get_qdrant_client()

        # Mark as failed
        manager._health["qdrant"] = ClientHealth(
            state=ClientState.FAILED,
            last_check=time.time(),
            last_error="Health check failed",
            consecutive_failures=3,
        )

        with pytest.raises(APIError, match="qdrant client is unhealthy"):
            await manager.get_qdrant_client()


class TestClientManagerHealthChecks:
    """Test cases for health check functionality."""

    @pytest.mark.asyncio
    async def test_qdrant_health_check_success(self):
        """Test successful Qdrant health check."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig()
        manager = ClientManager(config)

        mock_client = AsyncMock()
        mock_client.get_collections = AsyncMock()
        manager._clients["qdrant"] = mock_client

        result = await manager._check_qdrant_health()

        assert result is True
        mock_client.get_collections.assert_called_once()

    @pytest.mark.asyncio
    async def test_qdrant_health_check_failure(self):
        """Test failed Qdrant health check."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig()
        manager = ClientManager(config)

        mock_client = AsyncMock()
        mock_client.get_collections = AsyncMock(
            side_effect=Exception("Connection error")
        )
        manager._clients["qdrant"] = mock_client

        result = await manager._check_qdrant_health()

        assert result is False

    @pytest.mark.asyncio
    async def test_openai_health_check_success(self):
        """Test successful OpenAI health check."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig()
        manager = ClientManager(config)

        mock_client = AsyncMock()
        mock_client.models = AsyncMock()
        mock_client.models.list = AsyncMock()
        manager._clients["openai"] = mock_client

        result = await manager._check_openai_health()

        assert result is True
        mock_client.models.list.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_health_check_success(self):
        """Test successful Redis health check."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig()
        manager = ClientManager(config)

        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        manager._clients["redis"] = mock_client

        result = await manager._check_redis_health()

        assert result is True
        mock_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_firecrawl_health_check(self):
        """Test Firecrawl health check (always returns True if client exists)."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig()
        manager = ClientManager(config)

        mock_client = AsyncMock()
        manager._clients["firecrawl"] = mock_client

        result = await manager._check_firecrawl_health()
        assert result is True

        # Test with no client
        del manager._clients["firecrawl"]
        result = await manager._check_firecrawl_health()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_status_reporting(self):
        """Test health status reporting."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig()
        manager = ClientManager(config)

        # Add mock health data
        manager._health["qdrant"] = ClientHealth(
            state=ClientState.HEALTHY,
            last_check=1234567.0,
            last_error=None,
            consecutive_failures=0,
        )
        manager._circuit_breakers["qdrant"] = CircuitBreaker()

        status = await manager.get_health_status()

        assert "qdrant" in status
        assert status["qdrant"]["state"] == "healthy"
        assert status["qdrant"]["last_check"] == 1234567.0
        assert status["qdrant"]["last_error"] is None
        assert status["qdrant"]["consecutive_failures"] == 0
        assert status["qdrant"]["circuit_breaker_state"] == "healthy"


class TestClientManagerManagedClient:
    """Test cases for managed client context manager."""

    @pytest.mark.asyncio
    async def test_managed_client_qdrant(self):
        """Test managed client context manager for Qdrant."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig()
        manager = ClientManager(config)

        with patch.object(manager, "get_qdrant_client") as mock_get:
            mock_client = AsyncMock()
            mock_get.return_value = mock_client

            async with manager.managed_client("qdrant") as client:
                assert client is mock_client

            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_managed_client_invalid_type(self):
        """Test managed client with invalid client type."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig()
        manager = ClientManager(config)

        with pytest.raises(ValueError, match="Unknown client type: invalid"):
            async with manager.managed_client("invalid"):
                pass

    @pytest.mark.asyncio
    async def test_managed_client_exception_handling(self):
        """Test managed client exception handling."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig()
        manager = ClientManager(config)

        with patch.object(manager, "get_qdrant_client") as mock_get:
            mock_get.side_effect = Exception("Client creation failed")

            with pytest.raises(Exception, match="Client creation failed"):
                async with manager.managed_client("qdrant"):
                    pass


class TestClientManagerAsyncContextManager:
    """Test cases for async context manager functionality."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test ClientManager as async context manager."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig()

        async with ClientManager(config) as manager:
            assert manager._initialized is True

        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_async_context_manager_exception(self):
        """Test async context manager with exception."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig()

        try:
            async with ClientManager(config) as manager:
                assert manager._initialized is True
                raise ValueError("Test error")
        except ValueError:
            pass

        assert manager._initialized is False


class TestClientManagerConcurrency:
    """Test cases for concurrent client access."""

    @pytest.mark.asyncio
    async def test_concurrent_client_creation(self):
        """Test concurrent client creation is thread-safe."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig()
        manager = ClientManager(config)

        create_call_count = 0

        async def mock_create_client():
            nonlocal create_call_count
            await asyncio.sleep(0.01)  # Simulate async work
            create_call_count += 1
            return AsyncMock()

        with patch.object(
            manager, "_create_qdrant_client", side_effect=mock_create_client
        ):
            # Create multiple concurrent requests
            tasks = [manager.get_qdrant_client() for _ in range(5)]
            clients = await asyncio.gather(*tasks)

            # All should return same client instance
            assert all(client is clients[0] for client in clients)
            # Client should only be created once
            assert create_call_count == 1


class TestClientManagerIntegration:
    """Integration tests for ClientManager."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test full client manager lifecycle."""
        # Clear singleton
        ClientManager._instance = None

        config = ClientManagerConfig(
            health_check_interval=0.1,  # Fast for testing
            max_consecutive_failures=2,
        )
        manager = ClientManager(config)

        try:
            await manager.initialize()

            # Mock client creation
            with patch.object(manager, "_create_qdrant_client") as mock_create:
                mock_client = AsyncMock()
                mock_client.get_collections = AsyncMock()
                mock_create.return_value = mock_client

                # Get client
                client = await manager.get_qdrant_client()
                assert client is mock_client

                # Verify health check runs
                await asyncio.sleep(0.15)

                # Check that health status is available
                status = await manager.get_health_status()
                assert "qdrant" in status

        finally:
            await manager.cleanup()
