"""Unit tests for infrastructure client_manager module."""

import asyncio
import time
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from src.config import UnifiedConfig
from src.infrastructure.client_manager import CircuitBreaker
from src.infrastructure.client_manager import ClientHealth
from src.infrastructure.client_manager import ClientManager
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

        config = UnifiedConfig()
        manager1 = ClientManager(config)
        manager2 = ClientManager()

        assert manager1 is manager2
        assert ClientManager._instance is manager1

    def test_singleton_config_validation(self):
        """Test singleton config validation on re-initialization."""
        # Clear any existing instance
        ClientManager._instance = None

        config1 = UnifiedConfig()
        config1.qdrant.timeout = 30.0
        config2 = UnifiedConfig()
        config2.qdrant.timeout = 60.0

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

        config = UnifiedConfig()
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
    def test_from_unified_config(self, mock_config_loader):
        """Test ClientManager creation from unified config."""
        # Clear singleton
        ClientManager._instance = None

        mock_unified_config = UnifiedConfig()
        mock_config_loader.load_config.return_value = mock_unified_config

        manager = ClientManager.from_unified_config()

        assert manager.config is mock_unified_config
        mock_config_loader.load_config.assert_called_once()


class TestClientManagerInitialization:
    """Test cases for ClientManager initialization."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test client manager initialization."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
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

        config = UnifiedConfig()
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

        config = UnifiedConfig()
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

        config = UnifiedConfig()
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

        config = UnifiedConfig()
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

        config = UnifiedConfig()
        config.openai.api_key = "test-key"
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

        config = UnifiedConfig()
        config.openai.api_key = None
        manager = ClientManager(config)

        client = await manager.get_openai_client()
        assert client is None

    @pytest.mark.asyncio
    async def test_get_firecrawl_client_with_api_key(self):
        """Test Firecrawl client creation with API key."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
        config.firecrawl.api_key = "test-key"
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

        config = UnifiedConfig()
        config.firecrawl.api_key = None
        manager = ClientManager(config)

        client = await manager.get_firecrawl_client()
        assert client is None

    @pytest.mark.asyncio
    async def test_get_redis_client_success(self):
        """Test successful Redis client creation."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
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

        config = UnifiedConfig()
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

        config = UnifiedConfig()
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

        config = UnifiedConfig()
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


class TestClientManagerServiceGetters:
    """Test cases for service getter methods."""

    @pytest.mark.asyncio
    async def test_get_qdrant_service(self):
        """Test QdrantService getter."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
        manager = ClientManager(config)

        with patch(
            "src.services.vector_db.service.QdrantService"
        ) as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.return_value = mock_service

            service = await manager.get_qdrant_service()

            assert service is mock_service
            mock_service_class.assert_called_once_with(config, client_manager=manager)
            mock_service.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_embedding_manager(self):
        """Test EmbeddingManager getter."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
        manager = ClientManager(config)

        with patch(
            "src.services.embeddings.manager.EmbeddingManager"
        ) as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager

            service = await manager.get_embedding_manager()

            assert service is mock_manager
            mock_manager_class.assert_called_once_with(
                config=config,
                client_manager=manager,
            )
            mock_manager.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cache_manager(self):
        """Test CacheManager getter."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
        manager = ClientManager(config)

        with patch("src.services.cache.manager.CacheManager") as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager

            service = await manager.get_cache_manager()

            assert service is mock_manager
            mock_manager_class.assert_called_once()
            mock_manager.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_service_singleton_behavior(self):
        """Test that services are singletons within ClientManager."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
        manager = ClientManager(config)

        with patch(
            "src.services.vector_db.service.QdrantService"
        ) as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.return_value = mock_service

            service1 = await manager.get_qdrant_service()
            service2 = await manager.get_qdrant_service()

            assert service1 is service2
            # Should only be created once
            mock_service_class.assert_called_once()
            mock_service.initialize.assert_called_once()


class TestClientManagerHealthChecks:
    """Test cases for health check functionality."""

    @pytest.mark.asyncio
    async def test_qdrant_health_check_success(self):
        """Test successful Qdrant health check."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
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

        config = UnifiedConfig()
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

        config = UnifiedConfig()
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

        config = UnifiedConfig()
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

        config = UnifiedConfig()
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

        config = UnifiedConfig()
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

        config = UnifiedConfig()
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

        config = UnifiedConfig()
        manager = ClientManager(config)

        with pytest.raises(ValueError, match="Unknown client type: invalid"):
            async with manager.managed_client("invalid"):
                pass

    @pytest.mark.asyncio
    async def test_managed_client_exception_handling(self):
        """Test managed client exception handling."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
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

        config = UnifiedConfig()

        async with ClientManager(config) as manager:
            assert manager._initialized is True

        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_async_context_manager_exception(self):
        """Test async context manager with exception."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()

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

        config = UnifiedConfig()
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

        config = UnifiedConfig()
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


class TestClientManagerErrorHandling:
    """Test error handling and edge cases in ClientManager."""

    @pytest.mark.asyncio
    async def test_client_creation_with_invalid_config(self):
        """Test client creation with invalid configuration."""
        # Clear singleton
        ClientManager._instance = None

        # Create config with invalid Qdrant URL
        config = UnifiedConfig()
        config.qdrant.url = "invalid-url"
        manager = ClientManager(config)

        with pytest.raises(APIError, match="Failed to create qdrant client"):
            await manager.get_qdrant_client()

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Test that circuit breaker opens after consecutive failures."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
        manager = ClientManager(config)

        # Mock client that always fails
        with patch.object(manager, "_create_qdrant_client") as mock_create:
            mock_create.side_effect = Exception("Connection failed")

            # Try to get client multiple times to trigger circuit breaker
            for _ in range(6):  # More than failure_threshold (5)
                with pytest.raises(APIError):
                    await manager.get_qdrant_client()

            # Circuit breaker should now be open
            with pytest.raises(APIError, match="circuit breaker is open"):
                await manager.get_qdrant_client()

    @pytest.mark.asyncio
    async def test_health_check_with_missing_client(self):
        """Test health check when client doesn't exist."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
        manager = ClientManager(config)

        # Test Qdrant health check without client
        result = await manager._check_qdrant_health()
        assert result is False

        # Test OpenAI health check without client
        result = await manager._check_openai_health()
        assert result is False

        # Test Redis health check without client
        result = await manager._check_redis_health()
        assert result is False

    @pytest.mark.asyncio
    async def test_service_initialization_failure(self):
        """Test service initialization failure handling."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
        manager = ClientManager(config)

        # Mock service class import to fail during service creation
        with patch("src.services.vector_db.service.QdrantService") as mock_service:
            mock_service.side_effect = Exception("Service init failed")

            with pytest.raises(Exception, match="Service init failed"):
                await manager.get_qdrant_service()

    @pytest.mark.asyncio
    async def test_client_creation_with_none_config_values(self):
        """Test client creation with None configuration values."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
        config.openai.api_key = None
        config.firecrawl.api_key = None
        manager = ClientManager(config)

        # OpenAI client should return None without API key
        openai_client = await manager.get_openai_client()
        assert openai_client is None

        # Firecrawl client should return None without API key
        firecrawl_client = await manager.get_firecrawl_client()
        assert firecrawl_client is None

    @pytest.mark.asyncio
    async def test_redis_connection_failure(self):
        """Test Redis connection failure handling."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
        config.cache.enable_dragonfly_cache = True
        config.cache.dragonfly_url = "redis://invalid:6379"
        manager = ClientManager(config)

        # Mock redis connection to fail
        with patch("redis.asyncio.from_url") as mock_redis:
            mock_redis.side_effect = Exception("Connection failed")

            with pytest.raises(APIError, match="Failed to create redis client"):
                await manager.get_redis_client()

    @pytest.mark.asyncio
    async def test_health_monitoring_lifecycle(self):
        """Test health monitoring during initialization and cleanup."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
        manager = ClientManager(config)

        # Initialize manager (starts health monitoring)
        await manager.initialize()
        assert manager._health_check_task is not None
        assert not manager._health_check_task.done()

        # Cleanup manager (stops health monitoring)
        await manager.cleanup()
        # Task should be cancelled, not None
        assert manager._health_check_task.cancelled()

    @pytest.mark.asyncio
    async def test_concurrent_service_access(self):
        """Test concurrent access to the same service."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
        manager = ClientManager(config)

        async def get_service():
            return await manager.get_qdrant_service()

        # Create multiple concurrent requests for the same service
        tasks = [get_service() for _ in range(10)]
        services = await asyncio.gather(*tasks)

        # All should return the same instance (singleton)
        for service in services[1:]:
            assert service is services[0]

    @pytest.mark.asyncio
    async def test_client_cleanup_with_errors(self):
        """Test cleanup when client.close() raises an error."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
        manager = ClientManager(config)

        # Create mock client that raises error on close
        mock_client = AsyncMock()
        mock_client.close = AsyncMock(side_effect=Exception("Close failed"))
        manager._clients["test_client"] = mock_client

        # Cleanup should handle the error gracefully
        await manager.cleanup()

        # Client should still be removed from the clients dict
        assert "test_client" not in manager._clients

    @pytest.mark.asyncio
    async def test_get_health_status_comprehensive(self):
        """Test comprehensive health status reporting."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
        config.openai.api_key = "test-key"
        config.cache.enable_dragonfly_cache = True
        manager = ClientManager(config)

        # Create some clients to populate health status
        await manager.get_qdrant_client()  # This will create qdrant client

        # Check that health status reflects created clients
        status = await manager.get_health_status()

        # Should have health status for available services
        assert isinstance(status, dict)
        if status:  # Health status may be empty initially
            for _service_name, service_status in status.items():
                assert "state" in service_status
                assert "last_check" in service_status

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_firecrawl_health_check_edge_cases(self):
        """Test Firecrawl health check edge cases."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
        manager = ClientManager(config)

        # Test with None client (returns False when no client exists)
        result = await manager._check_firecrawl_health()
        assert result is False  # Returns False when no client exists

        # Test with existing client (should return True)
        mock_client = AsyncMock()
        manager._clients["firecrawl"] = mock_client

        result = await manager._check_firecrawl_health()
        assert result is True  # Returns True when client exists


class TestClientManagerConfiguration:
    """Test ClientManager configuration handling."""

    @pytest.mark.asyncio
    async def test_from_unified_config_class_method(self):
        """Test creating ClientManager from UnifiedConfig."""
        # Clear singleton
        ClientManager._instance = None

        # Mock the config loader to return a known config
        with patch("src.config.loader.ConfigLoader.load_config") as mock_loader:
            config = UnifiedConfig()
            mock_loader.return_value = config

            manager = ClientManager.from_unified_config()

            assert isinstance(manager, ClientManager)
            assert manager.config is config
            mock_loader.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_client_type_in_managed_client(self):
        """Test managed_client with invalid client type."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
        manager = ClientManager(config)

        with pytest.raises(ValueError, match="Unknown client type"):
            async with manager.managed_client("invalid_type"):
                pass

    @pytest.mark.asyncio
    async def test_qdrant_client_recreation_after_failure(self):
        """Test that Qdrant client is recreated after circuit breaker recovery."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
        manager = ClientManager(config)

        # Initialize health status by creating a client first
        await manager.get_qdrant_client()

        # Now simulate circuit breaker opening
        manager._health["qdrant"].state = ClientState.FAILED
        manager._circuit_breakers["qdrant"]._state = "open"
        manager._circuit_breakers["qdrant"]._last_failure_time = (
            time.time() - 61
        )  # Past recovery timeout

        # Remove the existing client to force recreation
        del manager._clients["qdrant"]

        # Mock successful client creation on retry
        with patch.object(manager, "_create_qdrant_client") as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client

            client = await manager.get_qdrant_client()
            assert client is mock_client
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_embedding_manager_with_different_configs(self):
        """Test EmbeddingManager creation with different configurations."""
        # Clear singleton
        ClientManager._instance = None

        config = UnifiedConfig()
        config.openai.api_key = "test-key"
        config.openai.model = "text-embedding-3-large"
        manager = ClientManager(config)

        embedding_manager = await manager.get_embedding_manager()
        assert embedding_manager is not None

        # Should return same instance on second call
        embedding_manager2 = await manager.get_embedding_manager()
        assert embedding_manager is embedding_manager2
