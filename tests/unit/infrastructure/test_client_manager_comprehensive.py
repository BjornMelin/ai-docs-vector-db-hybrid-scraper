"""Comprehensive tests for ClientManager with 90%+ coverage."""

import asyncio
import contextlib
import time
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.infrastructure.client_manager import CircuitBreaker
from src.infrastructure.client_manager import ClientHealth
from src.infrastructure.client_manager import ClientManager
from src.config.models import UnifiedConfig
from src.infrastructure.client_manager import ClientState
from src.services.errors import APIError


class TestClientManagerServiceGetters:
    """Test the service getter methods added for MCP tools refactoring."""

    @pytest.fixture
    async def client_manager(self):
        """Create a ClientManager instance for testing."""
        config = UnifiedConfig()
        manager = ClientManager(config)
        manager.unified_config = Mock()  # Mock unified config
        await manager.initialize()
        yield manager
        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_get_qdrant_service(self, client_manager):
        """Test QdrantService getter method."""
        with patch("src.services.vector_db.service.QdrantService") as mock_service:
            mock_instance = AsyncMock()
            mock_service.return_value = mock_instance

            # First call should create the service
            service1 = await client_manager.get_qdrant_service()
            assert service1 == mock_instance
            mock_service.assert_called_once_with(client_manager.unified_config)
            mock_instance.initialize.assert_called_once()

            # Second call should return the same instance
            service2 = await client_manager.get_qdrant_service()
            assert service2 == service1
            # Should not create a new instance
            assert mock_service.call_count == 1

    @pytest.mark.asyncio
    async def test_get_embedding_manager(self, client_manager):
        """Test EmbeddingManager getter method."""
        with patch("src.services.embeddings.manager.EmbeddingManager") as mock_manager:
            mock_instance = Mock()
            mock_manager.return_value = mock_instance

            service = await client_manager.get_embedding_manager()
            assert service == mock_instance
            mock_manager.assert_called_once_with(client_manager.unified_config)

    @pytest.mark.asyncio
    async def test_get_cache_manager(self, client_manager):
        """Test CacheManager getter method."""
        with patch("src.services.cache.manager.CacheManager") as mock_manager:
            mock_instance = Mock()
            mock_manager.return_value = mock_instance

            service = await client_manager.get_cache_manager()
            assert service == mock_instance
            mock_manager.assert_called_once_with(client_manager)

    @pytest.mark.asyncio
    async def test_get_crawl_manager(self, client_manager):
        """Test CrawlManager getter method."""
        with patch("src.services.crawling.manager.CrawlManager") as mock_manager:
            mock_instance = Mock()
            mock_manager.return_value = mock_instance

            service = await client_manager.get_crawl_manager()
            assert service == mock_instance
            mock_manager.assert_called_once_with(client_manager.unified_config)

    @pytest.mark.asyncio
    async def test_get_hyde_engine(self, client_manager):
        """Test HyDE engine getter method."""
        with patch("src.services.hyde.engine.HyDEQueryEngine") as mock_engine:
            mock_instance = Mock()
            mock_engine.return_value = mock_instance

            service = await client_manager.get_hyde_engine()
            assert service == mock_instance
            mock_engine.assert_called_once_with(client_manager.unified_config)

    @pytest.mark.asyncio
    async def test_get_project_storage(self, client_manager):
        """Test ProjectStorage getter method."""
        with patch("src.services.core.project_storage.ProjectStorage") as mock_storage:
            mock_instance = Mock()
            mock_storage.return_value = mock_instance

            service = await client_manager.get_project_storage()
            assert service == mock_instance
            mock_storage.assert_called_once_with(client_manager.unified_config)

    @pytest.mark.asyncio
    async def test_get_alias_manager(self, client_manager):
        """Test QdrantAliasManager getter method."""
        with patch(
            "src.services.core.qdrant_alias_manager.QdrantAliasManager"
        ) as mock_manager:
            mock_instance = Mock()
            mock_manager.return_value = mock_instance

            service = await client_manager.get_alias_manager()
            assert service == mock_instance
            mock_manager.assert_called_once_with(client_manager.unified_config)

    @pytest.mark.asyncio
    async def test_get_blue_green_deployment(self, client_manager):
        """Test BlueGreenDeployment getter method."""
        with patch(
            "src.services.deployment.blue_green.BlueGreenDeployment"
        ) as mock_deploy:
            mock_instance = Mock()
            mock_deploy.return_value = mock_instance

            service = await client_manager.get_blue_green_deployment()
            assert service == mock_instance
            mock_deploy.assert_called_once_with(client_manager.unified_config)

    @pytest.mark.asyncio
    async def test_get_ab_testing(self, client_manager):
        """Test ABTesting getter method."""
        with patch("src.services.deployment.ab_testing.ABTestingManager") as mock_ab:
            mock_instance = Mock()
            mock_ab.return_value = mock_instance

            service = await client_manager.get_ab_testing()
            assert service == mock_instance
            mock_ab.assert_called_once_with(client_manager.unified_config)

    @pytest.mark.asyncio
    async def test_get_canary_deployment(self, client_manager):
        """Test CanaryDeployment getter method."""
        with patch("src.services.deployment.canary.CanaryDeployment") as mock_canary:
            mock_instance = Mock()
            mock_canary.return_value = mock_instance

            service = await client_manager.get_canary_deployment()
            assert service == mock_instance
            mock_canary.assert_called_once_with(client_manager.unified_config)


class TestClientManagerErrorScenarios:
    """Test error scenarios to improve coverage."""

    @pytest.fixture
    async def client_manager(self):
        """Create a ClientManager instance for testing."""
        config = ClientManagerConfig()
        manager = ClientManager(config)
        await manager.initialize()
        yield manager
        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_with_service_errors(self, client_manager):
        """Test cleanup when services raise exceptions."""
        # Set up mock services with cleanup methods that raise exceptions
        mock_service = Mock()
        mock_service.cleanup = AsyncMock(side_effect=Exception("Cleanup failed"))
        client_manager._qdrant_service = mock_service

        mock_client = Mock()
        mock_client.close = AsyncMock(side_effect=Exception("Close failed"))
        client_manager._clients["test"] = mock_client

        # Cleanup should not raise exceptions even if services fail
        await client_manager.cleanup()

        # Verify cleanup was attempted
        mock_service.cleanup.assert_called_once()
        mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_close_with_aclose_error(self, client_manager):
        """Test client cleanup when aclose method fails."""
        mock_client = Mock()
        mock_client.aclose = AsyncMock(side_effect=Exception("aclose failed"))
        del mock_client.close  # No close method, only aclose
        client_manager._clients["test"] = mock_client

        await client_manager.cleanup()
        mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state(self, client_manager):
        """Test circuit breaker in open state blocks requests."""
        # Create a circuit breaker and force it to open state
        breaker = CircuitBreaker(failure_threshold=1)
        client_manager._circuit_breakers["test"] = breaker

        # Force circuit breaker to open by simulating failures
        with contextlib.suppress(Exception):
            await breaker.call(lambda: (_ for _ in ()).throw(Exception("Test error")))

        # Now the circuit breaker should be open and block requests
        with pytest.raises(APIError, match="test client circuit breaker is open"):
            await client_manager._get_or_create_client(
                "test", lambda: Mock(), lambda: True
            )

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_failure(self):
        """Test circuit breaker half-open state failure handling."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        # Force circuit breaker to open
        for _ in range(2):
            with contextlib.suppress(Exception):
                await breaker.call(
                    lambda: (_ for _ in ()).throw(Exception("Test error"))
                )

        # Wait for recovery timeout
        await asyncio.sleep(0.2)

        # Should be in half-open state, test failure
        with pytest.raises(Exception, match="Test error"):
            await breaker.call(lambda: (_ for _ in ()).throw(Exception("Test error")))

    @pytest.mark.asyncio
    async def test_health_check_timeout(self, client_manager):
        """Test health check timeout handling."""

        async def slow_health_check():
            await asyncio.sleep(10)  # Longer than timeout
            return True

        client_manager._clients["test"] = Mock()
        client_manager.config.health_check_timeout = 0.1

        # This should trigger timeout handling
        await client_manager._run_single_health_check("test", slow_health_check)

        # Health should be marked as failed due to timeout
        assert client_manager._health["test"].state == ClientState.FAILED
        assert "timeout" in client_manager._health["test"].last_error.lower()

    @pytest.mark.asyncio
    async def test_health_check_exception_handling(self, client_manager):
        """Test health check exception handling."""

        async def failing_health_check():
            raise Exception("Health check failed")

        client_manager._clients["test"] = Mock()

        await client_manager._run_single_health_check("test", failing_health_check)

        # Health should be marked as failed
        assert client_manager._health["test"].state == ClientState.FAILED
        assert "Health check failed" in client_manager._health["test"].last_error

    @pytest.mark.asyncio
    async def test_client_recreation_after_failure(self, client_manager):
        """Test client recreation after health recovery."""
        # Set up a failed client
        old_client = Mock()
        old_client.close = AsyncMock()
        client_manager._clients["test"] = old_client
        client_manager._health["test"] = ClientHealth(
            state=ClientState.FAILED, last_check=time.time(), consecutive_failures=5
        )

        # Simulate health recovery
        async def healthy_check():
            return True

        await client_manager._run_single_health_check("test", healthy_check)

        # Old client should be removed to force recreation
        assert "test" not in client_manager._clients
        old_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_recreation_with_close_error(self, client_manager):
        """Test client recreation when close fails."""
        old_client = Mock()
        old_client.close = AsyncMock(side_effect=Exception("Close failed"))
        client_manager._clients["test"] = old_client
        client_manager._health["test"] = ClientHealth(
            state=ClientState.FAILED, last_check=time.time()
        )

        # Should not raise exception even if close fails
        await client_manager._recreate_client_if_needed("test")
        old_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_unhealthy_client_blocking(self, client_manager):
        """Test that unhealthy clients are blocked."""
        # Set up an unhealthy client
        client_manager._clients["test"] = Mock()
        client_manager._health["test"] = ClientHealth(
            state=ClientState.FAILED,
            last_check=time.time(),
            consecutive_failures=10,  # Exceeds max_consecutive_failures
            last_error="Test error",
        )

        with pytest.raises(APIError, match="test client is unhealthy"):
            await client_manager._get_or_create_client(
                "test", lambda: Mock(), lambda: True
            )


class TestClientManagerFactory:
    """Test the factory method for ClientManager."""

    @patch("src.config.loader.ConfigLoader.load_config")
    @patch("os.getenv")
    def test_from_unified_config(self, mock_getenv, mock_load_config):
        """Test ClientManager.from_unified_config factory method."""
        mock_unified_config = Mock()
        mock_load_config.return_value = mock_unified_config

        # Mock environment variables
        mock_getenv.side_effect = lambda key, default=None: {
            "QDRANT_URL": "http://test:6333",
            "QDRANT_API_KEY": "test-key",
            "OPENAI_API_KEY": "sk-test-key",
            "FIRECRAWL_API_KEY": "fc-test-key",
            "REDIS_URL": "redis://test:6379",
        }.get(key, default)

        manager = ClientManager.from_unified_config()

        assert manager.unified_config == mock_unified_config
        assert manager.config.qdrant_url == "http://test:6333"
        assert manager.config.qdrant_api_key == "test-key"
        assert manager.config.openai_api_key == "sk-test-key"
        assert manager.config.firecrawl_api_key == "fc-test-key"
        assert manager.config.redis_url == "redis://test:6379"


class TestCircuitBreakerAdvanced:
    """Test advanced circuit breaker scenarios."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_success(self):
        """Test circuit breaker recovery through half-open state."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        # Force circuit breaker to open
        for _ in range(2):
            with contextlib.suppress(Exception):
                await breaker.call(
                    lambda: (_ for _ in ()).throw(Exception("Test error"))
                )

        assert breaker.state == ClientState.FAILED

        # Wait for recovery timeout
        await asyncio.sleep(0.2)

        # Should be in half-open state now
        assert breaker.state == ClientState.DEGRADED

        # Successful call should reset circuit breaker
        async def success_func():
            return "success"

        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state == ClientState.HEALTHY

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_request_limit(self):
        """Test circuit breaker half-open request limit."""
        breaker = CircuitBreaker(
            failure_threshold=1, recovery_timeout=0.1, half_open_requests=1
        )

        # Force circuit breaker to open
        with contextlib.suppress(Exception):
            await breaker.call(lambda: (_ for _ in ()).throw(Exception("Test error")))

        # Wait for recovery
        await asyncio.sleep(0.2)

        # First request should be allowed (half-open)
        with contextlib.suppress(Exception):

            async def failing_func():
                raise Exception("Still failing")

            await breaker.call(failing_func)

        # Second request should be blocked due to half_open_requests limit
        with pytest.raises(APIError, match="Circuit breaker is open"):

            async def test_func():
                return "test"

            await breaker.call(test_func)


class TestHealthStatusReporting:
    """Test health status reporting functionality."""

    @pytest.fixture
    async def client_manager(self):
        """Create a ClientManager instance for testing."""
        config = ClientManagerConfig()
        manager = ClientManager(config)
        await manager.initialize()
        yield manager
        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_get_health_status_comprehensive(self, client_manager):
        """Test comprehensive health status reporting."""
        # Set up various client health states
        client_manager._health["healthy"] = ClientHealth(
            state=ClientState.HEALTHY, last_check=time.time(), consecutive_failures=0
        )

        client_manager._health["degraded"] = ClientHealth(
            state=ClientState.DEGRADED,
            last_check=time.time(),
            consecutive_failures=2,
            last_error="Occasional failures",
        )

        client_manager._health["failed"] = ClientHealth(
            state=ClientState.FAILED,
            last_check=time.time(),
            consecutive_failures=5,
            last_error="Persistent failure",
        )

        # Set up circuit breakers
        client_manager._circuit_breakers["healthy"] = CircuitBreaker()
        client_manager._circuit_breakers["failed"] = CircuitBreaker()

        status = await client_manager.get_health_status()

        assert "healthy" in status
        assert status["healthy"]["state"] == "healthy"
        assert status["healthy"]["consecutive_failures"] == 0
        assert status["healthy"]["last_error"] is None

        assert "degraded" in status
        assert status["degraded"]["state"] == "degraded"
        assert status["degraded"]["consecutive_failures"] == 2
        assert "Occasional failures" in status["degraded"]["last_error"]

        assert "failed" in status
        assert status["failed"]["state"] == "failed"
        assert status["failed"]["consecutive_failures"] == 5
        assert "Persistent failure" in status["failed"]["last_error"]
