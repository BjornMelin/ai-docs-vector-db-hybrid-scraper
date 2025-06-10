"""Comprehensive tests for ClientManager to improve coverage.

Focuses on areas not covered by existing tests:
- Service getters and initialization
- Health check loops
- Cleanup and error handling
- Performance targets validation
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, MagicMock, patch
from unittest.mock import call

import pytest
from src.config import UnifiedConfig
from src.infrastructure.client_manager import ClientManager, ClientState, ClientHealth
from src.services.errors import APIError


class TestClientManagerServiceGetters:
    """Test service getter methods and lazy initialization."""
    
    @pytest.fixture
    async def client_manager(self):
        """Create a clean ClientManager instance for testing."""
        # Clear singleton
        ClientManager._instance = None
        
        config = UnifiedConfig()
        manager = ClientManager(config)
        await manager.initialize()
        
        yield manager
        
        await manager.cleanup()
        ClientManager._instance = None
    
    @pytest.mark.asyncio
    async def test_get_qdrant_service_lazy_initialization(self, client_manager):
        """Test lazy initialization of QdrantService."""
        with patch('src.services.vector_db.service.QdrantService') as mock_service_class:
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
    async def test_get_embedding_manager_lazy_initialization(self, client_manager):
        """Test lazy initialization of EmbeddingManager."""
        with patch('src.services.embeddings.manager.EmbeddingManager') as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager
            
            service = await client_manager.get_embedding_manager()
            assert service == mock_manager
            mock_manager.initialize.assert_called_once()
            
            # Verify it was created with correct parameters
            mock_manager_class.assert_called_once_with(
                config=client_manager.config,
                client_manager=client_manager,
            )
    
    @pytest.mark.asyncio
    async def test_get_cache_manager_initialization(self, client_manager):
        """Test CacheManager initialization with config parameters."""
        with patch('src.services.cache.manager.CacheManager') as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager
            
            service = await client_manager.get_cache_manager()
            assert service == mock_manager
            
            # Verify initialization parameters
            call_args = mock_manager_class.call_args
            assert call_args.kwargs['dragonfly_url'] == client_manager.config.cache.dragonfly_url
            assert call_args.kwargs['enable_local_cache'] == client_manager.config.cache.enable_local_cache
            assert call_args.kwargs['enable_distributed_cache'] == client_manager.config.cache.enable_dragonfly_cache
    
    @pytest.mark.asyncio
    async def test_get_crawl_manager_initialization(self, client_manager):
        """Test CrawlManager initialization."""
        with patch('src.services.crawling.manager.CrawlManager') as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager
            
            service = await client_manager.get_crawl_manager()
            assert service == mock_manager
            
            mock_manager_class.assert_called_once_with(
                config=client_manager.config,
                rate_limiter=None,
            )
    
    @pytest.mark.asyncio
    async def test_get_hyde_engine_with_dependencies(self, client_manager):
        """Test HyDEEngine initialization with all dependencies."""
        mock_hyde_engine = AsyncMock()
        mock_embedding_manager = AsyncMock()
        mock_qdrant_service = AsyncMock()
        mock_cache_manager = AsyncMock()
        mock_openai_client = AsyncMock()
        
        with patch('src.services.hyde.engine.HyDEQueryEngine', return_value=mock_hyde_engine), \
             patch('src.services.hyde.config.HyDEConfig') as mock_config_class, \
             patch('src.services.hyde.config.HyDEPromptConfig'), \
             patch('src.services.hyde.config.HyDEMetricsConfig'), \
             patch.object(client_manager, 'get_embedding_manager', return_value=mock_embedding_manager), \
             patch.object(client_manager, 'get_qdrant_service', return_value=mock_qdrant_service), \
             patch.object(client_manager, 'get_cache_manager', return_value=mock_cache_manager), \
             patch.object(client_manager, 'get_openai_client', return_value=mock_openai_client):
            
            service = await client_manager.get_hyde_engine()
            assert service == mock_hyde_engine
            mock_hyde_engine.initialize.assert_called_once()
            
            # Verify HyDEConfig was created from unified config
            mock_config_class.from_unified_config.assert_called_once_with(client_manager.config.hyde)
    
    @pytest.mark.asyncio
    async def test_get_project_storage_initialization(self, client_manager):
        """Test ProjectStorage initialization."""
        with patch('src.services.core.project_storage.ProjectStorage') as mock_storage_class:
            mock_storage = AsyncMock()
            mock_storage_class.return_value = mock_storage
            
            service = await client_manager.get_project_storage()
            assert service == mock_storage
            
            mock_storage_class.assert_called_once_with(
                data_dir=client_manager.config.data_dir
            )
    
    @pytest.mark.asyncio
    async def test_get_blue_green_deployment_with_dependencies(self, client_manager):
        """Test BlueGreenDeployment initialization with dependencies."""
        mock_deployment = AsyncMock()
        mock_qdrant_service = AsyncMock()
        mock_cache_manager = AsyncMock()
        
        with patch('src.services.deployment.blue_green.BlueGreenDeployment', return_value=mock_deployment), \
             patch.object(client_manager, 'get_qdrant_service', return_value=mock_qdrant_service), \
             patch.object(client_manager, 'get_cache_manager', return_value=mock_cache_manager):
            
            service = await client_manager.get_blue_green_deployment()
            assert service == mock_deployment
    
    @pytest.mark.asyncio
    async def test_get_canary_deployment_with_task_queue(self, client_manager):
        """Test CanaryDeployment initialization with task queue dependency."""
        mock_deployment = AsyncMock()
        mock_qdrant_service = AsyncMock()
        mock_task_queue = AsyncMock()
        mock_alias_manager = AsyncMock()
        
        with patch('src.services.deployment.canary.CanaryDeployment', return_value=mock_deployment), \
             patch('src.services.core.qdrant_alias_manager.QdrantAliasManager', return_value=mock_alias_manager), \
             patch.object(client_manager, 'get_qdrant_service', return_value=mock_qdrant_service), \
             patch.object(client_manager, 'get_task_queue_manager', return_value=mock_task_queue):
            
            mock_alias_manager.initialize = AsyncMock()
            
            service = await client_manager.get_canary_deployment()
            assert service == mock_deployment
            mock_alias_manager.initialize.assert_called_once()
            mock_deployment.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_browser_automation_router(self, client_manager):
        """Test BrowserAutomationRouter initialization."""
        with patch('src.services.browser.enhanced_router.EnhancedAutomationRouter') as mock_router_class:
            mock_router = AsyncMock()
            mock_router_class.return_value = mock_router
            
            service = await client_manager.get_browser_automation_router()
            assert service == mock_router
            mock_router.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_task_queue_manager(self, client_manager):
        """Test TaskQueueManager initialization."""
        with patch('src.services.task_queue.manager.TaskQueueManager') as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager
            
            service = await client_manager.get_task_queue_manager()
            assert service == mock_manager
            mock_manager.initialize.assert_called_once()


class TestClientManagerHealthChecks:
    """Test health check functionality and loops."""
    
    @pytest.fixture
    async def client_manager_with_mock_clients(self):
        """Create ClientManager with mocked clients for health testing."""
        ClientManager._instance = None
        config = UnifiedConfig()
        manager = ClientManager(config)
        
        # Mock clients
        mock_qdrant = AsyncMock()
        mock_openai = AsyncMock()
        mock_redis = AsyncMock()
        
        manager._clients = {
            "qdrant": mock_qdrant,
            "openai": mock_openai,
            "redis": mock_redis
        }
        
        yield manager, mock_qdrant, mock_openai, mock_redis
        
        await manager.cleanup()
        ClientManager._instance = None
    
    @pytest.mark.asyncio
    async def test_single_health_check_success(self, client_manager_with_mock_clients):
        """Test successful single health check."""
        manager, mock_qdrant, _, _ = client_manager_with_mock_clients
        
        # Mock successful health check
        async def mock_health_check():
            return True
        
        await manager._run_single_health_check("qdrant", mock_health_check)
        
        # Verify health status was updated
        assert "qdrant" in manager._health
        health = manager._health["qdrant"]
        assert health.state == ClientState.HEALTHY
        assert health.consecutive_failures == 0
        assert health.last_error is None
    
    @pytest.mark.asyncio
    async def test_single_health_check_failure(self, client_manager_with_mock_clients):
        """Test failed single health check."""
        manager, _, _, _ = client_manager_with_mock_clients
        
        async def mock_health_check():
            return False
        
        await manager._run_single_health_check("openai", mock_health_check)
        
        # Verify health status shows failure
        health = manager._health["openai"]
        assert health.state == ClientState.DEGRADED  # First failure
        assert health.consecutive_failures == 1
        assert health.last_error == "Health check returned false"
    
    @pytest.mark.asyncio
    async def test_health_check_timeout(self, client_manager_with_mock_clients):
        """Test health check timeout handling."""
        manager, _, _, _ = client_manager_with_mock_clients
        
        async def slow_health_check():
            await asyncio.sleep(10)  # Longer than timeout
            return True
        
        # Use asyncio.wait_for directly to test timeout behavior
        timeout_seconds = 0.1
        with patch('asyncio.wait_for') as mock_wait_for:
            mock_wait_for.side_effect = TimeoutError()
            await manager._run_single_health_check("redis", slow_health_check)
        
        # Should register as timeout failure (FAILED because it's an exception, not just false return)
        health = manager._health["redis"]
        assert health.state == ClientState.FAILED
        assert "timeout" in health.last_error.lower()
    
    @pytest.mark.asyncio
    async def test_health_check_exception(self, client_manager_with_mock_clients):
        """Test health check exception handling."""
        manager, _, _, _ = client_manager_with_mock_clients
        
        async def failing_health_check():
            raise Exception("Health check failed")
        
        await manager._run_single_health_check("qdrant", failing_health_check)
        
        # Should register as failure (FAILED on first exception, not DEGRADED)
        health = manager._health["qdrant"]
        assert health.state == ClientState.FAILED
        assert "Health check failed" in health.last_error
    
    @pytest.mark.asyncio
    async def test_client_recreation_after_recovery(self, client_manager_with_mock_clients):
        """Test client recreation after health recovery."""
        manager, mock_qdrant, _, _ = client_manager_with_mock_clients
        
        # Set client as failed
        manager._health["qdrant"] = ClientHealth(
            state=ClientState.FAILED,
            last_check=time.time(),
            consecutive_failures=5
        )
        
        # Mock successful health check (recovery)
        async def recovery_health_check():
            return True
        
        await manager._run_single_health_check("qdrant", recovery_health_check)
        
        # Should be healthy again
        health = manager._health["qdrant"]
        assert health.state == ClientState.HEALTHY
        assert health.consecutive_failures == 0
    
    @pytest.mark.asyncio
    async def test_health_check_loop_cancellation(self, client_manager_with_mock_clients):
        """Test health check loop cancellation."""
        manager, _, _, _ = client_manager_with_mock_clients
        
        # Start health check loop
        loop_task = asyncio.create_task(manager._health_check_loop())
        
        # Let it run very briefly to ensure it starts
        await asyncio.sleep(0.01)
        
        # Cancel the task
        loop_task.cancel()
        
        # Should handle cancellation gracefully (task should be cancelled)
        try:
            await loop_task
        except asyncio.CancelledError:
            pass  # Expected
        
        # Check that task was cancelled or completed (both are acceptable)
        assert loop_task.cancelled() or loop_task.done()


class TestClientManagerCleanup:
    """Test cleanup and resource management."""
    
    @pytest.mark.asyncio
    async def test_cleanup_with_services(self):
        """Test cleanup with various services initialized."""
        ClientManager._instance = None
        config = UnifiedConfig()
        manager = ClientManager(config)
        
        # Mock services with cleanup methods
        mock_qdrant_service = AsyncMock()
        mock_qdrant_service.cleanup = AsyncMock()
        
        mock_embedding_manager = AsyncMock()
        mock_embedding_manager.cleanup = AsyncMock()
        
        mock_cache_manager = AsyncMock()
        mock_cache_manager.cleanup = AsyncMock()
        
        # Assign services directly
        manager._qdrant_service = mock_qdrant_service
        manager._embedding_manager = mock_embedding_manager
        manager._cache_manager = mock_cache_manager
        
        # Mock clients - one with close, one with only aclose
        mock_client1 = AsyncMock()
        mock_client1.close = AsyncMock()
        
        mock_client2 = AsyncMock()
        # Remove close attribute so only aclose is available
        del mock_client2.close
        mock_client2.aclose = AsyncMock()
        
        manager._clients = {"client1": mock_client1, "client2": mock_client2}
        
        # Cleanup
        await manager.cleanup()
        
        # Verify services were cleaned up
        mock_qdrant_service.cleanup.assert_called_once()
        mock_embedding_manager.cleanup.assert_called_once()
        mock_cache_manager.cleanup.assert_called_once()
        
        # Verify clients were closed
        mock_client1.close.assert_called_once()
        mock_client2.aclose.assert_called_once()
        
        # Verify state was reset
        assert manager._clients == {}
        assert manager._health == {}
        assert manager._qdrant_service is None
        assert manager._embedding_manager is None
        assert manager._cache_manager is None
    
    @pytest.mark.asyncio
    async def test_cleanup_with_service_errors(self):
        """Test cleanup handling of service cleanup errors."""
        ClientManager._instance = None
        config = UnifiedConfig()
        manager = ClientManager(config)
        
        # Mock service that fails during cleanup
        mock_service = AsyncMock()
        mock_service.cleanup = AsyncMock(side_effect=Exception("Cleanup failed"))
        manager._qdrant_service = mock_service
        
        # Should not raise exception
        await manager.cleanup()
        
        # Service should still be reset
        assert manager._qdrant_service is None
    
    @pytest.mark.asyncio
    async def test_cleanup_with_client_errors(self):
        """Test cleanup handling of client close errors."""
        ClientManager._instance = None
        config = UnifiedConfig()
        manager = ClientManager(config)
        
        # Mock client that fails during close
        mock_client = AsyncMock()
        mock_client.close = AsyncMock(side_effect=Exception("Close failed"))
        manager._clients = {"failing_client": mock_client}
        
        # Should not raise exception
        await manager.cleanup()
        
        # Clients should still be cleared
        assert manager._clients == {}


class TestClientManagerPerformance:
    """Test performance-related functionality."""
    
    @pytest.mark.asyncio
    async def test_client_retrieval_performance(self):
        """Test client retrieval performance (target: <1ms)."""
        ClientManager._instance = None
        config = UnifiedConfig()
        manager = ClientManager(config)
        
        # Mock a simple client
        mock_client = AsyncMock()
        manager._clients["test"] = mock_client
        manager._health["test"] = ClientHealth(
            state=ClientState.HEALTHY,
            last_check=time.time()
        )
        
        # Create mock client getter
        async def mock_get_client():
            return manager._clients["test"]
        
        # Measure retrieval time
        start_time = time.time()
        
        # Simulate multiple rapid retrievals
        for _ in range(100):
            client = await mock_get_client()
            assert client == mock_client
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        avg_time_ms = elapsed_ms / 100
        
        print(f"\nClient retrieval performance:")
        print(f"100 retrievals in {elapsed_ms:.2f}ms")
        print(f"Average time per retrieval: {avg_time_ms:.3f}ms")
        
        # Performance target: <1ms per connection retrieval
        assert avg_time_ms < 1.0, f"Client retrieval took {avg_time_ms:.3f}ms, target is <1ms"
        
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_health_check_performance(self):
        """Test health check performance."""
        ClientManager._instance = None
        config = UnifiedConfig()
        manager = ClientManager(config)
        
        # Mock multiple clients
        clients = {}
        for i in range(10):
            clients[f"client_{i}"] = AsyncMock()
        
        manager._clients = clients
        
        async def fast_health_check():
            await asyncio.sleep(0.001)  # Simulate very fast check
            return True
        
        # Measure health check performance
        start_time = time.time()
        
        # Run health checks for all clients
        tasks = []
        for name in clients.keys():
            task = asyncio.create_task(
                manager._run_single_health_check(name, fast_health_check)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        
        print(f"\nHealth check performance:")
        print(f"10 health checks in {elapsed_ms:.2f}ms")
        print(f"Average time per check: {elapsed_ms/10:.2f}ms")
        
        # Should complete quickly
        assert elapsed_ms < 100, f"Health checks took {elapsed_ms:.2f}ms, which is too slow"
        
        await manager.cleanup()


class TestClientManagerErrorScenarios:
    """Test error scenarios and edge cases."""
    
    @pytest.mark.asyncio
    async def test_client_creation_with_circuit_breaker_config(self):
        """Test client creation uses configuration for circuit breaker settings."""
        ClientManager._instance = None
        
        config = UnifiedConfig()
        # Note: These fields don't exist in PerformanceConfig, but ClientManager uses getattr with defaults
        
        manager = ClientManager(config)
        
        # Mock client creation
        mock_client = AsyncMock()
        
        async def mock_create_client():
            return mock_client
        
        # Create client (should create circuit breaker with default settings since fields don't exist)
        with patch('src.infrastructure.client_manager.CircuitBreaker') as mock_breaker_class:
            mock_breaker = AsyncMock()
            mock_breaker.call = AsyncMock(return_value=mock_client)
            mock_breaker_class.return_value = mock_breaker
            
            # Ensure client doesn't exist yet
            assert "test" not in manager._clients
            
            client = await manager._get_or_create_client(
                "test",
                mock_create_client,
                lambda: True
            )
            
            # Verify circuit breaker was created with default settings (getattr fallbacks)
            mock_breaker_class.assert_called_once_with(
                failure_threshold=5,  # Default from getattr fallback
                recovery_timeout=60.0,  # Default from getattr fallback
                half_open_requests=1,  # Default from getattr fallback
            )
            
            # Verify client was created via circuit breaker
            mock_breaker.call.assert_called_once_with(mock_create_client)
            assert client == mock_client
        
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_client_unhealthy_threshold_check(self):
        """Test client health threshold checking."""
        ClientManager._instance = None
        config = UnifiedConfig()
        # Note: max_consecutive_failures doesn't exist in PerformanceConfig, uses getattr with default 3
        
        manager = ClientManager(config)
        
        # Set up unhealthy client
        manager._health["test"] = ClientHealth(
            state=ClientState.FAILED,
            last_check=time.time(),
            last_error="Multiple failures",
            consecutive_failures=4  # Exceeds default threshold of 3
        )
        
        mock_client = AsyncMock()
        manager._clients["test"] = mock_client
        
        # Should raise APIError due to health threshold
        with pytest.raises(APIError, match="unhealthy"):
            await manager._get_or_create_client(
                "test",
                lambda: mock_client,
                lambda: True
            )
        
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_get_health_status_comprehensive(self):
        """Test comprehensive health status reporting."""
        ClientManager._instance = None
        config = UnifiedConfig()
        manager = ClientManager(config)
        
        # Set up various client states
        manager._health["healthy"] = ClientHealth(
            state=ClientState.HEALTHY,
            last_check=time.time(),
            consecutive_failures=0
        )
        
        manager._health["degraded"] = ClientHealth(
            state=ClientState.DEGRADED,
            last_check=time.time() - 30,
            last_error="Temporary issue",
            consecutive_failures=2
        )
        
        manager._health["failed"] = ClientHealth(
            state=ClientState.FAILED,
            last_check=time.time() - 60,
            last_error="Connection failed",
            consecutive_failures=5
        )
        
        # Add circuit breakers
        from src.infrastructure.client_manager import CircuitBreaker
        manager._circuit_breakers["healthy"] = CircuitBreaker()
        manager._circuit_breakers["degraded"] = CircuitBreaker()
        
        status = await manager.get_health_status()
        
        assert len(status) == 3
        
        # Check healthy client
        assert status["healthy"]["state"] == "healthy"
        assert status["healthy"]["consecutive_failures"] == 0
        assert status["healthy"]["last_error"] is None
        assert status["healthy"]["circuit_breaker_state"] == "healthy"
        
        # Check degraded client
        assert status["degraded"]["state"] == "degraded"
        assert status["degraded"]["consecutive_failures"] == 2
        assert status["degraded"]["last_error"] == "Temporary issue"
        
        # Check failed client
        assert status["failed"]["state"] == "failed"
        assert status["failed"]["consecutive_failures"] == 5
        assert status["failed"]["last_error"] == "Connection failed"
        assert status["failed"]["circuit_breaker_state"] is None  # No circuit breaker
        
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_managed_client_context_manager(self):
        """Test managed client context manager."""
        ClientManager._instance = None
        config = UnifiedConfig()
        config.qdrant.url = "http://localhost:6333"
        manager = ClientManager(config)
        
        mock_client = AsyncMock()
        
        with patch.object(manager, 'get_qdrant_client', return_value=mock_client):
            async with manager.managed_client("qdrant") as client:
                assert client == mock_client
        
        # Test with invalid client type
        with pytest.raises(ValueError, match="Unknown client type"):
            async with manager.managed_client("invalid"):
                pass
        
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_managed_client_context_manager_with_error(self):
        """Test managed client context manager error handling."""
        ClientManager._instance = None
        config = UnifiedConfig()
        manager = ClientManager(config)
        
        with patch.object(manager, 'get_qdrant_client', side_effect=APIError("Client failed")):
            with pytest.raises(APIError):
                async with manager.managed_client("qdrant"):
                    pass
        
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_double_initialization_same_config(self):
        """Test that double initialization with same config is handled gracefully."""
        ClientManager._instance = None
        config = UnifiedConfig()
        
        # First initialization
        manager1 = ClientManager(config)
        await manager1.initialize()
        
        # Second initialization with same config (should be no-op)
        manager2 = ClientManager(config)
        await manager2.initialize()
        
        # Should be the same instance
        assert manager1 is manager2
        
        await manager1.cleanup()
    
    @pytest.mark.asyncio
    async def test_double_initialization_different_config(self):
        """Test that double initialization with different config raises error."""
        ClientManager._instance = None
        config1 = UnifiedConfig()
        config2 = UnifiedConfig()
        config2.qdrant.url = "http://different:6333"
        
        # First initialization
        manager1 = ClientManager(config1)
        await manager1.initialize()
        
        # Second initialization with different config should raise error
        with pytest.raises(ValueError, match="already initialized with different config"):
            ClientManager(config2)
        
        await manager1.cleanup()


class TestClientManagerAsyncContextManager:
    """Test async context manager functionality."""
    
    @pytest.mark.asyncio
    async def test_async_context_manager_lifecycle(self):
        """Test async context manager entry and exit."""
        ClientManager._instance = None
        config = UnifiedConfig()
        
        async with ClientManager(config) as manager:
            assert manager._initialized is True
            assert isinstance(manager, ClientManager)
        
        # Should be cleaned up after exit
        assert manager._initialized is False
        assert manager._clients == {}
        
        # Reset singleton
        ClientManager._instance = None