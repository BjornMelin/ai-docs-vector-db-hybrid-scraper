"""Comprehensive test suite for canary deployment with ARQ and DragonflyDB integration."""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from src.config import UnifiedConfig
from src.services.core.qdrant_alias_manager import QdrantAliasManager
from src.services.deployment.canary import CanaryDeployment
from src.services.deployment.canary import CanaryDeploymentConfig
from src.services.deployment.canary import CanaryMetrics
from src.services.deployment.canary import CanaryStage
from src.services.deployment.canary_router import CanaryRouter
from src.services.deployment.canary_router import RouteDecision
from src.services.deployment.deployment_state import DeploymentState
from src.services.deployment.deployment_state import DeploymentStateManager
from src.services.errors import ServiceError
from src.services.vector_db.search_interceptor import SearchInterceptor


@pytest.fixture
def mock_config():
    """Create mock unified config."""
    config = MagicMock(spec=UnifiedConfig)
    
    # Create nested mock objects
    performance_mock = MagicMock()
    performance_mock.canary_deployment_enabled = True
    performance_mock.canary_health_check_interval = 10
    performance_mock.canary_metrics_window = 300
    performance_mock.canary_max_error_rate = 0.1
    performance_mock.canary_min_success_count = 10
    
    cache_mock = MagicMock()
    cache_mock.dragonfly_url = "redis://localhost:6379"
    
    # Assign nested mocks
    config.performance = performance_mock
    config.cache = cache_mock
    
    # Add data_dir mock
    config.data_dir = Path("/tmp/test_data")
    
    return config


@pytest.fixture
def mock_alias_manager():
    """Create mock alias manager."""
    alias_manager = AsyncMock(spec=QdrantAliasManager)
    alias_manager.switch_alias = AsyncMock(return_value="old_collection")
    alias_manager.get_collection_for_alias = AsyncMock(return_value="old_collection")
    alias_manager.alias_exists = AsyncMock(return_value=True)
    alias_manager.safe_delete_collection = AsyncMock()
    return alias_manager


@pytest.fixture
def mock_task_queue_manager():
    """Create mock task queue manager."""
    manager = AsyncMock()
    manager.enqueue = AsyncMock(return_value="job_123")
    manager.get_job_status = AsyncMock(
        return_value={"status": "completed", "result": {"status": "success"}}
    )
    return manager


@pytest.fixture
def mock_qdrant_service():
    """Create mock Qdrant service."""
    service = AsyncMock()
    service.get_collection_info = AsyncMock(
        return_value={"status": "green", "vectors_count": 1000}
    )
    return service


@pytest.fixture
def mock_client_manager():
    """Create mock client manager."""
    manager = AsyncMock()
    redis_client = AsyncMock()
    redis_client.xadd = AsyncMock()
    redis_client.xrange = AsyncMock(return_value=[])
    redis_client.pipeline = MagicMock()
    redis_client.pipeline().__aenter__ = AsyncMock()
    redis_client.pipeline().__aexit__ = AsyncMock()
    manager.get_redis_client = AsyncMock(return_value=redis_client)
    manager.get_cache_manager = AsyncMock()
    return manager


@pytest.fixture
def mock_redis_client():
    """Create mock Redis/DragonflyDB client."""
    client = AsyncMock()
    client.set = AsyncMock(return_value=True)
    client.get = AsyncMock(return_value=None)
    client.delete = AsyncMock(return_value=1)
    client.scan = AsyncMock(return_value=(0, []))
    client.xadd = AsyncMock()
    client.xrange = AsyncMock(return_value=[])
    client.expire = AsyncMock(return_value=True)
    client.eval = AsyncMock(return_value=1)
    
    # Pipeline mock
    pipeline = AsyncMock()
    pipeline.watch = AsyncMock()
    pipeline.unwatch = AsyncMock()
    pipeline.multi = MagicMock()
    pipeline.setex = MagicMock()
    pipeline.execute = AsyncMock(return_value=[True])
    pipeline.get = MagicMock()
    
    client.pipeline = MagicMock(return_value=pipeline)
    
    return client


@pytest.fixture
def mock_cache():
    """Create mock distributed cache."""
    cache = AsyncMock()
    cache.set = AsyncMock(return_value=True)
    cache.get = AsyncMock(return_value=None)
    cache.hget = AsyncMock(return_value=None)
    cache.hset = AsyncMock(return_value=True)
    cache.hdel = AsyncMock(return_value=1)
    cache.hgetall = AsyncMock(return_value={})
    cache.expire = AsyncMock(return_value=True)
    cache.delete = AsyncMock(return_value=1)
    cache.incr = AsyncMock(return_value=1)
    cache.hincrby = AsyncMock(return_value=1)
    cache.hincrbyfloat = AsyncMock(return_value=1.0)
    return cache


class TestCanaryDeployment:
    """Test suite for CanaryDeployment service."""

    @pytest.mark.asyncio
    async def test_initialization(
        self,
        mock_config,
        mock_alias_manager,
        mock_task_queue_manager,
        mock_qdrant_service,
        mock_client_manager,
    ):
        """Test canary deployment initialization."""
        deployment = CanaryDeployment(
            config=mock_config,
            alias_manager=mock_alias_manager,
            task_queue_manager=mock_task_queue_manager,
            qdrant_service=mock_qdrant_service,
            client_manager=mock_client_manager,
        )
        
        # Mock cache manager
        cache_manager = AsyncMock()
        cache_manager.distributed_cache = AsyncMock()
        mock_client_manager.get_cache_manager.return_value = cache_manager
        
        await deployment.initialize()
        
        assert deployment._initialized
        # Router should be initialized with the cache
        assert deployment._router is not None

    @pytest.mark.asyncio
    async def test_start_canary_deployment_success(
        self,
        mock_config,
        mock_alias_manager,
        mock_task_queue_manager,
        mock_qdrant_service,
        mock_client_manager,
    ):
        """Test successful canary deployment start."""
        deployment = CanaryDeployment(
            config=mock_config,
            alias_manager=mock_alias_manager,
            task_queue_manager=mock_task_queue_manager,
            qdrant_service=mock_qdrant_service,
            client_manager=mock_client_manager,
        )
        await deployment.initialize()

        # Setup collection check
        mock_qdrant_service.get_collection_info.return_value = {
            "status": "green",
            "vectors_count": 1000,
        }

        # Start deployment
        stages = [
            {"percentage": 10, "duration_minutes": 10},
            {"percentage": 50, "duration_minutes": 20},
            {"percentage": 100, "duration_minutes": 0},
        ]
        
        deployment_id = await deployment.start_canary(
            alias_name="production",
            new_collection="new_collection",
            stages=stages,
        )
        
        assert deployment_id is not None
        assert deployment_id in deployment.deployments
        
        # Verify task queue was called
        mock_task_queue_manager.enqueue.assert_called_once()
        call_args = mock_task_queue_manager.enqueue.call_args
        assert call_args[0][0] == "run_canary_deployment"

    @pytest.mark.asyncio
    async def test_start_canary_deployment_collection_not_ready(
        self,
        mock_config,
        mock_alias_manager,
        mock_task_queue_manager,
        mock_qdrant_service,
        mock_client_manager,
    ):
        """Test canary deployment fails when collection not ready."""
        deployment = CanaryDeployment(
            config=mock_config,
            alias_manager=mock_alias_manager,
            task_queue_manager=mock_task_queue_manager,
            qdrant_service=mock_qdrant_service,
            client_manager=mock_client_manager,
        )
        await deployment.initialize()

        # Setup collection check to fail
        mock_qdrant_service.get_collection_info.return_value = {
            "status": "red",
            "vectors_count": 0,
        }

        with pytest.raises(ServiceError, match="Collection new_collection is not ready"):
            await deployment.start_canary(
                alias_name="production",
                new_collection="new_collection",
                stages=[{"percentage": 100, "duration_minutes": 0}],
            )

    @pytest.mark.asyncio
    async def test_pause_resume_deployment(
        self,
        mock_config,
        mock_alias_manager,
        mock_task_queue_manager,
        mock_qdrant_service,
        mock_client_manager,
    ):
        """Test pausing and resuming deployment."""
        deployment = CanaryDeployment(
            config=mock_config,
            alias_manager=mock_alias_manager,
            task_queue_manager=mock_task_queue_manager,
            qdrant_service=mock_qdrant_service,
            client_manager=mock_client_manager,
        )
        await deployment.initialize()

        # Create active deployment
        deployment_id = str(uuid.uuid4())
        config = CanaryDeploymentConfig(
            alias="production",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[CanaryStage(percentage=50, duration_minutes=30)],
            current_stage=0,
            metrics=CanaryMetrics(),
            start_time=time.time(),
            status="running",
        )
        deployment.deployments[deployment_id] = config

        # No need to mock state manager - the current implementation
        # uses internal deployments dict

        # Pause deployment
        result = await deployment.pause_deployment(deployment_id)
        assert result is True
        assert deployment.deployments[deployment_id].status == "paused"

        # Resume deployment
        result = await deployment.resume_deployment(deployment_id)
        assert result is True
        assert deployment.deployments[deployment_id].status == "running"

    @pytest.mark.asyncio
    async def test_rollback_deployment(
        self,
        mock_config,
        mock_alias_manager,
        mock_task_queue_manager,
        mock_qdrant_service,
        mock_client_manager,
    ):
        """Test deployment rollback."""
        deployment = CanaryDeployment(
            config=mock_config,
            alias_manager=mock_alias_manager,
            task_queue_manager=mock_task_queue_manager,
            qdrant_service=mock_qdrant_service,
            client_manager=mock_client_manager,
        )
        await deployment.initialize()

        # Create active deployment
        deployment_id = str(uuid.uuid4())
        config = CanaryDeploymentConfig(
            alias="production",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[CanaryStage(percentage=50, duration_minutes=30)],
            current_stage=0,
            metrics=CanaryMetrics(),
            start_time=time.time(),
            status="running",
        )
        deployment.deployments[deployment_id] = config

        # Mock router
        deployment._router.update_route = AsyncMock()
        deployment._router.remove_route = AsyncMock()

        # Rollback deployment
        result = await deployment.rollback_deployment(deployment_id)
        assert result is True
        assert deployment.deployments[deployment_id].status == "rolled_back"
        
        # Verify rollback operations
        mock_alias_manager.switch_alias.assert_called_with(
            "production", "old_collection", delete_old=False
        )
        deployment._router.remove_route.assert_called_with("production")

    @pytest.mark.asyncio
    async def test_get_deployment_status(
        self,
        mock_config,
        mock_alias_manager,
        mock_task_queue_manager,
        mock_qdrant_service,
        mock_client_manager,
    ):
        """Test getting deployment status."""
        deployment = CanaryDeployment(
            config=mock_config,
            alias_manager=mock_alias_manager,
            task_queue_manager=mock_task_queue_manager,
            qdrant_service=mock_qdrant_service,
            client_manager=mock_client_manager,
        )
        await deployment.initialize()

        # Create deployment
        deployment_id = str(uuid.uuid4())
        config = CanaryDeploymentConfig(
            alias="production",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[CanaryStage(percentage=50, duration_minutes=30)],
            current_stage=0,
            metrics=CanaryMetrics(
                latency={"p50": 10.0, "p95": 20.0, "p99": 30.0},
                error_rate=0.02,
                success_count=100,
                error_count=2,
                stage_start_time=time.time(),
            ),
            start_time=time.time(),
            status="running",
        )
        deployment.deployments[deployment_id] = config

        # Get status
        status = await deployment.get_deployment_status(deployment_id)
        
        assert status["deployment_id"] == deployment_id
        assert status["status"] == "running"
        assert status["current_stage"] == 0
        assert status["current_percentage"] == 50
        assert status["metrics"]["error_rate"] == 0.02
        assert status["metrics"]["success_count"] == 100

    @pytest.mark.asyncio
    async def test_cleanup_completed_deployments(
        self,
        mock_config,
        mock_alias_manager,
        mock_task_queue_manager,
        mock_qdrant_service,
        mock_client_manager,
    ):
        """Test cleanup of completed deployments."""
        deployment = CanaryDeployment(
            config=mock_config,
            alias_manager=mock_alias_manager,
            task_queue_manager=mock_task_queue_manager,
            qdrant_service=mock_qdrant_service,
            client_manager=mock_client_manager,
        )
        await deployment.initialize()

        # Create completed deployment
        deployment_id = str(uuid.uuid4())
        config = CanaryDeploymentConfig(
            alias="production",
            old_collection="old_collection",
            new_collection="new_collection",
            stages=[CanaryStage(percentage=100, duration_minutes=0)],
            current_stage=0,
            metrics=CanaryMetrics(),
            start_time=time.time() - 86400,  # 24 hours ago
            status="completed",
        )
        deployment.deployments[deployment_id] = config

        # Mock state manager
        deployment._state_manager.delete_state = AsyncMock(return_value=True)

        # Run cleanup
        await deployment._cleanup_completed_deployments()
        
        # Verify deployment was removed
        assert deployment_id not in deployment.deployments
        deployment._state_manager.delete_state.assert_called_with(deployment_id)


class TestCanaryRouter:
    """Test suite for CanaryRouter."""

    @pytest.mark.asyncio
    async def test_update_route(self, mock_cache, mock_config):
        """Test updating canary route."""
        router = CanaryRouter(cache=mock_cache, config=mock_config)
        
        await router.update_route(
            alias="production",
            deployment_id="deploy_123",
            old_collection="old_col",
            new_collection="new_col",
            percentage=30.0,
        )
        
        # Verify cache operations
        mock_cache.hset.assert_any_call(
            "canary:routes:production",
            "deployment_id",
            "deploy_123",
        )
        mock_cache.hset.assert_any_call(
            "canary:routes:production",
            "percentage",
            "30.0",
        )

    @pytest.mark.asyncio
    async def test_get_route_decision_no_canary(self, mock_cache, mock_config):
        """Test route decision when no canary is active."""
        router = CanaryRouter(cache=mock_cache, config=mock_config)
        
        # No route in cache
        mock_cache.hgetall.return_value = {}
        
        decision = await router.get_route_decision(
            alias="production",
            user_id="user_123",
        )
        
        assert decision.collection_name == "production"
        assert decision.is_canary is False
        assert decision.deployment_id is None

    @pytest.mark.asyncio
    async def test_get_route_decision_with_canary(self, mock_cache, mock_config):
        """Test route decision with active canary."""
        router = CanaryRouter(cache=mock_cache, config=mock_config)
        
        # Route in cache
        mock_cache.hgetall.return_value = {
            b"deployment_id": b"deploy_123",
            b"old_collection": b"old_col",
            b"new_collection": b"new_col",
            b"percentage": b"50.0",
        }
        
        # Test multiple decisions to verify distribution
        canary_count = 0
        for i in range(100):
            decision = await router.get_route_decision(
                alias="production",
                user_id=f"user_{i}",
            )
            if decision.is_canary:
                canary_count += 1
        
        # Should be roughly 50% canary (with some tolerance)
        assert 40 <= canary_count <= 60

    @pytest.mark.asyncio
    async def test_sticky_sessions(self, mock_cache, mock_config):
        """Test sticky session functionality."""
        router = CanaryRouter(cache=mock_cache, config=mock_config)
        
        # Setup canary route
        mock_cache.hgetall.return_value = {
            b"deployment_id": b"deploy_123",
            b"old_collection": b"old_col",
            b"new_collection": b"new_col",
            b"percentage": b"50.0",
        }
        
        # First decision
        mock_cache.get.return_value = None  # No sticky session yet
        decision1 = await router.get_route_decision(
            alias="production",
            user_id="user_123",
            use_sticky_sessions=True,
        )
        
        # Simulate sticky session stored
        if decision1.is_canary:
            mock_cache.get.return_value = b"new_col"
        else:
            mock_cache.get.return_value = b"old_col"
        
        # Second decision should be same
        decision2 = await router.get_route_decision(
            alias="production",
            user_id="user_123",
            use_sticky_sessions=True,
        )
        
        assert decision1.collection_name == decision2.collection_name
        assert decision1.is_canary == decision2.is_canary

    @pytest.mark.asyncio
    async def test_record_request_metrics(self, mock_cache, mock_config):
        """Test recording request metrics."""
        router = CanaryRouter(cache=mock_cache, config=mock_config)
        
        await router.record_request_metrics(
            deployment_id="deploy_123",
            collection_name="new_col",
            latency_ms=25.5,
            is_error=False,
        )
        
        # Verify metrics recorded
        mock_cache.hincrby.assert_called()
        mock_cache.hincrbyfloat.assert_called()

    @pytest.mark.asyncio
    async def test_get_collection_metrics(self, mock_cache, mock_config):
        """Test getting collection metrics."""
        router = CanaryRouter(cache=mock_cache, config=mock_config)
        
        # Setup mock metrics
        mock_cache.hgetall.return_value = {
            b"count": b"1000",
            b"errors": b"10",
            b"total_latency": b"25000.0",
        }
        
        metrics = await router.get_collection_metrics("deploy_123", "new_col")
        
        assert metrics["count"] == 1000
        assert metrics["errors"] == 10
        assert metrics["avg_latency"] == 25.0
        assert metrics["error_rate"] == 0.01


class TestDeploymentStateManager:
    """Test suite for DeploymentStateManager."""

    @pytest.mark.asyncio
    async def test_create_state(self, mock_redis_client):
        """Test creating deployment state."""
        manager = DeploymentStateManager(mock_redis_client)
        
        state = DeploymentState(
            deployment_id="deploy_123",
            alias="production",
            old_collection="old_col",
            new_collection="new_col",
            current_stage=0,
            current_percentage=50.0,
            status="running",
            start_time=time.time(),
            last_updated=time.time(),
            metrics={},
        )
        
        result = await manager.create_state(state)
        assert result is True
        
        # Verify Redis operations
        mock_redis_client.set.assert_called()
        mock_redis_client.xadd.assert_called()

    @pytest.mark.asyncio
    async def test_update_state_with_locking(self, mock_redis_client):
        """Test updating state with distributed locking."""
        manager = DeploymentStateManager(mock_redis_client)
        
        # Simulate existing state
        existing_state = {
            "deployment_id": "deploy_123",
            "status": "running",
            "current_percentage": 50.0,
            "version": 1,
        }
        mock_redis_client.get.return_value = json.dumps(existing_state).encode()
        
        # Update state
        result = await manager.update_state(
            deployment_id="deploy_123",
            updates={"current_percentage": 75.0, "current_stage": 1},
        )
        
        assert result is True
        
        # Verify lock was acquired and released
        assert mock_redis_client.set.call_count >= 1  # Lock acquisition
        assert mock_redis_client.delete.call_count >= 1  # Lock release

    @pytest.mark.asyncio
    async def test_list_deployments(self, mock_redis_client):
        """Test listing deployments."""
        manager = DeploymentStateManager(mock_redis_client)
        
        # Setup scan to return keys
        mock_redis_client.scan.return_value = (0, [b"deployment:state:deploy_1"])
        
        # Setup pipeline to return state data
        state_data = {
            "deployment_id": "deploy_1",
            "alias": "production",
            "old_collection": "old_col",
            "new_collection": "new_col",
            "current_stage": 0,
            "current_percentage": 50.0,
            "status": "running",
            "start_time": time.time(),
            "last_updated": time.time(),
            "metrics": {},
        }
        
        pipeline = mock_redis_client.pipeline.return_value
        pipeline.execute.return_value = [json.dumps(state_data).encode()]
        
        deployments = await manager.list_deployments()
        
        assert len(deployments) == 1
        assert deployments[0].deployment_id == "deploy_1"
        assert deployments[0].status == "running"

    @pytest.mark.asyncio
    async def test_acquire_release_lock(self, mock_redis_client):
        """Test acquiring and releasing deployment lock."""
        manager = DeploymentStateManager(mock_redis_client)
        
        # Test acquire lock
        result = await manager.acquire_deployment_lock(
            deployment_id="deploy_123",
            holder_id="worker_1",
            ttl=60,
        )
        assert result is True
        
        # Test release lock (with Lua script)
        mock_redis_client.eval.return_value = 1
        result = await manager.release_deployment_lock(
            deployment_id="deploy_123",
            holder_id="worker_1",
        )
        assert result is True
        
        # Verify Lua script was called
        mock_redis_client.eval.assert_called()


class TestSearchInterceptor:
    """Test suite for SearchInterceptor."""

    @pytest.mark.asyncio
    async def test_hybrid_search_with_canary_routing(self, mock_config):
        """Test hybrid search with canary routing."""
        # Mock search service
        search_service = AsyncMock()
        search_service.hybrid_search = AsyncMock(
            return_value=[{"id": "1", "score": 0.9}]
        )
        
        # Mock router
        router = AsyncMock(spec=CanaryRouter)
        router.get_route_decision = AsyncMock(
            return_value=RouteDecision(
                collection_name="new_col",
                is_canary=True,
                canary_percentage=50.0,
                deployment_id="deploy_123",
            )
        )
        router.record_request_metrics = AsyncMock()
        
        # Mock Redis client
        redis_client = AsyncMock()
        redis_client.xadd = AsyncMock()
        
        interceptor = SearchInterceptor(
            search_service=search_service,
            router=router,
            config=mock_config,
            redis_client=redis_client,
        )
        
        results = await interceptor.hybrid_search(
            collection_name="production",
            query_vector=[0.1, 0.2, 0.3],
            limit=10,
            user_id="user_123",
        )
        
        assert len(results) == 1
        assert results[0]["id"] == "1"
        
        # Verify routing was applied
        router.get_route_decision.assert_called_once()
        search_service.hybrid_search.assert_called_with(
            collection_name="new_col",  # Routed to canary
            query_vector=[0.1, 0.2, 0.3],
            sparse_vector=None,
            limit=10,
            score_threshold=0.0,
            fusion_type="rrf",
            search_accuracy="balanced",
        )
        
        # Verify metrics recorded
        router.record_request_metrics.assert_called_once()
        
        # Verify events published
        assert redis_client.xadd.call_count >= 2  # route and completion events

    @pytest.mark.asyncio
    async def test_search_error_handling(self, mock_config):
        """Test search error handling and metrics."""
        # Mock search service to fail
        search_service = AsyncMock()
        search_service.hybrid_search = AsyncMock(side_effect=Exception("Search failed"))
        
        # Mock router
        router = AsyncMock(spec=CanaryRouter)
        router.get_route_decision = AsyncMock(
            return_value=RouteDecision(
                collection_name="new_col",
                is_canary=True,
                canary_percentage=50.0,
                deployment_id="deploy_123",
            )
        )
        router.record_request_metrics = AsyncMock()
        
        # Mock Redis client
        redis_client = AsyncMock()
        redis_client.xadd = AsyncMock()
        
        interceptor = SearchInterceptor(
            search_service=search_service,
            router=router,
            config=mock_config,
            redis_client=redis_client,
        )
        
        with pytest.raises(Exception, match="Search failed"):
            await interceptor.hybrid_search(
                collection_name="production",
                query_vector=[0.1, 0.2, 0.3],
                limit=10,
                user_id="user_123",
            )
        
        # Verify error metrics recorded
        router.record_request_metrics.assert_called_with(
            deployment_id="deploy_123",
            collection_name="new_col",
            latency_ms=pytest.Any(float),
            is_error=True,
        )
        
        # Verify error event published
        error_event_call = None
        for call in redis_client.xadd.call_args_list:
            if call[0][1].get("type") == "search_failed":
                error_event_call = call
                break
        assert error_event_call is not None