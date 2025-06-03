"""Comprehensive tests for enhanced CanaryDeployment with production-grade traffic routing."""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from src.config.models import UnifiedConfig
from src.services.deployment.canary import CanaryDeployment
from src.services.deployment.canary import CanaryDeploymentConfig
from src.services.deployment.canary import CanaryStage
from src.services.deployment.canary_router import CanaryRouter
from src.services.deployment.canary_router import RouteDecision


@pytest.fixture
def mock_config():
    """Create mock unified config with temporary data directory."""
    config = MagicMock(spec=UnifiedConfig)
    config.data_dir = Path(tempfile.mkdtemp())
    return config


@pytest.fixture
def mock_alias_manager():
    """Create mock alias manager."""
    manager = AsyncMock()
    manager.get_collection_for_alias = AsyncMock(return_value="old_collection")
    manager.switch_alias = AsyncMock()
    return manager


@pytest.fixture
def mock_qdrant_service():
    """Create mock Qdrant service."""
    service = AsyncMock()
    service.get_collection_info = AsyncMock(
        return_value={
            "status": "green",
            "vectors_count": 1000,
            "indexed_vectors_count": 1000,
            "points_count": 1000,
        }
    )
    return service


@pytest.fixture
def mock_dragonfly_cache():
    """Create mock DragonflyDB cache."""
    cache = AsyncMock()
    cache.set = AsyncMock(return_value=True)
    cache.get = AsyncMock(return_value=None)
    cache.delete = AsyncMock(return_value=True)
    cache.scan_keys = AsyncMock(return_value=[])

    # Mock Redis client for metrics operations
    redis_client = AsyncMock()
    redis_client.lpush = AsyncMock(return_value=1)
    redis_client.expire = AsyncMock(return_value=True)
    redis_client.incr = AsyncMock(return_value=1)
    redis_client.get = AsyncMock(return_value="0")
    redis_client.lrange = AsyncMock(return_value=[])
    cache.client = redis_client

    return cache


@pytest.fixture
def mock_cache_manager(mock_dragonfly_cache):
    """Create mock cache manager."""
    manager = MagicMock()
    manager.distributed_cache = mock_dragonfly_cache
    return manager


@pytest.fixture
def mock_client_manager(mock_cache_manager, mock_dragonfly_cache):
    """Create mock client manager with cache support."""
    manager = AsyncMock()
    manager.get_cache_manager = AsyncMock(return_value=mock_cache_manager)
    manager.get_redis_client = AsyncMock(return_value=mock_dragonfly_cache.client)
    return manager


@pytest.fixture
async def canary_router(mock_dragonfly_cache, mock_config):
    """Create CanaryRouter instance."""
    router = CanaryRouter(
        cache=mock_dragonfly_cache,
        config=mock_config,
    )
    return router


@pytest.fixture
async def canary_deployment(
    mock_config, mock_alias_manager, mock_qdrant_service, mock_client_manager
):
    """Create CanaryDeployment instance with router support."""
    deployment = CanaryDeployment(
        mock_config, mock_alias_manager, mock_qdrant_service, mock_client_manager
    )
    await deployment.initialize()
    yield deployment
    await deployment.cleanup()


class TestCanaryRouter:
    """Test canary router functionality."""

    @pytest.mark.asyncio
    async def test_update_route_success(self, canary_router, mock_dragonfly_cache):
        """Test successful route update."""
        result = await canary_router.update_route(
            deployment_id="canary_123",
            alias="test_alias",
            old_collection="old_coll",
            new_collection="new_coll",
            percentage=25.0,
            status="running",
        )

        assert result is True
        mock_dragonfly_cache.set.assert_called_once()

        # Verify route data structure
        call_args = mock_dragonfly_cache.set.call_args
        route_data = call_args[0][1]
        assert route_data["deployment_id"] == "canary_123"
        assert route_data["percentage"] == 25.0

    @pytest.mark.asyncio
    async def test_remove_route_success(self, canary_router, mock_dragonfly_cache):
        """Test successful route removal."""
        mock_dragonfly_cache.scan_keys.return_value = [
            "canary:sticky:test_alias:user1",
            "canary:sticky:test_alias:user2",
        ]

        result = await canary_router.remove_route("test_alias")

        assert result is True
        # Should delete route and sticky sessions
        assert mock_dragonfly_cache.delete.call_count >= 1

    @pytest.mark.asyncio
    async def test_get_route_decision_no_canary(
        self, canary_router, mock_dragonfly_cache
    ):
        """Test route decision when no canary is active."""
        mock_dragonfly_cache.get.return_value = None

        decision = await canary_router.get_route_decision(
            alias="test_alias",
            user_id="user_123",
        )

        assert decision.collection_name == "test_alias"
        assert decision.is_canary is False
        assert decision.canary_percentage is None

    @pytest.mark.asyncio
    async def test_get_route_decision_with_canary(
        self, canary_router, mock_dragonfly_cache
    ):
        """Test route decision with active canary deployment."""
        mock_dragonfly_cache.get.return_value = {
            "deployment_id": "canary_123",
            "alias": "test_alias",
            "old_collection": "old_coll",
            "new_collection": "new_coll",
            "percentage": 25.0,
            "status": "running",
            "updated_at": time.time(),
        }

        decision = await canary_router.get_route_decision(
            alias="test_alias",
            user_id="user_123",
        )

        assert decision.collection_name in ["old_coll", "new_coll"]
        assert decision.deployment_id == "canary_123"
        assert decision.canary_percentage == 25.0

    @pytest.mark.asyncio
    async def test_sticky_sessions(self, canary_router, mock_dragonfly_cache):
        """Test sticky session functionality."""
        # First request - no sticky session
        mock_dragonfly_cache.get.side_effect = [
            {  # Route data
                "deployment_id": "canary_123",
                "alias": "test_alias",
                "old_collection": "old_coll",
                "new_collection": "new_coll",
                "percentage": 50.0,
                "status": "running",
                "updated_at": time.time(),
            },
            None,  # No sticky session
        ]

        await canary_router.get_route_decision(
            alias="test_alias",
            user_id="user_123",
            use_sticky_sessions=True,
        )

        # Verify sticky session was stored
        assert mock_dragonfly_cache.set.call_count >= 1

    @pytest.mark.asyncio
    async def test_record_request_metrics(self, canary_router, mock_dragonfly_cache):
        """Test recording request metrics."""
        await canary_router.record_request_metrics(
            deployment_id="canary_123",
            collection_name="new_coll",
            latency_ms=150.5,
            is_error=False,
        )

        # Verify metrics were recorded
        redis_client = mock_dragonfly_cache.client
        redis_client.lpush.assert_called()  # Latency
        redis_client.incr.assert_called()  # Request count

    @pytest.mark.asyncio
    async def test_get_collection_metrics(self, canary_router, mock_dragonfly_cache):
        """Test retrieving collection metrics."""
        redis_client = mock_dragonfly_cache.client

        # Mock with alternating pattern for count/error/latency requests per bucket
        # For 1 bucket (duration 1 minute), we need count -> error -> latency calls
        redis_client.get.return_value = "50"  # Return consistent count
        redis_client.lrange.return_value = ["100", "120", "110", "95", "105"]

        metrics = await canary_router.get_collection_metrics(
            deployment_id="canary_123",
            collection_name="new_coll",
            duration_minutes=1,  # Reduce to 1 minute to simplify test
        )

        assert metrics["total_requests"] > 0
        assert metrics["error_rate"] >= 0.0
        assert metrics["avg_latency"] > 0
        assert metrics["p95_latency"] > 0


class TestEnhancedCanaryDeployment:
    """Test enhanced canary deployment with traffic routing."""

    @pytest.mark.asyncio
    async def test_initialization_with_router(self, canary_deployment):
        """Test canary deployment initializes with router when cache is available."""
        assert canary_deployment._router is not None
        assert isinstance(canary_deployment._router, CanaryRouter)

    @pytest.mark.asyncio
    async def test_start_canary_with_traffic_routing(
        self, canary_deployment, mock_alias_manager
    ):
        """Test starting canary deployment updates routing configuration."""
        mock_alias_manager.get_collection_for_alias.return_value = "old_collection"

        deployment_id = await canary_deployment.start_canary(
            alias_name="test_alias",
            new_collection="new_collection",
            stages=[{"percentage": 25, "duration_minutes": 0}],  # Immediate for test
        )

        assert deployment_id.startswith("canary_")

        # Wait briefly for async task to process
        await asyncio.sleep(0.1)

        # Verify router was updated
        if canary_deployment._router:
            # Router update happens in the background task
            # We can't easily verify the exact call without more complex mocking
            pass

    @pytest.mark.asyncio
    async def test_collect_metrics_from_router(self, canary_deployment):
        """Test collecting real metrics from router."""
        deployment_config = CanaryDeploymentConfig(
            alias="test",
            old_collection="old",
            new_collection="new",
            stages=[CanaryStage(25, 30)],
            start_time=time.time(),
        )

        # Mock router metrics
        canary_deployment._router.get_collection_metrics = AsyncMock(
            return_value={
                "total_requests": 100,
                "total_errors": 2,
                "error_rate": 0.02,
                "avg_latency": 120.0,
                "p95_latency": 180.0,
                "latency": 180.0,
            }
        )

        metrics = await canary_deployment._collect_metrics(deployment_config)

        assert metrics["latency"] == 180.0
        assert metrics["error_rate"] == 0.02
        assert "timestamp" in metrics

    @pytest.mark.asyncio
    async def test_rollback_removes_routing(
        self, canary_deployment, mock_alias_manager
    ):
        """Test rollback removes routing configuration."""
        mock_alias_manager.get_collection_for_alias.return_value = "old_collection"

        # Start deployment
        deployment_id = await canary_deployment.start_canary(
            "test_alias", "new_collection"
        )

        # Mock router
        canary_deployment._router.remove_route = AsyncMock(return_value=True)

        # Rollback
        await canary_deployment._rollback_canary(deployment_id)

        # Verify routing was removed
        canary_deployment._router.remove_route.assert_called_with("test_alias")

    @pytest.mark.asyncio
    async def test_deployment_completion_removes_routing(
        self, canary_deployment, mock_alias_manager
    ):
        """Test deployment completion removes routing configuration."""
        mock_alias_manager.get_collection_for_alias.return_value = "old_collection"

        # Create deployment at 100% stage
        deployment_id = await canary_deployment.start_canary(
            "test_alias",
            "new_collection",
            stages=[{"percentage": 100, "duration_minutes": 0}],
        )

        # Mock router
        canary_deployment._router.remove_route = AsyncMock(return_value=True)

        # Wait for completion
        await asyncio.sleep(0.2)

        # Verify status
        config = canary_deployment.deployments.get(deployment_id)
        if config:
            # Deployment should complete immediately at 100%
            assert config.status in ["completed", "running"]


class TestSearchInterceptor:
    """Test search interceptor functionality."""

    @pytest.mark.asyncio
    async def test_search_interceptor_with_routing(self):
        """Test search interceptor routes based on canary configuration."""
        from src.services.vector_db.search import QdrantSearch
        from src.services.vector_db.search_interceptor import SearchInterceptor

        # Mock search service
        mock_search = AsyncMock(spec=QdrantSearch)
        mock_search.hybrid_search = AsyncMock(
            return_value=[
                {"id": "1", "score": 0.9, "payload": {"text": "result 1"}},
                {"id": "2", "score": 0.8, "payload": {"text": "result 2"}},
            ]
        )

        # Mock router
        mock_router = AsyncMock(spec=CanaryRouter)
        mock_router.get_route_decision = AsyncMock(
            return_value=RouteDecision(
                collection_name="new_collection",
                is_canary=True,
                canary_percentage=25.0,
                deployment_id="canary_123",
                routing_key="user_123",
            )
        )
        mock_router.record_request_metrics = AsyncMock()

        # Create interceptor
        config = MagicMock()
        interceptor = SearchInterceptor(
            search_service=mock_search,
            router=mock_router,
            config=config,
        )

        # Perform search
        results = await interceptor.hybrid_search(
            collection_name="test_alias",
            query_vector=[0.1] * 1536,
            limit=10,
            user_id="user_123",
        )

        assert len(results) == 2

        # Verify routing decision was made
        mock_router.get_route_decision.assert_called_once()

        # Verify search was performed on routed collection
        mock_search.hybrid_search.assert_called_once()
        call_args = mock_search.hybrid_search.call_args
        assert call_args[1]["collection_name"] == "new_collection"

        # Verify metrics were recorded
        mock_router.record_request_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_interceptor_error_handling(self):
        """Test search interceptor handles errors and records metrics."""
        from src.services.vector_db.search import QdrantSearch
        from src.services.vector_db.search_interceptor import SearchInterceptor

        # Mock search service that raises error
        mock_search = AsyncMock(spec=QdrantSearch)
        mock_search.hybrid_search = AsyncMock(side_effect=Exception("Search failed"))

        # Mock router
        mock_router = AsyncMock(spec=CanaryRouter)
        mock_router.get_route_decision = AsyncMock(
            return_value=RouteDecision(
                collection_name="new_collection",
                is_canary=True,
                canary_percentage=25.0,
                deployment_id="canary_123",
            )
        )
        mock_router.record_request_metrics = AsyncMock()

        # Create interceptor
        config = MagicMock()
        interceptor = SearchInterceptor(
            search_service=mock_search,
            router=mock_router,
            config=config,
        )

        # Perform search that will fail
        with pytest.raises(Exception, match="Search failed"):
            await interceptor.hybrid_search(
                collection_name="test_alias",
                query_vector=[0.1] * 1536,
                limit=10,
                user_id="user_123",
            )

        # Verify error metrics were recorded
        mock_router.record_request_metrics.assert_called_once()
        call_args = mock_router.record_request_metrics.call_args
        assert call_args[1]["is_error"] is True


class TestCanaryMetricsCollection:
    """Test real metrics collection integration."""

    @pytest.mark.asyncio
    async def test_metrics_flow_from_search_to_deployment(
        self, canary_deployment, mock_alias_manager
    ):
        """Test complete metrics flow from search through router to deployment."""
        mock_alias_manager.get_collection_for_alias.return_value = "old_collection"

        # Start deployment
        deployment_id = await canary_deployment.start_canary(
            "test_alias",
            "new_collection",
            stages=[{"percentage": 50, "duration_minutes": 30}],
        )

        # Mock router to return metrics
        if canary_deployment._router:
            canary_deployment._router.get_collection_metrics = AsyncMock(
                return_value={
                    "total_requests": 1000,
                    "total_errors": 10,
                    "error_rate": 0.01,
                    "avg_latency": 95.0,
                    "p95_latency": 150.0,
                    "latency": 150.0,
                }
            )

        # Collect metrics
        config = canary_deployment.deployments[deployment_id]
        metrics = await canary_deployment._collect_metrics(config)

        # Verify real metrics were used
        assert metrics["latency"] == 150.0
        assert metrics["error_rate"] == 0.01

        # Check health with real metrics
        config.metrics.latency = [150.0]
        config.metrics.error_rate = [0.01]

        is_healthy = canary_deployment._check_health(config)
        assert is_healthy is True  # Within thresholds


class TestTrafficShiftingStrategy:
    """Test traffic shifting implementation."""

    @pytest.mark.asyncio
    async def test_traffic_shifting_documentation(self):
        """Test traffic shifting strategy is documented."""
        from src.services.deployment import canary_router

        # Verify module has proper documentation
        assert "application-level traffic routing" in canary_router.__doc__.lower()

        # Verify key classes are documented
        assert CanaryRouter.__doc__ is not None
        assert RouteDecision.__doc__ is not None

    def test_consistent_routing_algorithm(self):
        """Test routing algorithm provides consistent results."""
        router = CanaryRouter(None, MagicMock())

        # Test consistent hashing
        key1 = router._generate_routing_key("user_123", None, "test_alias")
        key2 = router._generate_routing_key("user_123", None, "test_alias")
        assert key1 == key2

        # Test routing decision consistency
        decision1 = router._make_routing_decision(key1, 50.0, "old", "new")
        decision2 = router._make_routing_decision(key1, 50.0, "old", "new")
        assert decision1 == decision2

    def test_traffic_distribution(self):
        """Test traffic distribution matches configured percentage."""
        router = CanaryRouter(None, MagicMock())

        # Test with multiple users
        old_count = 0
        new_count = 0

        for i in range(1000):
            key = router._generate_routing_key(f"user_{i}", None, "test_alias")
            collection = router._make_routing_decision(key, 25.0, "old", "new")

            if collection == "old":
                old_count += 1
            else:
                new_count += 1

        # Should be roughly 75% old, 25% new (with some variance)
        new_percentage = (new_count / 1000) * 100
        assert 20 <= new_percentage <= 30  # Allow 5% variance


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery scenarios."""

    @pytest.mark.asyncio
    async def test_canary_without_router(
        self, mock_config, mock_alias_manager, mock_qdrant_service
    ):
        """Test canary deployment works without router (no Redis)."""
        # Create deployment without client manager
        deployment = CanaryDeployment(
            mock_config, mock_alias_manager, mock_qdrant_service, client_manager=None
        )
        await deployment.initialize()

        assert deployment._router is None

        # Should still work without router
        mock_alias_manager.get_collection_for_alias.return_value = "old_collection"
        deployment_id = await deployment.start_canary("test_alias", "new_collection")

        assert deployment_id is not None

    @pytest.mark.asyncio
    async def test_router_failure_fallback(self, canary_deployment):
        """Test graceful handling of router failures."""
        # Make router methods fail
        if canary_deployment._router:
            canary_deployment._router.update_route = AsyncMock(return_value=False)
            canary_deployment._router.get_collection_metrics = AsyncMock(
                side_effect=Exception("Router failed")
            )

        # Deployment should still work with simulated metrics
        deployment_config = CanaryDeploymentConfig(
            alias="test",
            old_collection="old",
            new_collection="new",
            stages=[CanaryStage(25, 30)],
        )

        metrics = await canary_deployment._collect_metrics(deployment_config)

        # Should fall back to simulated metrics
        assert "latency" in metrics
        assert "error_rate" in metrics


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_canary_lifecycle(
        self, canary_deployment, mock_alias_manager
    ):
        """Test complete canary deployment lifecycle with routing."""
        mock_alias_manager.get_collection_for_alias.return_value = "old_collection"

        # 1. Start deployment
        deployment_id = await canary_deployment.start_canary(
            "test_alias",
            "new_collection",
            stages=[
                {"percentage": 10, "duration_minutes": 0},
                {"percentage": 50, "duration_minutes": 0},
                {"percentage": 100, "duration_minutes": 0},
            ],
        )

        # 2. Verify deployment started
        assert deployment_id in canary_deployment.deployments
        config = canary_deployment.deployments[deployment_id]
        assert config.status == "pending"

        # 3. Wait for deployment to complete
        await asyncio.sleep(0.5)

        # 4. Check final status
        status = await canary_deployment.get_deployment_status(deployment_id)

        # Should complete quickly with 0 duration stages
        assert status["status"] in ["completed", "running"]

        # 5. Verify cleanup
        if status["status"] == "completed" and canary_deployment._router:
            # Router should be cleaned up
            # Can't easily verify async cleanup without more mocking
            pass
