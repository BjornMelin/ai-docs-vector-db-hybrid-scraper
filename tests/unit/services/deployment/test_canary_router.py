"""Tests for CanaryRouter traffic routing functionality."""

import hashlib
import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config import UnifiedConfig
from src.services.deployment.canary_router import CanaryRoute
from src.services.deployment.canary_router import CanaryRouter
from src.services.deployment.canary_router import RouteDecision


@pytest.fixture
def mock_config():
    """Create mock unified config."""
    config = MagicMock(spec=UnifiedConfig)
    return config


@pytest.fixture
def mock_dragonfly_cache():
    """Create mock DragonflyDB cache."""
    cache = AsyncMock()
    cache.set = AsyncMock(return_value=True)
    cache.get = AsyncMock(return_value=None)
    cache.delete = AsyncMock(return_value=True)
    cache.scan_keys = AsyncMock(return_value=[])

    # Mock Redis client for metrics
    redis_client = AsyncMock()
    redis_client.lpush = AsyncMock(return_value=1)
    redis_client.expire = AsyncMock(return_value=True)
    redis_client.incr = AsyncMock(return_value=1)
    redis_client.get = AsyncMock(return_value=None)
    redis_client.lrange = AsyncMock(return_value=[])
    cache.client = redis_client

    return cache


@pytest.fixture
def canary_router(mock_dragonfly_cache, mock_config):
    """Create CanaryRouter instance."""
    return CanaryRouter(
        cache=mock_dragonfly_cache,
        config=mock_config,
    )


class TestCanaryRoute:
    """Test CanaryRoute dataclass."""

    def test_canary_route_creation(self):
        """Test creating a canary route."""
        route = CanaryRoute(
            deployment_id="canary_123",
            alias="test_alias",
            old_collection="old_coll",
            new_collection="new_coll",
            percentage=25.0,
            status="running",
        )

        assert route.deployment_id == "canary_123"
        assert route.alias == "test_alias"
        assert route.old_collection == "old_coll"
        assert route.new_collection == "new_coll"
        assert route.percentage == 25.0
        assert route.status == "running"
        assert route.updated_at > 0


class TestRouteDecision:
    """Test RouteDecision dataclass."""

    def test_route_decision_creation(self):
        """Test creating a route decision."""
        decision = RouteDecision(
            collection_name="new_coll",
            is_canary=True,
            canary_percentage=25.0,
            deployment_id="canary_123",
            routing_key="hash_key",
        )

        assert decision.collection_name == "new_coll"
        assert decision.is_canary is True
        assert decision.canary_percentage == 25.0
        assert decision.deployment_id == "canary_123"
        assert decision.routing_key == "hash_key"


class TestCanaryRouterOperations:
    """Test CanaryRouter core operations."""

    @pytest.mark.asyncio
    async def test_update_route_creates_route_data(
        self, canary_router, mock_dragonfly_cache
    ):
        """Test updating route creates correct data structure."""
        result = await canary_router.update_route(
            deployment_id="canary_123",
            alias="test_alias",
            old_collection="old_coll",
            new_collection="new_coll",
            percentage=50.0,
            status="running",
        )

        assert result is True

        # Verify cache set was called
        mock_dragonfly_cache.set.assert_called_once()
        call_args = mock_dragonfly_cache.set.call_args

        # Check key format
        assert call_args[0][0] == "canary:routes:test_alias"

        # Check data structure
        route_data = call_args[0][1]
        assert route_data["deployment_id"] == "canary_123"
        assert route_data["alias"] == "test_alias"
        assert route_data["old_collection"] == "old_coll"
        assert route_data["new_collection"] == "new_coll"
        assert route_data["percentage"] == 50.0
        assert route_data["status"] == "running"
        assert "updated_at" in route_data

    @pytest.mark.asyncio
    async def test_update_route_handles_failure(
        self, canary_router, mock_dragonfly_cache
    ):
        """Test update route handles cache failures gracefully."""
        mock_dragonfly_cache.set.side_effect = Exception("Cache error")

        result = await canary_router.update_route(
            deployment_id="canary_123",
            alias="test_alias",
            old_collection="old_coll",
            new_collection="new_coll",
            percentage=25.0,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_remove_route_clears_all_data(
        self, canary_router, mock_dragonfly_cache
    ):
        """Test removing route clears route and sticky sessions."""
        # Mock sticky session keys
        mock_dragonfly_cache.scan_keys.return_value = [
            "canary:sticky:test_alias:user1",
            "canary:sticky:test_alias:user2",
            "canary:sticky:test_alias:user3",
        ]

        result = await canary_router.remove_route("test_alias")

        assert result is True

        # Should delete route
        assert mock_dragonfly_cache.delete.call_count >= 1
        first_delete = mock_dragonfly_cache.delete.call_args_list[0]
        assert first_delete[0][0] == "canary:routes:test_alias"

        # Should scan for sticky sessions
        mock_dragonfly_cache.scan_keys.assert_called_once_with(
            "canary:sticky:test_alias:*"
        )

        # Should delete all sticky sessions
        assert mock_dragonfly_cache.delete.call_count == 4  # 1 route + 3 sticky

    @pytest.mark.asyncio
    async def test_remove_route_handles_no_sticky_sessions(
        self, canary_router, mock_dragonfly_cache
    ):
        """Test removing route when no sticky sessions exist."""
        mock_dragonfly_cache.scan_keys.return_value = []

        result = await canary_router.remove_route("test_alias")

        assert result is True
        assert mock_dragonfly_cache.delete.call_count == 1


class TestRouteDecisionLogic:
    """Test route decision making logic."""

    @pytest.mark.asyncio
    async def test_route_decision_no_active_canary(
        self, canary_router, mock_dragonfly_cache
    ):
        """Test route decision when no canary deployment exists."""
        mock_dragonfly_cache.get.return_value = None

        decision = await canary_router.get_route_decision(
            alias="test_alias",
            user_id="user_123",
        )

        assert decision.collection_name == "test_alias"
        assert decision.is_canary is False
        assert decision.canary_percentage is None
        assert decision.deployment_id is None

    @pytest.mark.asyncio
    async def test_route_decision_with_inactive_canary(
        self, canary_router, mock_dragonfly_cache
    ):
        """Test route decision when canary is not running."""
        mock_dragonfly_cache.get.return_value = {
            "deployment_id": "canary_123",
            "alias": "test_alias",
            "old_collection": "old_coll",
            "new_collection": "new_coll",
            "percentage": 25.0,
            "status": "paused",  # Not running
            "updated_at": time.time(),
        }

        decision = await canary_router.get_route_decision(
            alias="test_alias",
            user_id="user_123",
        )

        # Should use alias when canary is not running
        assert decision.collection_name == "test_alias"
        assert decision.is_canary is False

    @pytest.mark.asyncio
    async def test_route_decision_error_fallback(
        self, canary_router, mock_dragonfly_cache
    ):
        """Test route decision falls back to alias on error."""
        mock_dragonfly_cache.get.side_effect = Exception("Cache error")

        decision = await canary_router.get_route_decision(
            alias="test_alias",
            user_id="user_123",
        )

        # Should fallback to alias
        assert decision.collection_name == "test_alias"
        assert decision.is_canary is False


class TestRoutingAlgorithm:
    """Test consistent hashing and routing algorithm."""

    def test_generate_routing_key_consistency(self, canary_router):
        """Test routing key generation is consistent."""
        # Same user should get same key
        key1 = canary_router._generate_routing_key("user_123", None, "alias1")
        key2 = canary_router._generate_routing_key("user_123", None, "alias1")
        assert key1 == key2

        # Different users get different keys
        key3 = canary_router._generate_routing_key("user_456", None, "alias1")
        assert key1 != key3

        # Request ID used when no user ID
        key4 = canary_router._generate_routing_key(None, "req_123", "alias1")
        key5 = canary_router._generate_routing_key(None, "req_123", "alias1")
        assert key4 == key5

    def test_make_routing_decision_consistency(self, canary_router):
        """Test routing decision is consistent for same key."""
        routing_key = "abc123def456"

        # Same key should always route to same collection
        decision1 = canary_router._make_routing_decision(
            routing_key, 50.0, "old", "new"
        )
        decision2 = canary_router._make_routing_decision(
            routing_key, 50.0, "old", "new"
        )
        assert decision1 == decision2

    def test_routing_distribution_accuracy(self, canary_router):
        """Test routing distribution matches configured percentage."""
        results = {"old": 0, "new": 0}

        # Test with 30% canary traffic
        for i in range(10000):
            key = hashlib.md5(f"user_{i}".encode()).hexdigest()
            collection = canary_router._make_routing_decision(key, 30.0, "old", "new")
            results[collection] += 1

        # Calculate actual percentage
        new_percentage = (results["new"] / 10000) * 100

        # Should be close to 30% (allow 2% variance)
        assert 28.0 <= new_percentage <= 32.0

    def test_routing_edge_cases(self, canary_router):
        """Test routing with edge case percentages."""
        key = "abcdef1234567890"  # Valid hex string

        # 0% should always route to old
        assert canary_router._make_routing_decision(key, 0.0, "old", "new") == "old"

        # 100% should always route to new
        assert canary_router._make_routing_decision(key, 100.0, "old", "new") == "new"


class TestStickySession:
    """Test sticky session functionality."""

    @pytest.mark.asyncio
    async def test_sticky_session_creation(self, canary_router, mock_dragonfly_cache):
        """Test sticky session is created on first request."""
        # Setup: active canary, no existing sticky session
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

        decision = await canary_router.get_route_decision(
            alias="test_alias",
            user_id="user_123",
            use_sticky_sessions=True,
        )

        # Should create sticky session
        assert mock_dragonfly_cache.set.call_count >= 1

        # Check sticky session data
        set_calls = mock_dragonfly_cache.set.call_args_list
        sticky_call = None
        for call in set_calls:
            if "canary:sticky:" in call[0][0]:
                sticky_call = call
                break

        assert sticky_call is not None
        sticky_data = sticky_call[0][1]
        assert sticky_data["collection"] == decision.collection_name
        assert sticky_data["deployment_id"] == "canary_123"

    @pytest.mark.asyncio
    async def test_sticky_session_reuse(self, canary_router, mock_dragonfly_cache):
        """Test sticky session is reused for returning user."""
        # Setup: active canary with existing sticky session
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
            {  # Existing sticky session
                "collection": "new_coll",
                "deployment_id": "canary_123",
                "created_at": time.time() - 3600,
            },
        ]

        # Mock check sticky session method to verify it's called
        with patch.object(canary_router, "_check_sticky_session") as mock_check:
            mock_check.return_value = RouteDecision(
                collection_name="new_coll",
                is_canary=True,
                canary_percentage=50.0,
                deployment_id="canary_123",
                routing_key="user_123",
            )

            decision = await canary_router.get_route_decision(
                alias="test_alias",
                user_id="user_123",
                use_sticky_sessions=True,
            )

            # Should use sticky session
            assert decision.collection_name == "new_coll"
            assert decision.is_canary is True
            mock_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_sticky_session_disabled(self, canary_router, mock_dragonfly_cache):
        """Test sticky sessions can be disabled."""
        mock_dragonfly_cache.get.return_value = {
            "deployment_id": "canary_123",
            "alias": "test_alias",
            "old_collection": "old_coll",
            "new_collection": "new_coll",
            "percentage": 50.0,
            "status": "running",
            "updated_at": time.time(),
        }

        await canary_router.get_route_decision(
            alias="test_alias",
            user_id="user_123",
            use_sticky_sessions=False,
        )

        # Should not create sticky session
        sticky_sets = [
            call
            for call in mock_dragonfly_cache.set.call_args_list
            if "canary:sticky:" in str(call)
        ]
        assert len(sticky_sets) == 0


class TestMetricsCollection:
    """Test metrics collection and retrieval."""

    @pytest.mark.asyncio
    async def test_record_request_metrics_success(
        self, canary_router, mock_dragonfly_cache
    ):
        """Test recording successful request metrics."""
        await canary_router.record_request_metrics(
            deployment_id="canary_123",
            collection_name="new_coll",
            latency_ms=125.5,
            is_error=False,
        )

        redis_client = mock_dragonfly_cache.client

        # Should record latency
        assert redis_client.lpush.called
        latency_call = redis_client.lpush.call_args
        assert "latency" in latency_call[0][0]
        assert latency_call[0][1] == "125.5"

        # Should increment request count
        incr_calls = [
            call for call in redis_client.incr.call_args_list if "count" in call[0][0]
        ]
        assert len(incr_calls) == 1

    @pytest.mark.asyncio
    async def test_record_request_metrics_error(
        self, canary_router, mock_dragonfly_cache
    ):
        """Test recording error metrics."""
        await canary_router.record_request_metrics(
            deployment_id="canary_123",
            collection_name="new_coll",
            latency_ms=500.0,
            is_error=True,
        )

        redis_client = mock_dragonfly_cache.client

        # Should record error count
        error_calls = [
            call for call in redis_client.incr.call_args_list if "errors" in call[0][0]
        ]
        assert len(error_calls) == 1

    @pytest.mark.asyncio
    async def test_record_request_metrics_handles_failure(
        self, canary_router, mock_dragonfly_cache
    ):
        """Test metrics recording handles failures gracefully."""
        redis_client = mock_dragonfly_cache.client
        redis_client.lpush.side_effect = Exception("Redis error")

        # Should not raise exception
        await canary_router.record_request_metrics(
            deployment_id="canary_123",
            collection_name="new_coll",
            latency_ms=100.0,
            is_error=False,
        )

    @pytest.mark.asyncio
    async def test_get_collection_metrics_aggregation(
        self, canary_router, mock_dragonfly_cache
    ):
        """Test metrics aggregation from multiple buckets."""
        redis_client = mock_dragonfly_cache.client

        # Mock metric data
        redis_client.get.side_effect = [
            "50",  # count bucket 1
            "2",  # errors bucket 1
            "50",  # count bucket 2
            "3",  # errors bucket 2
        ] * 10  # Multiple buckets

        redis_client.lrange.side_effect = [
            ["100", "120", "110"],  # latencies bucket 1
            ["95", "105", "115"],  # latencies bucket 2
        ] * 10

        metrics = await canary_router.get_collection_metrics(
            deployment_id="canary_123",
            collection_name="new_coll",
            duration_minutes=10,
        )

        # Verify aggregated metrics
        assert metrics["total_requests"] > 0
        assert metrics["total_errors"] > 0
        assert metrics["error_rate"] > 0
        assert metrics["avg_latency"] > 0
        assert metrics["p95_latency"] > 0

    @pytest.mark.asyncio
    async def test_get_collection_metrics_empty(
        self, canary_router, mock_dragonfly_cache
    ):
        """Test metrics when no data exists."""
        redis_client = mock_dragonfly_cache.client
        redis_client.get.return_value = None
        redis_client.lrange.return_value = []

        metrics = await canary_router.get_collection_metrics(
            deployment_id="canary_123",
            collection_name="new_coll",
            duration_minutes=10,
        )

        # Should return zeros
        assert metrics["total_requests"] == 0
        assert metrics["total_errors"] == 0
        assert metrics["error_rate"] == 0.0
        assert metrics["avg_latency"] == 0.0
        assert metrics["p95_latency"] == 0.0

    def test_metrics_p95_calculation(self, canary_router):
        """Test 95th percentile calculation."""
        # Create sorted latency list
        latencies = list(range(1, 101))  # 1 to 100

        # Manual p95 calculation
        p95_index = int(len(latencies) * 0.95)
        expected_p95 = sorted(latencies)[p95_index]

        # This would be done in get_collection_metrics
        # For 100 items, 95th percentile is at index 95 (value 96)
        assert expected_p95 == 96
