"""Tests for SearchInterceptor canary routing functionality."""

import asyncio
import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from src.config import UnifiedConfig
from src.services.deployment.canary_router import CanaryRouter
from src.services.deployment.canary_router import RouteDecision
from src.services.errors import QdrantServiceError
from src.services.vector_db.search import QdrantSearch
from src.services.vector_db.search_interceptor import SearchInterceptor


@pytest.fixture
def mock_config():
    """Create mock unified config."""
    config = MagicMock(spec=UnifiedConfig)
    config.performance = MagicMock()
    config.performance.enable_canary_metrics = True
    return config


@pytest.fixture
def mock_search_service():
    """Create mock QdrantSearch service."""
    search = AsyncMock(spec=QdrantSearch)
    search.hybrid_search = AsyncMock(
        return_value=[
            {"id": "1", "score": 0.95, "payload": {"text": "Result 1"}},
            {"id": "2", "score": 0.85, "payload": {"text": "Result 2"}},
            {"id": "3", "score": 0.75, "payload": {"text": "Result 3"}},
        ]
    )
    search.multi_stage_search = AsyncMock(
        return_value=[
            {"id": "4", "score": 0.90, "payload": {"text": "Multi-stage result"}},
        ]
    )
    search.hyde_search = AsyncMock(
        return_value=[
            {"id": "5", "score": 0.88, "payload": {"text": "HyDE result"}},
        ]
    )
    search.filtered_search = AsyncMock(
        return_value=[
            {"id": "6", "score": 0.92, "payload": {"text": "Filtered result"}},
        ]
    )
    return search


@pytest.fixture
def mock_router():
    """Create mock CanaryRouter."""
    router = AsyncMock(spec=CanaryRouter)
    router.get_route_decision = AsyncMock(
        return_value=RouteDecision(
            collection_name="test_collection",
            is_canary=False,
            canary_percentage=None,
            deployment_id=None,
            routing_key=None,
        )
    )
    router.record_request_metrics = AsyncMock()
    return router


@pytest.fixture
def search_interceptor(mock_search_service, mock_router, mock_config):
    """Create SearchInterceptor instance."""
    return SearchInterceptor(
        search_service=mock_search_service,
        router=mock_router,
        config=mock_config,
    )


class TestSearchInterceptorBasics:
    """Test basic SearchInterceptor functionality."""

    def test_initialization(self, mock_search_service, mock_router, mock_config):
        """Test SearchInterceptor initialization."""
        interceptor = SearchInterceptor(
            search_service=mock_search_service,
            router=mock_router,
            config=mock_config,
        )

        assert interceptor._search == mock_search_service
        assert interceptor._router == mock_router
        assert interceptor._config == mock_config
        assert interceptor._metrics_enabled is True

    def test_initialization_without_router(self, mock_search_service, mock_config):
        """Test SearchInterceptor works without router."""
        interceptor = SearchInterceptor(
            search_service=mock_search_service,
            router=None,
            config=mock_config,
        )

        assert interceptor._router is None


class TestHybridSearchInterception:
    """Test hybrid search with canary routing."""

    @pytest.mark.asyncio
    async def test_hybrid_search_no_canary(
        self, search_interceptor, mock_search_service, mock_router
    ):
        """Test hybrid search when no canary is active."""
        mock_router.get_route_decision.return_value = RouteDecision(
            collection_name="original_collection",
            is_canary=False,
        )

        results = await search_interceptor.hybrid_search(
            collection_name="test_alias",
            query_vector=[0.1] * 1536,
            limit=10,
            user_id="user_123",
        )

        assert len(results) == 3
        assert results[0]["id"] == "1"

        # Verify routing decision was made
        mock_router.get_route_decision.assert_called_once_with(
            alias="test_alias",
            user_id="user_123",
            request_id=None,
            use_sticky_sessions=True,
        )

        # Verify search was called with routed collection
        mock_search_service.hybrid_search.assert_called_once()
        call_args = mock_search_service.hybrid_search.call_args[1]
        assert call_args["collection_name"] == "original_collection"

    @pytest.mark.asyncio
    async def test_hybrid_search_with_canary(
        self, search_interceptor, mock_search_service, mock_router
    ):
        """Test hybrid search with active canary deployment."""
        mock_router.get_route_decision.return_value = RouteDecision(
            collection_name="new_collection_v2",
            is_canary=True,
            canary_percentage=25.0,
            deployment_id="canary_123",
            routing_key="user_123_hash",
        )

        results = await search_interceptor.hybrid_search(
            collection_name="test_alias",
            query_vector=[0.2] * 1536,
            sparse_vector={100: 0.5, 200: 0.3},
            limit=5,
            score_threshold=0.7,
            fusion_type="dbsf",
            search_accuracy="accurate",
            user_id="user_123",
            request_id="req_456",
        )

        assert len(results) == 3

        # Verify search used canary collection
        mock_search_service.hybrid_search.assert_called_once()
        call_args = mock_search_service.hybrid_search.call_args[1]
        assert call_args["collection_name"] == "new_collection_v2"
        assert call_args["sparse_vector"] == {100: 0.5, 200: 0.3}
        assert call_args["fusion_type"] == "dbsf"

    @pytest.mark.asyncio
    async def test_hybrid_search_metrics_recording(
        self, search_interceptor, mock_router
    ):
        """Test metrics are recorded for successful searches."""
        mock_router.get_route_decision.return_value = RouteDecision(
            collection_name="new_collection",
            is_canary=True,
            deployment_id="canary_123",
        )

        await search_interceptor.hybrid_search(
            collection_name="test_alias",
            query_vector=[0.1] * 1536,
            user_id="user_123",
        )

        # Verify metrics were recorded
        mock_router.record_request_metrics.assert_called_once()
        call_args = mock_router.record_request_metrics.call_args[1]
        assert call_args["deployment_id"] == "canary_123"
        assert call_args["collection_name"] == "new_collection"
        assert call_args["latency_ms"] > 0
        assert call_args["is_error"] is False

    @pytest.mark.asyncio
    async def test_hybrid_search_error_metrics(
        self, search_interceptor, mock_search_service, mock_router
    ):
        """Test error metrics are recorded on search failure."""
        mock_router.get_route_decision.return_value = RouteDecision(
            collection_name="new_collection",
            is_canary=True,
            deployment_id="canary_123",
        )

        mock_search_service.hybrid_search.side_effect = QdrantServiceError(
            "Search failed"
        )

        with pytest.raises(QdrantServiceError):
            await search_interceptor.hybrid_search(
                collection_name="test_alias",
                query_vector=[0.1] * 1536,
                user_id="user_123",
            )

        # Verify error metrics were recorded
        mock_router.record_request_metrics.assert_called_once()
        call_args = mock_router.record_request_metrics.call_args[1]
        assert call_args["is_error"] is True

    @pytest.mark.asyncio
    async def test_hybrid_search_without_router(self, mock_search_service, mock_config):
        """Test hybrid search works without router."""
        interceptor = SearchInterceptor(
            search_service=mock_search_service,
            router=None,
            config=mock_config,
        )

        results = await interceptor.hybrid_search(
            collection_name="test_collection",
            query_vector=[0.1] * 1536,
            user_id="user_123",
        )

        assert len(results) == 3

        # Should call search directly with original collection
        mock_search_service.hybrid_search.assert_called_once()
        call_args = mock_search_service.hybrid_search.call_args[1]
        assert call_args["collection_name"] == "test_collection"


class TestMultiStageSearchInterception:
    """Test multi-stage search with canary routing."""

    @pytest.mark.asyncio
    async def test_multi_stage_search_routing(
        self, search_interceptor, mock_search_service, mock_router
    ):
        """Test multi-stage search routes correctly."""
        mock_router.get_route_decision.return_value = RouteDecision(
            collection_name="canary_collection",
            is_canary=True,
            deployment_id="canary_456",
        )

        stages = [
            {
                "query_vector": [0.1] * 1536,
                "vector_name": "dense",
                "limit": 100,
            },
            {
                "query_vector": [0.2] * 1536,
                "vector_name": "dense",
                "limit": 50,
            },
        ]

        results = await search_interceptor.multi_stage_search(
            collection_name="test_alias",
            stages=stages,
            limit=10,
            fusion_algorithm="rrf",
            user_id="user_789",
        )

        assert len(results) == 1
        assert results[0]["id"] == "4"

        # Verify routing
        mock_router.get_route_decision.assert_called_once()

        # Verify search used routed collection
        mock_search_service.multi_stage_search.assert_called_once()
        call_args = mock_search_service.multi_stage_search.call_args[1]
        assert call_args["collection_name"] == "canary_collection"
        assert call_args["stages"] == stages

    @pytest.mark.asyncio
    async def test_multi_stage_search_metrics_disabled(
        self, mock_search_service, mock_router
    ):
        """Test multi-stage search with metrics disabled."""
        config = MagicMock()
        config.performance = MagicMock()
        config.performance.enable_canary_metrics = False

        interceptor = SearchInterceptor(
            search_service=mock_search_service,
            router=mock_router,
            config=config,
        )

        mock_router.get_route_decision.return_value = RouteDecision(
            collection_name="new_collection",
            is_canary=True,
            deployment_id="canary_123",
        )

        await interceptor.multi_stage_search(
            collection_name="test_alias",
            stages=[
                {"query_vector": [0.1] * 1536, "vector_name": "dense", "limit": 10}
            ],
            user_id="user_123",
        )

        # Should not record metrics when disabled
        mock_router.record_request_metrics.assert_not_called()


class TestHyDESearchInterception:
    """Test HyDE search with canary routing."""

    @pytest.mark.asyncio
    async def test_hyde_search_routing(
        self, search_interceptor, mock_search_service, mock_router
    ):
        """Test HyDE search routes correctly."""
        mock_router.get_route_decision.return_value = RouteDecision(
            collection_name="hyde_canary_collection",
            is_canary=True,
            canary_percentage=50.0,
            deployment_id="canary_hyde",
        )

        query = "What is machine learning?"
        query_embedding = [0.1] * 1536
        hypothetical_embeddings = [
            [0.2] * 1536,
            [0.3] * 1536,
            [0.4] * 1536,
        ]

        results = await search_interceptor.hyde_search(
            collection_name="test_alias",
            query=query,
            query_embedding=query_embedding,
            hypothetical_embeddings=hypothetical_embeddings,
            limit=10,
            fusion_algorithm="dbsf",
            search_accuracy="accurate",
            user_id="user_hyde",
            request_id="req_hyde",
        )

        assert len(results) == 1
        assert results[0]["id"] == "5"

        # Verify search parameters passed correctly
        mock_search_service.hyde_search.assert_called_once()
        call_args = mock_search_service.hyde_search.call_args[1]
        assert call_args["collection_name"] == "hyde_canary_collection"
        assert call_args["query"] == query
        assert len(call_args["hypothetical_embeddings"]) == 3


class TestFilteredSearchInterception:
    """Test filtered search with canary routing."""

    @pytest.mark.asyncio
    async def test_filtered_search_routing(
        self, search_interceptor, mock_search_service, mock_router
    ):
        """Test filtered search routes correctly."""
        mock_router.get_route_decision.return_value = RouteDecision(
            collection_name="filtered_canary",
            is_canary=True,
            deployment_id="canary_filtered",
        )

        filters = {
            "doc_type": "technical",
            "language": "en",
            "created_after": "2024-01-01",
        }

        results = await search_interceptor.filtered_search(
            collection_name="test_alias",
            query_vector=[0.5] * 1536,
            filters=filters,
            limit=20,
            search_accuracy="fast",
            user_id="user_filter",
        )

        assert len(results) == 1
        assert results[0]["id"] == "6"

        # Verify filters passed correctly
        mock_search_service.filtered_search.assert_called_once()
        call_args = mock_search_service.filtered_search.call_args[1]
        assert call_args["collection_name"] == "filtered_canary"
        assert call_args["filters"] == filters
        assert call_args["search_accuracy"] == "fast"


class TestErrorHandling:
    """Test error handling in search interceptor."""

    @pytest.mark.asyncio
    async def test_router_error_continues_search(
        self, search_interceptor, mock_search_service, mock_router
    ):
        """Test search continues when router fails."""
        mock_router.get_route_decision.side_effect = Exception("Router failed")

        results = await search_interceptor.hybrid_search(
            collection_name="test_alias",
            query_vector=[0.1] * 1536,
            user_id="user_123",
        )

        assert len(results) == 3

        # Should use original collection name
        mock_search_service.hybrid_search.assert_called_once()
        call_args = mock_search_service.hybrid_search.call_args[1]
        assert call_args["collection_name"] == "test_alias"

    @pytest.mark.asyncio
    async def test_metrics_error_does_not_fail_search(
        self, search_interceptor, mock_router
    ):
        """Test search succeeds even if metrics recording fails."""
        mock_router.get_route_decision.return_value = RouteDecision(
            collection_name="new_collection",
            is_canary=True,
            deployment_id="canary_123",
        )

        mock_router.record_request_metrics.side_effect = Exception("Metrics failed")

        # Should not raise exception
        results = await search_interceptor.hybrid_search(
            collection_name="test_alias",
            query_vector=[0.1] * 1536,
            user_id="user_123",
        )

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_error_propagates(
        self, search_interceptor, mock_search_service
    ):
        """Test search errors are propagated correctly."""
        mock_search_service.hybrid_search.side_effect = QdrantServiceError(
            "Collection not found"
        )

        with pytest.raises(QdrantServiceError, match="Collection not found"):
            await search_interceptor.hybrid_search(
                collection_name="missing_collection",
                query_vector=[0.1] * 1536,
            )


class TestPerformance:
    """Test performance aspects of search interceptor."""

    @pytest.mark.asyncio
    async def test_minimal_overhead_without_router(
        self, mock_search_service, mock_config
    ):
        """Test interceptor adds minimal overhead when no router."""
        interceptor = SearchInterceptor(
            search_service=mock_search_service,
            router=None,
            config=mock_config,
        )

        start_time = time.time()
        await interceptor.hybrid_search(
            collection_name="test_collection",
            query_vector=[0.1] * 1536,
        )
        elapsed = time.time() - start_time

        # Should complete very quickly (< 10ms overhead)
        assert elapsed < 0.01

        # Only one call to underlying service
        assert mock_search_service.hybrid_search.call_count == 1

    @pytest.mark.asyncio
    async def test_latency_measurement_accuracy(self, search_interceptor, mock_router):
        """Test latency measurement is accurate."""
        mock_router.get_route_decision.return_value = RouteDecision(
            collection_name="new_collection",
            is_canary=True,
            deployment_id="canary_123",
        )

        # Add artificial delay to search
        async def delayed_search(*args, **kwargs):
            await asyncio.sleep(0.1)  # 100ms delay
            return [{"id": "1", "score": 0.9, "payload": {}}]

        search_interceptor._search.hybrid_search = delayed_search

        await search_interceptor.hybrid_search(
            collection_name="test_alias",
            query_vector=[0.1] * 1536,
            user_id="user_123",
        )

        # Check recorded latency
        mock_router.record_request_metrics.assert_called_once()
        call_args = mock_router.record_request_metrics.call_args[1]
        latency_ms = call_args["latency_ms"]

        # Should be at least 100ms (with some tolerance)
        assert 95 <= latency_ms <= 150
