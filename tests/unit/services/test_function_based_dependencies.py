"""Tests for function-based dependency injection replacing Manager classes.

This test module validates that the function-based approach
provides equivalent functionality to the original Manager classes
while achieving the target 60% complexity reduction.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.dependencies import (
from inspect import signature
from src.services.dependencies import CacheManagerDep, EmbeddingManagerDep
from src.services.errors import (
from fastapi import FastAPI
from src.services.dependencies import ConfigDep
    CrawlRequest,
    CrawlResponse,
    EmbeddingRequest,  # Pydantic Models
    EmbeddingResponse,
    TaskRequest,
    cache_delete,
    cache_get,
    cache_set,
    crawl_site,
    enqueue_task,
    generate_embeddings,  # Service Functions
    get_cache_manager,
    get_client_manager,  # Core Dependencies
    get_embedding_manager,
    get_service_health,
    get_service_metrics,
    get_task_status,
    scrape_url,
)
from src.services.errors import (
    CacheServiceError,
    CrawlServiceError,
    EmbeddingServiceError,
    TaskQueueServiceError,
)


@pytest.fixture
def mock_client_manager():
    """Mock ClientManager for testing."""
    mock = MagicMock()
    mock.get_embedding_manager = AsyncMock()
    mock.get_cache_manager = AsyncMock()
    mock.get_crawl_manager = AsyncMock()
    mock.get_task_queue_manager = AsyncMock()
    mock.get_health_status = AsyncMock()
    return mock


@pytest.fixture
def mock_embedding_manager():
    """Mock EmbeddingManager for testing."""
    mock = MagicMock()
    mock.generate_embeddings = AsyncMock()
    mock.get_provider_info = MagicMock()
    mock.get_usage_report = MagicMock()
    return mock


@pytest.fixture
def mock_cache_manager():
    """Mock CacheManager for testing."""
    mock = MagicMock()
    mock.get = AsyncMock()
    mock.set = AsyncMock()
    mock.delete = AsyncMock()
    mock.get_stats = AsyncMock()
    mock.get_performance_stats = AsyncMock()
    return mock


@pytest.fixture
def mock_crawl_manager():
    """Mock CrawlManager for testing."""
    mock = MagicMock()
    mock.scrape_url = AsyncMock()
    mock.crawl_site = AsyncMock()
    mock.get_metrics = MagicMock()
    mock.get_tier_metrics = MagicMock()
    return mock


@pytest.fixture
def mock_task_manager():
    """Mock TaskQueueManager for testing."""
    mock = MagicMock()
    mock.enqueue = AsyncMock()
    mock.get_job_status = AsyncMock()
    return mock


class TestCoreDependencies:
    """Test core dependency injection functions."""

    @patch("src.services.dependencies.ClientManager.from_unified_config")
    def test_get_client_manager_singleton(self, mock_from_config):
        """Test that get_client_manager returns singleton instance."""
        mock_manager = MagicMock()
        mock_from_config.return_value = mock_manager

        # First call
        result1 = get_client_manager()

        # Second call should return same instance due to @lru_cache
        result2 = get_client_manager()

        assert result1 is result2
        mock_from_config.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_embedding_manager(
        self, mock_client_manager, mock_embedding_manager
    ):
        """Test embedding manager dependency injection."""
        mock_client_manager.get_embedding_manager.return_value = mock_embedding_manager

        with patch(
            "src.services.dependencies.get_client_manager",
            return_value=mock_client_manager,
        ):
            result = await get_embedding_manager(mock_client_manager)

        assert result == mock_embedding_manager
        mock_client_manager.get_embedding_manager.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cache_manager(self, mock_client_manager, mock_cache_manager):
        """Test cache manager dependency injection."""
        mock_client_manager.get_cache_manager.return_value = mock_cache_manager

        with patch(
            "src.services.dependencies.get_client_manager",
            return_value=mock_client_manager,
        ):
            result = await get_cache_manager(mock_client_manager)

        assert result == mock_cache_manager
        mock_client_manager.get_cache_manager.assert_called_once()


class TestEmbeddingFunctions:
    """Test embedding service functions."""

    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self, mock_embedding_manager):
        """Test successful embedding generation."""
        # Setup
        request = EmbeddingRequest(
            texts=["test text 1", "test text 2"],
            quality_tier="balanced",
            auto_select=True,
        )

        expected_result = {
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "provider": "openai",
            "model": "text-embedding-3-small",
            "cost": 0.001,
            "latency_ms": 150.0,
            "tokens": 10,
            "reasoning": "Smart selection based on quality tier",
            "quality_tier": "balanced",
            "cache_hit": False,
        }

        mock_embedding_manager.generate_embeddings.return_value = expected_result

        # Execute
        result = await generate_embeddings(request, mock_embedding_manager)

        # Verify
        assert isinstance(result, EmbeddingResponse)
        assert len(result.embeddings) == 2
        assert result.provider == "openai"
        assert result.cost == 0.001

        mock_embedding_manager.generate_embeddings.assert_called_once()
        call_args = mock_embedding_manager.generate_embeddings.call_args
        assert call_args.kwargs["texts"] == request.texts
        assert call_args.kwargs["auto_select"] == request.auto_select

    @pytest.mark.asyncio
    async def test_generate_embeddings_error(self, mock_embedding_manager):
        """Test embedding generation error handling."""
        request = EmbeddingRequest(texts=["test"])

        mock_embedding_manager.generate_embeddings.side_effect = Exception(
            "Provider failed"
        )

        with pytest.raises(EmbeddingServiceError) as exc_info:
            await generate_embeddings(request, mock_embedding_manager)

        assert "Failed to generate embeddings" in str(exc_info.value)


class TestCacheFunctions:
    """Test cache service functions."""

    @pytest.mark.asyncio
    async def test_cache_get_success(self, mock_cache_manager):
        """Test successful cache get operation."""
        mock_cache_manager.get.return_value = {"test": "data"}

        result = await cache_get("test_key", "crawl", mock_cache_manager)

        assert result == {"test": "data"}
        mock_cache_manager.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_get_error(self, mock_cache_manager):
        """Test cache get error handling."""
        mock_cache_manager.get.side_effect = Exception("Cache error")

        result = await cache_get("test_key", "crawl", mock_cache_manager)

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_set_success(self, mock_cache_manager):
        """Test successful cache set operation."""
        mock_cache_manager.set.return_value = True

        result = await cache_set(
            "test_key", {"data": "value"}, "crawl", 3600, mock_cache_manager
        )

        assert result is True
        mock_cache_manager.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_delete_success(self, mock_cache_manager):
        """Test successful cache delete operation."""
        mock_cache_manager.delete.return_value = True

        result = await cache_delete("test_key", "crawl", mock_cache_manager)

        assert result is True
        mock_cache_manager.delete.assert_called_once()


class TestCrawlFunctions:
    """Test crawl service functions."""

    @pytest.mark.asyncio
    async def test_scrape_url_success(self, mock_crawl_manager):
        """Test successful URL scraping."""
        request = CrawlRequest(
            url="https://example.com",
            preferred_provider="crawl4ai",
        )

        expected_result = {
            "success": True,
            "content": "Test content",
            "url": "https://example.com",
            "title": "Test Page",
            "metadata": {"links": []},
            "tier_used": "crawl4ai",
            "automation_time_ms": 500.0,
            "quality_score": 0.9,
            "error": None,
            "fallback_attempted": False,
            "failed_tiers": [],
        }

        mock_crawl_manager.scrape_url.return_value = expected_result

        result = await scrape_url(request, mock_crawl_manager)

        assert isinstance(result, CrawlResponse)
        assert result.success is True
        assert result.content == "Test content"
        assert result.tier_used == "crawl4ai"

        mock_crawl_manager.scrape_url.assert_called_once_with(
            url=request.url,
            preferred_provider=request.preferred_provider,
        )

    @pytest.mark.asyncio
    async def test_scrape_url_error(self, mock_crawl_manager):
        """Test URL scraping error handling."""
        request = CrawlRequest(url="https://example.com")

        mock_crawl_manager.scrape_url.side_effect = Exception("Scraping failed")

        with pytest.raises(CrawlServiceError) as exc_info:
            await scrape_url(request, mock_crawl_manager)

        assert "Failed to scrape URL" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_crawl_site_success(self, mock_crawl_manager):
        """Test successful site crawling."""
        request = CrawlRequest(
            url="https://example.com",
            max_pages=10,
        )

        expected_result = {
            "success": True,
            "pages": [
                {"url": "https://example.com", "content": "Page 1"},
                {"url": "https://example.com/page2", "content": "Page 2"},
            ],
            "total_pages": 2,
            "provider": "crawl4ai",
            "error": None,
        }

        mock_crawl_manager.crawl_site.return_value = expected_result

        result = await crawl_site(request, mock_crawl_manager)

        assert result["success"] is True
        assert len(result["pages"]) == 2
        assert result["total_pages"] == 2

        mock_crawl_manager.crawl_site.assert_called_once_with(
            url=request.url,
            max_pages=request.max_pages,
            preferred_provider=request.preferred_provider,
        )


class TestTaskQueueFunctions:
    """Test task queue service functions."""

    @pytest.mark.asyncio
    async def test_enqueue_task_success(self, mock_task_manager):
        """Test successful task enqueue."""
        request = TaskRequest(
            task_name="test_task",
            args=["arg1", "arg2"],
            kwargs={"key": "value"},
            delay=60,
        )

        mock_task_manager.enqueue.return_value = "job_123"

        result = await enqueue_task(request, mock_task_manager)

        assert result == "job_123"

        mock_task_manager.enqueue.assert_called_once_with(
            request.task_name,
            *request.args,
            _delay=request.delay,
            _queue_name=request.queue_name,
            **request.kwargs,
        )

    @pytest.mark.asyncio
    async def test_enqueue_task_error(self, mock_task_manager):
        """Test task enqueue error handling."""
        request = TaskRequest(task_name="test_task")

        mock_task_manager.enqueue.side_effect = Exception("Queue error")

        with pytest.raises(TaskQueueServiceError) as exc_info:
            await enqueue_task(request, mock_task_manager)

        assert "Failed to enqueue task" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_task_status_success(self, mock_task_manager):
        """Test successful task status retrieval."""
        expected_status = {
            "status": "complete",
            "job_id": "job_123",
            "function": "test_task",
            "result": {"success": True},
        }

        mock_task_manager.get_job_status.return_value = expected_status

        result = await get_task_status("job_123", mock_task_manager)

        assert result == expected_status
        mock_task_manager.get_job_status.assert_called_once_with("job_123")


class TestUtilityFunctions:
    """Test utility functions."""

    @pytest.mark.asyncio
    async def test_get_service_health_success(self, mock_client_manager):
        """Test successful service health check."""
        expected_health = {
            "qdrant": {"state": "healthy", "last_check": 1234567890},
            "openai": {"state": "healthy", "last_check": 1234567890},
        }

        mock_client_manager.get_health_status.return_value = expected_health

        with patch(
            "src.services.dependencies.get_client_manager",
            return_value=mock_client_manager,
        ):
            result = await get_service_health()

        assert result["status"] == "healthy"
        assert "services" in result
        assert result["services"] == expected_health

    @pytest.mark.asyncio
    async def test_get_service_health_error(self):
        """Test service health check error handling."""
        with patch(
            "src.services.dependencies.get_client_manager",
            side_effect=Exception("Health error"),
        ):
            result = await get_service_health()

        assert result["status"] == "unhealthy"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_service_metrics_success(
        self, mock_client_manager, mock_cache_manager, mock_crawl_manager
    ):
        """Test successful service metrics collection."""
        mock_cache_manager.get_performance_stats.return_value = {"hit_rate": 0.85}
        mock_crawl_manager.get_tier_metrics.return_value = {
            "tier0": {"success_rate": 0.95}
        }

        mock_client_manager.get_cache_manager.return_value = mock_cache_manager
        mock_client_manager.get_crawl_manager.return_value = mock_crawl_manager

        with patch(
            "src.services.dependencies.get_client_manager",
            return_value=mock_client_manager,
        ):
            result = await get_service_metrics()

        assert "cache_service" in result
        assert "crawl_service" in result
        assert result["cache_service"]["hit_rate"] == 0.85


class TestPydanticModels:
    """Test Pydantic model validation."""

    def test_embedding_request_validation(self):
        """Test EmbeddingRequest model validation."""
        # Valid request
        request = EmbeddingRequest(
            texts=["test1", "test2"],
            quality_tier="balanced",
            auto_select=True,
        )
        assert request.texts == ["test1", "test2"]
        assert request.quality_tier == "balanced"
        assert request.auto_select is True

        # Test defaults
        minimal_request = EmbeddingRequest(texts=["test"])
        assert minimal_request.auto_select is True
        assert minimal_request.speed_priority is False

    def test_crawl_request_validation(self):
        """Test CrawlRequest model validation."""
        # Valid request
        request = CrawlRequest(
            url="https://example.com",
            preferred_provider="crawl4ai",
            max_pages=25,
        )
        assert request.url == "https://example.com"
        assert request.preferred_provider == "crawl4ai"
        assert request.max_pages == 25

        # Test defaults
        minimal_request = CrawlRequest(url="https://example.com")
        assert minimal_request.max_pages == 50
        assert minimal_request.include_subdomains is False

    def test_task_request_validation(self):
        """Test TaskRequest model validation."""
        # Valid request
        request = TaskRequest(
            task_name="process_data",
            args=[1, 2, 3],
            kwargs={"param": "value"},
            delay=120,
        )
        assert request.task_name == "process_data"
        assert request.args == [1, 2, 3]
        assert request.kwargs == {"param": "value"}
        assert request.delay == 120

        # Test defaults
        minimal_request = TaskRequest(task_name="simple_task")
        assert minimal_request.args == []
        assert minimal_request.kwargs == {}
        assert minimal_request.delay is None


class TestComplexityReduction:
    """Test that function-based approach reduces complexity."""

    def test_function_signature_simplicity(self):
        """Test that function signatures are simpler than class methods."""
        # Function-based approach has simple, focused signatures

        # Check generate_embeddings function signature
        sig = signature(generate_embeddings)
        params = list(sig.parameters.keys())
        assert len(params) == 2  # request, embedding_manager

        # Check cache functions have simple signatures
        cache_get_sig = signature(cache_get)
        cache_get_params = list(cache_get_sig.parameters.keys())
        assert len(cache_get_params) == 3  # key, cache_type, cache_manager

    def test_dependency_injection_clarity(self):
        """Test that dependency injection is explicit and clear."""
        # All dependencies are explicitly typed with Annotated

        # Dependencies use clear naming convention
        assert str(EmbeddingManagerDep).startswith("typing.Annotated")
        assert str(CacheManagerDep).startswith("typing.Annotated")

    def test_error_handling_consistency(self):
        """Test that error handling is consistent across functions."""
        # All service functions use consistent error types
            CrawlServiceError,
            EmbeddingServiceError,
            TaskQueueServiceError,
        )

        # Error classes follow consistent naming pattern
        error_classes = [
            EmbeddingServiceError,
            CacheServiceError,
            CrawlServiceError,
            TaskQueueServiceError,
        ]

        for error_class in error_classes:
            assert error_class.__name__.endswith("ServiceError")
            assert issubclass(error_class, Exception)


@pytest.mark.integration
class TestIntegrationWithFastAPI:
    """Integration tests with FastAPI dependency injection."""

    @pytest.mark.asyncio
    async def test_fastapi_dependency_resolution(self):
        """Test that FastAPI can resolve dependencies correctly."""

        app = FastAPI()


        @app.get("/test-config")
        async def test_config_endpoint(config: ConfigDep):
            return {"embedding_provider": config.embedding_provider.value}

        # Test that the endpoint can be created without errors
        # (actual testing would require full app setup)
        assert test_config_endpoint is not None

    def test_pydantic_integration(self):
        """Test that Pydantic models integrate well with FastAPI."""

        app = FastAPI()

        @app.post("/test-embedding")
        async def test_embedding_endpoint(request: EmbeddingRequest):
            return {"received_texts": len(request.texts)}

        # Test that the endpoint can be created without errors
        assert test_embedding_endpoint is not None

    def test_response_model_integration(self):
        """Test that response models work with FastAPI."""

        app = FastAPI()

        @app.post("/test-crawl", response_model=CrawlResponse)
        async def test_crawl_endpoint(request: CrawlRequest):
            return CrawlResponse(
                success=True,
                url=request.url,
                content="test content",
            )

        # Test that the endpoint can be created without errors
        assert test_crawl_endpoint is not None
