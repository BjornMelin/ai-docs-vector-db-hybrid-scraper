"""Test infrastructure validation.

Tests to verify the  test infrastructure is working correctly,
including fixtures, parallel execution, and performance monitoring.
"""

import asyncio
import importlib
import time

import httpx
import pytest


class TestBasicInfrastructure:
    """Test basic test infrastructure components."""

    def test_app_config_session_fixture(self, app_config):
        """Test session-scoped app configuration fixture."""
        assert app_config["test_mode"] is True
        assert "parallel_workers" in app_config
        assert "timeouts" in app_config
        assert app_config["timeouts"]["default"] == 30

    def test_ci_environment_detection(self, ci_environment_config):
        """Test CI environment detection and configuration."""
        assert "is_ci" in ci_environment_config
        assert "parallel_workers" in ci_environment_config
        assert "platform" in ci_environment_config
        assert ci_environment_config["platform"]["os"] is not None

    def test_performance_monitor(self, performance_monitor):
        """Test performance monitoring fixture."""
        # Test basic monitoring
        performance_monitor.start()

        # Simulate some work
        time.sleep(0.01)

        metrics = performance_monitor.stop()

        assert "duration_seconds" in metrics
        assert metrics["duration_seconds"] > 0
        assert metrics["duration_seconds"] < 1.0  # Should be very fast

    def test_performance_monitor_checkpoints(self, performance_monitor):
        """Test performance monitoring with checkpoints."""
        performance_monitor.start()

        # Add checkpoints
        checkpoint1 = performance_monitor.checkpoint("start_processing")
        assert checkpoint1["name"] == "start_processing"
        assert "elapsed_seconds" in checkpoint1

        time.sleep(0.01)

        checkpoint2 = performance_monitor.checkpoint("end_processing")
        assert checkpoint2["elapsed_seconds"] > checkpoint1["elapsed_seconds"]

        metrics = performance_monitor.stop()
        assert len(metrics["checkpoints"]) == 2


class TestMockFactories:
    """Test mock factory infrastructure."""

    def test_external_service_factory(self, external_service_factory):
        """Test external service mock factory."""
        # Test OpenAI client creation
        openai_client = external_service_factory.create_openai_client()
        assert hasattr(openai_client, "embeddings")
        assert hasattr(openai_client.embeddings, "create")

        # Test Qdrant client creation
        qdrant_client = external_service_factory.create_qdrant_client()
        assert hasattr(qdrant_client, "create_collection")
        assert hasattr(qdrant_client, "search")

    def test_data_factory(self, data_factory):
        """Test data mock factory."""
        # Test embedding points creation
        points = data_factory.create_embedding_points(count=3, vector_size=384)
        assert len(points) == 3
        assert all(len(point["vector"]) == 384 for point in points)
        assert all("payload" in point for point in points)

        # Test crawl response creation
        response = data_factory.create_crawl_response(success=True)
        assert response["status_code"] == 200
        assert "html" in response
        assert "metadata" in response

    def test_mock_openai_factory(self, mock_openai_factory):
        """Test OpenAI mock factory fixture."""
        client = mock_openai_factory(embedding_dim=512, model="custom-model")

        # Verify configuration
        # The actual verification would depend on how the mock is set up
        assert client is not None

    def test_mock_qdrant_factory(self, mock_qdrant_factory):
        """Test Qdrant mock factory fixture."""
        client = mock_qdrant_factory(vector_size=512, collection_exists=False)

        # Verify configuration
        assert client is not None


@pytest.mark.asyncio
class TestAsyncInfrastructure:
    """Test async infrastructure components."""

    @pytest.mark.asyncio
    async def test_async_resource_manager(self, async_resource_manager):
        """Test async resource manager."""

        # Register a mock resource
        class MockResource:
            async def aclose(self):
                pass

        mock_resource = MockResource()
        await async_resource_manager.register_resource(mock_resource)

        # Manager should clean up automatically at test end
        assert mock_resource in async_resource_manager._resources

    @pytest.mark.asyncio
    async def test_async_test_context(self, async_test_context):
        """Test async test context fixture."""

        # Test task creation
        async def dummy_task():
            await asyncio.sleep(0.01)
            return "done"

        task = await async_test_context["create_task"](dummy_task())
        result = await task
        assert result == "done"

        # Test queue creation
        queue = async_test_context["create_queue"](maxsize=5)
        await queue.put("test_item")
        item = await queue.get()
        assert item == "test_item"

    @pytest.mark.asyncio
    async def test_async_timeout_manager(self, async_timeout_manager):
        """Test async timeout manager."""
        # Test successful operation within timeout
        async with async_timeout_manager.timeout(1.0):
            await asyncio.sleep(0.01)

        # Test timeout assertion
        with pytest.raises(AssertionError, match="Operation timed out"):
            async with async_timeout_manager.timeout(0.001):
                await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_async_performance_profiler(self, async_performance_profiler):
        """Test async performance profiler."""
        # Profile an async operation
        async with async_performance_profiler.profile("test_operation"):
            await asyncio.sleep(0.01)

        profile = async_performance_profiler.get_profile("test_operation")
        assert profile is not None
        assert "duration_seconds" in profile
        assert profile["duration_seconds"] > 0


@pytest.mark.respx
class TestRespxIntegration:
    """Test respx HTTP mocking integration."""

    @pytest.mark.asyncio
    async def test_respx_mock_fixture(self, respx_mock):
        """Test respx mock fixture with pre-configured routes."""

        async with httpx.AsyncClient() as client:
            # Test pre-configured OpenAI route
            response = await client.post("https://api.openai.com/v1/embeddings")
            assert response.status_code == 200

            data = response.json()
            assert "data" in data
            assert len(data["data"]) > 0
            assert "embedding" in data["data"][0]

            # Test generic test endpoint
            response = await client.get("https://test.example.com")
            assert response.status_code == 200
            assert "Test content" in response.text


class TestParallelExecution:
    """Test parallel execution support."""

    def test_worker_config(self, worker_config):
        """Test worker configuration fixture."""
        assert "worker_id" in worker_config
        assert "is_master" in worker_config
        assert "isolation_level" in worker_config

        # Worker should have either "master" ID or worker-specific ID
        assert (
            worker_config["worker_id"] in ["master"]
            or "gw" in worker_config["worker_id"]
        )

    def test_parallel_resource_manager(self, parallel_resource_manager):
        """Test parallel resource manager."""
        # Test port allocation
        port1 = parallel_resource_manager.get_free_port(8000)
        port2 = parallel_resource_manager.get_free_port(8000)

        assert port1 != port2  # Should get different ports
        assert port1 > 0
        assert port2 > 0

        # Test temp directory creation
        temp_dir = parallel_resource_manager.create_worker_temp_dir("test_dir")
        assert temp_dir.exists()
        assert temp_dir.is_dir()


@pytest.mark.fast
class TestPerformanceFixtures:
    """Test performance-related fixtures."""

    def test_performance_assertions(self, performance_monitor):
        """Test performance assertion capabilities."""
        performance_monitor.start()

        # Very fast operation
        time.sleep(0.001)

        performance_monitor.stop()

        # Should not raise - operation was fast enough
        performance_monitor.assert_performance(max_duration=1.0)

        # Should raise - operation was too slow for this constraint
        with pytest.raises(AssertionError, match="exceeded max duration"):
            performance_monitor.assert_performance(max_duration=0.0001)


@pytest.mark.hypothesis
class TestPropertyBasedInfrastructure:
    """Test property-based testing infrastructure."""

    def test_embedding_helper_available(self):
        """Test that deterministic embedding helper is exposed."""
        module = importlib.import_module("tests.utils.ai_testing_utilities")
        generate_embeddings = module.EmbeddingTestUtils.generate_test_embeddings

        embeddings = generate_embeddings(count=1, dim=128, seed=7)
        assert len(embeddings) == 1

        embedding = embeddings[0]
        assert len(embedding) == 128
        assert all(isinstance(value, int | float) for value in embedding)
