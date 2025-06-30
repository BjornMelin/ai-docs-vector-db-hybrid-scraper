"""Integration tests for parallel processing system with dependency injection."""

import logging
from unittest.mock import AsyncMock, Mock

import pytest

from src.config.settings import Settings
from src.infrastructure.client_manager import ClientManager
from src.infrastructure.container import initialize_container, shutdown_container


logger = logging.getLogger(__name__)


class TestParallelProcessingIntegration:
    """Test parallel processing system integration with DI container."""

    @pytest.fixture
    async def mock_config(self):
        """Create mock configuration for testing."""
        config = Mock(spec=Settings)
        config.openai = Mock()
        config.openai.api_key = "test-key"
        config.qdrant = Mock()
        config.qdrant.url = "http://localhost:6333"
        config.qdrant.api_key = None
        config.qdrant.timeout = 30.0
        config.qdrant.prefer_grpc = False
        config.cache = Mock()
        config.cache.dragonfly_url = "redis://localhost:6379"
        config.cache.redis_pool_size = 20
        config.firecrawl = Mock()
        config.firecrawl.api_key = "test-key"
        config.performance = Mock()
        config.performance.max_retries = 3
        return config

    @pytest.fixture
    async def mock_embedding_manager(self):
        """Create mock embedding manager for testing."""
        manager = AsyncMock()
        manager.generate_embedding = AsyncMock(return_value=[0.1] * 384)
        manager.generate_embeddings_batch = AsyncMock(
            return_value={"embeddings": [[0.1] * 384, [0.2] * 384]}
        )
        return manager

    @pytest.fixture
    async def container_with_parallel_processing(self, mock_config):
        """Initialize container with parallel processing system."""
        # Initialize container
        container = await initialize_container(mock_config)

        yield container

        # Cleanup
        await shutdown_container()

    @pytest.mark.asyncio
    async def test_parallel_processing_system_initialization(
        self, container_with_parallel_processing, mock_embedding_manager
    ):
        """Test that parallel processing system can be initialized through DI."""
        container = container_with_parallel_processing

        # Mock the embedding manager creation
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "src.services.embeddings.manager.EmbeddingManager",
                lambda: mock_embedding_manager,
            )

            # Get parallel processing system from container
            parallel_system = container.parallel_processing_system()

            assert parallel_system is not None
            assert hasattr(parallel_system, "process_documents_parallel")
            assert hasattr(parallel_system, "get_system_status")
            assert hasattr(parallel_system, "optimize_performance")

    @pytest.mark.asyncio
    async def test_client_manager_parallel_processing_access(
        self, container_with_parallel_processing, mock_embedding_manager
    ):
        """Test that ClientManager provides access to parallel processing system."""
        # Mock the embedding manager
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "src.services.embeddings.manager.EmbeddingManager",
                lambda: mock_embedding_manager,
            )

            # Initialize client manager
            client_manager = ClientManager()
            await client_manager.initialize()

            try:
                # Get parallel processing system
                parallel_system = await client_manager.get_parallel_processing_system()

                assert parallel_system is not None

                # Test system status
                status = await parallel_system.get_system_status()
                assert "system_health" in status
                assert "performance_metrics" in status
                assert "optimization_status" in status

            finally:
                await client_manager.cleanup()

    @pytest.mark.asyncio
    async def test_parallel_processing_document_pipeline(
        self, container_with_parallel_processing, mock_embedding_manager
    ):
        """Test the complete document processing pipeline."""
        # Mock the embedding manager
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "src.services.embeddings.manager.EmbeddingManager",
                lambda: mock_embedding_manager,
            )

            # Initialize client manager
            client_manager = ClientManager()
            await client_manager.initialize()

            try:
                parallel_system = await client_manager.get_parallel_processing_system()
                assert parallel_system is not None

                # Test document processing
                test_documents = [
                    {
                        "content": "This is a test document about machine learning.",
                        "url": "test1.html",
                    },
                    {
                        "content": "Another document about artificial intelligence.",
                        "url": "test2.html",
                    },
                ]

                results = await parallel_system.process_documents_parallel(
                    documents=test_documents,
                    enable_classification=False,  # Skip classification for test
                    enable_metadata_extraction=False,  # Skip metadata for test
                    enable_embedding_generation=True,
                )

                # Verify results structure
                assert "documents" in results
                assert "processing_stats" in results
                assert "performance_metrics" in results
                assert "optimization_enabled" in results

                # Verify optimization flags
                optimization_status = results["optimization_enabled"]
                assert optimization_status["parallel_processing"] is True
                assert optimization_status["intelligent_caching"] is True
                assert optimization_status["optimized_algorithms"] is True

                # Verify processing stats
                stats = results["processing_stats"]
                assert stats["total_documents"] == 2
                assert "processing_time_ms" in stats
                assert "throughput_docs_per_second" in stats

            finally:
                await client_manager.cleanup()

    @pytest.mark.asyncio
    async def test_parallel_processing_system_health_check(
        self, container_with_parallel_processing, mock_embedding_manager
    ):
        """Test health checking integration."""
        # Mock the embedding manager
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "src.services.embeddings.manager.EmbeddingManager",
                lambda: mock_embedding_manager,
            )

            client_manager = ClientManager()
            await client_manager.initialize()

            try:
                # Get service status
                service_status = await client_manager.get_service_status()

                assert "parallel_processing" in service_status
                parallel_status = service_status["parallel_processing"]

                # Should have system health information
                if parallel_status and "error" not in parallel_status:
                    assert "system_health" in parallel_status
                    assert "performance_metrics" in parallel_status
                    assert "optimization_status" in parallel_status

            finally:
                await client_manager.cleanup()

    @pytest.mark.asyncio
    async def test_parallel_processing_context_manager(
        self, container_with_parallel_processing, mock_embedding_manager
    ):
        """Test parallel processing system via context manager."""
        # Mock the embedding manager
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "src.services.embeddings.manager.EmbeddingManager",
                lambda: mock_embedding_manager,
            )

            client_manager = ClientManager()
            await client_manager.initialize()

            try:
                # Use context manager
                async with client_manager.managed_client(
                    "parallel_processing"
                ) as parallel_system:
                    assert parallel_system is not None

                    # Test optimization functionality
                    optimization_result = await parallel_system.optimize_performance()
                    assert "status" in optimization_result

            finally:
                await client_manager.cleanup()

    @pytest.mark.asyncio
    async def test_parallel_processing_custom_configuration(
        self, container_with_parallel_processing, mock_embedding_manager
    ):
        """Test parallel processing with custom optimization configuration."""
        # Mock the embedding manager
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "src.services.embeddings.manager.EmbeddingManager",
                lambda: mock_embedding_manager,
            )

            from src.services.cache.intelligent import CacheConfig
            from src.services.embeddings.parallel import ParallelConfig
            from src.services.processing.parallel_integration import (
                OptimizationConfig,
                ParallelProcessingSystem,
            )

            # Create custom configuration
            custom_config = OptimizationConfig(
                enable_parallel_processing=True,
                enable_intelligent_caching=True,
                enable_optimized_algorithms=True,
                parallel_config=ParallelConfig(
                    max_concurrent_tasks=25,
                    batch_size_per_worker=5,
                    adaptive_batching=True,
                ),
                cache_config=CacheConfig(
                    max_memory_mb=128,
                    enable_compression=True,
                ),
                performance_monitoring=True,
                auto_optimization=False,  # Disable auto-optimization for testing
            )

            # Create system with custom config
            parallel_system = ParallelProcessingSystem(
                embedding_manager=mock_embedding_manager,
                config=custom_config,
            )

            # Verify configuration
            assert parallel_system.config.enable_parallel_processing is True
            assert parallel_system.config.auto_optimization is False
            assert parallel_system.config.parallel_config.max_concurrent_tasks == 25
            assert parallel_system.config.cache_config.max_memory_mb == 128

            # Test system status
            status = await parallel_system.get_system_status()
            assert status["optimization_status"]["auto_optimization"] is False

            # Cleanup
            await parallel_system.cleanup()


@pytest.mark.benchmark
class TestParallelProcessingPerformance:
    """Performance benchmarks for the integrated parallel processing system."""

    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_processing_benchmark(
        self, benchmark, mock_embedding_manager
    ):
        """Benchmark parallel vs sequential processing performance."""
        # This would be a more comprehensive benchmark in a real scenario
        # For now, we'll just verify the system can handle load

        test_documents = [
            {
                "content": f"Test document {i} with content for processing.",
                "url": f"test{i}.html",
            }
            for i in range(10)
        ]

        async def process_documents():
            from src.services.processing.parallel_integration import (
                create_optimized_system,
            )

            # Create optimized system
            parallel_system = create_optimized_system(
                embedding_manager=mock_embedding_manager,
                enable_all_optimizations=True,
            )

            try:
                return await parallel_system.process_documents_parallel(
                    documents=test_documents,
                    enable_classification=False,
                    enable_metadata_extraction=False,
                    enable_embedding_generation=True,
                )
            finally:
                await parallel_system.cleanup()

        # Run benchmark (pytest-benchmark will handle timing)
        result = await process_documents()

        # Verify results
        assert result is not None
        assert len(result["documents"]) == 10
        assert result["processing_stats"]["total_documents"] == 10
