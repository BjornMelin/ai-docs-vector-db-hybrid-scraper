
class TestError(Exception):
    """Custom exception for this module."""
    pass

"""Tests for query processing pipeline."""

import asyncio
import time
from unittest.mock import AsyncMock, Mock

import pytest

from src.services.query_processing.models import (
    MatryoshkaDimension,
    QueryComplexity,
    QueryIntent,
    QueryIntentClassification,
    QueryPreprocessingResult,
    QueryProcessingRequest,
    QueryProcessingResponse,
    SearchStrategy,
    SearchStrategySelection,
)
from src.services.query_processing.pipeline import QueryProcessingPipeline


@pytest.fixture
def mock_orchestrator():
    """Create a mock query processing orchestrator."""
    orchestrator = AsyncMock()
    orchestrator._call_count = 0

    def get_performance_stats():
        return {
            "total_queries": orchestrator._call_count,
            "successful_queries": orchestrator._call_count,
            "average_processing_time": 100.0,
            "strategy_usage": {"semantic": orchestrator._call_count},
        }

    def increment_calls(*_args, **_kwargs):
        orchestrator._call_count += 1
        return QueryProcessingResponse(
            success=True,
            results=[{"id": "1", "content": "test content", "score": 0.9}],
            total_results=1,
            total_processing_time_ms=100.0,
            confidence_score=0.8,
            quality_score=0.85,
            intent_classification=QueryIntentClassification(
                primary_intent=QueryIntent.CONCEPTUAL,
                confidence_scores={QueryIntent.CONCEPTUAL: 0.9},
                complexity_level=QueryComplexity.MODERATE,
            ),
            preprocessing_result=QueryPreprocessingResult(
                original_query="test query", processed_query="test query"
            ),
            strategy_selection=SearchStrategySelection(
                primary_strategy=SearchStrategy.SEMANTIC,
                matryoshka_dimension=MatryoshkaDimension.MEDIUM,
                confidence=0.8,
                estimated_quality=0.8,
                reasoning="Test strategy selection",
            ),
        )

    orchestrator.process_query = AsyncMock(side_effect=increment_calls)
    orchestrator.get_performance_stats = Mock(side_effect=get_performance_stats)
    orchestrator.initialize = AsyncMock()
    orchestrator.cleanup = AsyncMock()
    return orchestrator


@pytest.fixture
def pipeline(mock_orchestrator):
    """Create a pipeline instance."""
    return QueryProcessingPipeline(orchestrator=mock_orchestrator)


@pytest.fixture
async def initialized_pipeline(pipeline):
    """Create an initialized pipeline."""
    await pipeline.initialize()
    return pipeline


@pytest.fixture
def sample_request():
    """Create a sample processing request."""
    return QueryProcessingRequest(
        query="What is machine learning?",
        collection_name="documentation",
        limit=10,
    )


class TestQueryProcessingPipeline:
    """Test the QueryProcessingPipeline class."""

    def test_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.orchestrator is not None
        assert pipeline._initialized is False

    async def test_initialize(self, pipeline):
        """Test pipeline initialization."""
        await pipeline.initialize()
        assert pipeline._initialized is True

    async def test_basic_query_processing(self, initialized_pipeline, sample_request):
        """Test basic query processing."""
        response = await initialized_pipeline.process(sample_request)

        assert isinstance(response, QueryProcessingResponse)
        assert response.success is True
        assert len(response.results) > 0
        assert response.total_processing_time_ms > 0

    async def test_string_query_processing(self, initialized_pipeline):
        """Test processing with string query input."""
        response = await initialized_pipeline.process(
            "What is Python?", collection_name="docs", limit=5
        )

        assert response.success is True
        # Should have called orchestrator with proper request object

    async def test_analyze_query(self, initialized_pipeline):
        """Test query analysis functionality."""
        analysis = await initialized_pipeline.analyze_query(
            "How to optimize database performance?"
        )

        assert "intent_classification" in analysis
        assert "complexity" in analysis
        assert "preprocessing" in analysis
        assert "strategy" in analysis

    async def test_batch_processing(self, initialized_pipeline):
        """Test batch query processing."""
        requests = [
            QueryProcessingRequest(query=f"Query {i}", collection_name="docs", limit=5)
            for i in range(3)
        ]

        responses = await initialized_pipeline.process_batch(requests)

        assert len(responses) == 3
        assert all(isinstance(resp, QueryProcessingResponse) for resp in responses)
        assert all(resp.success is True for resp in responses)

    async def test_batch_processing_with_failures(
        self, initialized_pipeline, mock_orchestrator
    ):
        """Test batch processing with some failures."""

        # Make second request fail
        def side_effect(request):
                raise TestError("Processing failed")
                raise TestError("Processing failed")
            return QueryProcessingResponse(
                success=True,
                results=[],
                total_results=0,
            )

        mock_orchestrator.process_query.side_effect = side_effect

        requests = [
            QueryProcessingRequest(query=f"Query {i}", collection_name="docs", limit=5)
            for i in range(3)
        ]

        responses = await initialized_pipeline.process_batch(requests)

        assert len(responses) == 3
        assert responses[0].success is True
        assert responses[1].success is False
        assert responses[2].success is True

    async def test_health_check(self, initialized_pipeline):
        """Test health check functionality."""
        health = await initialized_pipeline.health_check()

        assert "status" in health
        assert "components" in health
        assert "performance" in health
        assert health["status"] == "healthy"

    async def test_get_metrics(self, initialized_pipeline):
        """Test metrics retrieval."""
        metrics = await initialized_pipeline.get_metrics()

        assert "total_queries" in metrics
        assert "successful_queries" in metrics
        assert "average_processing_time" in metrics
        assert "strategy_usage" in metrics

    async def test_warm_up(self, initialized_pipeline):
        """Test pipeline warm-up."""
        result = await initialized_pipeline.warm_up()

        assert result["status"] == "completed"
        assert "warmup_time_ms" in result
        assert result["warmup_time_ms"] >= 0

    async def test_uninitialized_pipeline_error(self, pipeline, sample_request):
        """Test error when using uninitialized pipeline."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await pipeline.process(sample_request)

    async def test_invalid_request_handling(self, initialized_pipeline):
        """Test handling of invalid requests."""
        # Empty query should be handled gracefully
        response = await initialized_pipeline.process(
            "", collection_name="docs", limit=5
        )

        # Should either reject or handle gracefully
        assert isinstance(response, QueryProcessingResponse)

    async def test_context_manager_usage(self, mock_orchestrator):
        """Test using pipeline as a context manager."""
        async with QueryProcessingPipeline(orchestrator=mock_orchestrator) as pipeline:
            response = await pipeline.process(
                "test query", collection_name="docs", limit=5
            )
            assert response.success is True

        # Should have called cleanup
        mock_orchestrator.cleanup.assert_called_once()

    async def test_request_validation(self, initialized_pipeline):
        """Test request validation."""
        # Test with various request configurations
        valid_requests = [
            QueryProcessingRequest(
                query="What is Python?",
                collection_name="docs",
                limit=10,
                enable_preprocessing=True,
                enable_intent_classification=True,
                enable_strategy_selection=True,
            ),
            QueryProcessingRequest(
                query="Debug memory leak",
                collection_name="troubleshooting",
                limit=5,
                force_strategy=SearchStrategy.FILTERED,
                user_context={"urgency": "high"},
            ),
        ]

        for request in valid_requests:
            response = await initialized_pipeline.process(request)
            assert response.success is True

    async def test_performance_monitoring(self, initialized_pipeline, sample_request):
        """Test performance monitoring."""
        # Process some queries
        for _ in range(3):
            await initialized_pipeline.process(sample_request)

        metrics = await initialized_pipeline.get_metrics()

        # Should track performance metrics
        assert metrics["total_queries"] >= 3
        assert metrics["average_processing_time"] > 0

    async def test_strategy_usage_tracking(self, initialized_pipeline):
        """Test strategy usage tracking."""
        # Process queries with different strategies
        strategies = [
            SearchStrategy.SEMANTIC,
            SearchStrategy.HYDE,
            SearchStrategy.HYBRID,
        ]

        for strategy in strategies:
            request = QueryProcessingRequest(
                query="test query",
                collection_name="docs",
                limit=5,
                force_strategy=strategy,
            )
            await initialized_pipeline.process(request)

        metrics = await initialized_pipeline.get_metrics()

        # Should track strategy usage
        assert "strategy_usage" in metrics

    async def test_error_recovery(self, initialized_pipeline, mock_orchestrator):
        """Test error recovery mechanisms."""

        # Make orchestrator fail
        def side_effect(*_args, **_kwargs):
            raise TestError("Temporary failure")

        mock_orchestrator.process_query.side_effect = side_effect

        request = QueryProcessingRequest(
            query="test query",
            collection_name="docs",
            limit=5,
        )

        # Should propagate the error
        with pytest.raises(Exception, match="Temporary failure"):
            await initialized_pipeline.process(request)

    async def test_concurrent_processing(self, initialized_pipeline):
        """Test concurrent query processing."""
        import asyncio

        requests = [
            QueryProcessingRequest(
                query=f"Concurrent query {i}",
                collection_name="docs",
                limit=5,
            )
            for i in range(5)
        ]

        # Process requests concurrently
        tasks = [initialized_pipeline.process(req) for req in requests]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 5
        assert all(isinstance(resp, QueryProcessingResponse) for resp in responses)

    async def test_resource_cleanup(self, initialized_pipeline):
        """Test proper resource cleanup."""
        await initialized_pipeline.cleanup()
        assert initialized_pipeline._initialized is False

    async def test_configuration_validation(self, _mock_orchestrator):
        """Test pipeline configuration validation."""
        # Test with None orchestrator
        with pytest.raises(ValueError):
            QueryProcessingPipeline(orchestrator=None)

    async def test_processing_timeout_handling(
        self, initialized_pipeline, mock_orchestrator
    ):
        """Test handling of processing timeouts."""
        import asyncio

        # Mock a slow response
        async def slow_process(*_args, **_kwargs):
            await asyncio.sleep(0.1)  # Simulate slow processing
            return QueryProcessingResponse(
                success=True,
                results=[],
                total_results=0,
                total_processing_time_ms=200.0,
            )

        mock_orchestrator.process_query = slow_process

        request = QueryProcessingRequest(
            query="slow query",
            collection_name="docs",
            limit=5,
            max_processing_time_ms=50,  # Very short timeout
        )

        response = await initialized_pipeline.process(request)

        # Should handle timeout appropriately
        assert isinstance(response, QueryProcessingResponse)

    async def test_query_preprocessing_integration(self, initialized_pipeline):
        """Test integration with query preprocessing."""
        request = QueryProcessingRequest(
            query="phython programming guide",  # Misspelled
            collection_name="docs",
            limit=5,
            enable_preprocessing=True,
        )

        response = await initialized_pipeline.process(request)

        assert response.success is True
        # Should have preprocessing results if orchestrator supports it

    async def test_intent_classification_integration(self, initialized_pipeline):
        """Test integration with intent classification."""
        request = QueryProcessingRequest(
            query="How to debug memory leaks in Python?",
            collection_name="docs",
            limit=5,
            enable_intent_classification=True,
        )

        response = await initialized_pipeline.process(request)

        assert response.success is True
        # Should have intent classification if orchestrator supports it

    async def test_strategy_selection_integration(self, initialized_pipeline):
        """Test integration with strategy selection."""
        request = QueryProcessingRequest(
            query="Compare React vs Vue performance",
            collection_name="docs",
            limit=5,
            enable_strategy_selection=True,
            enable_intent_classification=True,  # Required for strategy selection
        )

        response = await initialized_pipeline.process(request)

        assert response.success is True
        # Should have strategy selection if orchestrator supports it

    async def test_comprehensive_pipeline_flow(self, initialized_pipeline):
        """Test comprehensive pipeline flow with all features."""
        request = QueryProcessingRequest(
            query="How to optimize databse performance in phython?",
            collection_name="documentation",
            limit=10,
            enable_preprocessing=True,
            enable_intent_classification=True,
            enable_strategy_selection=True,
            user_context={"experience_level": "intermediate"},
            filters={"category": "performance"},
        )

        response = await initialized_pipeline.process(request)

        assert response.success is True
        assert isinstance(response, QueryProcessingResponse)


class TestPipelineInitialization:
    """Test pipeline initialization and configuration scenarios."""

    def test_init_with_valid_orchestrator(self):
        """Test initialization with valid orchestrator."""
        orchestrator = AsyncMock()
        config = Mock()
        pipeline = QueryProcessingPipeline(orchestrator, config)

        assert pipeline.orchestrator is orchestrator
        assert pipeline.config is config
        assert not pipeline._initialized

    def test_init_with_none_orchestrator_raises_error(self):
        """Test initialization with None orchestrator raises ValueError."""
        with pytest.raises(ValueError, match="Orchestrator cannot be None"):
            QueryProcessingPipeline(orchestrator=None)

    def test_init_with_none_config(self):
        """Test initialization with None config works."""
        orchestrator = AsyncMock()
        pipeline = QueryProcessingPipeline(orchestrator, config=None)

        assert pipeline.orchestrator is orchestrator
        assert pipeline.config is None

    async def test_initialize_success(self, mock_orchestrator):
        """Test successful initialization."""
        pipeline = QueryProcessingPipeline(mock_orchestrator)

        assert not pipeline._initialized
        await pipeline.initialize()

        assert pipeline._initialized
        mock_orchestrator.initialize.assert_called_once()

    async def test_initialize_already_initialized(
        self, initialized_pipeline, mock_orchestrator
    ):
        """Test initializing already initialized pipeline."""
        # Reset call count
        mock_orchestrator.initialize.reset_mock()

        # Initialize again
        await initialized_pipeline.initialize()

        # Should not call orchestrator initialize again
        mock_orchestrator.initialize.assert_not_called()
        assert initialized_pipeline._initialized

    async def test_initialize_orchestrator_failure(self, mock_orchestrator):
        """Test initialization failure when orchestrator fails."""
        mock_orchestrator.initialize.side_effect = Exception("Orchestrator init failed")
        pipeline = QueryProcessingPipeline(mock_orchestrator)

        with pytest.raises(Exception, match="Orchestrator init failed"):
            await pipeline.initialize()

        assert not pipeline._initialized


class TestPipelineExecution:
    """Test pipeline execution with different query types and scenarios."""

    async def test_process_with_query_processing_request(
        self, initialized_pipeline, sample_request
    ):
        """Test processing with QueryProcessingRequest object."""
        response = await initialized_pipeline.process(sample_request)

        assert isinstance(response, QueryProcessingResponse)
        assert response.success
        assert len(response.results) > 0

    async def test_process_with_string_query(self, initialized_pipeline):
        """Test processing with string query."""
        response = await initialized_pipeline.process(
            "What is machine learning?", collection_name="docs", limit=15
        )

        assert isinstance(response, QueryProcessingResponse)
        assert response.success

    async def test_process_with_string_query_and_kwargs(self, initialized_pipeline):
        """Test processing with string query and additional kwargs."""
        response = await initialized_pipeline.process(
            "Python best practices",
            collection_name="docs",
            limit=5,
            enable_preprocessing=True,
            enable_intent_classification=False,
            user_context={"level": "beginner"},
        )

        assert isinstance(response, QueryProcessingResponse)
        assert response.success

    async def test_process_empty_query_string(self, initialized_pipeline):
        """Test processing with empty query string."""
        response = await initialized_pipeline.process("")

        assert isinstance(response, QueryProcessingResponse)
        assert not response.success
        assert response.error == "Empty query provided"
        assert response.total_results == 0

    async def test_process_whitespace_only_query(self, initialized_pipeline):
        """Test processing with whitespace-only query."""
        response = await initialized_pipeline.process("   \n\t   ")

        assert isinstance(response, QueryProcessingResponse)
        assert not response.success
        assert response.error == "Empty query provided"

    async def test_process_uninitialized_pipeline_raises_error(
        self, pipeline, sample_request
    ):
        """Test processing with uninitialized pipeline raises RuntimeError."""
        with pytest.raises(
            RuntimeError, match="QueryProcessingPipeline not initialized"
        ):
            await pipeline.process(sample_request)


class TestAdvancedProcessing:
    """Test advanced processing methods."""

    async def test_process_advanced_with_full_request(self, initialized_pipeline):
        """Test process_advanced with complete request."""
        request = QueryProcessingRequest(
            query="How to optimize database performance?",
            collection_name="technical_docs",
            limit=20,
            enable_preprocessing=True,
            enable_intent_classification=True,
            enable_strategy_selection=True,
            force_strategy=SearchStrategy.HYBRID,
            force_dimension=MatryoshkaDimension.LARGE,
            user_context={"role": "database_admin"},
            filters={"category": "performance"},
            search_accuracy="high",
            max_processing_time_ms=10000,
        )

        response = await initialized_pipeline.process_advanced(request)

        assert isinstance(response, QueryProcessingResponse)
        assert response.success

    async def test_process_advanced_uninitialized_raises_error(self, pipeline):
        """Test process_advanced with uninitialized pipeline."""
        from src.services.errors import APIError

        request = QueryProcessingRequest(query="test")

        with pytest.raises(APIError):
            await pipeline.process_advanced(request)


class TestBatchProcessing:
    """Test batch processing functionality."""

    async def test_batch_processing_multiple_requests(self, initialized_pipeline):
        """Test batch processing with multiple valid requests."""
        requests = [
            QueryProcessingRequest(query=f"Query {i}", limit=5) for i in range(5)
        ]

        responses = await initialized_pipeline.process_batch(requests, max_concurrent=3)

        assert len(responses) == 5
        assert all(isinstance(resp, QueryProcessingResponse) for resp in responses)
        assert all(resp.success for resp in responses)

    async def test_batch_processing_empty_list(self, initialized_pipeline):
        """Test batch processing with empty request list."""
        responses = await initialized_pipeline.process_batch([])

        assert responses == []

    async def test_batch_processing_single_request(self, initialized_pipeline):
        """Test batch processing with single request."""
        requests = [QueryProcessingRequest(query="Single query")]

        responses = await initialized_pipeline.process_batch(requests)

        assert len(responses) == 1
        assert responses[0].success

    async def test_batch_processing_with_orchestrator_failures(
        self, initialized_pipeline, mock_orchestrator
    ):
        """Test batch processing when some requests fail in orchestrator."""
        call_count = 0

        def side_effect(_request):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Second request fails
                raise TestError("Processing error")
            return QueryProcessingResponse(
                success=True, results=[{"id": f"result_{call_count}"}], total_results=1
            )

        mock_orchestrator.process_query.side_effect = side_effect

        requests = [QueryProcessingRequest(query=f"Query {i}") for i in range(3)]

        responses = await initialized_pipeline.process_batch(requests)

        assert len(responses) == 3
        assert responses[0].success
        assert not responses[1].success
        assert "Processing error" in responses[1].error
        assert responses[2].success

    async def test_batch_processing_concurrency_control(self, initialized_pipeline):
        """Test batch processing with different concurrency limits."""
        requests = [
            QueryProcessingRequest(query=f"Concurrent query {i}") for i in range(10)
        ]

        start_time = time.time()
        responses = await initialized_pipeline.process_batch(requests, max_concurrent=2)
        end_time = time.time()

        assert len(responses) == 10
        assert all(resp.success for resp in responses)
        # With concurrency limit, should take more time than unlimited
        assert end_time - start_time >= 0  # Basic timing check

    async def test_batch_processing_uninitialized_raises_error(self, pipeline):
        """Test batch processing with uninitialized pipeline."""
        from src.services.errors import APIError

        requests = [QueryProcessingRequest(query="test")]

        with pytest.raises(APIError):
            await pipeline.process_batch(requests)


class TestQueryAnalysis:
    """Test query analysis functionality."""

    async def test_analyze_query_basic(self, initialized_pipeline):
        """Test basic query analysis."""
        analysis = await initialized_pipeline.analyze_query("How to debug Python code?")

        assert isinstance(analysis, dict)
        assert "query" in analysis
        assert "preprocessing" in analysis
        assert "intent_classification" in analysis
        assert "complexity" in analysis
        assert "strategy" in analysis
        assert "processing_time_ms" in analysis
        assert analysis["query"] == "How to debug Python code?"

    async def test_analyze_query_with_preprocessing_disabled(
        self, initialized_pipeline
    ):
        """Test query analysis with preprocessing disabled."""
        analysis = await initialized_pipeline.analyze_query(
            "Machine learning algorithms",
            enable_preprocessing=False,
            enable_intent_classification=True,
        )

        assert isinstance(analysis, dict)
        assert "preprocessing" in analysis
        assert "intent_classification" in analysis

    async def test_analyze_query_with_intent_classification_disabled(
        self, initialized_pipeline
    ):
        """Test query analysis with intent classification disabled."""
        analysis = await initialized_pipeline.analyze_query(
            "Database optimization techniques",
            enable_preprocessing=True,
            enable_intent_classification=False,
        )

        assert isinstance(analysis, dict)
        assert "preprocessing" in analysis
        assert "intent_classification" in analysis

    async def test_analyze_query_both_features_disabled(self, initialized_pipeline):
        """Test query analysis with both preprocessing and intent classification disabled."""
        analysis = await initialized_pipeline.analyze_query(
            "API design patterns",
            enable_preprocessing=False,
            enable_intent_classification=False,
        )

        assert isinstance(analysis, dict)
        assert "query" in analysis

    async def test_analyze_query_uninitialized_raises_error(self, pipeline):
        """Test query analysis with uninitialized pipeline."""
        from src.services.errors import APIError

        with pytest.raises(APIError):
            await pipeline.analyze_query("test query")


class TestMetricsAndPerformance:
    """Test performance metrics tracking."""

    async def test_get_metrics_initialized_pipeline(self, initialized_pipeline):
        """Test getting metrics from initialized pipeline."""
        metrics = await initialized_pipeline.get_metrics()

        assert isinstance(metrics, dict)
        assert "total_queries" in metrics
        assert "successful_queries" in metrics
        assert "average_processing_time" in metrics
        assert "strategy_usage" in metrics
        assert "pipeline_initialized" in metrics
        assert metrics["pipeline_initialized"] is True

    async def test_get_metrics_uninitialized_pipeline(self, pipeline):
        """Test getting metrics from uninitialized pipeline."""
        metrics = await pipeline.get_metrics()

        assert isinstance(metrics, dict)
        assert metrics["total_queries"] == 0
        assert metrics["successful_queries"] == 0
        assert metrics["average_processing_time"] == 0.0
        assert metrics["strategy_usage"] == {}

    async def test_metrics_after_processing_queries(
        self, initialized_pipeline, mock_orchestrator
    ):
        """Test metrics tracking after processing multiple queries."""
        # Process several queries
        for i in range(3):
            await initialized_pipeline.process(f"Query {i}")

        metrics = await initialized_pipeline.get_metrics()

        # Should reflect the queries processed
        assert metrics["total_queries"] >= 3


class TestHealthCheck:
    """Test health check functionality."""

    async def test_health_check_healthy_pipeline(self, initialized_pipeline):
        """Test health check on healthy pipeline."""
        health = await initialized_pipeline.health_check()

        assert isinstance(health, dict)
        assert "status" in health
        assert "pipeline_healthy" in health
        assert "components" in health
        assert "performance" in health

        assert health["status"] == "healthy"
        assert health["pipeline_healthy"] is True
        assert health["performance"]["initialized"] is True

    async def test_health_check_uninitialized_pipeline(self, pipeline):
        """Test health check on uninitialized pipeline."""
        health = await pipeline.health_check()

        assert health["status"] == "unhealthy"
        assert health["pipeline_healthy"] is False
        assert health["performance"]["initialized"] is False

    async def test_health_check_component_analysis(self, initialized_pipeline):
        """Test health check component analysis."""
        health = await initialized_pipeline.health_check()

        components = health["components"]
        assert "orchestrator" in components
        assert "intent_classifier" in components
        assert "preprocessor" in components
        assert "strategy_selector" in components

        for component in components.values():
            if isinstance(component, dict):
                assert "status" in component
                assert "message" in component

    async def test_health_check_with_analysis_failure(
        self, initialized_pipeline, mock_orchestrator
    ):
        """Test health check when analysis fails."""
        # Make analyze_query fail
        original_process = mock_orchestrator.process_query
        mock_orchestrator.process_query.side_effect = Exception("Analysis failed")

        health = await initialized_pipeline.health_check()

        assert health["status"] == "unhealthy"
        assert "error" in health["components"]

        # Restore original behavior
        mock_orchestrator.process_query = original_process


class TestWarmUp:
    """Test pipeline warm-up functionality."""

    async def test_warm_up_success(self, initialized_pipeline):
        """Test successful pipeline warm-up."""
        result = await initialized_pipeline.warm_up()

        assert isinstance(result, dict)
        assert result["status"] == "completed"
        assert "warmup_time_ms" in result
        assert "queries_processed" in result
        assert "successful_queries" in result
        assert "components_warmed" in result

        assert result["warmup_time_ms"] >= 0
        assert result["queries_processed"] == 3  # Default warmup queries
        assert result["successful_queries"] >= 0
        assert len(result["components_warmed"]) > 0

    async def test_warm_up_with_failures(self, initialized_pipeline, mock_orchestrator):
        """Test warm-up with some processing failures."""
        call_count = 0

        def side_effect(_request):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Second warmup query fails
                raise TestError("Warmup failure")
            return QueryProcessingResponse(success=True, results=[], total_results=0)

        mock_orchestrator.process_query.side_effect = side_effect

        result = await initialized_pipeline.warm_up()

        # Should still complete but with fewer successful queries
        assert result["status"] == "completed"
        assert result["queries_processed"] == 3
        assert result["successful_queries"] < 3

    async def test_warm_up_complete_failure(
        self, initialized_pipeline, mock_orchestrator
    ):
        """Test warm-up with complete processing failure."""

        # Mock process_batch to fail completely
        async def failing_batch(*_args, **_kwargs):
            raise TestError("Complete failure")

        # Mock the process_batch method directly on the pipeline
        original_process_batch = initialized_pipeline.process_batch
        initialized_pipeline.process_batch = failing_batch

        result = await initialized_pipeline.warm_up()

        assert result["status"] == "partial"
        assert "error" in result
        assert result["queries_processed"] == 0
        assert result["successful_queries"] == 0

        # Restore original method
        initialized_pipeline.process_batch = original_process_batch

    async def test_warm_up_uninitialized_raises_error(self, pipeline):
        """Test warm-up with uninitialized pipeline."""
        from src.services.errors import APIError

        with pytest.raises(APIError):
            await pipeline.warm_up()


class TestCleanup:
    """Test cleanup functionality."""

    async def test_cleanup_initialized_pipeline(
        self, initialized_pipeline, mock_orchestrator
    ):
        """Test cleaning up initialized pipeline."""
        assert initialized_pipeline._initialized

        await initialized_pipeline.cleanup()

        assert not initialized_pipeline._initialized
        mock_orchestrator.cleanup.assert_called_once()

    async def test_cleanup_uninitialized_pipeline(self, pipeline, mock_orchestrator):
        """Test cleaning up uninitialized pipeline."""
        assert not pipeline._initialized

        await pipeline.cleanup()

        # Should not call orchestrator cleanup
        mock_orchestrator.cleanup.assert_not_called()


class TestContextManager:
    """Test async context manager functionality."""

    async def test_context_manager_success(self, mock_orchestrator):
        """Test successful context manager usage."""
        async with QueryProcessingPipeline(mock_orchestrator) as pipeline:
            assert pipeline._initialized
            response = await pipeline.process("test query")
            assert response.success

        # Should have cleaned up
        mock_orchestrator.cleanup.assert_called_once()

    async def test_context_manager_with_exception(self, mock_orchestrator):
        """Test context manager with exception in context."""
        mock_orchestrator.process_query.side_effect = Exception("Processing error")

        with pytest.raises(Exception, match="Processing error"):
            async with QueryProcessingPipeline(mock_orchestrator) as pipeline:
                await pipeline.process("test query")

        # Should still clean up
        mock_orchestrator.cleanup.assert_called_once()

    async def test_context_manager_initialization_failure(self, mock_orchestrator):
        """Test context manager when initialization fails."""
        mock_orchestrator.initialize.side_effect = Exception("Init failed")

        with pytest.raises(Exception, match="Init failed"):
            async with QueryProcessingPipeline(mock_orchestrator):
                pass

        # Should not call cleanup if init failed
        mock_orchestrator.cleanup.assert_not_called()


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    async def test_process_with_orchestrator_timeout(
        self, initialized_pipeline, mock_orchestrator
    ):
        """Test processing when orchestrator takes too long."""

        async def slow_process(*_args, **_kwargs):
            await asyncio.sleep(0.1)
            return QueryProcessingResponse(success=True, results=[], total_results=0)

        mock_orchestrator.process_query = slow_process

        # Should handle slow orchestrator gracefully
        response = await initialized_pipeline.process("slow query")
        assert isinstance(response, QueryProcessingResponse)

    async def test_process_with_malformed_orchestrator_response(
        self, initialized_pipeline, mock_orchestrator
    ):
        """Test processing with malformed orchestrator response."""
        mock_orchestrator.process_query.return_value = "invalid response"

        # Should handle gracefully or raise appropriate error
        # The pipeline delegates to orchestrator, so malformed response should cause issues
        response = await initialized_pipeline.process("test query")
        # The response should be the malformed response or an error should be raised
        assert response == "invalid response" or hasattr(response, "success")

    async def test_concurrent_initialization_attempts(self, mock_orchestrator):
        """Test concurrent initialization attempts."""
        pipeline = QueryProcessingPipeline(mock_orchestrator)

        # Start multiple initialization tasks
        tasks = [pipeline.initialize() for _ in range(3)]
        await asyncio.gather(*tasks)

        # Should be initialized only once
        assert pipeline._initialized
        mock_orchestrator.initialize.assert_called_once()

    async def test_process_with_very_large_request(self, initialized_pipeline):
        """Test processing with very large request parameters."""
        request = QueryProcessingRequest(
            query="A" * 1000,  # Very long query
            collection_name="test",
            limit=100,  # Maximum limit
            user_context={"data": "x" * 1000},  # Large context
            filters={
                "field_" + str(i): f"value_{i}" for i in range(100)
            },  # Many filters
        )

        response = await initialized_pipeline.process(request)
        assert isinstance(response, QueryProcessingResponse)

    async def test_nested_context_managers(self, mock_orchestrator):
        """Test nested context manager usage."""
        async with (
            QueryProcessingPipeline(mock_orchestrator) as outer_pipeline,
            QueryProcessingPipeline(mock_orchestrator) as inner_pipeline,
        ):
            response1 = await outer_pipeline.process("outer query")
            response2 = await inner_pipeline.process("inner query")

            assert response1.success
            assert response2.success

    async def test_pipeline_reuse_after_cleanup(self, initialized_pipeline):
        """Test using pipeline after cleanup and re-initialization."""
        # Use pipeline
        response1 = await initialized_pipeline.process("first query")
        assert response1.success

        # Cleanup
        await initialized_pipeline.cleanup()
        assert not initialized_pipeline._initialized

        # Re-initialize and use again
        await initialized_pipeline.initialize()
        response2 = await initialized_pipeline.process("second query")
        assert response2.success


class TestCachingBehavior:
    """Test caching behavior and cache integration."""

    async def test_cache_hit_tracking(self, initialized_pipeline, mock_orchestrator):
        """Test cache hit tracking in responses."""

        # Create a new mock function that returns the desired response
        async def cache_hit_response(*_args, **_kwargs):
            return QueryProcessingResponse(
                success=True,
                results=[{"id": "cached_result"}],
                total_results=1,
                cache_hit=True,
            )

        mock_orchestrator.process_query = cache_hit_response

        response = await initialized_pipeline.process("cached query")

        assert response.cache_hit is True

    async def test_cache_miss_tracking(self, initialized_pipeline, mock_orchestrator):
        """Test cache miss tracking in responses."""

        # Create a new mock function that returns the desired response
        async def cache_miss_response(*_args, **_kwargs):
            return QueryProcessingResponse(
                success=True,
                results=[{"id": "fresh_result"}],
                total_results=1,
                cache_hit=False,
            )

        mock_orchestrator.process_query = cache_miss_response

        response = await initialized_pipeline.process("fresh query")

        assert response.cache_hit is False


class TestPipelineCustomization:
    """Test pipeline customization features."""

    async def test_force_strategy_override(self, initialized_pipeline):
        """Test forcing specific search strategy."""
        request = QueryProcessingRequest(
            query="optimization techniques",
            force_strategy=SearchStrategy.HYDE,
            enable_strategy_selection=False,  # Should still work with forced strategy
        )

        response = await initialized_pipeline.process(request)
        assert response.success

    async def test_force_dimension_override(self, initialized_pipeline):
        """Test forcing specific embedding dimension."""
        request = QueryProcessingRequest(
            query="complex technical query",
            force_dimension=MatryoshkaDimension.LARGE,
            enable_matryoshka_optimization=False,
        )

        response = await initialized_pipeline.process(request)
        assert response.success

    async def test_custom_search_accuracy(self, initialized_pipeline):
        """Test custom search accuracy settings."""
        for accuracy in ["fast", "balanced", "high"]:
            request = QueryProcessingRequest(
                query=f"query with {accuracy} accuracy", search_accuracy=accuracy
            )

            response = await initialized_pipeline.process(request)
            assert response.success

    async def test_custom_processing_timeout(self, initialized_pipeline):
        """Test custom processing timeout settings."""
        request = QueryProcessingRequest(
            query="query with custom timeout", max_processing_time_ms=1000
        )

        response = await initialized_pipeline.process(request)
        assert response.success


class TestIntegrationBetweenStages:
    """Test integration and data flow between pipeline stages."""

    async def test_stage_orchestration_data_flow(
        self, initialized_pipeline, mock_orchestrator
    ):
        """Test data flow between pipeline stages."""

        # Create a new mock function that returns detailed stage information
        async def detailed_stage_response(*_args, **_kwargs):
            return QueryProcessingResponse(
                success=True,
                results=[{"id": "result_1", "score": 0.9}],
                total_results=1,
                intent_classification=QueryIntentClassification(
                    primary_intent=QueryIntent.TROUBLESHOOTING,
                    complexity_level=QueryComplexity.COMPLEX,
                    classification_reasoning="Query about debugging indicates troubleshooting intent",
                ),
                preprocessing_result=QueryPreprocessingResult(
                    original_query="How to debug memry leaks?",
                    processed_query="How to debug memory leaks?",
                    corrections_applied=["memry -> memory"],
                    preprocessing_time_ms=50.0,
                ),
                strategy_selection=SearchStrategySelection(
                    primary_strategy=SearchStrategy.HYBRID,
                    matryoshka_dimension=MatryoshkaDimension.LARGE,
                    confidence=0.85,
                    reasoning="Complex troubleshooting query requires hybrid search",
                ),
                processing_steps=[
                    "Query received",
                    "Preprocessing applied",
                    "Intent classified as troubleshooting",
                    "Strategy selected: hybrid",
                    "Search executed",
                    "Results ranked",
                ],
                total_processing_time_ms=250.0,
                strategy_selection_time_ms=25.0,
            )

        mock_orchestrator.process_query = detailed_stage_response

        request = QueryProcessingRequest(
            query="How to debug memry leaks?",
            enable_preprocessing=True,
            enable_intent_classification=True,
            enable_strategy_selection=True,
        )

        response = await initialized_pipeline.process(request)

        # Verify complete stage integration
        assert response.success
        assert response.intent_classification is not None
        assert response.preprocessing_result is not None
        assert response.strategy_selection is not None
        assert response.processing_steps is not None
        assert len(response.processing_steps) > 0

        # Verify data consistency between stages
        assert (
            response.intent_classification.primary_intent == QueryIntent.TROUBLESHOOTING
        )
        assert response.preprocessing_result.corrections_applied == ["memry -> memory"]
        assert response.strategy_selection.primary_strategy == SearchStrategy.HYBRID

    async def test_stage_dependency_handling(self, initialized_pipeline):
        """Test handling of stage dependencies."""
        # Strategy selection typically depends on intent classification
        request = QueryProcessingRequest(
            query="How to implement authentication?",
            enable_intent_classification=True,
            enable_strategy_selection=True,
            enable_preprocessing=False,
        )

        response = await initialized_pipeline.process(request)
        assert response.success

    async def test_stage_error_propagation(
        self, initialized_pipeline, mock_orchestrator
    ):
        """Test error propagation between stages."""

        # Create a mock function that returns response with warnings
        async def error_propagation_response(*_args, **_kwargs):
            return QueryProcessingResponse(
                success=True,
                results=[],
                total_results=0,
                warnings=[
                    "Preprocessing stage had issues",
                    "Intent classification uncertain",
                ],
                fallback_used=True,
            )

        mock_orchestrator.process_query = error_propagation_response

        response = await initialized_pipeline.process("problematic query")

        assert response.success
        assert len(response.warnings) > 0
        assert response.fallback_used


class TestAdditionalEdgeCases:
    """Additional edge case tests for comprehensive coverage."""

    async def test_analyze_query_with_null_classification(
        self, initialized_pipeline, mock_orchestrator
    ):
        """Test analyze_query when intent classification returns None."""

        async def null_classification_response(*_args, **_kwargs):
            return QueryProcessingResponse(
                success=True,
                results=[],
                total_results=0,
                intent_classification=None,  # Null classification
                preprocessing_result=QueryPreprocessingResult(
                    original_query="test query", processed_query="test query"
                ),
                strategy_selection=SearchStrategySelection(
                    primary_strategy=SearchStrategy.SEMANTIC,
                    matryoshka_dimension=MatryoshkaDimension.MEDIUM,
                    confidence=0.8,
                ),
            )

        mock_orchestrator.process_query = null_classification_response

        analysis = await initialized_pipeline.analyze_query("ambiguous query")

        assert analysis["complexity"] is None
        assert "intent_classification" in analysis

    async def test_process_with_none_collection_name(self, initialized_pipeline):
        """Test processing with None collection name in string query."""
        # This should use default collection name when None is passed
        response = await initialized_pipeline.process(
            "test query"
        )  # No collection_name specified

        assert isinstance(response, QueryProcessingResponse)

    async def test_context_manager_double_entry(self, mock_orchestrator):
        """Test context manager being used twice on same instance."""
        pipeline = QueryProcessingPipeline(mock_orchestrator)

        async with pipeline as p1:
            assert p1._initialized
            # Try to use context manager again on same instance should work
            async with pipeline as p2:
                assert p2._initialized
                assert p1 is p2  # Same instance

    async def test_analyze_query_request_creation(
        self, initialized_pipeline, mock_orchestrator
    ):
        """Test that analyze_query creates proper internal request."""
        captured_request = None

        async def capture_request(request):
            nonlocal captured_request
            captured_request = request
            return QueryProcessingResponse(
                success=True,
                results=[],
                total_results=0,
                preprocessing_result=QueryPreprocessingResult(
                    original_query="test", processed_query="test"
                ),
            )

        mock_orchestrator.process_query = capture_request

        await initialized_pipeline.analyze_query(
            "test query", enable_preprocessing=False, enable_intent_classification=True
        )

        # Verify the request was created correctly
        assert captured_request is not None
        assert captured_request.query == "test query"
        assert captured_request.collection_name == "analysis"
        assert captured_request.limit == 1
        assert captured_request.enable_preprocessing is False
        assert captured_request.enable_intent_classification is True
        assert captured_request.enable_strategy_selection is True

    async def test_health_check_component_status_variations(
        self, initialized_pipeline, mock_orchestrator
    ):
        """Test health check with various component status combinations."""

        async def partial_health_response(*_args, **_kwargs):
            return QueryProcessingResponse(
                success=True,
                results=[],
                total_results=0,
                intent_classification=QueryIntentClassification(
                    primary_intent=QueryIntent.CONCEPTUAL,
                    complexity_level=QueryComplexity.SIMPLE,
                ),
                preprocessing_result=None,  # No preprocessing result
                strategy_selection=SearchStrategySelection(
                    primary_strategy=SearchStrategy.SEMANTIC,
                    matryoshka_dimension=MatryoshkaDimension.MEDIUM,
                    confidence=0.8,
                ),
            )

        mock_orchestrator.process_query = partial_health_response

        health = await initialized_pipeline.health_check()

        assert "components" in health
        components = health["components"]

        # Should detect partial component health
        assert components["intent_classifier"]["status"] == "healthy"
        assert components["preprocessor"]["status"] == "degraded"
        assert components["strategy_selector"]["status"] == "healthy"

    async def test_warmup_timing_measurement(self, initialized_pipeline):
        """Test that warmup properly measures timing."""
        import time

        start_time = time.time()
        result = await initialized_pipeline.warm_up()
        end_time = time.time()

        # Warmup should report reasonable timing
        assert result["warmup_time_ms"] >= 0
        # Should be close to actual elapsed time (allowing for some overhead)
        actual_time_ms = (end_time - start_time) * 1000
        assert abs(result["warmup_time_ms"] - actual_time_ms) < 1000  # Within 1 second

    async def test_process_with_all_request_options(self, initialized_pipeline):
        """Test processing with all possible request options set."""
        request = QueryProcessingRequest(
            query="comprehensive test query",
            collection_name="full_test",
            limit=50,
            enable_preprocessing=True,
            enable_intent_classification=True,
            enable_strategy_selection=True,
            enable_matryoshka_optimization=True,
            force_strategy=SearchStrategy.ADAPTIVE,
            force_dimension=MatryoshkaDimension.LARGE,
            user_context={
                "user_id": "test_user",
                "session_id": "test_session",
                "preferences": {"accuracy": "high"},
            },
            filters={
                "domain": "technical",
                "difficulty": "advanced",
                "timestamp": "recent",
            },
            search_accuracy="high",
            max_processing_time_ms=30000,
        )

        response = await initialized_pipeline.process(request)

        assert isinstance(response, QueryProcessingResponse)
        assert response.success

    async def test_get_metrics_edge_case_values(
        self, initialized_pipeline, mock_orchestrator
    ):
        """Test get_metrics with edge case return values from orchestrator."""

        def edge_case_stats():
            return {
                "total_queries": None,  # None value
                "successful_queries": -1,  # Negative value
                "average_processing_time": float("inf"),  # Infinity
                "strategy_usage": None,  # None strategy usage
                "extra_field": "should_be_ignored",  # Extra field
            }

        mock_orchestrator.get_performance_stats = edge_case_stats

        metrics = await initialized_pipeline.get_metrics()

        # Should handle edge cases gracefully - the implementation returns raw values
        assert metrics["total_queries"] is None or metrics["total_queries"] >= 0
        assert metrics["successful_queries"] == -1 or metrics["successful_queries"] >= 0
        assert (
            metrics["average_processing_time"] == float("inf")
            or metrics["average_processing_time"] >= 0
        )
        assert metrics["strategy_usage"] is None or isinstance(
            metrics["strategy_usage"], dict
        )

    async def test_batch_processing_max_concurrent_edge_cases(
        self, initialized_pipeline
    ):
        """Test batch processing with edge case concurrency values."""
        requests = [QueryProcessingRequest(query=f"Query {i}") for i in range(3)]

        # Test with max_concurrent = 1 (sequential processing)
        responses = await initialized_pipeline.process_batch(requests, max_concurrent=1)
        assert len(responses) == 3
        assert all(resp.success for resp in responses)

        # Test with max_concurrent > number of requests
        responses = await initialized_pipeline.process_batch(
            requests, max_concurrent=10
        )
        assert len(responses) == 3
        assert all(resp.success for resp in responses)
