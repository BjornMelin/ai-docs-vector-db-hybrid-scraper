"""Tests for query processing pipeline."""

from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest
from src.services.query_processing.models import MatryoshkaDimension
from src.services.query_processing.models import QueryComplexity
from src.services.query_processing.models import QueryIntent
from src.services.query_processing.models import QueryIntentClassification
from src.services.query_processing.models import QueryPreprocessingResult
from src.services.query_processing.models import QueryProcessingRequest
from src.services.query_processing.models import QueryProcessingResponse
from src.services.query_processing.models import SearchStrategy
from src.services.query_processing.models import SearchStrategySelection
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

    def increment_calls(*args, **kwargs):
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
            if "Query 1" in request.query:
                raise Exception("Processing failed")
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
        def side_effect(*args, **kwargs):
            raise Exception("Temporary failure")

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

    async def test_configuration_validation(self, mock_orchestrator):
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
        async def slow_process(*args, **kwargs):
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
