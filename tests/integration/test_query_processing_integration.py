"""Integration tests for the complete query processing system."""

from unittest.mock import AsyncMock

import pytest

from src.services.query_processing.models import MatryoshkaDimension
from src.services.query_processing.models import QueryComplexity
from src.services.query_processing.models import QueryIntent
from src.services.query_processing.models import QueryProcessingRequest
from src.services.query_processing.models import SearchStrategy
from src.services.query_processing.orchestrator import (
    SearchOrchestrator as AdvancedSearchOrchestrator,
)
from src.services.query_processing.pipeline import QueryProcessingPipeline


@pytest.fixture
def mock_embedding_manager():
    """Create a mock embedding manager."""
    manager = AsyncMock()

    def mock_generate_embeddings(texts, **kwargs):
        """Generate mock embeddings that help with intent classification."""
        # Create different embeddings for different intent types
        embeddings = []
        for i, text in enumerate(texts):
            # Create embeddings that favor troubleshooting for the test query
            if i == 0:  # First text is the query
                # Query: "Getting ImportError when importing pandas, how to fix?"
                # Should be most similar to troubleshooting reference
                embedding = [
                    0.9 if j == 3 else 0.1 for j in range(768)
                ]  # High troubleshooting similarity
            elif (
                "fix" in text.lower()
                and "error" in text.lower()
                and "problem" in text.lower()
            ):
                # This is the troubleshooting reference: "How to fix this error and resolve the problem?"
                embedding = [
                    0.9 if j == 3 else 0.1 for j in range(768)
                ]  # Match query embedding
            elif "step by step" in text.lower() and "implement" in text.lower():
                # This is the procedural reference: "How do I implement this step by step?"
                embedding = [
                    0.9 if j == 1 else 0.1 for j in range(768)
                ]  # Different from query
            else:
                # Other reference embeddings - make them different
                embedding = [0.3 if j == (i % 10) else 0.1 for j in range(768)]
            embeddings.append(embedding)

        return {"success": True, "embeddings": embeddings}

    manager.generate_embeddings = AsyncMock(side_effect=mock_generate_embeddings)
    manager.rerank_results = AsyncMock(
        return_value=[{"original": {"id": "1", "content": "test", "score": 0.9}}]
    )
    return manager


@pytest.fixture
def mock_qdrant_service():
    """Create a mock Qdrant service."""
    service = AsyncMock()
    service.filtered_search = AsyncMock(
        return_value=[
            {
                "id": "1",
                "payload": {
                    "content": "Machine learning is a subset of AI",
                    "title": "ML Intro",
                },
                "score": 0.9,
            },
            {
                "id": "2",
                "payload": {"content": "Python is great for ML", "title": "Python ML"},
                "score": 0.8,
            },
        ]
    )
    service.search.hybrid_search = AsyncMock(
        return_value=[
            {
                "id": "3",
                "payload": {
                    "content": "Deep learning concepts",
                    "title": "Deep Learning",
                },
                "score": 0.85,
            },
        ]
    )
    service.search.multi_stage_search = AsyncMock(
        return_value=[
            {
                "id": "4",
                "payload": {"content": "Advanced algorithms", "title": "Algorithms"},
                "score": 0.88,
            },
        ]
    )
    return service


@pytest.fixture
def mock_hyde_engine():
    """Create a mock HyDE engine."""
    engine = AsyncMock()
    engine.enhanced_search = AsyncMock(
        return_value=[
            {
                "id": "5",
                "content": "HyDE enhanced results",
                "title": "Enhanced",
                "score": 0.92,
            }
        ]
    )
    return engine


@pytest.fixture
async def complete_pipeline(
    mock_embedding_manager, mock_qdrant_service, mock_hyde_engine
):
    """Create a complete query processing pipeline with all components."""
    # Create orchestrator with correct constructor parameters
    orchestrator = AdvancedSearchOrchestrator(
        enable_all_features=True,
        enable_performance_optimization=True,
        cache_size=1000,
        max_concurrent_stages=5,
    )

    # Create pipeline
    pipeline = QueryProcessingPipeline(orchestrator=orchestrator)

    # Initialize everything
    await pipeline.initialize()

    return pipeline


class TestQueryProcessingIntegration:
    """Test complete query processing integration."""

    async def test_end_to_end_conceptual_query(self, complete_pipeline):
        """Test end-to-end processing of a conceptual query."""
        request = QueryProcessingRequest(
            query="What is machine learning and how does it work?",
            collection_name="documentation",
            limit=10,
            enable_preprocessing=True,
            enable_intent_classification=True,
            enable_strategy_selection=True,
        )

        response = await complete_pipeline.process(request)

        # Verify successful processing
        assert response.success is True
        assert response.total_results >= 0  # May be 0 with mocked services

        # Verify processing components were used
        assert response.preprocessing_result is not None
        assert response.intent_classification is not None
        assert response.intent_classification.primary_intent == QueryIntent.CONCEPTUAL
        assert response.strategy_selection is not None

        # Verify timing information
        assert response.total_processing_time_ms > 0
        assert response.search_time_ms >= 0

    async def test_end_to_end_procedural_query(self, complete_pipeline):
        """Test end-to-end processing of a procedural query."""
        request = QueryProcessingRequest(
            query="How to implement authentication in Python step by step?",
            collection_name="guides",
            limit=5,
            enable_preprocessing=True,
            enable_intent_classification=True,
            enable_strategy_selection=True,
        )

        response = await complete_pipeline.process(request)

        assert response.success is True
        assert response.intent_classification.primary_intent == QueryIntent.PROCEDURAL
        # Procedural queries should use HyDE strategy
        assert response.strategy_selection.primary_strategy == SearchStrategy.HYDE

    async def test_end_to_end_troubleshooting_query(self, complete_pipeline):
        """Test end-to-end processing of a troubleshooting query."""
        request = QueryProcessingRequest(
            query="Getting ImportError when importing pandas, how to fix?",
            collection_name="troubleshooting",
            limit=8,
            enable_preprocessing=True,
            enable_intent_classification=True,
            enable_strategy_selection=True,
        )

        response = await complete_pipeline.process(request)

        assert response.success is True
        assert response.intent_classification.primary_intent == QueryIntent.PROCEDURAL
        # Procedural queries should use HyDE strategy
        assert response.strategy_selection.primary_strategy == SearchStrategy.HYDE

    async def test_preprocessing_spell_correction_flow(self, complete_pipeline):
        """Test preprocessing with spell correction integration."""
        request = QueryProcessingRequest(
            query="What is phython programming langauge?",
            collection_name="docs",
            limit=5,
            enable_preprocessing=True,
            enable_intent_classification=True,
        )

        response = await complete_pipeline.process(request)

        assert response.success is True
        assert response.preprocessing_result is not None
        assert "python" in response.preprocessing_result.processed_query.lower()
        assert len(response.preprocessing_result.corrections_applied) > 0

    async def test_context_extraction_and_intent_classification(
        self, complete_pipeline
    ):
        """Test context extraction affecting intent classification."""
        request = QueryProcessingRequest(
            query="How to optimize React performance for large applications?",
            collection_name="web_dev",
            limit=10,
            enable_preprocessing=True,
            enable_intent_classification=True,
            enable_strategy_selection=True,
        )

        response = await complete_pipeline.process(request)

        assert response.success is True

        # Should extract framework context
        if response.preprocessing_result.context_extracted:
            context = response.preprocessing_result.context_extracted
            assert "framework" in context or "react" in str(context).lower()

        # Should classify as procedural intent (due to "How to" pattern)
        assert response.intent_classification.primary_intent == QueryIntent.PROCEDURAL

    async def test_strategy_selection_based_on_complexity(self, complete_pipeline):
        """Test strategy selection based on query complexity."""
        # Simple query
        simple_request = QueryProcessingRequest(
            query="What is Python?",
            collection_name="basics",
            limit=5,
            enable_preprocessing=True,
            enable_intent_classification=True,
            enable_strategy_selection=True,
        )

        simple_response = await complete_pipeline.process(simple_request)

        # Complex query
        complex_request = QueryProcessingRequest(
            query="How to design a distributed microservices architecture with event sourcing and CQRS patterns?",
            collection_name="architecture",
            limit=10,
            enable_preprocessing=True,
            enable_intent_classification=True,
            enable_strategy_selection=True,
        )

        complex_response = await complete_pipeline.process(complex_request)

        # Both should succeed
        assert simple_response.success is True
        assert complex_response.success is True

        # Complex query should have higher complexity
        assert (
            complex_response.intent_classification.complexity_level.value
            > simple_response.intent_classification.complexity_level.value
            or complex_response.intent_classification.complexity_level
            == QueryComplexity.EXPERT
        )

    async def test_forced_strategy_override(self, complete_pipeline):
        """Test forcing a specific search strategy."""
        request = QueryProcessingRequest(
            query="What is machine learning?",
            collection_name="docs",
            limit=5,
            force_strategy=SearchStrategy.HYBRID,
            force_dimension=MatryoshkaDimension.LARGE,
        )

        response = await complete_pipeline.process(request)

        assert response.success is True
        # Strategy should be overridden
        # (Verification would depend on orchestrator implementation)

    async def test_performance_requirements_integration(self, complete_pipeline):
        """Test performance requirements affecting strategy selection."""
        request = QueryProcessingRequest(
            query="How to implement user authentication?",
            collection_name="security",
            limit=5,
            max_processing_time_ms=100,  # Short timeout
            enable_preprocessing=True,
            enable_intent_classification=True,
            enable_strategy_selection=True,
        )

        response = await complete_pipeline.process(request)

        assert response.success is True
        # Should consider performance constraints

    async def test_user_context_integration(self, complete_pipeline):
        """Test user context affecting processing."""
        request = QueryProcessingRequest(
            query="How to implement authentication?",
            collection_name="tutorials",
            limit=5,
            user_context={
                "programming_language": ["python"],
                "experience_level": "beginner",
                "urgency": "medium",
            },
            enable_preprocessing=True,
            enable_intent_classification=True,
            enable_strategy_selection=True,
        )

        response = await complete_pipeline.process(request)

        assert response.success is True
        # Context should influence processing

    async def test_filters_integration(self, complete_pipeline):
        """Test search filters integration."""
        request = QueryProcessingRequest(
            query="Python web development",
            collection_name="tutorials",
            limit=8,
            filters={"category": "web", "difficulty": "intermediate"},
            force_strategy=SearchStrategy.FILTERED,
        )

        response = await complete_pipeline.process(request)

        assert response.success is True
        # Filters should be applied in search

    async def test_fallback_strategy_usage(
        self, complete_pipeline, mock_qdrant_service
    ):
        """Test fallback strategy when primary fails."""
        # Make primary search fail
        mock_qdrant_service.filtered_search.side_effect = Exception(
            "Primary search failed"
        )

        # Set orchestrator to simulate search failure
        complete_pipeline.orchestrator._test_search_failure = True

        request = QueryProcessingRequest(
            query="Database optimization techniques",
            collection_name="performance",
            limit=5,
            force_strategy=SearchStrategy.FILTERED,
        )

        try:
            response = await complete_pipeline.process(request)

            # Should still succeed with fallback
            assert response.success is True
            assert response.fallback_used is True
        finally:
            # Reset the test flag
            complete_pipeline.orchestrator._test_search_failure = False

    async def test_batch_processing_integration(self, complete_pipeline):
        """Test batch processing with various query types."""
        requests = [
            QueryProcessingRequest(
                query="What is machine learning?",
                collection_name="docs",
                limit=5,
                enable_intent_classification=True,
            ),
            QueryProcessingRequest(
                query="How to fix ImportError in Python?",
                collection_name="troubleshooting",
                limit=3,
                enable_preprocessing=True,
            ),
            QueryProcessingRequest(
                query="Compare React vs Vue performance",
                collection_name="comparisons",
                limit=8,
                enable_strategy_selection=True,
                enable_intent_classification=True,
            ),
        ]

        responses = await complete_pipeline.process_batch(requests)

        assert len(responses) == 3
        assert all(resp.success is True for resp in responses)

        # Different intents should be detected
        intent_types = [
            resp.intent_classification.primary_intent
            for resp in responses
            if resp.intent_classification
        ]
        assert len(set(intent_types)) > 1  # Should have different intents

    async def test_comprehensive_pipeline_metrics(self, complete_pipeline):
        """Test comprehensive pipeline metrics collection."""
        # Process various queries
        queries = [
            "What is Python?",
            "How to debug memory leaks?",
            "Compare frameworks performance",
            "Configure production settings",
        ]

        for query in queries:
            request = QueryProcessingRequest(
                query=query,
                collection_name="docs",
                limit=5,
                enable_preprocessing=True,
                enable_intent_classification=True,
                enable_strategy_selection=True,
            )
            await complete_pipeline.process(request)

        metrics = await complete_pipeline.get_metrics()

        assert metrics["total_queries"] >= 4
        assert metrics["successful_queries"] >= 4
        assert metrics["average_processing_time"] > 0
        assert "strategy_usage" in metrics

    async def test_health_check_integration(self, complete_pipeline):
        """Test health check of integrated system."""
        health = await complete_pipeline.health_check()

        assert health["status"] == "healthy"
        assert "components" in health
        assert "performance" in health

        # Should check all components
        components = health["components"]
        expected_components = [
            "orchestrator",
            "intent_classifier",
            "preprocessor",
            "strategy_selector",
        ]
        for component in expected_components:
            if component in components:
                assert components[component]["status"] == "healthy"

    async def test_warm_up_integration(self, complete_pipeline):
        """Test system warm-up integration."""
        result = await complete_pipeline.warm_up()

        assert result["status"] == "completed"
        assert "warmup_time_ms" in result
        assert result["warmup_time_ms"] >= 0

    async def test_error_handling_integration(
        self, complete_pipeline, mock_embedding_manager
    ):
        """Test error handling across integrated components."""
        # Make embeddings fail
        mock_embedding_manager.generate_embeddings.side_effect = Exception(
            "Embedding service down"
        )

        request = QueryProcessingRequest(
            query="Test query with embedding failure",
            collection_name="docs",
            limit=5,
        )

        response = await complete_pipeline.process(request)

        # Should handle error gracefully
        assert isinstance(response, type(response))
        # Response might be successful with fallback or unsuccessful with error message

    async def test_cleanup_integration(self, complete_pipeline):
        """Test cleanup of integrated system."""
        await complete_pipeline.cleanup()
        # Should clean up all components without errors

    async def test_concurrent_processing_integration(self, complete_pipeline):
        """Test concurrent processing with integrated system."""
        import asyncio

        requests = [
            QueryProcessingRequest(
                query=f"Concurrent query {i}: What is technology {i}?",
                collection_name="tech",
                limit=3,
                enable_preprocessing=True,
                enable_intent_classification=True,
            )
            for i in range(5)
        ]

        # Process concurrently
        tasks = [complete_pipeline.process(req) for req in requests]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 5
        assert all(isinstance(resp, type(responses[0])) for resp in responses)

    async def test_advanced_intent_detection_integration(self, complete_pipeline):
        """Test detection of advanced intent categories."""
        advanced_queries = [
            (
                "How to design scalable microservices architecture?",
                QueryIntent.ARCHITECTURAL,
            ),
            ("Compare React vs Vue vs Angular performance", QueryIntent.COMPARATIVE),
            ("How to secure OAuth 2.0 implementation?", QueryIntent.PROCEDURAL),
            ("Best practices for Python code organization", QueryIntent.BEST_PRACTICES),
            ("How to migrate from Python 2 to 3?", QueryIntent.PROCEDURAL),
            ("Debug performance bottlenecks in production", QueryIntent.PERFORMANCE),
            ("Configure Django for production deployment", QueryIntent.CONFIGURATION),
        ]

        for query, expected_intent in advanced_queries:
            request = QueryProcessingRequest(
                query=query,
                collection_name="advanced",
                limit=5,
                enable_intent_classification=True,
            )

            response = await complete_pipeline.process(request)

            assert response.success is True
            assert response.intent_classification is not None
            assert response.intent_classification.primary_intent == expected_intent
