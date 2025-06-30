"""Integration tests for enhanced query processing orchestrator with portfolio features."""

from unittest.mock import AsyncMock

import pytest


try:
    import numpy as np
except ImportError:
    np = None

import random

from src.services.analytics.search_dashboard import SearchAnalyticsDashboard
from src.services.analytics.vector_visualization import VectorVisualizationEngine
from src.services.query_processing.orchestrator import (
    SearchMode,
    SearchOrchestrator,
    SearchPipeline,
    SearchRequest,
)


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for orchestrator."""
    deps = {
        "embedding_manager": AsyncMock(),
        "qdrant_service": AsyncMock(),
        "hyde_engine": AsyncMock(),
        "query_expansion_service": AsyncMock(),
        "clustering_service": AsyncMock(),
        "ranking_service": AsyncMock(),
        "federated_service": AsyncMock(),
        "rag_generator": AsyncMock(),
    }

    # Configure mock responses
    deps["embedding_manager"].generate_embeddings.return_value = {
        "success": True,
        "embeddings": [[0.1] * 768],
    }

    deps["qdrant_service"].search.hybrid_search.return_value = [
        {
            "id": "1",
            "payload": {
                "content": "Machine learning is a subset of AI",
                "title": "ML Basics",
            },
            "score": 0.95,
        },
        {
            "id": "2",
            "payload": {
                "content": "Deep learning uses neural networks",
                "title": "Deep Learning",
            },
            "score": 0.87,
        },
    ]

    deps["rag_generator"].get_metrics.return_value = {
        "total_answers": 42,
        "avg_confidence": 0.85,
        "avg_generation_time": 1.2,
        "success_rate": 0.94,
    }

    return deps


@pytest.fixture
async def orchestrator_with_portfolio_features(mock_dependencies):
    """Create orchestrator with enhanced portfolio features."""
    orchestrator = SearchOrchestrator(
        cache_size=100, enable_performance_optimization=True
    )

    # Initialize with mock dependencies
    await orchestrator.initialize()

    # Inject mock dependencies
    for name, mock_dep in mock_dependencies.items():
        setattr(orchestrator, f"_{name}", mock_dep)

    return orchestrator


@pytest.fixture
def search_analytics_dashboard():
    """Create search analytics dashboard."""
    return SearchAnalyticsDashboard()


@pytest.fixture
def vector_visualization_engine():
    """Create vector visualization engine."""
    return VectorVisualizationEngine()


class TestEnhancedOrchestratorIntegration:
    """Test integration of enhanced orchestrator with portfolio features."""

    async def test_enhanced_stats_with_rag_metrics(
        self, orchestrator_with_portfolio_features
    ):
        """Test enhanced statistics including RAG-specific metrics."""
        orchestrator = orchestrator_with_portfolio_features

        # Perform a search to generate stats
        request = SearchRequest(
            query="What is machine learning?",
            collection_name="documentation",
            mode=SearchMode.ENHANCED,
            pipeline=SearchPipeline.BALANCED,
            enable_rag=True,
        )

        await orchestrator.search(request)

        # Get enhanced stats
        stats = orchestrator.get_stats()

        # Verify basic orchestrator stats
        assert "total_searches" in stats
        assert "avg_processing_time" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats

        # Verify RAG-specific metrics
        assert "rag_answers_generated" in stats
        assert "rag_avg_confidence" in stats
        assert "rag_avg_generation_time" in stats
        assert "rag_success_rate" in stats

        # Verify feature utilization metrics
        assert "feature_utilization" in stats
        feature_util = stats["feature_utilization"]
        assert "query_expansion_usage" in feature_util
        assert "clustering_usage" in feature_util
        assert "personalization_usage" in feature_util
        assert "federation_usage" in feature_util
        assert "rag_usage" in feature_util

        # Verify performance trends
        assert "performance_trends" in stats
        trends = stats["performance_trends"]
        assert "avg_latency_trend" in trends
        assert "cache_hit_rate" in trends
        assert "error_rate" in trends

    async def test_search_analytics_integration(
        self, orchestrator_with_portfolio_features, search_analytics_dashboard
    ):
        """Test integration with search analytics dashboard."""
        orchestrator = orchestrator_with_portfolio_features
        dashboard = search_analytics_dashboard

        await dashboard.initialize()

        # Perform searches and track in analytics
        requests = [
            SearchRequest(
                query="machine learning algorithms",
                collection_name="documentation",
                mode=SearchMode.ENHANCED,
                enable_expansion=True,
            ),
            SearchRequest(
                query="deep learning neural networks",
                collection_name="papers",
                mode=SearchMode.FAST,
                enable_clustering=True,
            ),
            SearchRequest(
                query="data science methodology",
                collection_name="documentation",
                mode=SearchMode.COMPREHENSIVE,
                enable_rag=True,
            ),
        ]

        for request in requests:
            # Execute search
            result = await orchestrator.search(request)

            # Track in analytics dashboard
            await dashboard.track_query(
                {
                    "query": request.query,
                    "collection": request.collection_name,
                    "processing_time_ms": result.processing_time_ms,
                    "results_count": result.total_results,
                    "success": True,
                    "features_used": result.features_used,
                    "cache_hit": result.cache_hit,
                }
            )

        # Verify analytics data
        performance_metrics = await dashboard.get_performance_metrics()
        assert performance_metrics["total_queries"] == 3
        assert performance_metrics["success_rate"] == 1.0

        # Test query patterns detection
        patterns = await dashboard.detect_query_patterns()
        assert len(patterns) > 0

        # Test optimization recommendations
        recommendations = await dashboard.generate_optimization_recommendations()
        assert len(recommendations) > 0

    async def test_vector_visualization_integration(
        self, orchestrator_with_portfolio_features, vector_visualization_engine
    ):
        """Test integration with vector visualization."""
        orchestrator = orchestrator_with_portfolio_features
        viz_engine = vector_visualization_engine

        await viz_engine.initialize()

        # Perform search to get results with embeddings
        request = SearchRequest(
            query="machine learning concepts",
            collection_name="documentation",
            mode=SearchMode.ENHANCED,
        )

        result = await orchestrator.search(request)

        # Simulate embeddings for visualization
        if np is not None:
            rng = np.random.default_rng(42)
            # Create sample embeddings and documents based on search results
            embeddings = rng.random((len(result.results), 768))
        else:
            rng = random.Random(42)
            # Create sample embeddings using regular random
            embeddings = [
                [rng.random() for _ in range(768)] for _ in range(len(result.results))
            ]
        documents = [
            {
                "id": doc["id"],
                "text": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
            }
            for doc in result.results
        ]

        # Create visualization
        if len(documents) > 0:
            visualization = await viz_engine.create_visualization(
                embeddings=embeddings, documents=documents, dimension="2d"
            )

            # Verify visualization components
            assert "points" in visualization
            assert "clusters" in visualization
            assert "quality_metrics" in visualization

            points = visualization["points"]
            assert len(points) == len(result.results)

            # Test similarity analysis
            query_embedding = rng.random(768)
            similar_relations = await viz_engine.find_similar_vectors(
                embeddings=embeddings,
                documents=documents,
                query_vector=query_embedding,
                top_k=3,
            )

            assert len(similar_relations) <= 3

    async def test_full_portfolio_workflow(
        self,
        orchestrator_with_portfolio_features,
        search_analytics_dashboard,
        vector_visualization_engine,
    ):
        """Test complete portfolio workflow integration."""
        orchestrator = orchestrator_with_portfolio_features
        dashboard = search_analytics_dashboard
        viz_engine = vector_visualization_engine

        # Initialize all components
        await dashboard.initialize()
        await viz_engine.initialize()

        # Simulate a complete user search workflow
        user_queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "Explain deep learning architectures",
            "Compare supervised vs unsupervised learning",
            "What are transformer models?",
        ]

        all_results = []

        for i, query in enumerate(user_queries):
            # Create search request with varying configurations
            request = SearchRequest(
                query=query,
                collection_name="documentation",
                mode=SearchMode.ENHANCED,
                pipeline=SearchPipeline.BALANCED,
                enable_expansion=i % 2 == 0,  # Alternate expansion
                enable_clustering=i % 3 == 0,  # Every third query
                enable_rag=i % 2 == 1,  # Alternate RAG
                user_id=f"user_{i % 3}",  # 3 different users
                session_id=f"session_{i // 2}",  # 2-3 queries per session
            )

            # Execute search
            result = await orchestrator.search(request)
            all_results.append(result)

            # Track in analytics
            await dashboard.track_query(
                {
                    "query": query,
                    "collection": request.collection_name,
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "processing_time_ms": result.processing_time_ms,
                    "results_count": result.total_results,
                    "success": True,
                    "features_used": result.features_used,
                    "cache_hit": result.cache_hit,
                }
            )

        # Generate comprehensive analytics
        orchestrator_stats = orchestrator.get_stats()
        performance_metrics = await dashboard.get_performance_metrics()
        query_patterns = await dashboard.detect_query_patterns()
        user_insights = await dashboard.get_user_behavior_insights()
        top_queries = await dashboard.get_top_queries()

        # Verify comprehensive analytics
        assert orchestrator_stats["total_searches"] == 5
        assert performance_metrics["total_queries"] == 5
        assert len(query_patterns) > 0
        assert len(user_insights) > 0
        assert len(top_queries) > 0

        # Test visualization of aggregated results
        if np is not None:
            rng = np.random.default_rng(42)
        else:
            rng = random.Random(42)

        # Simulate embeddings for all results
        all_documents = []
        for result in all_results:
            for doc in result.results:
                all_documents.append(
                    {
                        "id": doc["id"],
                        "text": doc.get("content", ""),
                        "metadata": doc.get("metadata", {}),
                    }
                )

        if len(all_documents) > 0:
            if np is not None:
                embeddings = rng.random((len(all_documents), 768))
            else:
                embeddings = [
                    [rng.random() for _ in range(768)]
                    for _ in range(len(all_documents))
                ]

            visualization = await viz_engine.create_visualization(
                embeddings=embeddings, documents=all_documents, dimension="2d"
            )

            quality_metrics = await viz_engine.analyze_embedding_quality(embeddings)

            # Verify portfolio features work together
            assert len(visualization["points"]) == len(all_documents)
            assert quality_metrics.dimensionality == 768

            # Export analytics data
            analytics_export = await dashboard.export_analytics_data()
            viz_export = await viz_engine.export_visualization_data(
                visualization, format="json"
            )

            assert "queries" in analytics_export
            assert "performance_metrics" in analytics_export
            assert "points" in viz_export
            assert "clusters" in viz_export

    async def test_performance_monitoring_integration(
        self, orchestrator_with_portfolio_features, search_analytics_dashboard
    ):
        """Test performance monitoring across components."""
        orchestrator = orchestrator_with_portfolio_features
        dashboard = search_analytics_dashboard

        await dashboard.initialize()

        # Perform searches with different performance characteristics
        slow_request = SearchRequest(
            query="complex analytical query",
            collection_name="documentation",
            mode=SearchMode.COMPREHENSIVE,
            pipeline=SearchPipeline.COMPREHENSIVE,
            enable_expansion=True,
            enable_clustering=True,
            enable_rag=True,
        )

        fast_request = SearchRequest(
            query="simple query",
            collection_name="documentation",
            mode=SearchMode.FAST,
            pipeline=SearchPipeline.FAST,
        )

        # Execute requests
        slow_result = await orchestrator.search(slow_request)
        fast_result = await orchestrator.search(fast_request)

        # Track performance
        await dashboard.track_query(
            {
                "query": slow_request.query,
                "processing_time_ms": slow_result.processing_time_ms,
                "success": True,
                "features_used": slow_result.features_used,
            }
        )

        await dashboard.track_query(
            {
                "query": fast_request.query,
                "processing_time_ms": fast_result.processing_time_ms,
                "success": True,
                "features_used": fast_result.features_used,
            }
        )

        # Analyze performance patterns
        recommendations = await dashboard.generate_optimization_recommendations()
        performance_metrics = await dashboard.get_performance_metrics()

        # Verify performance insights
        assert performance_metrics["total_queries"] == 2
        assert len(recommendations) > 0

        # Should detect performance differences
        timeline = await dashboard.get_search_volume_timeline(hours=1)
        assert len(timeline) > 0

    async def test_error_handling_and_resilience(
        self, orchestrator_with_portfolio_features, search_analytics_dashboard
    ):
        """Test error handling and system resilience."""
        orchestrator = orchestrator_with_portfolio_features
        dashboard = search_analytics_dashboard

        await dashboard.initialize()

        # Test with various error conditions
        error_request = SearchRequest(
            query="",  # Empty query
            collection_name="nonexistent",
            mode=SearchMode.ENHANCED,
        )

        # Should handle gracefully
        try:
            result = await orchestrator.search(error_request)
            # Track even failed requests
            await dashboard.track_query(
                {
                    "query": error_request.query,
                    "processing_time_ms": getattr(result, "processing_time_ms", 0),
                    "success": False,
                    "error": "Invalid query",
                }
            )
        except (TimeoutError, ConnectionError, RuntimeError, ValueError):
            # Should not propagate unhandled exceptions
            pass

        # System should remain functional
        valid_request = SearchRequest(
            query="valid query", collection_name="documentation", mode=SearchMode.FAST
        )

        result = await orchestrator.search(valid_request)
        assert result is not None

        # Analytics should track both success and failure
        metrics = await dashboard.get_performance_metrics()
        assert metrics["total_queries"] > 0
