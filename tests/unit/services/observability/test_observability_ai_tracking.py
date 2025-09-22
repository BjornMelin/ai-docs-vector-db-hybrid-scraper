"""Tests for AI/ML operation tracking module."""

from unittest.mock import Mock, patch

import pytest

from src.services.observability.ai_tracking import (
    AIOperationMetrics,
    AIOperationTracker,
    get_ai_tracker,
    track_embedding_generation,
    track_llm_call,
    track_rag_pipeline,
    track_vector_search,
)


class TestAIOperationMetrics:
    """Test AIOperationMetrics dataclass."""

    def test_ai_operation_metrics_creation(self):
        """Test creating AI operation metrics."""
        metrics = AIOperationMetrics(
            operation_type="embedding_generation",
            provider="openai",
            model="text-embedding-ada-002",
            duration_ms=150.5,
            tokens_used=100,
            cost_usd=0.002,
            success=True,
        )

        assert metrics.operation_type == "embedding_generation"
        assert metrics.provider == "openai"
        assert metrics.model == "text-embedding-ada-002"
        assert metrics.duration_ms == 150.5
        assert metrics.tokens_used == 100
        assert metrics.cost_usd == 0.002
        assert metrics.success is True

    def test_ai_operation_metrics_defaults(self):
        """Test default values in AI operation metrics."""
        metrics = AIOperationMetrics(
            operation_type="llm_call",
            provider="anthropic",
            model="claude-3-sonnet",
            duration_ms=500.0,
        )

        assert metrics.tokens_used is None
        assert metrics.cost_usd is None
        assert metrics.success is True
        assert metrics.error_message is None
        assert metrics.input_size is None
        assert metrics.output_size is None
        assert metrics.quality_score is None


class TestAIOperationTracker:
    """Test AIOperationTracker class."""

    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = AIOperationTracker()

        assert tracker.meter is not None
        assert tracker.ai_cost_counter is not None
        assert tracker.token_counter is not None
        assert tracker.operation_duration is not None
        assert tracker.quality_gauge is not None
        assert tracker.cache_hit_rate is not None

    def test_record_operation(self):
        """Test recording AI operation metrics."""
        tracker = AIOperationTracker()

        metrics = AIOperationMetrics(
            operation_type="embedding_generation",
            provider="openai",
            model="text-embedding-ada-002",
            duration_ms=120.0,
            tokens_used=50,
            cost_usd=0.001,
            success=True,
        )

        # Should not raise any exceptions
        tracker.record_operation(metrics)

    def test_record_operation_without_optional_fields(self):
        """Test recording operation without optional fields."""
        tracker = AIOperationTracker()

        metrics = AIOperationMetrics(
            operation_type="vector_search",
            provider="qdrant",
            model="test_collection",
            duration_ms=50.0,
            success=True,
        )

        # Should not raise any exceptions
        tracker.record_operation(metrics)


class TestEmbeddingTracking:
    """Test embedding generation tracking."""

    def test_track_embedding_generation_single_text(self):
        """Test tracking single text embedding generation."""
        tracker = AIOperationTracker()

        with tracker.track_embedding_generation(
            provider="openai", model="text-embedding-ada-002", input_texts="Hello world"
        ) as result:
            # Simulate embedding generation
            result["embeddings"] = [0.1, 0.2, 0.3, 0.4]
            result["cost"] = 0.001
            result["cache_hit"] = False

    def test_track_embedding_generation_batch(self):
        """Test tracking batch embedding generation."""
        tracker = AIOperationTracker()

        texts = ["Text 1", "Text 2", "Text 3"]

        with tracker.track_embedding_generation(
            provider="fastembed",
            model="BAAI/bge-small-en",
            input_texts=texts,
            expected_dimensions=384,
        ) as result:
            # Simulate batch embedding generation
            result["embeddings"] = [[0.1, 0.2, 0.3] for _ in texts]
            result["cost"] = None  # Local model, no cost
            result["cache_hit"] = True

    def test_track_embedding_generation_with_exception(self):
        """Test embedding tracking with exceptions."""
        tracker = AIOperationTracker()

        error_msg = "API rate limit exceeded"
        with (
            tracker.track_embedding_generation(
                provider="openai",
                model="text-embedding-ada-002",
                input_texts="Test text",
            ),
            pytest.raises(ValueError, match="API rate limit exceeded"),
        ):
            raise ValueError(error_msg)

    def test_track_embedding_generation_cache_hit(self):
        """Test tracking embedding generation with cache hit."""
        tracker = AIOperationTracker()

        with tracker.track_embedding_generation(
            provider="openai",
            model="text-embedding-ada-002",
            input_texts=["cached text"],
        ) as result:
            # Simulate cache hit
            result["embeddings"] = [[0.5, 0.6, 0.7]]
            result["cache_hit"] = True
            result["cost"] = 0.0  # No cost for cache hit


class TestLLMTracking:
    """Test LLM call tracking."""

    def test_track_llm_call_basic(self):
        """Test basic LLM call tracking."""
        tracker = AIOperationTracker()

        with tracker.track_llm_call(
            provider="openai", model="gpt-4", operation="completion"
        ) as result:
            # Simulate LLM response
            mock_usage = Mock()
            mock_usage.prompt_tokens = 50
            mock_usage.completion_tokens = 30
            mock_usage._total_tokens = 80

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].finish_reason = "stop"

            result["response"] = mock_response
            result["usage"] = mock_usage
            result["cost"] = 0.024

    def test_track_llm_call_with_max_tokens(self):
        """Test LLM call tracking with max tokens specified."""
        tracker = AIOperationTracker()

        with tracker.track_llm_call(
            provider="anthropic",
            model="claude-3-sonnet",
            operation="completion",
            expected_max_tokens=1000,
        ) as result:
            # Simulate Claude response
            mock_usage = Mock()
            mock_usage.input_tokens = 100
            mock_usage.output_tokens = 200
            mock_usage._total_tokens = 300

            result["usage"] = mock_usage
            result["cost"] = 0.045

    def test_track_llm_call_with_exception(self):
        """Test LLM call tracking with exceptions."""
        tracker = AIOperationTracker()

        error_msg = "API service unavailable"
        with (
            tracker.track_llm_call(
                provider="openai", model="gpt-4", operation="completion"
            ),
            pytest.raises(ConnectionError, match="API service unavailable"),
        ):
            raise ConnectionError(error_msg)

    def test_track_llm_call_multiple_choices(self):
        """Test LLM call tracking with multiple response choices."""
        tracker = AIOperationTracker()

        with tracker.track_llm_call(
            provider="openai", model="gpt-3.5-turbo", operation="chat"
        ) as result:
            # Simulate multiple choices response
            mock_response = Mock()
            mock_response.choices = [Mock(), Mock(), Mock()]
            for choice in mock_response.choices:
                choice.finish_reason = "stop"

            result["response"] = mock_response


class TestVectorSearchTracking:
    """Test vector search tracking."""

    def test_track_vector_search_basic(self):
        """Test basic vector search tracking."""
        tracker = AIOperationTracker()

        with tracker.track_vector_search(
            collection_name="documents", query_type="semantic", top_k=10
        ) as result:
            # Simulate search results
            result["results"] = [
                {"id": "doc1", "metadata": {}},
                {"id": "doc2", "metadata": {}},
                {"id": "doc3", "metadata": {}},
            ]
            result["scores"] = [0.95, 0.87, 0.82]
            result["cache_hit"] = False

    def test_track_vector_search_hybrid(self):
        """Test hybrid vector search tracking."""
        tracker = AIOperationTracker()

        with tracker.track_vector_search(
            collection_name="hybrid_docs", query_type="hybrid"
        ) as result:
            # Simulate hybrid search results
            result["results"] = [{"id": f"doc{i}"} for i in range(5)]
            result["scores"] = [0.9, 0.85, 0.8, 0.75, 0.7]
            result["cache_hit"] = True

    def test_track_vector_search_with_exception(self):
        """Test vector search tracking with exceptions."""
        tracker = AIOperationTracker()

        error_msg = "Qdrant connection failed"
        with (
            tracker.track_vector_search(
                collection_name="documents", query_type="semantic"
            ),
            pytest.raises(ConnectionError, match="Qdrant connection failed"),
        ):
            raise ConnectionError(error_msg)

    def test_track_vector_search_empty_results(self):
        """Test vector search tracking with empty results."""
        tracker = AIOperationTracker()

        with tracker.track_vector_search(
            collection_name="empty_collection", query_type="semantic"
        ) as result:
            # Simulate empty search results
            result["results"] = []
            result["scores"] = []
            result["cache_hit"] = False


class TestRAGPipelineTracking:
    """Test RAG pipeline tracking."""

    def test_track_rag_pipeline_complete(self):
        """Test complete RAG pipeline tracking."""
        tracker = AIOperationTracker()

        query = "What is artificial intelligence?"

        with tracker.track_rag_pipeline(
            query=query, retrieval_method="hybrid", generation_model="gpt-4"
        ) as result:
            # Simulate complete RAG pipeline
            result["retrieved_docs"] = [
                {"id": "doc1", "content": "AI is..."},
                {"id": "doc2", "content": "Machine learning..."},
                {"id": "doc3", "content": "Neural networks..."},
            ]
            result["generated_answer"] = (
                "Artificial intelligence is a field of computer science..."
            )
            result["retrieval_time"] = 0.15
            result["generation_time"] = 0.8
            result["_total_cost"] = 0.035

    def test_track_rag_pipeline_minimal(self):
        """Test RAG pipeline tracking with minimal data."""
        tracker = AIOperationTracker()

        with tracker.track_rag_pipeline(
            query="Short query", retrieval_method="semantic"
        ) as result:
            # Simulate minimal pipeline result
            result["retrieved_docs"] = [{"id": "doc1"}]
            result["generated_answer"] = "Short answer"

    def test_track_rag_pipeline_with_exception(self):
        """Test RAG pipeline tracking with exceptions."""
        tracker = AIOperationTracker()

        error_msg = "Generation model failed"
        with (
            tracker.track_rag_pipeline(
                query="Test query", retrieval_method="hybrid", generation_model="gpt-4"
            ),
            pytest.raises(ValueError, match="Generation model failed"),
        ):
            raise ValueError(error_msg)

    def test_track_rag_pipeline_with_timing_breakdown(self):
        """Test RAG pipeline with detailed timing breakdown."""
        tracker = AIOperationTracker()

        with tracker.track_rag_pipeline(
            query="Complex query requiring detailed analysis",
            retrieval_method="hybrid",
            generation_model="claude-3-sonnet",
        ) as result:
            # Simulate pipeline with detailed timing
            result["retrieved_docs"] = [{"id": f"doc{i}"} for i in range(8)]
            result["generated_answer"] = "Detailed analysis of the query..."
            result["retrieval_time"] = 0.25
            result["generation_time"] = 1.2
            result["_total_cost"] = 0.078


class TestCachePerformanceTracking:
    """Test cache performance tracking."""

    def test_record_cache_performance(self):
        """Test recording cache performance metrics."""
        tracker = AIOperationTracker()

        # Should not raise any exceptions
        tracker.record_cache_performance(
            cache_type="embedding",
            operation="get",
            hit_rate=0.85,
            avg_retrieval_time_ms=2.5,
        )

    def test_record_cache_performance_different_types(self):
        """Test recording different cache types."""
        tracker = AIOperationTracker()

        cache_types = ["embedding", "search_results", "llm_responses"]
        operations = ["get", "set", "delete"]

        for cache_type in cache_types:
            for operation in operations:
                tracker.record_cache_performance(
                    cache_type=cache_type,
                    operation=operation,
                    hit_rate=0.7,
                    avg_retrieval_time_ms=1.0,
                )


class TestModelPerformanceTracking:
    """Test model performance tracking."""

    def test_record_model_performance(self):
        """Test recording model performance metrics."""
        tracker = AIOperationTracker()

        tracker.record_model_performance(
            provider="openai",
            model="gpt-4",
            operation_type="completion",
            success_rate=0.98,
            avg_latency_ms=1200.0,
            cost_per_operation=0.03,
        )

    def test_record_model_performance_without_cost(self):
        """Test recording model performance without cost."""
        tracker = AIOperationTracker()

        tracker.record_model_performance(
            provider="fastembed",
            model="BAAI/bge-small-en",
            operation_type="embedding",
            success_rate=0.99,
            avg_latency_ms=45.0,
        )


class TestGlobalTracker:
    """Test global tracker instance management."""

    def test_get_ai_tracker_singleton(self):
        """Test that get_ai_tracker returns singleton instance."""
        tracker1 = get_ai_tracker()
        tracker2 = get_ai_tracker()

        assert tracker1 is tracker2

    def test_convenience_functions(self):
        """Test convenience functions for tracking."""
        # Test convenience function for embedding tracking
        with track_embedding_generation(
            provider="openai", model="text-embedding-ada-002", input_texts="test"
        ) as result:
            result["embeddings"] = [[0.1, 0.2, 0.3]]

        # Test convenience function for LLM tracking
        with track_llm_call(provider="openai", model="gpt-4") as result:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            result["response"] = mock_response
            result["usage"] = Mock()

        # Test convenience function for vector search tracking
        with track_vector_search(
            collection_name="test", query_type="semantic"
        ) as result:
            result["results"] = []
            result["scores"] = []

        # Test convenience function for RAG pipeline tracking
        with track_rag_pipeline(
            query="test query", retrieval_method="hybrid"
        ) as result:
            result["retrieved_docs"] = []
            result["generated_answer"] = "test answer"


@pytest.fixture
def mock_metrics():
    """Fixture providing mocked OpenTelemetry metrics."""
    with patch("src.services.observability.ai_tracking.metrics") as mock:
        meter = Mock()

        # Mock all metric instruments
        meter.create_counter.return_value = Mock()
        meter.create_histogram.return_value = Mock()
        meter.create_gauge.return_value = Mock()

        mock.get_meter.return_value = meter
        yield mock, meter


class TestAITrackingIntegration:
    """Test integration scenarios with mocked dependencies."""

    def test_tracker_with_mocked_metrics(self, mock_metrics):
        """Test tracker initialization with mocked metrics."""
        metrics_module, meter = mock_metrics

        AIOperationTracker()

        # Verify meter creation
        metrics_module.get_meter.assert_called()

        # Verify instrument creation
        assert meter.create_counter.call_count >= 2
        assert meter.create_histogram.call_count >= 1
        assert meter.create_gauge.call_count >= 2

    def test_record_operation_with_mocked_instruments(self, mock_metrics):
        """Test recording operations with mocked instruments."""
        _metrics_module, meter = mock_metrics

        # Setup mocked instruments
        duration_histogram = Mock()
        cost_counter = Mock()
        token_counter = Mock()
        quality_gauge = Mock()

        meter.create_histogram.return_value = duration_histogram
        meter.create_counter.side_effect = [cost_counter, token_counter]
        meter.create_gauge.side_effect = [quality_gauge, Mock()]

        tracker = AIOperationTracker()

        # Record operation
        metrics = AIOperationMetrics(
            operation_type="test",
            provider="test_provider",
            model="test_model",
            duration_ms=100.0,
            tokens_used=50,
            cost_usd=0.01,
            quality_score=0.9,
        )

        tracker.record_operation(metrics)

        # Verify instrument calls
        duration_histogram.record.assert_called_once()
        cost_counter.add.assert_called_once()
        token_counter.add.assert_called_once()
        quality_gauge.set.assert_called_once()
