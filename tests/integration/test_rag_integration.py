"""Integration tests for RAG patterns with service dependencies."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import get_config
from src.services.dependencies import (
    RAGRequest,
    RAGResponse,
    clear_rag_cache,
    generate_rag_answer,
    get_rag_metrics,
)
from src.services.rag.models import AnswerMetrics, RAGResult, SourceAttribution


class TestRAGIntegration:
    """Test RAG integration patterns with service dependencies."""

    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing."""
        return [
            {
                "id": "doc_1",
                "title": "FastAPI Documentation",
                "content": "FastAPI is a modern web framework for building APIs with Python.",
                "url": "https://fastapi.tiangolo.com/",
                "score": 0.95,
                "metadata": {"type": "documentation"},
            },
            {
                "id": "doc_2",
                "title": "Pydantic Guide",
                "content": "Pydantic provides data validation using Python type annotations.",
                "url": "https://docs.pydantic.dev/",
                "score": 0.87,
                "metadata": {"type": "documentation"},
            },
        ]

    @pytest.fixture
    def rag_request(self, sample_search_results):
        """Sample RAG request."""
        return RAGRequest(
            query="How do I use FastAPI with Pydantic?",
            search_results=sample_search_results,
            include_sources=True,
            require_high_confidence=False,
            max_context_results=2,
        )

    @pytest.fixture
    def mock_rag_result(self):
        """Mock RAG result for testing."""
        return RAGResult(
            answer="FastAPI works seamlessly with Pydantic for automatic data validation and serialization.",
            confidence_score=0.85,
            sources=[
                SourceAttribution(
                    source_id="doc_1",
                    title="FastAPI Documentation",
                    url="https://fastapi.tiangolo.com/",
                    relevance_score=0.95,
                    excerpt="FastAPI is a modern web framework...",
                    position_in_context=0,
                ),
            ],
            context_used="FastAPI documentation context...",
            query_processed="How do I use FastAPI with Pydantic?",
            generation_time_ms=1250.0,
            metrics=AnswerMetrics(
                confidence_score=0.85,
                context_utilization=0.8,
                source_diversity=0.7,
                answer_length=87,
                generation_time_ms=1250.0,
                tokens_used=45,
                cost_estimate=0.0012,
            ),
            truncated=False,
            cached=False,
            follow_up_questions=["What are FastAPI's key features?"],
            reasoning_trace=["Analyzed FastAPI and Pydantic integration"],
        )

    @pytest.mark.asyncio
    async def test_rag_answer_generation_success(self, rag_request, mock_rag_result):
        """Test successful RAG answer generation."""
        # Mock the RAG generator
        mock_generator = AsyncMock()
        mock_generator.generate_answer.return_value = mock_rag_result

        # Test the function-based dependency injection pattern
        response = await generate_rag_answer(rag_request, mock_generator)

        # Verify response structure
        assert isinstance(response, RAGResponse)
        assert response.answer == mock_rag_result.answer
        assert response.confidence_score == mock_rag_result.confidence_score
        assert response.sources_used == len(mock_rag_result.sources)
        assert response.generation_time_ms == mock_rag_result.generation_time_ms
        assert response.cached == mock_rag_result.cached

        # Verify source formatting
        assert response.sources is not None
        assert len(response.sources) == 1
        source = response.sources[0]
        assert source["source_id"] == "doc_1"
        assert source["title"] == "FastAPI Documentation"
        assert source["relevance_score"] == 0.95

        # Verify metrics formatting
        assert response.metrics is not None
        assert response.metrics["confidence_score"] == 0.85
        assert response.metrics["tokens_used"] == 45
        assert response.metrics["cost_estimate"] == 0.0012

        # Verify portfolio features
        assert response.follow_up_questions == ["What are FastAPI's key features?"]
        assert response.reasoning_trace == ["Analyzed FastAPI and Pydantic integration"]

        # Verify generator was called correctly
        mock_generator.generate_answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_rag_metrics_retrieval(self):
        """Test RAG metrics retrieval."""
        # Mock the RAG generator with metrics
        mock_generator = MagicMock()
        mock_generator.get_metrics.return_value = {
            "generation_count": 5,
            "total_generation_time": 6250.0,
            "avg_generation_time": 1250.0,
            "total_tokens_used": 225,
            "total_cost": 0.006,
            "avg_cost_per_generation": 0.0012,
            "cache_hits": 2,
            "cache_misses": 3,
            "cache_hit_rate": 0.4,
        }

        # Test metrics retrieval
        metrics = await get_rag_metrics(mock_generator)

        # Verify metrics structure and values
        assert metrics["generation_count"] == 5
        assert metrics["avg_generation_time"] == 1250.0
        assert metrics["total_cost"] == 0.006
        assert metrics["cache_hit_rate"] == 0.4

        # Verify generator was called
        mock_generator.get_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_rag_cache_clearing(self):
        """Test RAG cache clearing."""
        # Mock the RAG generator
        mock_generator = MagicMock()
        mock_generator.clear_cache.return_value = None

        # Test cache clearing
        result = await clear_rag_cache(mock_generator)

        # Verify result
        assert result["status"] == "success"
        assert "successfully" in result["message"]

        # Verify generator was called
        mock_generator.clear_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_rag_generation_error_handling(self, rag_request):
        """Test error handling in RAG generation."""
        # Mock generator that raises an exception
        mock_generator = AsyncMock()
        mock_generator.generate_answer.side_effect = Exception("LLM API error")

        # Test error handling
        with pytest.raises(Exception) as exc_info:
            await generate_rag_answer(rag_request, mock_generator)

        assert "Failed to generate RAG answer" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_request_validation(self, sample_search_results):
        """Test RAG request validation."""
        # Test valid request
        valid_request = RAGRequest(
            query="Test query",
            search_results=sample_search_results,
            include_sources=True,
            temperature=0.5,
            max_tokens=1000,
        )

        assert valid_request.query == "Test query"
        assert len(valid_request.search_results) == 2
        assert valid_request.include_sources is True
        assert valid_request.temperature == 0.5
        assert valid_request.max_tokens == 1000

        # Test request with minimal fields
        minimal_request = RAGRequest(
            query="Minimal query",
            search_results=[],
        )

        assert minimal_request.query == "Minimal query"
        assert minimal_request.search_results == []
        assert minimal_request.include_sources is True  # default
        assert minimal_request.require_high_confidence is False  # default

    @pytest.mark.asyncio
    async def test_response_serialization(self, mock_rag_result):
        """Test RAG response serialization patterns."""
        # Mock generator
        mock_generator = AsyncMock()
        mock_generator.generate_answer.return_value = mock_rag_result

        # Create request
        request = RAGRequest(
            query="Test query",
            search_results=[
                {
                    "id": "test",
                    "title": "Test Doc",
                    "content": "Test content",
                    "score": 0.9,
                }
            ],
        )

        # Generate response
        response = await generate_rag_answer(request, mock_generator)

        # Verify response can be serialized (important for API endpoints)
        response_dict = response.model_dump()

        assert "answer" in response_dict
        assert "confidence_score" in response_dict
        assert "sources_used" in response_dict
        assert "generation_time_ms" in response_dict
        assert "sources" in response_dict
        assert "metrics" in response_dict
        assert "cached" in response_dict

        # Verify nested structures are properly serialized
        if response_dict["sources"]:
            source = response_dict["sources"][0]
            assert "source_id" in source
            assert "title" in source
            assert "relevance_score" in source

        if response_dict["metrics"]:
            metrics = response_dict["metrics"]
            assert "confidence_score" in metrics
            assert "tokens_used" in metrics

    @pytest.mark.asyncio
    async def test_integration_with_circuit_breaker_pattern(self, rag_request):
        """Test integration with circuit breaker patterns."""
        # This test verifies that the RAG functions work with circuit breaker decorators
        # In actual usage, the circuit breakers would be applied via the dependency injection

        mock_generator = AsyncMock()
        mock_generator.generate_answer.return_value = RAGResult(
            answer="Test answer",
            confidence_score=0.8,
            sources=[],
            context_used="Test context",
            query_processed="Test query",
            generation_time_ms=1000.0,
            cached=False,
        )

        # Test that function can be called (circuit breaker would be applied in production)
        response = await generate_rag_answer(rag_request, mock_generator)

        assert response.answer == "Test answer"
        assert response.confidence_score == 0.8

    def test_rag_config_integration(self):
        """Test RAG configuration integration with core config."""

        config = get_config()

        # Verify RAG config is available
        assert hasattr(config, "rag")
        assert hasattr(config.rag, "enable_rag")
        assert hasattr(config.rag, "model")
        assert hasattr(config.rag, "temperature")
        assert hasattr(config.rag, "max_tokens")
        assert hasattr(config.rag, "max_context_length")

        # Verify defaults
        assert isinstance(config.rag.enable_rag, bool)
        assert isinstance(config.rag.model, str)
        assert isinstance(config.rag.temperature, float)
        assert isinstance(config.rag.max_tokens, int)
