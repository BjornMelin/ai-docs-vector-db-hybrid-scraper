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
from src.services.rag.models import (
    AnswerMetrics,
    RAGResult,
    RAGServiceMetrics,
    SourceAttribution,
)


class TestRAGIntegration:
    """Test RAG integration patterns with service dependencies."""

    @pytest.fixture
    def rag_request(self):
        """Sample RAG request."""
        return RAGRequest(
            query="How do I use FastAPI with Pydantic?",
            search_results=[
                {
                    "id": "doc_1",
                    "content": "FastAPI documentation excerpt",
                    "metadata": {"title": "FastAPI Docs"},
                },
                {
                    "id": "doc_2",
                    "content": "Pydantic documentation excerpt",
                    "metadata": {"title": "Pydantic Docs"},
                },
            ],
            include_sources=True,
        )

    @pytest.fixture
    def mock_rag_result(self):
        """Mock RAG result for testing."""
        return RAGResult(
            answer=(
                "FastAPI works seamlessly with Pydantic for automatic data "
                "validation and serialization."
            ),
            confidence_score=0.85,
            sources=[
                SourceAttribution(
                    source_id="doc_1",
                    title="FastAPI Documentation",
                    url="https://fastapi.tiangolo.com/",
                    excerpt="FastAPI is a modern web framework...",
                    score=0.95,
                )
            ],
            generation_time_ms=1250.0,
            metrics=AnswerMetrics(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                generation_time_ms=1250.0,
            ),
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

        # Verify source formatting
        assert response.sources is not None
        assert len(response.sources) == 1
        source = response.sources[0]
        assert source["source_id"] == "doc_1"
        assert source["title"] == "FastAPI Documentation"
        assert source["relevance_score"] == 0.95

        # Verify metrics formatting
        assert response.metrics is not None
        assert response.metrics["generation_time_ms"] == pytest.approx(1250.0)

        # Verify generator was called correctly
        mock_generator.generate_answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_rag_metrics_retrieval(self):
        """Test RAG metrics retrieval."""
        # Mock the RAG generator with metrics
        mock_generator = MagicMock()
        mock_generator.get_metrics.return_value = RAGServiceMetrics(
            generation_count=5,
            avg_generation_time_ms=1250.0,
            total_generation_time_ms=6250.0,
        )

        # Test metrics retrieval
        metrics = await get_rag_metrics(mock_generator)

        # Verify metrics structure and values
        assert metrics["generation_count"] == 5
        assert metrics["avg_generation_time_ms"] == pytest.approx(1250.0)
        assert metrics["total_generation_time_ms"] == pytest.approx(6250.0)

        # Verify generator was called
        mock_generator.get_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_rag_cache_clearing(self):
        """Test RAG cache clearing."""
        # Mock the RAG generator
        mock_generator = MagicMock()
        result = await clear_rag_cache(mock_generator)
        assert result["status"] == "noop"
        assert "without a local cache" in result["message"]

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
    async def test_request_validation(self):
        """Test RAG request validation."""
        # Test valid request
        valid_request = RAGRequest(
            query="Test query",
            search_results=[
                {
                    "id": "doc_1",
                    "content": "Test content",
                    "metadata": {"category": "docs"},
                }
            ],
            include_sources=False,
            temperature=0.5,
            max_tokens=1000,
            require_high_confidence=True,
            max_context_results=5,
            preferred_source_types=["official"],
            exclude_source_ids=["old_doc"],
        )

        assert valid_request.query == "Test query"
        assert len(valid_request.search_results) == 1
        assert valid_request.include_sources is False
        assert valid_request.temperature == 0.5
        assert valid_request.max_tokens == 1000
        assert valid_request.require_high_confidence is True
        assert valid_request.max_context_results == 5
        assert valid_request.preferred_source_types == ["official"]
        assert valid_request.exclude_source_ids == ["old_doc"]

        # Test request with minimal fields
        minimal_request = RAGRequest(
            query="Minimal query",
            search_results=[],
        )

        assert minimal_request.query == "Minimal query"
        assert minimal_request.search_results == []
        assert minimal_request.include_sources is True
        assert minimal_request.max_tokens is None
        assert minimal_request.temperature is None

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
                    "id": "doc_1",
                    "content": "Test content",
                    "metadata": {"title": "Test Doc"},
                }
            ],
            include_sources=True,
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
        # Verify nested structures are properly serialized
        if response_dict["sources"]:
            source = response_dict["sources"][0]
            assert "source_id" in source
            assert "title" in source
            assert "relevance_score" in source

        if response_dict["metrics"]:
            metrics = response_dict["metrics"]
            assert "generation_time_ms" in metrics

    @pytest.mark.asyncio
    async def test_integration_with_circuit_breaker_pattern(self, rag_request):
        """Test integration with circuit breaker patterns."""
        # This test verifies that the RAG functions work with circuit breaker decorators
        # In actual usage, the circuit breakers
        # would be applied via the dependency injection

        mock_generator = AsyncMock()
        mock_generator.generate_answer.return_value = RAGResult(
            answer="Test answer",
            confidence_score=0.8,
            sources=[],
            generation_time_ms=1000.0,
            metrics=AnswerMetrics(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                generation_time_ms=1000.0,
            ),
        )

        # Test that function can be called
        # (circuit breaker would be applied in production)
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
