"""Comprehensive unit tests for EmbeddingManager using modern patterns.

This module demonstrates:
- Async test patterns with pytest-asyncio
- HTTP mocking with respx
- Property-based testing with hypothesis
- Performance testing
"""

import asyncio
import time

import httpx
import pytest
import respx
from hypothesis import assume, given, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from src.services.embeddings.manager import EmbeddingManager
from tests.fixtures.test_infrastructure import (
    PerformanceTestUtils,
    PropertyTestStrategies,
    async_test,
)


class TestEmbeddingManagerAsync:
    """Test EmbeddingManager with async patterns."""

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_single_text_embedding(self, respx_mock):
        """Test embedding generation for single text."""
        # Arrange
        text = "This is a test document for embedding generation."
        expected_embedding = [0.1] * 1536

        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [{"embedding": expected_embedding}],
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                },
            )
        )

        manager = EmbeddingManager(
            openai_api_key="test-key",
            model_name="text-embedding-3-small",
            dimension=1536,
        )
        await manager.initialize()

        # Act
        embedding = await manager.generate_embedding(text)

        # Assert
        assert len(embedding) == 1536
        assert embedding == expected_embedding

        # Cleanup
        await manager.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_batch_embeddings(self, respx_mock):
        """Test batch embedding generation."""
        # Arrange
        texts = [f"Document {i}" for i in range(5)]
        expected_embeddings = [[0.1 + i * 0.01] * 1536 for i in range(5)]

        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [{"embedding": emb} for emb in expected_embeddings],
                    "usage": {"prompt_tokens": 50, "total_tokens": 50},
                },
            )
        )

        manager = EmbeddingManager(
            openai_api_key="test-key",
            model_name="text-embedding-3-small",
            dimension=1536,
        )
        await manager.initialize()

        # Act
        embeddings = await manager.generate_embeddings_batch(texts)

        # Assert
        assert len(embeddings) == 5
        for i, embedding in enumerate(embeddings):
            assert len(embedding) == 1536
            assert embedding[0] == pytest.approx(0.1 + i * 0.01)

        # Cleanup
        await manager.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_concurrent_embedding_requests(self, respx_mock):
        """Test concurrent embedding generation."""
        # Arrange
        num_requests = 10
        texts = [f"Concurrent document {i}" for i in range(num_requests)]

        # Mock each request individually
        call_count = 0

        def mock_response(request):
            nonlocal call_count
            call_count += 1
            return httpx.Response(
                200,
                json={
                    "data": [{"embedding": [0.1] * 1536}],
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                },
            )

        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            side_effect=mock_response
        )

        manager = EmbeddingManager(
            openai_api_key="test-key",
            model_name="text-embedding-3-small",
            dimension=1536,
            max_concurrent_requests=5,  # Limit concurrency
        )
        await manager.initialize()

        # Act
        start_time = time.time()
        embeddings = await asyncio.gather(
            *[manager.generate_embedding(text) for text in texts]
        )
        duration = time.time() - start_time

        # Assert
        assert len(embeddings) == num_requests
        assert call_count == num_requests
        # With max_concurrent_requests=5, should complete reasonably fast
        assert duration < 5.0

        # Cleanup
        await manager.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.respx
    @pytest.mark.parametrize("dimension", [384, 768, 1536, 3072])
    async def test_different_embedding_dimensions(self, respx_mock, dimension):
        """Test embeddings with different dimensions."""
        # Arrange
        text = "Test text for dimension validation"
        expected_embedding = [0.1] * dimension

        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [{"embedding": expected_embedding}],
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                },
            )
        )

        manager = EmbeddingManager(
            openai_api_key="test-key",
            model_name="text-embedding-3-small",
            dimension=dimension,
        )
        await manager.initialize()

        # Act
        embedding = await manager.generate_embedding(text)

        # Assert
        assert len(embedding) == dimension

        # Cleanup
        await manager.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.respx
    async def test_error_handling_with_retry(self, respx_mock):
        """Test error handling and retry logic."""
        # Arrange
        text = "Test document"
        attempts = []

        def track_attempts(request):
            attempts.append(time.time())
            if len(attempts) < 3:
                return httpx.Response(500, json={"error": "Server error"})
            return httpx.Response(
                200,
                json={
                    "data": [{"embedding": [0.1] * 1536}],
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                },
            )

        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            side_effect=track_attempts
        )

        manager = EmbeddingManager(
            openai_api_key="test-key",
            model_name="text-embedding-3-small",
            dimension=1536,
            max_retries=3,
        )
        await manager.initialize()

        # Act
        embedding = await manager.generate_embedding(text)

        # Assert
        assert len(attempts) == 3  # Should retry twice before succeeding
        assert len(embedding) == 1536

        # Cleanup
        await manager.cleanup()


class TestEmbeddingManagerPropertyBased:
    """Property-based tests for EmbeddingManager."""

    @pytest.mark.asyncio
    @pytest.mark.respx
    @pytest.mark.hypothesis
    @given(
        text=PropertyTestStrategies.document_content(),
        dimension=PropertyTestStrategies.valid_embedding_dimensions(),
    )
    async def test_embedding_properties(self, respx_mock, text, dimension):
        """Test embedding generation maintains properties."""
        # Skip empty texts
        assume(len(text.strip()) > 0)

        # Arrange
        expected_embedding = [0.5] * dimension

        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [{"embedding": expected_embedding}],
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                },
            )
        )

        manager = EmbeddingManager(
            openai_api_key="test-key",
            model_name="text-embedding-3-small",
            dimension=dimension,
        )
        await manager.initialize()

        # Act
        embedding = await manager.generate_embedding(text)

        # Assert properties
        assert len(embedding) == dimension
        assert all(isinstance(val, float) for val in embedding)
        assert all(-1.0 <= val <= 1.0 for val in embedding)  # Normalized range

        # Cleanup
        await manager.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.respx
    @pytest.mark.hypothesis
    @given(
        texts=st.lists(
            PropertyTestStrategies.document_content(),
            min_size=1,
            max_size=10,
        )
    )
    async def test_batch_consistency(self, respx_mock, texts):
        """Test batch processing maintains consistency."""
        # Filter out empty texts
        texts = [t for t in texts if t.strip()]
        assume(len(texts) > 0)

        # Arrange
        dimension = 1536
        expected_embeddings = [[0.5] * dimension for _ in texts]

        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [{"embedding": emb} for emb in expected_embeddings],
                    "usage": {
                        "prompt_tokens": len(texts) * 10,
                        "total_tokens": len(texts) * 10,
                    },
                },
            )
        )

        manager = EmbeddingManager(
            openai_api_key="test-key",
            model_name="text-embedding-3-small",
            dimension=dimension,
        )
        await manager.initialize()

        # Act
        embeddings = await manager.generate_embeddings_batch(texts)

        # Assert
        assert len(embeddings) == len(texts)
        assert all(len(emb) == dimension for emb in embeddings)

        # Cleanup
        await manager.cleanup()


class TestEmbeddingManagerPerformance:
    """Performance tests for EmbeddingManager."""

    @pytest.mark.asyncio
    @pytest.mark.respx
    @pytest.mark.performance
    async def test_single_embedding_latency(self, respx_mock):
        """Test single embedding generation latency."""
        # Arrange
        text = "Performance test document"

        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [{"embedding": [0.1] * 1536}],
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                },
            )
        )

        manager = EmbeddingManager(
            openai_api_key="test-key",
            model_name="text-embedding-3-small",
            dimension=1536,
        )
        await manager.initialize()

        # Act & Measure
        async with PerformanceTestUtils.measure_async_time("single_embedding"):
            embedding = await manager.generate_embedding(text)

        # Assert
        assert len(embedding) == 1536
        # Target: < 200ms for single embedding (including mocked API call)

        # Cleanup
        await manager.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.respx
    @pytest.mark.performance
    async def test_batch_throughput(self, respx_mock):
        """Test batch processing throughput."""
        # Arrange
        batch_size = 100
        texts = [f"Document {i}" for i in range(batch_size)]

        respx_mock.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [{"embedding": [0.1] * 1536} for _ in range(batch_size)],
                    "usage": {
                        "prompt_tokens": batch_size * 10,
                        "total_tokens": batch_size * 10,
                    },
                },
            )
        )

        manager = EmbeddingManager(
            openai_api_key="test-key",
            model_name="text-embedding-3-small",
            dimension=1536,
            batch_size=25,  # Process in smaller batches
        )
        await manager.initialize()

        # Act & Measure
        start_time = time.time()
        embeddings = await manager.generate_embeddings_batch(texts)
        duration = time.time() - start_time

        # Calculate throughput
        throughput = batch_size / duration

        # Assert
        assert len(embeddings) == batch_size
        assert throughput > 50  # Should process > 50 docs/second with mocking

        # Cleanup
        await manager.cleanup()


class EmbeddingStateMachine(RuleBasedStateMachine):
    """Stateful testing for EmbeddingManager."""

    def __init__(self):
        super().__init__()
        self.manager = None
        self.embeddings_cache = {}
        self.mock = None

    @initialize()
    async def setup(self):
        """Initialize the embedding manager."""
        self.mock = respx.mock()
        self.mock.start()

        # Setup default mock
        self.mock.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [{"embedding": [0.1] * 1536}],
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                },
            )
        )

        self.manager = EmbeddingManager(
            openai_api_key="test-key",
            model_name="text-embedding-3-small",
            dimension=1536,
        )
        await self.manager.initialize()

    @rule(text=PropertyTestStrategies.document_content())
    async def generate_embedding(self, text):
        """Generate embedding for text."""
        if text.strip():
            embedding = await self.manager.generate_embedding(text)
            self.embeddings_cache[text] = embedding

    @rule()
    async def generate_batch(self):
        """Generate embeddings for cached texts."""
        if self.embeddings_cache:
            texts = list(self.embeddings_cache.keys())[:5]  # Limit batch size
            embeddings = await self.manager.generate_embeddings_batch(texts)
            assert len(embeddings) == len(texts)

    @invariant()
    def embeddings_have_correct_dimension(self):
        """All embeddings should have correct dimension."""
        for embedding in self.embeddings_cache.values():
            assert len(embedding) == 1536

    def teardown(self):
        """Cleanup after state machine test."""
        if self.mock:
            self.mock.stop()
        if self.manager:
            asyncio.run(self.manager.cleanup())
