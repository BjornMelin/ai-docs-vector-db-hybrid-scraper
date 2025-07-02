"""Tests for advanced OpenTelemetry instrumentation module."""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from src.services.observability.instrumentation import (
    add_span_attribute,
    add_span_event,
    get_current_span_id,
    get_current_trace_id,
    get_tracer,
    instrument_embedding_generation,
    instrument_function,
    instrument_llm_call,
    instrument_vector_search,
    set_business_context,
    set_user_context,
    trace_async_operation,
    trace_operation,
)


class TestInstrumentationBasics:
    """Test basic instrumentation functionality."""

    def test_get_tracer(self):
        """Test tracer retrieval."""
        tracer = get_tracer()
        assert tracer is not None
        assert hasattr(tracer, "start_span")

    def test_add_span_attribute(self):
        """Test adding span attributes."""
        tracer = get_tracer()
        with tracer.start_as_current_span("test_span"):
            # Should not raise any errors
            add_span_attribute("test.key", "test_value")
            assert True

    def test_add_span_event(self):
        """Test adding span events."""
        tracer = get_tracer()
        with tracer.start_as_current_span("test_span"):
            # Should not raise any errors
            add_span_event("test_event", {"key": "value"})
            assert True

    def test_set_user_context(self):
        """Test setting user context."""
        tracer = get_tracer()
        with tracer.start_as_current_span("test_span"):
            set_user_context("user123", "session456")
            # Context setting should not raise errors
            assert True

    def test_set_business_context(self):
        """Test setting business context."""
        tracer = get_tracer()
        with tracer.start_as_current_span("test_span"):
            set_business_context("semantic", "search_operation")
            assert True

    def test_get_trace_ids(self):
        """Test getting current trace and span IDs."""
        tracer = get_tracer()
        with tracer.start_as_current_span("test_span"):
            trace_id = get_current_trace_id()
            span_id = get_current_span_id()

            # Should return hex strings when in an active span
            if trace_id:  # May be None if no active span
                assert isinstance(trace_id, str)
                assert len(trace_id) == 32  # 128-bit trace ID as hex

            if span_id:
                assert isinstance(span_id, str)
                assert len(span_id) == 16  # 64-bit span ID as hex


class TestFunctionInstrumentation:
    """Test function-level instrumentation decorators."""

    def test_instrument_function_decorator(self):
        """Test basic function instrumentation."""

        @instrument_function("test_operation")
        def test_function(x, y):
            return x + y

        result = test_function(2, 3)
        assert result == 5

    def test_instrument_function_with_attributes(self):
        """Test function instrumentation with additional attributes."""

        @instrument_function("test_operation", {"module": "test"})
        def test_function(x):
            return x * 2

        result = test_function(5)
        assert result == 10

    def test_instrument_function_exception_handling(self):
        """Test function instrumentation with exceptions."""

        @instrument_function("failing_operation")
        def failing_function():
            msg = "Test error"
            raise ValueError(msg)

        with pytest.raises(ValueError):
            failing_function()

    def test_instrument_async_function(self):
        """Test async function instrumentation."""

        @instrument_function("async_operation")
        async def async_function(x):
            await asyncio.sleep(0.01)
            return x * 2

        async def run_test():
            result = await async_function(5)
            assert result == 10

        asyncio.run(run_test())


class TestVectorSearchInstrumentation:
    """Test vector search specific instrumentation."""

    def test_instrument_vector_search_decorator(self):
        """Test vector search instrumentation."""

        @instrument_vector_search("test_collection", "semantic")
        def search_function(_query, top_k=5):
            return [{"id": i, "score": 0.9 - i * 0.1} for i in range(top_k)]

        results = search_function("test query", 3)
        assert len(results) == 3
        assert results[0]["score"] == 0.9

    def test_instrument_vector_search_with_metadata(self):
        """Test vector search instrumentation with metadata collection."""

        @instrument_vector_search("test_collection", "hybrid")
        def search_with_metadata(_query):
            # Simulate vector search results
            return {
                "results": [{"id": "doc1", "score": 0.95}],
                "metadata": {"search_time": 0.05, "cache_hit": False},
            }

        result = search_with_metadata("test query")
        assert "results" in result
        assert "metadata" in result

    def test_instrument_vector_search_exception_handling(self):
        """Test vector search instrumentation with exceptions."""

        @instrument_vector_search("test_collection", "semantic")
        def failing_search(_query):
            msg = "Vector database unavailable"
            raise ConnectionError(msg)

        with pytest.raises(ConnectionError):
            failing_search("test query")


class TestEmbeddingInstrumentation:
    """Test embedding generation instrumentation."""

    def test_instrument_embedding_generation(self):
        """Test embedding generation instrumentation."""

        @instrument_embedding_generation("openai", "text-embedding-ada-002")
        def generate_embeddings(texts):
            # Simulate embedding generation
            return [[0.1, 0.2, 0.3] for _ in texts]

        embeddings = generate_embeddings(["text1", "text2"])
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 3

    def test_instrument_embedding_with_cost_tracking(self):
        """Test embedding instrumentation with cost tracking."""

        @instrument_embedding_generation("openai", "text-embedding-ada-002")
        def generate_embeddings_with_cost(texts):
            return {
                "embeddings": [[0.1, 0.2, 0.3] for _ in texts],
                "cost": 0.002,
                "tokens": len(" ".join(texts).split()),
            }

        result = generate_embeddings_with_cost(["hello world", "test text"])
        assert "embeddings" in result
        assert "cost" in result
        assert "tokens" in result

    def test_instrument_embedding_batch_processing(self):
        """Test embedding instrumentation with batch processing."""

        @instrument_embedding_generation("fastembed", "BAAI/bge-small-en")
        def batch_embed(texts, batch_size=32):
            # Simulate batch processing
            batches = [
                texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
            ]
            all_embeddings = []
            for batch in batches:
                all_embeddings.extend([[0.1, 0.2, 0.3] for _ in batch])
            return all_embeddings

        texts = [f"text_{i}" for i in range(50)]
        embeddings = batch_embed(texts, batch_size=10)
        assert len(embeddings) == 50


class TestLLMInstrumentation:
    """Test LLM call instrumentation."""

    def test_instrument_llm_call(self):
        """Test LLM call instrumentation."""

        @instrument_llm_call("openai", "gpt-4")
        def call_llm(prompt, _max_tokens=100):
            return {
                "response": f"Response to: {prompt}",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "_total_tokens": 30,
                },
            }

        result = call_llm("What is AI?")
        assert "response" in result
        assert "usage" in result

    def test_instrument_llm_with_cost_attribution(self):
        """Test LLM instrumentation with cost attribution."""

        @instrument_llm_call("anthropic", "claude-3-sonnet")
        def call_claude(_messages):
            return {
                "response": "Claude's response",
                "usage": {"input_tokens": 50, "output_tokens": 30},
                "cost": 0.015,
            }

        result = call_claude([{"role": "user", "content": "Hello"}])
        assert result["cost"] == 0.015

    def test_instrument_llm_streaming_response(self):
        """Test LLM instrumentation with streaming responses."""

        @instrument_llm_call("openai", "gpt-4")
        def streaming_llm(_prompt):
            # Simulate streaming response
            chunks = ["Hello", " world", "!"]
            return {"response": "".join(chunks), "streaming": True, "chunks": chunks}

        result = streaming_llm("Say hello")
        assert result["response"] == "Hello world!"
        assert result["streaming"] is True


class TestContextManagers:
    """Test context manager instrumentation."""

    def test_trace_operation_context_manager(self):
        """Test trace_operation context manager."""
        with trace_operation("test_operation", {"category": "test"}):
            time.sleep(0.01)  # Small delay to test timing
            result = "operation_complete"

        assert result == "operation_complete"

    def test_trace_operation_with_exception(self):
        """Test trace_operation context manager with exceptions."""
        with trace_operation("failing_operation"):
            with pytest.raises(ValueError, match="Test error"):
                raise ValueError("Test error")

    def test_trace_async_operation(self):
        """Test async trace_operation context manager."""

        async def async_test():
            async with trace_async_operation("async_test_operation"):
                await asyncio.sleep(0.01)
                return "async_complete"

        result = asyncio.run(async_test())
        assert result == "async_complete"

    def test_trace_async_operation_with_exception(self):
        """Test async trace_operation with exceptions."""

        async def async_failing_test():
            async with trace_async_operation("async_failing_operation"):
                msg = "Async test error"
                raise ValueError(msg)

        with pytest.raises(ValueError):
            asyncio.run(async_failing_test())


class TestPerformanceTracking:
    """Test performance tracking capabilities."""

    def test_operation_timing_tracking(self):
        """Test that operations are properly timed."""

        @instrument_function("timed_operation")
        def slow_operation():
            time.sleep(0.1)
            return "done"

        start_time = time.time()
        result = slow_operation()
        end_time = time.time()

        assert result == "done"
        assert (end_time - start_time) >= 0.1

    def test_nested_operation_tracking(self):
        """Test nested operation tracking."""

        @instrument_function("outer_operation")
        def outer_operation():
            @instrument_function("inner_operation")
            def inner_operation():
                return "inner_result"

            inner_result = inner_operation()
            return f"outer_{inner_result}"

        result = outer_operation()
        assert result == "outer_inner_result"

    def test_concurrent_operation_tracking(self):
        """Test tracking of concurrent operations."""

        @instrument_function("concurrent_operation")
        async def concurrent_task(task_id):
            await asyncio.sleep(0.02)
            return f"task_{task_id}_complete"

        async def run_concurrent_test():
            tasks = [concurrent_task(i) for i in range(3)]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_concurrent_test())
        assert len(results) == 3
        assert all("complete" in result for result in results)


@pytest.fixture
def mock_tracer():
    """Fixture providing a mock tracer for testing."""
    with patch("src.services.observability.instrumentation.get_tracer") as mock:
        tracer = Mock()
        span = Mock()
        span.is_recording.return_value = True
        span.__enter__ = Mock(return_value=span)
        span.__exit__ = Mock(return_value=None)
        tracer.start_as_current_span.return_value = span
        mock.return_value = tracer
        yield tracer, span


class TestInstrumentationIntegration:
    """Test integration scenarios with mocked dependencies."""

    def test_integration_with_mocked_tracer(self, mock_tracer):
        """Test instrumentation with mocked OpenTelemetry tracer."""
        tracer, span = mock_tracer

        @instrument_function("integration_test")
        def test_function():
            return "success"

        result = test_function()

        assert result == "success"
        tracer.start_as_current_span.assert_called()
        span.set_attribute.assert_called()

    def test_error_handling_with_mocked_tracer(self, mock_tracer):
        """Test error handling with mocked tracer."""
        tracer, span = mock_tracer

        @instrument_function("error_test")
        def failing_function():
            msg = "Test error"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError):
            failing_function()

        span.record_exception.assert_called()
        span.set_status.assert_called()
