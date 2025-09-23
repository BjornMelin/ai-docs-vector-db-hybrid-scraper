"""AI/ML operation tracking with cost attribution and performance monitoring.

This module provides specialized instrumentation for AI/ML operations including
LLM calls, embedding generation, vector search, and RAG pipeline monitoring
with detailed cost tracking and performance analysis.
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass

from opentelemetry import metrics
from opentelemetry.trace import Status, StatusCode

from .instrumentation import add_span_attribute, add_span_event, get_tracer


logger = logging.getLogger(__name__)


@dataclass
class AIOperationMetrics:
    """Metrics for AI/ML operations."""

    operation_type: str
    provider: str
    model: str
    duration_ms: float
    tokens_used: int | None = None
    cost_usd: float | None = None
    success: bool = True
    error_message: str | None = None
    input_size: int | None = None
    output_size: int | None = None
    quality_score: float | None = None


class AIOperationTracker:
    """Tracks AI/ML operations with detailed metrics and cost attribution."""

    def __init__(self):
        """Initialize AI operation tracker with metrics instruments."""
        self.meter = metrics.get_meter(__name__)

        # Cost tracking metrics
        self.ai_cost_counter = self.meter.create_counter(
            "ai_operation_cost_total_usd",
            description="Total cost of AI operations in USD",
            unit="USD",
        )

        # Token usage metrics
        self.token_counter = self.meter.create_counter(
            "ai_tokens_used_total", description="Total tokens used across AI operations"
        )

        # Performance metrics
        self.operation_duration = self.meter.create_histogram(
            "ai_operation_duration_ms",
            description="Duration of AI operations in milliseconds",
            unit="ms",
        )

        # Quality metrics
        self.quality_gauge = self.meter.create_gauge(
            "ai_operation_quality_score",
            description="Quality score of AI operation results",
        )

        # Cache hit rate
        self.cache_hit_rate = self.meter.create_gauge(
            "ai_cache_hit_rate", description="Cache hit rate for AI operations"
        )

    def record_operation(self, metrics: AIOperationMetrics) -> None:
        """Record AI operation metrics.

        Args:
            metrics: AI operation metrics to record

        """
        labels = {
            "operation_type": metrics.operation_type,
            "provider": metrics.provider,
            "model": metrics.model,
            "success": str(metrics.success),
        }

        # Record duration
        self.operation_duration.record(metrics.duration_ms, labels)

        # Record cost if available
        if metrics.cost_usd is not None:
            self.ai_cost_counter.add(metrics.cost_usd, labels)

        # Record token usage if available
        if metrics.tokens_used is not None:
            self.token_counter.add(metrics.tokens_used, labels)

        # Record quality score if available
        if metrics.quality_score is not None:
            self.quality_gauge.set(metrics.quality_score, labels)

    @contextmanager
    def track_embedding_generation(
        self,
        provider: str,
        model: str,
        input_texts: list[str] | str,
        expected_dimensions: int | None = None,
    ):
        """Context manager for tracking embedding generation operations.

        Args:
            provider: Embedding provider (openai, fastembed, cohere, etc.)
            model: Model name
            input_texts: Input text(s) for embedding
            expected_dimensions: Expected embedding dimensions

        Yields:
            Dictionary to store operation results

        """
        tracer = get_tracer()
        operation_result = {"embeddings": None, "cost": None, "cache_hit": False}

        with tracer.start_as_current_span("ai.embedding_generation") as span:
            start_time = time.time()

            # Set span attributes
            span.set_attribute("ai.operation_type", "embedding_generation")
            span.set_attribute("ai.provider", provider)
            span.set_attribute("ai.model", model)

            # Input characteristics
            if isinstance(input_texts, list):
                span.set_attribute("ai.input.batch_size", len(input_texts))
                span.set_attribute(
                    "ai.input.total_chars", sum(len(text) for text in input_texts)
                )
                span.set_attribute("ai.input.type", "batch")
            else:
                span.set_attribute("ai.input.chars", len(input_texts))
                span.set_attribute("ai.input.type", "single")

            if expected_dimensions:
                span.set_attribute(
                    "ai.embedding.expected_dimensions", expected_dimensions
                )

            try:
                yield operation_result

                # Process results
                embeddings = operation_result.get("embeddings")
                if embeddings:
                    if isinstance(embeddings, list) and embeddings:
                        span.set_attribute("ai.output.embedding_count", len(embeddings))
                        if hasattr(embeddings[0], "__len__"):
                            span.set_attribute(
                                "ai.output.dimensions", len(embeddings[0])
                            )
                    elif hasattr(embeddings, "__len__"):
                        span.set_attribute("ai.output.dimensions", len(embeddings))

                # Record cache performance
                cache_hit = operation_result.get("cache_hit", False)
                span.set_attribute("ai.cache.hit", cache_hit)
                add_span_event("embedding_cache_checked", {"hit": cache_hit})

                # Record cost if available
                cost = operation_result.get("cost")
                if cost:
                    span.set_attribute("ai.cost.usd", cost)

                span.set_status(Status(StatusCode.OK))

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

                # Record error metrics
                duration_ms = (time.time() - start_time) * 1000
                error_metrics = AIOperationMetrics(
                    operation_type="embedding_generation",
                    provider=provider,
                    model=model,
                    duration_ms=duration_ms,
                    success=False,
                    error_message=str(e),
                )
                self.record_operation(error_metrics)
                raise

            finally:
                # Record final metrics
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("ai.duration_ms", duration_ms)

                metrics = AIOperationMetrics(
                    operation_type="embedding_generation",
                    provider=provider,
                    model=model,
                    duration_ms=duration_ms,
                    cost_usd=operation_result.get("cost"),
                    success=True,
                    input_size=len(input_texts) if isinstance(input_texts, list) else 1,
                    output_size=len(operation_result.get("embeddings", []))
                    if operation_result.get("embeddings")
                    else 0,
                )
                self.record_operation(metrics)

    @contextmanager
    def track_llm_call(
        self,
        provider: str,
        model: str,
        operation: str = "completion",
        expected_max_tokens: int | None = None,
    ):
        """Context manager for tracking LLM API calls.

        Args:
            provider: LLM provider (openai, anthropic, cohere, etc.)
            model: Model name
            operation: Operation type (completion, chat, etc.)
            expected_max_tokens: Expected maximum tokens

        Yields:
            Dictionary to store operation results

        """
        tracer = get_tracer()
        operation_result = {"response": None, "usage": None, "cost": None}

        with tracer.start_as_current_span("ai.llm_call") as span:
            start_time = time.time()

            # Set span attributes
            span.set_attribute("ai.operation_type", "llm_call")
            span.set_attribute("ai.provider", provider)
            span.set_attribute("ai.model", model)
            span.set_attribute("llm.operation", operation)

            if expected_max_tokens:
                span.set_attribute("llm.max_tokens", expected_max_tokens)

            try:
                yield operation_result

                # Process usage information
                usage = operation_result.get("usage")
                if usage:
                    if hasattr(usage, "prompt_tokens"):
                        span.set_attribute(
                            "llm.usage.prompt_tokens", usage.prompt_tokens
                        )
                    if hasattr(usage, "completion_tokens"):
                        span.set_attribute(
                            "llm.usage.completion_tokens", usage.completion_tokens
                        )
                    if hasattr(usage, "total_tokens"):
                        span.set_attribute("llm.usage.total_tokens", usage.total_tokens)

                # Record cost if available
                cost = operation_result.get("cost")
                if cost:
                    span.set_attribute("ai.cost.usd", cost)

                # Record response characteristics
                response = operation_result.get("response")
                if response and hasattr(response, "choices") and response.choices:
                    span.set_attribute(
                        "llm.response.choices_count", len(response.choices)
                    )
                    if hasattr(response.choices[0], "finish_reason"):
                        span.set_attribute(
                            "llm.response.finish_reason",
                            response.choices[0].finish_reason,
                        )

                span.set_status(Status(StatusCode.OK))

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

                # Record error metrics
                duration_ms = (time.time() - start_time) * 1000
                error_metrics = AIOperationMetrics(
                    operation_type="llm_call",
                    provider=provider,
                    model=model,
                    duration_ms=duration_ms,
                    success=False,
                    error_message=str(e),
                )
                self.record_operation(error_metrics)

                return
                raise

            finally:
                # Record final metrics
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("ai.duration_ms", duration_ms)

                usage = operation_result.get("usage")
                total_tokens = (
                    usage.total_tokens
                    if usage and hasattr(usage, "total_tokens")
                    else None
                )

                metrics = AIOperationMetrics(
                    operation_type="llm_call",
                    provider=provider,
                    model=model,
                    duration_ms=duration_ms,
                    tokens_used=total_tokens,
                    cost_usd=operation_result.get("cost"),
                    success=True,
                )
                self.record_operation(metrics)

    @contextmanager
    def track_vector_search(
        self,
        collection_name: str,
        query_type: str = "semantic",
        top_k: int | None = None,
    ):
        """Context manager for tracking vector search operations.

        Args:
            collection_name: Name of the vector collection
            query_type: Type of search (semantic, hybrid, keyword)
            top_k: Number of results requested

        Yields:
            Dictionary to store operation results

        """
        tracer = get_tracer()
        operation_result = {"results": None, "scores": None, "cache_hit": False}

        with tracer.start_as_current_span("ai.vector_search") as span:
            start_time = time.time()

            # Set span attributes
            span.set_attribute("ai.operation_type", "vector_search")
            span.set_attribute("vector.collection", collection_name)
            span.set_attribute("vector.query_type", query_type)

            if top_k:
                span.set_attribute("vector.top_k", top_k)

            try:
                yield operation_result

                # Process search results
                results = operation_result.get("results")
                scores = operation_result.get("scores")

                if results:
                    span.set_attribute("vector.results_returned", len(results))

                if scores:
                    span.set_attribute("vector.top_score", max(scores))
                    span.set_attribute("vector.min_score", min(scores))
                    span.set_attribute("vector.avg_score", sum(scores) / len(scores))

                # Record cache performance
                cache_hit = operation_result.get("cache_hit", False)
                span.set_attribute("vector.cache.hit", cache_hit)

                # Calculate quality score based on top scores
                quality_score = None
                if scores and scores:
                    quality_score = max(scores)  # Use top score as quality indicator
                    span.set_attribute("vector.quality_score", quality_score)

                span.set_status(Status(StatusCode.OK))

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

                # Record error metrics
                duration_ms = (time.time() - start_time) * 1000
                error_metrics = AIOperationMetrics(
                    operation_type="vector_search",
                    provider="qdrant",
                    model=collection_name,
                    duration_ms=duration_ms,
                    success=False,
                    error_message=str(e),
                )
                self.record_operation(error_metrics)
                raise

            finally:
                # Record final metrics
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("ai.duration_ms", duration_ms)

                results = operation_result.get("results", [])
                scores = operation_result.get("scores", [])
                quality_score = max(scores) if scores else None

                metrics = AIOperationMetrics(
                    operation_type="vector_search",
                    provider="qdrant",
                    model=collection_name,
                    duration_ms=duration_ms,
                    success=True,
                    input_size=1,  # One query
                    output_size=len(results) if results else 0,
                    quality_score=quality_score,
                )
                self.record_operation(metrics)

    @contextmanager
    def track_rag_pipeline(
        self,
        query: str,
        retrieval_method: str = "hybrid",
        generation_model: str = "default",
    ):
        """Context manager for tracking end-to-end RAG pipeline operations.

        Args:
            query: User query
            retrieval_method: Retrieval method used
            generation_model: Model used for generation

        Yields:
            Dictionary to store pipeline results

        """
        tracer = get_tracer()
        pipeline_result = {
            "retrieved_docs": None,
            "generated_answer": None,
            "retrieval_time": None,
            "generation_time": None,
            "total_cost": None,
        }

        with tracer.start_as_current_span("ai.rag_pipeline") as span:
            start_time = time.time()

            # Set span attributes
            span.set_attribute("ai.operation_type", "rag_pipeline")
            span.set_attribute("rag.query_length", len(query))
            span.set_attribute("rag.retrieval_method", retrieval_method)
            span.set_attribute("rag.generation_model", generation_model)

            add_span_event(
                "rag_pipeline_started",
                {"query_chars": len(query), "method": retrieval_method},
            )

            try:
                yield pipeline_result

                # Process pipeline results
                retrieved_docs = pipeline_result.get("retrieved_docs") or []
                generated_answer = pipeline_result.get("generated_answer") or ""

                span.set_attribute("rag.retrieved_docs_count", len(retrieved_docs))
                span.set_attribute("rag.answer_length", len(generated_answer))

                # Record timing breakdown
                retrieval_time = pipeline_result.get("retrieval_time")
                generation_time = pipeline_result.get("generation_time")

                if retrieval_time:
                    span.set_attribute("rag.retrieval_time_ms", retrieval_time * 1000)
                if generation_time:
                    span.set_attribute("rag.generation_time_ms", generation_time * 1000)

                # Record cost if available
                total_cost = pipeline_result.get("total_cost")
                if total_cost:
                    span.set_attribute("ai.cost.usd", total_cost)

                add_span_event(
                    "rag_pipeline_completed",
                    {
                        "docs_retrieved": len(retrieved_docs),
                        "answer_generated": bool(generated_answer),
                    },
                )

                span.set_status(Status(StatusCode.OK))

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

                add_span_event("rag_pipeline_failed", {"error": str(e)})

                # Record error metrics
                duration_ms = (time.time() - start_time) * 1000
                error_metrics = AIOperationMetrics(
                    operation_type="rag_pipeline",
                    provider="combined",
                    model=generation_model,
                    duration_ms=duration_ms,
                    success=False,
                    error_message=str(e),
                )
                self.record_operation(error_metrics)
                raise

            finally:
                # Record final metrics
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("ai.duration_ms", duration_ms)

                answer = pipeline_result.get("generated_answer", "")
                metrics = AIOperationMetrics(
                    operation_type="rag_pipeline",
                    provider="combined",
                    model=generation_model,
                    duration_ms=duration_ms,
                    cost_usd=pipeline_result.get("total_cost"),
                    success=True,
                    input_size=len(query),
                    output_size=len(answer) if answer else 0,
                )
                self.record_operation(metrics)

    def record_cache_performance(
        self,
        cache_type: str,
        operation: str,
        hit_rate: float,
        avg_retrieval_time_ms: float,
    ) -> None:
        """Record cache performance metrics.

        Args:
            cache_type: Type of cache (embedding, search, etc.)
            operation: Operation type
            hit_rate: Cache hit rate (0.0 to 1.0)
            avg_retrieval_time_ms: Average retrieval time in milliseconds

        """
        labels = {"cache_type": cache_type, "operation": operation}

        self.cache_hit_rate.set(hit_rate, labels)

        # Also record in current span if available
        add_span_attribute("cache.hit_rate", hit_rate)
        add_span_attribute("cache.avg_retrieval_time_ms", avg_retrieval_time_ms)

    def record_model_performance(
        self,
        provider: str,
        model: str,
        operation_type: str,
        *,
        success_rate: float,
        avg_latency_ms: float,
        cost_per_operation: float | None = None,
    ) -> None:
        """Record aggregated model performance metrics.

        Args:
            provider: AI/ML provider
            model: Model name
            operation_type: Type of operation
            success_rate: Success rate (0.0 to 1.0)
            avg_latency_ms: Average latency in milliseconds
            cost_per_operation: Average cost per operation

        """
        # Record performance metrics
        add_span_event(
            "model_performance_summary",
            {
                "provider": provider,
                "model": model,
                "operation_type": operation_type,
                "success_rate": success_rate,
                "avg_latency_ms": avg_latency_ms,
                "cost_per_operation": cost_per_operation,
            },
        )


# Global AI operation tracker instance
_ai_tracker: AIOperationTracker | None = None


def get_ai_tracker() -> AIOperationTracker:
    """Get global AI operation tracker instance.

    Returns:
        Global AIOperationTracker instance

    """
    global _ai_tracker
    if _ai_tracker is None:
        _ai_tracker = AIOperationTracker()
    return _ai_tracker


def track_embedding_generation(
    provider: str,
    model: str,
    input_texts: list[str] | str,
    expected_dimensions: int | None = None,
):
    """Convenience function for tracking embedding generation.

    Args:
        provider: Embedding provider
        model: Model name
        input_texts: Input texts
        expected_dimensions: Expected embedding dimensions

    Returns:
        Context manager for tracking

    """
    tracker = get_ai_tracker()
    return tracker.track_embedding_generation(
        provider, model, input_texts, expected_dimensions
    )


def track_llm_call(
    provider: str,
    model: str,
    operation: str = "completion",
    expected_max_tokens: int | None = None,
):
    """Convenience function for tracking LLM calls.

    Args:
        provider: LLM provider
        model: Model name
        operation: Operation type
        expected_max_tokens: Expected maximum tokens

    Returns:
        Context manager for tracking

    """
    tracker = get_ai_tracker()
    return tracker.track_llm_call(provider, model, operation, expected_max_tokens)


def track_vector_search(
    collection_name: str,
    query_type: str = "semantic",
    top_k: int | None = None,
):
    """Convenience function for tracking vector search.

    Args:
        collection_name: Collection name
        query_type: Query type
        top_k: Number of results

    Returns:
        Context manager for tracking

    """
    tracker = get_ai_tracker()
    return tracker.track_vector_search(collection_name, query_type, top_k)


def track_rag_pipeline(
    query: str, retrieval_method: str = "hybrid", generation_model: str = "default"
):
    """Convenience function for tracking RAG pipeline.

    Args:
        query: User query
        retrieval_method: Retrieval method
        generation_model: Generation model

    Returns:
        Context manager for tracking

    """
    tracker = get_ai_tracker()
    return tracker.track_rag_pipeline(query, retrieval_method, generation_model)
