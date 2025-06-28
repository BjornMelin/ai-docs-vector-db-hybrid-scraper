"""RAG (Retrieval-Augmented Generation) service for answer generation.

This service generates contextual answers from search results using LLMs,
providing a portfolio-worthy implementation of modern RAG patterns.
"""

import asyncio
import hashlib
import logging
import time
from typing import Any

from src.infrastructure.client_manager import ClientManager
from src.services.base import BaseService
from src.services.errors import EmbeddingServiceError

from .models import AnswerMetrics, RAGConfig, RAGRequest, RAGResult, SourceAttribution


logger = logging.getLogger(__name__)


class RAGGenerator(BaseService):
    """Generate contextual answers from search results using LLMs.

    This service implements enterprise-grade RAG patterns including:
    - Source attribution and citation
    - Confidence scoring and quality metrics
    - Context optimization and truncation
    - Parallel processing and caching
    - Portfolio-worthy features for demonstration
    """

    def __init__(
        self,
        config: RAGConfig,
        client_manager: ClientManager | None = None,
    ):
        """Initialize RAG generator.

        Args:
            config: RAG configuration
            client_manager: Optional client manager (will create one if not provided)

        """
        super().__init__(config)
        self.config = config
        self.client_manager = client_manager or ClientManager.from_unified_config()
        self._llm_client = None

        # Performance tracking
        self.generation_count = 0
        self.total_generation_time = 0.0
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.cache_hits = 0
        self.cache_misses = 0

        # Answer cache (simple in-memory for V1)
        self._answer_cache: dict[str, RAGResult] = {}

        # Model pricing (tokens per 1K)
        self.model_pricing = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4o": {"input": 0.005, "output": 0.015},
        }

    async def initialize(self) -> None:
        """Initialize the RAG generator."""
        if self._initialized:
            return

        try:
            # Initialize client manager and get OpenAI client
            await self.client_manager.initialize()
            self._llm_client = await self.client_manager.get_openai_client()

            if not self._llm_client:
                msg = "OpenAI client not available for RAG"
                raise EmbeddingServiceError(msg)

            # Test LLM connection
            await self._llm_client.models.list()

            self._initialized = True
            logger.info("RAG generator initialized")

        except Exception as e:
            msg = f"Failed to initialize RAG generator: {e}"
            raise EmbeddingServiceError(msg) from e

    async def cleanup(self) -> None:
        """Cleanup generator resources."""
        if self.client_manager:
            await self.client_manager.cleanup()
        self._llm_client = None
        self._answer_cache.clear()
        self._initialized = False
        logger.info("RAG generator cleaned up")

    async def generate_answer(self, request: RAGRequest) -> RAGResult:
        """Generate a contextual answer from search results.

        Args:
            request: RAG request with query and search results

        Returns:
            RAGResult: Generated answer with sources, metrics, and metadata

        Raises:
            EmbeddingServiceError: If generation fails or LLM unavailable

        """
        self._validate_initialized()

        start_time = time.time()

        try:
            # Check cache first
            if self.config.enable_caching:
                cache_key = self._get_cache_key(request)
                if cache_key in self._answer_cache:
                    self.cache_hits += 1
                    cached_result = self._answer_cache[cache_key]
                    cached_result.cached = True
                    return cached_result
                self.cache_misses += 1

            # Process and filter search results
            processed_results = self._process_search_results(request)

            if not processed_results:
                return self._create_no_context_result(request, start_time)

            # Build context from search results
            context_info = self._build_context(request, processed_results)

            # Create source attributions
            sources = self._create_source_attributions(processed_results)

            # Generate answer using LLM
            answer_text, reasoning_trace = await self._generate_answer_with_llm(
                request, context_info["context"]
            )

            # Calculate confidence and metrics
            confidence_score = self._calculate_confidence(
                answer_text, context_info, request
            )

            metrics = self._calculate_metrics(
                answer_text, context_info, start_time, request
            )

            # Create result
            result = RAGResult(
                answer=answer_text,
                confidence_score=confidence_score,
                sources=sources,
                context_used=context_info["context"],
                query_processed=request.query,
                generation_time_ms=(time.time() - start_time) * 1000,
                metrics=metrics,
                truncated=context_info["truncated"],
                cached=False,
                reasoning_trace=reasoning_trace
                if self.config.enable_answer_metrics
                else None,
                follow_up_questions=self._generate_follow_up_questions(
                    answer_text, request.query
                ),
            )

            # Cache result
            if self.config.enable_caching:
                self._cache_result(cache_key, result)

            # Update tracking
            self._update_metrics(metrics)

            return result

        except Exception as e:
            logger.error(f"Failed to generate RAG answer: {e}", exc_info=True)  # TODO: Convert f-string to logging format
            msg = f"RAG generation failed: {e}"
            raise EmbeddingServiceError(msg) from e

    def _process_search_results(self, request: RAGRequest) -> list[dict[str, Any]]:
        """Process and filter search results for context building."""
        results = request.search_results.copy()

        # Filter by excluded source IDs
        if request.exclude_source_ids:
            results = [
                r for r in results if r.get("id") not in request.exclude_source_ids
            ]

        # Filter by preferred source types
        if request.preferred_source_types:
            results = [
                r
                for r in results
                if r.get("metadata", {}).get("type") in request.preferred_source_types
            ]

        # Limit number of results
        max_results = request.max_context_results or self.config.max_results_for_context
        results = results[:max_results]

        # Sort by relevance score if available
        if results and "score" in results[0]:
            results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        return results

    def _build_context(
        self, request: RAGRequest, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Build context string from search results with token management."""
        context_parts = []
        token_count = 0
        max_tokens = self.config.max_context_length
        truncated = False

        # Add query context
        query_context = f"User Query: {request.query}\n\nRelevant Information:\n\n"
        context_parts.append(query_context)
        token_count += len(query_context.split()) * 1.3  # Rough token estimate

        for i, result in enumerate(results):
            # Build result context
            result_text = self._format_result_for_context(result, i + 1)
            result_tokens = len(result_text.split()) * 1.3

            # Check if adding this result would exceed token limit
            if token_count + result_tokens > max_tokens:
                truncated = True
                break

            context_parts.append(result_text)
            token_count += result_tokens

        context = "\n".join(context_parts)

        return {
            "context": context,
            "token_count": int(token_count),
            "truncated": truncated,
            "results_used": len(context_parts) - 1,  # Subtract query context
        }

    def _format_result_for_context(self, result: dict[str, Any], index: int) -> str:
        """Format a single search result for inclusion in context."""
        title = result.get("title", f"Document {index}")
        content = result.get("content", "")
        url = result.get("url", "")
        score = result.get("score", 0.0)

        # Truncate content if too long
        if len(content) > 500:
            content = content[:500] + "..."

        formatted = f"[{index}] {title}\n"
        if url:
            formatted += f"URL: {url}\n"
        formatted += f"Relevance: {score:.2f}\n"
        formatted += f"Content: {content}\n"

        return formatted

    def _create_source_attributions(
        self, results: list[dict[str, Any]]
    ) -> list[SourceAttribution]:
        """Create source attribution objects from search results."""
        sources = []

        for i, result in enumerate(results):
            source = SourceAttribution(
                source_id=result.get("id", f"source_{i}"),
                title=result.get("title", f"Document {i + 1}"),
                url=result.get("url"),
                relevance_score=result.get("score", 0.0),
                excerpt=result.get("content", "")[:200],  # First 200 chars
                position_in_context=i,
            )
            sources.append(source)

        return sources

    async def _generate_answer_with_llm(
        self, request: RAGRequest, context: str
    ) -> tuple[str, list[str] | None]:
        """Generate answer using LLM with the provided context."""
        # Build system prompt
        system_prompt = self._build_system_prompt(request)

        # Build user prompt with context
        user_prompt = self._build_user_prompt(request.query, context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Configure generation parameters
        temperature = request.temperature or self.config.temperature
        max_tokens = request.max_tokens or self.config.max_tokens

        try:
            response = await asyncio.wait_for(
                self._llm_client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=30,
                ),
                timeout=self.config.timeout_seconds,
            )

            answer = response.choices[0].message.content.strip()

            # Extract reasoning if answer includes it
            reasoning_trace = None
            if "REASONING:" in answer:
                parts = answer.split("REASONING:", 1)
                if len(parts) == 2:
                    answer = parts[0].strip()
                    reasoning_trace = [parts[1].strip()]

            return answer, reasoning_trace

        except TimeoutError:
            logger.warning("RAG answer generation timed out")
            msg = "Answer generation timed out"
            raise EmbeddingServiceError(msg) from None
        except Exception as e:
            logger.warning(f"Failed to generate answer: {e}")  # TODO: Convert f-string to logging format
            msg = f"Answer generation failed: {e}"
            raise EmbeddingServiceError(msg) from e

    def _build_system_prompt(self, request: RAGRequest) -> str:
        """Build system prompt for RAG answer generation."""
        base_prompt = """You are an expert AI assistant that provides accurate, helpful answers based on the provided context.

Guidelines:
- Use ONLY the information provided in the context to answer the question
- If the context doesn't contain enough information, say so clearly
- Include specific references to sources when possible
- Be concise but comprehensive
- Maintain a professional, helpful tone
- If you're uncertain about something, express that uncertainty"""

        if request.include_sources:
            base_prompt += "\n- Always cite your sources using [1], [2], etc. notation"

        if request.require_high_confidence:
            base_prompt += "\n- Only provide answers you are highly confident about"
            base_prompt += "\n- If confidence is low, explain what additional information would be needed"

        return base_prompt

    def _build_user_prompt(self, query: str, context: str) -> str:
        """Build user prompt with query and context."""
        return f"""{context}

Based on the above information, please answer the following question:

{query}

Provide a clear, accurate answer based solely on the information provided above."""

    def _calculate_confidence(
        self, answer: str, context_info: dict[str, Any], _request: RAGRequest
    ) -> float:
        """Calculate confidence score for the generated answer."""
        confidence = 0.8  # Base confidence

        # Adjust based on context quality
        if context_info["results_used"] >= 3:
            confidence += 0.1  # More sources = higher confidence
        elif context_info["results_used"] == 1:
            confidence -= 0.2  # Single source = lower confidence

        # Adjust based on truncation
        if context_info["truncated"]:
            confidence -= 0.1

        # Adjust based on answer length and specificity
        if len(answer) < 50:
            confidence -= 0.1  # Very short answers might be incomplete
        elif len(answer) > 500:
            confidence += 0.05  # Detailed answers often more confident

        # Check for uncertainty indicators in answer
        uncertainty_phrases = [
            "i don't know",
            "unclear",
            "not enough information",
            "cannot determine",
            "uncertain",
            "unclear from the context",
        ]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence -= 0.3

        return max(0.0, min(1.0, confidence))

    def _calculate_metrics(
        self,
        answer: str,
        context_info: dict[str, Any],
        start_time: float,
        request: RAGRequest,
    ) -> AnswerMetrics:
        """Calculate comprehensive metrics for the generated answer."""
        generation_time = (time.time() - start_time) * 1000

        # Estimate tokens used
        input_tokens = context_info["token_count"]
        output_tokens = len(answer.split()) * 1.3
        total_tokens = input_tokens + output_tokens

        # Calculate cost
        cost = self._calculate_cost(input_tokens, output_tokens)

        # Calculate utilization metrics
        context_utilization = min(
            1.0, len(answer.split()) / max(1, context_info["token_count"] / 10)
        )
        source_diversity = min(
            1.0,
            context_info["results_used"] / max(1, self.config.max_results_for_context),
        )

        return AnswerMetrics(
            confidence_score=self._calculate_confidence(answer, context_info, request),
            context_utilization=context_utilization,
            source_diversity=source_diversity,
            answer_length=len(answer),
            generation_time_ms=generation_time,
            tokens_used=int(total_tokens),
            cost_estimate=cost,
        )

    def _calculate_cost(self, input_tokens: float, output_tokens: float) -> float:
        """Calculate estimated cost for token usage."""
        model_costs = self.model_pricing.get(
            self.config.model,
            {"input": 0.002, "output": 0.002},  # Default fallback
        )

        cost = (input_tokens / 1000) * model_costs["input"] + (
            output_tokens / 1000
        ) * model_costs["output"]

        return cost

    def _generate_follow_up_questions(self, answer: str, _query: str) -> list[str]:
        """Generate relevant follow-up questions based on the answer."""
        # Simple rule-based follow-up generation for V1
        follow_ups = []

        if "error" in answer.lower() or "issue" in answer.lower():
            follow_ups.append("How can I troubleshoot this issue?")
            follow_ups.append("What are common causes of this problem?")

        if "install" in answer.lower() or "setup" in answer.lower():
            follow_ups.append("What are the system requirements?")
            follow_ups.append("Are there any configuration options?")

        if len(follow_ups) == 0:
            follow_ups = [
                "Can you provide more details about this topic?",
                "Are there any related concepts I should know about?",
                "What are the best practices for this?",
            ]

        return follow_ups[:3]  # Limit to 3 follow-ups

    def _create_no_context_result(
        self, request: RAGRequest, start_time: float
    ) -> RAGResult:
        """Create result when no valid context is available."""
        return RAGResult(
            answer="I don't have enough relevant information to answer your question. Please try rephrasing your query or providing more specific details.",
            confidence_score=0.0,
            sources=[],
            context_used="",
            query_processed=request.query,
            generation_time_ms=(time.time() - start_time) * 1000,
            metrics=AnswerMetrics(
                confidence_score=0.0,
                context_utilization=0.0,
                source_diversity=0.0,
                answer_length=0,
                generation_time_ms=(time.time() - start_time) * 1000,
                tokens_used=0,
                cost_estimate=0.0,
            ),
            truncated=False,
            cached=False,
        )

    def _get_cache_key(self, request: RAGRequest) -> str:
        """Generate cache key for request."""
        # Create deterministic cache key
        key_parts = [
            request.query,
            str(len(request.search_results)),
            str(request.max_tokens or self.config.max_tokens),
            str(request.temperature or self.config.temperature),
            str(request.include_sources),
        ]

        # Add search result IDs for uniqueness
        result_ids = [
            r.get("id", str(i)) for i, r in enumerate(request.search_results[:5])
        ]
        key_parts.extend(result_ids)

        cache_string = "|".join(key_parts)
        return hashlib.sha256(cache_string.encode()).hexdigest()

    def _cache_result(self, cache_key: str, result: RAGResult) -> None:
        """Cache result with TTL management."""
        # Simple in-memory cache for V1 (would use Redis/Dragonfly in production)
        if len(self._answer_cache) >= 1000:  # Prevent memory issues
            # Remove oldest entries (simple FIFO)
            to_remove = list(self._answer_cache.keys())[:100]
            for key in to_remove:
                del self._answer_cache[key]

        self._answer_cache[cache_key] = result

    def _update_metrics(self, metrics: AnswerMetrics) -> None:
        """Update service-level metrics."""
        self.generation_count += 1
        self.total_generation_time += metrics.generation_time_ms
        self.total_tokens_used += metrics.tokens_used
        self.total_cost += metrics.cost_estimate

    def get_metrics(self) -> dict[str, float | int]:
        """Get RAG service metrics."""
        return {
            "generation_count": self.generation_count,
            "total_generation_time": self.total_generation_time,
            "avg_generation_time": (
                self.total_generation_time / self.generation_count
                if self.generation_count > 0
                else 0.0
            ),
            "total_tokens_used": self.total_tokens_used,
            "total_cost": self.total_cost,
            "avg_cost_per_generation": (
                self.total_cost / self.generation_count
                if self.generation_count > 0
                else 0.0
            ),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0.0
            ),
        }

    def clear_cache(self) -> None:
        """Clear the answer cache."""
        self._answer_cache.clear()
        logger.info("RAG answer cache cleared")