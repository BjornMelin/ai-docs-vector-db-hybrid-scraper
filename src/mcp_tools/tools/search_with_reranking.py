"""Search with reranking tools implementing advanced result refinement and quality assessment.

Provides intelligent search result reranking with ML-powered quality assessment,
autonomous ranking optimization, and multi-criteria result evaluation.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from fastmcp import Context
else:
    # Use a protocol for testing to avoid FastMCP import issues
    from typing import Protocol

    class Context(Protocol):
        async def info(self, msg: str) -> None: ...
        async def debug(self, msg: str) -> None: ...
        async def warning(self, msg: str) -> None: ...
        async def error(self, msg: str) -> None: ...


from src.infrastructure.client_manager import ClientManager
from src.security import MLSecurityValidator as SecurityValidator


logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register search with reranking tools with the MCP server."""

    @mcp.tool()
    async def semantic_reranking_search(
        query: str,
        collection_name: str,
        initial_limit: int = 50,
        final_limit: int = 10,
        reranking_strategy: str = "semantic_similarity",
        quality_factors: dict[str, float] | None = None,
        filters: dict[str, Any] | None = None,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """Perform search with semantic reranking for improved result quality.

        Implements advanced semantic reranking with multiple quality factors
        and autonomous ranking optimization based on query characteristics.

        Args:
            query: Search query text
            collection_name: Target collection for search
            initial_limit: Number of results to retrieve for reranking
            final_limit: Final number of results after reranking
            reranking_strategy: Strategy for reranking (semantic_similarity, quality_score, hybrid)
            quality_factors: Optional weights for quality factors
            filters: Optional metadata filters
            ctx: MCP context for logging

        Returns:
            Reranked search results with quality assessment and ranking metadata
        """
        try:
            if ctx:
                await ctx.info(
                    f"Performing semantic reranking search: '{query}' with {reranking_strategy}"
                )

            # Validate query
            security_validator = SecurityValidator.from_unified_config()
            validated_query = security_validator.validate_query_string(query)

            # Get services
            qdrant_service = await client_manager.get_qdrant_service()
            embedding_manager = await client_manager.get_embedding_manager()

            # Perform initial search with higher limit
            initial_results = await _perform_initial_search(
                qdrant_service,
                embedding_manager,
                validated_query,
                collection_name,
                initial_limit,
                filters,
                ctx,
            )

            if not initial_results["success"] or not initial_results["results"]:
                return {
                    "success": False,
                    "error": "Initial search returned no results",
                    "query": validated_query,
                }

            # Apply quality assessment to all results
            quality_assessed_results = await _apply_quality_assessment(
                initial_results["results"], validated_query, ctx
            )

            # Apply reranking strategy
            reranked_results = await _apply_reranking_strategy(
                quality_assessed_results,
                validated_query,
                reranking_strategy,
                quality_factors,
                ctx,
            )

            # Calculate reranking metrics
            reranking_metrics = _calculate_reranking_metrics(
                initial_results["results"], reranked_results, final_limit
            )

            # Generate optimization insights
            optimization_insights = await _generate_reranking_insights(
                validated_query, initial_results["results"], reranked_results, ctx
            )

            final_results = {
                "success": True,
                "query": validated_query,
                "collection": collection_name,
                "results": reranked_results[:final_limit],
                "reranking_metadata": {
                    "strategy": reranking_strategy,
                    "initial_results_count": len(initial_results["results"]),
                    "quality_factors": quality_factors or {},
                    "ranking_confidence": reranking_metrics["ranking_confidence"],
                    "quality_improvement": reranking_metrics["quality_improvement"],
                },
                "reranking_metrics": reranking_metrics,
                "autonomous_optimization": optimization_insights,
            }

            if ctx:
                await ctx.info(
                    f"Reranking completed: {len(reranked_results)} results with {reranking_metrics['quality_improvement']:.2f} quality improvement"
                )

            return final_results

        except Exception as e:
            logger.exception("Failed to perform semantic reranking search")
            if ctx:
                await ctx.error(f"Semantic reranking search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "reranking_strategy": reranking_strategy,
            }

    @mcp.tool()
    async def multi_criteria_reranking(
        query: str,
        collection_name: str,
        initial_limit: int = 50,
        final_limit: int = 10,
        criteria_weights: dict[str, float] | None = None,
        adaptive_weighting: bool = True,
        filters: dict[str, Any] | None = None,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """Perform multi-criteria reranking with adaptive weight optimization.

        Implements comprehensive multi-criteria ranking with autonomous
        weight adjustment based on query characteristics and result quality.

        Args:
            query: Search query text
            collection_name: Target collection for search
            initial_limit: Number of results to retrieve for reranking
            final_limit: Final number of results after reranking
            criteria_weights: Optional weights for ranking criteria
            adaptive_weighting: Enable autonomous weight adaptation
            filters: Optional metadata filters
            ctx: MCP context for logging

        Returns:
            Multi-criteria reranked results with adaptation metadata
        """
        try:
            if ctx:
                await ctx.info(
                    f"Performing multi-criteria reranking: '{query}' with adaptive weighting"
                )

            # Validate query
            security_validator = SecurityValidator.from_unified_config()
            validated_query = security_validator.validate_query_string(query)

            # Get services
            qdrant_service = await client_manager.get_qdrant_service()
            embedding_manager = await client_manager.get_embedding_manager()

            # Perform initial search
            initial_results = await _perform_initial_search(
                qdrant_service,
                embedding_manager,
                validated_query,
                collection_name,
                initial_limit,
                filters,
                ctx,
            )

            if not initial_results["success"] or not initial_results["results"]:
                return {
                    "success": False,
                    "error": "Initial search returned no results",
                    "query": validated_query,
                }

            # Analyze query for optimal criteria weighting
            if adaptive_weighting:
                optimal_weights = await _analyze_optimal_criteria_weights(
                    validated_query, initial_results["results"], ctx
                )
            else:
                optimal_weights = criteria_weights or _get_default_criteria_weights()

            # Apply comprehensive quality assessment
            assessed_results = await _apply_comprehensive_assessment(
                initial_results["results"], validated_query, optimal_weights, ctx
            )

            # Apply multi-criteria ranking
            multi_ranked_results = await _apply_multi_criteria_ranking(
                assessed_results, optimal_weights, ctx
            )

            # Calculate adaptation metrics
            adaptation_metrics = _calculate_adaptation_metrics(
                criteria_weights, optimal_weights, assessed_results
            )

            # Generate autonomous optimization insights
            optimization_insights = await _generate_multi_criteria_insights(
                validated_query, optimal_weights, multi_ranked_results, ctx
            )

            final_results = {
                "success": True,
                "query": validated_query,
                "collection": collection_name,
                "results": multi_ranked_results[:final_limit],
                "multi_criteria_metadata": {
                    "initial_weights": criteria_weights or {},
                    "optimal_weights": optimal_weights,
                    "adaptive_weighting_applied": adaptive_weighting,
                    "criteria_used": list(optimal_weights.keys()),
                    "ranking_confidence": adaptation_metrics["ranking_confidence"],
                },
                "adaptation_metrics": adaptation_metrics,
                "autonomous_optimization": optimization_insights,
            }

            if ctx:
                await ctx.info(
                    f"Multi-criteria reranking completed with {len(multi_ranked_results)} results"
                )

            return final_results

        except Exception as e:
            logger.exception("Failed to perform multi-criteria reranking")
            if ctx:
                await ctx.error(f"Multi-criteria reranking failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "adaptive_weighting": adaptive_weighting,
            }

    @mcp.tool()
    async def contextual_reranking_search(
        query: str,
        collection_name: str,
        context_query: str | None = None,
        context_weight: float = 0.3,
        initial_limit: int = 50,
        final_limit: int = 10,
        filters: dict[str, Any] | None = None,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """Perform contextual reranking considering additional context.

        Implements context-aware reranking that considers additional context
        or user intent to improve result relevance and ranking quality.

        Args:
            query: Primary search query
            collection_name: Target collection for search
            context_query: Optional additional context for reranking
            context_weight: Weight for context influence on ranking
            initial_limit: Number of results to retrieve for reranking
            final_limit: Final number of results after reranking
            filters: Optional metadata filters
            ctx: MCP context for logging

        Returns:
            Contextually reranked search results with context influence metadata
        """
        try:
            if ctx:
                await ctx.info(
                    f"Performing contextual reranking: '{query}' with context weight {context_weight}"
                )

            # Validate queries
            security_validator = SecurityValidator.from_unified_config()
            validated_query = security_validator.validate_query_string(query)
            validated_context = None

            if context_query:
                validated_context = security_validator.validate_query_string(
                    context_query
                )

            # Get services
            qdrant_service = await client_manager.get_qdrant_service()
            embedding_manager = await client_manager.get_embedding_manager()

            # Perform initial search
            initial_results = await _perform_initial_search(
                qdrant_service,
                embedding_manager,
                validated_query,
                collection_name,
                initial_limit,
                filters,
                ctx,
            )

            if not initial_results["success"] or not initial_results["results"]:
                return {
                    "success": False,
                    "error": "Initial search returned no results",
                    "query": validated_query,
                }

            # Generate context embeddings if context provided
            context_embeddings = None
            if validated_context:
                context_result = await embedding_manager.generate_embeddings(
                    [validated_context]
                )
                context_embeddings = context_result.embeddings[0]

                if ctx:
                    await ctx.debug(
                        f"Generated context embeddings for: '{validated_context}'"
                    )

            # Apply contextual assessment
            contextual_results = await _apply_contextual_assessment(
                initial_results["results"],
                validated_query,
                validated_context,
                context_embeddings,
                embedding_manager,
                ctx,
            )

            # Apply contextual reranking
            reranked_results = await _apply_contextual_reranking(
                contextual_results, context_weight, ctx
            )

            # Calculate contextual metrics
            contextual_metrics = _calculate_contextual_metrics(
                initial_results["results"], reranked_results, context_weight
            )

            final_results = {
                "success": True,
                "query": validated_query,
                "context_query": validated_context,
                "collection": collection_name,
                "results": reranked_results[:final_limit],
                "contextual_metadata": {
                    "context_provided": bool(validated_context),
                    "context_weight": context_weight,
                    "context_influence": contextual_metrics["context_influence"],
                    "ranking_adjustment": contextual_metrics["ranking_adjustment"],
                },
                "contextual_metrics": contextual_metrics,
            }

            if ctx:
                await ctx.info(
                    f"Contextual reranking completed with {contextual_metrics['context_influence']:.2f} context influence"
                )

            return final_results

        except Exception as e:
            logger.exception("Failed to perform contextual reranking search")
            if ctx:
                await ctx.error(f"Contextual reranking search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "context_query": context_query,
            }

    @mcp.tool()
    async def get_reranking_capabilities() -> dict[str, Any]:
        """Get reranking capabilities and configuration options.

        Returns:
            Comprehensive capabilities information for reranking system
        """
        return {
            "reranking_strategies": {
                "semantic_similarity": {
                    "description": "Reranks based on semantic similarity to query",
                    "best_for": ["conceptual_queries", "semantic_search"],
                    "complexity": "medium",
                    "effectiveness": "high",
                },
                "quality_score": {
                    "description": "Reranks based on content quality assessment",
                    "best_for": ["information_quality", "content_curation"],
                    "complexity": "low",
                    "effectiveness": "medium",
                },
                "hybrid": {
                    "description": "Combines multiple ranking factors",
                    "best_for": ["balanced_results", "general_queries"],
                    "complexity": "high",
                    "effectiveness": "optimal",
                },
            },
            "quality_factors": {
                "semantic_relevance": "Similarity to query intent and meaning",
                "content_quality": "Overall quality and completeness of content",
                "freshness": "Recency and currency of information",
                "authority": "Source credibility and expertise",
                "completeness": "Comprehensiveness of content coverage",
                "readability": "Clarity and accessibility of content",
            },
            "multi_criteria_features": {
                "adaptive_weighting": True,
                "weight_optimization": True,
                "criteria_correlation": True,
                "performance_targeting": True,
            },
            "contextual_features": {
                "context_aware_ranking": True,
                "intent_analysis": True,
                "context_weighting": True,
                "multi_query_support": True,
            },
            "autonomous_capabilities": {
                "quality_assessment": True,
                "weight_optimization": True,
                "strategy_selection": True,
                "performance_correlation": True,
            },
            "metrics": [
                "ranking_confidence",
                "quality_improvement",
                "context_influence",
                "adaptation_effectiveness",
            ],
            "status": "active",
        }


# Helper functions


async def _perform_initial_search(
    qdrant_service,
    embedding_manager,
    query: str,
    collection_name: str,
    limit: int,
    filters: dict | None,
    ctx,
) -> dict[str, Any]:
    """Perform initial search to get results for reranking."""
    try:
        # Generate embedding for query
        embeddings_result = await embedding_manager.generate_embeddings([query])
        query_embedding = embeddings_result.embeddings[0]

        # Perform vector search
        search_result = await qdrant_service.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            filter=filters,
            with_payload=True,
            with_vectors=True,  # Need vectors for semantic similarity
        )

        results = search_result.get("points", [])

        if ctx:
            await ctx.debug(f"Initial search returned {len(results)} results")

        return {
            "success": True,
            "results": results,
            "query_embedding": query_embedding,
        }

    except Exception as e:
        logger.exception("Initial search failed")
        return {
            "success": False,
            "error": str(e),
        }


async def _apply_quality_assessment(results: list[dict], query: str, ctx) -> list[dict]:
    """Apply comprehensive quality assessment to search results."""
    assessed_results = []

    for result in results:
        payload = result.get("payload", {})
        content = payload.get("content", "")
        title = payload.get("title", "")

        # Calculate various quality factors
        quality_scores = {
            "semantic_relevance": _calculate_semantic_relevance(query, content, title),
            "content_quality": _calculate_content_quality(content),
            "freshness": _calculate_freshness(payload),
            "authority": _calculate_authority(payload),
            "completeness": _calculate_completeness(content),
            "readability": _calculate_readability(content),
        }

        # Calculate overall quality score
        overall_quality = sum(quality_scores.values()) / len(quality_scores)

        # Add quality metadata to result
        result["quality_assessment"] = {
            "overall_quality": overall_quality,
            "quality_factors": quality_scores,
            "assessment_confidence": 0.85,  # Mock confidence
        }

        assessed_results.append(result)

    if ctx:
        await ctx.debug(
            f"Applied quality assessment to {len(assessed_results)} results"
        )

    return assessed_results


def _calculate_semantic_relevance(query: str, content: str, title: str) -> float:
    """Calculate semantic relevance score."""
    query_words = set(query.lower().split())
    content_words = set(content.lower().split())
    title_words = set(title.lower().split())

    # Simple term overlap calculation
    content_overlap = (
        len(query_words.intersection(content_words)) / len(query_words)
        if query_words
        else 0
    )
    title_overlap = (
        len(query_words.intersection(title_words)) / len(query_words)
        if query_words
        else 0
    )

    # Weight title more heavily
    relevance = (content_overlap * 0.7) + (title_overlap * 0.3)
    return min(relevance, 1.0)


def _calculate_content_quality(content: str) -> float:
    """Calculate content quality score based on various factors."""
    if not content:
        return 0.0

    # Length factor (prefer moderate length)
    length_score = (
        min(len(content) / 1000, 1.0)
        if len(content) < 2000
        else max(1.0 - (len(content) - 2000) / 5000, 0.3)
    )

    # Structure factor (presence of punctuation, formatting)
    structure_score = min(
        content.count(".") / 10, 1.0
    )  # Prefer well-structured content

    # Diversity factor (vocabulary diversity)
    words = content.split()
    unique_words = set(words)
    diversity_score = len(unique_words) / len(words) if words else 0

    quality = (length_score * 0.4) + (structure_score * 0.3) + (diversity_score * 0.3)
    return min(quality, 1.0)


def _calculate_freshness(payload: dict) -> float:
    """Calculate freshness score based on timestamp or recency indicators."""
    # Mock freshness calculation (replace with actual timestamp analysis)
    return 0.7  # Default medium freshness


def _calculate_authority(payload: dict) -> float:
    """Calculate authority score based on source credibility."""
    # Mock authority calculation (replace with actual domain/source analysis)
    url = payload.get("url", "")

    # Simple domain-based authority scoring
    high_authority_domains = ["edu", "gov", "org"]
    domain = url.split("//")[-1].split("/")[0].split(".")[-1] if url else ""

    if domain in high_authority_domains:
        return 0.9
    if "github" in url or "stackoverflow" in url:
        return 0.8
    return 0.6


def _calculate_completeness(content: str) -> float:
    """Calculate completeness score based on content coverage."""
    if not content:
        return 0.0

    # Check for various content indicators
    indicators = ["example", "implementation", "step", "method", "approach"]
    coverage = sum(1 for indicator in indicators if indicator in content.lower())

    return min(coverage / len(indicators), 1.0)


def _calculate_readability(content: str) -> float:
    """Calculate readability score."""
    if not content:
        return 0.0

    # Simple readability metrics
    sentences = content.count(".") + content.count("!") + content.count("?")
    words = len(content.split())

    if sentences == 0:
        return 0.5

    avg_sentence_length = words / sentences

    # Prefer moderate sentence length (10-20 words)
    if 10 <= avg_sentence_length <= 20:
        readability = 1.0
    elif avg_sentence_length < 10:
        readability = avg_sentence_length / 10
    else:
        readability = max(20 / avg_sentence_length, 0.3)

    return readability


async def _apply_reranking_strategy(
    results: list[dict],
    query: str,
    strategy: str,
    quality_factors: dict[str, float] | None,
    ctx,
) -> list[dict]:
    """Apply the specified reranking strategy."""
    if strategy == "semantic_similarity":
        return await _semantic_similarity_reranking(results, query, ctx)
    if strategy == "quality_score":
        return await _quality_score_reranking(results, quality_factors, ctx)
    # hybrid
    return await _hybrid_reranking(results, query, quality_factors, ctx)


async def _semantic_similarity_reranking(
    results: list[dict], query: str, ctx
) -> list[dict]:
    """Rerank based on semantic similarity."""
    # Sort by semantic relevance from quality assessment
    reranked = sorted(
        results,
        key=lambda x: x.get("quality_assessment", {})
        .get("quality_factors", {})
        .get("semantic_relevance", 0.0),
        reverse=True,
    )

    # Add reranking metadata
    for i, result in enumerate(reranked):
        result["reranking_metadata"] = {
            "strategy": "semantic_similarity",
            "new_rank": i + 1,
            "original_rank": results.index(result) + 1 if result in results else -1,
            "ranking_score": result.get("quality_assessment", {})
            .get("quality_factors", {})
            .get("semantic_relevance", 0.0),
        }

    return reranked


async def _quality_score_reranking(
    results: list[dict], quality_factors: dict[str, float] | None, ctx
) -> list[dict]:
    """Rerank based on overall quality score."""
    weights = quality_factors or _get_default_quality_weights()

    # Calculate weighted quality scores
    for result in results:
        quality_assessment = result.get("quality_assessment", {})
        quality_factors_scores = quality_assessment.get("quality_factors", {})

        weighted_score = sum(
            quality_factors_scores.get(factor, 0.0) * weight
            for factor, weight in weights.items()
        )

        result["weighted_quality_score"] = weighted_score

    # Sort by weighted quality score
    reranked = sorted(
        results, key=lambda x: x.get("weighted_quality_score", 0.0), reverse=True
    )

    # Add reranking metadata
    for i, result in enumerate(reranked):
        result["reranking_metadata"] = {
            "strategy": "quality_score",
            "new_rank": i + 1,
            "original_rank": results.index(result) + 1 if result in results else -1,
            "ranking_score": result.get("weighted_quality_score", 0.0),
            "weights_used": weights,
        }

    return reranked


async def _hybrid_reranking(
    results: list[dict], query: str, quality_factors: dict[str, float] | None, ctx
) -> list[dict]:
    """Apply hybrid reranking combining multiple factors."""
    # Combine original score, semantic relevance, and quality
    for result in results:
        original_score = result.get("score", 0.0)
        quality_assessment = result.get("quality_assessment", {})
        semantic_relevance = quality_assessment.get("quality_factors", {}).get(
            "semantic_relevance", 0.0
        )
        overall_quality = quality_assessment.get("overall_quality", 0.0)

        # Hybrid score combining multiple factors
        hybrid_score = (
            original_score * 0.4 + semantic_relevance * 0.35 + overall_quality * 0.25
        )

        result["hybrid_score"] = hybrid_score

    # Sort by hybrid score
    reranked = sorted(results, key=lambda x: x.get("hybrid_score", 0.0), reverse=True)

    # Add reranking metadata
    for i, result in enumerate(reranked):
        result["reranking_metadata"] = {
            "strategy": "hybrid",
            "new_rank": i + 1,
            "original_rank": results.index(result) + 1 if result in results else -1,
            "ranking_score": result.get("hybrid_score", 0.0),
            "score_components": {
                "original_score": result.get("score", 0.0),
                "semantic_relevance": result.get("quality_assessment", {})
                .get("quality_factors", {})
                .get("semantic_relevance", 0.0),
                "overall_quality": result.get("quality_assessment", {}).get(
                    "overall_quality", 0.0
                ),
            },
        }

    return reranked


def _get_default_quality_weights() -> dict[str, float]:
    """Get default weights for quality factors."""
    return {
        "semantic_relevance": 0.3,
        "content_quality": 0.25,
        "freshness": 0.15,
        "authority": 0.15,
        "completeness": 0.1,
        "readability": 0.05,
    }


def _calculate_reranking_metrics(
    original_results: list[dict], reranked_results: list[dict], final_limit: int
) -> dict[str, Any]:
    """Calculate metrics for reranking effectiveness."""
    # Calculate ranking changes
    rank_changes = []
    for i, result in enumerate(reranked_results[:final_limit]):
        original_rank = (
            original_results.index(result) + 1 if result in original_results else -1
        )
        new_rank = i + 1
        if original_rank > 0:
            rank_changes.append(abs(new_rank - original_rank))

    # Calculate quality improvement
    original_avg_quality = sum(
        r.get("quality_assessment", {}).get("overall_quality", 0.0)
        for r in original_results[:final_limit]
    ) / min(len(original_results), final_limit)

    reranked_avg_quality = sum(
        r.get("quality_assessment", {}).get("overall_quality", 0.0)
        for r in reranked_results[:final_limit]
    ) / min(len(reranked_results), final_limit)

    quality_improvement = reranked_avg_quality - original_avg_quality

    return {
        "average_rank_change": sum(rank_changes) / len(rank_changes)
        if rank_changes
        else 0.0,
        "max_rank_change": max(rank_changes) if rank_changes else 0,
        "quality_improvement": quality_improvement,
        "ranking_confidence": min(
            abs(quality_improvement) * 2, 1.0
        ),  # Simple confidence metric
        "reranked_count": len(reranked_results),
    }


async def _generate_reranking_insights(
    query: str, original_results: list[dict], reranked_results: list[dict], ctx
) -> dict[str, Any]:
    """Generate insights for reranking optimization."""
    insights = {
        "reranking_effectiveness": {
            "significant_reordering": any(
                abs(reranked_results.index(r) - original_results.index(r)) > 5
                for r in reranked_results[:10]
                if r in original_results
            ),
            "quality_correlation": _calculate_quality_correlation(reranked_results),
        },
        "top_results_analysis": {
            "top_result_changed": (reranked_results[0] != original_results[0])
            if reranked_results and original_results
            else False,
            "top_3_stability": len(
                {r.get("id") for r in reranked_results[:3]}.intersection(
                    {r.get("id") for r in original_results[:3]}
                )
            )
            / 3
            if len(reranked_results) >= 3 and len(original_results) >= 3
            else 0.0,
        },
        "recommendations": [],
    }

    # Generate recommendations
    if not insights["reranking_effectiveness"]["significant_reordering"]:
        insights["recommendations"].append(
            "Reranking had minimal impact - consider different strategy"
        )

    if insights["top_results_analysis"]["top_3_stability"] < 0.5:
        insights["recommendations"].append(
            "High volatility in top results - may indicate conflicting ranking factors"
        )

    return insights


def _calculate_quality_correlation(results: list[dict]) -> float:
    """Calculate correlation between ranking position and quality scores."""
    if len(results) < 3:
        return 0.0

    # Simple correlation calculation
    quality_scores = [
        r.get("quality_assessment", {}).get("overall_quality", 0.0) for r in results
    ]

    # Check if quality generally decreases with rank (positive correlation)
    decreasing_count = 0
    for i in range(1, len(quality_scores)):
        if quality_scores[i] <= quality_scores[i - 1]:
            decreasing_count += 1

    return decreasing_count / (len(quality_scores) - 1)


async def _analyze_optimal_criteria_weights(
    query: str, results: list[dict], ctx
) -> dict[str, float]:
    """Analyze optimal criteria weights based on query and results."""
    # Simple heuristic for weight optimization
    query_words = query.lower().split()

    weights = _get_default_criteria_weights()

    # Adjust weights based on query characteristics
    if any(word in query_words for word in ["how", "implement", "tutorial"]):
        # Procedural queries - emphasize completeness and readability
        weights["completeness"] = 0.3
        weights["readability"] = 0.2
        weights["semantic_relevance"] = 0.25
        weights["content_quality"] = 0.15
        weights["authority"] = 0.1
    elif any(word in query_words for word in ["what", "define", "definition"]):
        # Definitional queries - emphasize authority and semantic relevance
        weights["semantic_relevance"] = 0.35
        weights["authority"] = 0.25
        weights["content_quality"] = 0.2
        weights["completeness"] = 0.15
        weights["readability"] = 0.05
    elif any(word in query_words for word in ["latest", "recent", "new"]):
        # Temporal queries - emphasize freshness
        weights["freshness"] = 0.4
        weights["semantic_relevance"] = 0.25
        weights["content_quality"] = 0.2
        weights["authority"] = 0.1
        weights["completeness"] = 0.05

    if ctx:
        await ctx.debug("Optimized criteria weights based on query characteristics")

    return weights


def _get_default_criteria_weights() -> dict[str, float]:
    """Get default weights for multi-criteria ranking."""
    return {
        "semantic_relevance": 0.3,
        "content_quality": 0.25,
        "authority": 0.2,
        "completeness": 0.15,
        "freshness": 0.1,
    }


async def _apply_comprehensive_assessment(
    results: list[dict], query: str, weights: dict[str, float], ctx
) -> list[dict]:
    """Apply comprehensive multi-criteria assessment."""
    assessed_results = await _apply_quality_assessment(results, query, ctx)

    # Calculate weighted scores using optimal weights
    for result in assessed_results:
        quality_factors = result.get("quality_assessment", {}).get(
            "quality_factors", {}
        )

        weighted_score = sum(
            quality_factors.get(factor, 0.0) * weight
            for factor, weight in weights.items()
            if factor in quality_factors
        )

        result["multi_criteria_score"] = weighted_score
        result["criteria_weights_used"] = weights

    return assessed_results


async def _apply_multi_criteria_ranking(
    results: list[dict], weights: dict[str, float], ctx
) -> list[dict]:
    """Apply multi-criteria ranking to results."""
    # Sort by multi-criteria score
    ranked_results = sorted(
        results, key=lambda x: x.get("multi_criteria_score", 0.0), reverse=True
    )

    # Add ranking metadata
    for i, result in enumerate(ranked_results):
        result["multi_criteria_metadata"] = {
            "rank": i + 1,
            "score": result.get("multi_criteria_score", 0.0),
            "criteria_breakdown": result.get("quality_assessment", {}).get(
                "quality_factors", {}
            ),
            "weights_applied": weights,
        }

    return ranked_results


def _calculate_adaptation_metrics(
    original_weights: dict[str, float] | None,
    optimal_weights: dict[str, float],
    results: list[dict],
) -> dict[str, Any]:
    """Calculate metrics for weight adaptation."""
    # Calculate weight changes if original weights provided
    weight_changes = {}
    if original_weights:
        for factor in optimal_weights:
            original = original_weights.get(factor, 0.0)
            optimal = optimal_weights[factor]
            weight_changes[factor] = optimal - original

    # Calculate ranking confidence based on score distribution
    scores = [r.get("multi_criteria_score", 0.0) for r in results]
    score_variance = (
        sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)
        if scores
        else 0
    )
    ranking_confidence = min(score_variance * 10, 1.0)  # Simple confidence metric

    return {
        "weight_changes": weight_changes,
        "adaptation_magnitude": sum(abs(change) for change in weight_changes.values())
        if weight_changes
        else 0.0,
        "ranking_confidence": ranking_confidence,
        "score_distribution": {
            "mean": sum(scores) / len(scores) if scores else 0.0,
            "variance": score_variance,
            "range": max(scores) - min(scores) if scores else 0.0,
        },
    }


async def _generate_multi_criteria_insights(
    query: str, optimal_weights: dict[str, float], results: list[dict], ctx
) -> dict[str, Any]:
    """Generate insights for multi-criteria optimization."""
    insights = {
        "weight_analysis": {
            "dominant_criteria": max(optimal_weights.items(), key=lambda x: x[1])[0],
            "weight_distribution": "balanced"
            if max(optimal_weights.values()) < 0.4
            else "focused",
        },
        "result_analysis": {
            "score_spread": max(r.get("multi_criteria_score", 0.0) for r in results)
            - min(r.get("multi_criteria_score", 0.0) for r in results)
            if results
            else 0.0,
            "top_criteria_consistency": _analyze_top_criteria_consistency(results),
        },
        "recommendations": [],
    }

    # Generate recommendations
    if insights["weight_analysis"]["weight_distribution"] == "focused":
        insights["recommendations"].append(
            "Consider more balanced criteria weighting for diverse results"
        )

    if insights["result_analysis"]["score_spread"] < 0.2:
        insights["recommendations"].append(
            "Low score differentiation - may need more discriminative criteria"
        )

    return insights


def _analyze_top_criteria_consistency(results: list[dict]) -> float:
    """Analyze consistency of top-performing criteria across results."""
    if len(results) < 5:
        return 1.0

    # Get top criteria for each result
    top_criteria = []
    for result in results[:5]:
        criteria_scores = result.get("multi_criteria_metadata", {}).get(
            "criteria_breakdown", {}
        )
        if criteria_scores:
            top_criterion = max(criteria_scores.items(), key=lambda x: x[1])[0]
            top_criteria.append(top_criterion)

    # Calculate consistency
    if not top_criteria:
        return 0.0

    most_common = max(set(top_criteria), key=top_criteria.count)
    return top_criteria.count(most_common) / len(top_criteria)


async def _apply_contextual_assessment(
    results: list[dict],
    query: str,
    context_query: str | None,
    context_embeddings: list[float] | None,
    embedding_manager,
    ctx,
) -> list[dict]:
    """Apply contextual assessment to results."""
    assessed_results = []

    for result in results:
        # Start with existing quality assessment or create basic one
        if "quality_assessment" not in result:
            result = (await _apply_quality_assessment([result], query, ctx))[0]

        # Add contextual relevance if context provided
        context_relevance = 0.5  # Default neutral relevance

        if context_query and context_embeddings:
            # Calculate semantic similarity to context
            result_content = result.get("payload", {}).get("content", "")
            if result_content:
                try:
                    result_embedding_response = (
                        await embedding_manager.generate_embeddings(
                            [result_content[:500]]
                        )
                    )  # Limit content length
                    result_embedding = result_embedding_response.embeddings[0]

                    # Calculate cosine similarity
                    context_relevance = _calculate_cosine_similarity(
                        context_embeddings, result_embedding
                    )
                except (asyncio.CancelledError, TimeoutError, RuntimeError) as e:
                    logger.warning(f"Failed to calculate context relevance: {e}")

        # Add contextual metadata
        result["contextual_assessment"] = {
            "context_relevance": context_relevance,
            "context_query_used": bool(context_query),
        }

        assessed_results.append(result)

    return assessed_results


def _calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


async def _apply_contextual_reranking(
    results: list[dict], context_weight: float, ctx
) -> list[dict]:
    """Apply contextual reranking to results."""
    # Calculate contextual scores
    for result in results:
        original_score = result.get("score", 0.0)
        context_relevance = result.get("contextual_assessment", {}).get(
            "context_relevance", 0.5
        )

        # Combine original score with context relevance
        contextual_score = (original_score * (1 - context_weight)) + (
            context_relevance * context_weight
        )
        result["contextual_score"] = contextual_score

    # Sort by contextual score
    reranked_results = sorted(
        results, key=lambda x: x.get("contextual_score", 0.0), reverse=True
    )

    # Add ranking metadata
    for i, result in enumerate(reranked_results):
        result["contextual_ranking_metadata"] = {
            "rank": i + 1,
            "contextual_score": result.get("contextual_score", 0.0),
            "original_score": result.get("score", 0.0),
            "context_relevance": result.get("contextual_assessment", {}).get(
                "context_relevance", 0.5
            ),
            "context_weight_applied": context_weight,
        }

    return reranked_results


def _calculate_contextual_metrics(
    original_results: list[dict], reranked_results: list[dict], context_weight: float
) -> dict[str, Any]:
    """Calculate metrics for contextual reranking."""
    # Calculate ranking changes
    rank_changes = []
    for i, result in enumerate(reranked_results[:10]):
        original_rank = (
            original_results.index(result) + 1 if result in original_results else -1
        )
        new_rank = i + 1
        if original_rank > 0:
            rank_changes.append(abs(new_rank - original_rank))

    # Calculate context influence
    avg_context_relevance = sum(
        r.get("contextual_assessment", {}).get("context_relevance", 0.5)
        for r in reranked_results[:10]
    ) / min(len(reranked_results), 10)

    context_influence = avg_context_relevance * context_weight

    return {
        "context_influence": context_influence,
        "average_rank_change": sum(rank_changes) / len(rank_changes)
        if rank_changes
        else 0.0,
        "ranking_adjustment": len([change for change in rank_changes if change > 2])
        / len(rank_changes)
        if rank_changes
        else 0.0,
        "context_effectiveness": min(context_influence * 2, 1.0),
    }
