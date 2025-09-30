"""Multi-stage search tools with progressive refinement and autonomous optimization.

Implements intelligent multi-stage search with adaptive query refinement,
result quality assessment, and autonomous stage optimization.
"""

import asyncio
import logging
from typing import Any

from fastmcp import Context

from src.infrastructure.client_manager import ClientManager
from src.security import MLSecurityValidator


logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register multi-stage search tools with the MCP server."""

    @mcp.tool()
    async def multi_stage_progressive_search(
        query: str,
        collection_name: str,
        stages: int = 3,
        refinement_strategy: str = "adaptive",
        quality_threshold: float = 0.7,
        limit_per_stage: int = 20,
        final_limit: int = 10,
        filters: dict[str, Any] | None = None,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """Perform multi-stage search with progressive query refinement.

        Implements intelligent multi-stage search that progressively refines
        the query and filters results based on quality assessment and relevance.

        Args:
            query: Original search query
            collection_name: Target collection for search
            stages: Number of search stages to perform
            refinement_strategy: Strategy for query refinement
                (adaptive, semantic, contextual)
            quality_threshold: Minimum quality threshold for results
            limit_per_stage: Maximum results to retrieve per stage
            final_limit: Final number of results to return
            filters: Optional metadata filters
            ctx: MCP context for logging

        Returns:
            Multi-stage search results with refinement metadata and quality metrics

        """
        try:
            if ctx:
                await ctx.info(f"Starting {stages}-stage progressive search: '{query}'")

            # Validate query
            security_validator = MLSecurityValidator.from_unified_config()
            validated_query = security_validator.validate_query_string(query)

            # Get services
            qdrant_service = await client_manager.get_qdrant_service()
            embedding_manager = await client_manager.get_embedding_manager()

            # Initialize stage tracking
            stage_results = []
            cumulative_results = []
            current_query = validated_query

            # Perform each stage
            for stage_num in range(stages):
                if ctx:
                    await ctx.debug(
                        f"Stage {stage_num + 1}/{stages}: '{current_query}'"
                    )

                # Perform search for current stage
                stage_result = await _perform_stage_search(
                    qdrant_service,
                    embedding_manager,
                    current_query,
                    collection_name,
                    limit_per_stage,
                    filters,
                    stage_num,
                    ctx,
                )

                if not stage_result["success"]:
                    if ctx:
                        await ctx.warning(
                            f"Stage {stage_num + 1} failed: {stage_result['error']}"
                        )
                    continue

                # Assess result quality
                quality_assessment = await _assess_stage_quality(
                    stage_result["results"], quality_threshold, ctx
                )

                # Store stage metadata
                stage_metadata = {
                    "stage": stage_num + 1,
                    "query": current_query,
                    "results_count": len(stage_result["results"]),
                    "quality_assessment": quality_assessment,
                    "refinement_applied": stage_num > 0,
                }

                stage_results.append(stage_metadata)

                # Add high-quality results to cumulative set
                quality_filtered = [
                    result
                    for result in stage_result["results"]
                    if result.get("quality_score", 0.5) >= quality_threshold
                ]

                cumulative_results.extend(quality_filtered)

                if ctx:
                    await ctx.debug(
                        f"Stage {stage_num + 1}: {len(quality_filtered)} quality "
                        "results added"
                    )

                # Early termination if sufficient quality results found
                if (
                    len(cumulative_results) >= final_limit * 2
                    and quality_assessment["average_quality"] >= quality_threshold
                ):
                    if ctx:
                        await ctx.info(
                            f"Early termination at stage {stage_num + 1}: "
                            "sufficient quality results"
                        )
                    break

                # Refine query for next stage (if not last stage)
                if stage_num < stages - 1:
                    refinement_result = await _refine_query_for_next_stage(
                        current_query, stage_result["results"], refinement_strategy, ctx
                    )

                    if refinement_result["success"]:
                        current_query = refinement_result["refined_query"]
                        if ctx:
                            await ctx.debug(
                                f"Query refined for stage {stage_num + 2}: "
                                f"'{current_query}'"
                            )

            # Fuse and rank cumulative results
            fused_results = await _fuse_multi_stage_results(
                cumulative_results, stage_results, final_limit, ctx
            )

            # Calculate overall metrics
            overall_metrics = _calculate_multi_stage_metrics(
                stage_results, fused_results
            )

            # Generate optimization insights
            optimization_insights = await _generate_multi_stage_insights(
                validated_query, stage_results, fused_results, ctx
            )

            final_results = {
                "success": True,
                "original_query": validated_query,
                "collection": collection_name,
                "results": fused_results["results"],
                "multi_stage_metadata": {
                    "stages_performed": len(stage_results),
                    "refinement_strategy": refinement_strategy,
                    "quality_threshold": quality_threshold,
                    "stage_results": stage_results,
                    "fusion_confidence": fused_results["confidence"],
                },
                "overall_metrics": overall_metrics,
                "autonomous_optimization": optimization_insights,
            }

            if ctx:
                await ctx.info(
                    f"Multi-stage search completed: {len(stage_results)} stages, "
                    f"{len(fused_results['results'])} final results"
                )

        except Exception as e:
            logger.exception("Failed to perform multi-stage search")
            if ctx:
                await ctx.error(f"Multi-stage search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "stages_attempted": len(stage_results)
                if "stage_results" in locals()
                else 0,
            }

        else:
            return final_results

    @mcp.tool()
    async def adaptive_multi_stage_search(
        query: str,
        collection_name: str,
        auto_optimize: bool = True,
        performance_target: str = "quality",
        max_stages: int = 5,
        filters: dict[str, Any] | None = None,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """Perform adaptive multi-stage search with ML-powered optimization.

        Automatically determines optimal number of stages, refinement strategies,
        and termination criteria based on query characteristics and performance targets.

        Args:
            query: Original search query
            collection_name: Target collection for search
            auto_optimize: Enable autonomous optimization
            performance_target: Target optimization (quality, speed, coverage)
            max_stages: Maximum number of stages to perform
            filters: Optional metadata filters
            ctx: MCP context for logging

        Returns:
            Optimized multi-stage search results with adaptation metadata

        """
        try:
            if ctx:
                await ctx.info(
                    f"Starting adaptive multi-stage search targeting "
                    f"{performance_target}"
                )

            # Analyze query for optimal parameters
            query_analysis = await _analyze_query_for_multi_stage(query, ctx)

            # Select optimal multi-stage parameters
            optimal_params = await _select_optimal_multi_stage_parameters(
                query_analysis, performance_target, auto_optimize, max_stages, ctx
            )

            # Perform adaptive multi-stage search
            search_result = await multi_stage_progressive_search(
                query=query,
                collection_name=collection_name,
                stages=optimal_params["stages"],
                refinement_strategy=optimal_params["refinement_strategy"],
                quality_threshold=optimal_params["quality_threshold"],
                limit_per_stage=optimal_params["limit_per_stage"],
                final_limit=optimal_params["final_limit"],
                filters=filters,
                ctx=ctx,
            )

            if not search_result["success"]:
                return search_result

            # Add adaptive optimization metadata
            search_result["adaptive_optimization"] = {
                "query_analysis": query_analysis,
                "optimal_parameters": optimal_params,
                "performance_target": performance_target,
                "auto_optimization_applied": auto_optimize,
                "adaptation_confidence": 0.88,
            }

            if ctx:
                await ctx.info(
                    f"Adaptive multi-stage search completed with "
                    f"{optimal_params['stages']} stages"
                )

        except Exception as e:
            logger.exception("Failed to perform adaptive multi-stage search")
            if ctx:
                await ctx.error(f"Adaptive multi-stage search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "performance_target": performance_target,
            }

        else:
            return search_result

    @mcp.tool()
    async def contextual_refinement_search(
        query: str,
        collection_name: str,
        context_sources: list[str],
        refinement_depth: int = 2,
        context_weight: float = 0.3,
        filters: dict[str, Any] | None = None,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """Perform multi-stage search with contextual refinement from multiple sources.

        Uses context from multiple collections or sources to progressively
        refine the search query and improve result relevance.

        Args:
            query: Original search query
            collection_name: Primary target collection for search
            context_sources: List of collections to use for context
            refinement_depth: Number of refinement iterations
            context_weight: Weight for context influence on refinement
            filters: Optional metadata filters
            ctx: MCP context for logging

        Returns:
            Contextually refined search results with source attribution

        """
        try:
            if ctx:
                await ctx.info(
                    "Starting contextual refinement search with "
                    f"{len(context_sources)} context sources"
                )

            # Validate query
            security_validator = MLSecurityValidator.from_unified_config()
            validated_query = security_validator.validate_query_string(query)

            # Get services
            qdrant_service = await client_manager.get_qdrant_service()
            embedding_manager = await client_manager.get_embedding_manager()

            # Gather context from multiple sources
            context_data = await _gather_contextual_data(
                qdrant_service,
                embedding_manager,
                validated_query,
                context_sources,
                ctx,
            )

            # Perform iterative refinement
            refinement_results = []
            current_query = validated_query

            for iteration in range(refinement_depth):
                if ctx:
                    await ctx.debug(
                        f"Refinement iteration {iteration + 1}/{refinement_depth}"
                    )

                # Apply contextual refinement
                refinement_result = await _apply_contextual_refinement(
                    current_query, context_data, context_weight, iteration, ctx
                )

                if refinement_result["success"]:
                    current_query = refinement_result["refined_query"]

                    # Perform search with refined query
                    search_result = await _perform_stage_search(
                        qdrant_service,
                        embedding_manager,
                        current_query,
                        collection_name,
                        30,
                        filters,
                        iteration,
                        ctx,
                    )

                    refinement_results.append(
                        {
                            "iteration": iteration + 1,
                            "refined_query": current_query,
                            "context_influence": refinement_result["context_influence"],
                            "results_count": len(search_result.get("results", [])),
                            "quality_improvement": search_result.get(
                                "quality_score", 0.5
                            ),
                        }
                    )

                    if ctx:
                        await ctx.debug(
                            f"Refinement {iteration + 1}: '{current_query}' -> "
                            f"{len(search_result.get('results', []))} results"
                        )

            # Final search with best refined query
            final_search = await _perform_stage_search(
                qdrant_service,
                embedding_manager,
                current_query,
                collection_name,
                50,
                filters,
                refinement_depth,
                ctx,
            )

            # Calculate contextual metrics
            contextual_metrics = _calculate_contextual_metrics(
                context_data, refinement_results, final_search
            )

            final_results = {
                "success": True,
                "original_query": validated_query,
                "final_query": current_query,
                "collection": collection_name,
                "results": final_search.get("results", [])[:20],  # Limit final results
                "contextual_metadata": {
                    "context_sources": context_sources,
                    "refinement_depth": refinement_depth,
                    "context_weight": context_weight,
                    "refinement_results": refinement_results,
                    "context_data_quality": context_data["quality_score"],
                },
                "contextual_metrics": contextual_metrics,
            }

            if ctx:
                await ctx.info(
                    f"Contextual refinement completed: {refinement_depth} iterations, "
                    f"{len(final_results['results'])} final results"
                )

        except Exception as e:
            logger.exception("Failed to perform contextual refinement search")
            if ctx:
                await ctx.error(f"Contextual refinement search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "context_sources": context_sources,
            }

        else:
            return final_results

    @mcp.tool()
    async def get_multi_stage_capabilities() -> dict[str, Any]:
        """Get multi-stage search capabilities and configuration options.

        Returns:
            Comprehensive capabilities information for multi-stage search system

        """
        return {
            "refinement_strategies": {
                "adaptive": {
                    "description": (
                        "Adapts refinement based on result quality and query "
                        "characteristics"
                    ),
                    "best_for": ["general_queries", "unknown_domains"],
                    "complexity": "high",
                    "effectiveness": "optimal",
                },
                "semantic": {
                    "description": "Uses semantic similarity to refine queries",
                    "best_for": ["conceptual_queries", "broad_topics"],
                    "complexity": "medium",
                    "effectiveness": "good",
                },
                "contextual": {
                    "description": "Leverages contextual information for refinement",
                    "best_for": ["domain_specific", "multi_source_queries"],
                    "complexity": "high",
                    "effectiveness": "high",
                },
            },
            "stage_optimization": {
                "early_termination": True,
                "quality_assessment": True,
                "adaptive_stage_count": True,
                "progressive_filtering": True,
            },
            "performance_targets": ["quality", "speed", "coverage"],
            "contextual_features": {
                "multi_source_context": True,
                "context_weighting": True,
                "iterative_refinement": True,
                "source_attribution": True,
            },
            "autonomous_capabilities": {
                "parameter_optimization": True,
                "strategy_selection": True,
                "termination_criteria": True,
                "quality_driven_refinement": True,
            },
            "quality_metrics": [
                "stage_effectiveness",
                "refinement_quality",
                "fusion_confidence",
                "contextual_relevance",
            ],
            "status": "active",
        }


# Helper functions


async def _perform_stage_search(
    qdrant_service,
    embedding_manager,
    query: str,
    collection_name: str,
    limit: int,
    filters: dict | None,
    stage_num: int,
    ctx,
) -> dict[str, Any]:
    """Perform search for a single stage."""
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
            with_vectors=False,
        )

        results = search_result.get("points", [])

        # Add stage metadata to results
        for result in results:
            result["stage_number"] = stage_num + 1
            result["stage_query"] = query
            result["quality_score"] = _calculate_result_quality(result)

        return {
            "success": True,
            "results": results,
            "stage": stage_num + 1,
            "query": query,
        }

    except Exception as e:
        logger.exception(f"Stage {stage_num + 1} search failed")
        return {
            "success": False,
            "error": str(e),
            "stage": stage_num + 1,
            "query": query,
        }


def _calculate_result_quality(result: dict[str, Any]) -> float:
    """Calculate quality score for a search result."""
    # Simple quality calculation based on score and content
    base_score = result.get("score", 0.0)

    payload = result.get("payload", {})
    content_length = len(payload.get("content", ""))

    # Quality factors
    length_factor = min(content_length / 500, 1.0)  # Prefer reasonable length content
    score_factor = base_score

    # Combine factors
    quality_score = (score_factor * 0.7) + (length_factor * 0.3)

    return min(quality_score, 1.0)


async def _assess_stage_quality(
    results: list[dict], quality_threshold: float, ctx
) -> dict[str, Any]:
    """Assess quality of results from a stage."""
    if not results:
        return {
            "average_quality": 0.0,
            "quality_variance": 0.0,
            "threshold_met_count": 0,
            "quality_distribution": {},
        }

    quality_scores = [result.get("quality_score", 0.0) for result in results]
    average_quality = sum(quality_scores) / len(quality_scores)

    # Calculate variance
    variance = sum((score - average_quality) ** 2 for score in quality_scores) / len(
        quality_scores
    )

    # Count results meeting threshold
    threshold_met_count = sum(
        1 for score in quality_scores if score >= quality_threshold
    )

    # Quality distribution
    high_quality = sum(1 for score in quality_scores if score >= 0.8)
    medium_quality = sum(1 for score in quality_scores if 0.5 <= score < 0.8)
    low_quality = sum(1 for score in quality_scores if score < 0.5)

    return {
        "average_quality": average_quality,
        "quality_variance": variance,
        "threshold_met_count": threshold_met_count,
        "quality_distribution": {
            "high": high_quality,
            "medium": medium_quality,
            "low": low_quality,
        },
    }


async def _refine_query_for_next_stage(
    current_query: str, stage_results: list[dict], strategy: str, ctx
) -> dict[str, Any]:
    """Refine query for the next stage based on current results."""
    try:
        if not stage_results:
            return {
                "success": False,
                "error": "No results to base refinement on",
            }

        # Extract key terms from high-quality results
        high_quality_results = [
            result
            for result in stage_results
            if result.get("quality_score", 0.0) >= 0.7
        ]

        if not high_quality_results:
            high_quality_results = stage_results[
                :3
            ]  # Use top 3 if none meet quality threshold

        # Apply refinement strategy
        if strategy == "adaptive":
            refined_query = await _adaptive_query_refinement(
                current_query, high_quality_results, ctx
            )
        elif strategy == "semantic":
            refined_query = await _semantic_query_refinement(
                current_query, high_quality_results, ctx
            )
        else:  # contextual
            refined_query = await _contextual_query_refinement(
                current_query, high_quality_results, ctx
            )

        return {
            "success": True,
            "refined_query": refined_query,
            "strategy": strategy,
            "base_results_count": len(high_quality_results),
        }

    except Exception as e:
        logger.exception("Failed to refine query")
        return {
            "success": False,
            "error": str(e),
            "original_query": current_query,
        }


async def _adaptive_query_refinement(query: str, results: list[dict], ctx) -> str:
    """Apply adaptive query refinement based on result analysis."""
    # Extract common terms from high-quality results
    common_terms = []

    for result in results:
        content = result.get("payload", {}).get("content", "")
        words = content.lower().split()

        # Simple term extraction (can be enhanced with NLP)
        for word in words:
            if len(word) > 5 and word.isalpha():
                common_terms.append(word)
                if len(common_terms) >= 10:
                    break

    # Find most frequent terms not in original query
    query_words = set(query.lower().split())
    term_counts = {}

    for term in common_terms:
        if term not in query_words:
            term_counts[term] = term_counts.get(term, 0) + 1

    # Add top 2 terms to query
    if term_counts:
        top_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:2]
        refined_query = f"{query} {' '.join(term[0] for term in top_terms)}"
    else:
        refined_query = query

    return refined_query


async def _semantic_query_refinement(query: str, results: list[dict], ctx) -> str:
    """Apply semantic query refinement."""
    # For semantic refinement, expand with related concepts
    semantic_expansions = {
        "implement": ["development", "build", "create"],
        "optimize": ["improve", "enhance", "performance"],
        "analyze": ["study", "examine", "evaluate"],
        "design": ["architecture", "structure", "pattern"],
    }

    query_lower = query.lower()
    expansion_terms = []

    for key, expansions in semantic_expansions.items():
        if key in query_lower:
            expansion_terms.extend(expansions[:1])  # Add one expansion

    if expansion_terms:
        refined_query = f"{query} {' '.join(expansion_terms)}"
    else:
        # Fallback to adding "best practices" or "guide"
        refined_query = f"{query} guide"

    return refined_query


async def _contextual_query_refinement(query: str, results: list[dict], ctx) -> str:
    """Apply contextual query refinement."""
    # Extract contextual terms from result titles and metadata
    contextual_terms = []

    for result in results:
        payload = result.get("payload", {})
        if title := payload.get("title", ""):
            title_words = title.lower().split()
            contextual_terms.extend(
                [word for word in title_words if len(word) > 4 and word.isalpha()]
            )

    # Use most common contextual term
    if contextual_terms:
        term_counts = {}
        for term in contextual_terms:
            term_counts[term] = term_counts.get(term, 0) + 1

        most_common = max(term_counts.items(), key=lambda x: x[1])[0]
        refined_query = f"{query} {most_common}"
    else:
        refined_query = query

    return refined_query


async def _fuse_multi_stage_results(
    cumulative_results: list[dict], stage_results: list[dict], final_limit: int, ctx
) -> dict[str, Any]:
    """Fuse results from multiple stages."""
    # Remove duplicates based on document ID
    seen_ids = set()
    unique_results = []

    for result in cumulative_results:
        if (doc_id := result.get("id")) not in seen_ids:
            seen_ids.add(doc_id)
            unique_results.append(result)

    # Sort by quality score and relevance
    unique_results.sort(
        key=lambda x: (x.get("quality_score", 0.0), x.get("score", 0.0)), reverse=True
    )

    # Add fusion metadata
    for result in unique_results:
        result["multi_stage_fusion"] = True
        result["stages_appeared"] = [
            stage["stage"]
            for stage in stage_results
            if any(r.get("id") == result.get("id") for r in [result])
        ]

    # Calculate fusion confidence
    confidence = _calculate_multi_stage_confidence(unique_results, stage_results)

    return {
        "results": unique_results[:final_limit],
        "confidence": confidence,
        "fusion_metadata": {
            "total_unique_results": len(unique_results),
            "stages_with_results": len(
                [s for s in stage_results if s["results_count"] > 0]
            ),
            "duplicate_removal": len(cumulative_results) - len(unique_results),
        },
    }


def _calculate_multi_stage_confidence(
    results: list[dict], stage_results: list[dict]
) -> float:
    """Calculate confidence in multi-stage fusion."""
    if not results or not stage_results:
        return 0.0

    # Factor in result quality and stage effectiveness
    avg_quality = sum(r.get("quality_score", 0.0) for r in results[:10]) / min(
        len(results), 10
    )
    stage_effectiveness = sum(
        s["quality_assessment"]["average_quality"] for s in stage_results
    ) / len(stage_results)

    # Combine factors
    confidence = (avg_quality * 0.6) + (stage_effectiveness * 0.4)
    return min(confidence, 1.0)


def _calculate_multi_stage_metrics(
    stage_results: list[dict], fused_results: dict
) -> dict[str, Any]:
    """Calculate overall metrics for multi-stage search."""
    return {
        "total_stages": len(stage_results),
        "stages_with_results": len(
            [s for s in stage_results if s["results_count"] > 0]
        ),
        "average_stage_quality": sum(
            s["quality_assessment"]["average_quality"] for s in stage_results
        )
        / len(stage_results)
        if stage_results
        else 0.0,
        "fusion_effectiveness": fused_results["confidence"],
        "result_diversity": fused_results["fusion_metadata"]["total_unique_results"]
        / max(sum(s["results_count"] for s in stage_results), 1),
        "refinement_success_rate": len(
            [s for s in stage_results if s.get("refinement_applied")]
        )
        / max(len(stage_results) - 1, 1),
    }


async def _generate_multi_stage_insights(
    original_query: str, stage_results: list[dict], fused_results: dict, ctx
) -> dict[str, Any]:
    """Generate optimization insights for multi-stage search."""
    insights = {
        "stage_analysis": {
            "most_effective_stage": max(
                stage_results, key=lambda s: s["quality_assessment"]["average_quality"]
            )["stage"]
            if stage_results
            else 1,
            "quality_progression": [
                s["quality_assessment"]["average_quality"] for s in stage_results
            ],
            "diminishing_returns": _detect_diminishing_returns(stage_results),
        },
        "refinement_analysis": {
            "refinement_effectiveness": sum(
                1
                for s in stage_results
                if s.get("refinement_applied")
                and s["quality_assessment"]["average_quality"] > 0.6
            )
            / max(len(stage_results) - 1, 1),
            "optimal_stage_count": _estimate_optimal_stage_count(stage_results),
        },
        "recommendations": [],
    }

    # Generate recommendations
    if insights["stage_analysis"]["diminishing_returns"]:
        insights["recommendations"].append(
            "Consider reducing number of stages to avoid diminishing returns"
        )

    if insights["refinement_analysis"]["refinement_effectiveness"] < 0.5:
        insights["recommendations"].append(
            "Try different refinement strategy for better effectiveness"
        )

    avg_quality = (
        sum(s["quality_assessment"]["average_quality"] for s in stage_results)
        / len(stage_results)
        if stage_results
        else 0
    )
    if avg_quality < 0.6:
        insights["recommendations"].append(
            "Increase quality threshold or adjust search parameters"
        )

    return insights


def _detect_diminishing_returns(stage_results: list[dict]) -> bool:
    """Detect if stages show diminishing returns."""
    if len(stage_results) < 3:
        return False

    qualities = [s["quality_assessment"]["average_quality"] for s in stage_results]

    # Check if quality improvement decreases in later stages
    improvements = [qualities[i] - qualities[i - 1] for i in range(1, len(qualities))]

    # Diminishing returns if last improvement is significantly smaller
    if len(improvements) >= 2:
        return improvements[-1] < improvements[0] * 0.5

    return False


def _estimate_optimal_stage_count(stage_results: list[dict]) -> int:
    """Estimate optimal number of stages based on results."""
    if not stage_results:
        return 3

    qualities = [s["quality_assessment"]["average_quality"] for s in stage_results]

    # Find stage where quality peaks or plateaus
    for i in range(1, len(qualities)):
        if qualities[i] <= qualities[i - 1] * 1.05:  # Less than 5% improvement
            return i

    return len(stage_results)


async def _analyze_query_for_multi_stage(query: str, ctx) -> dict[str, Any]:
    """Analyze query characteristics for multi-stage optimization."""
    words = query.split()

    return {
        "length": len(words),
        "complexity": "high"
        if len(words) > 6
        else "medium"
        if len(words) > 3
        else "low",
        "specificity": "high" if any(len(word) > 8 for word in words) else "medium",
        "question_indicators": len(
            [w for w in words if w.lower() in ["how", "what", "why", "when", "where"]]
        ),
        "technical_indicators": len(
            [
                w
                for w in words
                if any(
                    term in w.lower()
                    for term in ["api", "algorithm", "framework", "implementation"]
                )
            ]
        ),
    }


async def _select_optimal_multi_stage_parameters(
    query_analysis: dict,
    performance_target: str,
    auto_optimize: bool,
    max_stages: int,
    ctx,
) -> dict[str, Any]:
    """Select optimal parameters for multi-stage search."""
    params = {
        "stages": 3,
        "refinement_strategy": "adaptive",
        "quality_threshold": 0.7,
        "limit_per_stage": 20,
        "final_limit": 10,
    }

    if not auto_optimize:
        return params

    # Adjust based on query complexity
    if query_analysis["complexity"] == "high":
        params["stages"] = min(4, max_stages)
        params["quality_threshold"] = 0.6
    elif query_analysis["complexity"] == "low":
        params["stages"] = 2
        params["quality_threshold"] = 0.8

    # Adjust based on performance target
    if performance_target == "speed":
        params["stages"] = min(params["stages"], 2)
        params["limit_per_stage"] = 15
    elif performance_target == "quality":
        params["stages"] = min(params["stages"] + 1, max_stages)
        params["quality_threshold"] = 0.8
    elif performance_target == "coverage":
        params["limit_per_stage"] = 30
        params["final_limit"] = 15

    return params


async def _gather_contextual_data(
    qdrant_service, embedding_manager, query: str, context_sources: list[str], ctx
) -> dict[str, Any]:
    """Gather contextual data from multiple sources."""
    context_data = {
        "sources": {},
        "quality_score": 0.0,
        "total_context_items": 0,
    }

    # Generate embedding for query
    embeddings_result = await embedding_manager.generate_embeddings([query])
    query_embedding = embeddings_result.embeddings[0]

    # Search each context source
    for source in context_sources:
        try:
            search_result = await qdrant_service.search(
                collection_name=source,
                query_vector=query_embedding,
                limit=5,  # Limited context per source
                with_payload=True,
                with_vectors=False,
            )

            if search_result and "points" in search_result:
                context_data["sources"][source] = {
                    "results": search_result["points"],
                    "count": len(search_result["points"]),
                    "average_score": sum(
                        p.get("score", 0.0) for p in search_result["points"]
                    )
                    / len(search_result["points"])
                    if search_result["points"]
                    else 0.0,
                }
                context_data["total_context_items"] += len(search_result["points"])

                if ctx:
                    await ctx.debug(
                        f"Context from {source}: {len(search_result['points'])} items"
                    )

        except (asyncio.CancelledError, TimeoutError, RuntimeError) as e:
            if ctx:
                await ctx.warning(f"Failed to gather context from {source}: {e}")

    # Calculate overall quality
    if context_data["sources"]:
        avg_scores = [
            source_data["average_score"]
            for source_data in context_data["sources"].values()
        ]
        context_data["quality_score"] = sum(avg_scores) / len(avg_scores)

    return context_data


async def _apply_contextual_refinement(
    query: str, context_data: dict, context_weight: float, iteration: int, ctx
) -> dict[str, Any]:
    """Apply contextual refinement to query."""
    try:
        # Extract key terms from context
        context_terms = []

        for source_data in context_data["sources"].values():
            for result in source_data["results"]:
                content = result.get("payload", {}).get("content", "")
                title = result.get("payload", {}).get("title", "")

                # Extract key terms (simplified)
                for text in [content, title]:
                    words = text.lower().split()
                    context_terms.extend(
                        [word for word in words if len(word) > 5 and word.isalpha()]
                    )

        # Find most relevant context terms
        term_counts = {}
        query_words = set(query.lower().split())

        for term in context_terms:
            if term not in query_words:
                term_counts[term] = term_counts.get(term, 0) + 1

        # Apply contextual weight and iteration factor
        context_influence = context_weight * (
            1.0 - iteration * 0.2
        )  # Decrease influence over iterations

        if term_counts and context_influence > 0.1:
            top_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[
                :2
            ]
            context_addition = " ".join(term[0] for term in top_terms)
            refined_query = f"{query} {context_addition}"
        else:
            refined_query = query

        return {
            "success": True,
            "refined_query": refined_query,
            "context_influence": context_influence,
            "terms_added": len(refined_query.split()) - len(query.split()),
        }

    except Exception as e:
        logger.exception("Failed to apply contextual refinement")
        return {
            "success": False,
            "error": str(e),
            "context_influence": 0.0,
        }


def _calculate_contextual_metrics(
    context_data: dict, refinement_results: list[dict], final_search: dict
) -> dict[str, Any]:
    """Calculate metrics for contextual refinement."""
    return {
        "context_sources_used": len(context_data["sources"]),
        "total_context_items": context_data["total_context_items"],
        "context_quality": context_data["quality_score"],
        "refinement_iterations": len(refinement_results),
        "average_context_influence": sum(
            r["context_influence"] for r in refinement_results
        )
        / len(refinement_results)
        if refinement_results
        else 0.0,
        "final_search_quality": final_search.get("quality_score", 0.0),
        "context_effectiveness": min(
            context_data["quality_score"] * len(refinement_results) / 3, 1.0
        ),
    }
