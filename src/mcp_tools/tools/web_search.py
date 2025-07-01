"""Web search orchestration tools implementing I5 research findings.

Provides autonomous web search orchestration with multi-provider support,
intelligent result fusion, and adaptive search strategy optimization.
"""

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
    """Register web search orchestration tools with the MCP server."""

    @mcp.tool()
    async def autonomous_web_search(
        query: str,
        providers: list[str] | None = None,
        max_results: int = 20,
        fusion_strategy: str = "intelligent",
        quality_threshold: float = 0.7,
        auto_expand: bool = True,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """Perform autonomous web search with multi-provider orchestration.

        Implements I5 research findings for intelligent web search orchestration
        with autonomous provider selection, result fusion, and quality optimization.

        Args:
            query: Search query text
            providers: Optional list of search providers to use
            max_results: Maximum number of results to return
            fusion_strategy: Strategy for result fusion (intelligent, weighted, ranked)
            quality_threshold: Minimum quality threshold for results
            auto_expand: Enable autonomous query expansion
            ctx: MCP context for logging

        Returns:
            Autonomous web search results with orchestration metadata
        """
        try:
            if ctx:
                await ctx.info(
                    f"Starting autonomous web search: '{query}' with fusion strategy '{fusion_strategy}'"
                )

            # Validate query
            security_validator = SecurityValidator.from_unified_config()
            validated_query = security_validator.validate_query_string(query)

            # Autonomous provider selection if not specified
            if not providers:
                provider_analysis = await _analyze_optimal_providers(
                    validated_query, ctx
                )
                providers = provider_analysis["recommended_providers"]

                if ctx:
                    await ctx.debug(f"Auto-selected providers: {', '.join(providers)}")

            # Autonomous query expansion if enabled
            search_queries = [validated_query]
            if auto_expand:
                expansion_result = await _autonomous_query_expansion(
                    validated_query, ctx
                )
                if expansion_result["success"]:
                    search_queries.extend(
                        expansion_result["expanded_queries"][:2]
                    )  # Limit expansions

                    if ctx:
                        await ctx.debug(f"Expanded to {len(search_queries)} queries")

            # Perform multi-provider search
            provider_results = {}
            total_raw_results = 0

            for provider in providers:
                provider_result = await _perform_provider_search(
                    provider, search_queries, max_results, ctx
                )

                if provider_result["success"]:
                    provider_results[provider] = provider_result
                    total_raw_results += sum(
                        len(query_results.get("results", []))
                        for query_results in provider_result["query_results"].values()
                    )

                    if ctx:
                        await ctx.debug(
                            f"Provider {provider}: {sum(len(qr.get('results', [])) for qr in provider_result['query_results'].values())} results"
                        )

            if not provider_results:
                return {
                    "success": False,
                    "error": "No providers returned results",
                    "query": validated_query,
                    "providers_attempted": providers,
                }

            # Apply intelligent result fusion
            fused_results = await _apply_intelligent_fusion(
                provider_results, fusion_strategy, quality_threshold, ctx
            )

            # Calculate orchestration metrics
            orchestration_metrics = _calculate_orchestration_metrics(
                provider_results, fused_results, total_raw_results
            )

            # Generate autonomous optimization insights
            optimization_insights = await _generate_web_search_insights(
                validated_query, provider_results, fused_results, ctx
            )

            final_results = {
                "success": True,
                "query": validated_query,
                "expanded_queries": search_queries[1:]
                if len(search_queries) > 1
                else [],
                "results": fused_results["results"][:max_results],
                "orchestration_metadata": {
                    "providers_used": list(provider_results.keys()),
                    "fusion_strategy": fusion_strategy,
                    "quality_threshold": quality_threshold,
                    "auto_expansion_applied": auto_expand and len(search_queries) > 1,
                    "fusion_confidence": fused_results["confidence"],
                },
                "orchestration_metrics": orchestration_metrics,
                "autonomous_optimization": optimization_insights,
                "i5_research_features": {
                    "multi_provider_orchestration": True,
                    "intelligent_fusion": True,
                    "autonomous_optimization": True,
                    "quality_driven_filtering": True,
                },
            }

            if ctx:
                await ctx.info(
                    f"Autonomous web search completed: {len(fused_results['results'])} results from {len(provider_results)} providers"
                )

            return final_results

        except Exception as e:
            logger.exception("Failed to perform autonomous web search")
            if ctx:
                await ctx.error(f"Autonomous web search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "fusion_strategy": fusion_strategy,
            }

    @mcp.tool()
    async def adaptive_web_search_orchestration(
        query: str,
        search_intent: str = "general",
        performance_target: str = "balanced",
        max_iterations: int = 3,
        learning_enabled: bool = True,
        ctx: Context = None,
    ) -> dict[str, Any]:
        """Perform adaptive web search with ML-powered orchestration optimization.

        Implements advanced adaptive orchestration that learns from search patterns
        and automatically optimizes provider selection and fusion strategies.

        Args:
            query: Search query text
            search_intent: Intent classification (research, factual, procedural, comparative)
            performance_target: Target optimization (speed, quality, coverage, balanced)
            max_iterations: Maximum optimization iterations
            learning_enabled: Enable ML-powered learning and adaptation
            ctx: MCP context for logging

        Returns:
            Adaptively optimized web search results with learning metadata
        """
        try:
            if ctx:
                await ctx.info(
                    f"Starting adaptive web search orchestration: intent='{search_intent}', target='{performance_target}'"
                )

            # Validate query
            security_validator = SecurityValidator.from_unified_config()
            validated_query = security_validator.validate_query_string(query)

            # Analyze query characteristics for adaptive optimization
            query_analysis = await _analyze_query_characteristics(
                validated_query, search_intent, ctx
            )

            # Adaptive parameter optimization
            best_result = None
            iteration_results = []

            for iteration in range(max_iterations):
                if ctx:
                    await ctx.debug(
                        f"Adaptive iteration {iteration + 1}/{max_iterations}"
                    )

                # Select optimal parameters for this iteration
                iteration_params = await _select_adaptive_parameters(
                    query_analysis, performance_target, iteration, best_result, ctx
                )

                # Perform web search with current parameters
                search_result = await autonomous_web_search(
                    query=validated_query,
                    providers=iteration_params["providers"],
                    max_results=iteration_params["max_results"],
                    fusion_strategy=iteration_params["fusion_strategy"],
                    quality_threshold=iteration_params["quality_threshold"],
                    auto_expand=iteration_params["auto_expand"],
                    ctx=ctx,
                )

                if search_result["success"]:
                    # Evaluate iteration performance
                    performance_score = await _evaluate_iteration_performance(
                        search_result, performance_target, query_analysis, ctx
                    )

                    iteration_results.append(
                        {
                            "iteration": iteration + 1,
                            "parameters": iteration_params,
                            "performance_score": performance_score,
                            "results_count": len(search_result["results"]),
                            "orchestration_confidence": search_result[
                                "orchestration_metadata"
                            ]["fusion_confidence"],
                        }
                    )

                    # Update best result if this iteration performed better
                    if (
                        best_result is None
                        or performance_score > best_result["performance_score"]
                    ):
                        best_result = search_result.copy()
                        best_result["performance_score"] = performance_score
                        best_result["iteration_metadata"] = {
                            "iteration": iteration + 1,
                            "parameters_used": iteration_params,
                        }

                        if ctx:
                            await ctx.debug(
                                f"New best result from iteration {iteration + 1}: score={performance_score:.3f}"
                            )

                    # Early termination if performance target met
                    if performance_score >= 0.9:  # High performance threshold
                        if ctx:
                            await ctx.info(
                                f"Performance target achieved in iteration {iteration + 1}"
                            )
                        break

            if not best_result:
                return {
                    "success": False,
                    "error": "All adaptive iterations failed",
                    "query": validated_query,
                    "iterations_attempted": len(iteration_results),
                }

            # Apply learning if enabled
            learning_insights = {}
            if learning_enabled:
                learning_insights = await _apply_adaptive_learning(
                    query_analysis, iteration_results, best_result, ctx
                )

            # Add adaptive metadata
            best_result["adaptive_optimization"] = {
                "query_analysis": query_analysis,
                "search_intent": search_intent,
                "performance_target": performance_target,
                "iterations_performed": len(iteration_results),
                "iteration_results": iteration_results,
                "best_iteration": best_result.get("iteration_metadata", {}),
                "learning_applied": learning_enabled,
                "learning_insights": learning_insights,
            }

            if ctx:
                await ctx.info(
                    f"Adaptive orchestration completed in {len(iteration_results)} iterations with score {best_result.get('performance_score', 0.0):.3f}"
                )

            return best_result

        except Exception as e:
            logger.exception("Failed to perform adaptive web search orchestration")
            if ctx:
                await ctx.error(f"Adaptive web search orchestration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "search_intent": search_intent,
            }

    @mcp.tool()
    async def multi_provider_result_synthesis(
        query: str,
        provider_preferences: dict[str, float] | None = None,
        synthesis_depth: str = "comprehensive",
        deduplication_strategy: str = "semantic",
        ctx: Context = None,
    ) -> dict[str, Any]:
        """Synthesize results from multiple web search providers with intelligent deduplication.

        Implements advanced result synthesis with semantic deduplication,
        quality correlation, and intelligent provider weighting.

        Args:
            query: Search query text
            provider_preferences: Optional provider weight preferences
            synthesis_depth: Depth of synthesis (basic, standard, comprehensive)
            deduplication_strategy: Strategy for removing duplicates (url, semantic, hybrid)
            ctx: MCP context for logging

        Returns:
            Synthesized results with deduplication and quality correlation metadata
        """
        try:
            if ctx:
                await ctx.info(
                    f"Starting multi-provider result synthesis with {synthesis_depth} depth"
                )

            # Validate query
            security_validator = SecurityValidator.from_unified_config()
            validated_query = security_validator.validate_query_string(query)

            # Get all available providers
            available_providers = await _get_available_providers(ctx)

            # Gather results from all providers
            provider_results = {}
            for provider in available_providers:
                provider_result = await _perform_provider_search(
                    provider,
                    [validated_query],
                    30,
                    ctx,  # Get more results for synthesis
                )

                if provider_result["success"]:
                    provider_results[provider] = provider_result

                    if ctx:
                        await ctx.debug(f"Gathered results from {provider}")

            if not provider_results:
                return {
                    "success": False,
                    "error": "No providers returned results for synthesis",
                    "query": validated_query,
                }

            # Apply intelligent deduplication
            deduplicated_results = await _apply_intelligent_deduplication(
                provider_results, deduplication_strategy, ctx
            )

            # Apply result synthesis based on depth
            synthesized_results = await _apply_result_synthesis(
                deduplicated_results, synthesis_depth, provider_preferences, ctx
            )

            # Calculate synthesis metrics
            synthesis_metrics = _calculate_synthesis_metrics(
                provider_results, deduplicated_results, synthesized_results
            )

            final_results = {
                "success": True,
                "query": validated_query,
                "results": synthesized_results["results"],
                "synthesis_metadata": {
                    "providers_synthesized": len(provider_results),
                    "synthesis_depth": synthesis_depth,
                    "deduplication_strategy": deduplication_strategy,
                    "provider_preferences": provider_preferences or {},
                    "synthesis_confidence": synthesized_results["confidence"],
                },
                "synthesis_metrics": synthesis_metrics,
                "deduplication_details": deduplicated_results["metadata"],
            }

            if ctx:
                await ctx.info(
                    f"Synthesis completed: {len(synthesized_results['results'])} results from {len(provider_results)} providers"
                )

            return final_results

        except Exception as e:
            logger.exception("Failed to perform multi-provider result synthesis")
            if ctx:
                await ctx.error(f"Multi-provider result synthesis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "synthesis_depth": synthesis_depth,
            }

    @mcp.tool()
    async def get_web_search_capabilities() -> dict[str, Any]:
        """Get web search orchestration capabilities and configuration options.

        Returns:
            Comprehensive capabilities information for web search orchestration system
        """
        return {
            "search_providers": {
                "google": {
                    "type": "traditional_search",
                    "strengths": ["general_queries", "comprehensive_coverage"],
                    "rate_limits": "1000_requests_per_day",
                    "quality": "high",
                },
                "bing": {
                    "type": "traditional_search",
                    "strengths": ["enterprise_content", "news_results"],
                    "rate_limits": "1000_requests_per_day",
                    "quality": "high",
                },
                "duckduckgo": {
                    "type": "privacy_focused",
                    "strengths": ["privacy", "unbiased_results"],
                    "rate_limits": "unlimited",
                    "quality": "medium",
                },
                "searx": {
                    "type": "meta_search",
                    "strengths": ["aggregation", "open_source"],
                    "rate_limits": "depends_on_instance",
                    "quality": "variable",
                },
            },
            "fusion_strategies": {
                "intelligent": {
                    "description": "ML-powered fusion with quality assessment",
                    "best_for": ["complex_queries", "research_tasks"],
                    "complexity": "high",
                    "effectiveness": "optimal",
                },
                "weighted": {
                    "description": "Provider-weighted result combination",
                    "best_for": ["known_provider_preferences", "consistent_results"],
                    "complexity": "medium",
                    "effectiveness": "good",
                },
                "ranked": {
                    "description": "Rank-based fusion with position weighting",
                    "best_for": ["authoritative_ranking", "simple_queries"],
                    "complexity": "low",
                    "effectiveness": "medium",
                },
            },
            "adaptive_features": {
                "provider_optimization": True,
                "query_expansion": True,
                "performance_targeting": True,
                "learning_adaptation": True,
            },
            "synthesis_capabilities": {
                "semantic_deduplication": True,
                "quality_correlation": True,
                "provider_weighting": True,
                "comprehensive_synthesis": True,
            },
            "autonomous_capabilities": {
                "provider_selection": True,
                "strategy_optimization": True,
                "quality_assessment": True,
                "performance_correlation": True,
            },
            "performance_targets": ["speed", "quality", "coverage", "balanced"],
            "search_intents": [
                "research",
                "factual",
                "procedural",
                "comparative",
                "general",
            ],
            "i5_research_implementation": {
                "multi_provider_orchestration": True,
                "autonomous_optimization": True,
                "intelligent_fusion": True,
                "adaptive_learning": True,
            },
            "status": "active",
        }


# Helper functions


async def _analyze_optimal_providers(query: str, ctx) -> dict[str, Any]:
    """Analyze query to select optimal search providers."""
    query_lower = query.lower()

    # Provider selection heuristics
    recommended = ["google"]  # Default reliable provider

    # Add specialized providers based on query characteristics
    if any(term in query_lower for term in ["news", "current", "recent", "latest"]):
        recommended.append("bing")  # Good for news

    if any(term in query_lower for term in ["privacy", "anonymous", "secure"]):
        recommended.append("duckduckgo")  # Privacy-focused

    if len(query.split()) > 5:  # Complex queries
        recommended.append("searx")  # Meta-search aggregation

    # Limit to 3 providers for performance
    recommended = recommended[:3]

    return {
        "recommended_providers": recommended,
        "selection_reasoning": f"Selected {len(recommended)} providers based on query characteristics",
        "query_complexity": "high"
        if len(query.split()) > 5
        else "medium"
        if len(query.split()) > 2
        else "low",
    }


async def _autonomous_query_expansion(query: str, ctx) -> dict[str, Any]:
    """Perform autonomous query expansion for better coverage."""
    try:
        query_words = query.split()
        expanded_queries = []

        # Simple expansion strategies
        if len(query_words) <= 3:
            # Add context for short queries
            expanded_queries.append(f"{query} guide")
            expanded_queries.append(f"{query} tutorial")
        # Create variations for longer queries
        elif "how to" in query.lower():
            expanded_queries.append(query.replace("how to", "steps to"))
        elif "what is" in query.lower():
            expanded_queries.append(query.replace("what is", "definition of"))

        # Add domain-specific expansions
        if any(
            term in query.lower() for term in ["programming", "code", "development"]
        ):
            expanded_queries.append(f"{query} example")

        return {
            "success": True,
            "expanded_queries": expanded_queries[:3],  # Limit expansions
            "expansion_strategy": "autonomous_heuristic",
        }

    except Exception as e:
        logger.exception("Failed to perform autonomous query expansion")
        return {
            "success": False,
            "error": str(e),
        }


async def _perform_provider_search(
    provider: str, queries: list[str], max_results: int, ctx
) -> dict[str, Any]:
    """Perform search using a specific provider."""
    try:
        # Mock provider search implementation
        # In real implementation, this would use actual provider APIs

        query_results = {}

        for query in queries:
            # Simulate provider-specific results
            mock_results = _generate_mock_provider_results(
                provider, query, max_results // len(queries)
            )

            query_results[query] = {
                "results": mock_results,
                "provider": provider,
                "query": query,
                "results_count": len(mock_results),
            }

        return {
            "success": True,
            "provider": provider,
            "query_results": query_results,
            "total_results": sum(len(qr["results"]) for qr in query_results.values()),
        }

    except Exception as e:
        logger.exception(f"Provider {provider} search failed")
        return {
            "success": False,
            "provider": provider,
            "error": str(e),
        }


def _generate_mock_provider_results(
    provider: str, query: str, limit: int
) -> list[dict[str, Any]]:
    """Generate mock results for a provider (replace with actual API calls)."""
    base_results = []

    for i in range(min(limit, 10)):  # Generate up to 10 mock results
        result = {
            "id": f"{provider}_{query.replace(' ', '_')}_{i}",
            "title": f"Result {i + 1} for '{query}' from {provider}",
            "url": f"https://example.com/{provider}/{i}",
            "snippet": f"This is a mock result snippet {i + 1} from {provider} for the query '{query}'. It contains relevant information and demonstrates the provider's response format.",
            "provider": provider,
            "rank": i + 1,
            "quality_score": 0.8 - (i * 0.05),  # Decreasing quality scores
            "relevance_score": 0.9 - (i * 0.03),
            "timestamp": "2024-01-01T00:00:00Z",
        }

        # Add provider-specific characteristics
        if provider == "google":
            result["featured_snippet"] = i == 0  # First result is featured
            result["knowledge_panel"] = i < 2  # Top results have knowledge panels
        elif provider == "bing":
            result["news_cluster"] = "news" in query.lower()
            result["entity_recognition"] = True
        elif provider == "duckduckgo":
            result["privacy_protected"] = True
            result["tracking_blocked"] = True
        elif provider == "searx":
            result["aggregated_sources"] = ["google", "bing", "yahoo"]
            result["source_diversity"] = 0.8

        base_results.append(result)

    return base_results


async def _apply_intelligent_fusion(
    provider_results: dict[str, dict],
    fusion_strategy: str,
    quality_threshold: float,
    ctx,
) -> dict[str, Any]:
    """Apply intelligent fusion strategy to combine provider results."""
    all_results = []

    # Collect all results from all providers
    for provider, provider_data in provider_results.items():
        for query, query_data in provider_data["query_results"].items():
            for result in query_data["results"]:
                # Add fusion metadata
                result["fusion_metadata"] = {
                    "source_provider": provider,
                    "source_query": query,
                    "original_rank": result.get("rank", 0),
                }
                all_results.append(result)

    # Apply quality filtering
    quality_filtered = [
        result
        for result in all_results
        if result.get("quality_score", 0.0) >= quality_threshold
    ]

    # Apply fusion strategy
    if fusion_strategy == "intelligent":
        fused_results = await _intelligent_fusion(quality_filtered, ctx)
    elif fusion_strategy == "weighted":
        fused_results = await _weighted_fusion(quality_filtered, ctx)
    else:  # ranked
        fused_results = await _ranked_fusion(quality_filtered, ctx)

    # Calculate fusion confidence
    confidence = _calculate_fusion_confidence(fused_results, len(all_results))

    return {
        "results": fused_results,
        "confidence": confidence,
        "fusion_metadata": {
            "total_raw_results": len(all_results),
            "quality_filtered_results": len(quality_filtered),
            "final_results": len(fused_results),
            "quality_filter_rate": len(quality_filtered) / len(all_results)
            if all_results
            else 0,
        },
    }


async def _intelligent_fusion(results: list[dict], ctx) -> list[dict]:
    """Apply intelligent ML-powered fusion."""
    # Remove URL duplicates first
    seen_urls = set()
    unique_results = []

    for result in results:
        url = result.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)

    # Calculate fusion scores
    for result in unique_results:
        quality_score = result.get("quality_score", 0.0)
        relevance_score = result.get("relevance_score", 0.0)
        provider_weight = _get_provider_weight(
            result.get("fusion_metadata", {}).get("source_provider", "")
        )

        # Intelligent fusion score
        fusion_score = (
            (quality_score * 0.4) + (relevance_score * 0.4) + (provider_weight * 0.2)
        )
        result["fusion_score"] = fusion_score

    # Sort by fusion score
    return sorted(
        unique_results, key=lambda x: x.get("fusion_score", 0.0), reverse=True
    )


async def _weighted_fusion(results: list[dict], ctx) -> list[dict]:
    """Apply weighted fusion based on provider weights."""
    provider_weights = {
        "google": 0.4,
        "bing": 0.3,
        "duckduckgo": 0.2,
        "searx": 0.1,
    }

    # Apply provider weights
    for result in results:
        provider = result.get("fusion_metadata", {}).get("source_provider", "")
        weight = provider_weights.get(provider, 0.1)
        original_rank = result.get("fusion_metadata", {}).get("original_rank", 10)

        # Weighted score (higher weight, lower rank = higher score)
        weighted_score = weight * (1.0 / (original_rank + 1))
        result["weighted_score"] = weighted_score

    # Sort by weighted score
    return sorted(results, key=lambda x: x.get("weighted_score", 0.0), reverse=True)


async def _ranked_fusion(results: list[dict], ctx) -> list[dict]:
    """Apply rank-based fusion."""
    # Simple rank-based fusion using reciprocal rank
    for result in results:
        original_rank = result.get("fusion_metadata", {}).get("original_rank", 10)
        rank_score = 1.0 / (original_rank + 1)
        result["rank_score"] = rank_score

    # Sort by rank score
    return sorted(results, key=lambda x: x.get("rank_score", 0.0), reverse=True)


def _get_provider_weight(provider: str) -> float:
    """Get weight for a specific provider."""
    weights = {
        "google": 0.4,
        "bing": 0.3,
        "duckduckgo": 0.2,
        "searx": 0.1,
    }
    return weights.get(provider, 0.1)


def _calculate_fusion_confidence(results: list[dict], total_results: int) -> float:
    """Calculate confidence in fusion results."""
    if not results or total_results == 0:
        return 0.0

    # Factor in result diversity and quality
    providers_represented = len(
        {r.get("fusion_metadata", {}).get("source_provider", "") for r in results[:10]}
    )

    avg_quality = sum(r.get("quality_score", 0.0) for r in results[:10]) / min(
        len(results), 10
    )
    provider_diversity = providers_represented / 4  # Max 4 providers

    confidence = (avg_quality * 0.7) + (provider_diversity * 0.3)
    return min(confidence, 1.0)


def _calculate_orchestration_metrics(
    provider_results: dict, fused_results: dict, total_raw_results: int
) -> dict[str, Any]:
    """Calculate metrics for orchestration performance."""
    return {
        "providers_used": len(provider_results),
        "total_raw_results": total_raw_results,
        "final_results": len(fused_results["results"]),
        "fusion_efficiency": len(fused_results["results"]) / total_raw_results
        if total_raw_results > 0
        else 0,
        "provider_coverage": len(provider_results) / 4,  # Assuming 4 max providers
        "quality_filter_effectiveness": fused_results.get("fusion_metadata", {}).get(
            "quality_filter_rate", 0.0
        ),
        "orchestration_confidence": fused_results["confidence"],
    }


async def _generate_web_search_insights(
    query: str, provider_results: dict, fused_results: dict, ctx
) -> dict[str, Any]:
    """Generate insights for web search optimization."""
    # Analyze provider performance
    provider_performance = {}
    for provider, data in provider_results.items():
        total_results = data["total_results"]
        avg_quality = (
            sum(
                sum(r.get("quality_score", 0.0) for r in qr["results"])
                for qr in data["query_results"].values()
            )
            / total_results
            if total_results > 0
            else 0
        )

        provider_performance[provider] = {
            "results_count": total_results,
            "average_quality": avg_quality,
            "effectiveness": avg_quality
            * (total_results / 20),  # Normalize by expected count
        }

    # Generate insights
    best_provider = (
        max(provider_performance.items(), key=lambda x: x[1]["effectiveness"])[0]
        if provider_performance
        else None
    )

    insights = {
        "provider_analysis": {
            "best_performing_provider": best_provider,
            "provider_performance": provider_performance,
            "provider_diversity": len(provider_results),
        },
        "fusion_analysis": {
            "fusion_confidence": fused_results["confidence"],
            "result_diversity": fused_results.get("fusion_metadata", {}).get(
                "quality_filter_rate", 0.0
            ),
        },
        "recommendations": [],
    }

    # Generate recommendations
    if fused_results["confidence"] < 0.6:
        insights["recommendations"].append(
            "Consider using additional providers for better coverage"
        )

    if len(provider_results) < 2:
        insights["recommendations"].append(
            "Use multiple providers for result diversity"
        )

    avg_provider_quality = (
        sum(p["average_quality"] for p in provider_performance.values())
        / len(provider_performance)
        if provider_performance
        else 0
    )
    if avg_provider_quality < 0.7:
        insights["recommendations"].append(
            "Adjust quality threshold or provider selection"
        )

    return insights


async def _analyze_query_characteristics(
    query: str, search_intent: str, ctx
) -> dict[str, Any]:
    """Analyze query characteristics for adaptive optimization."""
    words = query.split()

    return {
        "length": len(words),
        "complexity": "high"
        if len(words) > 6
        else "medium"
        if len(words) > 3
        else "low",
        "search_intent": search_intent,
        "question_type": _classify_question_type(query),
        "domain_indicators": _detect_domain_indicators(query),
        "temporal_indicators": _detect_temporal_indicators(query),
    }


def _classify_question_type(query: str) -> str:
    """Classify the type of question being asked."""
    query_lower = query.lower()

    if any(word in query_lower for word in ["how", "implement", "build", "create"]):
        return "procedural"
    if any(word in query_lower for word in ["what", "define", "explain", "describe"]):
        return "definitional"
    if any(word in query_lower for word in ["why", "reason", "cause", "purpose"]):
        return "causal"
    if any(word in query_lower for word in ["compare", "difference", "vs", "versus"]):
        return "comparative"
    return "informational"


def _detect_domain_indicators(query: str) -> list[str]:
    """Detect domain-specific indicators in query."""
    technical_terms = [
        "algorithm",
        "implementation",
        "architecture",
        "framework",
        "api",
    ]
    business_terms = ["strategy", "market", "revenue", "cost", "optimization"]
    academic_terms = ["research", "study", "analysis", "methodology", "theory"]
    news_terms = ["news", "current", "latest", "recent", "breaking"]

    domains = []
    query_lower = query.lower()

    if any(term in query_lower for term in technical_terms):
        domains.append("technical")
    if any(term in query_lower for term in business_terms):
        domains.append("business")
    if any(term in query_lower for term in academic_terms):
        domains.append("academic")
    if any(term in query_lower for term in news_terms):
        domains.append("news")

    return domains or ["general"]


def _detect_temporal_indicators(query: str) -> str:
    """Detect temporal requirements in query."""
    query_lower = query.lower()

    if any(
        term in query_lower for term in ["latest", "recent", "current", "new", "today"]
    ):
        return "recent"
    if any(term in query_lower for term in ["historical", "past", "before", "old"]):
        return "historical"
    return "neutral"


async def _select_adaptive_parameters(
    query_analysis: dict,
    performance_target: str,
    iteration: int,
    best_result: dict | None,
    ctx,
) -> dict[str, Any]:
    """Select optimal parameters for adaptive iteration."""
    base_params = {
        "providers": ["google", "bing"],
        "max_results": 20,
        "fusion_strategy": "intelligent",
        "quality_threshold": 0.7,
        "auto_expand": True,
    }

    # Adjust based on query characteristics
    if query_analysis["complexity"] == "high":
        base_params["providers"].append("searx")  # Add meta-search for complex queries
        base_params["auto_expand"] = True

    if "news" in query_analysis["domain_indicators"]:
        base_params["providers"] = [
            "bing",
            "google",
        ]  # Prioritize news-friendly providers

    # Adjust based on performance target
    if performance_target == "speed":
        base_params["providers"] = base_params["providers"][:2]  # Limit providers
        base_params["fusion_strategy"] = "ranked"  # Faster fusion
        base_params["max_results"] = 15
    elif performance_target == "quality":
        base_params["quality_threshold"] = 0.8
        base_params["fusion_strategy"] = "intelligent"
    elif performance_target == "coverage":
        base_params["providers"] = ["google", "bing", "duckduckgo", "searx"]
        base_params["max_results"] = 30

    # Iterative adjustments
    if iteration > 0 and best_result:
        current_confidence = best_result.get("orchestration_metadata", {}).get(
            "fusion_confidence", 0.0
        )

        if current_confidence < 0.6:
            # Low confidence - try different approach
            if base_params["fusion_strategy"] == "intelligent":
                base_params["fusion_strategy"] = "weighted"
            base_params["quality_threshold"] = max(
                base_params["quality_threshold"] - 0.1, 0.5
            )

    return base_params


async def _evaluate_iteration_performance(
    search_result: dict, performance_target: str, query_analysis: dict, ctx
) -> float:
    """Evaluate performance of a search iteration."""
    # Base performance factors
    result_count = len(search_result.get("results", []))
    fusion_confidence = search_result.get("orchestration_metadata", {}).get(
        "fusion_confidence", 0.0
    )
    provider_count = len(
        search_result.get("orchestration_metadata", {}).get("providers_used", [])
    )

    # Calculate performance score based on target
    if performance_target == "speed":
        # Favor fewer providers and good confidence
        score = (fusion_confidence * 0.6) + ((4 - provider_count) / 4 * 0.4)
    elif performance_target == "quality":
        # Favor high confidence and reasonable result count
        score = (fusion_confidence * 0.8) + (min(result_count / 20, 1.0) * 0.2)
    elif performance_target == "coverage":
        # Favor more results and providers
        score = (
            (min(result_count / 30, 1.0) * 0.5)
            + (provider_count / 4 * 0.3)
            + (fusion_confidence * 0.2)
        )
    else:  # balanced
        score = (
            (fusion_confidence * 0.4)
            + (min(result_count / 20, 1.0) * 0.3)
            + (provider_count / 4 * 0.3)
        )

    return min(score, 1.0)


async def _apply_adaptive_learning(
    query_analysis: dict, iteration_results: list[dict], best_result: dict, ctx
) -> dict[str, Any]:
    """Apply adaptive learning from iteration results."""
    # Analyze what worked best
    best_params = best_result.get("iteration_metadata", {}).get("parameters_used", {})

    return {
        "optimal_providers": best_params.get("providers", []),
        "optimal_fusion_strategy": best_params.get("fusion_strategy", "intelligent"),
        "optimal_quality_threshold": best_params.get("quality_threshold", 0.7),
        "performance_correlation": {
            "provider_count_impact": _analyze_provider_count_impact(iteration_results),
            "fusion_strategy_effectiveness": _analyze_fusion_strategy_effectiveness(
                iteration_results
            ),
            "quality_threshold_impact": _analyze_quality_threshold_impact(
                iteration_results
            ),
        },
        "learning_confidence": 0.8,  # Mock learning confidence
    }


def _analyze_provider_count_impact(iteration_results: list[dict]) -> float:
    """Analyze impact of provider count on performance."""
    if len(iteration_results) < 2:
        return 0.0

    # Simple correlation analysis
    provider_counts = [len(ir["parameters"]["providers"]) for ir in iteration_results]

    # Calculate simple correlation
    if len(set(provider_counts)) > 1:
        # Find best performing provider count
        best_iteration = max(iteration_results, key=lambda x: x["performance_score"])
        optimal_count = len(best_iteration["parameters"]["providers"])
        return optimal_count / 4  # Normalize to 0-1

    return 0.5


def _analyze_fusion_strategy_effectiveness(
    iteration_results: list[dict],
) -> dict[str, float]:
    """Analyze effectiveness of different fusion strategies."""
    strategy_performance = {}

    for result in iteration_results:
        strategy = result["parameters"]["fusion_strategy"]
        score = result["performance_score"]

        if strategy not in strategy_performance:
            strategy_performance[strategy] = []
        strategy_performance[strategy].append(score)

    # Average performance by strategy
    return {
        strategy: sum(scores) / len(scores)
        for strategy, scores in strategy_performance.items()
    }


def _analyze_quality_threshold_impact(iteration_results: list[dict]) -> float:
    """Analyze impact of quality threshold on performance."""
    if len(iteration_results) < 2:
        return 0.7  # Default threshold

    best_iteration = max(iteration_results, key=lambda x: x["performance_score"])
    return best_iteration["parameters"]["quality_threshold"]


async def _get_available_providers(ctx) -> list[str]:
    """Get list of available search providers."""
    # In real implementation, this would check actual provider availability
    return ["google", "bing", "duckduckgo", "searx"]


async def _apply_intelligent_deduplication(
    provider_results: dict, strategy: str, ctx
) -> dict[str, Any]:
    """Apply intelligent deduplication to provider results."""
    all_results = []

    # Collect all results
    for provider_data in provider_results.values():
        for query_data in provider_data["query_results"].values():
            all_results.extend(query_data["results"])

    if strategy == "url":
        deduplicated = _url_deduplication(all_results)
    elif strategy == "semantic":
        deduplicated = await _semantic_deduplication(all_results, ctx)
    else:  # hybrid
        deduplicated = await _hybrid_deduplication(all_results, ctx)

    return {
        "results": deduplicated,
        "metadata": {
            "original_count": len(all_results),
            "deduplicated_count": len(deduplicated),
            "duplication_rate": (len(all_results) - len(deduplicated))
            / len(all_results)
            if all_results
            else 0,
            "strategy_used": strategy,
        },
    }


def _url_deduplication(results: list[dict]) -> list[dict]:
    """Remove duplicates based on URL."""
    seen_urls = set()
    unique_results = []

    for result in results:
        url = result.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)

    return unique_results


async def _semantic_deduplication(results: list[dict], ctx) -> list[dict]:
    """Remove duplicates based on semantic similarity."""
    # Simple semantic deduplication using title and snippet similarity
    unique_results = []

    for result in results:
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        content = f"{title} {snippet}".lower()

        # Check similarity with existing results
        is_duplicate = False
        for existing in unique_results:
            existing_title = existing.get("title", "")
            existing_snippet = existing.get("snippet", "")
            existing_content = f"{existing_title} {existing_snippet}".lower()

            # Simple word overlap check
            content_words = set(content.split())
            existing_words = set(existing_content.split())

            if content_words and existing_words:
                overlap = len(content_words.intersection(existing_words))
                similarity = overlap / min(len(content_words), len(existing_words))

                if similarity > 0.8:  # High similarity threshold
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique_results.append(result)

    return unique_results


async def _hybrid_deduplication(results: list[dict], ctx) -> list[dict]:
    """Apply hybrid deduplication combining URL and semantic approaches."""
    # First apply URL deduplication
    url_deduplicated = _url_deduplication(results)

    # Then apply semantic deduplication
    return await _semantic_deduplication(url_deduplicated, ctx)


async def _apply_result_synthesis(
    deduplicated_results: dict,
    synthesis_depth: str,
    provider_preferences: dict[str, float] | None,
    ctx,
) -> dict[str, Any]:
    """Apply result synthesis based on specified depth."""
    results = deduplicated_results["results"]

    if synthesis_depth == "basic":
        synthesized = await _basic_synthesis(results, ctx)
    elif synthesis_depth == "comprehensive":
        synthesized = await _comprehensive_synthesis(results, provider_preferences, ctx)
    else:  # standard
        synthesized = await _standard_synthesis(results, ctx)

    return synthesized


async def _basic_synthesis(results: list[dict], ctx) -> dict[str, Any]:
    """Apply basic synthesis - simple ranking."""
    # Sort by quality score
    sorted_results = sorted(
        results, key=lambda x: x.get("quality_score", 0.0), reverse=True
    )

    return {
        "results": sorted_results,
        "confidence": 0.7,  # Basic confidence
        "synthesis_method": "quality_ranking",
    }


async def _standard_synthesis(results: list[dict], ctx) -> dict[str, Any]:
    """Apply standard synthesis with moderate sophistication."""
    # Combine quality score and relevance
    for result in results:
        quality = result.get("quality_score", 0.0)
        relevance = result.get("relevance_score", 0.0)
        combined_score = (quality * 0.6) + (relevance * 0.4)
        result["synthesis_score"] = combined_score

    # Sort by synthesis score
    sorted_results = sorted(
        results, key=lambda x: x.get("synthesis_score", 0.0), reverse=True
    )

    return {
        "results": sorted_results,
        "confidence": 0.8,
        "synthesis_method": "quality_relevance_combination",
    }


async def _comprehensive_synthesis(
    results: list[dict], provider_preferences: dict[str, float] | None, ctx
) -> dict[str, Any]:
    """Apply comprehensive synthesis with full sophistication."""
    preferences = provider_preferences or {}

    # Multi-factor synthesis
    for result in results:
        quality = result.get("quality_score", 0.0)
        relevance = result.get("relevance_score", 0.0)
        provider = result.get("provider", "")
        provider_weight = preferences.get(provider, 0.25)  # Default weight

        # Comprehensive synthesis score
        comprehensive_score = (
            (quality * 0.4) + (relevance * 0.4) + (provider_weight * 0.2)
        )
        result["comprehensive_score"] = comprehensive_score

    # Sort by comprehensive score
    sorted_results = sorted(
        results, key=lambda x: x.get("comprehensive_score", 0.0), reverse=True
    )

    return {
        "results": sorted_results,
        "confidence": 0.9,
        "synthesis_method": "comprehensive_multi_factor",
    }


def _calculate_synthesis_metrics(
    provider_results: dict, deduplicated_results: dict, synthesized_results: dict
) -> dict[str, Any]:
    """Calculate metrics for synthesis process."""
    total_original = sum(
        provider_data["total_results"] for provider_data in provider_results.values()
    )

    return {
        "original_results": total_original,
        "deduplicated_results": len(deduplicated_results["results"]),
        "final_synthesized": len(synthesized_results["results"]),
        "deduplication_rate": deduplicated_results["metadata"]["duplication_rate"],
        "synthesis_efficiency": len(synthesized_results["results"])
        / len(deduplicated_results["results"])
        if deduplicated_results["results"]
        else 0,
        "overall_confidence": synthesized_results["confidence"],
    }
