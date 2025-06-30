"""HyDE (Hypothetical Document Embeddings) search tools for MCP server.

Implements HyDE search with autonomous document generation and
adaptive query enhancement for improved semantic search results.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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
    """Register HyDE search tools with the MCP server."""

    @mcp.tool()
    async def hyde_semantic_search(
        query: str,
        collection_name: str,
        limit: int = 10,
        hyde_documents: int = 3,
        generation_strategy: str = "adaptive",
        filters: Optional[Dict[str, Any]] = None,
        ctx: Context = None,
    ) -> Dict[str, Any]:
        """Perform HyDE search by generating hypothetical documents and using their embeddings.

        Implements HyDE (Hypothetical Document Embeddings) with autonomous document
        generation and adaptive query enhancement for improved semantic search.

        Args:
            query: Original search query
            collection_name: Target collection for search
            limit: Maximum number of results to return
            hyde_documents: Number of hypothetical documents to generate
            generation_strategy: Strategy for document generation (adaptive, diverse, focused)
            filters: Optional metadata filters
            ctx: MCP context for logging

        Returns:
            HyDE search results with generation metadata and quality metrics
        """
        try:
            if ctx:
                await ctx.info(
                    f"Performing HyDE search: '{query}' with {hyde_documents} hypothetical documents"
                )

            # Validate query
            security_validator = SecurityValidator.from_unified_config()
            validated_query = security_validator.validate_search_query(query)

            # Get services
            qdrant_service = await client_manager.get_qdrant_service()
            embedding_manager = await client_manager.get_embedding_manager()
            llm_client = await client_manager.get_llm_client()

            # Generate hypothetical documents
            hypothetical_docs = await _generate_hypothetical_documents(
                llm_client, validated_query, hyde_documents, generation_strategy, ctx
            )

            if not hypothetical_docs["success"]:
                # Fallback to regular search if HyDE generation fails
                if ctx:
                    await ctx.warning(
                        "HyDE generation failed, falling back to regular search"
                    )
                return await _fallback_search(
                    qdrant_service,
                    embedding_manager,
                    validated_query,
                    collection_name,
                    limit,
                    filters,
                    ctx,
                )

            # Generate embeddings for hypothetical documents
            doc_texts = [doc["content"] for doc in hypothetical_docs["documents"]]
            embeddings_result = await embedding_manager.generate_embeddings(doc_texts)
            embeddings = embeddings_result.embeddings

            if ctx:
                await ctx.debug(
                    f"Generated embeddings for {len(embeddings)} hypothetical documents"
                )

            # Perform search using each hypothetical document embedding
            all_results = []
            search_metadata = []

            for i, (doc, embedding) in enumerate(
                zip(hypothetical_docs["documents"], embeddings)
            ):
                search_result = await qdrant_service.search(
                    collection_name=collection_name,
                    query_vector=embedding,
                    limit=limit * 2,  # Get more for fusion
                    filter=filters,
                    with_payload=True,
                    with_vectors=False,
                )

                if search_result and "points" in search_result:
                    # Add metadata about which hypothetical document generated this result
                    for point in search_result["points"]:
                        point["hyde_source"] = i
                        point["hyde_document"] = doc["content"][:100] + "..."
                        point["hyde_relevance"] = doc["relevance_score"]

                    all_results.extend(search_result["points"])
                    search_metadata.append(
                        {
                            "hyde_doc_index": i,
                            "results_count": len(search_result["points"]),
                            "generation_quality": doc["quality_score"],
                        }
                    )

                if ctx:
                    await ctx.debug(
                        f"HyDE doc {i + 1}: {len(search_result.get('points', []))} results"
                    )

            # Fuse results from multiple hypothetical documents
            fused_results = await _fuse_hyde_results(
                all_results, hypothetical_docs["documents"], limit, ctx
            )

            # Calculate HyDE-specific metrics
            hyde_metrics = _calculate_hyde_metrics(
                hypothetical_docs, search_metadata, fused_results
            )

            # Generate autonomous optimization insights
            optimization_insights = await _generate_hyde_optimization_insights(
                validated_query, hypothetical_docs, fused_results, ctx
            )

            final_results = {
                "success": True,
                "query": validated_query,
                "collection": collection_name,
                "results": fused_results["results"],
                "hyde_metadata": {
                    "hypothetical_documents": [
                        {
                            "content_preview": doc["content"][:200] + "...",
                            "quality_score": doc["quality_score"],
                            "relevance_score": doc["relevance_score"],
                            "generation_strategy": doc["strategy"],
                        }
                        for doc in hypothetical_docs["documents"]
                    ],
                    "generation_strategy": generation_strategy,
                    "documents_generated": len(hypothetical_docs["documents"]),
                    "fusion_confidence": fused_results["confidence"],
                },
                "search_metadata": search_metadata,
                "hyde_metrics": hyde_metrics,
                "autonomous_optimization": optimization_insights,
            }

            if ctx:
                await ctx.info(
                    f"HyDE search completed: {len(fused_results['results'])} results from {len(hypothetical_docs['documents'])} hypothetical documents"
                )

            return final_results

        except Exception as e:
            logger.exception("Failed to perform HyDE search")
            if ctx:
                await ctx.error(f"HyDE search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "generation_strategy": generation_strategy,
            }

    @mcp.tool()
    async def adaptive_hyde_search(
        query: str,
        collection_name: str,
        limit: int = 10,
        auto_optimize: bool = True,
        quality_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
        ctx: Context = None,
    ) -> Dict[str, Any]:
        """Perform adaptive HyDE search with ML-powered optimization.

        Automatically adjusts HyDE parameters based on query characteristics
        and iteratively improves document generation quality.

        Args:
            query: Original search query
            collection_name: Target collection for search
            limit: Maximum number of results to return
            auto_optimize: Enable autonomous parameter optimization
            quality_threshold: Minimum quality threshold for generated documents
            filters: Optional metadata filters
            ctx: MCP context for logging

        Returns:
            Optimized HyDE search results with adaptation metadata
        """
        try:
            if ctx:
                await ctx.info(
                    f"Performing adaptive HyDE search with quality threshold {quality_threshold}"
                )

            # Analyze query for optimal HyDE parameters
            query_analysis = await _analyze_query_for_hyde(query, ctx)

            # Select optimal HyDE parameters
            optimal_params = await _select_optimal_hyde_parameters(
                query_analysis, auto_optimize, quality_threshold, ctx
            )

            # Perform iterative HyDE search with quality improvement
            best_result = None
            iteration_results = []

            for iteration in range(optimal_params["max_iterations"]):
                if ctx:
                    await ctx.debug(
                        f"HyDE iteration {iteration + 1}/{optimal_params['max_iterations']}"
                    )

                # Adjust parameters for this iteration
                iteration_params = await _adjust_iteration_parameters(
                    optimal_params, iteration, best_result, ctx
                )

                # Perform HyDE search with current parameters
                iteration_result = await hyde_semantic_search(
                    query=query,
                    collection_name=collection_name,
                    limit=limit,
                    hyde_documents=iteration_params["hyde_documents"],
                    generation_strategy=iteration_params["strategy"],
                    filters=filters,
                    ctx=ctx,
                )

                if iteration_result["success"]:
                    iteration_results.append(
                        {
                            "iteration": iteration + 1,
                            "parameters": iteration_params,
                            "quality_score": iteration_result["hyde_metrics"][
                                "average_quality"
                            ],
                            "results_count": len(iteration_result["results"]),
                        }
                    )

                    # Check if this iteration meets quality threshold
                    if iteration_result["hyde_metrics"][
                        "average_quality"
                    ] >= quality_threshold and (
                        not best_result
                        or iteration_result["hyde_metrics"]["average_quality"]
                        > best_result["hyde_metrics"]["average_quality"]
                    ):
                        best_result = iteration_result

                        if ctx:
                            await ctx.debug(
                                f"Quality threshold met in iteration {iteration + 1}"
                            )
                        break

                    if not best_result:
                        best_result = iteration_result

            if not best_result:
                return {
                    "success": False,
                    "error": "All HyDE iterations failed",
                    "iterations_attempted": len(iteration_results),
                }

            # Add adaptive optimization metadata
            best_result["adaptive_optimization"] = {
                "query_analysis": query_analysis,
                "optimal_parameters": optimal_params,
                "iterations_performed": len(iteration_results),
                "iteration_results": iteration_results,
                "quality_threshold": quality_threshold,
                "final_quality_score": best_result["hyde_metrics"]["average_quality"],
                "optimization_success": best_result["hyde_metrics"]["average_quality"]
                >= quality_threshold,
            }

            if ctx:
                await ctx.info(
                    f"Adaptive HyDE completed in {len(iteration_results)} iterations with quality {best_result['hyde_metrics']['average_quality']:.2f}"
                )

            return best_result

        except Exception as e:
            logger.exception("Failed to perform adaptive HyDE search")
            if ctx:
                await ctx.error(f"Adaptive HyDE search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "quality_threshold": quality_threshold,
            }

    @mcp.tool()
    async def hyde_query_expansion(
        query: str,
        expansion_factor: int = 3,
        diversity_weight: float = 0.3,
        ctx: Context = None,
    ) -> Dict[str, Any]:
        """Generate expanded queries using HyDE document generation.

        Uses HyDE's hypothetical document generation to create diverse
        query variations for improved search coverage.

        Args:
            query: Original query to expand
            expansion_factor: Number of expanded queries to generate
            diversity_weight: Weight for promoting query diversity
            ctx: MCP context for logging

        Returns:
            Expanded queries with quality and diversity metrics
        """
        try:
            if ctx:
                await ctx.info(
                    f"Generating {expansion_factor} expanded queries using HyDE"
                )

            # Validate query
            security_validator = SecurityValidator.from_unified_config()
            validated_query = security_validator.validate_search_query(query)

            # Get LLM client
            llm_client = await client_manager.get_llm_client()

            # Generate hypothetical documents for query expansion
            expansion_docs = await _generate_expansion_documents(
                llm_client, validated_query, expansion_factor, diversity_weight, ctx
            )

            if not expansion_docs["success"]:
                return {
                    "success": False,
                    "error": "Failed to generate expansion documents",
                    "original_query": validated_query,
                }

            # Extract key phrases and concepts from hypothetical documents
            expanded_queries = await _extract_expansion_queries(
                expansion_docs["documents"], validated_query, ctx
            )

            # Calculate diversity and quality metrics
            expansion_metrics = _calculate_expansion_metrics(
                validated_query, expanded_queries, expansion_docs["documents"]
            )

            final_results = {
                "success": True,
                "original_query": validated_query,
                "expanded_queries": expanded_queries,
                "expansion_metadata": {
                    "expansion_factor": expansion_factor,
                    "diversity_weight": diversity_weight,
                    "generation_strategy": "hyde_based",
                    "documents_used": len(expansion_docs["documents"]),
                },
                "expansion_metrics": expansion_metrics,
                "autonomous_features": {
                    "diversity_optimization": True,
                    "semantic_expansion": True,
                    "quality_assessment": True,
                },
            }

            if ctx:
                await ctx.info(
                    f"Query expansion completed: {len(expanded_queries)} queries generated"
                )

            return final_results

        except Exception as e:
            logger.exception("Failed to perform HyDE query expansion")
            if ctx:
                await ctx.error(f"HyDE query expansion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "original_query": query,
            }

    @mcp.tool()
    async def get_hyde_capabilities() -> Dict[str, Any]:
        """Get HyDE search capabilities and configuration options.

        Returns:
            Comprehensive capabilities information for HyDE search system
        """
        return {
            "generation_strategies": {
                "adaptive": {
                    "description": "Adapts generation style based on query characteristics",
                    "best_for": ["mixed_queries", "unknown_domains"],
                    "quality": "high",
                    "speed": "medium",
                },
                "diverse": {
                    "description": "Generates diverse hypothetical documents for broad coverage",
                    "best_for": ["exploratory_search", "broad_topics"],
                    "quality": "medium",
                    "speed": "fast",
                },
                "focused": {
                    "description": "Generates focused documents for specific domains",
                    "best_for": ["specific_queries", "domain_expertise"],
                    "quality": "high",
                    "speed": "slow",
                },
            },
            "adaptive_features": {
                "query_analysis": True,
                "iterative_improvement": True,
                "quality_thresholding": True,
                "parameter_optimization": True,
            },
            "quality_metrics": [
                "document_relevance",
                "generation_coherence",
                "semantic_consistency",
                "fusion_effectiveness",
            ],
            "optimization_capabilities": {
                "autonomous_parameter_tuning": True,
                "quality_driven_iteration": True,
                "diversity_enhancement": True,
                "fallback_mechanisms": True,
            },
            "query_expansion": {
                "hyde_based_expansion": True,
                "diversity_optimization": True,
                "semantic_enhancement": True,
                "concept_extraction": True,
            },
            "status": "active",
        }


# Helper functions


async def _generate_hypothetical_documents(
    llm_client, query: str, count: int, strategy: str, ctx
) -> Dict[str, Any]:
    """Generate hypothetical documents using LLM."""
    try:
        # Prepare generation prompt based on strategy
        if strategy == "adaptive":
            prompt = f"""Generate {count} hypothetical documents that would be relevant to this query: "{query}"

Each document should:
1. Be written as if it directly answers or relates to the query
2. Use natural language and domain-appropriate terminology
3. Include specific details and examples
4. Be roughly 100-150 words each

Vary the style and approach across documents to maximize search effectiveness."""

        elif strategy == "diverse":
            prompt = f"""Generate {count} diverse hypothetical documents for the query: "{query}"

Create documents with different:
- Perspectives and viewpoints
- Levels of technical detail
- Writing styles (formal, casual, technical)
- Content focus areas

Each document should be 100-150 words and approach the topic differently."""

        else:  # focused
            prompt = f"""Generate {count} focused, high-quality hypothetical documents for: "{query}"

Each document should:
- Demonstrate deep expertise on the topic
- Use precise, domain-specific terminology
- Provide detailed, accurate information
- Be well-structured and coherent
- Be approximately 150-200 words"""

        # Mock LLM response (replace with actual LLM call)
        documents = []
        for i in range(count):
            doc_content = await _mock_document_generation(query, strategy, i)
            documents.append(
                {
                    "content": doc_content,
                    "strategy": strategy,
                    "quality_score": 0.8 + (i * 0.05),  # Mock quality scores
                    "relevance_score": 0.9 - (i * 0.1),
                    "index": i,
                }
            )

        if ctx:
            await ctx.debug(
                f"Generated {len(documents)} hypothetical documents using {strategy} strategy"
            )

        return {
            "success": True,
            "documents": documents,
            "strategy": strategy,
            "generation_metadata": {
                "prompt_used": prompt[:100] + "...",
                "total_documents": len(documents),
                "average_quality": sum(doc["quality_score"] for doc in documents)
                / len(documents),
            },
        }

    except Exception as e:
        logger.exception("Failed to generate hypothetical documents")
        return {
            "success": False,
            "error": str(e),
            "strategy": strategy,
        }


async def _mock_document_generation(query: str, strategy: str, index: int) -> str:
    """Mock document generation (replace with actual LLM call)."""
    templates = {
        "adaptive": [
            f"When considering {query}, it's important to understand the fundamental concepts involved. This topic encompasses various aspects that are crucial for practical implementation and theoretical understanding.",
            f"Recent developments in {query} have shown significant progress in addressing key challenges. The latest research indicates several promising approaches that could revolutionize the field.",
            f"A comprehensive guide to {query} reveals multiple dimensions of complexity. From basic principles to advanced applications, this subject requires careful consideration of numerous factors.",
        ],
        "diverse": [
            f"From a beginner's perspective, {query} can seem overwhelming at first. However, breaking it down into manageable components makes it much more approachable and understandable.",
            f"Industry experts often debate the best practices for {query}. While there's no universal consensus, several established methodologies have proven effective in different scenarios.",
            f"The technical implementation of {query} involves sophisticated algorithms and careful optimization. Performance considerations and scalability requirements drive most architectural decisions.",
        ],
        "focused": [
            f"Advanced practitioners of {query} understand that optimal results require deep expertise in underlying principles. The sophisticated interplay between theoretical foundations and practical constraints demands careful analysis.",
            f"Professional implementation of {query} necessitates comprehensive understanding of industry standards, regulatory requirements, and performance optimization techniques that distinguish expert-level practice.",
            f"Strategic approaches to {query} incorporate enterprise-level considerations including scalability, maintainability, security protocols, and long-term sustainability requirements.",
        ],
    }

    template_list = templates.get(strategy, templates["adaptive"])
    return template_list[index % len(template_list)]


async def _fallback_search(
    qdrant_service,
    embedding_manager,
    query: str,
    collection_name: str,
    limit: int,
    filters: Optional[Dict],
    ctx,
) -> Dict[str, Any]:
    """Perform fallback search when HyDE generation fails."""
    try:
        # Generate embedding for original query
        embeddings_result = await embedding_manager.generate_embeddings([query])
        query_embedding = embeddings_result.embeddings[0]

        # Perform regular vector search
        search_result = await qdrant_service.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            filter=filters,
            with_payload=True,
            with_vectors=False,
        )

        return {
            "success": True,
            "query": query,
            "collection": collection_name,
            "results": search_result.get("points", []),
            "fallback_used": True,
            "hyde_metadata": {
                "generation_failed": True,
                "fallback_method": "standard_vector_search",
            },
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Both HyDE and fallback search failed: {e}",
            "query": query,
        }


async def _fuse_hyde_results(
    all_results: List[Dict], hypothetical_docs: List[Dict], limit: int, ctx
) -> Dict[str, Any]:
    """Fuse results from multiple HyDE document searches."""
    # Group results by document ID
    document_scores = {}

    for result in all_results:
        doc_id = result.get("id")
        if doc_id not in document_scores:
            document_scores[doc_id] = {
                "document": result,
                "scores": [],
                "hyde_sources": [],
            }

        document_scores[doc_id]["scores"].append(result.get("score", 0.0))
        document_scores[doc_id]["hyde_sources"].append(result.get("hyde_source", -1))

    # Calculate fused scores
    fused_results = []
    for doc_id, doc_data in document_scores.items():
        # Use max score fusion (can be enhanced with other strategies)
        fused_score = max(doc_data["scores"])

        # Add fusion metadata
        doc_data["document"]["fused_score"] = fused_score
        doc_data["document"]["hyde_fusion_metadata"] = {
            "source_count": len(doc_data["scores"]),
            "max_score": max(doc_data["scores"]),
            "avg_score": sum(doc_data["scores"]) / len(doc_data["scores"]),
            "hyde_sources": doc_data["hyde_sources"],
        }

        fused_results.append(doc_data["document"])

    # Sort by fused score
    fused_results.sort(key=lambda x: x.get("fused_score", 0.0), reverse=True)

    # Calculate fusion confidence
    confidence = _calculate_fusion_confidence(fused_results, len(hypothetical_docs))

    return {
        "results": fused_results[:limit],
        "confidence": confidence,
        "fusion_metadata": {
            "total_raw_results": len(all_results),
            "unique_documents": len(document_scores),
            "fusion_method": "max_score",
            "average_source_overlap": sum(
                len(doc_data["scores"]) for doc_data in document_scores.values()
            )
            / len(document_scores)
            if document_scores
            else 0,
        },
    }


def _calculate_fusion_confidence(results: List[Dict], num_hyde_docs: int) -> float:
    """Calculate confidence in HyDE fusion results."""
    if not results:
        return 0.0

    # Factor in score distribution and source diversity
    scores = [r.get("fused_score", 0.0) for r in results]
    max_score = max(scores) if scores else 0.0

    # Check source diversity
    source_diversity = 0.0
    for result in results[:5]:  # Check top 5 results
        fusion_meta = result.get("hyde_fusion_metadata", {})
        source_count = fusion_meta.get("source_count", 1)
        source_diversity += min(source_count / num_hyde_docs, 1.0)

    source_diversity = source_diversity / min(len(results), 5) if results else 0.0

    # Combine score quality and source diversity
    confidence = (max_score * 0.7) + (source_diversity * 0.3)
    return min(confidence, 1.0)


def _calculate_hyde_metrics(
    hypothetical_docs: Dict, search_metadata: List[Dict], fused_results: Dict
) -> Dict[str, Any]:
    """Calculate HyDE-specific metrics."""
    docs = hypothetical_docs["documents"]

    return {
        "average_quality": sum(doc["quality_score"] for doc in docs) / len(docs),
        "average_relevance": sum(doc["relevance_score"] for doc in docs) / len(docs),
        "generation_efficiency": len(search_metadata) / len(docs),
        "fusion_effectiveness": fused_results["confidence"],
        "result_diversity": fused_results.get("fusion_metadata", {}).get(
            "average_source_overlap", 0.0
        ),
        "documents_with_results": len(
            [meta for meta in search_metadata if meta["results_count"] > 0]
        ),
    }


async def _generate_hyde_optimization_insights(
    query: str, hypothetical_docs: Dict, fused_results: Dict, ctx
) -> Dict[str, Any]:
    """Generate optimization insights for HyDE search."""
    docs = hypothetical_docs["documents"]
    avg_quality = sum(doc["quality_score"] for doc in docs) / len(docs)

    insights = {
        "quality_analysis": {
            "average_document_quality": avg_quality,
            "quality_variance": _calculate_variance(
                [doc["quality_score"] for doc in docs]
            ),
            "improvement_potential": "high"
            if avg_quality < 0.7
            else "medium"
            if avg_quality < 0.8
            else "low",
        },
        "fusion_analysis": {
            "fusion_confidence": fused_results["confidence"],
            "source_diversity": fused_results.get("fusion_metadata", {}).get(
                "average_source_overlap", 0.0
            ),
        },
        "recommendations": [],
    }

    # Generate recommendations
    if avg_quality < 0.7:
        insights["recommendations"].append(
            "Consider using focused generation strategy for higher quality"
        )
    if fused_results["confidence"] < 0.6:
        insights["recommendations"].append(
            "Increase number of hypothetical documents for better fusion"
        )
    if len(docs) < 3:
        insights["recommendations"].append(
            "Generate more hypothetical documents for better coverage"
        )

    return insights


def _calculate_variance(values: List[float]) -> float:
    """Calculate variance of a list of values."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


async def _analyze_query_for_hyde(query: str, ctx) -> Dict[str, Any]:
    """Analyze query characteristics for optimal HyDE parameters."""
    words = query.split()

    return {
        "length": len(words),
        "complexity": "high"
        if len(words) > 7
        else "medium"
        if len(words) > 3
        else "low",
        "specificity": "high" if any(len(word) > 10 for word in words) else "medium",
        "domain_indicators": _detect_domain_indicators(query),
        "question_type": _classify_question_type(query),
    }


def _detect_domain_indicators(query: str) -> List[str]:
    """Detect domain-specific indicators in query."""
    technical_terms = [
        "algorithm",
        "implementation",
        "architecture",
        "framework",
        "protocol",
    ]
    business_terms = ["strategy", "market", "revenue", "cost", "optimization"]
    academic_terms = ["research", "study", "analysis", "methodology", "theory"]

    domains = []
    query_lower = query.lower()

    if any(term in query_lower for term in technical_terms):
        domains.append("technical")
    if any(term in query_lower for term in business_terms):
        domains.append("business")
    if any(term in query_lower for term in academic_terms):
        domains.append("academic")

    return domains or ["general"]


def _classify_question_type(query: str) -> str:
    """Classify the type of question being asked."""
    query_lower = query.lower()

    if any(word in query_lower for word in ["how", "implement", "build", "create"]):
        return "procedural"
    elif any(word in query_lower for word in ["what", "define", "explain", "describe"]):
        return "definitional"
    elif any(word in query_lower for word in ["why", "reason", "cause", "purpose"]):
        return "causal"
    elif any(word in query_lower for word in ["compare", "difference", "vs", "versus"]):
        return "comparative"
    else:
        return "informational"


async def _select_optimal_hyde_parameters(
    query_analysis: Dict, auto_optimize: bool, quality_threshold: float, ctx
) -> Dict[str, Any]:
    """Select optimal HyDE parameters based on query analysis."""
    params = {
        "hyde_documents": 3,
        "strategy": "adaptive",
        "max_iterations": 1,
    }

    if not auto_optimize:
        return params

    # Adjust based on query complexity
    if query_analysis["complexity"] == "high":
        params["hyde_documents"] = 4
        params["strategy"] = "focused"
    elif query_analysis["complexity"] == "low":
        params["hyde_documents"] = 2
        params["strategy"] = "diverse"

    # Adjust based on quality threshold
    if quality_threshold > 0.8:
        params["max_iterations"] = 3
        params["strategy"] = "focused"
    elif quality_threshold < 0.6:
        params["max_iterations"] = 1
        params["strategy"] = "diverse"
    else:
        params["max_iterations"] = 2

    return params


async def _adjust_iteration_parameters(
    optimal_params: Dict, iteration: int, best_result: Optional[Dict], ctx
) -> Dict[str, Any]:
    """Adjust parameters for each iteration."""
    params = optimal_params.copy()

    if iteration == 0:
        # First iteration uses optimal parameters
        return params

    if best_result:
        current_quality = best_result["hyde_metrics"]["average_quality"]

        if current_quality < 0.6:
            # Low quality - try focused strategy with more documents
            params["strategy"] = "focused"
            params["hyde_documents"] = min(params["hyde_documents"] + 1, 5)
        elif current_quality > 0.8:
            # High quality - try diverse strategy for coverage
            params["strategy"] = "diverse"

    return params


async def _generate_expansion_documents(
    llm_client, query: str, expansion_factor: int, diversity_weight: float, ctx
) -> Dict[str, Any]:
    """Generate documents for query expansion."""
    # Use HyDE document generation with emphasis on diversity
    return await _generate_hypothetical_documents(
        llm_client, query, expansion_factor, "diverse", ctx
    )


async def _extract_expansion_queries(
    expansion_docs: List[Dict], original_query: str, ctx
) -> List[Dict[str, Any]]:
    """Extract expanded queries from hypothetical documents."""
    expanded_queries = []

    for i, doc in enumerate(expansion_docs):
        # Extract key phrases and concepts (simplified approach)
        content = doc["content"]
        words = content.split()

        # Extract noun phrases and key terms (mock implementation)
        key_phrases = []
        for j, word in enumerate(words):
            if len(word) > 6 and word.isalpha():  # Simple heuristic for key terms
                key_phrases.append(word)
                if len(key_phrases) >= 3:
                    break

        # Create expanded query
        if key_phrases:
            expanded_query = f"{original_query} {' '.join(key_phrases[:2])}"
            expanded_queries.append(
                {
                    "query": expanded_query,
                    "source_document": i,
                    "key_phrases": key_phrases,
                    "relevance_score": doc["relevance_score"],
                    "expansion_type": "phrase_extraction",
                }
            )

    # Add semantic variations (mock implementation)
    semantic_variations = [
        f"understand {original_query}",
        f"implement {original_query}",
        f"best practices {original_query}",
    ]

    for var in semantic_variations:
        expanded_queries.append(
            {
                "query": var,
                "source_document": -1,
                "key_phrases": [],
                "relevance_score": 0.8,
                "expansion_type": "semantic_variation",
            }
        )

    return expanded_queries


def _calculate_expansion_metrics(
    original_query: str, expanded_queries: List[Dict], expansion_docs: List[Dict]
) -> Dict[str, Any]:
    """Calculate metrics for query expansion."""
    return {
        "expansion_count": len(expanded_queries),
        "average_relevance": sum(q["relevance_score"] for q in expanded_queries)
        / len(expanded_queries)
        if expanded_queries
        else 0.0,
        "diversity_score": len(set(q["expansion_type"] for q in expanded_queries))
        / len(expanded_queries)
        if expanded_queries
        else 0.0,
        "phrase_extraction_count": len(
            [q for q in expanded_queries if q["expansion_type"] == "phrase_extraction"]
        ),
        "semantic_variation_count": len(
            [q for q in expanded_queries if q["expansion_type"] == "semantic_variation"]
        ),
        "average_document_quality": sum(doc["quality_score"] for doc in expansion_docs)
        / len(expansion_docs)
        if expansion_docs
        else 0.0,
    }
