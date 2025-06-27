"""Function-based embedding service with FastAPI dependency injection.

Transforms the complex EmbeddingManager class into pure functions with
dependency injection. Maintains all functionality while improving testability.
"""

import logging
from typing import Annotated, Any, Dict, List

from fastapi import Depends, HTTPException

from ..embeddings.manager import QualityTier, TextAnalysis
from .circuit_breaker import CircuitBreakerConfig, circuit_breaker
from .dependencies import get_embedding_client


logger = logging.getLogger(__name__)


@circuit_breaker(CircuitBreakerConfig.simple_mode())
async def generate_embeddings(
    texts: list[str],
    quality_tier: QualityTier | None = None,
    provider_name: str | None = None,
    max_cost: float | None = None,
    speed_priority: bool = False,
    auto_select: bool = True,
    generate_sparse: bool = False,
    embedding_client: Annotated[object, Depends(get_embedding_client)] = None,
) -> dict[str, Any]:
    """Generate embeddings with smart provider selection.

    Pure function replacement for EmbeddingManager.generate_embeddings().
    Uses dependency injection instead of instance state.

    Args:
        texts: Text strings to embed
        quality_tier: Quality tier (FAST, BALANCED, BEST)
        provider_name: Explicit provider ("openai" or "fastembed")
        max_cost: Optional maximum cost constraint
        speed_priority: Whether to prioritize speed over quality
        auto_select: Use smart selection (True) or legacy logic (False)
        generate_sparse: Whether to generate sparse embeddings
        embedding_client: Injected embedding manager

    Returns:
        Dictionary containing embeddings and metadata

    Raises:
        HTTPException: If embedding generation fails
    """
    try:
        if not embedding_client:
            raise HTTPException(
                status_code=500, detail="Embedding client not available"
            )

        result = await embedding_client.generate_embeddings(
            texts=texts,
            quality_tier=quality_tier,
            provider_name=provider_name,
            max_cost=max_cost,
            speed_priority=speed_priority,
            auto_select=auto_select,
            generate_sparse=generate_sparse,
        )

        logger.info(
            f"Generated embeddings for {len(texts)} texts "
            f"using {result.get('provider', 'unknown')} provider"
        )

    except Exception as e:
        logger.exception("Embedding generation failed")
        raise HTTPException(
            status_code=500, detail=f"Embedding generation failed: {e!s}"
        )
    else:
        return result


@circuit_breaker(CircuitBreakerConfig.simple_mode())
async def rerank_results(
    query: str,
    results: list[dict[str, Any]],
    embedding_client: Annotated[object, Depends(get_embedding_client)] = None,
) -> list[dict[str, Any]]:
    """Rerank search results using BGE reranker.

    Pure function replacement for EmbeddingManager.rerank_results().

    Args:
        query: Search query
        results: List of search results with 'content' field
        embedding_client: Injected embedding manager

    Returns:
        Reranked results sorted by relevance

    Raises:
        HTTPException: If reranking fails
    """
    try:
        if not embedding_client:
            logger.warning("Embedding client not available, returning original results")
            return results

        reranked = await embedding_client.rerank_results(query, results)
        logger.info(f"Reranked {len(results)} results")

    except Exception as e:
        logger.exception("Reranking failed")
        # Return original results on failure (graceful degradation)
        return results
    else:
        return reranked


async def analyze_text_characteristics(
    texts: list[str],
    embedding_client: Annotated[object, Depends(get_embedding_client)] = None,
) -> TextAnalysis:
    """Analyze text characteristics for smart model selection.

    Pure function replacement for EmbeddingManager.analyze_text_characteristics().

    Args:
        texts: List of texts to analyze
        embedding_client: Injected embedding manager

    Returns:
        TextAnalysis with characteristics data

    Raises:
        HTTPException: If analysis fails
    """
    try:
        if not embedding_client:
            raise HTTPException(
                status_code=500, detail="Embedding client not available"
            )

        analysis = embedding_client.analyze_text_characteristics(texts)

        logger.debug(
            f"Analyzed {len(texts)} texts: "
            f"type={analysis.text_type}, "
            f"complexity={analysis.complexity_score:.2f}"
        )

    except Exception as e:
        logger.exception("Text analysis failed")
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {e!s}")
    else:
        return analysis


async def estimate_embedding_cost(
    texts: list[str],
    provider_name: str | None = None,
    embedding_client: Annotated[object, Depends(get_embedding_client)] = None,
) -> dict[str, Dict[str, float]]:
    """Estimate embedding generation cost.

    Pure function replacement for EmbeddingManager.estimate_cost().

    Args:
        texts: List of texts to estimate cost for
        provider_name: Optional specific provider to estimate
        embedding_client: Injected embedding manager

    Returns:
        Cost estimation per provider

    Raises:
        HTTPException: If cost estimation fails
    """
    try:
        if not embedding_client:
            raise HTTPException(
                status_code=500, detail="Embedding client not available"
            )

        costs = embedding_client.estimate_cost(texts, provider_name)
        logger.debug(f"Estimated costs for {len(texts)} texts")

    except Exception as e:
        logger.exception("Cost estimation failed")
        raise HTTPException(status_code=500, detail=f"Cost estimation failed: {e!s}")
    else:
        return costs


async def get_provider_info(
    embedding_client: Annotated[object, Depends(get_embedding_client)] = None,
) -> dict[str, Dict[str, Any]]:
    """Get information about available embedding providers.

    Pure function replacement for EmbeddingManager.get_provider_info().

    Args:
        embedding_client: Injected embedding manager

    Returns:
        Provider information dictionary

    Raises:
        HTTPException: If provider info retrieval fails
    """
    try:
        if not embedding_client:
            return {}

        info = embedding_client.get_provider_info()
        logger.debug(f"Retrieved info for {len(info)} providers")

    except Exception as e:
        logger.exception("Provider info retrieval failed")
        raise HTTPException(
            status_code=500, detail=f"Provider info retrieval failed: {e!s}"
        )
    else:
        return info


async def get_smart_recommendation(
    texts: list[str],
    quality_tier: QualityTier | None = None,
    max_cost: float | None = None,
    speed_priority: bool = False,
    embedding_client: Annotated[object, Depends(get_embedding_client)] = None,
) -> dict[str, Any]:
    """Get smart provider recommendation based on text analysis.

    Pure function that combines text analysis with provider recommendation.

    Args:
        texts: List of texts to analyze
        quality_tier: Optional quality tier override
        max_cost: Optional maximum cost constraint
        speed_priority: Whether to prioritize speed over quality
        embedding_client: Injected embedding manager

    Returns:
        Recommendation with provider, model, cost, and reasoning

    Raises:
        HTTPException: If recommendation fails
    """
    try:
        if not embedding_client:
            raise HTTPException(
                status_code=500, detail="Embedding client not available"
            )

        # Analyze text characteristics
        text_analysis = await analyze_text_characteristics(texts, embedding_client)

        # Get smart recommendation
        recommendation = embedding_client.get_smart_provider_recommendation(
            text_analysis=text_analysis,
            quality_tier=quality_tier,
            max_cost=max_cost,
            speed_priority=speed_priority,
        )

        logger.info(
            f"Smart recommendation: {recommendation['provider']}/{recommendation['model']} "
            f"(${recommendation['estimated_cost']:.4f}) - {recommendation['reasoning']}"
        )

    except Exception as e:
        logger.exception("Smart recommendation failed")
        raise HTTPException(
            status_code=500, detail=f"Smart recommendation failed: {e!s}"
        )
    else:
        return recommendation


async def get_usage_report(
    embedding_client: Annotated[object, Depends(get_embedding_client)] = None,
) -> dict[str, Any]:
    """Get comprehensive usage report.

    Pure function replacement for EmbeddingManager.get_usage_report().

    Args:
        embedding_client: Injected embedding manager

    Returns:
        Usage statistics dictionary

    Raises:
        HTTPException: If usage report retrieval fails
    """
    try:
        if not embedding_client:
            return {
                "summary": {"total_requests": 0, "total_cost": 0.0},
                "by_tier": {},
                "by_provider": {},
                "budget": {"daily_limit": None, "daily_usage": 0.0},
            }

        report = embedding_client.get_usage_report()
        logger.debug("Retrieved usage report")

    except Exception as e:
        logger.exception("Usage report retrieval failed")
        raise HTTPException(
            status_code=500, detail=f"Usage report retrieval failed: {e!s}"
        )
    else:
        return report


# Batch processing function (new functionality)
@circuit_breaker(CircuitBreakerConfig.enterprise_mode())
async def batch_generate_embeddings(
    text_batches: list[List[str]],
    quality_tier: QualityTier | None = None,
    max_parallel: int = 3,
    embedding_client: Annotated[object, Depends(get_embedding_client)] = None,
) -> list[dict[str, Any]]:
    """Generate embeddings for multiple batches in parallel.

    New function-based capability that demonstrates composition patterns.

    Args:
        text_batches: List of text batches to process
        quality_tier: Quality tier for all batches
        max_parallel: Maximum parallel executions
        embedding_client: Injected embedding manager

    Returns:
        List of embedding results for each batch

    Raises:
        HTTPException: If batch processing fails
    """
    try:
        import asyncio

        semaphore = asyncio.Semaphore(max_parallel)

        async def process_batch(texts: list[str]) -> dict[str, Any]:
            async with semaphore:
                return await generate_embeddings(
                    texts=texts,
                    quality_tier=quality_tier,
                    embedding_client=embedding_client,
                )

        # Process batches in parallel with concurrency control
        tasks = [process_batch(batch) for batch in text_batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch {i} failed: {result}")
                processed_results.append(
                    {
                        "success": False,
                        "error": str(result),
                        "embeddings": [],
                    }
                )
            else:
                processed_results.append(result)

        logger.info(
            f"Processed {len(text_batches)} batches with "
            f"{sum(1 for r in processed_results if r.get('success', True))} successes"
        )

    except Exception as e:
        logger.exception("Batch embedding generation failed")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {e!s}")
    else:
        return processed_results
