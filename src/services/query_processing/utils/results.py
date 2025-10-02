"""Result post-processing helpers shared across services."""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterable
from typing import Any, TypeVar

import numpy as np

from .similarity import distance_for_metric
from .text import STOP_WORDS


_TOKEN_PATTERN = re.compile(r"\b\w+\b")


T = TypeVar("T")


def _tokenize(content: str, stop_words: Collection[str]) -> set[str]:
    """Tokenize text content while filtering stop words.

    Args:
        content: Text to tokenize.
        stop_words: Words to exclude from tokens.

    Returns:
        Set of filtered tokens.
    """

    normalized_stop_words = {word.lower() for word in stop_words}
    return {
        match.group(0).lower()
        for match in _TOKEN_PATTERN.finditer(content)
        if match.group(0).lower() not in normalized_stop_words
    }


def _text_similarity(
    left: str,
    right: str,
    *,
    stop_words: Collection[str],
) -> float:
    """Calculate Jaccard similarity between two text strings using token overlap.

    Args:
        left: First text string for comparison.
        right: Second text string for comparison.
        stop_words: Collection of words to exclude from tokenization.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    left_tokens = _tokenize(left, stop_words)
    right_tokens = _tokenize(right, stop_words)

    if not left_tokens or not right_tokens:
        return 0.0

    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return intersection / union if union else 0.0


def deduplicate_results(
    results: Iterable[T],
    *,
    content_getter: Callable[[T], str | None],
    threshold: float = 0.9,
    stop_words: Collection[str] | None = None,
    embedding_getter: Callable[[T], np.ndarray | None] | None = None,
    metric: str | Any = "cosine",
) -> list[T]:
    """Remove near-duplicate results using embeddings and text overlap.

    Args:
        results: Iterable of result objects to deduplicate.
        content_getter: Callable that extracts textual content from objects.
        threshold: Similarity threshold in the range [0, 1]. Values below the
            threshold are considered distinct.
        stop_words: Optional custom stop-word collection used for tokenization.
        embedding_getter: Optional callable producing numpy arrays for vector
            similarity comparison.
        metric: Similarity metric name compatible with ``distance_for_metric``.

    Returns:
        List of results with duplicates removed in iteration order.
    """

    if threshold >= 1.0:
        return list(results)

    stop_words = stop_words or STOP_WORDS
    deduplicated: list[T] = []

    for candidate in results:
        candidate_content = content_getter(candidate) or ""
        candidate_embedding = (
            embedding_getter(candidate) if embedding_getter is not None else None
        )

        is_duplicate = False
        for existing in deduplicated:
            if candidate_embedding is not None and embedding_getter is not None:
                existing_embedding = embedding_getter(existing)
                if existing_embedding is not None:
                    distance = distance_for_metric(
                        candidate_embedding, existing_embedding, metric
                    )
                    similarity = 1.0 - float(distance)
                    if similarity >= threshold:
                        is_duplicate = True
                        break

            existing_content = content_getter(existing) or ""
            if candidate_content and existing_content:
                similarity = _text_similarity(
                    candidate_content, existing_content, stop_words=stop_words
                )
                if similarity >= threshold:
                    is_duplicate = True
                    break

        if not is_duplicate:
            deduplicated.append(candidate)

    return deduplicated


def merge_performance_metadata(
    *,
    performance_stats: dict[str, Any],
    cache_tracker: Any,
    cache_size: int,
) -> dict[str, Any]:
    """Combine performance tracker output with cache statistics.

    Args:
        performance_stats: Serialized tracker statistics.
        cache_tracker: CacheTracker instance exposing hit/miss counters.
        cache_size: Current size of the cache backing the service.

    Returns:
        Dictionary merging performance metrics with cache data.
    """

    return {
        **performance_stats,
        "cache_stats": {
            "hits": getattr(cache_tracker, "hits", 0),
            "misses": getattr(cache_tracker, "misses", 0),
        },
        "cache_size": cache_size,
    }
