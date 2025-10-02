"""Similarity and distance helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..clustering import SimilarityMetric


def _normalize_metric(metric: str | SimilarityMetric) -> str:
    """Normalize metric name to lowercase string.

    Args:
        metric: Metric name or enum value.

    Returns:
        Normalized metric name as lowercase string.
    """
    # Check if metric is an enum member by checking if it has a 'value' attribute
    value = getattr(metric, "value", None)
    if value is not None:
        return str(value).lower()
    return str(metric).lower()


def distance_for_metric(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    metric: str | SimilarityMetric,
) -> float:
    """Compute distance between two vectors using the requested metric.

    Args:
        vector_a: First vector.
        vector_b: Second vector.
        metric: Distance metric to use (cosine, manhattan, or euclidean).

    Returns:
        Distance value between vectors.
    """

    normalized = _normalize_metric(metric)

    if normalized == "cosine":
        norm_a = np.linalg.norm(vector_a) or 1.0
        norm_b = np.linalg.norm(vector_b) or 1.0
        cosine_similarity = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
        return float(1.0 - cosine_similarity)

    if normalized == "manhattan":
        return float(np.sum(np.abs(vector_a - vector_b)))

    return float(np.linalg.norm(vector_a - vector_b))


def sklearn_metric_for(metric: str | SimilarityMetric) -> str:
    """Return the metric name expected by scikit-learn.

    Args:
        metric: Metric name or enum value.

    Returns:
        Scikit-learn compatible metric name.
    """

    normalized = _normalize_metric(metric)

    if normalized == "cosine":
        return "cosine"
    if normalized == "manhattan":
        return "manhattan"
    return "euclidean"
