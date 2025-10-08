"""Utility helpers for content intelligence services."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, sqrt


__all__ = ["cosine_similarity"]


def cosine_similarity(
    vec1: Sequence[float | int | str], vec2: Sequence[float | int | str]
) -> float:
    """Calculate the cosine similarity between two numeric vectors.

    Args:
        vec1: First vector of numeric components.
        vec2: Second vector of numeric components.

    Returns:
        float: Cosine similarity value in the range [-1.0, 1.0].
    """

    try:
        vector_one = tuple(float(component) for component in vec1)
        vector_two = tuple(float(component) for component in vec2)
    except (TypeError, ValueError):
        return 0.0

    if not vector_one or not vector_two:
        return 0.0

    if not all(isfinite(value) for value in vector_one + vector_two):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vector_one, vector_two, strict=False))
    magnitude_one = sqrt(sum(component * component for component in vector_one))
    magnitude_two = sqrt(sum(component * component for component in vector_two))

    if magnitude_one == 0.0 or magnitude_two == 0.0:
        return 0.0

    return dot_product / (magnitude_one * magnitude_two)
