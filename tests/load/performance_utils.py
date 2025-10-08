"""Shared performance-related utilities for load testing."""

from __future__ import annotations

from collections.abc import Iterable


def grade_from_score(score: float) -> str:
    """Convert a numeric performance score into a letter grade."""
    thresholds: Iterable[tuple[float, str]] = (
        (90, "A"),
        (80, "B"),
        (70, "C"),
        (60, "D"),
    )

    for threshold, grade in thresholds:
        if score >= threshold:
            return grade
    return "F"
