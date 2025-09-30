"""Lean validation for the remaining core constants."""

from __future__ import annotations

import pytest

from src.core import constants


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("DEFAULT_REQUEST_TIMEOUT", 30.0),
        ("DEFAULT_CACHE_TTL", 3600),
        ("DEFAULT_CHUNK_SIZE", 1600),
        ("DEFAULT_CHUNK_OVERLAP", 320),
        ("MAX_RETRIES", 3),
    ],
)
def test_core_defaults(name: str, expected: float) -> None:
    """Core defaults should remain stable for consumer tooling."""
    assert getattr(constants, name) == expected


def test_search_dimension_bounds() -> None:
    """Vector dimension bounds guard against invalid embedding sizes."""
    assert constants.MIN_VECTOR_DIMENSIONS < constants.MAX_VECTOR_DIMENSIONS
    assert constants.DEFAULT_VECTOR_DIMENSIONS in constants.COMMON_VECTOR_DIMENSIONS


def test_supported_extensions_cover_common_formats() -> None:
    """Ensure the supported extension map includes documentation staples."""
    extensions = constants.SUPPORTED_EXTENSIONS
    assert {".md", ".txt", ".html"} <= set(extensions)


def test_quality_threshold_shapes() -> None:
    """Structured threshold dictionaries should expose the final surface keys."""
    assert set(constants.QUALITY_THRESHOLDS) == {"fast", "balanced", "best"}
    assert set(constants.SPEED_THRESHOLDS) == {"fast", "balanced", "slow"}
    assert set(constants.COST_THRESHOLDS) == {"cheap", "moderate", "expensive"}


def test_budget_thresholds_are_ordered() -> None:
    """Budget thresholds define warning and critical trip-points."""
    assert (
        0 < constants.BUDGET_WARNING_THRESHOLD < constants.BUDGET_CRITICAL_THRESHOLD < 1
    )
