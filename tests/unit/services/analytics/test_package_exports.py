"""Tests covering the analytics service package exports."""

from __future__ import annotations

import pytest

from src.services import analytics as analytics_module
from src.services.analytics import SearchAnalyticsDashboard, VectorVisualizationEngine


@pytest.mark.parametrize(
    ("export_name", "expected_object"),
    (
        ("SearchAnalyticsDashboard", SearchAnalyticsDashboard),
        ("VectorVisualizationEngine", VectorVisualizationEngine),
    ),
)
def test_analytics_package_exports(
    export_name: str, expected_object: type[object]
) -> None:
    """Confirm analytics exports resolve the curated public services.

    Args:
        export_name: Symbol exposed through :mod:`src.services.analytics`.
        expected_object: Implementation type imported via the package facade.

    Returns:
        None: This test asserts the module returns the canonical implementations.
    """

    exported_object = getattr(analytics_module, export_name)

    assert exported_object is expected_object


def test_analytics_all_exports() -> None:
    """Validate ``__all__`` enumerates the public analytics API surface.

    Also asserts that ``__all__`` is a tuple and is immutable.
    """
    all_exports = getattr(analytics_module, "__all__")

    # Assert __all__ is a tuple
    assert isinstance(all_exports, tuple)

    # Assert __all__ is immutable
    with pytest.raises(TypeError):
        all_exports += ("new_export",)
    with pytest.raises(TypeError):
        all_exports[0] = "something_else"

    Returns:
        None: This test ensures the package boundary remains intentional.
    """

    expected_exports = (
        "SearchAnalyticsDashboard",
        "VectorVisualizationEngine",
    )

    assert analytics_module.__all__ == expected_exports
