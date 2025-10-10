"""Tests for dependency singleton cache management."""

from src.services import dependencies


def test_reset_dependency_singletons_clears_cached_instances() -> None:
    """Cached helpers should be cleared when reset is invoked."""

    dependencies._automation_router_instance = object()  # type: ignore[attr-defined]
    dependencies._health_manager_instance = object()  # type: ignore[attr-defined]

    dependencies.reset_dependency_singletons()

    assert dependencies._automation_router_instance is None  # type: ignore[attr-defined]
    assert dependencies._health_manager_instance is None  # type: ignore[attr-defined]
