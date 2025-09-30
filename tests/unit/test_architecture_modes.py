"""Focused tests for the dual-mode architecture helpers."""

from __future__ import annotations

from typing import Any

import pytest

from src.architecture.features import (
    FeatureFlag,
    ModeAwareFeatureManager,
    conditional_feature,
    enterprise_only,
    get_feature_config,
    register_feature,
    service_required,
)
from src.architecture.modes import (
    ENTERPRISE_MODE_CONFIG,
    SIMPLE_MODE_CONFIG,
    ApplicationMode,
    detect_mode_from_environment,
    get_enabled_services,
    get_feature_setting,
    get_resource_limit,
)


@pytest.fixture(autouse=True)
def reset_mode_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure each test starts from a clean mode configuration."""
    monkeypatch.delenv("AI_DOCS_MODE", raising=False)
    manager = ModeAwareFeatureManager()
    monkeypatch.setattr("src.architecture.features._feature_manager", manager)


@pytest.mark.parametrize(
    ("env", "expected"),
    [("simple", ApplicationMode.SIMPLE), ("enterprise", ApplicationMode.ENTERPRISE)],
)
def test_detect_mode_from_environment(
    monkeypatch: pytest.MonkeyPatch, env: str, expected: ApplicationMode
) -> None:
    """Mode detection should follow the configured environment variable."""
    monkeypatch.setenv("AI_DOCS_MODE", env)
    assert detect_mode_from_environment() is expected


@pytest.mark.parametrize(
    ("mode", "service", "is_enabled"),
    [
        (ApplicationMode.SIMPLE, "basic_search", True),
        (ApplicationMode.SIMPLE, "advanced_search", False),
        (ApplicationMode.ENTERPRISE, "advanced_search", True),
    ],
)
def test_mode_configuration_service_matrix(
    monkeypatch: pytest.MonkeyPatch,
    mode: ApplicationMode,
    service: str,
    is_enabled: bool,
) -> None:
    """Enabled services should reflect the expected complexity tier."""
    monkeypatch.setenv("AI_DOCS_MODE", mode.value)
    services = set(get_enabled_services())
    assert (service in services) is is_enabled


@pytest.mark.parametrize(
    ("mode", "feature", "expected"),
    [
        (ApplicationMode.SIMPLE, "enable_hybrid_search", False),
        (ApplicationMode.ENTERPRISE, "enable_hybrid_search", True),
    ],
)
def test_feature_flags_match_mode_settings(
    monkeypatch: pytest.MonkeyPatch, mode: ApplicationMode, feature: str, expected: bool
) -> None:
    """Feature flags mirror the configuration selected for the current mode."""
    monkeypatch.setenv("AI_DOCS_MODE", mode.value)
    assert get_feature_setting(feature) is expected


def test_feature_flag_reports_current_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """FeatureFlag helpers reflect the current runtime mode."""
    monkeypatch.setenv("AI_DOCS_MODE", ApplicationMode.SIMPLE.value)
    flag = FeatureFlag(SIMPLE_MODE_CONFIG)
    assert flag.is_simple_mode() is True
    assert flag.is_enterprise_mode() is False

    monkeypatch.setenv("AI_DOCS_MODE", ApplicationMode.ENTERPRISE.value)
    flag = FeatureFlag(ENTERPRISE_MODE_CONFIG)
    assert flag.is_simple_mode() is False
    assert flag.is_enterprise_mode() is True


def test_enterprise_only_decorator_respects_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enterprise-only blocks should return fallbacks when disabled."""

    @enterprise_only(fallback_value="fallback")
    def expensive_feature() -> str:
        return "enterprise"

    monkeypatch.setenv("AI_DOCS_MODE", "simple")
    assert expensive_feature() == "fallback"

    monkeypatch.setenv("AI_DOCS_MODE", "enterprise")
    assert expensive_feature() == "enterprise"


def test_conditional_feature_allows_mode_specific_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conditional features honour the configured capability switches."""

    @conditional_feature("enable_hybrid_search", fallback_value={"mode": "dense"})
    def select_strategy() -> dict[str, Any]:
        return {"mode": "hybrid"}

    monkeypatch.setenv("AI_DOCS_MODE", "simple")
    assert select_strategy() == {"mode": "dense"}

    monkeypatch.setenv("AI_DOCS_MODE", "enterprise")
    assert select_strategy() == {"mode": "hybrid"}


def test_service_required_gates_unavailable_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """service_required should return fallbacks when the service is unavailable."""

    @service_required("advanced_search", fallback_value=None)
    def execute_pipeline() -> str | None:
        return "complete"

    monkeypatch.setenv("AI_DOCS_MODE", "simple")
    assert execute_pipeline() is None

    monkeypatch.setenv("AI_DOCS_MODE", "enterprise")
    assert execute_pipeline() == "complete"


def test_mode_aware_feature_manager_serves_registered_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The feature manager should return mode-specific registrations."""
    feature_name = "agentic_search"
    register_feature(
        feature_name,
        {"enabled": False, "parallelism": 1},
        {"enabled": True, "parallelism": 4},
    )

    monkeypatch.setenv("AI_DOCS_MODE", "simple")
    simple_config = get_feature_config(feature_name)
    assert simple_config == {"enabled": False, "parallelism": 1}

    monkeypatch.setenv("AI_DOCS_MODE", "enterprise")
    enterprise_config = get_feature_config(feature_name)
    assert enterprise_config == {"enabled": True, "parallelism": 4}


def test_mode_resource_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resource budgets should reflect each deployment mode."""
    monkeypatch.setenv("AI_DOCS_MODE", "simple")
    simple_limit = get_resource_limit("max_concurrent_requests")

    monkeypatch.setenv("AI_DOCS_MODE", "enterprise")
    enterprise_limit = get_resource_limit("max_concurrent_requests")

    assert simple_limit < enterprise_limit
