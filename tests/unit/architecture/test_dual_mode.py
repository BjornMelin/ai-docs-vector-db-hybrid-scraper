"""Unit tests for dual-mode architecture utilities."""

# pylint: disable=duplicate-code

from __future__ import annotations

from collections.abc import Generator
from types import SimpleNamespace
from typing import cast

import pytest

from src.architecture.features import (
    FeatureFlag,
    ModeAwareFeatureManager,
    conditional_feature,
    enterprise_only,
    get_feature_config,
    get_feature_manager,
    register_feature,
    service_required,
)
from src.architecture.modes import (
    ENTERPRISE_MODE_CONFIG,
    SIMPLE_MODE_CONFIG,
    ApplicationMode,
    get_mode_config,
    resolve_mode,
)
from src.architecture.service_factory import ModeAwareServiceFactory
from src.config import get_config, reset_config, set_config
from src.config.models import Environment


EXPECTED_SIMPLE_SERVICES = {
    "qdrant_client",
    "embedding_service",
    "basic_search",
    "simple_caching",
}
EXPECTED_ENTERPRISE_SERVICES = {
    "qdrant_client",
    "embedding_service",
    "advanced_search",
    "multi_tier_caching",
    "deployment_services",
    "advanced_analytics",
}
FORBIDDEN_SIMPLE_SERVICES = {
    "advanced_analytics",
    "deployment_services",
    "multi_tier_caching",
    "a_b_testing",
}


@pytest.fixture(autouse=True)
def _install_default_config(
    config_factory, monkeypatch: pytest.MonkeyPatch
) -> Generator[None, None, None]:
    """Provision an isolated Config instance for each test."""
    monkeypatch.delenv("AI_DOCS_MODE", raising=False)
    set_config(
        config_factory(
            mode=ApplicationMode.SIMPLE,
            environment=Environment.TESTING,
        )
    )
    yield
    reset_config()


def _install_flag_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enterprise: bool = False,
    feature_enabled: bool = False,
    service_enabled: bool = False,
) -> None:
    """Replace FeatureFlag with a stub that returns deterministic results."""

    class _Stub:
        def is_enterprise_mode(self) -> bool:
            return enterprise

        def is_simple_mode(self) -> bool:
            return not enterprise

        def is_feature_enabled(self, _name: str) -> bool:
            return feature_enabled

        def is_service_enabled(self, _name: str) -> bool:
            return service_enabled

    monkeypatch.setattr(
        "src.architecture.features.FeatureFlag", lambda *_, **__: _Stub()
    )


class TestModeConfiguration:
    """Validate simple and enterprise configuration characteristics."""

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            (SIMPLE_MODE_CONFIG, EXPECTED_SIMPLE_SERVICES),
            (ENTERPRISE_MODE_CONFIG, EXPECTED_ENTERPRISE_SERVICES),
        ],
    )
    def test_config_contains_expected_services(
        self, config, expected: set[str]
    ) -> None:
        enabled = set(config.enabled_services)
        assert expected <= enabled

    def test_simple_mode_excludes_enterprise_capabilities(self) -> None:
        enabled = set(SIMPLE_MODE_CONFIG.enabled_services)
        assert enabled.isdisjoint(FORBIDDEN_SIMPLE_SERVICES)

    @pytest.mark.parametrize(
        "resource",
        ["max_concurrent_requests", "max_memory_usage_mb", "cache_size_mb"],
    )
    def test_simple_resource_limits_are_lower(self, resource: str) -> None:
        assert (
            SIMPLE_MODE_CONFIG.resource_limits[resource]
            < ENTERPRISE_MODE_CONFIG.resource_limits[resource]
        )

    def test_simple_has_smaller_middleware_stack(self) -> None:
        assert len(SIMPLE_MODE_CONFIG.middleware_stack) < len(
            ENTERPRISE_MODE_CONFIG.middleware_stack
        )


class TestModeDetection:
    """Validate environment-driven mode detection and derived helpers."""

    @pytest.mark.parametrize(
        "expected",
        [
            ApplicationMode.SIMPLE,
            ApplicationMode.ENTERPRISE,
        ],
    )
    def test_resolve_mode_reflects_settings(self, expected: ApplicationMode) -> None:
        current = get_config()
        set_config(current.model_copy(update={"mode": expected}))
        assert resolve_mode() is expected

    def test_resolve_mode_defaults_to_current_settings(self) -> None:
        assert resolve_mode() is ApplicationMode.SIMPLE

    def test_get_mode_config_auto_uses_detection(
        self,
    ) -> None:
        set_config(get_config().model_copy(update={"mode": ApplicationMode.ENTERPRISE}))
        assert get_mode_config() == ENTERPRISE_MODE_CONFIG

    def test_helper_functions_reflect_mode(self) -> None:
        set_config(get_config().model_copy(update={"mode": ApplicationMode.ENTERPRISE}))
        assert resolve_mode() is ApplicationMode.ENTERPRISE
        enterprise = get_mode_config(config=get_config())
        assert "advanced_search" in enterprise.enabled_services
        assert enterprise.max_complexity_features["enable_advanced_monitoring"] is True
        assert enterprise.resource_limits["max_concurrent_requests"] == 100

        set_config(get_config().model_copy(update={"mode": ApplicationMode.SIMPLE}))
        assert resolve_mode() is ApplicationMode.SIMPLE
        simple = get_mode_config(config=get_config())
        assert "advanced_search" not in simple.enabled_services
        assert simple.max_complexity_features["enable_advanced_monitoring"] is False
        assert simple.resource_limits["max_concurrent_requests"] == 5


class TestFeatureFlag:
    """Validate FeatureFlag helpers against both configurations."""

    def test_feature_flag_respects_modes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "src.architecture.features.resolve_mode", lambda *_: ApplicationMode.SIMPLE
        )
        flag = FeatureFlag(SIMPLE_MODE_CONFIG)
        assert flag.is_simple_mode() is True
        assert flag.is_enterprise_mode() is False
        assert flag.is_feature_enabled("enable_advanced_monitoring") is False

        monkeypatch.setattr(
            "src.architecture.features.resolve_mode",
            lambda *_: ApplicationMode.ENTERPRISE,
        )
        flag = FeatureFlag(ENTERPRISE_MODE_CONFIG)
        assert flag.is_simple_mode() is False
        assert flag.is_enterprise_mode() is True
        assert flag.is_feature_enabled("enable_advanced_monitoring") is True
        assert flag.is_service_enabled("advanced_analytics") is True


class TestDecorators:
    """Validate decorator behaviours for mode and feature enforcement."""

    @pytest.mark.asyncio
    async def test_enterprise_only_async(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_flag_stub(monkeypatch, enterprise=True)

        @enterprise_only(fallback_value="simple")
        async def privileged() -> str:
            return "enterprise"

        assert await privileged() == "enterprise"

        _install_flag_stub(monkeypatch, enterprise=False)
        assert await privileged() == "simple"

    def test_enterprise_only_sync(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_flag_stub(monkeypatch, enterprise=True)

        @enterprise_only(fallback_value="simple")
        def privileged() -> str:
            return "enterprise"

        assert privileged() == "enterprise"

        _install_flag_stub(monkeypatch, enterprise=False)
        assert privileged() == "simple"

    @pytest.mark.asyncio
    async def test_conditional_feature_async(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_flag_stub(monkeypatch, feature_enabled=True)

        @conditional_feature("enable_advanced_monitoring", fallback_value="disabled")
        async def monitored() -> str:
            return "enabled"

        assert await monitored() == "enabled"

        _install_flag_stub(monkeypatch, feature_enabled=False)
        assert await monitored() == "disabled"

    @pytest.mark.asyncio
    async def test_service_required_async(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_flag_stub(monkeypatch, service_enabled=True)

        @service_required("rag_service", fallback_value="missing")
        async def service_call() -> str:
            return "available"

        assert await service_call() == "available"

        _install_flag_stub(monkeypatch, service_enabled=False)
        assert await service_call() == "missing"


class TestServiceFactory:
    """Validate service factory registration and inspection helpers."""

    def _factory_with_copy(self, mode: ApplicationMode) -> ModeAwareServiceFactory:
        factory = ModeAwareServiceFactory(mode)
        factory.mode_config = factory.mode_config.model_copy(deep=True)
        return factory

    def test_auto_detect_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "src.architecture.service_factory.resolve_mode",
            lambda: ApplicationMode.ENTERPRISE,
        )
        factory = ModeAwareServiceFactory()
        assert factory.mode is ApplicationMode.ENTERPRISE
        assert factory.mode_config == ENTERPRISE_MODE_CONFIG

    def test_service_registration(self) -> None:
        factory = self._factory_with_copy(ApplicationMode.SIMPLE)

        class SimpleService:
            async def initialize(
                self,
            ) -> None:  # pragma: no cover - interface requirement
                return None

            async def cleanup(self) -> None:  # pragma: no cover - interface requirement
                return None

            def get_service_name(
                self,
            ) -> str:  # pragma: no cover - interface requirement
                return "simple"

        class EnterpriseService(SimpleService):
            pass

        factory.register_service("test_service", SimpleService, EnterpriseService)
        implementations = factory.get_registered_service_implementations("test_service")
        assert implementations["simple"] is SimpleService
        assert implementations["enterprise"] is EnterpriseService

    def test_service_availability_helpers(self) -> None:
        factory = self._factory_with_copy(ApplicationMode.SIMPLE)
        factory.mode_config.enabled_services.append("test_service")

        class Impl:
            async def initialize(
                self,
            ) -> None:  # pragma: no cover - interface requirement
                return None

            async def cleanup(self) -> None:  # pragma: no cover - interface requirement
                return None

            def get_service_name(
                self,
            ) -> str:  # pragma: no cover - interface requirement
                return "impl"

        factory.register_service("test_service", Impl, Impl)
        assert factory.is_service_available("test_service") is True
        assert "test_service" in factory.get_available_services()
        status = factory.get_service_status("test_service")
        assert status == {
            "name": "test_service",
            "available": True,
            "enabled": True,
            "initialized": False,
            "mode": "simple",
        }

    def test_get_mode_info(self) -> None:
        factory = self._factory_with_copy(ApplicationMode.ENTERPRISE)
        info = factory.get_mode_info()
        assert info["mode"] == "enterprise"
        assert info["enabled_services"] == ENTERPRISE_MODE_CONFIG.enabled_services
        assert info["resource_limits"] == ENTERPRISE_MODE_CONFIG.resource_limits
        assert info["advanced_monitoring"] is True
        assert info["deployment_features"] is True


class TestFeatureManager:
    """Validate the global feature manager helpers."""

    @pytest.fixture
    def feature_manager(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> ModeAwareFeatureManager:
        manager = ModeAwareFeatureManager()
        manager._feature_flags = cast(
            FeatureFlag, SimpleNamespace(is_enterprise_mode=lambda: False)
        )
        monkeypatch.setattr("src.architecture.features._feature_manager", manager)
        return manager

    def test_register_and_retrieve_feature_configs(
        self, feature_manager: ModeAwareFeatureManager
    ) -> None:
        register_feature(
            "hybrid_search",
            {"enabled": True, "top_k": 5},
            {"enabled": True, "top_k": 20},
        )

        config = get_feature_config("hybrid_search")
        assert config == {"enabled": True, "top_k": 5}

        feature_manager._feature_flags = cast(
            FeatureFlag, SimpleNamespace(is_enterprise_mode=lambda: True)
        )
        config = get_feature_config("hybrid_search")
        assert config == {"enabled": True, "top_k": 20}

    def test_is_feature_available(
        self, feature_manager: ModeAwareFeatureManager
    ) -> None:
        register_feature(
            "observability",
            {"enabled": False},
            {"enabled": True},
        )
        assert feature_manager.is_feature_available("observability") is False

        feature_manager._feature_flags = cast(
            FeatureFlag, SimpleNamespace(is_enterprise_mode=lambda: True)
        )
        assert feature_manager.is_feature_available("observability") is True

        assert get_feature_manager() is feature_manager
