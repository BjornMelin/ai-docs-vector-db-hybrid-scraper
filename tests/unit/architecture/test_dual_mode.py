"""Tests for dual-mode architecture functionality."""

import os
from unittest.mock import patch

import pytest

from src.architecture.features import FeatureFlag, conditional_feature, enterprise_only
from src.architecture.modes import (
    ENTERPRISE_MODE_CONFIG,
    SIMPLE_MODE_CONFIG,
    ApplicationMode,
    detect_mode_from_environment,
    get_current_mode,
    get_enabled_services,
    get_feature_setting,
    get_mode_config,
    get_resource_limit,
    is_enterprise_mode,
    is_service_enabled,
    is_simple_mode,
)
from src.architecture.service_factory import ModeAwareServiceFactory


class TestApplicationModes:
    """Test application mode detection and configuration."""

    def test_simple_mode_configuration(self):
        """Test simple mode has appropriate configuration."""
        config = SIMPLE_MODE_CONFIG

        # Check enabled services - should be minimal
        assert "qdrant_client" in config.enabled_services
        assert "embedding_service" in config.enabled_services
        assert "basic_search" in config.enabled_services
        assert "simple_caching" in config.enabled_services

        # Should not have enterprise services
        assert "advanced_analytics" not in config.enabled_services
        assert "deployment_services" not in config.enabled_services
        assert "a_b_testing" not in config.enabled_services

        # Check resource limits are conservative
        assert config.resource_limits["max_concurrent_requests"] == 5
        assert config.resource_limits["max_memory_usage_mb"] == 500
        assert config.resource_limits["cache_size_mb"] == 50

        # Check features are disabled
        assert config.max_complexity_features["enable_advanced_monitoring"] is False
        assert config.max_complexity_features["enable_deployment_features"] is False
        assert config.max_complexity_features["max_concurrent_crawls"] == 5

    def test_enterprise_mode_configuration(self):
        """Test enterprise mode has full feature set."""
        config = ENTERPRISE_MODE_CONFIG

        # Check enabled services - should be comprehensive
        assert "qdrant_client" in config.enabled_services
        assert "embedding_service" in config.enabled_services
        assert "advanced_search" in config.enabled_services
        assert "multi_tier_caching" in config.enabled_services
        assert "deployment_services" in config.enabled_services
        assert "a_b_testing" in config.enabled_services
        assert "advanced_analytics" in config.enabled_services

        # Check resource limits are scaled up
        assert config.resource_limits["max_concurrent_requests"] == 100
        assert config.resource_limits["max_memory_usage_mb"] == 4000
        assert config.resource_limits["cache_size_mb"] == 1000

        # Check features are enabled
        assert config.max_complexity_features["enable_advanced_monitoring"] is True
        assert config.max_complexity_features["enable_deployment_features"] is True
        assert config.max_complexity_features["max_concurrent_crawls"] == 50

    @patch.dict(os.environ, {"AI_DOCS_MODE": "simple"})
    def test_detect_simple_mode_from_environment(self):
        """Test detecting simple mode from environment."""
        mode = detect_mode_from_environment()
        assert mode == ApplicationMode.SIMPLE

    @patch.dict(os.environ, {"AI_DOCS_MODE": "enterprise"})
    def test_detect_enterprise_mode_from_environment(self):
        """Test detecting enterprise mode from environment."""
        mode = detect_mode_from_environment()
        assert mode == ApplicationMode.ENTERPRISE

    @patch.dict(os.environ, {"AI_DOCS_MODE": ""})
    def test_detect_mode_defaults_to_simple(self):
        """Test mode detection defaults to simple when not set."""
        mode = detect_mode_from_environment()
        assert mode == ApplicationMode.SIMPLE

    @patch.dict(os.environ, {"AI_DOCS_MODE": "invalid"})
    def test_detect_mode_handles_invalid_value(self):
        """Test mode detection handles invalid values gracefully."""
        mode = detect_mode_from_environment()
        assert mode == ApplicationMode.SIMPLE

    @patch.dict(os.environ, {"AI_DOCS_DEPLOYMENT__TIER": "enterprise"})
    def test_legacy_environment_variable_support(self):
        """Test support for legacy deployment tier environment variable."""
        # Clear the new variable
        with patch.dict(os.environ, {"AI_DOCS_MODE": ""}, clear=False):
            mode = detect_mode_from_environment()
            assert mode == ApplicationMode.ENTERPRISE

    def test_get_mode_config_simple(self):
        """Test getting simple mode configuration."""
        config = get_mode_config(ApplicationMode.SIMPLE)
        assert config == SIMPLE_MODE_CONFIG

    def test_get_mode_config_enterprise(self):
        """Test getting enterprise mode configuration."""
        config = get_mode_config(ApplicationMode.ENTERPRISE)
        assert config == ENTERPRISE_MODE_CONFIG

    @patch("src.architecture.modes.detect_mode_from_environment")
    def test_get_mode_config_auto_detect(self, mock_detect):
        """Test auto-detecting mode when none provided."""
        mock_detect.return_value = ApplicationMode.ENTERPRISE
        config = get_mode_config()
        assert config == ENTERPRISE_MODE_CONFIG
        mock_detect.assert_called_once()

    @patch("src.architecture.modes.get_current_mode")
    def test_utility_functions(self, mock_current_mode):
        """Test utility functions for mode checking."""
        # Test simple mode utilities
        mock_current_mode.return_value = ApplicationMode.SIMPLE
        assert is_simple_mode() is True
        assert is_enterprise_mode() is False

        # Test enterprise mode utilities
        mock_current_mode.return_value = ApplicationMode.ENTERPRISE
        assert is_simple_mode() is False
        assert is_enterprise_mode() is True

    @patch("src.architecture.modes.get_mode_config")
    def test_service_and_feature_utilities(self, mock_get_config):
        """Test service and feature utility functions."""
        mock_config = SIMPLE_MODE_CONFIG
        mock_get_config.return_value = mock_config

        # Test service utilities
        enabled_services = get_enabled_services()
        assert enabled_services == mock_config.enabled_services

        assert is_service_enabled("qdrant_client") is True
        assert is_service_enabled("advanced_analytics") is False

        # Test feature utilities
        assert get_feature_setting("max_concurrent_crawls") == 5
        assert get_feature_setting("nonexistent_feature", "default") == "default"

        # Test resource utilities
        assert get_resource_limit("max_concurrent_requests") == 5
        assert get_resource_limit("nonexistent_resource", 999) == 999


class TestFeatureFlags:
    """Test feature flag system."""

    def test_feature_flag_initialization(self):
        """Test feature flag initialization."""
        feature_flag = FeatureFlag(SIMPLE_MODE_CONFIG)
        assert feature_flag.mode_config == SIMPLE_MODE_CONFIG

    def test_feature_flag_auto_detect_mode(self):
        """Test feature flag auto-detects mode when config not provided."""
        with patch("src.architecture.features.get_mode_config") as mock_get_config:
            mock_get_config.return_value = ENTERPRISE_MODE_CONFIG
            feature_flag = FeatureFlag()
            mock_get_config.assert_called_once()

    def test_feature_flag_mode_detection(self):
        """Test feature flag mode detection methods."""
        simple_flag = FeatureFlag(SIMPLE_MODE_CONFIG)
        enterprise_flag = FeatureFlag(ENTERPRISE_MODE_CONFIG)

        # Test with mocked current mode
        with patch("src.architecture.features.get_current_mode") as mock_current:
            mock_current.return_value = ApplicationMode.SIMPLE
            assert simple_flag.is_simple_mode() is True
            assert simple_flag.is_enterprise_mode() is False

            mock_current.return_value = ApplicationMode.ENTERPRISE
            assert enterprise_flag.is_simple_mode() is False
            assert enterprise_flag.is_enterprise_mode() is True

    def test_feature_enabled_check(self):
        """Test feature enabled checking."""
        simple_flag = FeatureFlag(SIMPLE_MODE_CONFIG)
        enterprise_flag = FeatureFlag(ENTERPRISE_MODE_CONFIG)

        # Simple mode should have features disabled
        assert simple_flag.is_feature_enabled("enable_advanced_monitoring") is False
        assert simple_flag.is_feature_enabled("enable_deployment_features") is False

        # Enterprise mode should have features enabled
        assert enterprise_flag.is_feature_enabled("enable_advanced_monitoring") is True
        assert enterprise_flag.is_feature_enabled("enable_deployment_features") is True

    def test_service_enabled_check(self):
        """Test service enabled checking."""
        simple_flag = FeatureFlag(SIMPLE_MODE_CONFIG)
        enterprise_flag = FeatureFlag(ENTERPRISE_MODE_CONFIG)

        # Both should have basic services
        assert simple_flag.is_service_enabled("qdrant_client") is True
        assert enterprise_flag.is_service_enabled("qdrant_client") is True

        # Only enterprise should have advanced services
        assert simple_flag.is_service_enabled("advanced_analytics") is False
        assert enterprise_flag.is_service_enabled("advanced_analytics") is True

    @pytest.mark.asyncio
    async def test_enterprise_only_decorator_async(self):
        """Test enterprise_only decorator with async functions."""

        @enterprise_only(fallback_value="simple_result")
        async def test_function():
            return "enterprise_result"

        # Mock enterprise mode
        with patch("src.architecture.features.FeatureFlag") as mock_flag_class:
            mock_flag = mock_flag_class.return_value
            mock_flag.is_enterprise_mode.return_value = True

            result = await test_function()
            assert result == "enterprise_result"

        # Mock simple mode
        with patch("src.architecture.features.FeatureFlag") as mock_flag_class:
            mock_flag = mock_flag_class.return_value
            mock_flag.is_enterprise_mode.return_value = False

            result = await test_function()
            assert result == "simple_result"

    def test_enterprise_only_decorator_sync(self):
        """Test enterprise_only decorator with sync functions."""

        @enterprise_only(fallback_value="simple_result")
        def test_function():
            return "enterprise_result"

        # Mock enterprise mode
        with patch("src.architecture.features.FeatureFlag") as mock_flag_class:
            mock_flag = mock_flag_class.return_value
            mock_flag.is_enterprise_mode.return_value = True

            result = test_function()
            assert result == "enterprise_result"

        # Mock simple mode
        with patch("src.architecture.features.FeatureFlag") as mock_flag_class:
            mock_flag = mock_flag_class.return_value
            mock_flag.is_enterprise_mode.return_value = False

            result = test_function()
            assert result == "simple_result"

    @pytest.mark.asyncio
    async def test_conditional_feature_decorator_async(self):
        """Test conditional_feature decorator with async functions."""

        @conditional_feature("enable_advanced_monitoring", fallback_value="disabled")
        async def test_function():
            return "enabled"

        # Mock feature enabled
        with patch("src.architecture.features.FeatureFlag") as mock_flag_class:
            mock_flag = mock_flag_class.return_value
            mock_flag.is_feature_enabled.return_value = True

            result = await test_function()
            assert result == "enabled"

        # Mock feature disabled
        with patch("src.architecture.features.FeatureFlag") as mock_flag_class:
            mock_flag = mock_flag_class.return_value
            mock_flag.is_feature_enabled.return_value = False

            result = await test_function()
            assert result == "disabled"


class TestServiceFactory:
    """Test mode-aware service factory."""

    def test_service_factory_initialization(self):
        """Test service factory initialization."""
        factory = ModeAwareServiceFactory(ApplicationMode.SIMPLE)
        assert factory.mode == ApplicationMode.SIMPLE
        assert factory.mode_config == SIMPLE_MODE_CONFIG

    def test_service_factory_auto_detect_mode(self):
        """Test service factory auto-detects mode when not provided."""
        with patch("src.architecture.service_factory.get_current_mode") as mock_current:
            mock_current.return_value = ApplicationMode.ENTERPRISE
            factory = ModeAwareServiceFactory()
            assert factory.mode == ApplicationMode.ENTERPRISE
            mock_current.assert_called_once()

    def test_service_registration(self):
        """Test service registration."""
        factory = ModeAwareServiceFactory(ApplicationMode.SIMPLE)

        class SimpleService:
            pass

        class EnterpriseService:
            pass

        factory.register_service("test_service", SimpleService, EnterpriseService)

        assert "test_service" in factory._service_registry
        assert factory._service_registry["test_service"]["simple"] == SimpleService
        assert (
            factory._service_registry["test_service"]["enterprise"] == EnterpriseService
        )

    def test_universal_service_registration(self):
        """Test universal service registration."""
        factory = ModeAwareServiceFactory(ApplicationMode.SIMPLE)

        class UniversalService:
            pass

        factory.register_universal_service("universal_service", UniversalService)

        assert (
            factory._service_registry["universal_service"]["simple"] == UniversalService
        )
        assert (
            factory._service_registry["universal_service"]["enterprise"]
            == UniversalService
        )

    def test_service_availability_check(self):
        """Test service availability checking."""
        factory = ModeAwareServiceFactory(ApplicationMode.SIMPLE)

        class TestService:
            pass

        # Register service and check availability through service list modification
        factory.register_service("test_service", TestService, TestService)

        # Create a copy of enabled services to avoid mutating the global config
        original_services = factory.mode_config.enabled_services.copy()
        factory.mode_config.enabled_services = original_services + ["test_service"]

        assert factory.is_service_available("test_service") is True
        assert factory.is_service_available("nonexistent_service") is False

        # Restore original services to avoid affecting other tests
        factory.mode_config.enabled_services = original_services

    def test_get_available_services(self):
        """Test getting list of available services."""
        factory = ModeAwareServiceFactory(ApplicationMode.SIMPLE)

        class TestService:
            pass

        # Register services
        factory.register_service("service1", TestService, TestService)
        factory.register_service("service2", TestService, TestService)

        # Create a copy and add test services to avoid mutating the global config
        original_services = factory.mode_config.enabled_services.copy()
        factory.mode_config.enabled_services = original_services + [
            "service1",
            "service2",
        ]

        available = factory.get_available_services()
        assert "service1" in available
        assert "service2" in available

        # Restore original services to avoid affecting other tests
        factory.mode_config.enabled_services = original_services

    def test_get_service_status(self):
        """Test getting service status information."""
        factory = ModeAwareServiceFactory(ApplicationMode.SIMPLE)

        class TestService:
            pass

        factory.register_service("test_service", TestService, TestService)

        # Create a copy and add test service to avoid mutating the global config
        original_services = factory.mode_config.enabled_services.copy()
        factory.mode_config.enabled_services = original_services + ["test_service"]

        status = factory.get_service_status("test_service")

        assert status["name"] == "test_service"
        assert status["available"] is True
        assert status["enabled"] is True
        assert status["initialized"] is False
        assert status["mode"] == "simple"

        # Restore original services to avoid affecting other tests
        factory.mode_config.enabled_services = original_services

    def test_get_mode_info(self):
        """Test getting mode information."""
        factory = ModeAwareServiceFactory(ApplicationMode.ENTERPRISE)

        mode_info = factory.get_mode_info()

        assert mode_info["mode"] == "enterprise"
        assert mode_info["enabled_services"] == ENTERPRISE_MODE_CONFIG.enabled_services
        assert mode_info["resource_limits"] == ENTERPRISE_MODE_CONFIG.resource_limits
        assert mode_info["advanced_monitoring"] is True
        assert mode_info["deployment_features"] is True


class TestComplexityReduction:
    """Test that simple mode achieves target complexity reduction."""

    def test_service_count_reduction(self):
        """Test that simple mode has significantly fewer services."""
        simple_services = len(SIMPLE_MODE_CONFIG.enabled_services)
        enterprise_services = len(ENTERPRISE_MODE_CONFIG.enabled_services)

        # Simple mode should have at least 50% fewer services
        reduction_ratio = simple_services / enterprise_services
        assert reduction_ratio <= 0.5, (
            f"Simple mode has {reduction_ratio:.2%} of enterprise services, should be ≤50%"
        )

    def test_resource_limit_reduction(self):
        """Test that simple mode has reduced resource limits."""
        simple_limits = SIMPLE_MODE_CONFIG.resource_limits
        enterprise_limits = ENTERPRISE_MODE_CONFIG.resource_limits

        # Check key resource limits are reduced
        assert (
            simple_limits["max_concurrent_requests"]
            <= enterprise_limits["max_concurrent_requests"] / 5
        )
        assert (
            simple_limits["max_memory_usage_mb"]
            <= enterprise_limits["max_memory_usage_mb"] / 5
        )
        assert simple_limits["cache_size_mb"] <= enterprise_limits["cache_size_mb"] / 10

    def test_feature_reduction(self):
        """Test that simple mode has disabled enterprise features."""
        simple_features = SIMPLE_MODE_CONFIG.max_complexity_features
        enterprise_features = ENTERPRISE_MODE_CONFIG.max_complexity_features

        # Count enabled features
        simple_enabled = sum(1 for v in simple_features.values() if v is True)
        enterprise_enabled = sum(1 for v in enterprise_features.values() if v is True)

        # Simple mode should have significantly fewer enabled features
        feature_reduction_ratio = (
            simple_enabled / enterprise_enabled if enterprise_enabled > 0 else 0
        )
        assert feature_reduction_ratio <= 0.3, (
            f"Simple mode has {feature_reduction_ratio:.2%} of enterprise features enabled"
        )

    def test_middleware_stack_reduction(self):
        """Test that simple mode has fewer middleware components."""
        simple_middleware = len(SIMPLE_MODE_CONFIG.middleware_stack)
        enterprise_middleware = len(ENTERPRISE_MODE_CONFIG.middleware_stack)

        # Simple mode should have at least 50% fewer middleware components
        middleware_reduction_ratio = simple_middleware / enterprise_middleware
        assert middleware_reduction_ratio <= 0.5, (
            f"Simple mode has {middleware_reduction_ratio:.2%} of enterprise middleware"
        )
