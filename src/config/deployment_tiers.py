
"""Deployment Tier Configuration System.

This module defines three configuration tiers for the application:
- Personal: Simple FastAPI + basic features for individual use
- Professional: + Flagsmith + monitoring for professional projects
- Enterprise: + Full deployment services for portfolio showcase

The tier system allows the application to be both simple for personal use
and sophisticated for enterprise demonstrations.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from .enums import DeploymentTier


class TierCapability(str, Enum):
    """Available capabilities across deployment tiers."""

    # Core features (available in all tiers)
    BASIC_SEARCH = "basic_search"
    LOCAL_CACHE = "local_cache"
    SIMPLE_DEPLOYMENT = "simple_deployment"
    HEALTH_CHECKS = "health_checks"

    # Professional tier features
    FEATURE_FLAGS = "feature_flags"
    MONITORING = "monitoring"
    METRICS_COLLECTION = "metrics_collection"

    # Enterprise tier features
    AB_TESTING = "ab_testing"
    BLUE_GREEN_DEPLOYMENT = "blue_green_deployment"
    CANARY_DEPLOYMENT = "canary_deployment"
    ADVANCED_MONITORING = "advanced_monitoring"
    TRAFFIC_ROUTING = "traffic_routing"
    DEPLOYMENT_AUTOMATION = "deployment_automation"


class TierConfiguration(BaseModel):
    """Configuration settings for a deployment tier."""

    tier: DeploymentTier = Field(..., description="Deployment tier")
    display_name: str = Field(..., description="Human-readable tier name")
    description: str = Field(..., description="Tier description")

    # Capabilities
    enabled_capabilities: list[TierCapability] = Field(
        ..., description="Available capabilities"
    )

    # Performance settings
    max_concurrent_requests: int = Field(..., description="Maximum concurrent requests")
    cache_size_mb: int = Field(..., description="Cache size in MB")
    monitoring_interval_seconds: int = Field(
        ..., description="Monitoring check interval"
    )

    # Feature flags
    feature_flags: dict[str, Any] = Field(
        default_factory=dict, description="Default feature flags"
    )

    # Service configurations
    deployment_services: dict[str, Any] = Field(
        default_factory=dict, description="Deployment service configs"
    )


# Tier Definitions
PERSONAL_TIER = TierConfiguration(
    tier=DeploymentTier.PERSONAL,
    display_name="Personal",
    description="Simple configuration for individual use and development",
    enabled_capabilities=[
        TierCapability.BASIC_SEARCH,
        TierCapability.LOCAL_CACHE,
        TierCapability.SIMPLE_DEPLOYMENT,
        TierCapability.HEALTH_CHECKS,
    ],
    max_concurrent_requests=10,
    cache_size_mb=100,
    monitoring_interval_seconds=300,  # 5 minutes
    feature_flags={
        "enable_feature_flags": False,
        "enable_monitoring": False,
        "enable_deployment_services": False,
        "log_level": "INFO",
    },
    deployment_services={
        "ab_testing": {"enabled": False},
        "blue_green": {"enabled": False},
        "canary": {"enabled": False},
    },
)

PROFESSIONAL_TIER = TierConfiguration(
    tier=DeploymentTier.PROFESSIONAL,
    display_name="Professional",
    description="Enhanced configuration with monitoring and feature flags for professional projects",
    enabled_capabilities=[
        TierCapability.BASIC_SEARCH,
        TierCapability.LOCAL_CACHE,
        TierCapability.SIMPLE_DEPLOYMENT,
        TierCapability.HEALTH_CHECKS,
        TierCapability.FEATURE_FLAGS,
        TierCapability.MONITORING,
        TierCapability.METRICS_COLLECTION,
    ],
    max_concurrent_requests=50,
    cache_size_mb=500,
    monitoring_interval_seconds=60,  # 1 minute
    feature_flags={
        "enable_feature_flags": True,
        "enable_monitoring": True,
        "enable_deployment_services": False,  # Only basic deployment
        "log_level": "INFO",
        "health_check_interval": 30,
    },
    deployment_services={
        "ab_testing": {"enabled": False},
        "blue_green": {"enabled": False},
        "canary": {"enabled": False},
        "monitoring": {
            "enabled": True,
            "metrics_retention_days": 7,
            "alert_thresholds": {
                "error_rate": 5.0,
                "response_time_ms": 1000,
            },
        },
    },
)

ENTERPRISE_TIER = TierConfiguration(
    tier=DeploymentTier.ENTERPRISE,
    display_name="Enterprise",
    description="Full enterprise deployment capabilities for portfolio showcase and production use",
    enabled_capabilities=[
        TierCapability.BASIC_SEARCH,
        TierCapability.LOCAL_CACHE,
        TierCapability.SIMPLE_DEPLOYMENT,
        TierCapability.HEALTH_CHECKS,
        TierCapability.FEATURE_FLAGS,
        TierCapability.MONITORING,
        TierCapability.METRICS_COLLECTION,
        TierCapability.AB_TESTING,
        TierCapability.BLUE_GREEN_DEPLOYMENT,
        TierCapability.CANARY_DEPLOYMENT,
        TierCapability.ADVANCED_MONITORING,
        TierCapability.TRAFFIC_ROUTING,
        TierCapability.DEPLOYMENT_AUTOMATION,
    ],
    max_concurrent_requests=1000,
    cache_size_mb=2048,  # 2GB
    monitoring_interval_seconds=10,  # 10 seconds
    feature_flags={
        "enable_feature_flags": True,
        "enable_monitoring": True,
        "enable_deployment_services": True,
        "log_level": "DEBUG",
        "health_check_interval": 15,
        # Deployment feature flags
        "ab_testing": True,
        "blue_green_deployment": True,
        "canary_deployment": True,
        "advanced_monitoring": True,
        "traffic_routing": True,
        "deployment_automation": True,
    },
    deployment_services={
        "ab_testing": {
            "enabled": True,
            "default_duration_days": 14,
            "min_sample_size": 1000,
            "confidence_level": 0.95,
            "auto_promote": True,
        },
        "blue_green": {
            "enabled": True,
            "health_check_timeout": 30,
            "health_check_retries": 3,
            "switch_delay_seconds": 5,
            "auto_switch": True,
            "auto_rollback": True,
        },
        "canary": {
            "enabled": True,
            "initial_traffic_percentage": 5.0,
            "stage_duration_minutes": 15,
            "auto_promote": True,
            "max_error_rate": 5.0,
            "rollback_on_failure": True,
        },
        "monitoring": {
            "enabled": True,
            "metrics_retention_days": 30,
            "real_time_alerts": True,
            "alert_thresholds": {
                "error_rate": 2.0,
                "response_time_ms": 500,
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
            },
            "dashboard_enabled": True,
        },
    },
)


# Tier Registry
TIER_CONFIGURATIONS: dict[DeploymentTier, TierConfiguration] = {
    DeploymentTier.PERSONAL: PERSONAL_TIER,
    DeploymentTier.PROFESSIONAL: PROFESSIONAL_TIER,
    DeploymentTier.ENTERPRISE: ENTERPRISE_TIER,
}


class TierManager:
    """Manager for deployment tier configuration and capabilities."""

    def __init__(self, current_tier: DeploymentTier = DeploymentTier.PERSONAL):
        """Initialize tier manager.

        Args:
            current_tier: Current deployment tier
        """
        self.current_tier = current_tier
        self._config_cache: dict[DeploymentTier, TierConfiguration] = {}

    def get_tier_config(self, tier: DeploymentTier | None = None) -> TierConfiguration:
        """Get configuration for a specific tier.

        Args:
            tier: Tier to get config for (defaults to current tier)

        Returns:
            TierConfiguration: Configuration for the tier
        """
        target_tier = tier or self.current_tier

        if target_tier not in self._config_cache:
            self._config_cache[target_tier] = TIER_CONFIGURATIONS[target_tier]

        return self._config_cache[target_tier]

    def is_capability_enabled(
        self, capability: TierCapability, tier: DeploymentTier | None = None
    ) -> bool:
        """Check if a capability is enabled in the current or specified tier.

        Args:
            capability: Capability to check
            tier: Tier to check (defaults to current tier)

        Returns:
            bool: True if capability is enabled
        """
        config = self.get_tier_config(tier)
        return capability in config.enabled_capabilities

    def get_feature_flag_value(
        self, flag_name: str, default: Any = False, tier: DeploymentTier | None = None
    ) -> Any:
        """Get feature flag value for the current or specified tier.

        Args:
            flag_name: Feature flag name
            default: Default value if flag not found
            tier: Tier to check (defaults to current tier)

        Returns:
            Any: Feature flag value
        """
        config = self.get_tier_config(tier)
        return config.feature_flags.get(flag_name, default)

    def get_deployment_service_config(
        self, service_name: str, tier: DeploymentTier | None = None
    ) -> dict[str, Any]:
        """Get deployment service configuration for a tier.

        Args:
            service_name: Service name (e.g., 'ab_testing', 'canary')
            tier: Tier to check (defaults to current tier)

        Returns:
            dict[str, Any]: Service configuration
        """
        config = self.get_tier_config(tier)
        return config.deployment_services.get(service_name, {"enabled": False})

    def set_tier(self, tier: DeploymentTier) -> None:
        """Change the current tier.

        Args:
            tier: New deployment tier
        """
        self.current_tier = tier

    def get_tier_comparison(self) -> dict[str, Any]:
        """Get comparison of all tiers and their capabilities.

        Returns:
            dict[str, Any]: Tier comparison data
        """
        comparison = {
            "tiers": {},
            "capability_matrix": {},
        }

        # Build tier information
        for tier in DeploymentTier:
            config = self.get_tier_config(tier)
            comparison["tiers"][tier.value] = {
                "display_name": config.display_name,
                "description": config.description,
                "max_concurrent_requests": config.max_concurrent_requests,
                "cache_size_mb": config.cache_size_mb,
                "monitoring_interval_seconds": config.monitoring_interval_seconds,
            }

        # Build capability matrix
        all_capabilities = list(TierCapability)
        for capability in all_capabilities:
            comparison["capability_matrix"][capability.value] = {}
            for tier in DeploymentTier:
                config = self.get_tier_config(tier)
                comparison["capability_matrix"][capability.value][tier.value] = (
                    capability in config.enabled_capabilities
                )

        return comparison

    def validate_tier_requirements(self, tier: DeploymentTier) -> dict[str, Any]:
        """Validate if system meets requirements for a specific tier.

        Args:
            tier: Tier to validate

        Returns:
            dict[str, Any]: Validation results
        """
        results = {
            "tier": tier.value,
            "valid": True,
            "requirements": [],
            "warnings": [],
        }

        # Check tier-specific requirements
        if tier == DeploymentTier.PROFESSIONAL:
            results["requirements"].extend(
                [
                    "Feature flag service configuration (Flagsmith recommended)",
                    "Monitoring infrastructure setup",
                    "Metrics collection endpoint",
                ]
            )

        elif tier == DeploymentTier.ENTERPRISE:
            results["requirements"].extend(
                [
                    "Feature flag service with enterprise features",
                    "Load balancer for traffic routing",
                    "Monitoring and alerting system",
                    "Database for deployment state storage",
                    "Service mesh or API gateway (optional)",
                ]
            )

        # Add general recommendations
        if tier in (DeploymentTier.PROFESSIONAL, DeploymentTier.ENTERPRISE):
            results["warnings"].extend(
                [
                    "Ensure adequate system resources for increased concurrent requests",
                    "Configure backup and disaster recovery procedures",
                    "Review security configurations for production use",
                ]
            )

        return results


# Default tier manager instance
default_tier_manager = TierManager()


def get_current_tier_config() -> TierConfiguration:
    """Get configuration for the current tier.

    Returns:
        TierConfiguration: Current tier configuration
    """
    return default_tier_manager.get_tier_config()


def is_feature_enabled(capability: TierCapability) -> bool:
    """Check if a feature is enabled in the current tier.

    Args:
        capability: Capability to check

    Returns:
        bool: True if feature is enabled
    """
    return default_tier_manager.is_capability_enabled(capability)
