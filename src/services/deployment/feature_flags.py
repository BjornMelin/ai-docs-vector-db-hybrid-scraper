"""Feature Flag Management for Deployment Tiers.

This module implements feature flag-driven configuration tiers using Flagsmith
for controlling access to enterprise deployment features while maintaining
simplicity for personal use.
"""

import logging
import os
import time
from typing import Any

from pydantic import BaseModel, Field

from src.config.models import DeploymentTier


# Optional dependency handling
try:
    from flagsmith import Flagsmith
except ImportError:
    Flagsmith = None


logger = logging.getLogger(__name__)


class FeatureFlagConfig(BaseModel):
    """Configuration for feature flag integration."""

    model_config = {"arbitrary_types_allowed": True}

    enabled: bool = Field(default=False, description="Enable feature flag integration")
    provider: str = Field(default="flagsmith", description="Feature flag provider")
    api_key: str | None = Field(
        default=None, description="API key for feature flag service"
    )
    environment_key: str | None = Field(
        default=None, description="Environment key for feature flags"
    )
    api_url: str = Field(
        default="https://edge.api.flagsmith.com/api/v1/",
        description="Feature flag API URL",
    )
    cache_ttl: int = Field(default=300, description="Cache TTL for flags in seconds")
    fallback_tier: DeploymentTier = Field(
        default=DeploymentTier.PERSONAL,
        description="Fallback tier when flags unavailable",
    )


class FeatureFlagManager:
    """Feature flag manager with tier-based deployment configuration."""

    def __init__(self, config: FeatureFlagConfig):
        """Initialize feature flag manager.

        Args:
            config: Feature flag configuration

        """
        self.config = config
        self._client = None
        self._cache: dict[str, Any] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._current_tier = config.fallback_tier
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize feature flag client if enabled."""
        if self._initialized:
            return

        try:
            await self._initialize_feature_flag_client()
            self._current_tier = await self._determine_tier()
            self._initialized = True
        except Exception:
            logger.exception("Failed to initialize feature flag manager: %s")
            self._current_tier = self.config.fallback_tier
            self._initialized = True

    async def _initialize_feature_flag_client(self) -> None:
        """Initialize the feature flag client if enabled."""
        if not (self.config.enabled and self.config.api_key):
            logger.info(
                "Feature flags disabled, using fallback tier: %s",
                self.config.fallback_tier,
            )
            return

        if Flagsmith is None:
            logger.warning("Flagsmith not available - feature flags disabled")
            return

        try:
            self._create_flagsmith_client()
        except ImportError:
            self._handle_flagsmith_import_error()

    def _create_flagsmith_client(self) -> None:
        """Create and configure the Flagsmith client."""
        assert Flagsmith is not None
        self._client = Flagsmith(
            environment_key=self.config.environment_key,
            api_url=self.config.api_url,
        )
        logger.info("Flagsmith client initialized successfully")

    def _handle_flagsmith_import_error(self) -> None:
        """Handle Flagsmith import errors gracefully."""
        logger.warning(
            "Flagsmith not available. Install with: pip install flagsmith"
            " Using fallback tier: %s",
            self.config.fallback_tier,
        )
        self._client = None

    async def get_deployment_tier(self) -> DeploymentTier:
        """Get current deployment tier based on feature flags.

        Returns:
            DeploymentTier: Current deployment configuration tier
        """
        if not self._initialized:
            await self.initialize()

        return self._current_tier

    async def is_feature_enabled(
        self, feature_name: str, user_id: str | None = None
    ) -> bool:
        """Check if a specific feature is enabled.

        Args:
            feature_name: Name of the feature flag
            user_id: Optional user ID for personalized flags

        Returns:
            bool: True if feature is enabled
        """
        if not self._initialized:
            await self.initialize()

        try:
            return await self._check_feature_flag(feature_name, user_id)
        except Exception:
            logger.exception("Error checking feature flag %s", feature_name)
            return self._get_feature_by_tier(feature_name)

    async def _check_feature_flag(self, feature_name: str, user_id: str | None) -> bool:
        """Check feature flag using client or tier fallback."""
        if self._client:
            # Use Flagsmith client
            flags = await self._get_flags_from_client(user_id)
            return flags.get(feature_name, False)
        # Use tier-based fallback
        return self._get_feature_by_tier(feature_name)

    async def get_config_value(
        self, config_key: str, default: Any = None, user_id: str | None = None
    ) -> Any:
        """Get configuration value from feature flags.

        Args:
            config_key: Configuration key to retrieve
            default: Default value if not found
            user_id: Optional user ID for personalized config

        Returns:
            Any: Configuration value or default
        """
        if not self._initialized:
            await self.initialize()

        try:
            return await self._get_config_from_flags(config_key, default, user_id)
        except Exception:
            logger.exception("Error getting config value %s", config_key)
            return self._get_config_by_tier(config_key, default)

    async def _get_config_from_flags(
        self, config_key: str, default: Any, user_id: str | None
    ) -> Any:
        """Get configuration value from flags or tier fallback."""
        if self._client:
            # Use Flagsmith client for remote config
            flags = await self._get_flags_from_client(user_id)
            return flags.get(config_key, default)
        # Use tier-based fallback
        return self._get_config_by_tier(config_key, default)

    def _get_feature_by_tier(self, feature_name: str) -> bool:
        """Get feature availability based on current tier.

        Args:
            feature_name: Feature name to check

        Returns:
            bool: True if feature is available in current tier
        """
        tier_features = {
            DeploymentTier.PERSONAL: {
                "basic_search",
                "simple_deployment",
                "local_cache",
            },
            DeploymentTier.PROFESSIONAL: {
                "basic_search",
                "simple_deployment",
                "local_cache",
                "monitoring",
                "feature_flags",
                "health_checks",
            },
            DeploymentTier.ENTERPRISE: {
                "basic_search",
                "simple_deployment",
                "local_cache",
                "monitoring",
                "feature_flags",
                "health_checks",
                "ab_testing",
                "blue_green_deployment",
                "canary_deployment",
                "advanced_monitoring",
                "traffic_routing",
                "deployment_automation",
            },
        }

        return feature_name in tier_features.get(self._current_tier, set())

    def _get_config_by_tier(self, config_key: str, default: Any) -> Any:
        """Get configuration value based on current tier.

        Args:
            config_key: Configuration key
            default: Default value

        Returns:
            Any: Configuration value for current tier
        """
        tier_configs = {
            DeploymentTier.PERSONAL: {
                "max_concurrent_requests": 10,
                "cache_size": "100MB",
                "monitoring_interval": 300,
                "log_level": "INFO",
            },
            DeploymentTier.PROFESSIONAL: {
                "max_concurrent_requests": 50,
                "cache_size": "500MB",
                "monitoring_interval": 60,
                "log_level": "INFO",
                "health_check_interval": 30,
            },
            DeploymentTier.ENTERPRISE: {
                "max_concurrent_requests": 1000,
                "cache_size": "2GB",
                "monitoring_interval": 10,
                "log_level": "DEBUG",
                "health_check_interval": 15,
                "canary_traffic_percentage": 5,
                "blue_green_enabled": True,
                "ab_test_duration_days": 14,
            },
        }

        tier_config = tier_configs.get(self._current_tier, {})
        return tier_config.get(config_key, default)

    async def _determine_tier(self) -> DeploymentTier:
        """Determine deployment tier from feature flags or environment.

        Returns:
            DeploymentTier: Determined tier
        """
        try:
            return await self._determine_tier_from_flags_or_env()
        except Exception:
            logger.exception("Error determining deployment tier: %s")
            return self.config.fallback_tier

    async def _determine_tier_from_flags_or_env(self) -> DeploymentTier:
        """Determine tier from feature flags or environment."""
        if self._client:
            return await self._determine_tier_from_flags()

        # Fallback to environment-based detection
        tier_env = os.getenv("DEPLOYMENT_TIER", "personal").lower()
        return DeploymentTier(tier_env)

    async def _determine_tier_from_flags(self) -> DeploymentTier:
        """Determine tier from feature flags."""
        flags = await self._get_flags_from_client()

        if flags.get("enterprise_features_enabled", False):
            return DeploymentTier.ENTERPRISE
        if flags.get("professional_features_enabled", False):
            return DeploymentTier.PROFESSIONAL
        return DeploymentTier.PERSONAL

    async def _get_flags_from_client(
        self, user_id: str | None = None
    ) -> dict[str, Any]:
        """Get flags from Flagsmith client with caching.

        Args:
            user_id: Optional user ID for personalized flags

        Returns:
            dict[str, Any]: Feature flags and config values
        """
        if not self._client:
            return {}

        try:
            return await self._fetch_and_cache_flags(user_id)
        except Exception:
            logger.exception("Error fetching flags from client: %s")
            return {}

    async def _fetch_and_cache_flags(self, user_id: str | None) -> dict[str, Any]:
        """Fetch flags from client and handle caching."""
        cache_key = f"flags_{user_id or 'anonymous'}"
        current_time = time.time()

        # Check cache first
        if self._is_cache_valid(cache_key, current_time):
            return self._cache[cache_key]

        # Fetch from Flagsmith
        flags_response = self._fetch_flags_from_flagsmith(user_id)
        flags = self._convert_flags_to_dict(flags_response)

        # Cache results
        self._cache[cache_key] = flags
        self._cache_timestamps[cache_key] = current_time
        return flags

    def _is_cache_valid(self, cache_key: str, current_time: float) -> bool:
        """Check if cached flags are still valid."""
        return (
            cache_key in self._cache
            and current_time - self._cache_timestamps.get(cache_key, 0)
            < self.config.cache_ttl
        )

    def _fetch_flags_from_flagsmith(self, user_id: str | None) -> Any:
        """Fetch flags from Flagsmith service."""
        assert self._client is not None
        if user_id:
            identity = {"identifier": user_id}
            return self._client.get_identity_flags(identity["identifier"])
        return self._client.get_environment_flags()

    def _convert_flags_to_dict(self, flags_response: Any) -> dict[str, Any]:
        """Convert Flagsmith response to simple dict format."""
        flags = {}
        for flag in flags_response:
            flags[flag.feature.name] = flag.enabled
            if flag.feature_state_value:
                try:
                    flags[flag.feature.name] = self._parse_flag_value(
                        flag.feature_state_value
                    )
                except (AttributeError, ValueError):
                    flags[flag.feature.name] = flag.feature_state_value
        return flags

    def _parse_flag_value(self, value: str) -> Any:
        """Parse flag value to appropriate type."""
        # Try to parse as number or boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        if value.replace(".", "").isdigit():
            return float(value)
        return value

    async def cleanup(self) -> None:
        """Cleanup feature flag manager resources."""
        if self._client:
            # Flagsmith client doesn't require explicit cleanup
            pass
        self._cache.clear()
        self._cache_timestamps.clear()
        self._initialized = False
