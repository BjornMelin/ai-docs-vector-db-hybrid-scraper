"""Application mode definitions and configurations.

This module defines the core dual-mode architecture that resolves the Enterprise Paradox
by providing distinct simple and enterprise modes with different feature sets and
complexity levels.
"""

from enum import Enum
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from src.config import Config


class ApplicationMode(Enum):
    """Application modes supporting different complexity levels."""

    SIMPLE = "simple"
    ENTERPRISE = "enterprise"


class ModeConfig(BaseModel):
    """Configuration for a specific application mode."""

    enabled_services: list[str] = Field(
        ..., description="Services enabled in this mode"
    )
    max_complexity_features: dict[str, Any] = Field(
        ..., description="Maximum complexity feature settings"
    )
    resource_limits: dict[str, int] = Field(..., description="Resource usage limits")
    middleware_stack: list[str] = Field(
        ..., description="Middleware components to load"
    )
    enable_advanced_monitoring: bool = Field(
        default=False, description="Enable comprehensive monitoring"
    )
    enable_deployment_features: bool = Field(
        default=False, description="Enable enterprise deployment features"
    )
    enable_a_b_testing: bool = Field(default=False, description="Enable A/B testing")
    enable_comprehensive_observability: bool = Field(
        default=False, description="Enable full observability stack"
    )


# Simple Mode Configuration - Optimized for solo developers
SIMPLE_MODE_CONFIG = ModeConfig(
    enabled_services=[
        "embedding_service",
        "vector_db_service",
    ],
    max_complexity_features={
        "max_concurrent_crawls": 5,
        "enable_advanced_monitoring": False,
        "enable_deployment_features": False,
        "enable_a_b_testing": False,
        "enable_comprehensive_observability": False,
        "enable_advanced_analytics": False,
        "enable_multi_tier_caching": False,
        "enable_circuit_breakers": False,
        "enable_blue_green_deployment": False,
        "enable_canary_deployment": False,
        "enable_feature_flags": False,
        "enable_query_expansion": False,
        "enable_hybrid_search": False,
        "enable_advanced_chunking": False,
        "enable_ml_quality_assessment": False,
        "enable_content_intelligence": False,
        "enable_batch_processing": False,
    },
    resource_limits={
        "max_connections": 25,
        "cache_size_mb": 50,
        "max_embeddings_batch": 25,
        "max_concurrent_requests": 5,
        "max_memory_usage_mb": 500,
        "max_vector_dimensions": 1536,
    },
    middleware_stack=["security", "timeout", "performance"],
    enable_advanced_monitoring=False,
    enable_deployment_features=False,
    enable_a_b_testing=False,
    enable_comprehensive_observability=False,
)

# Enterprise Mode Configuration - Full feature set
ENTERPRISE_MODE_CONFIG = ModeConfig(
    enabled_services=[
        "embedding_service",
        "vector_db_service",
    ],
    max_complexity_features={
        "max_concurrent_crawls": 50,
        "enable_advanced_monitoring": True,
        "enable_deployment_features": True,
        "enable_a_b_testing": True,
        "enable_comprehensive_observability": True,
        "enable_advanced_analytics": True,
        "enable_multi_tier_caching": True,
        "enable_circuit_breakers": True,
        "enable_blue_green_deployment": True,
        "enable_canary_deployment": True,
        "enable_feature_flags": True,
        "enable_query_expansion": True,
        "enable_hybrid_search": True,
        "enable_advanced_chunking": True,
        "enable_ml_quality_assessment": True,
        "enable_content_intelligence": True,
        "enable_batch_processing": True,
    },
    resource_limits={
        "max_connections": 500,
        "cache_size_mb": 1000,
        "max_embeddings_batch": 200,
        "max_concurrent_requests": 100,
        "max_memory_usage_mb": 4000,
        "max_vector_dimensions": 3072,
    },
    middleware_stack=["security", "timeout", "performance"],
    enable_advanced_monitoring=True,
    enable_deployment_features=True,
    enable_a_b_testing=True,
    enable_comprehensive_observability=True,
)


def _resolve_settings(config: "Config | None" = None) -> Any:
    """Return the active settings object, defaulting to the shared provider."""

    if config is not None:
        return config

    try:
        from src.config import get_config  # Local import to avoid circular dependency
    except ImportError:
        return SimpleNamespace(mode=ApplicationMode.SIMPLE)

    try:
        return get_config()
    except Exception:  # pragma: no cover - configuration not yet initialized
        # During module import the global configuration provider may still be wiring
        # dependencies. Default to simple mode to avoid circular imports.
        return SimpleNamespace(mode=ApplicationMode.SIMPLE)


def resolve_mode(config: "Config | None" = None) -> ApplicationMode:
    """Resolve the current application mode from configuration settings."""

    settings = _resolve_settings(config)
    return getattr(settings, "mode", ApplicationMode.SIMPLE)


def get_mode_config(
    mode: ApplicationMode | None = None, *, config: "Config | None" = None
) -> ModeConfig:
    """Get configuration for the specified mode.

    Args:
        mode: Application mode to get config for. If None, detects from environment.
        config: Optional configuration override providing the active mode.

    Returns:
        ModeConfig instance for the specified mode.
    """

    if mode is None:
        mode = resolve_mode(config)

    if mode == ApplicationMode.SIMPLE:
        return SIMPLE_MODE_CONFIG
    if mode == ApplicationMode.ENTERPRISE:
        return ENTERPRISE_MODE_CONFIG

    msg = f"Unknown application mode: {mode}"
    raise ValueError(msg)


def get_current_mode(config: "Config | None" = None) -> ApplicationMode:
    """Get the current application mode from configuration settings."""

    return resolve_mode(config)


def is_simple_mode(config: "Config | None" = None) -> bool:
    """Check if running in simple mode."""

    return get_current_mode(config) == ApplicationMode.SIMPLE


def is_enterprise_mode(config: "Config | None" = None) -> bool:
    """Check if running in enterprise mode."""

    return get_current_mode(config) == ApplicationMode.ENTERPRISE


def get_enabled_services(config: "Config | None" = None) -> list[str]:
    """Get list of services enabled in current mode."""

    mode_config = get_mode_config(config=config)
    return mode_config.enabled_services


def is_service_enabled(service_name: str, *, config: "Config | None" = None) -> bool:
    """Check if a service is enabled in the current mode."""

    return service_name in get_enabled_services(config=config)


def get_feature_setting(
    feature_name: str,
    default: Any = False,
    *,
    config: "Config | None" = None,
) -> Any:
    """Get a feature setting value for the current mode."""

    mode_config = get_mode_config(config=config)
    config_dump = mode_config.model_dump()
    features = cast(dict[str, Any], config_dump["max_complexity_features"])
    return features.get(feature_name, default)


def get_resource_limit(
    resource_name: str,
    default: int = 0,
    *,
    config: "Config | None" = None,
) -> int:
    """Get a resource limit for the current mode."""

    mode_config = get_mode_config(config=config)
    config_dump = mode_config.model_dump()
    limits = cast(dict[str, int], config_dump["resource_limits"])
    return limits.get(resource_name, default)
