"""Application mode definitions and configurations."""

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from src.config import Settings


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
        "basic_search",
        "qdrant_client",
        "simple_caching",
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
    middleware_stack=["security", "timeout"],
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
        "qdrant_client",
        "advanced_search",
        "multi_tier_caching",
        "deployment_services",
        "advanced_analytics",
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
    middleware_stack=["security", "timeout", "performance", "observability"],
    enable_advanced_monitoring=True,
    enable_deployment_features=True,
    enable_a_b_testing=True,
    enable_comprehensive_observability=True,
)


def _get_global_settings() -> "Settings":
    """Return the cached settings instance without creating import cycles."""
    # pylint: disable=import-outside-toplevel
    from src.config.loader import get_settings as _get_settings

    return _get_settings()


def resolve_mode(config: "Settings | None" = None) -> ApplicationMode:
    """Resolve the current application mode from configuration settings."""

    active_settings = config or _get_global_settings()
    return active_settings.mode


def get_mode_config(
    mode: ApplicationMode | None = None, *, config: "Settings | None" = None
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

    if mode is ApplicationMode.SIMPLE:
        return SIMPLE_MODE_CONFIG
    if mode is ApplicationMode.ENTERPRISE:
        return ENTERPRISE_MODE_CONFIG

    msg = f"Unknown application mode: {mode}"
    raise ValueError(msg)
