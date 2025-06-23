import typing

"""OpenTelemetry observability configuration integrated with unified config system.

Provides configuration for OpenTelemetry distributed tracing, AI/ML monitoring,
and cost tracking that integrates seamlessly with the existing configuration.
"""

import logging
from functools import lru_cache
from typing import Any

from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger(__name__)


class ObservabilityConfig(BaseModel):
    """Configuration for OpenTelemetry observability features."""

    # Core observability toggle
    enabled: bool = Field(
        default=False, description="Enable OpenTelemetry observability"
    )

    # Service identification
    service_name: str = Field(
        default="ai-docs-vector-db", description="Service name for traces"
    )
    service_version: str = Field(default="1.0.0", description="Service version")
    service_namespace: str = Field(default="ai-docs", description="Service namespace")

    # OTLP Exporter configuration
    otlp_endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP gRPC endpoint for trace export",
    )
    otlp_headers: dict[str, str] = Field(
        default_factory=dict, description="Headers for OTLP export"
    )
    otlp_insecure: bool = Field(
        default=True, description="Use insecure OTLP connection"
    )

    # Sampling configuration
    trace_sample_rate: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Trace sampling rate (0.0-1.0)"
    )

    # Resource attributes
    deployment_environment: str = Field(
        default="development", description="Deployment environment"
    )

    # AI/ML specific configuration
    track_ai_operations: bool = Field(
        default=True, description="Track AI operations (embeddings, LLM calls)"
    )
    track_costs: bool = Field(default=True, description="Track AI service costs")
    track_performance: bool = Field(
        default=True, description="Track performance metrics"
    )

    # Instrumentation configuration
    instrument_fastapi: bool = Field(
        default=True, description="Auto-instrument FastAPI"
    )
    instrument_httpx: bool = Field(
        default=True, description="Auto-instrument HTTP clients"
    )
    instrument_redis: bool = Field(default=True, description="Auto-instrument Redis")
    instrument_sqlalchemy: bool = Field(
        default=True, description="Auto-instrument SQLAlchemy"
    )

    # Console debugging (development)
    console_exporter: bool = Field(
        default=False, description="Export traces to console (development)"
    )

    # Batch span processor configuration
    batch_schedule_delay: int = Field(
        default=5000, description="Batch schedule delay in milliseconds"
    )
    batch_max_queue_size: int = Field(
        default=2048, description="Maximum batch queue size"
    )
    batch_max_export_batch_size: int = Field(
        default=512, description="Maximum export batch size"
    )
    batch_export_timeout: int = Field(
        default=30000, description="Batch export timeout in milliseconds"
    )


@lru_cache
def get_observability_config() -> ObservabilityConfig:
    """Get observability configuration from unified config system.

    Integrates with the existing configuration to add observability settings
    while maintaining the KISS principle and existing patterns.

    Returns:
        ObservabilityConfig instance with settings
    """
    try:
        # Try to get from main config if available
        from src.config import get_config

        main_config = get_config()

        # Extract observability settings from environment or create defaults
        config_dict: dict[str, Any] = {}

        # Use monitoring config if available
        if hasattr(main_config, "monitoring") and main_config.monitoring:
            config_dict["enabled"] = main_config.monitoring.enable_metrics

        # Environment-based configuration
        if hasattr(main_config, "environment"):
            config_dict["deployment_environment"] = main_config.environment.value

        # Service metadata
        if hasattr(main_config, "app_name"):
            # Convert app name to service name format
            service_name = main_config.app_name.lower().replace(" ", "-")
            config_dict["service_name"] = service_name

        if hasattr(main_config, "version"):
            config_dict["service_version"] = main_config.version

        return ObservabilityConfig(**config_dict)

    except Exception as e:
        logger.warning(f"Could not load from main config, using defaults: {e}")
        return ObservabilityConfig()


def get_resource_attributes(config: ObservabilityConfig) -> dict[str, str]:
    """Get OpenTelemetry resource attributes from configuration.

    Args:
        config: Observability configuration

    Returns:
        Dictionary of resource attributes
    """
    return {
        "service.name": config.service_name,
        "service.version": config.service_version,
        "service.namespace": config.service_namespace,
        "deployment.environment": config.deployment_environment,
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.language": "python",
        "application.type": "ai-documentation-system",
        "application.features": "vector-search,embeddings,web-scraping",
    }
