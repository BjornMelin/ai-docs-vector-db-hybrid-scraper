"""OpenTelemetry observability configuration integrated with unified config system.

Provides configuration for OpenTelemetry distributed tracing, AI/ML monitoring,
and cost tracking that integrates seamlessly with the existing configuration.
"""

import logging
import os
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter


# Try to import config helpers - may not be available in all contexts
try:
    from src.config import get_config, reset_config
except ImportError:
    get_config = None
    reset_config = None


logger = logging.getLogger(__name__)


def _raise_config_system_unavailable() -> None:
    """Raise ImportError for unavailable config system."""
    msg = "Config system not available"
    raise ImportError(msg)


class ObservabilityConfig(BaseModel):
    """Configuration for OpenTelemetry observability features."""

    enabled: bool = Field(default=False, description="Enable OpenTelemetry observability")
    service_name: str = Field(
        default="ai-docs-vector-db", description="Service name for traces"
    )
    service_version: str = Field(default="1.0.0", description="Service version")
    service_namespace: str = Field(default="ai-docs", description="Service namespace")
    otlp_endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP gRPC endpoint for trace export",
    )
    otlp_headers: dict[str, str] = Field(
        default_factory=dict, description="Headers for OTLP export"
    )
    otlp_insecure: bool = Field(default=True, description="Use insecure OTLP connection")
    trace_sample_rate: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Trace sampling rate"
    )
    deployment_environment: str = Field(
        default="development", description="Deployment environment"
    )
    track_ai_operations: bool = Field(
        default=True, description="Track AI operations"
    )
    track_costs: bool = Field(default=True, description="Track AI service costs")
    track_performance: bool = Field(
        default=True, description="Track performance metrics"
    )
    instrument_fastapi: bool = Field(default=True, description="Auto-instrument FastAPI")
    instrument_httpx: bool = Field(
        default=True, description="Auto-instrument HTTP clients"
    )
    instrument_redis: bool = Field(default=True, description="Auto-instrument Redis")
    instrument_sqlalchemy: bool = Field(
        default=True, description="Auto-instrument SQLAlchemy"
    )
    console_exporter: bool = Field(
        default=False, description="Export traces to console (development)"
    )
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


_CACHED_OBSERVABILITY_CONFIG: ObservabilityConfig | None = None


def clear_observability_cache() -> None:
    """Clear cached observability configuration."""

    global _CACHED_OBSERVABILITY_CONFIG
    _CACHED_OBSERVABILITY_CONFIG = None


def _slugify_app_name(value: str) -> str:
    """Convert application name to a slug suitable for service identifiers."""

    return "-".join(value.lower().split())


def get_observability_config(
    main_config: Any | None = None,
    *,
    force_refresh: bool = False,
) -> ObservabilityConfig:
    """Get observability configuration from unified config system.

    Integrates with the existing configuration to add observability settings
    while maintaining the KISS principle and existing patterns.

    Returns:
        ObservabilityConfig instance with settings

    """
    global _CACHED_OBSERVABILITY_CONFIG

    if main_config is None and not force_refresh and _CACHED_OBSERVABILITY_CONFIG:
        return _CACHED_OBSERVABILITY_CONFIG

    try:
        if main_config is None:
            if get_config is None or reset_config is None:
                _raise_config_system_unavailable()
            main_config = get_config()

        config_dict: dict[str, Any] = {}

        if hasattr(main_config, "observability") and main_config.observability:
            try:
                config_dict.update(main_config.observability.model_dump())
            except AttributeError:
                config_dict.update(main_config.observability.__dict__)

        if hasattr(main_config, "monitoring") and main_config.monitoring:
            config_dict["enabled"] = main_config.monitoring.enable_metrics

        if hasattr(main_config, "environment"):
            config_dict["deployment_environment"] = main_config.environment.value

        env_overrides: dict[str, Any] = {}
        for field_name, field_info in ObservabilityConfig.model_fields.items():
            env_key = f"AI_DOCS_OBSERVABILITY__{field_name.upper()}"
            raw_value = os.getenv(env_key)
            if raw_value is None:
                continue

            adapter = TypeAdapter(field_info.annotation or Any)
            try:
                env_overrides[field_name] = adapter.validate_python(raw_value)
            except ValueError:
                # Fallback handling for common boolean string values
                if field_info.annotation in {bool, bool | None}:
                    env_overrides[field_name] = raw_value.strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "on",
                    }
                else:
                    logger.debug(
                        "Failed to parse environment override for %s", field_name
                    )

        if "service_name" not in env_overrides and "service_name" in config_dict:
            app_name_override = os.getenv("AI_DOCS_APP_NAME")
            if app_name_override:
                env_overrides["service_name"] = _slugify_app_name(app_name_override)

        if "service_version" not in env_overrides and os.getenv("AI_DOCS_VERSION"):
            env_overrides["service_version"] = os.getenv("AI_DOCS_VERSION")

        config_dict.update(env_overrides)

        result = ObservabilityConfig(**config_dict)

        if main_config is not None and hasattr(main_config, "observability"):
            try:
                object.__setattr__(main_config, "observability", result)
            except (AttributeError, TypeError):
                main_config.observability = result

        if main_config is None:
            _CACHED_OBSERVABILITY_CONFIG = result

        return result

    except (ImportError, ValueError, TypeError, UnicodeDecodeError) as e:
        logger.warning(
            f"Could not load from main config, using defaults: {e}"
        )  # TODO: Convert f-string to logging format
        fallback = ObservabilityConfig()
        if main_config is None:
            _CACHED_OBSERVABILITY_CONFIG = fallback
        return fallback


get_observability_config.cache_clear = clear_observability_cache  # type: ignore[attr-defined]


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
