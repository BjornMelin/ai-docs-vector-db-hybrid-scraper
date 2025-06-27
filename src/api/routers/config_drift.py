"""API endpoints for configuration drift detection and monitoring.

Provides REST API endpoints to interact with the configuration drift detection
system, view drift events, manage alerting, and trigger manual drift checks.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ...config.core import get_config
from ...services.config_drift_service import (
    get_drift_service,
    get_drift_service_status,
    run_manual_drift_detection,
)
from ...services.observability.performance import monitor_operation


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/config-drift", tags=["Configuration Drift"])


class DriftServiceStatusResponse(BaseModel):
    """Response model for drift service status."""

    service_running: bool = Field(description="Whether the drift service is running")
    drift_detection_enabled: bool = Field(
        description="Whether drift detection is enabled"
    )
    detector_initialized: bool = Field(
        description="Whether the detector is initialized"
    )
    monitored_paths_count: int = Field(
        description="Number of monitored configuration paths"
    )
    config: dict[str, Any] = Field(description="Current drift detection configuration")
    drift_summary: dict[str, Any] | None = Field(
        default=None, description="Current drift status summary"
    )
    drift_summary_error: str | None = Field(
        default=None, description="Error getting drift summary"
    )


class DriftDetectionRequest(BaseModel):
    """Request model for manual drift detection."""

    force_snapshot: bool = Field(
        default=True, description="Force taking new snapshots before detection"
    )
    alert_on_detection: bool = Field(
        default=True, description="Send alerts for detected drift events"
    )


class DriftDetectionResponse(BaseModel):
    """Response model for drift detection results."""

    manual_detection: bool = Field(description="Whether this was a manual detection")
    timestamp: str = Field(description="Detection timestamp")
    snapshot_results: dict[str, Any] = Field(
        description="Configuration snapshot results"
    )
    comparison_results: dict[str, Any] = Field(
        description="Configuration comparison results"
    )


class DriftEventResponse(BaseModel):
    """Response model for individual drift events."""

    id: str = Field(description="Unique event identifier")
    source: str = Field(description="Configuration source")
    type: str = Field(description="Drift type")
    severity: str = Field(description="Drift severity level")
    description: str = Field(description="Event description")
    auto_remediable: bool = Field(description="Whether event can be auto-remediated")
    timestamp: str = Field(description="Event timestamp")


@router.get(
    "/status",
    response_model=DriftServiceStatusResponse,
    summary="Get configuration drift service status",
    description="Retrieve current status and configuration of the drift detection service",
)
async def get_drift_status():
    """Get current status of the configuration drift detection service."""
    try:
        with monitor_operation("api_config_drift_status", category="api"):
            status = await get_drift_service_status()
            return DriftServiceStatusResponse(**status)

    except Exception:
        logger.exception("Failed to get drift service status")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve drift service status: {e!s}",
        ) from e


@router.post(
    "/detect",
    response_model=DriftDetectionResponse,
    summary="Run manual configuration drift detection",
    description="Trigger manual configuration drift detection and return results",
)
async def run_drift_detection(
    _request: DriftDetectionRequest = DriftDetectionRequest(),
):
    """Run manual configuration drift detection."""
    try:
        with monitor_operation("api_config_drift_detect", category="api"):
            logger.info("Starting manual configuration drift detection via API")

            # Run manual detection
            results = await run_manual_drift_detection()

            logger.info(
                f"Manual drift detection completed - "
                f"snapshots: {results['snapshot_results']['snapshots_taken']}, "
                f"drift events: {len(results['comparison_results']['drift_events'])}"
            )

            return DriftDetectionResponse(**results)

    except Exception:
        logger.exception("Manual drift detection failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run drift detection: {e!s}",
        ) from e


@router.get(
    "/events",
    response_model=list[DriftEventResponse],
    summary="Get recent configuration drift events",
    description="Retrieve recent configuration drift events with filtering options",
)
async def get_drift_events(
    _limit: int = 50,
    _severity: str | None = None,
    _source: str | None = None,
    _hours: int = 24,
):
    """Get recent configuration drift events.

    Args:
        limit: Maximum number of events to return
        severity: Filter by severity level (low, medium, high, critical)
        source: Filter by configuration source
        hours: Number of hours back to search
    """
    try:
        with monitor_operation("api_config_drift_events", category="api"):
            service = get_drift_service()

            # Get drift summary which includes recent events
            summary = await service.get_service_status()

            # For now, return the drift events from the last detection
            # In a full implementation, this would query a persistent store
            if "drift_summary" not in summary:
                return []

            # This is a simplified implementation
            # A full version would maintain a persistent event store
            events = []

            return events

    except Exception:
        logger.exception("Failed to get drift events")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve drift events: {e!s}",
        ) from e


@router.get(
    "/summary",
    summary="Get configuration drift summary",
    description="Get a summary of configuration drift status and statistics",
)
async def get_drift_summary():
    """Get configuration drift summary and statistics."""
    try:
        with monitor_operation("api_config_drift_summary", category="api"):
            service = get_drift_service()
            status_info = await service.get_service_status()

            # Extract drift summary if available
            drift_summary = status_info.get("drift_summary", {})

            summary = {
                "service_status": {
                    "running": status_info["service_running"],
                    "enabled": status_info["drift_detection_enabled"],
                    "initialized": status_info["detector_initialized"],
                },
                "monitoring": {
                    "monitored_paths": status_info["monitored_paths_count"],
                    "snapshot_interval_minutes": status_info["config"][
                        "snapshot_interval_minutes"
                    ],
                    "comparison_interval_minutes": status_info["config"][
                        "comparison_interval_minutes"
                    ],
                },
                "drift_statistics": drift_summary,
                "timestamp": datetime.now(tz=UTC).isoformat(),
            }

            return summary

    except Exception:
        logger.exception("Failed to get drift summary")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve drift summary: {e!s}",
        ) from e


@router.get(
    "/health",
    summary="Check configuration drift service health",
    description="Simple health check for the configuration drift detection service",
)
async def check_drift_health():
    """Check health of the configuration drift detection service."""
    try:
        with monitor_operation("api_config_drift_health", category="api"):
            status = await get_drift_service_status()

            # Determine health based on service status
            healthy = (
                status["service_running"]
                and status["detector_initialized"]
                and status["drift_detection_enabled"]
            )

            health_status = {
                "healthy": healthy,
                "service_running": status["service_running"],
                "detector_initialized": status["detector_initialized"],
                "drift_detection_enabled": status["drift_detection_enabled"],
                "timestamp": datetime.now(tz=UTC).isoformat(),
            }

            if not healthy:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Configuration drift service is not healthy",
                )

            return health_status

    except HTTPException:
        raise
    except Exception:
        logger.exception("Drift health check failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {e!s}",
        ) from e


@router.get(
    "/config",
    summary="Get configuration drift detection settings",
    description="Retrieve current configuration for drift detection",
)
async def get_drift_config():
    """Get current configuration drift detection settings."""
    try:
        with monitor_operation("api_config_drift_config", category="api"):
            config = get_config()

            drift_config = {
                "enabled": config.drift_detection.enabled,
                "snapshot_interval_minutes": config.drift_detection.snapshot_interval_minutes,
                "comparison_interval_minutes": config.drift_detection.comparison_interval_minutes,
                "monitored_paths": config.drift_detection.monitored_paths,
                "excluded_paths": config.drift_detection.excluded_paths,
                "alert_on_severity": config.drift_detection.alert_on_severity,
                "max_alerts_per_hour": config.drift_detection.max_alerts_per_hour,
                "snapshot_retention_days": config.drift_detection.snapshot_retention_days,
                "events_retention_days": config.drift_detection.events_retention_days,
                "auto_remediation_enabled": config.drift_detection.enable_auto_remediation,
                "integrations": {
                    "task20_anomaly": config.drift_detection.integrate_with_task20_anomaly,
                    "performance_monitoring": config.drift_detection.use_performance_monitoring,
                },
            }

            return drift_config

    except Exception:
        logger.exception("Failed to get drift config")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve drift configuration: {e!s}",
        ) from e
