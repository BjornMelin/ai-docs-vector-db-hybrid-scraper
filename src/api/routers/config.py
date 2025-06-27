"""Configuration management API endpoints.

This module provides FastAPI endpoints for managing configuration reloading,
monitoring reload operations, and accessing configuration status information.
"""

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from ...config.reload import (
    ReloadOperation,
    ReloadTrigger,
    get_config_reloader,
)
from ...services.observability.config_instrumentation import (
    ConfigOperationType,
    instrument_config_operation,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["Configuration Management"])


class ReloadRequest(BaseModel):
    """Configuration reload request."""

    force: bool = Field(
        default=False, description="Force reload even if no changes detected"
    )
    config_source: str | None = Field(
        default=None, description="Optional specific config source path"
    )


class ReloadResponse(BaseModel):
    """Configuration reload response."""

    operation_id: str
    status: str
    success: bool
    message: str | None = None

    # Timing information
    total_duration_ms: float
    validation_duration_ms: float
    apply_duration_ms: float

    # Change information
    previous_config_hash: str | None = None
    new_config_hash: str | None = None
    changes_applied: list[str] = Field(default_factory=list)
    services_notified: list[str] = Field(default_factory=list)

    # Validation results
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)


class ReloadHistoryResponse(BaseModel):
    """Configuration reload history response."""

    operations: list[ReloadResponse]
    total_count: int
    limit: int


class ReloadStatsResponse(BaseModel):
    """Configuration reload statistics response."""

    total_operations: int
    successful_operations: int
    failed_operations: int
    success_rate: float
    average_duration_ms: float
    listeners_registered: int
    backups_available: int
    current_config_hash: str | None = None


class RollbackRequest(BaseModel):
    """Configuration rollback request."""

    target_hash: str | None = Field(
        default=None, description="Specific config hash to rollback to"
    )


def _operation_to_response(operation: ReloadOperation) -> ReloadResponse:
    """Convert ReloadOperation to API response."""
    return ReloadResponse(
        operation_id=operation.operation_id,
        status=operation.status.value,
        success=operation.success,
        message=operation.error_message,
        total_duration_ms=operation.total_duration_ms,
        validation_duration_ms=operation.validation_duration_ms,
        apply_duration_ms=operation.apply_duration_ms,
        previous_config_hash=operation.previous_config_hash,
        new_config_hash=operation.new_config_hash,
        changes_applied=operation.changes_applied,
        services_notified=operation.services_notified,
        validation_errors=operation.validation_errors,
        validation_warnings=operation.validation_warnings,
    )


@router.post("/reload", response_model=ReloadResponse)
@instrument_config_operation(
    operation_type=ConfigOperationType.UPDATE,
    operation_name="api.config.reload",
)
async def reload_configuration(request: ReloadRequest) -> ReloadResponse:
    """Trigger a configuration reload operation.

    This endpoint provides a safe way to reload configuration with proper
    validation and rollback capabilities. The operation is performed
    asynchronously with comprehensive error handling.

    Args:
        request: Reload request parameters

    Returns:
        Reload operation results and metrics

    Raises:
        HTTPException: If reload operation fails
    """
    try:
        reloader = get_config_reloader()

        # Perform reload operation
        operation = await reloader.reload_config(
            trigger=ReloadTrigger.API,
            config_source=Path(request.config_source)
            if request.config_source
            else None,
            force=request.force,
        )

        response = _operation_to_response(operation)

        if not operation.success:
            # Return detailed error information but don't raise exception
            # This allows clients to get full operation details
            logger.warning(f"Configuration reload failed: {operation.error_message}")

    except Exception as e:
        logger.exception("Unexpected error during configuration reload")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration reload failed: {e!s}",
        )
    else:
        return response


def _raise_rollback_error(operation: object) -> None:
    """Raise HTTPException for failed rollback operation."""
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Configuration rollback failed: {operation.error_message}",
    )


@router.post("/rollback", response_model=ReloadResponse)
@instrument_config_operation(
    operation_type=ConfigOperationType.ROLLBACK,
    operation_name="api.config.rollback",
)
async def rollback_configuration(request: RollbackRequest) -> ReloadResponse:
    """Rollback to a previous configuration.

    This endpoint allows rolling back to a previous configuration state
    in case of issues with the current configuration. If no target hash
    is specified, rolls back to the most recent backup.

    Args:
        request: Rollback request parameters

    Returns:
        Rollback operation results and metrics

    Raises:
        HTTPException: If rollback operation fails
    """
    try:
        reloader = get_config_reloader()

        # Perform rollback operation
        operation = await reloader.rollback_config(target_hash=request.target_hash)

        response = _operation_to_response(operation)

        if not operation.success:
            _raise_rollback_error(operation)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during configuration rollback")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration rollback failed: {e!s}",
        )
    else:
        return response


@router.get("/history", response_model=ReloadHistoryResponse)
async def get_reload_history(
    limit: int = Query(
        default=20, ge=1, le=100, description="Number of operations to return"
    ),
) -> ReloadHistoryResponse:
    """Get configuration reload operation history.

    Returns recent configuration reload operations with their results,
    timing information, and change details.

    Args:
        limit: Maximum number of operations to return

    Returns:
        Historical reload operations
    """
    try:
        reloader = get_config_reloader()
        history = reloader.get_reload_history(limit=limit)

        operations = [_operation_to_response(op) for op in history]

        return ReloadHistoryResponse(
            operations=operations,
            total_count=len(history),
            limit=limit,
        )

    except Exception as e:
        logger.exception("Error retrieving reload history")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve reload history: {e!s}",
        )


@router.get("/stats", response_model=ReloadStatsResponse)
async def get_reload_stats() -> ReloadStatsResponse:
    """Get configuration reload statistics.

    Returns comprehensive statistics about configuration reload operations
    including success rates, timing metrics, and system status.

    Returns:
        Configuration reload statistics
    """
    try:
        reloader = get_config_reloader()
        stats = reloader.get_reload_stats()

        return ReloadStatsResponse(**stats)

    except Exception as e:
        logger.exception("Error retrieving reload statistics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve reload statistics: {e!s}",
        )


@router.get("/status")
async def get_config_status() -> dict[str, Any]:
    """Get current configuration status and health information.

    Returns:
        Current configuration status including hash, listeners, and settings
    """
    try:
        reloader = get_config_reloader()
        stats = reloader.get_reload_stats()

        # Additional status information
        status_info = {
            "config_reloader_enabled": True,
            "current_config_hash": stats.get("current_config_hash"),
            "registered_listeners": stats.get("listeners_registered", 0),
            "available_backups": stats.get("backups_available", 0),
            "file_watching_enabled": hasattr(reloader, "_file_watch_enabled")
            and reloader._file_watch_enabled,
            "signal_handler_enabled": reloader.enable_signal_handler,
            "reload_statistics": {
                "total_operations": stats.get("total_operations", 0),
                "success_rate": stats.get("success_rate", 0.0),
                "average_duration_ms": stats.get("average_duration_ms", 0.0),
            },
        }

    except Exception as e:
        logger.exception("Error retrieving configuration status")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve configuration status: {e!s}",
        )
    else:
        return status_info


@router.post("/file-watch/enable")
async def enable_file_watching(
    poll_interval: float = Query(
        default=1.0, ge=0.1, le=60.0, description="Polling interval in seconds"
    ),
) -> dict[str, Any]:
    """Enable automatic configuration file watching.

    Enables monitoring of the configuration file for changes and automatic
    reload when changes are detected.

    Args:
        poll_interval: File polling interval in seconds

    Returns:
        File watching status
    """
    try:
        reloader = get_config_reloader()
        await reloader.enable_file_watching(poll_interval=poll_interval)

        return {
            "file_watching_enabled": True,
            "poll_interval_seconds": poll_interval,
            "config_source": str(reloader.config_source),
            "message": "Configuration file watching enabled",
        }

    except Exception as e:
        logger.exception("Error enabling file watching")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enable file watching: {e!s}",
        )


@router.post("/file-watch/disable")
async def disable_file_watching() -> dict[str, Any]:
    """Disable automatic configuration file watching.

    Returns:
        File watching status
    """
    try:
        reloader = get_config_reloader()
        await reloader.disable_file_watching()

    except Exception as e:
        logger.exception("Error disabling file watching")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to disable file watching: {e!s}",
        )
    else:
        return {
            "file_watching_enabled": False,
            "message": "Configuration file watching disabled",
        }


@router.get("/backups")
async def list_config_backups() -> dict[str, Any]:
    """List available configuration backups.

    Returns information about available configuration backups that can
    be used for rollback operations.

    Returns:
        Available configuration backups
    """
    try:
        reloader = get_config_reloader()

        # Access backup information (note: this requires exposing backup list)
        if hasattr(reloader, "_config_backups"):
            backups = [
                {
                    "hash": backup_hash,
                    "created_at": backup_config.model_dump().get(
                        "created_at", "unknown"
                    ),
                    "environment": str(backup_config.environment)
                    if hasattr(backup_config, "environment")
                    else "unknown",
                }
                for backup_hash, backup_config in reloader._config_backups
            ]
        else:
            backups = []

        return {
            "available_backups": len(backups),
            "backups": backups,
            "max_backup_count": reloader.backup_count,
        }

    except Exception as e:
        logger.exception("Error retrieving configuration backups")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve configuration backups: {e!s}",
        )
