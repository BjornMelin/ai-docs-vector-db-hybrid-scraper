"""FastAPI router for configuration lifecycle management."""

import logging
import os
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from src.config import get_config, get_config_reloader
from src.config.reloader import (
    ConfigError,
    ConfigLoadError,
    ReloadOperation,
    ReloadTrigger,
)
from src.services.monitoring.file_integrity import (
    FileIntegrityProvider,
    OsqueryFileIntegrityProvider,
)
from src.services.observability.tracing import (
    ConfigOperationType,
    instrument_config_operation,
)


logger = logging.getLogger(__name__)


async def require_config_access(request: Request) -> None:
    """Ensure configuration endpoints are accessed with the required credentials."""

    security_config = get_config().security
    if not security_config.api_key_required:
        return

    header_name = security_config.api_key_header
    api_key = request.headers.get(header_name)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required"
        )

    if security_config.api_keys and api_key not in security_config.api_keys:
        logger.warning("Config API key rejected", extra={"header": header_name})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )


router = APIRouter(
    prefix="/config",
    tags=["Configuration Management"],
    dependencies=[Depends(require_config_access)],
)


class ReloadRequest(BaseModel):
    """Request payload used when triggering a reload."""

    force: bool = Field(
        default=False, description="Reload even when no changes are detected"
    )
    config_source: str | None = Field(
        default=None, description="Optional configuration source override"
    )


class ReloadResponse(BaseModel):
    """Structured response returned by reload and rollback operations."""

    operation_id: str
    status: str
    success: bool
    message: str | None = None
    total_duration_ms: float
    validation_duration_ms: float
    apply_duration_ms: float
    previous_config_hash: str | None = None
    new_config_hash: str | None = None
    changes_applied: list[str] = Field(default_factory=list)
    services_notified: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)


class ReloadHistoryResponse(BaseModel):
    """Response body containing recent reload operations."""

    operations: list[ReloadResponse]
    total_count: int
    limit: int


class ReloadStatsResponse(BaseModel):
    """Aggregated statistics describing reload execution history."""

    total_operations: int
    successful_operations: int
    failed_operations: int
    success_rate: float
    average_duration_ms: float
    backups_available: int
    current_config_hash: str | None = None


RELOAD_REQUEST_BODY = Body(...)
ROLLBACK_REQUEST_BODY = Body(...)


class RollbackRequest(BaseModel):
    """Request payload for rolling back to a previous snapshot."""

    target_hash: str | None = Field(
        default=None, description="Specific configuration hash to restore"
    )


ReloadRequest.model_rebuild(_types_namespace=globals())
RollbackRequest.model_rebuild(_types_namespace=globals())


def _operation_to_response(operation: ReloadOperation) -> ReloadResponse:
    """Convert a ``ReloadOperation`` into the public response model."""

    return ReloadResponse(
        operation_id=getattr(operation, "operation_id", "unknown"),
        status=getattr(getattr(operation, "status", None), "value", "unknown"),
        success=getattr(operation, "success", False),
        message=getattr(operation, "error_message", None),
        total_duration_ms=getattr(operation, "total_duration_ms", 0.0),
        validation_duration_ms=getattr(operation, "validation_duration_ms", 0.0),
        apply_duration_ms=getattr(operation, "apply_duration_ms", 0.0),
        previous_config_hash=getattr(operation, "previous_config_hash", None),
        new_config_hash=getattr(operation, "new_config_hash", None),
        changes_applied=list(getattr(operation, "changes_applied", [])),
        services_notified=list(getattr(operation, "services_notified", [])),
        validation_errors=list(getattr(operation, "validation_errors", [])),
        validation_warnings=list(getattr(operation, "validation_warnings", [])),
    )


def _apply_config_source(*, raw_path: str, reloader: Any) -> Path:
    """Validate and register a configuration source with the reloader."""

    candidate = Path(raw_path).expanduser()
    try:
        reloader.set_default_config_source(candidate)
    except ConfigLoadError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    default_source = reloader.get_default_config_source()
    return cast(Path, default_source) if default_source is not None else candidate


def _osquery_results_log_path() -> Path:
    """Return the configured osquery results log path."""

    raw_path = os.getenv("OSQUERY_RESULTS_LOG", "/var/log/osquery/osqueryd.results.log")
    return Path(raw_path).expanduser().resolve()


async def _provider_health_snapshot(
    provider: FileIntegrityProvider | None,
) -> dict[str, Any] | None:
    """Return provider health details when available."""

    if provider is None:
        return None
    return await provider.health_check()


@router.post("/reload", response_model=ReloadResponse)
@instrument_config_operation(
    operation_type=ConfigOperationType.UPDATE,
    operation_name="api.config.reload",
)
async def reload_configuration(
    request: ReloadRequest = RELOAD_REQUEST_BODY,
) -> ReloadResponse:
    """Reload configuration from disk via the API.

    Args:
        request: Reload parameters supplied by the caller.

    Returns:
        ReloadResponse: Metrics and metadata associated with the reload run.

    Raises:
        HTTPException: Raised when the reloader surfaces an unexpected error.
    """

    reloader = get_config_reloader()
    config_source: Path | None = None

    if request.config_source:
        config_source = _apply_config_source(
            raw_path=request.config_source,
            reloader=reloader,
        )

    try:
        operation = await reloader.reload_config(
            trigger=ReloadTrigger.API,
            config_source=config_source,
            force=request.force,
        )
    except ConfigError as exc:  # pragma: no cover - defensive guard
        logger.warning("Configuration reload rejected: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001 - propagate as HTTP error
        logger.exception("Unexpected error during configuration reload")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration reload failed: {exc!s}",
        ) from exc

    if not operation.success:
        detail = operation.error_message or "Configuration reload failed"
        logger.warning("Configuration reload failed: %s", detail)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=detail)

    return _operation_to_response(operation)


@router.post("/rollback", response_model=ReloadResponse)
@instrument_config_operation(
    operation_type=ConfigOperationType.ROLLBACK,
    operation_name="api.config.rollback",
)
async def rollback_configuration(
    request: RollbackRequest = ROLLBACK_REQUEST_BODY,
) -> ReloadResponse:
    """Rollback to a previous configuration snapshot.

    Args:
        request: Rollback parameters supplied by the caller.

    Returns:
        ReloadResponse: Outcome for the rollback attempt.

    Raises:
        HTTPException: Raised if rollback fails or the reloader raises an exception.
    """

    try:
        reloader = get_config_reloader()
        operation = await reloader.rollback_config(target_hash=request.target_hash)

        response = _operation_to_response(operation)
        if not operation.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Configuration rollback failed: {operation.error_message}",
            )

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - propagate as HTTP error
        logger.exception("Unexpected error during configuration rollback")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration rollback failed: {exc!s}",
        ) from exc
    return response


@router.get("/history", response_model=ReloadHistoryResponse)
async def get_reload_history(
    limit: int = Query(
        default=20, ge=1, le=100, description="Number of operations to return"
    ),
) -> ReloadHistoryResponse:
    """Return recent reload operations recorded by the reloader.

    Args:
        limit: Maximum number of operations to return.

    Returns:
        ReloadHistoryResponse: Collection of historical operations.
    """

    try:
        reloader = get_config_reloader()
        history = reloader.get_reload_history(limit)
        operations = [_operation_to_response(op) for op in history]
        stats = reloader.get_reload_stats()
        total_count = cast(int, stats.get("total_operations", len(history)))
        return ReloadHistoryResponse(
            operations=operations, total_count=total_count, limit=limit
        )

    except Exception as exc:  # noqa: BLE001 - propagate as HTTP error
        logger.exception("Error retrieving reload history")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve reload history: {exc!s}",
        ) from exc


@router.get("/stats", response_model=ReloadStatsResponse)
async def get_reload_stats() -> ReloadStatsResponse:
    """Return aggregate statistics describing reload execution."""

    try:
        reloader = get_config_reloader()
        stats = reloader.get_reload_stats()
        current_hash_value = stats.get("current_config_hash")
        current_hash = (
            str(current_hash_value) if current_hash_value is not None else None
        )

        return ReloadStatsResponse(
            total_operations=cast(int, stats.get("total_operations", 0)),
            successful_operations=cast(int, stats.get("successful_operations", 0)),
            failed_operations=cast(int, stats.get("failed_operations", 0)),
            success_rate=cast(float, stats.get("success_rate", 0.0)),
            average_duration_ms=cast(float, stats.get("average_duration_ms", 0.0)),
            backups_available=cast(int, stats.get("backups_available", 0)),
            current_config_hash=current_hash,
        )

    except Exception as exc:  # noqa: BLE001 - propagate as HTTP error
        logger.exception("Error retrieving reload statistics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve reload statistics: {exc!s}",
        ) from exc


@router.get("/status")
async def get_config_status() -> dict[str, Any]:
    """Return the current reloader state and key health indicators."""

    try:
        reloader = get_config_reloader()
        stats = reloader.get_reload_stats()
        provider = reloader.get_file_integrity_provider()
        health = await _provider_health_snapshot(provider)
        return {
            "config_reloader_enabled": True,
            "current_config_hash": stats.get("current_config_hash"),
            "available_backups": stats.get("backups_available", 0),
            "file_watching_enabled": reloader.is_file_watch_enabled(),
            "file_watching_ready": bool(health and health.get("ready")),
            "file_watching_health": health,
            "reload_statistics": {
                "total_operations": stats.get("total_operations", 0),
                "success_rate": stats.get("success_rate", 0.0),
                "average_duration_ms": stats.get("average_duration_ms", 0.0),
            },
        }

    except Exception as exc:  # noqa: BLE001 - propagate as HTTP error
        logger.exception("Error retrieving configuration status")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve configuration status: {exc!s}",
        ) from exc


@router.post("/file-watch/enable")
async def enable_file_watching(
    poll_interval: float = Query(
        default=1.0, ge=0.1, le=60.0, description="Polling interval in seconds"
    ),
    config_source: str | None = Query(
        default=None,
        description="Optional configuration source to register before watching",
    ),
    readiness_timeout: float = Query(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="Maximum seconds to wait for the provider to report ready",
    ),
) -> dict[str, Any]:
    """Enable on-disk file watching for configuration updates."""

    reloader = get_config_reloader()

    if config_source:
        _apply_config_source(raw_path=config_source, reloader=reloader)

    default_source = reloader.get_default_config_source()
    if default_source is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=(
                "No configuration source configured; provide config_source when "
                "enabling file watching."
            ),
        )

    results_log = _osquery_results_log_path()
    if not results_log.exists():
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=(
                "Osquery results log not found at "
                f"{results_log}. Ensure osqueryd is emitting file_events."
            ),
        )

    provider: FileIntegrityProvider | None = None
    try:
        provider = OsqueryFileIntegrityProvider(
            results_log=results_log,
            poll_interval=poll_interval,
        )
        await reloader.attach_file_integrity_provider(provider)
        await reloader.enable_file_watching(poll_interval=poll_interval)
        ready = await provider.wait_until_ready(timeout=readiness_timeout)
        if not ready:
            await reloader.disable_file_watching()
            await reloader.attach_file_integrity_provider(None)
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "File integrity provider did not become ready. "
                    "Check osqueryd deployment and configuration."
                ),
            )

        health = await provider.health_check()
        return {
            "file_watching_enabled": True,
            "file_watching_ready": health.get("ready", False),
            "config_source": str(default_source),
            "results_log": str(results_log),
            "poll_interval_seconds": poll_interval,
        }

    except HTTPException:
        raise
    except ConfigError as exc:
        logger.warning("File watching configuration rejected: %s", exc)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001 - propagate as HTTP error
        logger.exception("Error enabling file watching")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enable file watching: {exc!s}",
        ) from exc


@router.post("/file-watch/disable")
async def disable_file_watching() -> dict[str, Any]:
    """Disable configuration file watching."""

    try:
        reloader = get_config_reloader()
        await reloader.disable_file_watching()

    except Exception as exc:  # noqa: BLE001 - propagate as HTTP error
        logger.exception("Error disabling file watching")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to disable file watching: {exc!s}",
        ) from exc
    return {
        "file_watching_enabled": False,
    }


@router.get("/backups")
async def list_config_backups() -> dict[str, Any]:
    """Return metadata for configuration backups exposed by the reloader."""

    try:
        reloader = get_config_reloader()

        backups: list[dict[str, Any]] = []
        for backup_hash, backup_config in reloader.get_config_backups():
            backups.append(
                {
                    "hash": backup_hash,
                    "created_at": backup_config.created_at,
                    "environment": backup_config.environment,
                }
            )

        max_backup_count = getattr(getattr(reloader, "_backups", None), "maxlen", 0)
        return {
            "available_backups": len(backups),
            "backups": backups,
            "max_backup_count": max_backup_count or len(backups),
        }

    except Exception as exc:  # noqa: BLE001 - propagate as HTTP error
        logger.exception("Error retrieving configuration backups")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve configuration backups: {exc!s}",
        ) from exc
