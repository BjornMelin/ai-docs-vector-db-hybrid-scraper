"""FastAPI router exposing read-only application settings endpoints."""

import logging
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,  # type: ignore
    status,  # type: ignore
)
from pydantic import BaseModel, Field

from src.config.loader import get_settings, refresh_settings


logger = logging.getLogger(__name__)


async def require_config_access(request: Request) -> None:
    """Ensure configuration endpoints enforce API key requirements."""
    security_config = get_settings().security
    if not security_config.api_key_required:
        return

    header_name = security_config.api_key_header
    api_key = request.headers.get(header_name)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required"
        )
    if security_config.api_keys and api_key not in security_config.api_keys:
        logger.warning("Configuration API key rejected", extra={"header": header_name})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )


router = APIRouter(
    prefix="/config",
    tags=["Configuration"],
    dependencies=[Depends(require_config_access)],
)


class SettingsSnapshot(BaseModel):
    """Serializable snapshot of core settings attributes."""

    app_name: str
    version: str
    mode: str
    environment: str
    debug: bool
    observability_enabled: bool = Field(
        description="Indicates whether observability features are active."
    )
    feature_flags: dict[str, bool] = Field(
        default_factory=dict,
        description="Active feature flags for the unified application.",
    )


class RefreshRequest(BaseModel):
    """Request payload accepted when refreshing settings."""

    overrides: dict[str, Any] | None = Field(
        default=None,
        description="Optional override values merged into the refreshed settings.",
    )


class RefreshResponse(BaseModel):
    """Response returned after refreshing application settings."""

    snapshot: SettingsSnapshot


@router.get("/", response_model=SettingsSnapshot)
def read_settings() -> SettingsSnapshot:
    """Return a sanitized snapshot of the current settings."""
    try:
        settings = get_settings()
        return SettingsSnapshot(
            app_name=settings.app_name,
            version=settings.version,
            mode=settings.mode,
            environment=settings.environment.value,
            debug=settings.debug,
            observability_enabled=settings.observability.enabled,
            feature_flags=settings.get_feature_flags(),
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to serialize settings snapshot")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Settings snapshot unavailable",
        ) from exc


@router.get("/status", response_model=SettingsSnapshot)
def read_status() -> SettingsSnapshot:
    """Return the same snapshot with emphasis on current mode."""
    try:
        settings = get_settings()
        return SettingsSnapshot(
            app_name=settings.app_name,
            version=settings.version,
            mode=settings.mode,
            environment=settings.environment.value,
            debug=settings.debug,
            observability_enabled=settings.observability.enabled,
            feature_flags=settings.get_feature_flags(),
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to serialize status snapshot")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Settings snapshot unavailable",
        ) from exc


@router.post("/refresh", response_model=RefreshResponse, status_code=status.HTTP_200_OK)
def refresh_settings_endpoint(request: RefreshRequest) -> RefreshResponse:
    """Refresh the cached settings instance, applying optional overrides."""
    overrides = request.overrides or {}
    try:
        settings = refresh_settings(**overrides)
        return RefreshResponse(
            snapshot=SettingsSnapshot(
                app_name=settings.app_name,
                version=settings.version,
                mode=settings.mode,
                environment=settings.environment.value,
                debug=settings.debug,
                observability_enabled=settings.observability.enabled,
                feature_flags=settings.get_feature_flags(),
            )
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to refresh settings with overrides")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to refresh settings",
        ) from exc
