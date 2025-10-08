"""Configuration management tools exposing the unified Config model."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from enum import Enum
from pathlib import Path
from typing import Any, cast

from fastmcp import Context

from src.config import Config, get_config as load_unified_config
from src.infrastructure.client_manager import ClientManager


logger = logging.getLogger(__name__)

_SENSITIVE_SUFFIXES = ("api_key", "api_keys", "token", "secret", "password")


def _should_mask(key: str) -> bool:
    lowered = key.lower()
    return lowered.endswith(_SENSITIVE_SUFFIXES) or lowered in {"api_token"}


def _sanitize(value: Any) -> Any:
    """Convert config structures to serializable data with masked secrets."""

    processed = value
    if isinstance(processed, Enum):
        return processed.value
    if isinstance(processed, Path):
        return str(processed)
    if isinstance(processed, Config) or (
        hasattr(processed, "model_dump") and callable(processed.model_dump)
    ):
        processed = processed.model_dump()
    if isinstance(processed, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in processed.items():
            sanitized[key] = "<redacted>" if _should_mask(key) else _sanitize(item)
        return sanitized
    if isinstance(processed, (list, tuple, set)):
        return [_sanitize(item) for item in processed]
    return processed


def _collect_provider_warnings(config: Config) -> list[str]:
    """Identify potential provider configuration gaps."""

    warnings: list[str] = []
    embedding_provider = getattr(config, "embedding_provider", None)
    openai_settings = getattr(config, "openai", None)
    openai_api_key = getattr(openai_settings, "api_key", None)
    firecrawl_settings = getattr(config, "firecrawl", None)
    firecrawl_api_key = getattr(firecrawl_settings, "api_key", None)
    crawl_provider = getattr(config, "crawl_provider", None)

    if (
        embedding_provider is not None
        and getattr(embedding_provider, "name", "").upper() == "OPENAI"
        and not (openai_api_key or getattr(openai_settings, "api_key", None))
    ):
        warnings.append(
            "OpenAI embeddings selected but API key not configured; embeddings will "
            "be unavailable (consider switching to FastEmbed)"
        )
    if (
        firecrawl_api_key is None
        and getattr(crawl_provider, "name", "").upper() == "FIRECRAWL"
    ):
        warnings.append(
            "Firecrawl provider selected without API key; "
            "crawling features may be degraded"
        )
    return warnings


async def _grouping_support_status(
    config: Config, client_manager: ClientManager
) -> tuple[list[str], bool | None]:
    """Check QueryPointGroups readiness and emit warnings if required."""

    qdrant_settings = getattr(config, "qdrant", None)
    if not getattr(qdrant_settings, "enable_grouping", False):
        return [], None

    grouping_supported = await _probe_grouping_support(client_manager)
    if grouping_supported is False:
        return [
            "QueryPointGroups requested but backend lacks support; grouping will "
            "be disabled at runtime"
        ], grouping_supported
    if grouping_supported is None:
        return [
            "Unable to verify QueryPointGroups support; ensure the Qdrant "
            "server is >= 1.9"
        ], grouping_supported
    return [], grouping_supported


def _extract_child(current: Any, token: str) -> Any:
    """Traverse config structures case-insensitively and with list indexes."""

    candidates = {token, token.lower(), token.upper()}
    if isinstance(current, Mapping):
        for candidate in candidates:
            if candidate in current:
                return current[candidate]
    model_dump = getattr(current, "model_dump", None)
    if callable(model_dump):
        data = model_dump()
        if isinstance(data, Mapping):
            for candidate in candidates:
                if candidate in data:
                    return data[candidate]
    for candidate in candidates:
        if hasattr(current, candidate):
            return getattr(current, candidate)
    if token.isdigit():
        index = int(token)
        if isinstance(current, (list, tuple)) and 0 <= index < len(current):
            return current[index]
    return None


def _resolve_key(config: Config, key: str) -> Any:
    """Resolve dotted configuration keys (case-insensitive)."""

    parts = [part for part in key.replace("__", ".").split(".") if part]
    current: Any = config
    for part in parts:
        next_value = _extract_child(current, part)
        if next_value is None:
            return None
        current = next_value
    return current


async def _probe_grouping_support(client_manager: ClientManager) -> bool | None:
    """Detect whether the active vector service supports QueryPointGroups."""

    try:
        vector_service = await client_manager.get_vector_store_service()
        server_side_probe = getattr(
            vector_service, "supports_server_side_grouping", None
        )
        if callable(server_side_probe):
            async_probe = cast(Callable[[], Awaitable[bool | None]], server_side_probe)
            return await async_probe()

        raw_probe = getattr(vector_service, "supports_query_groups", None)
        if raw_probe is not None:
            if asyncio.iscoroutinefunction(raw_probe):
                async_query = cast(Callable[[], Awaitable[bool | None]], raw_probe)
                return await async_query()
            if callable(raw_probe):
                sync_query = cast(Callable[[], bool], raw_probe)
                return bool(sync_query())
            return bool(raw_probe)
    except Exception as exc:  # pragma: no cover - defensive guardrail
        logger.warning("Failed to probe grouping support: %s", exc)
        return None
    return None


def register_tools(mcp, client_manager: ClientManager):
    """Register configuration management tools."""

    config_loader = load_unified_config

    @mcp.tool()
    async def get_config(
        key: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Return configuration values from the unified settings model."""

        try:
            config: Config = cast(Config, config_loader())
            if key:
                raw_value = _resolve_key(config, key)
                if raw_value is None:
                    message = f"Config key '{key}' not found"
                    if ctx:
                        await ctx.warning(message)
                    return {"key": key, "found": False, "value": None}
                sanitized_value = _sanitize(raw_value)
                if ctx:
                    await ctx.info(f"Retrieved config value for '{key}'")
                final_token = key.split(".")[-1]
                if _should_mask(final_token):
                    sanitized_value = "<redacted>"
                return {"key": key, "found": True, "value": sanitized_value}

            sanitized = _sanitize(config)
            if ctx:
                await ctx.info("Retrieved complete configuration snapshot")
            return {"config": sanitized}
        except Exception as exc:  # pragma: no cover - unexpected failure
            logger.exception("Failed to read configuration")
            if ctx:
                await ctx.error(f"Config error: {exc}")
            return {"error": str(exc)}

    @mcp.tool()
    async def validate_config(
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Validate configuration health and highlight potential issues."""

        try:
            config: Config = cast(Config, config_loader())
            issues: list[str] = []
            warnings = _collect_provider_warnings(config)
            grouping_warnings, grouping_supported = await _grouping_support_status(
                config, client_manager
            )
            warnings.extend(grouping_warnings)

            qdrant_settings = getattr(config, "qdrant", None)
            if not getattr(qdrant_settings, "url", ""):
                issues.append("Qdrant URL is not configured")

            if ctx:
                if issues:
                    await ctx.error(
                        f"Configuration validation failed with {len(issues)} issues"
                    )
                else:
                    await ctx.info("Configuration validation completed")
                for warning in warnings:
                    await ctx.warning(warning)

            sanitized = _sanitize(config)
            populated_source: Mapping[str, Any] | None = None
            if isinstance(sanitized, Mapping):
                populated_source = sanitized
            populated_fields = sum(
                1
                for value in (populated_source or {}).values()
                if value not in (None, "")
            )
            return {
                "valid": not issues,
                "issues": issues,
                "warnings": warnings,
                "populated_fields": populated_fields,
                "grouping_supported": grouping_supported,
                "config": sanitized,
            }
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("Config validation failed")
            if ctx:
                await ctx.error(f"Validation error: {exc}")
            return {"valid": False, "error": str(exc)}
