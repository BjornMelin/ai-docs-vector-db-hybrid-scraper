"""Embedding management tools for MCP server."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from fastmcp import Context

from src.mcp_tools.models.requests import EmbeddingRequest
from src.mcp_tools.models.responses import (
    EmbeddingGenerationResponse,
    EmbeddingProviderInfo,
)
from src.mcp_tools.utils.provider_metadata import normalize_provider_catalog


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ProviderCacheState:
    """Small helper to cache provider metadata without global statements."""

    data: list[EmbeddingProviderInfo] | None = None
    expiry: datetime | None = None


_PROVIDER_CACHE_TTL = timedelta(seconds=300)
_PROVIDER_CACHE = _ProviderCacheState()


def register_tools(
    mcp,
    *,
    embedding_manager: Any,
) -> None:
    """Register embedding management tools with the MCP server."""

    @mcp.tool()
    async def generate_embeddings(
        request: EmbeddingRequest, ctx: Context | None = None
    ) -> EmbeddingGenerationResponse:
        """Generate embeddings using the optimal provider.

        Auto-selects the best embedding model based on cost,
        performance, and availability.
        """

        if ctx:
            await ctx.info(
                f"Generating embeddings for {len(request.texts)} texts using model: "
                f"{request.model or 'default'}"
            )

        try:
            manager = embedding_manager

            if ctx:
                await ctx.debug(
                    f"Using batch size: {request.batch_size}, "
                    f"sparse embeddings: {request.generate_sparse}"
                )

            # Generate embeddings
            result = await manager.generate_embeddings(
                texts=request.texts,
                provider_name=request.model,
                generate_sparse=request.generate_sparse,
            )

            provider_name = str(result.get("provider", "unknown"))

            cost_estimate = _safe_estimate_cost(
                manager,
                request.texts,
                provider_name,
            )

            if ctx:
                await ctx.info(
                    f"Successfully generated embeddings for "
                    f"{len(request.texts)} texts using provider: "
                    f"{provider_name}"
                )

            total_tokens = _extract_total_tokens(result)
            if total_tokens is None and not request.texts:
                total_tokens = 0

            embeddings = _normalize_dense_embeddings(result.get("embeddings"))
            sparse_embeddings = _normalize_sparse_embeddings(
                result.get("sparse_embeddings")
            )
            model_name = result.get("model")

            extra_payload = {
                key: value
                for key, value in result.items()
                if key not in {"embeddings", "sparse_embeddings", "model", "provider"}
            }

            return EmbeddingGenerationResponse(
                embeddings=embeddings,
                sparse_embeddings=sparse_embeddings,
                model=model_name if isinstance(model_name, str) else None,
                provider=provider_name,
                cost_estimate=cost_estimate,
                total_tokens=total_tokens,
                **extra_payload,
            )

        except Exception as e:
            if ctx:
                await ctx.error(f"Embedding generation failed: {e}")
            logger.exception("Embedding generation failed")
            raise

    @mcp.tool()
    async def list_embedding_providers(
        ctx: Context | None = None,
    ) -> list[EmbeddingProviderInfo]:
        """List available embedding providers and their capabilities.

        Returns information about supported models, costs, and current status.
        """

        if ctx:
            await ctx.info("Retrieving available embedding providers")

        try:
            # Get embedding manager from DI container or override
            manager = embedding_manager

            providers = _get_normalized_providers(manager)

            if ctx:
                await ctx.info(f"Found {len(providers)} available embedding providers")

            return providers

        except Exception as e:
            if ctx:
                await ctx.error(f"Failed to list embedding providers: {e}")
            logger.exception("Failed to list embedding providers")
            raise


def _extract_total_tokens(result: Any) -> int | None:
    """Safely extract total token count from embedding result."""

    if isinstance(result, dict):
        tokens = result.get("tokens") or result.get("total_tokens")
        if tokens is None:
            return None
        try:
            return int(tokens)
        except (TypeError, ValueError):
            return None

    if hasattr(result, "total_tokens") and result.total_tokens is not None:
        return int(result.total_tokens)
    if hasattr(result, "_total_tokens") and result._total_tokens is not None:
        return int(result._total_tokens)
    return None


def _safe_estimate_cost(
    embedding_manager: Any,
    texts: Iterable[str],
    provider_name: str,
) -> float | None:
    """Estimate embedding cost using the manager when supported."""

    try:
        cost_map = embedding_manager.estimate_cost(list(texts), provider_name)
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("Embedding cost estimation unavailable", exc_info=True)
        return None

    if not isinstance(cost_map, dict):
        return None

    provider_cost = cost_map.get(provider_name)
    if provider_cost and "total_cost" in provider_cost:
        return provider_cost["total_cost"]

    # Fall back to the first available entry if no direct match
    for entry in cost_map.values():
        if isinstance(entry, dict) and "total_cost" in entry:
            return entry["total_cost"]
    return None


def _get_normalized_providers(
    embedding_manager: Any,
) -> list[EmbeddingProviderInfo]:
    """Return provider metadata with caching and fallbacks."""

    now = datetime.now(tz=UTC)
    if (
        _PROVIDER_CACHE.data is not None
        and _PROVIDER_CACHE.expiry is not None
        and now < _PROVIDER_CACHE.expiry
    ):
        return _PROVIDER_CACHE.data

    raw_catalog: dict[str, dict[str, Any]] = {}
    try:
        raw_catalog = embedding_manager.get_provider_info() or {}
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("Failed to fetch provider info from manager", exc_info=True)

    normalized = normalize_provider_catalog(raw_catalog)
    _PROVIDER_CACHE.data = normalized
    _PROVIDER_CACHE.expiry = now + _PROVIDER_CACHE_TTL
    return normalized


def _normalize_dense_embeddings(data: Any) -> list[list[float]]:
    """Coerce embedding payloads to a float matrix, dropping invalid rows."""

    if not isinstance(data, list):
        return []

    matrix: list[list[float]] = []
    for row in data:
        if not isinstance(row, Iterable) or isinstance(row, str | bytes):
            continue
        try:
            vector = [float(value) for value in row]
        except (TypeError, ValueError):
            continue
        matrix.append(vector)
    return matrix


def _normalize_sparse_embeddings(data: Any) -> list[list[float]] | None:
    """Return sparse embeddings when they align with the expected schema."""

    if not isinstance(data, list):
        return None

    normalized: list[list[float]] = []
    for entry in data:
        if not isinstance(entry, Iterable) or isinstance(entry, str | bytes):
            return None
        try:
            normalized.append([float(value) for value in entry])
        except (TypeError, ValueError):
            return None
    return normalized
