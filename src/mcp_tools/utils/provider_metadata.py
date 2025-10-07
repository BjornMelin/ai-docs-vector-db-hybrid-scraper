"""Utilities for normalizing embedding provider metadata."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from src.mcp_tools.models.responses import EmbeddingProviderInfo


FALLBACK_PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {
        "name": "openai",
        "status": "available",
        "features": ["dense", "async", "batching"],
        "models": [
            {
                "name": "text-embedding-3-small",
                "dimensions": 1536,
                "max_tokens": 8191,
                "cost_per_million": 0.02,
            },
            {
                "name": "text-embedding-3-large",
                "dimensions": 3072,
                "max_tokens": 8191,
                "cost_per_million": 0.13,
            },
            {
                "name": "text-embedding-ada-002",
                "dimensions": 1536,
                "max_tokens": 8191,
                "cost_per_million": 0.10,
            },
        ],
    },
    "fastembed": {
        "name": "fastembed",
        "status": "available",
        "features": ["dense", "sparse", "local", "cpu", "gpu"],
        "models": [
            {
                "name": "BAAI/bge-small-en-v1.5",
                "dimensions": 384,
                "max_tokens": 512,
                "cost_per_million": 0.0,
            },
            {
                "name": "BAAI/bge-base-en-v1.5",
                "dimensions": 768,
                "max_tokens": 512,
                "cost_per_million": 0.0,
            },
            {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": 384,
                "max_tokens": 256,
                "cost_per_million": 0.0,
            },
        ],
    },
}


def normalize_provider_catalog(
    catalog: Mapping[str, Mapping[str, Any]],
) -> list[EmbeddingProviderInfo]:
    """Normalize embedding provider metadata using manager data and fallbacks."""

    provider_ids = sorted(set(catalog.keys()) | set(FALLBACK_PROVIDERS.keys()))
    normalized: list[EmbeddingProviderInfo] = []

    for provider_id in provider_ids:
        raw = catalog.get(provider_id, {})
        fallback = FALLBACK_PROVIDERS.get(provider_id, {})
        provider_dict = _merge_provider_metadata(provider_id, raw, fallback)
        normalized.append(EmbeddingProviderInfo(**provider_dict))

    return normalized


def _merge_provider_metadata(
    provider_id: str,
    raw: Mapping[str, Any],
    fallback: Mapping[str, Any],
) -> dict[str, Any]:
    data: dict[str, Any] = {}

    data["name"] = fallback.get("name", provider_id)
    data["status"] = fallback.get("status")
    data["features"] = deepcopy(fallback.get("features", [])) or None

    primary_dims = raw.get("dimensions") or _first_model_field(fallback, "dimensions")
    if primary_dims is not None:
        data["dims"] = int(primary_dims)

    context_length = raw.get("max_tokens") or _first_model_field(fallback, "max_tokens")
    if context_length is not None:
        data["context_length"] = int(context_length)

    data["models"] = _merge_models(raw, fallback)

    cost_per_token = raw.get("cost_per_token")
    if cost_per_token is not None:
        data["cost_per_token"] = cost_per_token
        data.setdefault("models", [])
        if data["models"]:
            data["models"][0].setdefault("cost_per_token", cost_per_token)
            data["models"][0].setdefault("cost_per_million", cost_per_token * 1_000_000)
    else:
        primary_cost = _first_model_field(fallback, "cost_per_million")
        if primary_cost is not None:
            data["cost_per_million"] = primary_cost

    # Clean None values for readability
    return {key: value for key, value in data.items() if value is not None}


def _merge_models(
    raw: Mapping[str, Any], fallback: Mapping[str, Any]
) -> list[dict[str, Any]] | None:
    fallback_models = [deepcopy(model) for model in fallback.get("models", [])]
    raw_model_name = raw.get("model")
    if not raw_model_name:
        return fallback_models or None

    raw_entry = {
        "name": raw_model_name,
        "dimensions": raw.get("dimensions"),
        "max_tokens": raw.get("max_tokens"),
    }
    if raw.get("cost_per_token") is not None:
        raw_entry["cost_per_token"] = raw["cost_per_token"]
        raw_entry["cost_per_million"] = raw["cost_per_token"] * 1_000_000

    for existing in fallback_models:
        if existing.get("name") == raw_model_name:
            existing.update({k: v for k, v in raw_entry.items() if v is not None})
            break
    else:
        fallback_models.insert(0, raw_entry)

    # Remove None values inside model dicts
    for model in fallback_models:
        keys_to_del = [key for key, value in model.items() if value is None]
        for key in keys_to_del:
            del model[key]

    return fallback_models


def _first_model_field(data: Mapping[str, Any], field: str) -> Any:
    models = data.get("models") or []
    if not models:
        return None
    return models[0].get(field)
