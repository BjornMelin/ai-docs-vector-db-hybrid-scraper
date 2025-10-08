"""Tests for provider metadata normalization utilities."""

from __future__ import annotations

from src.mcp_tools.utils.provider_metadata import normalize_provider_catalog


def test_normalize_with_manager_data_merges_models():
    """Manager metadata should update model details while keeping fallbacks."""

    manager_catalog = {
        "openai": {
            "model": "text-embedding-3-small",
            "dimensions": 1536,
            "max_tokens": 8191,
            "cost_per_token": 0.0000004,
        }
    }

    providers = normalize_provider_catalog(manager_catalog)

    openai = next(p for p in providers if p.name == "openai")
    model_dump = openai.model_dump()

    assert model_dump["dims"] == 1536
    assert model_dump["context_length"] == 8191
    assert model_dump["models"][0]["name"] == "text-embedding-3-small"
    assert model_dump["models"][0]["cost_per_token"] == 0.0000004
    assert model_dump["models"][0]["cost_per_million"] == 0.0000004 * 1_000_000
    assert model_dump["models"][0]["dimensions"] == 1536


def test_normalize_uses_fallback_when_manager_missing():
    """When manager lacks provider, defaults should be returned."""

    providers = normalize_provider_catalog({})
    names = {provider.name for provider in providers}

    assert "openai" in names
    assert "fastembed" in names

    fastembed = next(p for p in providers if p.name == "fastembed")
    model_dump = fastembed.model_dump()

    assert model_dump["dims"] == 384
    assert model_dump["models"][0]["name"] == "BAAI/bge-small-en-v1.5"


def test_normalize_preserves_unknown_provider():
    """Providers without fallbacks should still be returned with raw data."""

    manager_catalog = {
        "custom-provider": {
            "model": "acme-embed-v1",
            "dimensions": 2048,
            "max_tokens": 1024,
        }
    }

    providers = normalize_provider_catalog(manager_catalog)

    custom = next(p for p in providers if p.name == "custom-provider")
    model_dump = custom.model_dump()

    assert model_dump["dims"] == 2048
    assert model_dump["context_length"] == 1024
    assert model_dump["models"][0]["name"] == "acme-embed-v1"
