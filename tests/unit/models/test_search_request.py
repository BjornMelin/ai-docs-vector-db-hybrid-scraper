"""Tests for the canonical :mod:`src.models.search` module."""

from __future__ import annotations

import pytest

from src.models.search import SearchRequest


def test_from_input_returns_existing_instance_when_no_overrides() -> None:
    """Existing :class:`SearchRequest` instances pass through untouched."""

    original = SearchRequest(query="install agent", limit=3)
    result = SearchRequest.from_input(original)
    assert result is original


def test_from_input_applies_overrides() -> None:
    """Field overrides should yield a new instance with updates applied."""

    original = SearchRequest(query="install agent", limit=3)
    result = SearchRequest.from_input(
        original,
        collection="docs",
        limit=5,
        enable_rag=True,
    )
    assert result is not original
    assert result.collection == "docs"
    assert result.limit == 5
    assert result.enable_rag is True


def test_from_input_accepts_plain_string() -> None:
    """Raw query strings are normalised into a :class:`SearchRequest`."""

    result = SearchRequest.from_input(
        "install agent",
        collection="docs",
        limit=7,
        enable_personalization=True,
    )
    assert result.query == "install agent"
    assert result.collection == "docs"
    assert result.limit == 7
    assert result.enable_personalization is True


def test_from_input_accepts_mapping() -> None:
    """Dictionary payloads should be merged with overrides."""

    payload = {"query": "install agent", "limit": 2}
    result = SearchRequest.from_input(
        payload,
        collection="docs",
        enable_expansion=False,
    )
    assert result.limit == 2
    assert result.collection == "docs"
    assert result.enable_expansion is False


def test_from_input_rejects_unsupported_type() -> None:
    """Unsupported payloads raise :class:`TypeError`."""

    with pytest.raises(TypeError):
        SearchRequest.from_input(123)  # type: ignore[arg-type]
