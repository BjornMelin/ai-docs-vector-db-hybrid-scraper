"""Tests for query processing data models."""

from __future__ import annotations

import pytest

from src.services.query_processing.models import SearchRequest


def test_search_request_from_input_returns_existing_instance() -> None:
    original = SearchRequest(query="install agent", limit=3)
    result = SearchRequest.from_input(original)
    assert result is original


def test_search_request_from_input_applies_overrides() -> None:
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


def test_search_request_from_input_accepts_string() -> None:
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


def test_search_request_from_input_accepts_mapping() -> None:
    payload = {"query": "install agent", "limit": 2}
    result = SearchRequest.from_input(
        payload,
        collection="docs",
        enable_expansion=False,
    )
    assert result.limit == 2  # existing value preserved
    assert result.collection == "docs"
    assert result.enable_expansion is False


def test_search_request_from_input_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError):
        SearchRequest.from_input(123)  # type: ignore[arg-type]
