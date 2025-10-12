"""Unit tests for the unified search request model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.models import (
    FusionAlgorithm,
    SearchAccuracy,
    SearchStrategy,
    VectorType,
)
from src.models.search import SearchRequest


def test_search_request_defaults() -> None:
    """Ensure minimal payloads populate default search parameters."""

    request = SearchRequest.model_validate({"query": "docs"})

    assert request.collection == "documentation"
    assert request.limit == 10
    assert request.vector_type is VectorType.DENSE
    assert request.search_accuracy is SearchAccuracy.BALANCED
    assert request.fusion_algorithm is FusionAlgorithm.RRF
    assert request.include_metadata is True
    assert request.include_vectors is False


def test_search_request_invalid_filter_key() -> None:
    """Reject filter keys that violate the allowed naming pattern."""

    with pytest.raises(ValidationError) as exc_info:
        SearchRequest.model_validate(
            {"query": "docs", "filters": {"invalid-field!": "value"}}
        )

    assert "Invalid filter key" in str(exc_info.value)


def test_search_request_filter_injection_guard() -> None:
    """Detect potentially dangerous patterns in filter values."""

    with pytest.raises(ValidationError) as exc_info:
        SearchRequest.model_validate(
            {"query": "docs", "filters": {"category": "DROP TABLE users"}}
        )

    assert "Potentially dangerous pattern" in str(exc_info.value)


def test_search_request_query_vector_validation() -> None:
    """Enforce dense vector dimension and numeric bounds."""

    large_vector = [0.1] * 5000
    with pytest.raises(ValidationError) as exc_info:
        SearchRequest.model_validate({"query": "docs", "query_vector": large_vector})
    assert "exceeds maximum allowed" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        SearchRequest.model_validate(
            {"query": "docs", "query_vector": [0.1, float("inf")]}
        )
    assert "Non-finite" in str(exc_info.value)


def test_search_request_sparse_vector_required() -> None:
    """Require sparse_vector payload when using sparse-compatible vector types."""

    with pytest.raises(ValidationError) as exc_info:
        SearchRequest.model_validate(
            {"query": "docs", "vector_type": VectorType.HYBRID}
        )

    assert "sparse_vector is required" in str(exc_info.value)


def test_search_request_force_dimension_mismatch() -> None:
    """Validate forced dimensions against provided query vectors."""

    with pytest.raises(ValidationError) as exc_info:
        SearchRequest.model_validate(
            {
                "query": "docs",
                "query_vector": [0.1, 0.2],
                "force_dimension": 3,
            }
        )

    assert "force_dimension" in str(exc_info.value)


def test_search_request_from_input_string_payload() -> None:
    """Normalize string payloads via SearchRequest.from_input."""

    request = SearchRequest.from_input("docs", collection="api", limit=5)

    assert isinstance(request, SearchRequest)
    assert request.query == "docs"
    assert request.collection == "api"
    assert request.limit == 5


def test_search_request_from_input_instance_passthrough() -> None:
    """Return the original object when overrides are absent."""

    base = SearchRequest.model_validate({"query": "docs", "limit": 20})
    clone = SearchRequest.from_input(base)

    assert clone is base


def test_search_request_valid_filter_groups() -> None:
    """Allow nested filter groups that respect structural limits."""

    request = SearchRequest.model_validate(
        {
            "query": "docs",
            "filters": {"category": "api"},
            "filter_groups": [
                {
                    "operator": "and",
                    "filters": [
                        {"tag": "python"},
                        {
                            "operator": "or",
                            "filters": [
                                {"tag": "fastapi"},
                                {"tag": "pydantic"},
                            ],
                        },
                    ],
                }
            ],
        }
    )

    assert request.filter_groups is not None
    assert len(request.filter_groups) == 1


def test_search_request_invalid_filter_group_operator() -> None:
    """Reject filter groups with unsupported operators."""

    with pytest.raises(ValidationError) as exc_info:
        SearchRequest.model_validate(
            {
                "query": "docs",
                "filter_groups": [
                    {
                        "operator": "xor",
                        "filters": [{"tag": "python"}],
                    }
                ],
            }
        )

    assert "operator" in str(exc_info.value)


def test_search_request_strategy_vector_alignment() -> None:
    """Enforce consistency between search_strategy and vector_type."""

    with pytest.raises(ValidationError) as exc_info:
        SearchRequest.model_validate(
            {
                "query": "docs",
                "search_strategy": SearchStrategy.SPARSE,
                "vector_type": VectorType.DENSE,
                "sparse_vector": {0: 0.1},
            }
        )

    assert "Sparse search_strategy" in str(exc_info.value)
