"""Unit tests for the canonical SearchRequest contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import ValidationError

from src.config.models import SearchAccuracy, SearchStrategy, VectorType
from src.models.search import SearchRequest


@pytest.fixture
def base_payload() -> dict[str, Any]:
    """Provide a minimal payload used across tests."""

    return {"query": "vector databases"}


class TestConstruction:
    """Construction-time behaviours."""

    def test_minimal_payload_applies_defaults(
        self, base_payload: Mapping[str, Any]
    ) -> None:
        """Model defaults should match canonical configuration."""

        request = SearchRequest.model_validate(base_payload)

        assert request.collection == "documentation"
        assert request.limit == 10
        assert request.search_strategy == SearchStrategy.HYBRID
        assert request.search_accuracy == SearchAccuracy.BALANCED
        assert request.vector_type == VectorType.DENSE
        assert request.enable_expansion is True
        assert request.enable_personalization is False
        assert request.filters is None

    def test_rejects_empty_query(self) -> None:
        """Empty queries violate minimum length constraints."""

        with pytest.raises(ValidationError):
            SearchRequest.model_validate({"query": ""})


class TestFromInput:
    """Normalisation helper behaviour."""

    def test_returns_existing_instance_when_no_overrides(
        self, base_payload: Mapping[str, Any]
    ) -> None:
        """Existing instances are returned verbatim if no overrides provided."""

        original = SearchRequest.model_validate(base_payload)
        result = SearchRequest.from_input(original)

        assert result is original

    def test_applies_overrides_and_returns_new_instance(
        self, base_payload: Mapping[str, Any]
    ) -> None:
        """Overrides should produce an updated copy."""

        original = SearchRequest.model_validate(base_payload)
        result = SearchRequest.from_input(
            original,
            limit=25,
            enable_rag=True,
            search_accuracy=SearchAccuracy.EXACT,
        )

        assert result is not original
        assert result.limit == 25
        assert result.enable_rag is True
        assert result.search_accuracy == SearchAccuracy.EXACT
        assert original.limit == 10

    def test_accepts_plain_string_payload(self) -> None:
        """String payloads are turned into SearchRequest instances."""

        request = SearchRequest.from_input(
            "install agent",
            collection="docs",
            limit=5,
        )

        assert request.query == "install agent"
        assert request.collection == "docs"
        assert request.limit == 5

    def test_accepts_mapping_payload(self, base_payload: Mapping[str, Any]) -> None:
        """Dictionary payloads merge with explicit overrides."""

        payload = dict(base_payload)
        payload["limit"] = 3
        request = SearchRequest.from_input(payload, enable_expansion=False)

        assert request.limit == 3
        assert request.enable_expansion is False

    def test_rejects_unsupported_payload_type(self) -> None:
        """Unsupported types raise :class:`TypeError`."""

        with pytest.raises(TypeError):
            SearchRequest.from_input(123)  # type: ignore[arg-type]


class TestValidation:
    """Field and cross-field validation behaviour."""

    @pytest.mark.parametrize(
        "vector_type",
        [
            VectorType.SPARSE,
            VectorType.HYBRID,
        ],
    )
    def test_sparse_vector_required_for_sparse_types(
        self,
        vector_type: VectorType,
        base_payload: Mapping[str, Any],
    ) -> None:
        """Sparse-capable vector types mandate sparse_vector data."""

        payload = dict(base_payload)
        payload["vector_type"] = vector_type

        with pytest.raises(ValidationError, match="sparse_vector is required"):
            SearchRequest.model_validate(payload)

    def test_sparse_strategy_requires_sparse_vector_type(
        self, base_payload: Mapping[str, Any]
    ) -> None:
        """Sparse strategy cannot be paired with a purely dense vector type."""

        payload = dict(base_payload)
        payload["search_strategy"] = SearchStrategy.SPARSE

        with pytest.raises(ValidationError, match="sparse-compatible vector_type"):
            SearchRequest.model_validate(payload)

    def test_dense_strategy_rejects_sparse_vector_type(
        self, base_payload: Mapping[str, Any]
    ) -> None:
        """Dense strategy rejects sparse-only vector configurations."""

        payload = dict(base_payload)
        payload["vector_type"] = VectorType.SPARSE
        payload["search_strategy"] = SearchStrategy.DENSE
        payload["sparse_vector"] = {0: 1.0}

        with pytest.raises(
            ValidationError,
            match="Dense search_strategy cannot be paired with sparse-only",
        ):
            SearchRequest.model_validate(payload)

    def test_query_vector_length_must_match_force_dimension(
        self, base_payload: Mapping[str, Any]
    ) -> None:
        """force_dimension must match query_vector length."""

        payload = dict(base_payload)
        payload["query_vector"] = [0.1, 0.2, 0.3]
        payload["force_dimension"] = 2

        with pytest.raises(ValidationError, match="force_dimension"):
            SearchRequest.model_validate(payload)

    def test_query_vector_rejects_non_finite_values(
        self, base_payload: Mapping[str, Any]
    ) -> None:
        """Non-finite values are invalid."""

        payload = dict(base_payload)
        payload["query_vector"] = [0.1, float("nan")]

        with pytest.raises(ValidationError, match="Non-finite vector value"):
            SearchRequest.model_validate(payload)

    def test_sparse_vector_rejects_non_numeric_indices(
        self, base_payload: Mapping[str, Any]
    ) -> None:
        """Sparse vector indices must coerce to integers."""

        payload = dict(base_payload)
        payload["vector_type"] = VectorType.HYBRID
        payload["sparse_vector"] = {"bad": 0.5}

        with pytest.raises(
            ValidationError, match="unable to parse string as an integer"
        ):
            SearchRequest.model_validate(payload)

    def test_filters_disallow_invalid_keys(
        self, base_payload: Mapping[str, Any]
    ) -> None:
        """Filter keys must match the allowed regex."""

        payload = dict(base_payload)
        payload["filters"] = {"invalid key": "value"}

        with pytest.raises(ValidationError, match="Invalid filter key"):
            SearchRequest.model_validate(payload)

    def test_filters_disallow_dangerous_values(
        self, base_payload: Mapping[str, Any]
    ) -> None:
        """Dangerous patterns in filter values are rejected."""

        payload = dict(base_payload)
        payload["filters"] = {"title": "DROP TABLE docs"}

        with pytest.raises(ValidationError, match="dangerous pattern"):
            SearchRequest.model_validate(payload)

    def test_filter_groups_validate_structure(
        self, base_payload: Mapping[str, Any]
    ) -> None:
        """Nested filter groups must use permitted operators and entries."""

        payload = dict(base_payload)
        payload["filter_groups"] = [
            {"operator": "and", "filters": [{"field": "language", "value": "en"}]}
        ]

        request = SearchRequest.model_validate(payload)

        assert request.filter_groups is not None
        assert request.filter_groups[0]["operator"] == "and"

    def test_filter_groups_reject_invalid_operator(
        self, base_payload: Mapping[str, Any]
    ) -> None:
        """Invalid operators should trigger validation failure."""

        payload = dict(base_payload)
        payload["filter_groups"] = [
            {"operator": "xor", "filters": [{"field": "language", "value": "en"}]}
        ]

        with pytest.raises(
            ValidationError, match="operator must be 'and', 'or', or 'not'"
        ):
            SearchRequest.model_validate(payload)
