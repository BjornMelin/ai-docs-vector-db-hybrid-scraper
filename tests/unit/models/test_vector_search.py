"""Vector search request validation tests."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from src.config.models import SearchStrategy, VectorType
from src.models.search import SearchRequest


@pytest.fixture
def base_payload() -> dict[str, object]:
    """Provide a reusable payload for mutation in tests."""

    return {"query": "find docs"}


class TestHybridAndSparseValidation:
    """Hybrid/sparse vector rules."""

    def test_sparse_strategy_requires_sparse_payload(
        self, base_payload: Mapping[str, object]
    ) -> None:
        payload = dict(base_payload)
        payload.update(
            {"search_strategy": SearchStrategy.SPARSE, "vector_type": VectorType.SPARSE}
        )

        with pytest.raises(ValidationError, match="sparse_vector is required"):
            SearchRequest.model_validate(payload)

    def test_hybrid_request_with_sparse_vector_valid(
        self, base_payload: Mapping[str, object]
    ) -> None:
        payload = dict(base_payload)
        payload.update(
            {
                "vector_type": VectorType.HYBRID,
                "search_strategy": SearchStrategy.HYBRID,
                "sparse_vector": {0: 0.7, 5: 0.2},
            }
        )

        request = SearchRequest.model_validate(payload)

        assert request.sparse_vector == {0: 0.7, 5: 0.2}
        assert request.vector_type is VectorType.HYBRID

    def test_sparse_vector_weight_must_be_numeric(
        self, base_payload: Mapping[str, object]
    ) -> None:
        payload = dict(base_payload)
        payload.update(
            {
                "vector_type": VectorType.HYBRID,
                "sparse_vector": {0: "weight"},
            }
        )

        with pytest.raises(ValidationError, match="unable to parse string as a number"):
            SearchRequest.model_validate(payload)


class TestDenseVectorValidation:
    """Dense vector bounds."""

    def test_query_vector_dimension_upper_bound(
        self, base_payload: Mapping[str, object]
    ) -> None:
        payload = dict(base_payload)
        payload["query_vector"] = [0.1] * 5001

        with pytest.raises(ValidationError, match="exceeds maximum allowed dimension"):
            SearchRequest.model_validate(payload)

    def test_query_vector_value_range(self, base_payload: Mapping[str, object]) -> None:
        payload = dict(base_payload)
        payload["query_vector"] = [0.1, 2e7]

        with pytest.raises(ValidationError, match="Vector value out of allowed range"):
            SearchRequest.model_validate(payload)


class TestFilterValidation:
    """Filtering rules and protections."""

    def test_filter_list_length_limit(self, base_payload: Mapping[str, object]) -> None:
        payload = dict(base_payload)
        payload["filters"] = {f"field_{idx}": "value" for idx in range(51)}

        with pytest.raises(ValidationError, match="Too many filter keys supplied"):
            SearchRequest.model_validate(payload)

    def test_exclude_filters_require_mappings(
        self, base_payload: Mapping[str, object]
    ) -> None:
        payload = dict(base_payload)
        payload["exclude_filters"] = ["not-a-mapping"]  # type: ignore[list-item]

        with pytest.raises(ValidationError, match="Input should be a valid dictionary"):
            SearchRequest.model_validate(payload)

    def test_filter_groups_depth_limit(
        self, base_payload: Mapping[str, object]
    ) -> None:
        nested_group: dict[str, object] = {
            "operator": "and",
            "filters": [{"field": "lang", "value": "en"}],
        }
        for _ in range(12):
            nested_group = {"operator": "and", "filters": [nested_group]}

        payload = dict(base_payload)
        payload["filter_groups"] = [nested_group]

        with pytest.raises(ValidationError, match="Filter group nesting too deep"):
            SearchRequest.model_validate(payload)


class TestExtraFields:
    """Model config extra forbid behaviour."""

    def test_extra_fields_are_rejected(
        self, base_payload: Mapping[str, object]
    ) -> None:
        payload = dict(base_payload)
        payload["unexpected_field"] = "value"

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            SearchRequest.model_validate(payload)
