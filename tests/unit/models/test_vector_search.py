"""Unit tests for secure vector search models."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from src.models.vector_search import (
    AdvancedFilteredSearchRequest,
    BasicSearchRequest,
    DimensionError,
    FilterValidationError,
    SecureFilterGroupModel,
    SecureFilterModel,
    SecureSparseVectorModel,
    SecureVectorModel,
)


class TestSecureVectorModel:
    """Validate vector sanitisation."""

    def test_valid_vector_exposes_dimension_and_magnitude(self) -> None:
        vector = SecureVectorModel(values=[0.0, 3.0, 4.0])

        assert vector.dimension == 3
        assert math.isclose(vector.magnitude, 5.0)

    def test_invalid_vector_raises_dimension_error(self) -> None:
        with pytest.raises(DimensionError):
            SecureVectorModel(values=list(range(4097)))

    def test_invalid_value_raises_value_error(self) -> None:
        with pytest.raises(ValidationError):
            SecureVectorModel(values=[0.0, float("nan")])


class TestSecureSparseVectorModel:
    """Ensure sparse vector validation catches mismatches."""

    def test_duplicate_indices_raise(self) -> None:
        with pytest.raises(ValidationError):
            SecureSparseVectorModel(indices=[0, 0], values=[0.1, 0.2])

    def test_valid_sparse_vector(self) -> None:
        sparse = SecureSparseVectorModel(indices=[0, 2], values=[0.5, 0.1])
        assert sparse.indices == [0, 2]


class TestSecureFilterModels:
    """Security validation for filters and groups."""

    def test_filter_detects_dangerous_pattern(self) -> None:
        with pytest.raises(FilterValidationError):
            SecureFilterModel(
                field="metadata.title",
                operator="regex",
                value="DROP TABLE users",
            )

    def test_filter_group_depth_limit(self) -> None:
        group = SecureFilterGroupModel(
            operator="and",
            filters=[SecureFilterModel(field="a", operator="eq", value="1")],
        )
        assert group.operator == "and"

        deep_group = group
        for _ in range(10):
            deep_group = SecureFilterGroupModel(operator="and", filters=[deep_group])

        with pytest.raises(FilterValidationError):
            SecureFilterGroupModel(
                operator="and",
                filters=[deep_group],
            )


class TestSearchRequests:
    """Ensure request wrappers enforce sane defaults."""

    def test_basic_search_request_sets_defaults(self) -> None:
        request = BasicSearchRequest(
            query_vector=SecureVectorModel(values=[0.1, 0.2]),
        )

        params = request.model_dump()["search_params"]
        assert params["limit"] == 10
        assert params["ef_search"] >= params["limit"]

    def test_advanced_filtered_request_accepts_groups(self) -> None:
        request = AdvancedFilteredSearchRequest(
            query_vector=SecureVectorModel(values=[0.5, 0.9]),
            filter_groups=[
                SecureFilterGroupModel(
                    operator="and",
                    filters=[
                        SecureFilterModel(field="tags", operator="in", value=["docs"]),
                    ],
                )
            ],
        )

        filters_dump = request.model_dump()["filter_groups"][0]["filters"][0]
        assert filters_dump["field"] == "tags"
