"""Minimal regression coverage for vector search data models."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.models.vector_search import (
    BatchSearchResponse,
    DimensionError,
    FilterValidationError,
    PrefetchConfig,
    SecureFilterModel,
    SecureVectorModel,
    VectorType,
)

BatchSearchResponse.model_rebuild(_types_namespace={"datetime": datetime})


def test_secure_vector_model_validates_dimensions() -> None:
    """Secure vector models should retain valid vectors."""
    model = SecureVectorModel(values=[0.1, -0.2, 0.3])
    assert model.values == [0.1, -0.2, 0.3]

    with pytest.raises(DimensionError):
        SecureVectorModel(values=[0.0] * 5000)


def test_secure_filter_model_blocks_injection_patterns() -> None:
    """Filter models enforce strict operator and value validation."""
    valid = SecureFilterModel(field="metadata.author", operator="eq", value="Ada")
    assert valid.operator == "eq"

    with pytest.raises(FilterValidationError):
        SecureFilterModel(field="metadata", operator="eq", value="DROP TABLE users")


def test_prefetch_config_scaling() -> None:
    """Prefetch limits should scale with vector types but respect caps."""
    config = PrefetchConfig(
        dense_multiplier=2.0,
        sparse_multiplier=1.5,
        hyde_multiplier=3.0,
        max_prefetch_limit=100,
    )
    assert config.calculate_prefetch_limit(VectorType.DENSE, 10) == 20
    assert config.calculate_prefetch_limit(VectorType.SPARSE, 10) == 15
    assert config.calculate_prefetch_limit(VectorType.HYDE, 50) == 100


def test_batch_search_response_success_rate() -> None:
    """Batch responses compute aggregate success rates."""
    response = BatchSearchResponse(
        responses=[], batch_time_ms=120, successful_queries=3, failed_queries=1
    )
    assert pytest.approx(response.success_rate, rel=1e-6) == 0.75
