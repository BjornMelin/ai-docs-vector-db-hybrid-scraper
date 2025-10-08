"""Tests covering the retrieval-augmented generation package exports."""

from __future__ import annotations

import operator

import pytest

from src.services import rag as rag_module
from src.services.rag import (
    AnswerMetrics,
    CompressionStats,
    LangGraphRAGPipeline,
    RAGConfig,
    RAGGenerator,
    RAGRequest,
    RAGResult,
    RAGServiceMetrics,
    RagTracingCallback,
    SourceAttribution,
    VectorServiceRetriever,
    build_default_rag_config,
    initialise_rag_generator,
)


EXPECTED_EXPORTS: tuple[tuple[str, object], ...] = (
    ("RAGGenerator", RAGGenerator),
    ("LangGraphRAGPipeline", LangGraphRAGPipeline),
    ("RagTracingCallback", RagTracingCallback),
    ("VectorServiceRetriever", VectorServiceRetriever),
    ("CompressionStats", CompressionStats),
    ("RAGConfig", RAGConfig),
    ("RAGRequest", RAGRequest),
    ("RAGResult", RAGResult),
    ("RAGServiceMetrics", RAGServiceMetrics),
    ("AnswerMetrics", AnswerMetrics),
    ("SourceAttribution", SourceAttribution),
    ("build_default_rag_config", build_default_rag_config),
    ("initialise_rag_generator", initialise_rag_generator),
)


@pytest.mark.parametrize("export_name, expected_object", EXPECTED_EXPORTS)
def test_rag_package_exports_resolve_types(
    export_name: str, expected_object: object
) -> None:
    """Ensure RAG exports expose the expected curated API.

    Args:
        export_name: Symbol name published via :mod:`src.services.rag`.

    Returns:
        None: This test asserts each export is discoverable at the package boundary.
    """

    exported_object = getattr(rag_module, export_name)

    assert exported_object is expected_object


def test_rag_all_exports_alignment() -> None:
    """Validate ``__all__`` enumerates the canonical RAG package API.

    Returns:
        None: This test ensures the module contract remains deliberate.
    """

    assert isinstance(rag_module.__all__, tuple)

    with pytest.raises(TypeError):
        operator.setitem(rag_module.__all__, 0, "something_else")  # type: ignore[operator]

    expected_exports = tuple(name for name, _ in EXPECTED_EXPORTS)

    assert rag_module.__all__ == expected_exports


def test_compression_stats_dataclass_behaviour() -> None:
    """Ensure compression statistics expose actionable ratios for observability.

    Returns:
        None: This test validates the ratios remain accessible to callers.
    """

    stats = CompressionStats(documents_processed=10, tokens_before=100, tokens_after=40)

    assert stats.reduction_ratio == pytest.approx(0.6)
    assert stats.to_dict()["reduction_ratio"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("tokens_before", "tokens_after", "expected_ratio"),
    (
        (0, 10, 0.0),
        (-5, 10, 0.0),
        (10, -5, 1.0),
        (0, 0, 0.0),
    ),
)
def test_compression_stats_edge_cases(
    tokens_before: int, tokens_after: int, expected_ratio: float
) -> None:
    """Ensure ``CompressionStats`` handles non-positive token counts safely."""

    stats = CompressionStats(tokens_before=tokens_before, tokens_after=tokens_after)

    assert stats.reduction_ratio == expected_ratio
