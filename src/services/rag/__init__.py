"""Retrieval-augmented generation service package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from .generator import RAGGenerator
from .langgraph_pipeline import LangGraphRAGPipeline, RagTracingCallback
from .models import (
    AnswerMetrics,
    RAGConfig,
    RAGRequest,
    RAGResult,
    RAGServiceMetrics,
    SourceAttribution,
)
from .utils import build_default_rag_config, initialise_rag_generator


if TYPE_CHECKING:
    from .retriever import CompressionStats, VectorServiceRetriever


__all__: Final[tuple[str, ...]] = (
    "RAGGenerator",
    "LangGraphRAGPipeline",
    "RagTracingCallback",
    "VectorServiceRetriever",
    "CompressionStats",
    "RAGConfig",
    "RAGRequest",
    "RAGResult",
    "RAGServiceMetrics",
    "AnswerMetrics",
    "SourceAttribution",
    "build_default_rag_config",
    "initialise_rag_generator",
)

_LAZY_EXPORTS: Final[dict[str, str]] = {
    "CompressionStats": "CompressionStats",
    "VectorServiceRetriever": "VectorServiceRetriever",
}


def __getattr__(name: str) -> Any:
    """Lazily import optional retriever dependencies on first access."""
    if name in _LAZY_EXPORTS:
        from . import retriever as _retriever  # pylint: disable=import-outside-toplevel

        value = getattr(_retriever, _LAZY_EXPORTS[name])
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazily-loaded exports when introspecting the module."""
    return sorted(set(__all__) | set(globals().keys()))
