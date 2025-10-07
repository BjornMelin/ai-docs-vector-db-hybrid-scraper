"""Retrieval-augmented generation service package."""

from __future__ import annotations

from typing import Final

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
from .retriever import CompressionStats, VectorServiceRetriever
from .utils import build_default_rag_config, initialise_rag_generator


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
