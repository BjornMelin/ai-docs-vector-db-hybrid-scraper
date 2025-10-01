"""Utility helpers for Retrieval-Augmented Generation services."""

from __future__ import annotations

from src.config import Config
from src.services.vector_db.service import VectorStoreService

from .generator import RAGGenerator
from .models import RAGConfig
from .retriever import VectorServiceRetriever


def build_default_rag_config(config: Config) -> RAGConfig:
    """Construct RAG configuration values from the global application config."""

    return RAGConfig(
        model=config.rag.model,
        temperature=config.rag.temperature,
        max_tokens=config.rag.max_tokens,
        retriever_top_k=getattr(config.rag, "max_results_for_context", 5),
        include_sources=config.rag.include_sources,
        confidence_from_scores=getattr(config.rag, "include_confidence_score", True),
    )


async def initialise_rag_generator(
    config: Config, vector_store: VectorStoreService
) -> tuple[RAGGenerator, RAGConfig]:
    """Create and initialise a LangChain-backed RAG generator."""

    rag_config = build_default_rag_config(config)
    retriever = VectorServiceRetriever(
        vector_service=vector_store,
        collection=getattr(config.qdrant, "collection_name", "documents"),
        k=rag_config.retriever_top_k,
    )
    generator = RAGGenerator(rag_config, retriever)
    await generator.initialize()
    return generator, rag_config
