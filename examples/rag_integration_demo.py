#!/usr/bin/env python3
# ruff: noqa: E402, I001

"""RAG Integration Patterns Demonstration.

This script demonstrates the updated LangChain-backed RAG pipeline that now
retrieves context directly from the vector adapter. It showcases:

1. Initialising the shared VectorStoreService and LangChain retriever
2. Executing the RAG generator via the function-based dependency wrappers
3. Emitting metrics and managing cache endpoints

Usage:
    python examples/rag_integration_demo.py
"""

# pylint: disable=wrong-import-position
from __future__ import annotations

import asyncio
import sys
from pathlib import Path


SRC_ROOT = Path(__file__).parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.config import Config, get_config
from src.infrastructure.client_manager import ClientManager
from src.services.dependencies import (
    RAGResponse,
    RAGRequest,
    clear_rag_cache,
    generate_rag_answer,
    get_rag_metrics,
)
from src.services.errors import EmbeddingServiceError
from src.services.rag.generator import RAGGenerator
from src.services.rag.models import RAGConfig as ServiceRAGConfig
from src.services.rag.utils import initialise_rag_generator


def _print_banner() -> None:
    """Display the demonstration banner."""

    print("🚀 RAG Integration Patterns Demonstration")
    print("=" * 50)


def _print_configuration(config: Config) -> None:
    """Print the active RAG configuration."""

    print(f"✅ RAG enabled with model: {config.rag.model}")
    print(
        "🔧 Configuration: "
        f"{config.rag.max_tokens} max tokens, {config.rag.temperature} temperature"
    )
    print()


async def _initialise_generator(
    client_manager: ClientManager, config: Config
) -> tuple[RAGGenerator, ServiceRAGConfig]:
    """Initialise the vector-backed RAG generator."""

    vector_store = await client_manager.get_vector_store_service()
    generator, rag_config = await initialise_rag_generator(config, vector_store)
    print("✅ RAG generator initialized successfully")
    return generator, rag_config


def _display_initial_metrics(generator: RAGGenerator) -> None:
    """Print initial generator metrics."""

    metrics = generator.get_metrics()
    print(f"📊 Initial metrics: {metrics.generation_count} generations recorded")
    print()


def _demo_rag_request(rag_config: ServiceRAGConfig) -> RAGRequest:
    """Create the sample RAG request used in the demo."""

    request = RAGRequest(
        query=("How do I use type hints with FastAPI and Pydantic for API validation?"),
        include_sources=True,
        search_results=[],
        max_tokens=None,
        temperature=None,
    )
    print("📋 Pattern 2: Function-based Dependency Injection")
    print("-" * 40)
    print(f"🔍 Query: {request.query}")
    print(f"📚 Retrieving up to {rag_config.retriever_top_k} documents")
    print()
    return request


def _display_answer(response: RAGResponse) -> None:
    """Pretty-print the generated answer and metadata."""

    print("✅ RAG answer generated successfully")
    if response.confidence_score is not None:
        print(f"🎯 Confidence Score: {response.confidence_score:.2f}")
    print(f"📊 Sources Used: {response.sources_used}")
    print(f"⏱️  Generation Time: {response.generation_time_ms:.1f} ms")
    print()

    print("💬 Generated Answer:")
    print("-" * 20)
    print(response.answer)
    print()

    if response.sources:
        print("📖 Source Attribution:")
        print("-" * 20)
        for idx, source in enumerate(response.sources, 1):
            relevance = source.get("relevance_score")
            score_text = f" (Score: {relevance:.2f})" if relevance else ""
            print(f"{idx}. {source['title']}{score_text}")
            excerpt = source.get("excerpt")
            if excerpt:
                print(f"   {excerpt[:100]}...")
            if source.get("url"):
                print(f"   🔗 {source['url']}")
            print()

    if response.metrics:
        print("📈 Quality Metrics:")
        print("-" * 20)
        metrics = response.metrics
        print(f"• Generation Time: {metrics['generation_time_ms']:.1f} ms")
        if metrics.get("total_tokens") is not None:
            print(f"• Total Tokens: {metrics['total_tokens']}")
        print()


async def _run_rag_workflow(
    rag_request: RAGRequest, rag_generator: RAGGenerator
) -> None:
    """Execute the answer generation workflow and display results."""

    print("📋 Pattern 3: RAG Answer Generation with Circuit Breaker")
    print("-" * 40)
    try:
        response: RAGResponse = await generate_rag_answer(rag_request, rag_generator)
    except EmbeddingServiceError as err:
        print("⚠️  Retrieval produced no context; add documents to the vector store.")
        print(f"   Details: {err}")
        print()
        return

    _display_answer(response)


async def _display_observability(rag_generator: RAGGenerator) -> None:
    """Showcase metrics and cache helpers."""

    print("📋 Pattern 4: Metrics & Observability")
    print("-" * 40)
    metrics = await get_rag_metrics(rag_generator)
    for key, value in metrics.items():
        print(f"• {key}: {value}")
    print()

    print("📋 Pattern 5: Cache Management")
    print("-" * 40)
    cache_result = await clear_rag_cache(rag_generator)
    print(f"🧹 Cache Clear Result: {cache_result['message']}")
    print()


async def demonstrate_rag_patterns() -> None:
    """Demonstrate the updated LangChain-enabled RAG workflow."""

    _print_banner()
    config = get_config()
    if not config.rag.enable_rag:
        print("❌ RAG is not enabled in configuration")
        print("💡 To enable RAG, set AI_DOCS_RAG__ENABLE_RAG=true in your .env file")
        return

    _print_configuration(config)

    client_manager = ClientManager.from_unified_config()
    await client_manager.initialize()

    rag_generator: RAGGenerator | None = None
    try:
        print("📋 Pattern 1: Direct Service Integration")
        print("-" * 40)
        rag_generator, rag_config = await _initialise_generator(client_manager, config)
        _display_initial_metrics(rag_generator)

        rag_request = _demo_rag_request(rag_config)
        await _run_rag_workflow(rag_request, rag_generator)
        await _display_observability(rag_generator)
    finally:
        if rag_generator and rag_generator.llm_client_available:
            await rag_generator.cleanup()
        await client_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(demonstrate_rag_patterns())
