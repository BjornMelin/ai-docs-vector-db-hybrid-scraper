#!/usr/bin/env python3
"""RAG Integration Patterns Demonstration.

This script demonstrates the modern RAG implementation patterns integrated
into the AI Documentation Vector DB Hybrid Scraper system. It showcases:

1. Function-based dependency injection for RAG services
2. Circuit breaker patterns for LLM API resilience
3. Integration with existing vector search and embedding services
4. Modern RAG patterns with source attribution and confidence scoring
5. Portfolio-worthy implementation for 2025 AI/ML opportunities

Usage:
    python examples/rag_integration_demo.py
"""

import asyncio  # noqa: PLC0415
import logging  # noqa: PLC0415

# Add the src directory to the path
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import get_config
from src.infrastructure.client_manager import ClientManager
from src.services.dependencies import (
    RAGRequest,
    RAGResponse,
    clear_rag_cache,
    generate_rag_answer,
    get_rag_metrics,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_rag_patterns():
    """Demonstrate modern RAG integration patterns."""

    print("🚀 RAG Integration Patterns Demonstration")
    print("=" * 50)

    # Load configuration
    config = get_config()

    # Check if RAG is enabled
    if not config.rag.enable_rag:
        print("❌ RAG is not enabled in configuration")
        print("💡 To enable RAG, set AI_DOCS_RAG__ENABLE_RAG=true in your .env file")
        return

    print(f"✅ RAG enabled with model: {config.rag.model}")
    print(
        f"🔧 Configuration: {config.rag.max_tokens} max tokens, {config.rag.temperature} temperature"
    )
    print()

    # Initialize client manager
    client_manager = ClientManager.from_unified_config()
    await client_manager.initialize()

    try:
        # Pattern 1: Direct Service Integration
        print("📋 Pattern 1: Direct Service Integration")
        print("-" * 40)

        rag_generator = await client_manager.get_rag_generator()
        print("✅ RAG generator initialized successfully")

        # Get initial metrics
        initial_metrics = rag_generator.get_metrics()
        print(f"📊 Initial metrics: {initial_metrics['generation_count']} generations")
        print()

        # Pattern 2: Function-based Dependency Injection
        print("📋 Pattern 2: Function-based Dependency Injection")
        print("-" * 40)

        # Create sample search results (simulating vector search output)
        sample_search_results = [
            {
                "id": "doc_1",
                "title": "FastAPI Documentation - Getting Started",
                "content": "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints. It provides automatic API documentation, data validation, and serialization.",
                "url": "https://fastapi.tiangolo.com/tutorial/",
                "score": 0.95,
                "metadata": {"type": "documentation", "framework": "fastapi"},
            },
            {
                "id": "doc_2",
                "title": "Pydantic Models and Validation",
                "content": "Pydantic provides runtime type checking and data validation using Python type annotations. It automatically validates data, converts types, and provides detailed error messages for invalid data.",
                "url": "https://docs.pydantic.dev/",
                "score": 0.87,
                "metadata": {"type": "documentation", "framework": "pydantic"},
            },
            {
                "id": "doc_3",
                "title": "Python Type Hints Best Practices",
                "content": "Type hints in Python help with code clarity, IDE support, and static analysis. They don't affect runtime performance but significantly improve developer experience and code maintainability.",
                "url": "https://docs.python.org/3/library/typing.html",
                "score": 0.82,
                "metadata": {"type": "documentation", "language": "python"},
            },
        ]

        # Create RAG request
        rag_request = RAGRequest(
            query="How do I use type hints with FastAPI and Pydantic for API validation?",
            search_results=sample_search_results,
            include_sources=True,
            require_high_confidence=False,
            max_context_results=3,
        )

        print(f"🔍 Query: {rag_request.query}")
        print(f"📚 Using {len(rag_request.search_results)} search results as context")
        print()

        # Pattern 3: RAG Answer Generation with Error Handling
        print("📋 Pattern 3: RAG Answer Generation with Circuit Breaker")
        print("-" * 40)

        try:
            # Use function-based dependency injection pattern
            response: RAGResponse = await generate_rag_answer(
                rag_request, rag_generator
            )

            print("✅ RAG answer generated successfully")
            print(f"🎯 Confidence Score: {response.confidence_score:.2f}")
            print(f"📊 Sources Used: {response.sources_used}")
            print(f"⏱️  Generation Time: {response.generation_time_ms:.1f}ms")
            print()

            print("💬 Generated Answer:")
            print("-" * 20)
            print(response.answer)
            print()

            # Pattern 4: Source Attribution and Metrics
            if response.sources:
                print("📖 Source Attribution:")
                print("-" * 20)
                for i, source in enumerate(response.sources, 1):
                    print(
                        f"{i}. {source['title']} (Score: {source['relevance_score']:.2f})"
                    )
                    print(f"   {source['excerpt'][:100]}...")
                    if source["url"]:
                        print(f"   🔗 {source['url']}")
                    print()

            # Pattern 5: Advanced Metrics and Quality Assessment
            if response.metrics:
                print("📈 Quality Metrics:")
                print("-" * 20)
                metrics = response.metrics
                print(f"• Context Utilization: {metrics['context_utilization']:.2f}")
                print(f"• Source Diversity: {metrics['source_diversity']:.2f}")
                print(f"• Answer Length: {metrics['answer_length']} characters")
                print(f"• Tokens Used: {metrics['tokens_used']}")
                print(f"• Estimated Cost: ${metrics['cost_estimate']:.4f}")
                print()

            # Pattern 6: Follow-up Questions (Portfolio Feature)
            if response.follow_up_questions:
                print("❓ Follow-up Questions:")
                print("-" * 20)
                for question in response.follow_up_questions:
                    print(f"• {question}")
                print()

        except Exception as e:
            print(f"❌ RAG generation failed: {e}")
            print("🔧 This might be due to missing OpenAI API key or network issues")
            print()

        # Pattern 7: Service Metrics and Monitoring
        print("📋 Pattern 7: Service Metrics and Monitoring")
        print("-" * 40)

        final_metrics = await get_rag_metrics(rag_generator)
        print("📊 RAG Service Metrics:")
        for key, value in final_metrics.items():
            if isinstance(value, float):
                print(f"• {key.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"• {key.replace('_', ' ').title()}: {value}")
        print()

        # Pattern 8: Cache Management
        print("📋 Pattern 8: Cache Management")
        print("-" * 40)

        cache_result = await clear_rag_cache(rag_generator)
        print(f"🗑️  Cache Status: {cache_result['status']}")
        print(f"💬 Message: {cache_result['message']}")
        print()

        # Pattern 9: Health Monitoring
        print("📋 Pattern 9: Health Monitoring")
        print("-" * 40)

        health_status = await client_manager.get_health_status()
        if "rag_generator" in health_status:
            rag_health = health_status["rag_generator"]
            print(f"🏥 RAG Service Health: {rag_health.get('state', 'unknown')}")
            print(f"🔧 Last Check: {rag_health.get('last_check', 'never')}")
            if rag_health.get("last_error"):
                print(f"❌ Last Error: {rag_health['last_error']}")
        else:
            print("🏥 RAG Service Health: Not monitored")
        print()

        print("🎉 RAG Integration Patterns Demonstration Complete!")
        print()
        print("💼 Portfolio Value:")
        print("• Modern RAG implementation with LLM integration")
        print("• Function-based dependency injection patterns")
        print("• Circuit breaker resilience for production systems")
        print("• Comprehensive metrics and observability")
        print("• Source attribution and confidence scoring")
        print("• Enterprise-ready error handling and monitoring")

    finally:
        await client_manager.cleanup()


async def demonstrate_integration_with_search():
    """Demonstrate RAG integration with existing search capabilities."""

    print("\n🔍 RAG + Vector Search Integration")
    print("=" * 50)

    config = get_config()
    if not config.rag.enable_rag:
        print("❌ RAG not enabled - skipping integration demo")
        return

    client_manager = ClientManager.from_unified_config()
    await client_manager.initialize()

    try:
        # This would typically integrate with the existing vector search
        print("💡 Integration Points:")
        print("• Vector search results → RAG context")
        print("• HyDE query expansion → Enhanced RAG context")
        print("• Content intelligence → Source quality scoring")
        print("• Embedding services → Semantic similarity")
        print("• Qdrant collections → Multi-source retrieval")
        print()

        print("🏗️  Architecture Benefits:")
        print("• Unified service dependency pattern")
        print("• Consistent error handling and resilience")
        print("• Shared observability and metrics")
        print("• Function-based composition")
        print("• Production-ready patterns")

    finally:
        await client_manager.cleanup()


if __name__ == "__main__":
    """Main execution."""
    print("🤖 AI Documentation Vector DB - RAG Integration Patterns")
    print("🎯 Demonstrating 2025 RAG implementation best practices")
    print()

    asyncio.run(demonstrate_rag_patterns())
    asyncio.run(demonstrate_integration_with_search())
