"""Demo of simplified observability for AI documentation scraper.

This example shows how to use the optimized observability features
that are perfect for a portfolio project - impressive but not over-engineered.
"""

import asyncio

# Import from the simplified observability module
from src.services.observability import (
    cost_tracker,
    setup_tracing,
    trace_operation,
    track_ai_cost,
    track_ai_operation,
    track_vector_search_simple,
)


# Example 1: Track embedding generation with cost
@track_ai_cost(provider="openai", model="text-embedding-3-small", operation="embedding")
async def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    print(f"Generating embeddings for {len(texts)} texts...")

    # Simulate embedding generation
    await asyncio.sleep(0.05)  # 50ms latency

    # Return mock embeddings (1536 dimensions for OpenAI)
    return [[0.1] * 1536 for _ in texts]


# Example 2: Track vector search operations
@track_vector_search_simple(collection="documentation", top_k=5)
async def search_similar_docs(query_embedding: list[float]) -> list[dict]:  # noqa: ARG001
    """Search for similar documents using vector similarity."""
    print("Searching vector database...")

    # Simulate search
    await asyncio.sleep(0.01)  # 10ms latency

    # Return mock results
    return [
        {"id": f"doc_{i}", "score": 0.95 - (i * 0.05), "title": f"Document {i}"}
        for i in range(5)
    ]


# Example 3: Track a complete RAG pipeline
async def rag_pipeline(query: str) -> str:
    """Complete RAG pipeline with tracking."""

    # Track the overall pipeline
    with trace_operation("rag_pipeline", operation_type="pipeline", query=query):
        # Step 1: Generate query embedding
        with track_ai_operation(
            operation_type="embedding_generation",
            provider="openai",
            model="text-embedding-3-small",
            input_texts=[query],
        ) as embed_context:
            query_embedding = await generate_embeddings([query])
            embed_context["embeddings"] = query_embedding
            embed_context["cost"] = 0.00002  # $0.02 per 1M tokens

        # Step 2: Search for relevant documents
        search_results = await search_similar_docs(query_embedding[0])

        # Step 3: Generate response using LLM
        with track_ai_operation(
            operation_type="llm_call",
            provider="openai",
            model="gpt-3.5-turbo",
            operation="completion",
        ) as llm_context:
            # Simulate LLM call
            await asyncio.sleep(0.1)  # 100ms latency

            response = f"Based on {len(search_results)} relevant documents, here's the answer to '{query}'..."

            # Track token usage and cost
            llm_context["usage"] = type("Usage", (), {"total_tokens": 150})()
            llm_context["cost"] = 0.0015  # ~$0.001 per 1K tokens
            llm_context["response"] = response

        return response


# Example 4: Demonstrate cost tracking
async def process_documentation_batch():
    """Process a batch of documentation with cost tracking."""

    documents = [
        "FastAPI is a modern web framework for building APIs with Python.",
        "OpenTelemetry provides observability for distributed systems.",
        "Vector databases enable semantic search capabilities.",
        "Qdrant is a high-performance vector similarity search engine.",
        "AI cost optimization is crucial for production systems.",
    ]

    # Generate embeddings for all documents
    embeddings = await generate_embeddings(documents)
    print(f"Generated {len(embeddings)} embeddings")

    # Perform searches
    for i, embedding in enumerate(embeddings[:3]):
        results = await search_similar_docs(embedding)
        print(f"Found {len(results)} similar documents for document {i}")

    # Run a RAG query
    answer = await rag_pipeline("What is FastAPI used for?")
    print(f"RAG Response: {answer}")

    # Get cost summary
    summary = cost_tracker.get_summary()
    print("\n=== AI Operations Cost Summary ===")
    print(f"Total Cost: ${summary['total_cost_usd']:.4f}")
    print(f"Total Operations: {summary['total_operations']}")

    print("\nBreakdown by Operation:")
    for op_key, stats in summary["operations_by_type"].items():
        provider, model, op_type = op_key.split(":")
        print(f"\n{provider} - {model} ({op_type}):")
        print(f"  Count: {stats['count']}")
        print(f"  Total Tokens: {stats['total_tokens']}")
        print(f"  Total Cost: ${stats['total_cost_usd']:.4f}")
        print(f"  Avg Latency: {stats['avg_latency_ms']:.2f}ms")


async def main():
    """Run the observability demo."""

    # Initialize tracing (optional - can connect to Jaeger/Tempo)
    setup_tracing(
        service_name="ai-doc-scraper-demo",
        # otlp_endpoint="localhost:4317"  # Uncomment to export traces
    )

    print("Starting AI Documentation Scraper Observability Demo\n")

    # Process some documents
    await process_documentation_batch()

    print("\n✅ Demo completed! Check the cost summary above.")
    print("💡 In production, these metrics would be available at:")
    print("   - GET /api/v1/metrics/ai-costs")
    print("   - GET /api/v1/metrics/performance")


if __name__ == "__main__":
    asyncio.run(main())
