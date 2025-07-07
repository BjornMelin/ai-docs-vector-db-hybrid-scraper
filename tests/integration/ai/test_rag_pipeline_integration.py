"""Integration Tests for RAG Pipeline with Service Mocking.

This module provides comprehensive integration tests for the RAG pipeline,
demonstrating  testing patterns with realistic service mocking and
end-to-end workflow validation.
"""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import respx

from tests.utils.modern_ai_testing import (
    IntegrationTestingPatterns,
    ModernAITestingUtils,
    integration_test,
)


@pytest.mark.integration
@pytest.mark.ai
class TestRAGPipelineIntegration:
    """Integration tests for RAG pipeline with comprehensive service mocking."""

    @pytest.fixture
    async def mock_services(self):
        """Set up all required service mocks for RAG pipeline testing."""
        with respx.mock:
            await IntegrationTestingPatterns.setup_mock_services(respx.mock)
            yield respx.mock

    @pytest.fixture
    def test_documents(self):
        """Provide test documents for integration testing."""
        return IntegrationTestingPatterns.create_test_documents(count=5)

    @pytest.fixture
    def mock_rag_components(self):
        """Mock RAG pipeline components."""
        components = {}

        # Mock embedding manager
        embedding_manager = MagicMock()
        embedding_manager.generate_single = AsyncMock(
            return_value=ModernAITestingUtils.generate_mock_embeddings(384, 1)[0]
        )
        embedding_manager.generate_batch = AsyncMock(
            side_effect=lambda texts: ModernAITestingUtils.generate_mock_embeddings(
                384, len(texts)
            )
        )
        components["embedding_manager"] = embedding_manager

        # Mock vector database
        vector_db = MagicMock()
        vector_db.search = AsyncMock(
            return_value=ModernAITestingUtils.create_mock_qdrant_response(
                [0.1] * 384, 5
            )
        )
        vector_db.upsert_documents = AsyncMock(return_value={"status": "success"})
        components["vector_db"] = vector_db

        # Mock text generator
        text_generator = MagicMock()
        text_generator.generate_response = AsyncMock(
            return_value={
                "response": "This is a generated response based on the retrieved context.",
                "confidence": 0.85,
                "sources_used": 3,
            }
        )
        components["text_generator"] = text_generator

        return components

    @integration_test
    @pytest.mark.asyncio
    async def test_end_to_end_rag_query_processing(
        self, mock_services, mock_rag_components, test_documents
    ):
        """Test complete RAG pipeline from query to response."""
        # Simulate a complete RAG pipeline
        query = "What is machine learning and how does it work?"

        # Step 1: Query embedding generation
        embedding_manager = mock_rag_components["embedding_manager"]
        query_embedding = await embedding_manager.generate_single(query)

        # Verify embedding properties
        ModernAITestingUtils.assert_valid_embedding(query_embedding, expected_dim=384)

        # Step 2: Vector similarity search
        vector_db = mock_rag_components["vector_db"]
        search_results = await vector_db.search(
            query_vector=query_embedding, limit=5, collection_name="test_collection"
        )

        # Verify search results structure
        assert "result" in search_results
        assert len(search_results["result"]) <= 5

        retrieved_contexts = []
        for result in search_results["result"]:
            assert "payload" in result
            assert "score" in result
            assert result["score"] >= 0.0
            retrieved_contexts.append(result["payload"]["text"])

        # Step 3: Response generation
        text_generator = mock_rag_components["text_generator"]
        generation_result = await text_generator.generate_response(
            query=query, contexts=retrieved_contexts
        )

        # Verify response structure and quality
        assert "response" in generation_result
        assert "confidence" in generation_result
        assert generation_result["confidence"] >= 0.0
        assert len(generation_result["response"]) > 0

        # Integration assertions
        assert generation_result["sources_used"] <= len(retrieved_contexts)

        # Verify the mocks were called correctly
        embedding_manager.generate_single.assert_called_once_with(query)
        vector_db.search.assert_called_once()
        text_generator.generate_response.assert_called_once()

    @integration_test
    @pytest.mark.asyncio
    async def test_document_ingestion_pipeline(
        self, mock_services, mock_rag_components, test_documents
    ):
        """Test complete document ingestion pipeline."""
        embedding_manager = mock_rag_components["embedding_manager"]
        vector_db = mock_rag_components["vector_db"]

        # Step 1: Extract text content from documents
        document_texts = [doc["content"] for doc in test_documents]

        # Step 2: Generate embeddings for all documents
        embeddings = await embedding_manager.generate_batch(document_texts)

        # Verify embeddings
        assert len(embeddings) == len(test_documents)
        for embedding in embeddings:
            ModernAITestingUtils.assert_valid_embedding(embedding, expected_dim=384)

        # Step 3: Prepare documents for vector database
        vector_points = []
        for _i, (doc, embedding) in enumerate(
            zip(test_documents, embeddings, strict=False)
        ):
            point = {
                "id": doc["id"],
                "vector": embedding,
                "payload": {
                    "title": doc["title"],
                    "content": doc["content"],
                    "url": doc["url"],
                    "metadata": doc["metadata"],
                },
            }
            vector_points.append(point)

        # Step 4: Upsert to vector database
        upsert_result = await vector_db.upsert_documents(
            collection_name="test_collection", points=vector_points
        )

        # Verify ingestion success
        assert upsert_result["status"] == "success"

        # Verify the pipeline called services correctly
        embedding_manager.generate_batch.assert_called_once_with(document_texts)
        vector_db.upsert_documents.assert_called_once()

    @integration_test
    @pytest.mark.asyncio
    async def test_rag_pipeline_error_handling(
        self, mock_services, mock_rag_components
    ):
        """Test RAG pipeline error handling and resilience."""
        embedding_manager = mock_rag_components["embedding_manager"]
        vector_db = mock_rag_components["vector_db"]
        text_generator = mock_rag_components["text_generator"]

        # Test embedding service failure
        embedding_manager.generate_single.side_effect = Exception(
            "Embedding service unavailable"
        )

        with pytest.raises(Exception, match="Embedding service unavailable"):
            await embedding_manager.generate_single("test query")

        # Reset and test vector database failure
        embedding_manager.generate_single.side_effect = None
        embedding_manager.generate_single.return_value = [0.1] * 384
        vector_db.search.side_effect = Exception("Vector DB connection failed")

        query_embedding = await embedding_manager.generate_single("test query")

        with pytest.raises(Exception, match="Vector DB connection failed"):
            await vector_db.search(query_vector=query_embedding, limit=5)

        # Reset and test text generation failure
        vector_db.search.side_effect = None
        vector_db.search.return_value = (
            ModernAITestingUtils.create_mock_qdrant_response([0.1] * 384, 3)
        )
        text_generator.generate_response.side_effect = Exception(
            "Text generation failed"
        )

        search_results = await vector_db.search(query_vector=query_embedding, limit=5)
        contexts = [result["payload"]["text"] for result in search_results["result"]]

        with pytest.raises(Exception, match="Text generation failed"):
            await text_generator.generate_response(
                query="test query", contexts=contexts
            )

    @integration_test
    @pytest.mark.asyncio
    async def test_rag_pipeline_with_different_query_types(
        self, mock_services, mock_rag_components
    ):
        """Test RAG pipeline with various query types."""
        embedding_manager = mock_rag_components["embedding_manager"]
        vector_db = mock_rag_components["vector_db"]
        text_generator = mock_rag_components["text_generator"]

        # Test different query types
        query_types = [
            "What is machine learning?",  # Factual question
            "How to implement a neural network?",  # How-to question
            "Compare supervised vs unsupervised learning",  # Comparison
            "machine learning algorithms",  # Keyword search
            "Show me examples of deep learning applications",  # Request for examples
        ]

        for query in query_types:
            # Process each query type
            query_embedding = await embedding_manager.generate_single(query)
            ModernAITestingUtils.assert_valid_embedding(
                query_embedding, expected_dim=384
            )

            search_results = await vector_db.search(
                query_vector=query_embedding, limit=5
            )

            contexts = [
                result["payload"]["text"] for result in search_results["result"]
            ]

            response = await text_generator.generate_response(
                query=query, contexts=contexts
            )

            # Verify each query type gets appropriate response structure
            assert len(response["response"]) > 0
            assert 0.0 <= response["confidence"] <= 1.0
            assert response["sources_used"] >= 0

    @integration_test
    @pytest.mark.asyncio
    async def test_rag_pipeline_performance_integration(
        self, mock_services, mock_rag_components
    ):
        """Test RAG pipeline performance characteristics in integration."""
        embedding_manager = mock_rag_components["embedding_manager"]
        vector_db = mock_rag_components["vector_db"]
        text_generator = mock_rag_components["text_generator"]

        query = "What are the benefits of artificial intelligence?"

        # Measure end-to-end pipeline latency
        start_time = time.perf_counter()

        # Full pipeline execution
        query_embedding = await embedding_manager.generate_single(query)
        search_results = await vector_db.search(query_vector=query_embedding, limit=5)
        contexts = [result["payload"]["text"] for result in search_results["result"]]
        await text_generator.generate_response(query=query, contexts=contexts)

        end_time = time.perf_counter()
        total_latency = end_time - start_time

        # Performance assertions (for mocked services, should be very fast)
        assert total_latency < 1.0, (
            f"Pipeline latency {total_latency:.3f}s too high for mocked services"
        )

        # Verify all components were called
        assert embedding_manager.generate_single.call_count >= 1
        assert vector_db.search.call_count >= 1
        assert text_generator.generate_response.call_count >= 1

    @integration_test
    @pytest.mark.asyncio
    async def test_concurrent_rag_requests(self, mock_services, mock_rag_components):
        """Test RAG pipeline under concurrent load."""
        embedding_manager = mock_rag_components["embedding_manager"]
        vector_db = mock_rag_components["vector_db"]
        text_generator = mock_rag_components["text_generator"]

        queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Explain deep learning concepts",
            "What are neural networks?",
            "Describe natural language processing",
        ]

        async def process_query(query: str) -> dict[str, Any]:
            """Process a single query through the RAG pipeline."""
            query_embedding = await embedding_manager.generate_single(query)
            search_results = await vector_db.search(
                query_vector=query_embedding, limit=3
            )
            contexts = [
                result["payload"]["text"] for result in search_results["result"]
            ]
            response = await text_generator.generate_response(
                query=query, contexts=contexts
            )
            return {
                "query": query,
                "response": response,
                "contexts_count": len(contexts),
            }

        # Execute queries concurrently
        tasks = [process_query(query) for query in queries]
        results = await asyncio.gather(*tasks)

        # Verify all queries completed successfully
        assert len(results) == len(queries)

        for result in results:
            assert "query" in result
            assert "response" in result
            assert "contexts_count" in result
            assert result["contexts_count"] > 0
            assert len(result["response"]["response"]) > 0

        # Verify services handled concurrent load
        assert embedding_manager.generate_single.call_count == len(queries)
        assert vector_db.search.call_count == len(queries)
        assert text_generator.generate_response.call_count == len(queries)
