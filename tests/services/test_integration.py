"""Integration tests for service layer with mocked API responses."""

from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.config.enums import CrawlProvider
from src.config.enums import EmbeddingProvider
from src.config.models import UnifiedConfig
from src.services.crawling.manager import CrawlManager
from src.services.embeddings.manager import EmbeddingManager
from src.services.embeddings.manager import QualityTier
from src.services.qdrant_service import QdrantService


@pytest.fixture
def integration_config():
    """Integration test configuration.

    Using environment variable syntax (double underscore) to simulate
    how the config would be loaded in production from environment variables.
    Alternative syntax would be:
        UnifiedConfig(
            openai=OpenAIConfig(api_key="sk-test"),
            firecrawl=FirecrawlConfig(api_key="fc-test"),
            qdrant=QdrantConfig(url="http://localhost:6333")
        )
    """
    return UnifiedConfig(
        openai__api_key="sk-test-integration-key",
        firecrawl__api_key="fc-test-integration-key",
        qdrant__url="http://localhost:6333",
        embedding_provider=EmbeddingProvider.OPENAI,
        crawl_provider=CrawlProvider.FIRECRAWL,
    )


class TestServiceIntegration:
    """Integration tests for complete service workflows."""

    @pytest.mark.asyncio
    async def test_document_processing_workflow(self, integration_config):
        """Test complete document processing workflow."""
        # Mock responses
        crawl_response = {
            "success": True,
            "content": "# Test Document\n\nThis is test content for integration.",
            "metadata": {"title": "Test Document", "url": "https://example.com"},
        }

        embedding_response = [[0.1, 0.2, 0.3] * 512]  # 1536 dimensions

        # Test workflow
        crawl_manager = CrawlManager(integration_config)
        await crawl_manager.initialize()
        try:
            with patch.object(
                crawl_manager.providers.get("crawl4ai", Mock()),
                "scrape_url",
                return_value=crawl_response,
            ):
                # Crawl document
                crawl_result = await crawl_manager.scrape_url("https://example.com")
                assert crawl_result["success"] is True
                content = crawl_result["content"]

                # Generate embeddings
                embedding_manager = EmbeddingManager(integration_config)
                await embedding_manager.initialize()
                try:
                    with patch.object(
                        embedding_manager.providers.get("openai", Mock()),
                        "generate_embeddings",
                        return_value=embedding_response,
                    ):
                        embeddings = await embedding_manager.generate_embeddings(
                            [content]
                        )
                        assert len(embeddings) == 1
                        assert len(embeddings[0]) == 1536

                        # Store in Qdrant
                        qdrant_service = QdrantService(
                            url=integration_config.qdrant.url,
                            timeout=integration_config.qdrant.timeout,
                        )
                        await qdrant_service.initialize()
                        try:
                            with patch.object(
                                qdrant_service._client,
                                "upsert",
                                return_value=Mock(status="completed"),
                            ):
                                # Create collection
                                await qdrant_service.create_collection(
                                    collection_name="test_collection",
                                    vector_size=1536,
                                )

                                # Store document
                                await qdrant_service.upsert_documents(
                                    collection_name="test_collection",
                                    documents=[
                                        {
                                            "id": "test_doc_1",
                                            "content": content,
                                            "embedding": embeddings[0],
                                            "metadata": crawl_result["metadata"],
                                        }
                                    ],
                                )

                                # Search document
                                with patch.object(
                                    qdrant_service._client,
                                    "search",
                                    return_value=[
                                        Mock(
                                            id="test_doc_1",
                                            score=0.95,
                                            payload={
                                                "content": content,
                                                "metadata": crawl_result["metadata"],
                                            },
                                        )
                                    ],
                                ):
                                    search_results = await qdrant_service.search(
                                        collection_name="test_collection",
                                        query_vector=embeddings[0],
                                        limit=5,
                                    )
                                    assert len(search_results) == 1
                                    assert search_results[0].score > 0.9
                        finally:
                            await qdrant_service.cleanup()
                finally:
                    await embedding_manager.cleanup()
        finally:
            await crawl_manager.cleanup()

    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self, integration_config):
        """Test batch document processing workflow."""
        urls = [
            "https://example.com/doc1",
            "https://example.com/doc2",
            "https://example.com/doc3",
        ]

        # Mock batch crawl response
        crawl_responses = {
            "success": True,
            "data": [
                {
                    "url": urls[0],
                    "content": "Document 1 content",
                    "metadata": {"title": "Doc 1"},
                },
                {
                    "url": urls[1],
                    "content": "Document 2 content",
                    "metadata": {"title": "Doc 2"},
                },
                {
                    "url": urls[2],
                    "content": "Document 3 content",
                    "metadata": {"title": "Doc 3"},
                },
            ],
        }

        # Mock batch embeddings
        batch_embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]

        crawl_manager = CrawlManager(integration_config)
        await crawl_manager.initialize()
        try:
            with patch.object(
                crawl_manager.providers.get("firecrawl", Mock()),
                "scrape_multiple_urls",
                return_value=crawl_responses,
            ):
                # Batch crawl
                crawl_results = await crawl_manager.scrape_multiple_urls(urls)
                assert crawl_results["success"] is True
                assert len(crawl_results["data"]) == 3

                # Extract contents
                contents = [doc["content"] for doc in crawl_results["data"]]

                # Generate embeddings
                embedding_manager = EmbeddingManager(integration_config)
                await embedding_manager.initialize()
                try:
                    with patch.object(
                        embedding_manager.providers.get("openai", Mock()),
                        "generate_embeddings",
                        return_value=batch_embeddings,
                    ):
                        embeddings = await embedding_manager.generate_embeddings(
                            contents
                        )
                        assert len(embeddings) == 3

                        # Store in Qdrant
                        qdrant_service = QdrantService(
                            url=integration_config.qdrant.url,
                            timeout=integration_config.qdrant.timeout,
                        )
                        await qdrant_service.initialize()
                        try:
                            with patch.object(
                                qdrant_service._client,
                                "upsert",
                                return_value=Mock(status="completed"),
                            ):
                                # Create collection
                                await qdrant_service.create_collection(
                                    collection_name="batch_collection",
                                    vector_size=1536,
                                )

                                # Prepare documents
                                documents = [
                                    {
                                        "id": f"doc_{i}",
                                        "content": doc["content"],
                                        "embedding": embedding,
                                        "metadata": doc["metadata"],
                                    }
                                    for i, (doc, embedding) in enumerate(
                                        zip(
                                            crawl_results["data"],
                                            embeddings,
                                            strict=False,
                                        )
                                    )
                                ]

                                # Store batch
                                await qdrant_service.upsert_documents(
                                    collection_name="batch_collection",
                                    documents=documents,
                                )
                        finally:
                            await qdrant_service.cleanup()
                finally:
                    await embedding_manager.cleanup()
        finally:
            await crawl_manager.cleanup()

    @pytest.mark.asyncio
    async def test_smart_provider_selection(self, integration_config):
        """Test smart provider selection based on content."""
        # Different types of content
        test_contents = [
            "Short text",  # Should use FastEmbed (low quality tier)
            "def calculate_sum(a, b):\n    return a + b",  # Code - should use OpenAI
            "A" * 10000,  # Very long text - should batch properly
        ]

        embedding_manager = EmbeddingManager(integration_config)
        await embedding_manager.initialize()
        try:
            # Mock both providers
            with (
                patch.object(
                    embedding_manager.providers.get("openai", Mock()),
                    "generate_embeddings",
                    return_value=[[0.1] * 1536],
                ) as mock_openai,
                patch.object(
                    embedding_manager.providers.get("fastembed", Mock()),
                    "generate_embeddings",
                    return_value=[[0.2] * 384],
                ) as mock_fastembed,
            ):
                # Test short text - should use FastEmbed
                embeddings = await embedding_manager.generate_embeddings(
                    [test_contents[0]], quality_tier=QualityTier.LOW
                )
                assert len(embeddings) == 1
                mock_fastembed.assert_called()

                # Test code - should use OpenAI
                embeddings = await embedding_manager.generate_embeddings(
                    [test_contents[1]], quality_tier=QualityTier.HIGH
                )
                assert len(embeddings) == 1
                mock_openai.assert_called()

        finally:
            await embedding_manager.cleanup()

    @pytest.mark.asyncio
    async def test_error_handling_and_fallback(self, integration_config):
        """Test error handling and provider fallback."""
        crawl_manager = CrawlManager(integration_config)
        await crawl_manager.initialize()
        try:
            # Mock Firecrawl failure
            with (
                patch.object(
                    crawl_manager.providers.get("firecrawl", Mock()),
                    "scrape_url",
                    return_value={"success": False, "error": "API error"},
                ),
                patch.object(
                    crawl_manager.providers.get("crawl4ai", Mock()),
                    "scrape_url",
                    return_value={
                        "success": True,
                        "content": "Fallback content",
                        "metadata": {"title": "Fallback"},
                    },
                ),
            ):
                # Should fallback to Crawl4AI
                result = await crawl_manager.scrape_url("https://example.com")
                assert result["success"] is True
                assert result["content"] == "Fallback content"

        finally:
            await crawl_manager.cleanup()

    @pytest.mark.asyncio
    async def test_reranking_workflow(self, integration_config):
        """Test document reranking workflow."""
        # Mock search results
        search_results = [
            {"content": "Result 1", "score": 0.7},
            {"content": "Result 2", "score": 0.8},
            {"content": "Result 3", "score": 0.6},
            {"content": "Result 4", "score": 0.9},
            {"content": "Result 5", "score": 0.5},
        ]

        embedding_manager = EmbeddingManager(integration_config)
        await embedding_manager.initialize()
        try:
            # Mock reranker
            if embedding_manager._reranker is not None:
                with patch.object(
                    embedding_manager._reranker,
                    "compute_score",
                    return_value=[0.95, 0.85, 0.75, 0.65, 0.55],
                ):
                    query = "test query"
                    documents = [r["content"] for r in search_results]

                    reranked = await embedding_manager.rerank_documents(
                        query, documents, top_k=3
                    )

                    assert len(reranked) == 3
                    assert reranked[0]["score"] == 0.95  # Highest reranked score

        finally:
            await embedding_manager.cleanup()

    @pytest.mark.asyncio
    async def test_hybrid_search_workflow(self, integration_config):
        """Test hybrid search (dense + sparse) workflow."""
        qdrant_service = QdrantService(
            url=integration_config.qdrant.url,
            timeout=integration_config.qdrant.timeout,
        )
        await qdrant_service.initialize()
        try:
            # Mock collection info and hybrid search
            with (
                patch.object(
                    qdrant_service._client,
                    "get_collection",
                    return_value=Mock(
                        config=Mock(
                            params=Mock(
                                sparse_vectors={
                                    "text-sparse": Mock(index=Mock(on_disk=False))
                                }
                            )
                        )
                    ),
                ),
                patch.object(
                    qdrant_service._client,
                    "search_batch",
                    return_value=[
                        [
                            Mock(id="doc1", score=0.9, payload={"content": "Doc 1"}),
                            Mock(id="doc2", score=0.8, payload={"content": "Doc 2"}),
                        ]
                    ],
                ),
            ):
                results = await qdrant_service.hybrid_search(
                    collection_name="test_collection",
                    query_text="test query",
                    dense_vector=[0.1] * 1536,
                    sparse_vector={
                        "indices": [1, 5, 10],
                        "values": [0.5, 0.3, 0.2],
                    },
                    limit=5,
                )

                assert len(results) > 0
                assert results[0].score > 0.8

        finally:
            await qdrant_service.cleanup()

    @pytest.mark.asyncio
    async def test_complete_pipeline_with_no_api_keys(self):
        """Test pipeline with no API keys (using local providers only)."""
        config = UnifiedConfig()  # No API keys

        # Should use Crawl4AI and FastEmbed
        crawl_manager = CrawlManager(config)
        await crawl_manager.initialize()
        try:
            assert "crawl4ai" in crawl_manager.providers
            assert "firecrawl" not in crawl_manager.providers

            embedding_manager = EmbeddingManager(config)
            await embedding_manager.initialize()
            try:
                assert "fastembed" in embedding_manager.providers
                assert "openai" not in embedding_manager.providers

                # Test with local providers
                with (
                    patch.object(
                        crawl_manager.providers["crawl4ai"],
                        "scrape_url",
                        return_value={
                            "success": True,
                            "content": "Local content",
                            "metadata": {"title": "Local"},
                        },
                    ),
                    patch.object(
                        embedding_manager.providers["fastembed"],
                        "generate_embeddings",
                        return_value=[[0.1] * 384],
                    ),
                ):
                    # Process document
                    crawl_result = await crawl_manager.scrape_url("https://example.com")
                    assert crawl_result["success"] is True

                    embeddings = await embedding_manager.generate_embeddings(
                        [crawl_result["content"]]
                    )
                    assert len(embeddings) == 1
                    assert len(embeddings[0]) == 384  # FastEmbed dimension

            finally:
                await embedding_manager.cleanup()
        finally:
            await crawl_manager.cleanup()
