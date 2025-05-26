# Testing & Quality Enhancements

> **Status:** Test Plan Ready  
> **Priority:** High  
> **Estimated Effort:** 2-3 weeks  
> **Documentation Created:** 2025-05-22

## Overview

Comprehensive testing strategy and quality enhancement plan to achieve 95%+ test coverage, async testing patterns, performance benchmarks, and production-ready quality assurance. This builds on existing test foundation to ensure reliability at scale.

## Current Testing Status

### Existing Test Coverage

- **Unit Tests**: Basic functionality covered
- **Integration Tests**: Limited MCP integration tests
- **Performance Tests**: Basic benchmarks
- **Coverage**: ~60-70% estimated

### Target Quality Metrics

- **Test Coverage**: 95%+ (measured with pytest-cov)
- **Async Test Coverage**: 100% of async operations
- **Performance Regression**: <5% degradation tolerance
- **Error Recovery**: 100% error condition coverage
- **Integration Coverage**: All external APIs and services

## Testing Architecture

### Test Structure

```plaintext
tests/
├── unit/               # Fast, isolated unit tests
│   ├── test_models/    # Pydantic model validation
│   ├── test_services/  # Service layer testing
│   ├── test_providers/ # Provider implementations
│   └── test_utils/     # Utility functions
├── integration/        # Service integration tests
│   ├── test_qdrant/    # Vector database operations
│   ├── test_embedding/ # Embedding generation
│   ├── test_crawling/  # Web crawling
│   └── test_mcp/       # MCP server functionality
├── performance/        # Performance and load tests
│   ├── benchmarks/     # Performance benchmarks
│   ├── stress/         # Stress testing
│   └── regression/     # Performance regression tests
├── e2e/               # End-to-end workflows
│   ├── test_complete_pipeline.py
│   ├── test_search_workflows.py
│   └── test_mcp_workflows.py
└── fixtures/          # Shared test data and fixtures
    ├── sample_data/
    ├── mock_responses/
    └── test_configs/
```

## Implementation Plan

### Phase 1: Async Testing Foundation

#### Pytest-Asyncio Configuration

```python
# tests/conftest.py
import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import AsyncGenerator, Generator, Dict, Any

# Configure async testing
pytest_asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy() if os.name == 'nt' else None)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client for testing."""
    async with AsyncClient() as client:
        yield client

@pytest.fixture
def mock_config() -> UnifiedConfig:
    """Test configuration."""
    return UnifiedConfig(
        qdrant=QdrantConfig(
            url="http://test-qdrant:6333",
            api_key="test-key"
        ),
        embedding=EmbeddingConfig(
            provider="fastembed",
            model="test-model",
            dimensions=384
        ),
        max_concurrent_requests=5,
        chunk_size=500,
        enable_cache=False  # Disable cache for tests
    )

@pytest.fixture
async def mock_qdrant_client():
    """Mock Qdrant client."""
    client = AsyncMock()
    
    # Mock common methods
    client.create_collection.return_value = True
    client.upsert.return_value = True
    client.search.return_value = [
        MockPoint(id=1, score=0.9, payload={"text": "test result"})
    ]
    client.query_points.return_value = MockQueryResponse([
        MockPoint(id=1, score=0.9, payload={"text": "hybrid result"})
    ])
    
    return client

@pytest.fixture
async def mock_embedding_service():
    """Mock embedding service."""
    service = AsyncMock()
    service.generate_embeddings.return_value = [[0.1] * 384]
    service.dimensions = 384
    return service

@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """Sample documents for testing."""
    return [
        {
            "url": "https://example.com/doc1",
            "title": "Test Document 1",
            "content": "This is a test document for embeddings.",
            "metadata": {"source": "test", "type": "documentation"}
        },
        {
            "url": "https://example.com/doc2", 
            "title": "Test Document 2",
            "content": "Another test document with different content.",
            "metadata": {"source": "test", "type": "guide"}
        }
    ]
```

#### Mock Classes for Testing

```python
# tests/mocks/qdrant_mocks.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class MockPoint:
    """Mock Qdrant point."""
    id: int
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None

@dataclass
class MockQueryResponse:
    """Mock Qdrant query response."""
    points: List[MockPoint]

@dataclass
class MockCollectionInfo:
    """Mock collection info."""
    status: str = "green"
    vectors_count: int = 1000
    indexed_vectors_count: int = 1000

class MockQdrantClient:
    """Mock Qdrant client for integration tests."""
    
    def __init__(self):
        self.collections: Dict[str, MockCollectionInfo] = {}
        self.points: Dict[str, List[MockPoint]] = {}
    
    async def create_collection(self, collection_name: str, **kwargs) -> bool:
        """Mock collection creation."""
        self.collections[collection_name] = MockCollectionInfo()
        self.points[collection_name] = []
        return True
    
    async def upsert(self, collection_name: str, points: List, **kwargs) -> bool:
        """Mock point upsert."""
        if collection_name not in self.points:
            self.points[collection_name] = []
        
        for point in points:
            self.points[collection_name].append(
                MockPoint(
                    id=point.id,
                    score=0.0,
                    payload=point.payload,
                    vector=point.vector
                )
            )
        return True
    
    async def search(
        self, 
        collection_name: str, 
        query_vector: List[float],
        limit: int = 10,
        **kwargs
    ) -> List[MockPoint]:
        """Mock search."""
        if collection_name not in self.points:
            return []
        
        # Simple mock scoring based on vector similarity
        results = []
        for point in self.points[collection_name][:limit]:
            # Mock score calculation
            score = 0.9 - (len(results) * 0.1)
            results.append(MockPoint(
                id=point.id,
                score=max(0.1, score),
                payload=point.payload
            ))
        
        return results
    
    async def query_points(
        self, 
        collection_name: str, 
        query,
        limit: int = 10,
        **kwargs
    ) -> MockQueryResponse:
        """Mock hybrid query."""
        points = await self.search(collection_name, [], limit)
        return MockQueryResponse(points=points)
```

### Phase 2: Comprehensive Unit Tests

#### Service Layer Testing

```python
# tests/unit/test_services/test_embedding_service.py
import pytest
from unittest.mock import AsyncMock, patch
from src.services.embedding_service import UnifiedEmbeddingService
from src.providers.openai_provider import OpenAIEmbeddingProvider
from src.providers.fastembed_provider import FastEmbedProvider

class TestUnifiedEmbeddingService:
    """Test embedding service with different providers."""
    
    @pytest.mark.asyncio
    async def test_openai_provider_selection(self, mock_config, client_manager):
        """Test OpenAI provider selection."""
        mock_config.embedding.provider = "openai"
        mock_config.embedding.api_key = "test-key"
        
        with patch.object(client_manager, 'get_openai_client') as mock_client:
            mock_client.return_value = AsyncMock()
            
            service = UnifiedEmbeddingService(client_manager, mock_config.embedding)
            provider = await service._get_provider()
            
            assert isinstance(provider, OpenAIEmbeddingProvider)
            mock_client.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fastembed_fallback(self, mock_config, client_manager):
        """Test FastEmbed fallback when no API key."""
        mock_config.embedding.provider = "auto"
        mock_config.embedding.api_key = None
        
        with patch.object(client_manager, 'get_openai_client') as mock_client:
            mock_client.return_value = None
            
            with patch('src.providers.fastembed_provider.TextEmbedding') as mock_fe:
                mock_fe.return_value.dim = 384
                
                service = UnifiedEmbeddingService(client_manager, mock_config.embedding)
                provider = await service._get_provider()
                
                assert isinstance(provider, FastEmbedProvider)
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self, mock_embedding_service):
        """Test embedding generation."""
        texts = ["test text 1", "test text 2"]
        expected_embeddings = [[0.1] * 384, [0.2] * 384]
        
        mock_embedding_service.generate_embeddings.return_value = expected_embeddings
        
        result = await mock_embedding_service.generate_embeddings(texts)
        
        assert result == expected_embeddings
        mock_embedding_service.generate_embeddings.assert_called_once_with(texts)
    
    @pytest.mark.asyncio
    async def test_empty_input_handling(self, mock_embedding_service):
        """Test handling of empty input."""
        mock_embedding_service.generate_embeddings.return_value = []
        
        result = await mock_embedding_service.generate_embeddings([])
        
        assert result == []

# tests/unit/test_services/test_vector_service.py
class TestOptimizedVectorService:
    """Test optimized vector operations."""
    
    @pytest.mark.asyncio
    async def test_bulk_upsert_batching(self, mock_qdrant_client):
        """Test bulk upsert with batching."""
        from src.services.optimized_vector_service import OptimizedVectorService
        
        # Mock connection pool
        pool = AsyncMock()
        pool.get_connection.return_value = mock_qdrant_client
        pool.return_connection = AsyncMock()
        
        service = OptimizedVectorService(pool)
        
        # Large dataset to test batching
        vectors = [
            {"id": i, "vector": [0.1] * 384, "payload": {"text": f"doc {i}"}}
            for i in range(2500)  # More than batch size
        ]
        
        result = await service.bulk_upsert_optimized(
            "test_collection",
            vectors,
            batch_size=1000
        )
        
        assert result is True
        # Should be called 3 times (1000, 1000, 500)
        assert mock_qdrant_client.upsert.call_count == 3
    
    @pytest.mark.asyncio
    async def test_optimized_search_dense_only(self, mock_qdrant_client):
        """Test optimized dense-only search."""
        from src.services.optimized_vector_service import OptimizedVectorService
        
        pool = AsyncMock()
        pool.get_connection.return_value = mock_qdrant_client
        pool.return_connection = AsyncMock()
        
        service = OptimizedVectorService(pool)
        
        query_vector = [0.1] * 384
        
        result = await service.optimized_search(
            "test_collection",
            query_vector,
            limit=5
        )
        
        assert len(result) > 0
        mock_qdrant_client.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_hybrid_search_with_sparse(self, mock_qdrant_client):
        """Test hybrid search with sparse vectors."""
        from src.services.optimized_vector_service import OptimizedVectorService
        
        pool = AsyncMock()
        pool.get_connection.return_value = mock_qdrant_client
        pool.return_connection = AsyncMock()
        
        service = OptimizedVectorService(pool)
        
        query_vector = [0.1] * 384
        sparse_vector = {0: 0.5, 1: 0.3, 10: 0.8}
        
        result = await service.optimized_search(
            "test_collection",
            query_vector,
            sparse_vector=sparse_vector,
            limit=5
        )
        
        assert len(result) > 0
        mock_qdrant_client.query_points.assert_called_once()
```

#### Provider Testing

```python
# tests/unit/test_providers/test_openai_provider.py
import pytest
from unittest.mock import AsyncMock, patch
from src.providers.openai_provider import OpenAIEmbeddingProvider
from src.exceptions.embedding_exceptions import EmbeddingError

class TestOpenAIEmbeddingProvider:
    """Test OpenAI embedding provider."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        client = AsyncMock()
        
        # Mock embeddings response
        mock_response = AsyncMock()
        mock_response.data = [
            AsyncMock(embedding=[0.1] * 1536),
            AsyncMock(embedding=[0.2] * 1536)
        ]
        client.embeddings.create.return_value = mock_response
        
        return client
    
    @pytest.mark.asyncio
    async def test_single_batch_embedding(self, mock_openai_client, mock_config):
        """Test single batch embedding generation."""
        provider = OpenAIEmbeddingProvider(mock_openai_client, mock_config.embedding)
        
        texts = ["text 1", "text 2"]
        result = await provider.generate_embeddings(texts)
        
        assert len(result) == 2
        assert len(result[0]) == 1536
        mock_openai_client.embeddings.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multi_batch_embedding(self, mock_openai_client, mock_config):
        """Test multi-batch embedding generation."""
        mock_config.embedding.batch_size = 2
        provider = OpenAIEmbeddingProvider(mock_openai_client, mock_config.embedding)
        
        texts = ["text 1", "text 2", "text 3", "text 4", "text 5"]
        result = await provider.generate_embeddings(texts)
        
        assert len(result) == 5
        # Should be called 3 times (2, 2, 1)
        assert mock_openai_client.embeddings.create.call_count == 3
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, mock_openai_client, mock_config):
        """Test API error handling."""
        mock_openai_client.embeddings.create.side_effect = Exception("API Error")
        provider = OpenAIEmbeddingProvider(mock_openai_client, mock_config.embedding)
        
        with pytest.raises(EmbeddingError) as exc_info:
            await provider.generate_embeddings(["test"])
        
        assert "OpenAI embedding generation failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_empty_input(self, mock_openai_client, mock_config):
        """Test empty input handling."""
        provider = OpenAIEmbeddingProvider(mock_openai_client, mock_config.embedding)
        
        result = await provider.generate_embeddings([])
        
        assert result == []
        mock_openai_client.embeddings.create.assert_not_called()
```

### Phase 3: Integration Tests

#### Qdrant Integration Testing

```python
# tests/integration/test_qdrant/test_vector_operations.py
import pytest
from testcontainers.compose import DockerCompose
from qdrant_client import QdrantClient
from src.services.optimized_vector_service import OptimizedVectorService

@pytest.mark.integration
class TestQdrantIntegration:
    """Integration tests with real Qdrant instance."""
    
    @pytest.fixture(scope="class")
    def qdrant_container(self):
        """Start Qdrant container for testing."""
        with DockerCompose("tests/fixtures/docker", compose_file_name="qdrant.yml") as compose:
            # Wait for Qdrant to be ready
            import time
            time.sleep(10)
            yield compose
    
    @pytest.fixture
    def qdrant_client(self, qdrant_container):
        """Real Qdrant client."""
        client = QdrantClient(host="localhost", port=6333, prefer_grpc=False)
        yield client
        
        # Cleanup collections
        try:
            collections = client.get_collections()
            for collection in collections.collections:
                if collection.name.startswith("test_"):
                    client.delete_collection(collection.name)
        except:
            pass
    
    @pytest.mark.asyncio
    async def test_real_collection_lifecycle(self, qdrant_client):
        """Test complete collection lifecycle."""
        collection_name = "test_lifecycle"
        
        # Create collection
        await qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": 384, "distance": "Cosine"}
        )
        
        # Verify collection exists
        collections = await qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        assert collection_name in collection_names
        
        # Insert points
        points = [
            {
                "id": i,
                "vector": [0.1 * i] * 384,
                "payload": {"text": f"Document {i}", "category": "test"}
            }
            for i in range(100)
        ]
        
        await qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        # Search
        results = await qdrant_client.search(
            collection_name=collection_name,
            query_vector=[0.1] * 384,
            limit=10
        )
        
        assert len(results) == 10
        assert all(result.score > 0 for result in results)
        
        # Filter search
        filtered_results = await qdrant_client.search(
            collection_name=collection_name,
            query_vector=[0.1] * 384,
            query_filter={"must": [{"key": "category", "match": {"value": "test"}}]},
            limit=5
        )
        
        assert len(filtered_results) == 5
    
    @pytest.mark.asyncio
    async def test_quantization_performance(self, qdrant_client):
        """Test quantization impact on performance."""
        import time
        
        # Test with regular collection
        regular_collection = "test_regular"
        await qdrant_client.create_collection(
            collection_name=regular_collection,
            vectors_config={"size": 384, "distance": "Cosine"}
        )
        
        # Test with quantized collection
        quantized_collection = "test_quantized"
        await qdrant_client.create_collection(
            collection_name=quantized_collection,
            vectors_config={"size": 384, "distance": "Cosine"},
            quantization_config={
                "scalar": {
                    "type": "int8",
                    "quantile": 0.99,
                    "always_ram": True
                }
            }
        )
        
        # Insert same data to both
        points = [
            {
                "id": i,
                "vector": [0.1 * (i % 100)] * 384,
                "payload": {"text": f"Doc {i}"}
            }
            for i in range(1000)
        ]
        
        await qdrant_client.upsert(regular_collection, points)
        await qdrant_client.upsert(quantized_collection, points)
        
        # Benchmark search performance
        query_vector = [0.1] * 384
        
        start_time = time.time()
        regular_results = await qdrant_client.search(
            regular_collection, query_vector, limit=50
        )
        regular_time = time.time() - start_time
        
        start_time = time.time()
        quantized_results = await qdrant_client.search(
            quantized_collection, query_vector, limit=50
        )
        quantized_time = time.time() - start_time
        
        # Quantized should be faster (or at least not much slower)
        assert quantized_time <= regular_time * 1.5  # Allow 50% tolerance
        assert len(regular_results) == len(quantized_results)
```

#### End-to-End Pipeline Testing

```python
# tests/e2e/test_complete_pipeline.py
import pytest
from unittest.mock import patch, AsyncMock
from src.services.document_service import DocumentService
from src.services.search_service import SearchService

@pytest.mark.e2e
class TestCompletePipeline:
    """End-to-end pipeline testing."""
    
    @pytest.mark.asyncio
    async def test_document_indexing_pipeline(
        self, 
        mock_config, 
        sample_documents,
        mock_qdrant_client,
        mock_embedding_service
    ):
        """Test complete document indexing pipeline."""
        
        # Mock services
        with patch('src.infrastructure.client_manager.ClientManager') as mock_manager:
            mock_manager.return_value.get_qdrant_client.return_value = mock_qdrant_client
            
            # Initialize document service
            doc_service = DocumentService(mock_manager.return_value, mock_config)
            doc_service.embedding_service = mock_embedding_service
            
            # Process documents
            results = await doc_service.process_documents(
                sample_documents,
                collection_name="test_collection"
            )
            
            # Verify results
            assert results["success"] is True
            assert results["processed_count"] == len(sample_documents)
            assert results["failed_count"] == 0
            
            # Verify embeddings were generated
            mock_embedding_service.generate_embeddings.assert_called()
            
            # Verify vectors were stored
            mock_qdrant_client.upsert.assert_called()
    
    @pytest.mark.asyncio
    async def test_search_pipeline_with_reranking(
        self,
        mock_config,
        mock_qdrant_client,
        mock_embedding_service
    ):
        """Test search pipeline with reranking."""
        
        with patch('src.infrastructure.client_manager.ClientManager') as mock_manager:
            mock_manager.return_value.get_qdrant_client.return_value = mock_qdrant_client
            
            # Mock reranker
            with patch('src.services.optimized_reranking.OptimizedReranker') as mock_reranker:
                mock_reranker.return_value.adaptive_reranking.return_value = [
                    {
                        "document": "Reranked result",
                        "rerank_score": 0.95,
                        "combined_score": 0.93
                    }
                ]
                
                # Initialize search service
                search_service = SearchService(mock_manager.return_value, mock_config)
                search_service.embedding_service = mock_embedding_service
                
                # Perform search
                results = await search_service.search(
                    query="test query",
                    collection_name="test_collection",
                    limit=10,
                    enable_reranking=True
                )
                
                # Verify search pipeline
                assert len(results) > 0
                mock_embedding_service.generate_embeddings.assert_called_with(["test query"])
                mock_qdrant_client.search.assert_called()
                mock_reranker.return_value.adaptive_reranking.assert_called()
    
    @pytest.mark.asyncio
    async def test_error_recovery_pipeline(
        self,
        mock_config,
        sample_documents
    ):
        """Test error recovery in pipeline."""
        
        # Mock failing embedding service
        failing_embedding_service = AsyncMock()
        failing_embedding_service.generate_embeddings.side_effect = [
            Exception("Temporary failure"),  # First call fails
            [[0.1] * 384] * 5  # Second call succeeds
        ]
        
        with patch('src.infrastructure.client_manager.ClientManager') as mock_manager:
            with patch('src.utils.decorators.retry_async') as mock_retry:
                # Configure retry to actually retry
                mock_retry.return_value = lambda f: f
                
                doc_service = DocumentService(mock_manager.return_value, mock_config)
                doc_service.embedding_service = failing_embedding_service
                
                # Should recover from failure
                results = await doc_service.process_documents_with_retry(
                    sample_documents[:1],  # Single document for simplicity
                    collection_name="test_collection"
                )
                
                # Verify retry occurred
                assert failing_embedding_service.generate_embeddings.call_count == 2
```

### Phase 4: Performance Testing

#### Performance Benchmarks

```python
# tests/performance/benchmarks/test_embedding_performance.py
import pytest
import time
import statistics
from typing import List
from src.providers.openai_provider import OpenAIEmbeddingProvider
from src.providers.fastembed_provider import FastEmbedProvider

@pytest.mark.performance
class TestEmbeddingPerformance:
    """Embedding performance benchmarks."""
    
    @pytest.fixture
    def benchmark_texts(self) -> List[str]:
        """Generate benchmark texts of varying lengths."""
        return [
            "Short text",
            "Medium length text with more content to test embedding generation performance",
            "Long text with substantial content that would be typical of documentation chunks. " * 10,
            "Very long text that exceeds typical chunk sizes and tests the limits of embedding generation. " * 25
        ]
    
    @pytest.mark.asyncio
    async def test_fastembed_performance(self, benchmark_texts):
        """Benchmark FastEmbed performance."""
        provider = FastEmbedProvider(EmbeddingConfig(model="BAAI/bge-small-en-v1.5"))
        
        # Warmup
        await provider.generate_embeddings(benchmark_texts[:1])
        
        # Benchmark
        times = []
        for _ in range(5):  # Run 5 times for stability
            start_time = time.time()
            embeddings = await provider.generate_embeddings(benchmark_texts)
            duration = time.time() - start_time
            times.append(duration)
        
        avg_time = statistics.mean(times)
        per_text_time = avg_time / len(benchmark_texts) * 1000  # ms
        
        # Performance targets
        assert per_text_time < 50, f"FastEmbed too slow: {per_text_time:.2f}ms per text"
        assert len(embeddings) == len(benchmark_texts)
        assert all(len(emb) == 384 for emb in embeddings)
        
        print(f"FastEmbed Performance: {per_text_time:.2f}ms per text")
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key required")
    async def test_openai_performance(self, benchmark_texts):
        """Benchmark OpenAI performance."""
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,
            batch_size=len(benchmark_texts)
        )
        
        provider = OpenAIEmbeddingProvider(client, config)
        
        # Benchmark
        times = []
        for _ in range(3):  # Fewer runs due to API costs
            start_time = time.time()
            embeddings = await provider.generate_embeddings(benchmark_texts)
            duration = time.time() - start_time
            times.append(duration)
        
        avg_time = statistics.mean(times)
        per_text_time = avg_time / len(benchmark_texts) * 1000  # ms
        
        # Performance targets
        assert per_text_time < 100, f"OpenAI too slow: {per_text_time:.2f}ms per text"
        assert len(embeddings) == len(benchmark_texts)
        assert all(len(emb) == 1536 for emb in embeddings)
        
        print(f"OpenAI Performance: {per_text_time:.2f}ms per text")

# tests/performance/benchmarks/test_search_performance.py
@pytest.mark.performance
class TestSearchPerformance:
    """Search performance benchmarks."""
    
    @pytest.mark.asyncio
    async def test_vector_search_latency(self, mock_qdrant_client):
        """Benchmark vector search latency."""
        from src.services.optimized_vector_service import OptimizedVectorService
        
        pool = AsyncMock()
        pool.get_connection.return_value = mock_qdrant_client
        pool.return_connection = AsyncMock()
        
        service = OptimizedVectorService(pool)
        
        # Benchmark search
        query_vector = [0.1] * 384
        times = []
        
        for _ in range(100):  # Many iterations for stable results
            start_time = time.time()
            await service.optimized_search(
                "test_collection",
                query_vector,
                limit=10
            )
            duration = time.time() - start_time
            times.append(duration * 1000)  # Convert to ms
        
        avg_time = statistics.mean(times)
        p95_time = statistics.quantiles(times, n=20)[18]
        
        # Performance targets
        assert avg_time < 25, f"Search too slow: {avg_time:.2f}ms average"
        assert p95_time < 50, f"P95 too slow: {p95_time:.2f}ms"
        
        print(f"Search Performance - Avg: {avg_time:.2f}ms, P95: {p95_time:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_concurrent_search_performance(self, mock_qdrant_client):
        """Test concurrent search performance."""
        import asyncio
        
        from src.services.optimized_vector_service import OptimizedVectorService
        
        pool = AsyncMock()
        pool.get_connection.return_value = mock_qdrant_client
        pool.return_connection = AsyncMock()
        
        service = OptimizedVectorService(pool)
        
        async def single_search():
            query_vector = [0.1] * 384
            return await service.optimized_search(
                "test_collection",
                query_vector,
                limit=10
            )
        
        # Test concurrent searches
        start_time = time.time()
        tasks = [single_search() for _ in range(50)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        throughput = len(tasks) / total_time
        
        # Performance targets
        assert throughput > 100, f"Throughput too low: {throughput:.2f} searches/sec"
        assert all(len(result) > 0 for result in results)
        
        print(f"Concurrent Search Throughput: {throughput:.2f} searches/sec")
```

#### Stress Testing

```python
# tests/performance/stress/test_memory_stress.py
import pytest
import psutil
import gc
from src.utils.memory_optimization import MemoryOptimizer

@pytest.mark.stress
class TestMemoryStress:
    """Memory stress testing."""
    
    @pytest.mark.asyncio
    async def test_large_batch_processing(self):
        """Test memory usage with large batches."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate large dataset
        large_texts = [f"Document {i} with substantial content. " * 50 for i in range(10000)]
        
        # Process with memory monitoring
        async with MemoryOptimizer.memory_monitor(threshold_mb=500):
            # Simulate embedding generation
            embeddings = []
            for i in range(0, len(large_texts), 100):
                batch = large_texts[i:i + 100]
                # Mock embedding generation
                batch_embeddings = [[0.1] * 384 for _ in batch]
                embeddings.extend(batch_embeddings)
                
                # Check memory periodically
                if i % 1000 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_increase = current_memory - initial_memory
                    
                    # Memory increase should be reasonable
                    assert memory_increase < 1000, f"Memory usage too high: {memory_increase:.2f}MB"
        
        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory
        
        assert total_increase < 100, f"Memory leak detected: {total_increase:.2f}MB increase"
        assert len(embeddings) == len(large_texts)
    
    @pytest.mark.asyncio
    async def test_concurrent_memory_usage(self):
        """Test memory usage under concurrent load."""
        import asyncio
        
        async def memory_intensive_task():
            # Simulate embedding generation
            texts = [f"Text {i}" for i in range(1000)]
            embeddings = [[0.1] * 384 for _ in texts]
            return len(embeddings)
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Run many concurrent tasks
        tasks = [memory_intensive_task() for _ in range(50)]
        results = await asyncio.gather(*tasks)
        
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = current_memory - initial_memory
        
        # Memory usage should be reasonable
        assert memory_increase < 500, f"Concurrent memory usage too high: {memory_increase:.2f}MB"
        assert all(result == 1000 for result in results)
```

### Phase 5: Quality Assurance

#### Code Quality Checks

```python
# tests/quality/test_code_quality.py
import ast
import pytest
from pathlib import Path
from typing import List, Set

class TestCodeQuality:
    """Code quality and standards testing."""
    
    def test_no_hardcoded_secrets(self):
        """Ensure no hardcoded secrets in code."""
        src_dir = Path("src")
        secret_patterns = [
            "api_key", "password", "token", "secret", "key"
        ]
        
        for py_file in src_dir.glob("**/*.py"):
            with open(py_file, 'r') as f:
                content = f.read().lower()
                
                for pattern in secret_patterns:
                    # Check for potential hardcoded secrets
                    if f'{pattern}="' in content or f"{pattern}='" in content:
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if f'{pattern}=' in line and not line.strip().startswith('#'):
                                # Allow environment variable references
                                if 'os.getenv' not in line and 'env' not in line:
                                    pytest.fail(
                                        f"Potential hardcoded secret in {py_file}:{i+1}: {line.strip()}"
                                    )
    
    def test_proper_async_patterns(self):
        """Ensure proper async/await patterns."""
        src_dir = Path("src")
        
        for py_file in src_dir.glob("**/*.py"):
            with open(py_file, 'r') as f:
                try:
                    tree = ast.parse(f.read())
                except SyntaxError:
                    continue
                
                for node in ast.walk(tree):
                    # Check for sync calls in async functions
                    if isinstance(node, ast.FunctionDef) and any(
                        isinstance(decorator, ast.Name) and decorator.id == 'async'
                        for decorator in getattr(node, 'decorator_list', [])
                    ):
                        # This is an async function
                        for child in ast.walk(node):
                            if isinstance(child, ast.Call):
                                # Check for blocking calls that should be awaited
                                blocking_calls = ['time.sleep', 'requests.get', 'requests.post']
                                if any(self._is_call_to(child, call) for call in blocking_calls):
                                    pytest.fail(
                                        f"Blocking call in async function {node.name} in {py_file}"
                                    )
    
    def _is_call_to(self, call_node: ast.Call, target: str) -> bool:
        """Check if AST call node matches target function."""
        if isinstance(call_node.func, ast.Attribute):
            if isinstance(call_node.func.value, ast.Name):
                full_name = f"{call_node.func.value.id}.{call_node.func.attr}"
                return full_name == target
        return False
    
    def test_type_hints_coverage(self):
        """Ensure adequate type hint coverage."""
        src_dir = Path("src")
        
        for py_file in src_dir.glob("**/*.py"):
            if py_file.name == "__init__.py":
                continue
                
            with open(py_file, 'r') as f:
                try:
                    tree = ast.parse(f.read())
                except SyntaxError:
                    continue
                
                functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                
                if not functions:
                    continue
                
                typed_functions = 0
                for func in functions:
                    # Check if function has return type annotation
                    if func.returns is not None:
                        typed_functions += 1
                    # Check if parameters have type annotations
                    elif any(arg.annotation is not None for arg in func.args.args):
                        typed_functions += 1
                
                # Require at least 80% type hint coverage
                coverage = typed_functions / len(functions) if functions else 1
                assert coverage >= 0.8, f"Low type hint coverage in {py_file}: {coverage:.1%}"
```

## Test Execution Strategy

### Continuous Integration Pipeline

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12, 3.13]
    
    services:
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - 6333:6333
      redis:
        image: redis:latest
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install uv
      run: pip install uv
    
    - name: Install dependencies
      run: uv pip install -r requirements.txt
    
    - name: Lint with ruff
      run: |
        uv run ruff check .
        uv run ruff format --check .
    
    - name: Type check with mypy
      run: uv run mypy src/
    
    - name: Run unit tests
      run: uv run pytest tests/unit/ -v --cov=src --cov-report=xml
    
    - name: Run integration tests
      run: uv run pytest tests/integration/ -v -m "not slow"
    
    - name: Run performance benchmarks
      run: uv run pytest tests/performance/benchmarks/ -v --benchmark-only
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Local Development Testing

```bash
#!/bin/bash
# scripts/run_tests.sh

set -e

echo "Running comprehensive test suite..."

# Setup
echo "Setting up test environment..."
docker-compose -f tests/fixtures/docker/qdrant.yml up -d
sleep 10

# Lint and format
echo "Checking code quality..."
uv run ruff check --fix .
uv run ruff format .

# Type checking
echo "Type checking..."
uv run mypy src/

# Unit tests
echo "Running unit tests..."
uv run pytest tests/unit/ -v --cov=src --cov-report=term-missing --cov-fail-under=95

# Integration tests
echo "Running integration tests..."
uv run pytest tests/integration/ -v

# Performance tests (subset)
echo "Running performance benchmarks..."
uv run pytest tests/performance/benchmarks/ -v -k "not stress"

# Quality checks
echo "Running quality checks..."
uv run pytest tests/quality/ -v

# Cleanup
echo "Cleaning up..."
docker-compose -f tests/fixtures/docker/qdrant.yml down

echo "All tests completed successfully!"
```

## Official Documentation References

### Testing Frameworks

- **pytest-asyncio**: <https://pytest-asyncio.readthedocs.io/>
- **pytest**: <https://docs.pytest.org/en/stable/>
- **unittest.mock**: <https://docs.python.org/3/library/unittest.mock.html>
- **pytest-cov**: <https://pytest-cov.readthedocs.io/>

### Performance Testing

- **pytest-benchmark**: <https://pytest-benchmark.readthedocs.io/>
- **memory-profiler**: <https://pypi.org/project/memory-profiler/>
- **psutil**: <https://psutil.readthedocs.io/>

### Quality Assurance

- **mypy**: <https://mypy.readthedocs.io/>
- **ruff**: <https://docs.astral.sh/ruff/>
- **codecov**: <https://docs.codecov.com/>

### Container Testing

- **testcontainers**: <https://testcontainers-python.readthedocs.io/>
- **Docker Compose**: <https://docs.docker.com/compose/>

## Success Criteria

### Coverage Metrics

- [ ] 95%+ overall test coverage
- [ ] 100% async operation coverage
- [ ] 100% error condition coverage
- [ ] 90%+ integration test coverage

### Performance Standards

- [ ] All benchmarks meet target performance
- [ ] No performance regression >5%
- [ ] Memory usage within acceptable limits
- [ ] Concurrent load testing passes

### Quality Gates

- [ ] Zero critical code quality issues
- [ ] All type hints properly defined
- [ ] No hardcoded secrets detected
- [ ] Proper async patterns throughout

This comprehensive testing strategy ensures production-ready quality and reliability.
