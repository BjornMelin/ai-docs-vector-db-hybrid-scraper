"""Comprehensive integration test examples demonstrating P5 testing patterns.

This file contains real-world integration test implementations that follow
the patterns outlined in the integration-testing.md strategy document.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from faker import Faker
from hypothesis import given, strategies as st


# Test utilities
fake = Faker()


class IntegrationTestBase:
    """Base class for integration tests with common utilities."""

    @staticmethod
    async def measure_performance(func, *args, **kwargs):
        """Measure performance of async operations."""
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result, elapsed_time

    @staticmethod
    def create_test_documents(count: int = 10) -> list[dict[str, Any]]:
        """Create test documents for integration testing."""
        return [
            {
                "id": str(uuid.uuid4()),
                "content": fake.text(max_nb_chars=1000),
                "url": fake.url(),
                "metadata": {
                    "created_at": datetime.utcnow().isoformat(),
                    "author": fake.name(),
                    "category": fake.random_element(["tech", "science", "business"]),
                    "project_id": fake.uuid4(),
                },
            }
            for _ in range(count)
        ]


# Example 1: End-to-End Document Processing Pipeline
class TestDocumentProcessingPipeline(IntegrationTestBase):
    """Test complete document processing pipeline integration."""

    @pytest.fixture
    async def pipeline_services(self):
        """Provide integrated pipeline services."""
        from src.infrastructure.clients.qdrant_client import QdrantClient
        from src.services.cache.intelligent import IntelligentCache
        from src.services.embeddings.manager import EmbeddingManager
        from src.services.managers.crawling_manager import CrawlingManager
        from src.services.processing.batch_optimizer import BatchOptimizer

        return {
            "crawling": CrawlingManager(),
            "embeddings": EmbeddingManager(),
            "vector_db": QdrantClient(),
            "cache": IntelligentCache(),
            "batch_optimizer": BatchOptimizer(),
        }

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_document_pipeline(self, pipeline_services):
        """Test complete document ingestion and retrieval pipeline."""
        # Step 1: Crawl document
        doc_url = "https://example.com/technical-article.html"

        with respx.mock:
            # Mock HTTP response
            respx.get(doc_url).mock(
                return_value=httpx.Response(
                    200,
                    content="""
                    <html>
                        <body>
                            <h1>Advanced Machine Learning Techniques</h1>
                            <p>This article discusses state-of-the-art ML algorithms...</p>
                            <p>Deep learning has revolutionized computer vision...</p>
                        </body>
                    </html>
                    """,
                )
            )

            # Crawl the document
            crawled_doc = await pipeline_services["crawling"].crawl(
                url=doc_url, options={"extract_metadata": True, "clean_html": True}
            )

            assert crawled_doc.content
            assert (
                crawled_doc.metadata["title"] == "Advanced Machine Learning Techniques"
            )

        # Step 2: Process content in batches
        chunks = await pipeline_services["batch_optimizer"].chunk_content(
            content=crawled_doc.content, chunk_size=512, overlap=50
        )

        assert len(chunks) > 0
        assert all(len(chunk.text) <= 512 for chunk in chunks)

        # Step 3: Generate embeddings with caching
        embeddings = []
        cache = pipeline_services["cache"]

        for chunk in chunks:
            # Check cache first
            cache_key = f"embedding:{hash(chunk.text)}"
            cached_embedding = await cache.get(cache_key)

            if cached_embedding:
                embeddings.append(cached_embedding)
            else:
                # Generate new embedding
                embedding = await pipeline_services["embeddings"].generate_embedding(
                    text=chunk.text, model="text-embedding-3-small"
                )

                # Cache the result
                await cache.set(cache_key, embedding, ttl=3600)
                embeddings.append(embedding)

        assert len(embeddings) == len(chunks)
        assert all(len(emb) == 1536 for emb in embeddings)  # OpenAI dimension

        # Step 4: Store in vector database
        vector_db = pipeline_services["vector_db"]
        await vector_db.initialize()

        # Create collection if not exists
        collection_name = "test_documents"
        await vector_db.create_collection(
            name=collection_name, vector_size=1536, distance="Cosine"
        )

        # Prepare points for insertion
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
            points.append(
                {
                    "id": f"{crawled_doc.doc_id}_chunk_{i}",
                    "vector": embedding,
                    "payload": {
                        "content": chunk.text,
                        "doc_id": crawled_doc.doc_id,
                        "url": doc_url,
                        "chunk_index": i,
                        "metadata": crawled_doc.metadata,
                    },
                }
            )

        # Batch insert
        await vector_db.upsert_points(collection_name=collection_name, points=points)

        # Step 5: Verify retrieval with search
        query = "deep learning computer vision"
        query_embedding = await pipeline_services["embeddings"].generate_embedding(
            text=query, model="text-embedding-3-small"
        )

        search_results = await vector_db.search(
            collection_name=collection_name, query_vector=query_embedding, limit=5
        )

        assert len(search_results) > 0
        assert search_results[0].score > 0.7  # Relevance threshold
        assert "deep learning" in search_results[0].payload["content"].lower()

        # Performance validation
        assert crawled_doc.processing_time < 2.0  # Under 2 seconds

        # Cleanup
        await vector_db.delete_collection(collection_name)


# Example 2: Service Orchestration and Communication
class TestServiceOrchestration(IntegrationTestBase):
    """Test service orchestration and inter-service communication."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multi_service_orchestration(self):
        """Test orchestration across multiple services."""
        from src.services.agents.tool_orchestration import ToolOrchestrationService
        from src.services.enterprise.search import EnterpriseSearchService
        from src.services.hyde.engine import HyDEEngine
        from src.services.query_processing.pipeline import QueryPipeline

        # Initialize services
        orchestrator = ToolOrchestrationService()
        search_service = EnterpriseSearchService()
        hyde_engine = HyDEEngine()
        query_pipeline = QueryPipeline()

        # Complex query requiring orchestration
        user_query = (
            "What are the latest advances in transformer architectures for NLP?"
        )

        # Step 1: Query processing and expansion
        processed_query = await query_pipeline.process(
            query=user_query,
            options={
                "expand_synonyms": True,
                "extract_entities": True,
                "detect_intent": True,
            },
        )

        assert processed_query.intent == "research_query"
        assert "transformer" in processed_query.keywords
        assert "NLP" in processed_query.entities

        # Step 2: Generate hypothetical documents (HyDE)
        hypothetical_docs = await hyde_engine.generate_hypothetical_documents(
            query=processed_query.enhanced_query, num_documents=3
        )

        assert len(hypothetical_docs) == 3
        assert all("transformer" in doc.lower() for doc in hypothetical_docs)

        # Step 3: Orchestrate search across multiple sources
        search_workflow = await orchestrator.create_workflow(
            steps=[
                {
                    "service": "vector_search",
                    "action": "search",
                    "params": {
                        "query": processed_query.enhanced_query,
                        "hypothetical_docs": hypothetical_docs,
                        "limit": 10,
                    },
                },
                {
                    "service": "keyword_search",
                    "action": "search",
                    "params": {
                        "query": processed_query.enhanced_query,
                        "filters": {"category": "research"},
                        "limit": 10,
                    },
                },
                {
                    "service": "result_fusion",
                    "action": "merge",
                    "params": {"strategy": "reciprocal_rank_fusion", "k": 60},
                },
            ]
        )

        # Execute workflow
        results = await orchestrator.execute_workflow(search_workflow)

        assert results["status"] == "success"
        assert len(results["final_results"]) > 0
        assert results["execution_time"] < 3.0  # Performance constraint

        # Verify result quality
        top_result = results["final_results"][0]
        assert top_result["relevance_score"] > 0.8
        assert any(
            keyword in top_result["content"].lower()
            for keyword in ["transformer", "architecture", "nlp"]
        )


# Example 3: Security and Authentication Integration
class TestSecurityIntegration(IntegrationTestBase):
    """Test security components integration."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_security_flow(self):
        """Test complete security flow from auth to API access."""
        from src.api.app_factory import create_app
        from src.services.security.auth import AuthService
        from src.services.security.rate_limiter import DistributedRateLimiter

        # Create test app with security
        app = create_app(testing=True)
        auth_service = AuthService()
        rate_limiter = DistributedRateLimiter()

        # Test user registration and authentication
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            # Register new user
            register_response = await client.post(
                "/api/v1/auth/register",
                json={
                    "email": "security_test@example.com",
                    "password": "SecureP@ssw0rd123!",
                    "name": "Security Tester",
                },
            )
            assert register_response.status_code == 201
            user_data = register_response.json()

            # Login
            login_response = await client.post(
                "/api/v1/auth/login",
                json={
                    "email": "security_test@example.com",
                    "password": "SecureP@ssw0rd123!",
                },
            )
            assert login_response.status_code == 200
            tokens = login_response.json()

            # Test authenticated API access
            headers = {"Authorization": f"Bearer {tokens['access_token']}"}

            # Make authenticated requests
            for _i in range(10):
                response = await client.get("/api/v1/documents", headers=headers)
                assert response.status_code == 200

            # Test rate limiting
            for _i in range(100):  # Exceed rate limit
                response = await client.get("/api/v1/documents", headers=headers)
                if response.status_code == 429:
                    # Rate limit hit
                    assert "X-RateLimit-Remaining" in response.headers
                    assert response.headers["X-RateLimit-Remaining"] == "0"
                    break
            else:
                pytest.fail("Rate limiting did not trigger")

            # Test token refresh
            refresh_response = await client.post(
                "/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
            )
            assert refresh_response.status_code == 200
            new_tokens = refresh_response.json()
            assert new_tokens["access_token"] != tokens["access_token"]

            # Test security headers
            response = await client.get("/api/v1/health")
            security_headers = [
                "X-Content-Type-Options",
                "X-Frame-Options",
                "X-XSS-Protection",
                "Strict-Transport-Security",
            ]
            for header in security_headers:
                assert header in response.headers


# Example 4: Performance and Load Testing
class TestPerformanceIntegration(IntegrationTestBase):
    """Test system performance under load."""

    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_load_handling(self):
        """Test system handling concurrent load."""
        from src.infrastructure.clients.qdrant_client import QdrantClient
        from src.services.cache.intelligent import IntelligentCache
        from src.services.embeddings.manager import EmbeddingManager

        # Initialize services
        embedding_manager = EmbeddingManager()
        vector_db = QdrantClient()
        cache = IntelligentCache()

        await vector_db.initialize()

        # Create test data
        documents = self.create_test_documents(count=100)

        # Measure baseline performance
        single_doc = documents[0]
        _, baseline_time = await self.measure_performance(
            embedding_manager.generate_embedding, single_doc["content"]
        )

        print(f"Baseline embedding time: {baseline_time:.3f}s")

        # Test concurrent processing
        async def process_document(doc):
            """Process a single document through the pipeline."""
            start = time.time()

            # Generate embedding
            embedding = await embedding_manager.generate_embedding(doc["content"])

            # Store in vector DB
            await vector_db.upsert_point(
                collection_name="performance_test",
                point={
                    "id": doc["id"],
                    "vector": embedding,
                    "payload": doc["metadata"],
                },
            )

            # Cache result
            await cache.set(
                f"doc:{doc['id']}",
                {"embedding": embedding, "processed_at": datetime.utcnow().isoformat()},
            )

            return time.time() - start

        # Process documents concurrently
        start_time = time.time()
        tasks = [process_document(doc) for doc in documents]
        processing_times = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Filter out exceptions
        successful_times = [t for t in processing_times if isinstance(t, float)]
        failed_count = len(processing_times) - len(successful_times)

        # Performance assertions
        assert failed_count < 5  # Less than 5% failure rate
        assert total_time < baseline_time * 20  # Better than sequential

        # Calculate throughput
        throughput = len(successful_times) / total_time
        avg_latency = sum(successful_times) / len(successful_times)
        p95_latency = sorted(successful_times)[int(len(successful_times) * 0.95)]

        print(f"Throughput: {throughput:.2f} docs/sec")
        print(f"Average latency: {avg_latency:.3f}s")
        print(f"P95 latency: {p95_latency:.3f}s")

        # Performance requirements
        assert throughput > 5.0  # At least 5 docs/sec
        assert avg_latency < 2.0  # Average under 2 seconds
        assert p95_latency < 5.0  # P95 under 5 seconds


# Example 5: Disaster Recovery and Resilience Testing
class TestDisasterRecovery(IntegrationTestBase):
    """Test system resilience and recovery capabilities."""

    @pytest.mark.integration
    @pytest.mark.resilience
    @pytest.mark.asyncio
    async def test_service_failure_recovery(self):
        """Test recovery from service failures."""
        from src.infrastructure.database.connection_manager import ConnectionManager
        from src.services.managers.database_manager import DatabaseManager
        from src.services.monitoring.health import HealthMonitor

        # Initialize services
        db_manager = DatabaseManager()
        health_monitor = HealthMonitor()
        connection_manager = ConnectionManager()

        # Simulate vector database failure
        with patch.object(db_manager, "_vector_db") as mock_db:
            # First call fails
            mock_db.search.side_effect = [
                Exception("Connection lost"),
                Exception("Connection lost"),
                # Then recovers
                [{"id": "1", "score": 0.9, "content": "Recovered result"}],
            ]

            # Attempt search with retry logic
            attempts = 0
            max_attempts = 3
            result = None

            while attempts < max_attempts:
                try:
                    result = await db_manager.search(
                        query_vector=[0.1] * 1536, limit=10
                    )
                    break
                except Exception as e:
                    attempts += 1
                    if attempts < max_attempts:
                        # Exponential backoff
                        await asyncio.sleep(2**attempts)
                        # Trigger health check
                        await health_monitor.check_service("vector_db")

            assert result is not None
            assert len(result) > 0
            assert attempts == 2  # Failed twice, succeeded on third

        # Test circuit breaker behavior
        circuit_breaker = connection_manager.circuit_breaker

        # Simulate repeated failures
        for _ in range(5):
            with pytest.raises(Exception):
                await connection_manager.execute_with_circuit_breaker(
                    lambda: asyncio.sleep(0) or 1 / 0  # Always fails
                )

        # Circuit should be open
        assert circuit_breaker.state == "open"

        # Further calls should fail fast
        start = time.time()
        with pytest.raises(Exception, match="Circuit breaker is open"):
            await connection_manager.execute_with_circuit_breaker(
                lambda: asyncio.sleep(1)
            )
        assert time.time() - start < 0.1  # Failed immediately

        # Wait for circuit to half-open
        await asyncio.sleep(circuit_breaker.timeout)

        # Successful call should close circuit
        result = await connection_manager.execute_with_circuit_breaker(
            lambda: "success"
        )
        assert result == "success"
        assert circuit_breaker.state == "closed"


# Example 6: Data Consistency and Transaction Testing
class TestDataConsistency(IntegrationTestBase):
    """Test data consistency across distributed components."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_distributed_transaction_consistency(self):
        """Test consistency in distributed transactions."""
        from src.infrastructure.clients.redis_client import RedisClient
        from src.services.cache.intelligent import IntelligentCache
        from src.services.managers.database_manager import DatabaseManager

        db_manager = DatabaseManager()
        cache = IntelligentCache()
        redis_client = RedisClient()

        # Test document update consistency
        doc_id = "consistency_test_doc"
        original_content = "Original document content"
        updated_content = "Updated document content with new information"

        # Step 1: Initial document creation
        await db_manager.create_document(
            {"id": doc_id, "content": original_content, "version": 1}
        )

        # Cache the document
        await cache.set(f"doc:{doc_id}", {"content": original_content, "version": 1})

        # Step 2: Concurrent update attempts
        async def update_document(new_content, version):
            """Attempt to update document with optimistic locking."""
            # Get current version
            current_doc = await db_manager.get_document(doc_id)
            if current_doc["version"] != version:
                msg = "Version conflict"
                raise Exception(msg)

            # Update document
            await db_manager.update_document(
                doc_id, {"content": new_content, "version": version + 1}
            )

            # Invalidate cache
            await cache.delete(f"doc:{doc_id}")

            # Update cache with new version
            await cache.set(
                f"doc:{doc_id}", {"content": new_content, "version": version + 1}
            )

            return True

        # Simulate concurrent updates
        update_tasks = [
            update_document("Update A", 1),
            update_document("Update B", 1),
            update_document("Update C", 1),
        ]

        results = await asyncio.gather(*update_tasks, return_exceptions=True)

        # Only one should succeed
        successes = [r for r in results if r is True]
        failures = [r for r in results if isinstance(r, Exception)]

        assert len(successes) == 1
        assert len(failures) == 2
        assert all("Version conflict" in str(e) for e in failures)

        # Verify final state consistency
        final_doc = await db_manager.get_document(doc_id)
        cached_doc = await cache.get(f"doc:{doc_id}")

        assert final_doc["version"] == 2
        assert cached_doc["version"] == 2
        assert final_doc["content"] == cached_doc["content"]


# Test fixtures and utilities
@pytest.fixture
async def mock_external_services():
    """Mock external service dependencies."""
    with respx.mock:
        # Mock OpenAI embeddings
        respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [{"embedding": [0.1] * 1536}],
                    "model": "text-embedding-3-small",
                    "usage": {"total_tokens": 10},
                },
            )
        )

        # Mock Anthropic API
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "content": [{"text": "Mocked response"}],
                    "model": "claude-3-opus-20240229",
                },
            )
        )

        yield


@pytest.fixture
async def test_database():
    """Provide test database with cleanup."""
    from src.infrastructure.database.connection_manager import ConnectionManager

    manager = ConnectionManager(test_mode=True)
    await manager.initialize()

    yield manager

    # Cleanup
    await manager.cleanup()


# Performance monitoring utilities
class PerformanceMonitor:
    """Monitor performance metrics during tests."""

    def __init__(self):
        self.metrics = {"latencies": [], "throughput": [], "errors": []}

    def record_latency(self, operation: str, duration: float):
        """Record operation latency."""
        self.metrics["latencies"].append(
            {
                "operation": operation,
                "duration": duration,
                "timestamp": datetime.utcnow(),
            }
        )

    def calculate_stats(self):
        """Calculate performance statistics."""
        if not self.metrics["latencies"]:
            return {}

        latencies = [m["duration"] for m in self.metrics["latencies"]]
        return {
            "avg_latency": sum(latencies) / len(latencies),
            "p50_latency": sorted(latencies)[len(latencies) // 2],
            "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99_latency": sorted(latencies)[int(len(latencies) * 0.99)],
            "total_operations": len(latencies),
        }


if __name__ == "__main__":
    print("P5 Integration Test Examples")
    print("===========================")
    print("This file contains comprehensive integration test examples.")
    print(
        "Run with: uv run pytest planning/in-progress/P5/integration-test-examples.py -v"
    )
