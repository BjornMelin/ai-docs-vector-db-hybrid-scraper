# Phase 5: Integration & System Testing Strategy

## Overview

This document outlines the comprehensive integration testing strategy for the AI Docs Vector DB Hybrid Scraper project, focusing on component interactions, data flow validation, security framework integration, and enterprise feature validation.

## 1. System Integration Testing Strategy

### 1.1 Component Interaction Testing

```python
# tests/integration/services/test_component_interactions.py
"""Test interactions between major system components."""

import asyncio
from typing import Dict, Any
import pytest
import respx
from httpx import AsyncClient

from src.services.embeddings.manager import EmbeddingManager
from src.services.managers.database_manager import DatabaseManager
from src.services.managers.crawling_manager import CrawlingManager
from src.services.cache.intelligent import IntelligentCache
from src.services.query_processing.pipeline import QueryPipeline


@pytest.mark.integration
class TestComponentInteractions:
    """Test component interactions and data flow."""
    
    @pytest.fixture
    async def integration_stack(self):
        """Provide integrated service stack."""
        return {
            "embedding_manager": EmbeddingManager(),
            "db_manager": DatabaseManager(),
            "crawling_manager": CrawlingManager(),
            "cache": IntelligentCache(),
            "query_pipeline": QueryPipeline()
        }
    
    async def test_document_ingestion_flow(self, integration_stack):
        """Test complete document ingestion pipeline."""
        # 1. Crawl document
        doc_url = "https://example.com/doc.html"
        
        async with respx.mock:
            respx.get(doc_url).mock(return_value=httpx.Response(
                200,
                content="<html><body>Test content</body></html>"
            ))
            
            # Crawl
            crawled_doc = await integration_stack["crawling_manager"].crawl(doc_url)
            assert crawled_doc.content
            
            # Process embeddings
            embeddings = await integration_stack["embedding_manager"].generate_embeddings(
                crawled_doc.content
            )
            assert embeddings.shape[0] > 0
            
            # Store in vector DB
            doc_id = await integration_stack["db_manager"].store_document(
                content=crawled_doc.content,
                embeddings=embeddings,
                metadata={"url": doc_url}
            )
            assert doc_id
            
            # Cache the result
            await integration_stack["cache"].set(
                f"doc:{doc_id}",
                {"content": crawled_doc.content, "embeddings": embeddings.tolist()}
            )
            
            # Verify retrieval
            cached_doc = await integration_stack["cache"].get(f"doc:{doc_id}")
            assert cached_doc is not None
```

### 1.2 Service-to-Service Communication

```python
# tests/integration/services/test_service_communication.py
"""Test service-to-service communication patterns."""

import pytest
from unittest.mock import AsyncMock, patch
import asyncio

from src.services.agents.tool_orchestration import ToolOrchestrationService
from src.services.enterprise.search import EnterpriseSearchService
from src.services.monitoring.health import HealthMonitor


@pytest.mark.integration
class TestServiceCommunication:
    """Test inter-service communication patterns."""
    
    async def test_orchestration_to_search_communication(self):
        """Test tool orchestration communicating with search service."""
        orchestrator = ToolOrchestrationService()
        search_service = EnterpriseSearchService()
        
        # Mock dependencies
        with patch.object(search_service, "_vector_db") as mock_db:
            mock_db.search.return_value = AsyncMock(return_value=[
                {"id": "1", "content": "Result 1", "score": 0.95}
            ])
            
            # Execute orchestrated search
            result = await orchestrator.execute_search_workflow(
                query="test query",
                search_service=search_service
            )
            
            assert result["status"] == "success"
            assert len(result["results"]) > 0
    
    async def test_health_monitoring_across_services(self):
        """Test health monitoring across all services."""
        health_monitor = HealthMonitor()
        
        # Register services
        services = {
            "embedding": AsyncMock(health_check=AsyncMock(return_value=True)),
            "database": AsyncMock(health_check=AsyncMock(return_value=True)),
            "cache": AsyncMock(health_check=AsyncMock(return_value=True))
        }
        
        for name, service in services.items():
            health_monitor.register_service(name, service)
        
        # Check all services
        health_status = await health_monitor.check_all_services()
        assert all(status["healthy"] for status in health_status.values())
```

### 1.3 Data Flow Validation

```python
# tests/integration/data_flow/test_data_consistency.py
"""Test data consistency across system components."""

import pytest
import hashlib
from typing import Dict, Any

from src.models.document_processing import ProcessedDocument
from src.services.processing.algorithms import DataProcessor


@pytest.mark.integration
class TestDataFlowConsistency:
    """Validate data consistency through processing pipeline."""
    
    async def test_data_integrity_through_pipeline(self):
        """Ensure data integrity is maintained through pipeline."""
        original_content = "Test document content for integrity validation"
        content_hash = hashlib.sha256(original_content.encode()).hexdigest()
        
        # Process through pipeline stages
        processor = DataProcessor()
        
        # Stage 1: Chunking
        chunks = await processor.chunk_document(original_content)
        reconstructed = "".join(chunk.content for chunk in chunks)
        assert hashlib.sha256(reconstructed.encode()).hexdigest() == content_hash
        
        # Stage 2: Embedding generation (mock)
        embeddings = await processor.generate_embeddings(chunks)
        assert len(embeddings) == len(chunks)
        
        # Stage 3: Storage and retrieval
        stored_data = await processor.store_processed_data(chunks, embeddings)
        retrieved_data = await processor.retrieve_data(stored_data["document_id"])
        
        # Verify integrity
        retrieved_content = "".join(chunk["content"] for chunk in retrieved_data["chunks"])
        assert hashlib.sha256(retrieved_content.encode()).hexdigest() == content_hash
```

## 2. API & Interface Testing Framework

### 2.1 FastAPI Integration Tests

```python
# tests/integration/api/test_fastapi_endpoints.py
"""Comprehensive FastAPI endpoint testing."""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
import json
from datetime import datetime

from src.api.app_factory import create_app
from src.config import Config


@pytest.mark.integration
@pytest.mark.asyncio
class TestFastAPIEndpoints:
    """Test FastAPI endpoints with full integration."""
    
    @pytest.fixture
    async def test_app(self):
        """Create test application."""
        config = Config(testing=True)
        app = create_app(config)
        return app
    
    @pytest.fixture
    async def async_client(self, test_app):
        """Create async test client."""
        async with AsyncClient(app=test_app, base_url="http://test") as client:
            yield client
    
    async def test_document_ingestion_endpoint(self, async_client):
        """Test document ingestion API endpoint."""
        # Test valid document submission
        response = await async_client.post(
            "/api/v1/documents/ingest",
            json={
                "url": "https://example.com/doc.html",
                "metadata": {"category": "test", "priority": "high"}
            }
        )
        assert response.status_code == 202  # Accepted
        assert "task_id" in response.json()
        
        # Test validation
        response = await async_client.post(
            "/api/v1/documents/ingest",
            json={"invalid": "data"}
        )
        assert response.status_code == 422  # Validation error
    
    async def test_search_endpoint_with_pagination(self, async_client):
        """Test search endpoint with pagination."""
        response = await async_client.post(
            "/api/v1/search",
            json={
                "query": "test query",
                "filters": {"category": "documentation"},
                "page": 1,
                "page_size": 10
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert "results" in result
        assert "total" in result
        assert "page" in result
    
    async def test_rate_limiting(self, async_client):
        """Test API rate limiting."""
        # Make multiple requests
        responses = []
        for _ in range(15):  # Exceed rate limit
            response = await async_client.get("/api/v1/health")
            responses.append(response)
        
        # Check that rate limiting kicked in
        assert any(r.status_code == 429 for r in responses[-5:])
```

### 2.2 Authentication & Authorization Testing

```python
# tests/integration/api/test_auth_flows.py
"""Test authentication and authorization flows."""

import pytest
import jwt
from datetime import datetime, timedelta
from httpx import AsyncClient

from src.services.security.auth import AuthService
from src.models.api_contracts import UserRole


@pytest.mark.integration
class TestAuthenticationFlows:
    """Test complete authentication flows."""
    
    @pytest.fixture
    async def auth_service(self):
        """Provide auth service instance."""
        return AuthService()
    
    async def test_complete_auth_flow(self, async_client, auth_service):
        """Test complete authentication flow."""
        # 1. Register user
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
                "role": "user"
            }
        )
        assert response.status_code == 201
        user_data = response.json()
        
        # 2. Login
        response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!"
            }
        )
        assert response.status_code == 200
        tokens = response.json()
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        
        # 3. Access protected resource
        response = await async_client.get(
            "/api/v1/user/profile",
            headers={"Authorization": f"Bearer {tokens['access_token']}"}
        )
        assert response.status_code == 200
        
        # 4. Refresh token
        response = await async_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": tokens["refresh_token"]}
        )
        assert response.status_code == 200
        new_tokens = response.json()
        assert new_tokens["access_token"] != tokens["access_token"]
    
    async def test_role_based_authorization(self, async_client, auth_service):
        """Test role-based access control."""
        # Create tokens for different roles
        user_token = await auth_service.create_token(
            user_id="user123",
            role=UserRole.USER
        )
        admin_token = await auth_service.create_token(
            user_id="admin123",
            role=UserRole.ADMIN
        )
        
        # Test user access to admin endpoint
        response = await async_client.get(
            "/api/v1/admin/users",
            headers={"Authorization": f"Bearer {user_token}"}
        )
        assert response.status_code == 403  # Forbidden
        
        # Test admin access
        response = await async_client.get(
            "/api/v1/admin/users",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
```

### 2.3 Input Validation & Error Response Testing

```python
# tests/integration/api/test_input_validation.py
"""Test input validation and error responses."""

import pytest
from typing import Dict, Any
import json


@pytest.mark.integration
class TestInputValidation:
    """Test API input validation and error handling."""
    
    async def test_sql_injection_protection(self, async_client):
        """Test SQL injection protection."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "1; UPDATE users SET role='admin' WHERE id=1;"
        ]
        
        for malicious_input in malicious_inputs:
            response = await async_client.post(
                "/api/v1/search",
                json={"query": malicious_input}
            )
            # Should sanitize input, not error
            assert response.status_code in [200, 400]
            # Verify no SQL error in response
            assert "sql" not in response.text.lower()
    
    async def test_xss_prevention(self, async_client):
        """Test XSS prevention in responses."""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>"
        ]
        
        for payload in xss_payloads:
            response = await async_client.post(
                "/api/v1/documents/ingest",
                json={
                    "url": "https://example.com",
                    "metadata": {"description": payload}
                }
            )
            
            # Check response doesn't contain unescaped payload
            response_text = response.text
            assert payload not in response_text
            assert "&lt;script&gt;" in response_text or response.status_code == 422
    
    async def test_comprehensive_validation_errors(self, async_client):
        """Test comprehensive validation error responses."""
        test_cases = [
            {
                "endpoint": "/api/v1/documents/ingest",
                "data": {"url": "not-a-url"},
                "expected_error": "Invalid URL format"
            },
            {
                "endpoint": "/api/v1/search",
                "data": {"query": "", "page": -1},
                "expected_error": "Page must be positive"
            },
            {
                "endpoint": "/api/v1/embeddings/generate",
                "data": {"text": "x" * 10001},  # Exceeds max length
                "expected_error": "Text exceeds maximum length"
            }
        ]
        
        for test_case in test_cases:
            response = await async_client.post(
                test_case["endpoint"],
                json=test_case["data"]
            )
            assert response.status_code == 422
            error_detail = response.json()["detail"]
            assert any(test_case["expected_error"] in str(err) for err in error_detail)
```

## 3. Database and Vector Operations Testing

### 3.1 Qdrant Integration Testing

```python
# tests/integration/database/test_qdrant_operations.py
"""Test Qdrant vector database operations."""

import pytest
import numpy as np
from typing import List, Dict, Any
import asyncio

from src.infrastructure.clients.qdrant_client import QdrantClient
from src.models.vector_search import VectorSearchRequest, VectorSearchResult


@pytest.mark.integration
class TestQdrantIntegration:
    """Test Qdrant vector database integration."""
    
    @pytest.fixture
    async def qdrant_client(self):
        """Provide Qdrant client for testing."""
        client = QdrantClient(test_mode=True)
        await client.initialize()
        yield client
        await client.cleanup()
    
    async def test_vector_crud_operations(self, qdrant_client):
        """Test vector CRUD operations."""
        # Create collection
        collection_name = "test_documents"
        await qdrant_client.create_collection(
            name=collection_name,
            vector_size=768,
            distance="Cosine"
        )
        
        # Insert vectors
        vectors = [
            {
                "id": f"doc_{i}",
                "vector": np.random.rand(768).tolist(),
                "payload": {
                    "content": f"Document {i}",
                    "metadata": {"category": "test", "index": i}
                }
            }
            for i in range(100)
        ]
        
        await qdrant_client.upsert_batch(collection_name, vectors)
        
        # Search vectors
        query_vector = np.random.rand(768).tolist()
        results = await qdrant_client.search(
            collection_name,
            query_vector,
            limit=10,
            score_threshold=0.7
        )
        
        assert len(results) <= 10
        assert all(r.score >= 0.7 for r in results)
        
        # Update vector
        await qdrant_client.update_payload(
            collection_name,
            point_id="doc_0",
            payload={"updated": True}
        )
        
        # Delete vector
        await qdrant_client.delete_points(collection_name, ["doc_0"])
        
        # Verify deletion
        point = await qdrant_client.get_point(collection_name, "doc_0")
        assert point is None
    
    async def test_batch_operations_performance(self, qdrant_client):
        """Test batch operation performance."""
        collection_name = "performance_test"
        await qdrant_client.create_collection(
            name=collection_name,
            vector_size=768
        )
        
        # Prepare large batch
        batch_size = 1000
        vectors = [
            {
                "id": f"perf_{i}",
                "vector": np.random.rand(768).tolist(),
                "payload": {"index": i}
            }
            for i in range(batch_size)
        ]
        
        # Measure insertion time
        start_time = asyncio.get_event_loop().time()
        await qdrant_client.upsert_batch(collection_name, vectors, batch_size=100)
        insertion_time = asyncio.get_event_loop().time() - start_time
        
        # Should complete within reasonable time
        assert insertion_time < 10.0  # 10 seconds for 1000 vectors
        
        # Verify all inserted
        count = await qdrant_client.count_points(collection_name)
        assert count == batch_size
```

### 3.2 Redis Cache Integration

```python
# tests/integration/cache/test_redis_operations.py
"""Test Redis cache integration."""

import pytest
import asyncio
import json
from datetime import timedelta

from src.infrastructure.clients.redis_client import RedisClient
from src.services.cache.intelligent import IntelligentCache


@pytest.mark.integration
class TestRedisIntegration:
    """Test Redis cache operations."""
    
    @pytest.fixture
    async def redis_client(self):
        """Provide Redis client for testing."""
        client = RedisClient(test_mode=True)
        await client.initialize()
        yield client
        await client.cleanup()
    
    async def test_cache_operations(self, redis_client):
        """Test basic cache operations."""
        # Set value
        await redis_client.set("test_key", {"data": "test_value"}, ttl=60)
        
        # Get value
        value = await redis_client.get("test_key")
        assert value == {"data": "test_value"}
        
        # Set with expiration
        await redis_client.set("expire_key", "expire_value", ttl=1)
        await asyncio.sleep(2)
        value = await redis_client.get("expire_key")
        assert value is None
        
        # Delete
        await redis_client.delete("test_key")
        value = await redis_client.get("test_key")
        assert value is None
    
    async def test_intelligent_cache_patterns(self, redis_client):
        """Test intelligent caching patterns."""
        cache = IntelligentCache(redis_client)
        
        # Test cache-aside pattern
        async def expensive_operation():
            await asyncio.sleep(0.1)
            return {"result": "expensive_data"}
        
        # First call - cache miss
        start_time = asyncio.get_event_loop().time()
        result1 = await cache.get_or_compute(
            "expensive_key",
            expensive_operation,
            ttl=300
        )
        first_call_time = asyncio.get_event_loop().time() - start_time
        
        # Second call - cache hit
        start_time = asyncio.get_event_loop().time()
        result2 = await cache.get_or_compute(
            "expensive_key",
            expensive_operation,
            ttl=300
        )
        second_call_time = asyncio.get_event_loop().time() - start_time
        
        assert result1 == result2
        assert second_call_time < first_call_time / 10  # Much faster
    
    async def test_cache_invalidation_patterns(self, redis_client):
        """Test cache invalidation patterns."""
        cache = IntelligentCache(redis_client)
        
        # Set related cache entries
        await cache.set("user:123:profile", {"name": "Test User"})
        await cache.set("user:123:posts", ["post1", "post2"])
        await cache.set("user:123:settings", {"theme": "dark"})
        
        # Tag-based invalidation
        await cache.invalidate_by_pattern("user:123:*")
        
        # Verify all invalidated
        assert await cache.get("user:123:profile") is None
        assert await cache.get("user:123:posts") is None
        assert await cache.get("user:123:settings") is None
```

### 3.3 Database Migration Testing

```python
# tests/integration/database/test_migrations.py
"""Test database migration operations."""

import pytest
from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import inspect

from src.infrastructure.database.monitoring import DatabaseMetrics


@pytest.mark.integration
class TestDatabaseMigrations:
    """Test database migration integrity."""
    
    @pytest.fixture
    def alembic_config(self):
        """Provide Alembic configuration."""
        config = AlembicConfig("alembic.ini")
        config.set_main_option("sqlalchemy.url", "postgresql://test@localhost/test_db")
        return config
    
    async def test_migration_up_and_down(self, alembic_config):
        """Test migration up and down operations."""
        # Migrate to head
        command.upgrade(alembic_config, "head")
        
        # Verify schema
        inspector = inspect(alembic_config.engine)
        tables = inspector.get_table_names()
        assert "documents" in tables
        assert "embeddings" in tables
        assert "users" in tables
        
        # Downgrade one revision
        command.downgrade(alembic_config, "-1")
        
        # Verify changes
        tables_after = inspector.get_table_names()
        assert len(tables_after) < len(tables)
    
    async def test_migration_data_integrity(self, alembic_config):
        """Test data integrity during migrations."""
        # Insert test data
        with alembic_config.engine.begin() as conn:
            conn.execute(
                "INSERT INTO documents (id, content) VALUES (1, 'Test content')"
            )
        
        # Run migration that modifies schema
        command.upgrade(alembic_config, "+1")
        
        # Verify data still exists
        with alembic_config.engine.begin() as conn:
            result = conn.execute("SELECT * FROM documents WHERE id = 1")
            assert result.fetchone() is not None
```

## 4. Security Framework Integration Testing

### 4.1 Authentication Middleware Testing

```python
# tests/integration/security/test_auth_middleware.py
"""Test authentication middleware integration."""

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
import jwt

from src.services.security.middleware import AuthMiddleware
from src.services.security.auth import get_current_user


@pytest.mark.integration
class TestAuthMiddleware:
    """Test authentication middleware integration."""
    
    @pytest.fixture
    def app_with_auth(self):
        """Create app with auth middleware."""
        app = FastAPI()
        app.add_middleware(AuthMiddleware)
        
        @app.get("/protected")
        async def protected_route(user=Depends(get_current_user)):
            return {"user_id": user.id}
        
        @app.get("/public")
        async def public_route():
            return {"message": "public"}
        
        return app
    
    def test_middleware_blocks_unauthorized(self, app_with_auth):
        """Test middleware blocks unauthorized requests."""
        client = TestClient(app_with_auth)
        
        # No token
        response = client.get("/protected")
        assert response.status_code == 401
        
        # Invalid token
        response = client.get(
            "/protected",
            headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == 401
        
        # Public route accessible
        response = client.get("/public")
        assert response.status_code == 200
    
    def test_middleware_allows_valid_tokens(self, app_with_auth):
        """Test middleware allows valid tokens."""
        client = TestClient(app_with_auth)
        
        # Create valid token
        valid_token = jwt.encode(
            {"sub": "user123", "exp": datetime.utcnow() + timedelta(hours=1)},
            "secret_key",
            algorithm="HS256"
        )
        
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {valid_token}"}
        )
        assert response.status_code == 200
        assert response.json()["user_id"] == "user123"
```

### 4.2 Input Sanitization Testing

```python
# tests/integration/security/test_input_sanitization.py
"""Test input sanitization across the system."""

import pytest
from typing import List, Dict, Any

from src.services.security.sanitization import InputSanitizer
from src.services.crawling.extractors import ContentExtractor


@pytest.mark.integration
class TestInputSanitization:
    """Test input sanitization integration."""
    
    @pytest.fixture
    def sanitizer(self):
        """Provide input sanitizer."""
        return InputSanitizer()
    
    async def test_end_to_end_sanitization(self, sanitizer):
        """Test sanitization through complete flow."""
        # Malicious input
        malicious_input = {
            "content": "<script>alert('xss')</script>Normal content",
            "metadata": {
                "title": "'; DROP TABLE users; --",
                "tags": ["<img src=x onerror=alert(1)>", "normal_tag"]
            }
        }
        
        # Sanitize
        sanitized = await sanitizer.sanitize_document(malicious_input)
        
        # Verify sanitization
        assert "<script>" not in sanitized["content"]
        assert "DROP TABLE" not in sanitized["metadata"]["title"]
        assert all("<img" not in tag for tag in sanitized["metadata"]["tags"])
        
        # Process through content extractor
        extractor = ContentExtractor()
        processed = await extractor.extract_content(sanitized["content"])
        
        # Verify no malicious content in processed result
        assert "<script>" not in processed.text
        assert all(char in processed.text for char in "Normal content")
```

### 4.3 Encryption and Data Protection

```python
# tests/integration/security/test_encryption.py
"""Test encryption and data protection integration."""

import pytest
from cryptography.fernet import Fernet

from src.services.security.encryption import EncryptionService
from src.infrastructure.database.models import SensitiveData


@pytest.mark.integration
class TestEncryptionIntegration:
    """Test encryption integration across services."""
    
    @pytest.fixture
    async def encryption_service(self):
        """Provide encryption service."""
        service = EncryptionService()
        await service.initialize()
        return service
    
    async def test_field_level_encryption(self, encryption_service, db_session):
        """Test field-level encryption in database."""
        # Create sensitive data
        sensitive_data = SensitiveData(
            user_id="user123",
            api_key="secret_api_key_12345",
            personal_info={"ssn": "123-45-6789"}
        )
        
        # Encrypt before storage
        encrypted_data = await encryption_service.encrypt_model(sensitive_data)
        
        # Store in database
        db_session.add(encrypted_data)
        await db_session.commit()
        
        # Retrieve and decrypt
        retrieved = await db_session.get(SensitiveData, sensitive_data.id)
        decrypted = await encryption_service.decrypt_model(retrieved)
        
        # Verify
        assert decrypted.api_key == "secret_api_key_12345"
        assert decrypted.personal_info["ssn"] == "123-45-6789"
        
        # Verify encrypted in DB
        assert retrieved.api_key != "secret_api_key_12345"
```

## 5. Performance Integration Testing

### 5.1 Load Testing Integration

```python
# tests/integration/performance/test_load_scenarios.py
"""Test system performance under load."""

import pytest
import asyncio
from typing import List, Dict, Any
import aiohttp

from src.benchmarks.load_test_runner import LoadTestRunner
from src.services.monitoring.performance_monitor import PerformanceMonitor


@pytest.mark.integration
@pytest.mark.performance
class TestLoadScenarios:
    """Test system behavior under various load scenarios."""
    
    @pytest.fixture
    async def load_runner(self):
        """Provide load test runner."""
        runner = LoadTestRunner()
        yield runner
        await runner.cleanup()
    
    async def test_concurrent_user_load(self, load_runner):
        """Test system with concurrent users."""
        # Define load scenario
        scenario = {
            "users": 100,
            "ramp_up": 10,  # seconds
            "duration": 60,  # seconds
            "endpoints": [
                {"method": "GET", "path": "/api/v1/health", "weight": 0.2},
                {"method": "POST", "path": "/api/v1/search", "weight": 0.5},
                {"method": "GET", "path": "/api/v1/documents/{id}", "weight": 0.3}
            ]
        }
        
        # Run load test
        results = await load_runner.run_scenario(scenario)
        
        # Verify performance targets
        assert results["success_rate"] > 0.99  # 99% success rate
        assert results["p95_latency"] < 100  # 95th percentile < 100ms
        assert results["throughput"] > 1000  # > 1000 req/sec
    
    async def test_spike_load_handling(self, load_runner):
        """Test system response to traffic spikes."""
        # Normal load
        normal_load = await load_runner.generate_load(users=50, duration=30)
        
        # Spike load
        spike_load = await load_runner.generate_load(users=500, duration=10)
        
        # Return to normal
        post_spike = await load_runner.generate_load(users=50, duration=30)
        
        # System should handle spike and recover
        assert spike_load["error_rate"] < 0.05  # < 5% errors during spike
        assert post_spike["p95_latency"] < normal_load["p95_latency"] * 1.2
```

### 5.2 Resource Usage Monitoring

```python
# tests/integration/performance/test_resource_monitoring.py
"""Test resource usage monitoring integration."""

import pytest
import psutil
import asyncio

from src.services.monitoring.resource_monitor import ResourceMonitor


@pytest.mark.integration
class TestResourceMonitoring:
    """Test resource monitoring during operations."""
    
    @pytest.fixture
    async def resource_monitor(self):
        """Provide resource monitor."""
        monitor = ResourceMonitor()
        await monitor.start()
        yield monitor
        await monitor.stop()
    
    async def test_memory_usage_during_processing(self, resource_monitor):
        """Monitor memory usage during document processing."""
        # Baseline memory
        baseline = resource_monitor.get_memory_usage()
        
        # Process large documents
        documents = [f"Large document content {i} " * 1000 for i in range(100)]
        
        # Monitor during processing
        peak_memory = baseline
        for doc in documents:
            # Simulate processing
            await asyncio.sleep(0.01)
            current_memory = resource_monitor.get_memory_usage()
            peak_memory = max(peak_memory, current_memory)
        
        # Verify memory usage reasonable
        memory_increase = peak_memory - baseline
        assert memory_increase < 500 * 1024 * 1024  # Less than 500MB increase
```

## 6. Enterprise Feature Integration Testing

### 6.1 Zero-Trust Security Testing

```python
# tests/integration/enterprise/test_zero_trust.py
"""Test zero-trust security architecture."""

import pytest
from typing import Dict, Any

from src.services.security.zero_trust import ZeroTrustValidator
from src.services.enterprise.audit import AuditLogger


@pytest.mark.integration
@pytest.mark.enterprise
class TestZeroTrustIntegration:
    """Test zero-trust security integration."""
    
    async def test_request_validation_chain(self):
        """Test complete request validation chain."""
        validator = ZeroTrustValidator()
        
        # Mock request
        request = {
            "user_id": "user123",
            "action": "read_document",
            "resource": "doc_456",
            "context": {
                "ip": "192.168.1.100",
                "device_id": "device_789",
                "location": "office"
            }
        }
        
        # Validate request through zero-trust chain
        validation_result = await validator.validate_request(request)
        
        assert validation_result["status"] in ["allowed", "denied", "requires_mfa"]
        assert "risk_score" in validation_result
        assert "validation_steps" in validation_result
```

### 6.2 Audit Trail Integration

```python
# tests/integration/enterprise/test_audit_trail.py
"""Test audit trail integration."""

import pytest
from datetime import datetime, timedelta

from src.services.enterprise.audit import AuditService
from src.models.audit import AuditEvent


@pytest.mark.integration
@pytest.mark.enterprise
class TestAuditTrailIntegration:
    """Test audit trail functionality."""
    
    async def test_complete_audit_trail(self, db_session):
        """Test complete audit trail for user actions."""
        audit_service = AuditService(db_session)
        
        # Simulate user journey
        user_id = "test_user_123"
        
        # Login
        await audit_service.log_event(
            AuditEvent(
                user_id=user_id,
                action="login",
                resource="auth_system",
                result="success"
            )
        )
        
        # Search
        await audit_service.log_event(
            AuditEvent(
                user_id=user_id,
                action="search",
                resource="document_index",
                details={"query": "test query"}
            )
        )
        
        # Access document
        await audit_service.log_event(
            AuditEvent(
                user_id=user_id,
                action="access_document",
                resource="doc_789",
                result="success"
            )
        )
        
        # Retrieve audit trail
        trail = await audit_service.get_user_trail(
            user_id,
            start_time=datetime.utcnow() - timedelta(hours=1)
        )
        
        assert len(trail) == 3
        assert trail[0].action == "login"
        assert trail[-1].action == "access_document"
```

## 7. Integration Test Infrastructure

### 7.1 Test Fixtures and Utilities

```python
# tests/integration/fixtures/service_fixtures.py
"""Shared fixtures for integration testing."""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import AsyncMock

from src.services.dependencies import ServiceRegistry


@pytest.fixture(scope="session")
async def service_registry():
    """Provide fully configured service registry."""
    registry = ServiceRegistry()
    
    # Initialize all services
    await registry.initialize_all()
    
    yield registry
    
    # Cleanup
    await registry.shutdown_all()


@pytest.fixture
async def mock_external_services():
    """Mock external service dependencies."""
    return {
        "openai": AsyncMock(
            embed=AsyncMock(return_value=[[0.1] * 768])
        ),
        "firecrawl": AsyncMock(
            scrape=AsyncMock(return_value={"content": "Mocked content"})
        ),
        "external_api": AsyncMock(
            fetch=AsyncMock(return_value={"status": "success"})
        )
    }


@pytest.fixture
async def integration_context(service_registry, mock_external_services):
    """Provide complete integration test context."""
    return {
        "services": service_registry,
        "mocks": mock_external_services,
        "config": {
            "test_mode": True,
            "use_mocks": True
        }
    }
```

### 7.2 Test Data Management

```python
# tests/integration/fixtures/test_data.py
"""Test data management for integration tests."""

import pytest
from typing import List, Dict, Any
import json
from pathlib import Path


class TestDataManager:
    """Manage test data for integration tests."""
    
    def __init__(self):
        self.data_dir = Path("tests/integration/data")
        self.data_dir.mkdir(exist_ok=True)
    
    def load_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Load test scenario data."""
        scenario_file = self.data_dir / f"{scenario_name}.json"
        if scenario_file.exists():
            return json.loads(scenario_file.read_text())
        return {}
    
    def create_test_documents(self, count: int = 10) -> List[Dict[str, Any]]:
        """Create test documents."""
        return [
            {
                "id": f"test_doc_{i}",
                "content": f"Test document content {i}. " * 50,
                "metadata": {
                    "category": "test",
                    "index": i,
                    "tags": ["integration", "test", f"batch_{i // 5}"]
                }
            }
            for i in range(count)
        ]
    
    def create_test_users(self, count: int = 5) -> List[Dict[str, Any]]:
        """Create test users."""
        roles = ["user", "admin", "moderator"]
        return [
            {
                "id": f"test_user_{i}",
                "email": f"user{i}@test.com",
                "role": roles[i % len(roles)],
                "active": True
            }
            for i in range(count)
        ]


@pytest.fixture
def test_data_manager():
    """Provide test data manager."""
    return TestDataManager()
```

## 8. Continuous Integration Configuration

### 8.1 GitHub Actions Integration Tests

```yaml
# .github/workflows/integration-tests.yml
name: Integration Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: testdb
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
      
      qdrant:
        image: qdrant/qdrant
        ports:
          - 6333:6333
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install uv
          uv pip install -e ".[test]"
      
      - name: Run integration tests
        env:
          DATABASE_URL: postgresql://postgres:testpass@localhost/testdb
          REDIS_URL: redis://localhost:6379
          QDRANT_URL: http://localhost:6333
          TESTING: true
        run: |
          uv run pytest tests/integration/ -v --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

## 9. Integration Testing Best Practices

### 9.1 Test Organization

```
tests/integration/
├── api/                    # API endpoint tests
├── auth/                   # Authentication/authorization tests
├── cache/                  # Cache layer tests
├── database/               # Database operation tests
├── enterprise/             # Enterprise feature tests
├── performance/            # Performance integration tests
├── security/               # Security integration tests
├── services/               # Service interaction tests
└── fixtures/               # Shared fixtures and utilities
```

### 9.2 Testing Guidelines

1. **Boundary Testing**: Test at service boundaries, not internal implementations
2. **Realistic Data**: Use realistic test data that represents production scenarios
3. **Async Patterns**: Properly handle async operations with pytest-asyncio
4. **Resource Cleanup**: Always cleanup resources in fixtures
5. **Performance Targets**: Include performance assertions in integration tests
6. **Security Validation**: Validate security measures in every relevant test

### 9.3 Anti-Patterns to Avoid

1. **Over-mocking**: Don't mock internal components, only external services
2. **Shared State**: Avoid tests that depend on execution order
3. **Hardcoded Values**: Use fixtures and factories for test data
4. **Missing Cleanup**: Always cleanup database/cache state after tests
5. **Ignoring Errors**: Don't suppress errors in integration tests

## Conclusion

This comprehensive integration testing strategy ensures:
- Complete validation of component interactions
- Robust API testing with security validation
- Database and cache operation verification
- Enterprise feature integration testing
- Performance validation under load
- Security framework validation

The strategy follows the project's testing principles while providing thorough coverage of all system integration points.