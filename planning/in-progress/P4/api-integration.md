# API Integration Implementation Plan

## Executive Summary

This plan outlines the implementation of a production-ready API layer with FastAPI for the AI Docs Vector DB Hybrid Scraper project. The implementation focuses on scalability, performance (sub-100ms P95 latency), developer experience, and library-first patterns following Pydantic v2 standards.

## Current State Analysis

### Existing Infrastructure
- **FastAPI app factory** with dual-mode support (Simple/Enterprise)
- **Basic middleware stack**: CORS, Security, Timeout, Performance
- **Simple search endpoints** with Pydantic v2 models
- **Advanced error handling** with circuit breakers and retry logic
- **API contract models** for MCP requests/responses

### Gaps Identified
1. **Limited API endpoints** - only search and documents implemented
2. **No API versioning strategy** beyond `/api/v1` prefix
3. **Missing webhook system** for async operations
4. **No GraphQL endpoint** implementation
5. **Basic response caching** without advanced strategies
6. **No SDK generation** or client libraries
7. **Limited batch processing** capabilities
8. **No streaming responses** for large datasets

## Implementation Strategy

### Phase 1: Enhanced API Framework (Days 1-3)

#### 1.1 Advanced FastAPI Features
```python
# src/api/core/app_enhancer.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import asyncio

class APIEnhancer:
    """Enhance FastAPI with advanced features."""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self._setup_lifecycle()
        self._setup_exception_handlers()
        self._setup_middleware_hooks()
    
    def _setup_lifecycle(self):
        """Configure advanced lifecycle management."""
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self._warmup_services()
            await self._initialize_background_tasks()
            yield
            # Shutdown
            await self._graceful_shutdown()
        
        self.app.router.lifespan_context = lifespan
    
    async def _warmup_services(self):
        """Pre-warm critical services and caches."""
        # Connection pooling warmup
        # Cache preloading
        # Service health checks
        pass
```

#### 1.2 Comprehensive Error Handling
```python
# src/api/core/error_handlers.py
from fastapi import Request, status
from fastapi.responses import JSONResponse
from src.services.errors import BaseError, ValidationError

class ErrorHandlerRegistry:
    """Centralized error handling with detailed responses."""
    
    def register_handlers(self, app: FastAPI):
        """Register all error handlers."""
        
        @app.exception_handler(ValidationError)
        async def validation_error_handler(
            request: Request, 
            exc: ValidationError
        ) -> JSONResponse:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "error": exc.message,
                    "error_code": exc.error_code,
                    "details": exc.context,
                    "request_id": request.state.request_id
                }
            )
        
        @app.exception_handler(BaseError)
        async def base_error_handler(
            request: Request,
            exc: BaseError
        ) -> JSONResponse:
            status_code = getattr(exc, 'status_code', 500)
            return JSONResponse(
                status_code=status_code,
                content=exc.to_dict()
            )
```

#### 1.3 Request/Response Validation Enhancement
```python
# src/api/core/validators.py
from pydantic import BaseModel, ConfigDict, Field
from typing import Generic, TypeVar

T = TypeVar('T')

class PaginatedRequest(BaseModel):
    """Base paginated request model."""
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    sort_by: str | None = None
    sort_order: str = Field(default="asc", pattern="^(asc|desc)$")

class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response."""
    items: list[T]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
```

#### 1.4 API Versioning Strategy
```python
# src/api/core/versioning.py
from fastapi import APIRouter, Request
from typing import Callable

class APIVersionManager:
    """Manage API versioning with backward compatibility."""
    
    def __init__(self):
        self.versions: dict[str, APIRouter] = {}
        self.deprecation_notices: dict[str, str] = {}
    
    def register_version(
        self, 
        version: str, 
        router: APIRouter,
        deprecated: bool = False,
        sunset_date: str | None = None
    ):
        """Register API version with deprecation support."""
        self.versions[version] = router
        if deprecated and sunset_date:
            self.deprecation_notices[version] = sunset_date
    
    def get_versioned_app(self, app: FastAPI):
        """Apply versioning to FastAPI app."""
        for version, router in self.versions.items():
            prefix = f"/api/{version}"
            
            # Add deprecation headers if needed
            if version in self.deprecation_notices:
                @router.middleware("http")
                async def add_deprecation_header(request: Request, call_next):
                    response = await call_next(request)
                    response.headers["Sunset"] = self.deprecation_notices[version]
                    response.headers["Deprecation"] = "true"
                    return response
            
            app.include_router(router, prefix=prefix)
```

### Phase 2: Integration Patterns (Days 4-6)

#### 2.1 Circuit Breaker Pattern Enhancement
```python
# src/api/patterns/circuit_breaker_api.py
from src.services.errors import AdvancedCircuitBreaker
from fastapi import HTTPException

class APICircuitBreaker(AdvancedCircuitBreaker):
    """API-specific circuit breaker with endpoint protection."""
    
    async def protect_endpoint(
        self, 
        func: Callable,
        *args,
        **kwargs
    ):
        """Protect API endpoint with circuit breaker."""
        try:
            return await self.call(func, *args, **kwargs)
        except ExternalServiceError as e:
            if self.state == CircuitState.OPEN:
                raise HTTPException(
                    status_code=503,
                    detail="Service temporarily unavailable",
                    headers={
                        "Retry-After": str(int(self.adaptive_timeout)),
                        "X-Circuit-State": self.state.value
                    }
                )
            raise
```

#### 2.2 Retry Logic with Exponential Backoff
```python
# src/api/patterns/retry_client.py
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

class RetryableAPIClient:
    """HTTP client with configurable retry logic."""
    
    def __init__(
        self,
        base_url: str,
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ):
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=30.0,
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=100
            )
        )
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError))
    )
    async def request(self, method: str, path: str, **kwargs):
        """Make HTTP request with retry logic."""
        response = await self.client.request(method, path, **kwargs)
        response.raise_for_status()
        return response
```

#### 2.3 Service Mesh Integration
```python
# src/api/patterns/service_mesh.py
from typing import Optional
import os

class ServiceMeshIntegration:
    """Integration with service mesh (Istio/Linkerd)."""
    
    def __init__(self):
        self.mesh_headers = self._detect_mesh_headers()
        self.enable_tracing = os.getenv("ENABLE_SERVICE_MESH_TRACING", "true") == "true"
    
    def _detect_mesh_headers(self) -> dict[str, str]:
        """Detect service mesh headers for propagation."""
        headers = {}
        
        # Istio headers
        for header in [
            "x-request-id",
            "x-b3-traceid",
            "x-b3-spanid",
            "x-b3-parentspanid",
            "x-b3-sampled",
            "x-b3-flags"
        ]:
            if value := os.getenv(header.upper().replace("-", "_")):
                headers[header] = value
        
        return headers
    
    def inject_headers(self, request_headers: dict) -> dict:
        """Inject service mesh headers for tracing."""
        return {**request_headers, **self.mesh_headers}
```

#### 2.4 Webhook System
```python
# src/api/webhooks/manager.py
from pydantic import BaseModel, HttpUrl
import asyncio
from datetime import datetime, timedelta

class WebhookConfig(BaseModel):
    """Webhook configuration model."""
    url: HttpUrl
    events: list[str]
    secret: str | None = None
    retry_policy: dict = {"max_attempts": 3, "backoff": 5}
    active: bool = True

class WebhookManager:
    """Manage webhook subscriptions and delivery."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.delivery_queue = asyncio.Queue()
        self._running = False
    
    async def register_webhook(
        self,
        webhook_id: str,
        config: WebhookConfig
    ) -> str:
        """Register new webhook subscription."""
        await self.redis.hset(
            f"webhooks:{webhook_id}",
            mapping=config.model_dump()
        )
        return webhook_id
    
    async def trigger_event(
        self,
        event_type: str,
        payload: dict
    ):
        """Trigger webhook for event."""
        # Find matching webhooks
        webhooks = await self._find_webhooks_for_event(event_type)
        
        for webhook_id, config in webhooks:
            await self.delivery_queue.put({
                "webhook_id": webhook_id,
                "config": config,
                "payload": payload,
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def start_delivery_worker(self):
        """Start webhook delivery background worker."""
        self._running = True
        while self._running:
            try:
                delivery = await asyncio.wait_for(
                    self.delivery_queue.get(),
                    timeout=1.0
                )
                asyncio.create_task(self._deliver_webhook(delivery))
            except asyncio.TimeoutError:
                continue
```

### Phase 3: Developer Experience (Days 7-9)

#### 3.1 OpenAPI/Swagger Enhancement
```python
# src/api/docs/openapi_customizer.py
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

class OpenAPICustomizer:
    """Enhance OpenAPI documentation."""
    
    def customize_openapi(self, app: FastAPI):
        """Add custom OpenAPI documentation."""
        def custom_openapi():
            if app.openapi_schema:
                return app.openapi_schema
            
            openapi_schema = get_openapi(
                title="AI Docs Vector DB API",
                version="2.0.0",
                description=self._get_enhanced_description(),
                routes=app.routes,
            )
            
            # Add custom sections
            openapi_schema["x-logo"] = {
                "url": "/static/logo.png",
                "altText": "AI Docs Logo"
            }
            
            # Add authentication schemes
            openapi_schema["components"]["securitySchemes"] = {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                },
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer"
                }
            }
            
            # Add example responses
            self._add_example_responses(openapi_schema)
            
            app.openapi_schema = openapi_schema
            return app.openapi_schema
        
        app.openapi = custom_openapi
    
    def _get_enhanced_description(self) -> str:
        """Get enhanced API description with examples."""
        return """
        # AI Docs Vector DB API
        
        Production-ready API for hybrid AI documentation search with vector database integration.
        
        ## Features
        - 🚀 Sub-100ms search latency
        - 🔄 Hybrid search (dense + sparse)
        - 📊 Real-time analytics
        - 🔐 Enterprise security
        - 📦 Batch processing
        - 🌊 Streaming responses
        
        ## Quick Start
        ```python
        import httpx
        
        client = httpx.AsyncClient(
            base_url="https://api.example.com",
            headers={"X-API-Key": "your-key"}
        )
        
        response = await client.post(
            "/api/v2/search",
            json={"query": "FastAPI best practices"}
        )
        ```
        """
```

#### 3.2 SDK Generation
```python
# src/api/sdk/generator.py
from typing import Any
import json
from pathlib import Path

class SDKGenerator:
    """Generate client SDKs from OpenAPI spec."""
    
    def __init__(self, openapi_spec: dict):
        self.spec = openapi_spec
    
    async def generate_python_sdk(self, output_dir: Path):
        """Generate Python SDK using openapi-python-client."""
        # Save OpenAPI spec
        spec_path = output_dir / "openapi.json"
        spec_path.write_text(json.dumps(self.spec))
        
        # Run generator
        import subprocess
        subprocess.run([
            "openapi-python-client",
            "generate",
            "--path", str(spec_path),
            "--output-path", str(output_dir / "python"),
            "--package-name", "aidocs_client"
        ])
        
        # Add custom enhancements
        await self._enhance_python_sdk(output_dir / "python")
    
    async def generate_typescript_sdk(self, output_dir: Path):
        """Generate TypeScript SDK using openapi-typescript."""
        spec_path = output_dir / "openapi.json"
        spec_path.write_text(json.dumps(self.spec))
        
        # Generate types
        import subprocess
        subprocess.run([
            "npx",
            "openapi-typescript",
            str(spec_path),
            "--output", str(output_dir / "typescript" / "types.ts")
        ])
        
        # Generate client
        await self._generate_ts_client(output_dir / "typescript")
```

#### 3.3 API Testing Harness
```python
# src/api/testing/harness.py
from fastapi.testclient import TestClient
from typing import Any
import pytest

class APITestHarness:
    """Comprehensive API testing utilities."""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.client = TestClient(app)
    
    def generate_test_suite(self) -> str:
        """Generate comprehensive test suite from OpenAPI."""
        tests = []
        
        for path, methods in self.app.openapi()["paths"].items():
            for method, operation in methods.items():
                if method in ["get", "post", "put", "delete", "patch"]:
                    test_name = f"test_{operation.get('operationId', path.replace('/', '_'))}"
                    test_code = self._generate_test_case(
                        path, method, operation
                    )
                    tests.append(test_code)
        
        return "\n\n".join(tests)
    
    def _generate_test_case(
        self, 
        path: str, 
        method: str, 
        operation: dict
    ) -> str:
        """Generate individual test case."""
        return f'''
@pytest.mark.asyncio
async def {operation.get('operationId', 'test')}(api_client):
    """Test {method.upper()} {path}"""
    # Arrange
    {self._generate_test_data(operation)}
    
    # Act
    response = await api_client.{method}(
        "{path}",
        {self._generate_request_params(operation)}
    )
    
    # Assert
    assert response.status_code == 200
    {self._generate_assertions(operation)}
'''
```

#### 3.4 GraphQL Endpoint
```python
# src/api/graphql/schema.py
import strawberry
from typing import Optional

@strawberry.type
class SearchResult:
    """GraphQL search result type."""
    id: str
    score: float
    title: Optional[str]
    content: Optional[str]
    url: Optional[str]
    metadata: strawberry.scalars.JSON

@strawberry.type
class Query:
    """GraphQL query root."""
    
    @strawberry.field
    async def search(
        self,
        query: str,
        limit: int = 10,
        collection: str = "documents"
    ) -> list[SearchResult]:
        """Search documents with GraphQL."""
        # Implementation
        pass
    
    @strawberry.field
    async def document(self, id: str) -> Optional[SearchResult]:
        """Get document by ID."""
        # Implementation
        pass

@strawberry.type
class Mutation:
    """GraphQL mutation root."""
    
    @strawberry.mutation
    async def index_document(
        self,
        url: str,
        force_recrawl: bool = False
    ) -> bool:
        """Index new document."""
        # Implementation
        pass

# Create GraphQL app
def create_graphql_app():
    schema = strawberry.Schema(query=Query, mutation=Mutation)
    from strawberry.fastapi import GraphQLRouter
    return GraphQLRouter(schema)
```

### Phase 4: Performance Optimization (Days 10-12)

#### 4.1 Response Caching Strategy
```python
# src/api/caching/strategies.py
from functools import wraps
import hashlib
import json

class ResponseCacheStrategy:
    """Advanced response caching with multiple strategies."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def cache_response(
        self,
        ttl: int = 300,
        vary_on: list[str] = ["query", "limit"],
        cache_errors: bool = False
    ):
        """Decorator for response caching."""
        def decorator(func):
            @wraps(func)
            async def wrapper(request: Request, *args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(
                    request, vary_on, args, kwargs
                )
                
                # Check cache
                cached = await self.redis.get(cache_key)
                if cached:
                    return JSONResponse(
                        content=json.loads(cached),
                        headers={"X-Cache": "HIT"}
                    )
                
                # Execute function
                response = await func(request, *args, **kwargs)
                
                # Cache response
                if response.status_code == 200 or cache_errors:
                    await self.redis.setex(
                        cache_key,
                        ttl,
                        json.dumps(response.body)
                    )
                
                response.headers["X-Cache"] = "MISS"
                return response
            
            return wrapper
        return decorator
```

#### 4.2 Connection Pooling
```python
# src/api/connections/pool_manager.py
from contextlib import asynccontextmanager
import asyncpg
import httpx

class ConnectionPoolManager:
    """Manage connection pools for various services."""
    
    def __init__(self, config):
        self.config = config
        self.pools = {}
    
    async def initialize(self):
        """Initialize all connection pools."""
        # Database pool
        self.pools['database'] = await asyncpg.create_pool(
            self.config.database_url,
            min_size=10,
            max_size=50,
            max_queries=50000,
            max_inactive_connection_lifetime=300
        )
        
        # HTTP client pool
        self.pools['http'] = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=100,
                max_connections=1000,
                keepalive_expiry=30
            ),
            timeout=httpx.Timeout(30.0, pool=5.0)
        )
    
    @asynccontextmanager
    async def get_connection(self, pool_name: str):
        """Get connection from pool."""
        pool = self.pools.get(pool_name)
        if pool_name == 'database':
            async with pool.acquire() as conn:
                yield conn
        else:
            yield pool
```

#### 4.3 Batch Processing Endpoints
```python
# src/api/endpoints/batch.py
from fastapi import BackgroundTasks
import asyncio

@router.post("/batch/search")
async def batch_search(
    requests: list[SearchRequest],
    background_tasks: BackgroundTasks
) -> dict:
    """Process multiple search requests in batch."""
    # Validate batch size
    if len(requests) > 100:
        raise HTTPException(400, "Batch size exceeds limit (100)")
    
    # Create batch job
    batch_id = str(uuid.uuid4())
    
    # Process in background
    background_tasks.add_task(
        process_batch_search,
        batch_id,
        requests
    )
    
    return {
        "batch_id": batch_id,
        "status": "processing",
        "size": len(requests),
        "webhook_url": f"/api/v2/batch/{batch_id}/status"
    }

async def process_batch_search(batch_id: str, requests: list[SearchRequest]):
    """Process batch search with concurrency control."""
    semaphore = asyncio.Semaphore(10)  # Limit concurrent processing
    
    async def process_single(request: SearchRequest):
        async with semaphore:
            return await search_service.search(request)
    
    # Process all requests
    results = await asyncio.gather(
        *[process_single(req) for req in requests],
        return_exceptions=True
    )
    
    # Store results
    await store_batch_results(batch_id, results)
```

#### 4.4 Streaming Responses
```python
# src/api/streaming/generators.py
from fastapi import StreamingResponse
import asyncio

class StreamingResponseGenerator:
    """Generate streaming responses for large datasets."""
    
    @staticmethod
    async def stream_search_results(
        query: str,
        chunk_size: int = 10
    ):
        """Stream search results as they become available."""
        async def generate():
            offset = 0
            while True:
                # Fetch chunk
                results = await search_service.search_chunk(
                    query, offset, chunk_size
                )
                
                if not results:
                    break
                
                # Yield results as NDJSON
                for result in results:
                    yield json.dumps(result) + "\n"
                
                offset += chunk_size
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
        
        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"  # Disable Nginx buffering
            }
        )
```

### Phase 5: Integration & Testing (Days 13-15)

#### 5.1 Integration Test Suite
```python
# tests/integration/api/test_full_stack.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
class TestAPIIntegration:
    """Full API integration tests."""
    
    async def test_search_with_caching(self, api_client: AsyncClient):
        """Test search endpoint with caching."""
        # First request - cache miss
        response1 = await api_client.post(
            "/api/v2/search",
            json={"query": "test query"}
        )
        assert response1.headers["X-Cache"] == "MISS"
        
        # Second request - cache hit
        response2 = await api_client.post(
            "/api/v2/search",
            json={"query": "test query"}
        )
        assert response2.headers["X-Cache"] == "HIT"
        assert response1.json() == response2.json()
    
    async def test_circuit_breaker_protection(self, api_client: AsyncClient):
        """Test circuit breaker functionality."""
        # Simulate failures
        for _ in range(5):
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await api_client.get("/api/v2/external-service")
            assert exc_info.value.response.status_code == 503
        
        # Circuit should be open
        response = await api_client.get("/api/v2/external-service")
        assert response.status_code == 503
        assert "Retry-After" in response.headers
    
    async def test_webhook_delivery(self, api_client: AsyncClient):
        """Test webhook system."""
        # Register webhook
        webhook_response = await api_client.post(
            "/api/v2/webhooks",
            json={
                "url": "https://example.com/webhook",
                "events": ["document.indexed", "search.completed"]
            }
        )
        webhook_id = webhook_response.json()["webhook_id"]
        
        # Trigger event
        await api_client.post(
            "/api/v2/documents",
            json={"url": "https://example.com/doc"}
        )
        
        # Verify webhook was called
        # (Mock verification in tests)
```

#### 5.2 Performance Benchmarks
```python
# tests/benchmarks/api/test_latency.py
import asyncio
import time

class APIPerformanceBenchmark:
    """Benchmark API performance metrics."""
    
    async def test_p95_latency(self, api_client):
        """Verify P95 latency < 100ms."""
        latencies = []
        
        # Run 1000 requests
        for _ in range(1000):
            start = time.time()
            await api_client.post(
                "/api/v2/search",
                json={"query": "test", "limit": 10}
            )
            latencies.append((time.time() - start) * 1000)
        
        # Calculate P95
        latencies.sort()
        p95_index = int(len(latencies) * 0.95)
        p95_latency = latencies[p95_index]
        
        assert p95_latency < 100, f"P95 latency {p95_latency}ms exceeds 100ms"
```

## Implementation Schedule

### Week 1: Core Enhancement
- Days 1-3: Enhanced API Framework
- Days 4-6: Integration Patterns

### Week 2: Developer Experience & Performance
- Days 7-9: Developer Experience
- Days 10-12: Performance Optimization

### Week 3: Testing & Deployment
- Days 13-15: Integration & Testing

## Quality Gates

### Subagent Quality Gate Process
1. **Code Quality**
   - All code passes `ruff check . --fix` and `ruff format .`
   - Type hints on all public functions
   - Google-style docstrings
   - 90%+ test coverage

2. **Performance Validation**
   - P95 latency < 100ms verified
   - Load tests pass (1000 RPS)
   - Memory usage stable under load
   - Connection pools properly sized

3. **API Contract Compliance**
   - OpenAPI spec validates
   - All endpoints documented
   - Examples for all operations
   - SDK generation succeeds

4. **Integration Testing**
   - All patterns working (circuit breaker, retry, etc.)
   - Webhook delivery verified
   - GraphQL queries functional
   - Streaming responses work

## Success Metrics

### Technical Metrics
- **P95 Latency**: < 100ms ✓
- **Throughput**: > 1000 RPS ✓
- **Error Rate**: < 0.1% ✓
- **API Coverage**: 100% endpoints documented ✓

### Developer Experience Metrics
- **SDK Languages**: Python, TypeScript, Go ✓
- **Documentation**: Interactive Swagger UI ✓
- **Testing**: Automated test generation ✓
- **GraphQL**: Full query/mutation support ✓

## Risk Mitigation

### Performance Risks
- **Mitigation**: Extensive caching, connection pooling, async processing

### Backward Compatibility
- **Mitigation**: Versioning strategy with deprecation notices

### Integration Complexity
- **Mitigation**: Comprehensive integration tests, monitoring

## Next Steps

1. Begin Phase 1 implementation
2. Set up performance monitoring
3. Create SDK generation pipeline
4. Document all new endpoints
5. Prepare migration guide for v1 → v2

## Dependencies

- FastAPI 0.115.6+
- Pydantic 2.10.6+
- httpx 0.28.2+
- strawberry-graphql (for GraphQL)
- openapi-python-client (SDK generation)