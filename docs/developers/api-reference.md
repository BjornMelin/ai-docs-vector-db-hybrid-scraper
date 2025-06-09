# API Reference

> **Status**: Active  
> **Last Updated**: 2025-01-09  
> **Purpose**: Complete API reference for all system interfaces  
> **Audience**: Developers integrating with or contributing to the system

This comprehensive API reference covers all interfaces in the AI Documentation Vector DB system: REST APIs, Browser Automation APIs, MCP Tools, and data models.

## 🚀 Quick API Start

### Core APIs Available

- **REST API**: HTTP endpoints for search, documents, and collections
- **Browser API**: 5-tier browser automation with intelligent routing
- **MCP Tools**: 25+ tools for Claude Desktop/Code integration
- **Python SDK**: Direct programmatic access to all services

### Fast Start Example

```python
# Python SDK usage
from src.config import get_config
from src.services import EmbeddingManager, QdrantService

config = get_config()
async with QdrantService(config) as qdrant:
    results = await qdrant.search_vectors(
        collection_name="documents",
        query="vector database optimization",
        limit=10
    )
```

## 📡 REST API Reference

### Base Configuration

```bash
# API Base URL
BASE_URL=http://localhost:8000/api/v1

# Authentication (when enabled)
Authorization: Bearer <your-api-key>
Content-Type: application/json
```

### Search Endpoints

#### POST /search

Basic semantic search with hybrid vector search.

```bash
curl -X POST $BASE_URL/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "vector database optimization",
    "collection_name": "documents",
    "limit": 10,
    "score_threshold": 0.7,
    "enable_hyde": false
  }'
```

**Request Model:**

```python
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    collection_name: str = Field("documents", description="Target collection")
    limit: int = Field(10, ge=1, le=100, description="Result limit")
    score_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Min score")
    enable_hyde: bool = Field(False, description="Enable HyDE expansion")
    filters: dict = Field({}, description="Optional metadata filters")
```

**Response Model:**

```python
class SearchResponse(BaseModel):
    success: bool
    timestamp: float
    results: list[SearchResultItem]
    total_count: int
    query_time_ms: float
    search_strategy: str  # "dense", "sparse", "hybrid"
    cache_hit: bool
```

#### POST /search/advanced

Advanced search with multiple strategies and reranking.

```bash
curl -X POST $BASE_URL/search/advanced \
  -H "Content-Type: application/json" \
  -d '{
    "query": "advanced search techniques",
    "search_strategy": "hybrid",
    "accuracy_level": "balanced",
    "enable_reranking": true,
    "hyde_config": {"temperature": 0.7},
    "limit": 20
  }'
```

**Request Model:**

```python
class AdvancedSearchRequest(BaseModel):
    query: str
    search_strategy: str = Field("hybrid", enum=["dense", "sparse", "hybrid"])
    accuracy_level: str = Field("balanced", enum=["fast", "balanced", "accurate", "exact"])
    enable_reranking: bool = Field(True)
    hyde_config: dict = Field({})
    limit: int = Field(20, ge=1, le=100)
```

### Document Management

#### POST /documents

Add single document to index.

```bash
curl -X POST $BASE_URL/documents \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://docs.example.com/api",
    "collection_name": "api_docs",
    "doc_type": "api_reference",
    "metadata": {"version": "v1.0", "tags": ["api"]},
    "force_recrawl": false
  }'
```

#### POST /documents/bulk

Add multiple documents in batch.

```bash
curl -X POST $BASE_URL/documents/bulk \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://docs.example.com/guide1",
      "https://docs.example.com/guide2"
    ],
    "max_concurrent": 5,
    "collection_name": "guides",
    "force_recrawl": false
  }'
```

#### GET /documents/{document_id}

Retrieve document by ID.

#### DELETE /documents/{document_id}

Remove document from index.

### Collection Management

#### GET /collections

List all collections with statistics.

```bash
curl $BASE_URL/collections
```

**Response:**

```json
{
  "success": true,
  "collections": [
    {
      "name": "documents",
      "points_count": 1500,
      "vectors_count": 1500,
      "indexed_fields": ["title", "content", "url"],
      "status": "green"
    }
  ]
}
```

#### POST /collections

Create new collection.

```bash
curl -X POST $BASE_URL/collections \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "technical_docs",
    "vector_size": 1536,
    "distance_metric": "Cosine",
    "enable_hybrid": true,
    "hnsw_config": {"m": 16, "ef_construct": 200}
  }'
```

#### DELETE /collections/{collection_name}

Delete collection and all documents.

### Analytics

#### GET /analytics/usage

Get usage statistics.

```bash
curl "$BASE_URL/analytics/usage?time_range=24h&metric_types=searches,documents"
```

#### GET /analytics/performance

Get performance metrics.

## 🌐 Browser Automation API

### 5-Tier Architecture Overview

The browser automation system implements intelligent 5-tier routing:

1. **Tier 0: Lightweight HTTP** - httpx + BeautifulSoup (5-10x faster for static content)
2. **Tier 1: Crawl4AI Basic** - Standard browser automation for dynamic content
3. **Tier 2: Crawl4AI Enhanced** - Interactive content with custom JavaScript
4. **Tier 3: Browser-use AI** - Complex interactions with AI-powered automation
5. **Tier 4: Playwright + Firecrawl** - Maximum control + API fallback

### UnifiedBrowserManager

#### Core Scraping Method

```python
async def scrape(
    request: UnifiedScrapingRequest | None = None,
    url: str | None = None,
    **kwargs
) -> UnifiedScrapingResponse
```

**Basic Usage:**

```python
from src.services.browser.unified_manager import UnifiedBrowserManager, UnifiedScrapingRequest

# Simple scraping (automatic tier selection)
manager = UnifiedBrowserManager(config)
await manager.initialize()

response = await manager.scrape(url="https://docs.example.com")
print(f"Success: {response.success}")
print(f"Tier used: {response.tier_used}")
print(f"Content length: {response.content_length}")
```

**Structured Request:**

```python
request = UnifiedScrapingRequest(
    url="https://complex-spa.com",
    tier="browser_use",  # Force specific tier
    interaction_required=True,
    custom_actions=[
        {"type": "wait_for_selector", "selector": ".dynamic-content"},
        {"type": "click", "selector": "#load-more"},
        {"type": "extract", "target": "documentation"}
    ],
    timeout=30000,
    wait_for_selector=".content",
    extract_metadata=True
)

response = await manager.scrape(request)
```

#### URL Analysis

```python
async def analyze_url(url: str) -> dict
```

Analyze URL to determine optimal tier and provide performance insights.

```python
analysis = await manager.analyze_url("https://docs.example.com")
# Returns:
{
    "url": "https://docs.example.com",
    "domain": "docs.example.com",
    "recommended_tier": "crawl4ai",
    "expected_performance": {
        "estimated_time_ms": 1500.0,
        "success_rate": 0.95
    }
}
```

#### System Status

```python
def get_system_status() -> dict
```

Get comprehensive system health and performance information.

```python
status = manager.get_system_status()
# Returns detailed status including:
# - Overall health, success rates
# - Tier-specific metrics
# - Cache performance
# - Monitoring status
```

### Tier-Specific APIs

#### Tier 0: Lightweight HTTP

**Best For:** Static content, documentation sites, API endpoints

```python
response = await manager.scrape(
    url="https://docs.python.org/3/tutorial/",
    tier="lightweight"
)
```

**Performance:**

- 5-10x faster than browser-based tiers
- 95%+ success rate for static content
- Minimal resource usage

#### Tier 1: Crawl4AI Basic

**Best For:** Standard dynamic content, most documentation sites

```python
response = await manager.scrape(
    url="https://react.dev/learn",
    tier="crawl4ai"
)
```

#### Tier 2: Crawl4AI Enhanced

**Best For:** Interactive content, SPAs with custom JavaScript

```python
response = await manager.scrape(
    url="https://interactive-docs.com",
    tier="crawl4ai_enhanced",
    custom_actions=[
        {"type": "execute_js", "script": "expandAllSections()"},
        {"type": "wait", "duration": 2000}
    ]
)
```

#### Tier 3: Browser-use AI

**Best For:** Complex interactions requiring AI reasoning

```python
request = UnifiedScrapingRequest(
    url="https://complex-dashboard.com",
    tier="browser_use",
    interaction_required=True,
    custom_actions=[
        {
            "type": "ai_task",
            "instruction": "Navigate to documentation section and extract all API endpoints"
        }
    ]
)
response = await manager.scrape(request)
```

#### Tier 4: Playwright + Firecrawl

**Best For:** Maximum control, authentication, complex workflows

```python
request = UnifiedScrapingRequest(
    url="https://authenticated-site.com",
    tier="playwright",
    interaction_required=True,
    custom_actions=[
        {"type": "fill", "selector": "#username", "value": "demo"},
        {"type": "fill", "selector": "#password", "value": "demo"},
        {"type": "click", "selector": "#login"},
        {"type": "wait_for_selector", "selector": ".dashboard"},
        {"type": "extract_content", "selector": ".documentation"}
    ]
)
```

### Advanced Features

#### Caching System

```python
from src.services.cache.browser_cache import BrowserCache

cache = BrowserCache(
    default_ttl=3600,
    dynamic_content_ttl=300,    # Short TTL for dynamic content
    static_content_ttl=86400,   # Long TTL for static content
)

# Cache stats
stats = cache.get_stats()
# Returns hit rates, entry counts, size metrics
```

#### Rate Limiting

```python
from src.services.browser.tier_rate_limiter import TierRateLimiter, RateLimitContext

# Acquire rate limit permission
async with RateLimitContext(rate_limiter, "browser_use") as allowed:
    if allowed:
        result = await perform_scraping()
    else:
        wait_time = rate_limiter.get_wait_time("browser_use")
        await asyncio.sleep(wait_time)
```

#### Monitoring and Alerting

```python
from src.services.browser.monitoring import BrowserAutomationMonitor

monitor = BrowserAutomationMonitor(config)
await monitor.start_monitoring()

# Record metrics
await monitor.record_request_metrics(
    tier="crawl4ai",
    success=True,
    response_time_ms=1500.0,
    cache_hit=False
)

# Get system health
health = monitor.get_system_health()
alerts = monitor.get_active_alerts()
```

## 🔌 MCP Tools Reference

### Setup and Configuration

#### Basic Claude Desktop Configuration

Add to your Claude Desktop config file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`  
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`  
**Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "ai-docs-vector-db": {
      "command": "uv",
      "args": ["run", "python", "src/unified_mcp_server.py"],
      "cwd": "/absolute/path/to/ai-docs-vector-db-hybrid-scraper",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

#### Advanced Production Configuration

```json
{
  "mcpServers": {
    "ai-docs-vector-db": {
      "command": "uv",
      "args": ["run", "python", "src/unified_mcp_server.py"],
      "cwd": "/absolute/path/to/ai-docs-vector-db-hybrid-scraper",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "FIRECRAWL_API_KEY": "fc-...",
        "QDRANT_URL": "http://localhost:6333",
        "REDIS_URL": "redis://localhost:6379",
        "LOG_LEVEL": "INFO",
        "ENABLE_CACHE": "true",
        "CACHE_TTL": "3600"
      }
    }
  }
}
```

### Available MCP Tools

#### Search Tools

- **`search_documents`** - Hybrid vector search with reranking
- **`search_by_collection`** - Search within specific collections
- **`search_similar`** - Find similar documents

#### Document Management - `document_management`

- **`add_url`** - Add single URL to index
- **`add_urls`** - Bulk URL addition
- **`update_document`** - Update existing documents
- **`delete_document`** - Remove documents

#### Collection Management - `collection_management`

- **`list_collections`** - Show all collections
- **`create_collection`** - Create new collection
- **`delete_collection`** - Remove collection
- **`get_collection_stats`** - Collection metrics

#### Project Management - `project_management`

- **`create_project`** - Initialize new project
- **`list_projects`** - Show all projects
- **`update_project`** - Modify project settings
- **`delete_project`** - Remove project

#### Analytics - `analytics`

- **`get_usage_stats`** - API usage metrics
- **`get_performance_metrics`** - Search performance
- **`get_cache_stats`** - Cache hit rates

### Testing MCP Setup

#### Verify Server Startup

1. Open Claude Desktop
2. Start a new conversation
3. Type: "Can you list my vector collections?"
4. Claude should use the `list_collections` tool

#### Test Search Functionality

```plaintext
You: "Search for documentation about authentication"
Claude: [Uses search_documents tool with your query]
```

#### Test Document Addition

```plaintext
You: "Add https://docs.example.com to my documentation index"
Claude: [Uses add_url tool to crawl and index the page]
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key for embeddings |
| `FIRECRAWL_API_KEY` | No | - | Firecrawl API key for premium features |
| `QDRANT_URL` | No | <http://localhost:6333> | Qdrant database URL |
| `REDIS_URL` | No | - | Redis URL for caching |
| `LOG_LEVEL` | No | INFO | Logging level |
| `ENABLE_CACHE` | No | true | Enable caching layer |
| `CACHE_TTL` | No | 3600 | Cache TTL in seconds |

## 📊 Data Models Reference

### Configuration Models

#### UnifiedConfig

Central configuration model with Pydantic v2 validation.

```python
from src.config import UnifiedConfig, get_config

# Load configuration
config = get_config()

# Access sections
embedding_config = config.embedding
qdrant_config = config.qdrant
security_config = config.security
```

#### Configuration Enums

```python
from src.config.enums import (
    Environment, EmbeddingProvider, EmbeddingModel,
    SearchStrategy, VectorType, ChunkingStrategy
)

# Environment types
Environment.DEVELOPMENT
Environment.TESTING  
Environment.PRODUCTION

# Embedding providers
EmbeddingProvider.OPENAI     # "openai"
EmbeddingProvider.FASTEMBED  # "fastembed"

# Search strategies
SearchStrategy.DENSE   # "dense"
SearchStrategy.SPARSE  # "sparse" 
SearchStrategy.HYBRID  # "hybrid"
```

### API Contract Models

#### SearchRequest

```python
from src.models.api_contracts import SearchRequest

request = SearchRequest(
    query="vector database optimization",
    collection_name="documents",           # Default: "documents"
    limit=10,                             # Range: 1-100
    score_threshold=0.7,                  # Range: 0.0-1.0
    enable_hyde=False,                    # HyDE query expansion
    filters={"category": "technical"}     # Optional filters
)
```

#### SearchResponse

```python
{
    "success": True,
    "timestamp": 1641024000.0,
    "results": [SearchResultItem(...)],
    "total_count": 5,
    "query_time_ms": 150.0,
    "search_strategy": "hybrid",
    "cache_hit": True
}
```

#### DocumentRequest

```python
from src.models.api_contracts import DocumentRequest

request = DocumentRequest(
    url="https://docs.example.com/api",
    collection_name="api_docs",
    doc_type="api_reference",
    metadata={"version": "v1.0", "tags": ["api"]},
    force_recrawl=False
)
```

### Document Processing Models

#### Chunk

```python
from src.models.document_processing import Chunk, ChunkType

chunk = Chunk(
    content="This is a chunk of content...",
    chunk_index=0,
    chunk_type=ChunkType.TEXT,
    start_index=0,
    end_index=500,
    metadata={
        "source_url": "https://docs.example.com",
        "section": "introduction"
    }
)
```

#### ChunkType Enum

```python
ChunkType.TEXT       # "text"
ChunkType.CODE       # "code"  
ChunkType.HEADING    # "heading"
ChunkType.LIST       # "list"
ChunkType.TABLE      # "table"
```

### Vector Search Models

#### SearchParams

```python
from src.models.vector_search import SearchParams, VectorType

params = SearchParams(
    vector=embeddings,                    # List[float]
    limit=10,
    score_threshold=0.7,
    vector_name="default",               # For named vectors
    with_payload=True,
    with_vectors=False
)
```

#### FusionConfig

```python
from src.models.vector_search import FusionConfig, FusionAlgorithm

fusion = FusionConfig(
    algorithm=FusionAlgorithm.RRF,       # RRF (Reciprocal Rank Fusion)
    dense_weight=0.7,                    # Range: 0.0-1.0
    sparse_weight=0.3,                   # Range: 0.0-1.0
    rrf_k=60                             # RRF parameter
)
```

## 🔧 Service APIs

### EmbeddingManager

```python
from src.services.embeddings import EmbeddingManager

async with EmbeddingManager(config) as embeddings:
    # Generate dense embeddings
    dense_vectors = await embeddings.generate_embeddings(
        texts=["Hello world", "Vector search"],
        quality_tier="BALANCED"
    )
    
    # Generate sparse embeddings (if supported)
    sparse_vectors = await embeddings.generate_sparse_embeddings(
        texts=["Hello world"]
    )
    
    # Get embedding dimensions
    dimensions = embeddings.get_embedding_dimensions()
```

### QdrantService

```python
from src.services.vector_db import QdrantService

async with QdrantService(config) as qdrant:
    # Create collection
    await qdrant.create_collection(
        collection_name="documents",
        vector_size=1536,
        distance="Cosine"
    )
    
    # Hybrid search
    results = await qdrant.hybrid_search(
        collection_name="documents",
        query_vector=dense_vector,
        sparse_vector=sparse_vector,
        limit=10
    )
    
    # Add documents
    await qdrant.add_documents(
        collection_name="documents",
        documents=[doc1, doc2, doc3]
    )
```

### CrawlManager

```python
from src.services.crawling import CrawlManager

async with CrawlManager(config) as crawler:
    # Crawl single URL
    result = await crawler.crawl_url(
        url="https://docs.example.com",
        max_depth=2
    )
    
    # Bulk crawl site
    results = await crawler.crawl_site(
        base_url="https://docs.example.com",
        max_pages=100
    )
```

### CacheManager

```python
from src.services.cache import CacheManager

async with CacheManager(config) as cache:
    # Get or compute value
    result = await cache.get_or_compute(
        key="embeddings:hello_world",
        compute_fn=lambda: generate_embeddings(["Hello world"]),
        ttl=3600
    )
    
    # Cache search results
    await cache.cache_search_results(
        query="vector search",
        results=search_results,
        ttl=1800
    )
```

## ✅ Validation Functions

### API Key Validation

```python
from src.models.validators import (
    validate_api_key_common,
    openai_api_key_validator,
    firecrawl_api_key_validator
)

# Validate OpenAI API key
key = openai_api_key_validator("sk-1234567890abcdef")

# Validate Firecrawl API key  
key = firecrawl_api_key_validator("fc-abcdefghijklmnop")

# Generic API key validation
key = validate_api_key_common("sk-test", "sk-", "OpenAI")
```

### URL and String Validation

```python
from src.models.validators import (
    validate_url_format,
    validate_collection_name,
    validate_positive_int
)

# URL validation
url = validate_url_format("https://docs.example.com")

# Collection name validation (alphanumeric, hyphens, underscores)
name = validate_collection_name("my-collection_v1")

# Positive integer validation
count = validate_positive_int(42)
```

### Configuration Validation

```python
from src.models.validators import (
    validate_chunk_sizes,
    validate_rate_limit_config,
    validate_vector_dimensions
)

# Validate chunk configuration
validate_chunk_sizes(chunk_size=1600, chunk_overlap=200)

# Validate rate limiting
validate_rate_limit_config({
    "requests_per_minute": 60,
    "burst_limit": 10
})

# Validate vector dimensions
validate_vector_dimensions(1536, min_dim=1, max_dim=4096)
```

## 🚨 Error Handling

### Standard Error Response

All API endpoints return standardized error responses:

```python
{
    "success": False,
    "timestamp": 1641024000.0,
    "error": "Detailed error message",
    "error_type": "validation_error",
    "context": {
        "field": "query",
        "value": "",
        "constraint": "non_empty_string"
    }
}
```

### Common Error Types

- **`validation_error`**: Input validation failures
- **`authentication_error`**: Invalid API keys
- **`rate_limit_error`**: Rate limit exceeded
- **`service_error`**: External service failures
- **`configuration_error`**: Invalid configuration
- **`network_error`**: Network connectivity issues

### Exception Hierarchy

```python
class ServiceError(Exception):
    """Base exception for service errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}

class ValidationError(ServiceError):
    """Raised when input validation fails."""
    pass

class ConfigurationError(ServiceError):
    """Raised when configuration is invalid."""
    pass

class ExternalServiceError(ServiceError):
    """Raised when external service calls fail."""
    pass
```

## 📝 Usage Examples

### Complete Search Workflow

```python
from src.config import get_config
from src.services import EmbeddingManager, QdrantService
from src.models.api_contracts import SearchRequest

# Load configuration
config = get_config()

# Initialize services
async with EmbeddingManager(config) as embeddings, \
           QdrantService(config) as qdrant:
    
    # Create search request
    request = SearchRequest(
        query="vector database optimization",
        limit=10,
        score_threshold=0.7
    )
    
    # Generate query embedding
    query_vector = await embeddings.generate_embeddings([request.query])
    
    # Perform search
    results = await qdrant.search_vectors(
        collection_name=request.collection_name,
        query_vector=query_vector[0],
        limit=request.limit,
        score_threshold=request.score_threshold
    )
    
    # Format response
    response = SearchResponse(
        success=True,
        timestamp=time.time(),
        results=results,
        total_count=len(results),
        search_strategy="dense"
    )
```

### Document Processing Pipeline

```python
from src.services import CrawlManager, EmbeddingManager, QdrantService
from src.models.api_contracts import DocumentRequest

async def process_document(request: DocumentRequest):
    config = get_config()
    
    async with CrawlManager(config) as crawler, \
               EmbeddingManager(config) as embeddings, \
               QdrantService(config) as qdrant:
        
        # Crawl document
        crawl_result = await crawler.crawl_url(request.url)
        
        if not crawl_result["success"]:
            return DocumentResponse(
                success=False,
                error=f"Failed to crawl {request.url}",
                timestamp=time.time()
            )
        
        # Process and chunk content
        processed_doc = await process_crawl_result(crawl_result, request)
        
        # Generate embeddings for chunks
        chunk_texts = [chunk.content for chunk in processed_doc.chunks]
        embeddings_result = await embeddings.generate_embeddings(chunk_texts)
        
        # Store in vector database
        await qdrant.add_document(
            collection_name=request.collection_name,
            document=processed_doc,
            embeddings=embeddings_result
        )
        
        return DocumentResponse(
            success=True,
            timestamp=time.time(),
            document_id=processed_doc.metadata.content_hash,
            url=request.url,
            chunks_created=len(processed_doc.chunks),
            status="processed"
        )
```

### Production-Ready Browser Scraping

```python
from src.config import UnifiedConfig
from src.services.browser.unified_manager import UnifiedBrowserManager

# Load configuration
config = UnifiedConfig()

# Initialize with custom settings
config.cache.enable_browser_cache = True
config.cache.browser_cache_ttl = 3600
config.performance.enable_monitoring = True
config.performance.enable_rate_limiting = True

# Create and initialize manager
manager = UnifiedBrowserManager(config)
await manager.initialize()

# Production-ready scraping
try:
    response = await manager.scrape(
        url="https://production-docs.com",
        tier="auto",  # Let system choose optimal tier
        extract_metadata=True
    )
    
    if response.success:
        print(f"Success! Used {response.tier_used} tier")
        print(f"Content length: {response.content_length}")
        print(f"Quality score: {response.quality_score:.2f}")
        print(f"Execution time: {response.execution_time_ms:.1f}ms")
    else:
        print(f"Failed: {response.error}")
        print(f"Failed tiers: {response.failed_tiers}")
        
finally:
    await manager.cleanup()
```

## 📊 Performance and Monitoring

### Performance Optimization

#### Connection Pooling

The system uses optimized connection pooling:

- Qdrant: 10 connections
- Redis: 20 connections  
- HTTP: 100 connections

#### Batch Processing

```python
# Enable batch operations
config.performance.enable_batch_processing = True
config.performance.batch_size = 32
```

#### Resource Limits

```python
# Set appropriate limits
config.performance.max_memory_mb = 2048
config.performance.request_timeout = 30
```

### Monitoring and Alerting - `monitoring`

#### System Health Monitoring

```python
# Regular performance monitoring
async def monitor_system_health():
    status = manager.get_system_status()
    
    if status["overall_success_rate"] < 0.9:
        logger.warning("System performance degraded")
        
    if status["cache_stats"]["hit_rate"] < 0.7:
        logger.info("Consider cache optimization")
        
    # Check tier-specific metrics
    for tier, metrics in status["tier_metrics"].items():
        if metrics["success_rate"] < 0.8:
            logger.warning(f"Tier {tier} performance issues")
```

#### Cache Performance

```python
# Cache performance monitoring
cache_stats = manager._browser_cache.get_stats()
# Returns:
{
    "hit_rate": 0.78,
    "total_entries": 1250,
    "total_size_mb": 45.2,
    "avg_ttl_seconds": 3600,
    "eviction_count": 23
}
```

## 🛠️ Best Practices

### API Usage Patterns

#### Error Handling

```python
async def robust_api_request(url: str) -> dict:
    """Production-ready API request with comprehensive error handling."""
    try:
        response = await manager.scrape(url)
        
        if not response.success:
            logger.error(f"Request failed for {url}: {response.error}")
            
            # Implement custom fallback logic if needed
            if "authentication" in response.error.lower():
                return await handle_auth_required(url)
            
        return response
        
    except Exception as e:
        logger.exception(f"Unexpected error for {url}")
        return UnifiedScrapingResponse(
            success=False,
            error=str(e),
            url=url,
            tier_used="none",
            execution_time_ms=0,
            content_length=0
        )
```

#### Resource Management

```python
# Always use context managers or try/finally
async def safe_api_session():
    manager = UnifiedBrowserManager(config)
    
    try:
        await manager.initialize()
        
        # Perform API operations
        results = []
        for url in urls:
            result = await manager.scrape(url)
            results.append(result)
            
        return results
        
    finally:
        await manager.cleanup()  # Essential for resource cleanup
```

#### Batch Operations

```python
# Efficient bulk processing with concurrency control
async def efficient_bulk_processing(urls: list[str]):
    manager = UnifiedBrowserManager(config)
    await manager.initialize()
    
    try:
        # Batch process with concurrency control
        semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
        
        async def process_single(url):
            async with semaphore:
                return await manager.scrape(url)
        
        tasks = [process_single(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    finally:
        await manager.cleanup()
```

### Security Best Practices

#### Input Validation

```python
# Always validate inputs
request = SearchRequest.model_validate(user_input)

# Check for malicious URLs
if not validate_url_format(request.url):
    raise ValidationError("Invalid URL format")
```

#### Rate Limiting - `rate_limiting`

```python
# Implement rate limiting for external APIs
async with RateLimitContext(rate_limiter, "api_calls") as allowed:
    if not allowed:
        raise RateLimitError("Rate limit exceeded")
    
    result = await perform_api_call()
```

#### Secret Management

```python
# Never log secrets
logger.info(f"Using API key: {api_key[:8]}...")  # Only log prefix

# Use environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ConfigurationError("API key not configured")
```

---

*📚 This comprehensive API reference provides complete documentation for all system interfaces. For implementation examples and advanced usage patterns, refer to the test suite and service implementations.*
