# API Reference Documentation

This document provides comprehensive API reference for all models, services, and interfaces in the vector knowledge base system.

## Table of Contents

- [Configuration Models](#configuration-models)
- [API Contract Models](#api-contract-models)  
- [Document Processing Models](#document-processing-models)
- [Vector Search Models](#vector-search-models)
- [MCP Protocol Models](#mcp-protocol-models)
- [Service APIs](#service-apis)
- [Validation Functions](#validation-functions)

## Configuration Models

### UnifiedConfig

Central configuration model that unifies all system settings with Pydantic v2 validation.

```python
from src.config import UnifiedConfig, get_config

# Load configuration from environment and files
config = get_config()

# Access configuration sections
embedding_config = config.embedding
qdrant_config = config.qdrant
security_config = config.security
```

**Key Sections:**

- `embedding`: Embedding provider settings (OpenAI, FastEmbed)
- `qdrant`: Vector database configuration  
- `crawl4ai`: Web crawling settings
- `firecrawl`: Premium crawling configuration
- `openai`: OpenAI API settings
- `cache`: Caching configuration (DragonflyDB, local)
- `security`: Security validation settings
- `chunking`: Document chunking configuration
- `hyde`: HyDE search implementation

### Configuration Enums

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

## API Contract Models

### Request Models

#### SearchRequest

Basic search request with validation.

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

#### AdvancedSearchRequest

Extended search with advanced options.

```python
from src.models.api_contracts import AdvancedSearchRequest

request = AdvancedSearchRequest(
    query="advanced search techniques",
    search_strategy="hybrid",             # dense, sparse, hybrid
    accuracy_level="balanced",            # fast, balanced, accurate, exact
    enable_reranking=True,
    hyde_config={"temperature": 0.7},
    limit=20
)
```

#### DocumentRequest

Single document addition request.

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

#### BulkDocumentRequest

Batch document processing request.

```python
from src.models.api_contracts import BulkDocumentRequest

request = BulkDocumentRequest(
    urls=[
        "https://docs.example.com/guide1",
        "https://docs.example.com/guide2"
    ],
    max_concurrent=5,                     # Range: 1-20
    collection_name="guides",
    force_recrawl=False
)
```

### Response Models

#### SearchResponse

Search results with metadata.

```python
from src.models.api_contracts import SearchResponse, SearchResultItem

# Response structure
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

#### SearchResultItem

Individual search result.

```python
SearchResultItem(
    id="doc_123",
    score=0.95,
    title="Vector Database Optimization",
    content="This document covers...",
    url="https://docs.example.com/optimization",
    doc_type="guide",
    language="en",
    metadata={"tags": ["performance", "optimization"]}
)
```

#### DocumentResponse

Document processing result.

```python
DocumentResponse(
    success=True,
    timestamp=1641024000.0,
    document_id="doc_123",
    url="https://docs.example.com/guide",
    chunks_created=15,
    processing_time_ms=2500.0,
    status="processed"
)
```

### Collection Management

#### CollectionRequest

Collection creation/modification.

```python
from src.models.api_contracts import CollectionRequest

request = CollectionRequest(
    collection_name="technical_docs",
    vector_size=1536,                    # Must be > 0
    distance_metric="Cosine",           # Cosine, Dot, Euclidean
    enable_hybrid=True,
    hnsw_config={"m": 16, "ef_construct": 200}
)
```

#### CollectionInfo

Collection metadata and statistics.

```python
CollectionInfo(
    name="technical_docs",
    points_count=1500,
    vectors_count=1500,
    indexed_fields=["title", "content", "url"],
    status="green",
    config={"vector_size": 1536, "distance": "Cosine"}
)
```

### Analytics Models

#### AnalyticsRequest

Analytics query request.

```python
from src.models.api_contracts import AnalyticsRequest

request = AnalyticsRequest(
    collection_name="documents",
    time_range="24h",                   # 24h, 7d, 30d
    metric_types=["searches", "documents", "performance"]
)
```

#### MetricData

Individual metric data point.

```python
MetricData(
    name="search_latency_p95",
    value=150.5,
    unit="ms",
    timestamp=1641024000.0
)
```

## Document Processing Models

### Chunk

Document chunk with metadata.

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

### ChunkType Enum

```python
ChunkType.TEXT       # "text"
ChunkType.CODE       # "code"  
ChunkType.HEADING    # "heading"
ChunkType.LIST       # "list"
ChunkType.TABLE      # "table"
```

### DocumentMetadata

Document metadata with timestamps.

```python
from src.models.document_processing import DocumentMetadata

metadata = DocumentMetadata(
    url="https://docs.example.com/guide",
    title="Getting Started Guide",
    description="Comprehensive getting started guide",
    language="en",
    doc_type="tutorial",
    crawled_at=datetime.now(),  # Auto-generated
    content_hash="sha256:abc123...",
    metadata_tags=["beginner", "tutorial"]
)
```

### ProcessedDocument

Complete processed document.

```python
ProcessedDocument(
    url="https://docs.example.com/guide",
    title="Guide Title",
    content="Full document content...",
    chunks=[chunk1, chunk2, chunk3],
    metadata=document_metadata,
    processing_stats={
        "total_chunks": 15,
        "processing_time_ms": 2500.0,
        "content_length": 25000
    }
)
```

## Vector Search Models

### SearchParams

Basic search parameters.

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

### SearchStage

Multi-stage search configuration.

```python
from src.models.vector_search import SearchStage

stage = SearchStage(
    stage_name="prefetch",
    vector_type=VectorType.DENSE,
    limit=100,
    score_threshold=0.5
)
```

### FusionConfig

Hybrid search fusion configuration.

```python
from src.models.vector_search import FusionConfig, FusionAlgorithm

fusion = FusionConfig(
    algorithm=FusionAlgorithm.RRF,       # RRF (Reciprocal Rank Fusion)
    dense_weight=0.7,                    # Range: 0.0-1.0
    sparse_weight=0.3,                   # Range: 0.0-1.0
    rrf_k=60                             # RRF parameter
)
```

### PrefetchConfig

Prefetch optimization settings.

```python
PrefetchConfig(
    enable_prefetch=True,
    dense_multiplier=2.0,               # Prefetch 2x more dense results
    sparse_multiplier=5.0,              # Prefetch 5x more sparse results
    max_prefetch_limit=500              # Hard limit
)
```

## MCP Protocol Models

### MCPRequest/MCPResponse

Base MCP protocol models.

```python
from src.mcp.models.requests import MCPRequest
from src.mcp.models.responses import MCPResponse

# All MCP requests inherit from MCPRequest
class MyToolRequest(MCPRequest):
    parameter: str
    
# All MCP responses inherit from MCPResponse  
class MyToolResponse(MCPResponse):
    result: str
    
# MCPResponse includes:
# - success: bool
# - timestamp: float
# - error: Optional[str]
```

## Service APIs

### EmbeddingManager

Embedding generation service.

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

Vector database operations.

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

Web crawling service.

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

Caching service with DragonflyDB.

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

## Validation Functions

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

## Error Handling

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

- `validation_error`: Input validation failures
- `authentication_error`: Invalid API keys
- `rate_limit_error`: Rate limit exceeded
- `service_error`: External service failures
- `configuration_error`: Invalid configuration
- `network_error`: Network connectivity issues

## Usage Examples

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

This API reference provides comprehensive coverage of all models, services, and interfaces available in the vector knowledge base system. All models include full Pydantic v2 validation with detailed error messages and type safety.