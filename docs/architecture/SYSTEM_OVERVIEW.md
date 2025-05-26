# System Architecture Overview

**Status**: Current  
**Last Updated**: 2025-05-26

## Overview

AI Documentation Vector DB is a production-ready hybrid scraping system that combines bulk processing with on-demand retrieval, powered by advanced vector search and modern service architecture.

## Core Architecture

### High-Level Component Flow

```
Documentation URLs → Crawl4AI/Firecrawl → Enhanced Chunking → 
→ Embedding Pipeline → Qdrant Vector DB → MCP Server → Claude Desktop
```

### Service Layer Architecture

The system implements a clean service layer pattern with dependency injection:

```
┌─────────────────────────────────────────────────────────────┐
│                    Unified MCP Server                        │
│                 (FastMCP 2.0 Framework)                      │
├─────────────────────────────────────────────────────────────┤
│                    Service Layer                             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │EmbeddingMgr │  │ QdrantService│  │  CrawlManager   │   │
│  └─────────────┘  └──────────────┘  └─────────────────┘   │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │CacheManager │  │ProjectStorage│  │ SecurityValidator│   │
│  └─────────────┘  └──────────────┘  └─────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│              Unified Configuration (Pydantic v2)             │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Unified Configuration System
- **Location**: `src/config/`
- **Pattern**: Centralized Pydantic v2 models
- **Features**: 
  - Environment variable support
  - Nested configuration objects
  - Runtime validation
  - Type safety

### 2. Service Layer Components

#### EmbeddingManager
- Multi-provider support (OpenAI, FastEmbed, local models)
- Batch processing for cost optimization
- Smart model selection based on use case
- Sparse + dense embedding generation

#### QdrantService
- Direct SDK integration (no MCP overhead)
- Hybrid search with RRF fusion
- Vector quantization for storage efficiency
- Connection pooling and circuit breakers

#### CrawlManager
- Crawl4AI for bulk processing (4-6x faster)
- Firecrawl for on-demand scraping
- Unified interface for both providers
- Intelligent content extraction

#### CacheManager
- Multi-tier caching (memory + Redis)
- Content-based cache keys
- TTL and LRU eviction policies
- 80%+ cache hit rate target

### 3. Enhanced Chunking System
- **Basic**: Character-based splitting
- **Enhanced**: Code-aware with overlap
- **AST-Based**: Tree-sitter parsing for code

### 4. Unified MCP Server
- FastMCP 2.0 implementation
- 25+ tools for comprehensive functionality
- Streaming support for large responses
- Structured error handling

## Data Flow

### 1. Document Ingestion
```python
URL → CrawlManager → Raw Content → Enhanced Chunking → 
→ Text Chunks → EmbeddingManager → Vectors → QdrantService
```

### 2. Search Pipeline
```python
Query → SecurityValidator → EmbeddingManager → Query Vector →
→ QdrantService (Hybrid Search) → Results → Reranking → Response
```

### 3. Caching Strategy
```python
Request → CacheManager → Cache Hit? → Return Cached
                     ↓ (Cache Miss)
                  Process Request → Store in Cache → Return Result
```

## Key Design Decisions

### Direct SDK Integration
- **Before**: MCP-proxying added 50-80% overhead
- **After**: Direct Qdrant/OpenAI SDK calls
- **Result**: Faster API calls, better error handling

### Service Layer Pattern
- **Benefit**: Clean separation of concerns
- **Testing**: Easy to mock and test in isolation
- **Extensibility**: Simple to add new providers

### Unified Configuration
- **Single Source**: All config in one place
- **Validation**: Runtime type checking
- **Environment**: Easy deployment configuration

## Performance Characteristics

### Speed
- **Embedding Generation**: 1000+ embeddings/second
- **Search Latency**: < 100ms (95th percentile)
- **Cache Hit Rate**: 80%+ for common queries

### Storage
- **Vector Quantization**: 83-99% size reduction
- **Hybrid Indexing**: Optimized for both speed and accuracy
- **Persistence**: JSON-based project storage

### Accuracy
- **Hybrid Search**: 10-20% better than dense-only
- **BGE Reranking**: Additional 10-20% improvement
- **Chunking**: 30-50% better retrieval with AST parsing

## Integration Points

### MCP Server Tools
- `search_documents()` - Hybrid vector search
- `add_url()` - On-demand document addition
- `manage_collections()` - Database operations
- `get_analytics()` - Usage and performance metrics

### Environment Variables
```bash
OPENAI_API_KEY=sk-...          # Required for embeddings
FIRECRAWL_API_KEY=fc-...       # Optional for premium features
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
```

## Security Features

- URL validation and sanitization
- Collection name restrictions
- Query input validation
- Rate limiting on all endpoints
- No direct SQL/injection vulnerabilities

## Monitoring & Observability

- Structured logging with correlation IDs
- Performance metrics collection
- Error tracking and alerting
- Resource usage monitoring

## Related Documentation

- [Unified Configuration](./UNIFIED_CONFIGURATION.md) - Config system details
- [Client Management](./CENTRALIZED_CLIENT_MANAGEMENT.md) - Service patterns
- [Advanced Search](../features/ADVANCED_SEARCH_IMPLEMENTATION.md) - Search implementation
- [Performance Guide](../operations/PERFORMANCE_GUIDE.md) - Optimization strategies