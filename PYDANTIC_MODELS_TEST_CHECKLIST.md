# Pydantic Models Test Checklist

This checklist tracks all Pydantic models in the codebase and their testing status.

## Testing Best Practices (Pydantic v2)

Based on Pydantic v2 documentation research:

1. **Field Validation Testing**
   - Test valid inputs with edge cases
   - Test invalid inputs with expected ValidationError
   - Test field constraints (gt, ge, lt, le, min_length, max_length)
   - Test optional vs required fields

2. **Validator Testing**
   - Test field_validator functions
   - Test model_validator functions
   - Test custom validation logic

3. **Model Configuration Testing**
   - Test extra="forbid" behavior
   - Test aliases and field renaming
   - Test serialization/deserialization

4. **Error Testing**
   - Use pytest.raises(ValidationError)
   - Verify error types and messages
   - Check error locations (loc)

## Configuration Models (`src/config/models.py`) ✅ COMPLETED (94% coverage)

### Core Configuration Classes
- [x] `ModelBenchmark` - Performance benchmark for embedding models
- [x] `CacheConfig` - Cache configuration settings
- [x] `HNSWConfig` - HNSW index configuration
- [x] `CollectionHNSWConfigs` - Collection-specific HNSW configurations
- [x] `QdrantConfig` - Qdrant vector database configuration
- [x] `OpenAIConfig` - OpenAI API configuration
- [x] `FastEmbedConfig` - FastEmbed configuration
- [x] `FirecrawlConfig` - Firecrawl API configuration
- [x] `Crawl4AIConfig` - Crawl4AI configuration
- [x] `ChunkingConfig` - Text chunking configuration
- [x] `DocumentationSite` - Documentation site configuration
- [x] `PerformanceConfig` - Performance and optimization settings
- [x] `HyDEConfig` - HyDE configuration
- [x] `SmartSelectionConfig` - Smart model selection configuration
- [x] `EmbeddingConfig` - Advanced embedding configuration
- [x] `SecurityConfig` - Security settings
- [x] `UnifiedConfig` - Main unified configuration (BaseSettings)

### Validators to Test
- [x] API key validation functions
- [x] URL validation in QdrantConfig
- [x] Model name validation in OpenAIConfig
- [x] Chunk size validation in ChunkingConfig
- [x] Weight sum validation in SmartSelectionConfig
- [x] Provider key validation in UnifiedConfig
- [x] Directory creation in UnifiedConfig

## MCP Models (`src/mcp/models/`)

### Request Models (`requests.py`)
- [ ] `SearchRequest` - Search request with advanced options
- [ ] `EmbeddingRequest` - Embedding generation request
- [ ] `DocumentRequest` - Document processing request
- [ ] `CollectionRequest` - Collection management request
- [ ] `CacheRequest` - Cache operation request
- [ ] `MultiStageSearchRequest` - Multi-stage search request
- [ ] `HyDESearchRequest` - HyDE search request
- [ ] `FilteredSearchRequest` - Filtered search request

### Response Models (`responses.py`)
- [ ] `SearchResult` - Individual search result
- [ ] `SearchResponse` - Search operation response
- [ ] `EmbeddingResponse` - Embedding generation response
- [ ] `DocumentResponse` - Document processing response
- [ ] `CollectionResponse` - Collection operation response
- [ ] `ErrorResponse` - Error response model

## Domain Models (`src/models/`)

### API Contracts (`api_contracts.py`)
- [ ] `MCPRequest` - Base MCP request
- [ ] `MCPResponse` - Base MCP response
- [ ] `ErrorResponse` - Standard error response
- [ ] `SearchRequest` - Search API request
- [ ] `SearchResponse` - Search API response
- [ ] `DocumentRequest` - Document API request
- [ ] `DocumentResponse` - Document API response

### Configuration Models (`configuration.py`)
- [ ] Additional configuration models if any

### Document Processing Models (`document_processing.py`)
- [ ] `Document` - Document model
- [ ] `ProcessedDocument` - Processed document model
- [ ] `ChunkMetadata` - Chunk metadata model
- [ ] `DocumentChunk` - Document chunk model

### Vector Search Models (`vector_search.py`)
- [ ] `VectorSearchQuery` - Vector search query
- [ ] `VectorSearchResult` - Vector search result
- [ ] `HybridSearchQuery` - Hybrid search query
- [ ] `SearchMetadata` - Search metadata

### Validators (`validators.py`)
- [ ] Custom validator models
- [ ] Field validation models

## Service Models

### Browser Action Schemas (`src/services/browser/action_schemas.py`)
- [ ] `BrowserAction` - Browser action model
- [ ] `ClickAction` - Click action model
- [ ] `TypeAction` - Type action model
- [ ] `NavigateAction` - Navigation action model

### HyDE Models (`src/services/hyde/`)
- [ ] `HyDEConfig` - HyDE configuration (in config.py)
- [ ] `HyDEQuery` - HyDE query model
- [ ] `HyDEResult` - HyDE result model

### Search Models (`src/services/utilities/search_models.py`)
- [ ] Additional search utility models

## Other Models

### Chunking Models (`src/chunking.py`)
- [ ] Chunking-related models if any

### Error Models (`src/core/errors.py`, `src/services/errors.py`)
- [ ] Custom error models with Pydantic

## Testing Infrastructure

### Setup Requirements
- [ ] Install pytest-asyncio for async testing
- [ ] Set up conftest.py with common fixtures
- [ ] Create mock fixtures for external services
- [ ] Configure pytest settings in pyproject.toml

### Test Organization
```
tests/
├── unit/
│   ├── config/
│   │   ├── test_cache_config.py
│   │   ├── test_qdrant_config.py
│   │   ├── test_unified_config.py
│   │   └── ...
│   ├── mcp/
│   │   ├── test_requests.py
│   │   └── test_responses.py
│   ├── models/
│   │   ├── test_api_contracts.py
│   │   ├── test_document_processing.py
│   │   └── test_vector_search.py
│   └── services/
│       ├── test_browser_schemas.py
│       └── test_hyde_models.py
├── integration/
│   └── test_model_integration.py
└── conftest.py
```

## Progress Summary

**Total Models Identified**: ~60+ Pydantic models
**Tests Written**: 0
**Coverage Target**: >90%

## Next Steps

1. Set up test infrastructure (pytest-asyncio, fixtures)
2. Start with core configuration models (highest priority)
3. Test all validators and constraints
4. Add integration tests for model interactions
5. Run coverage reports and ensure >90% coverage