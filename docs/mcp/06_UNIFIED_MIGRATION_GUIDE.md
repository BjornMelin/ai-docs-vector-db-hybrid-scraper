# Unified MCP Server Migration Guide

This guide helps you migrate from the previous MCP server implementations to the new unified MCP server.

## Overview

The unified MCP server (`src/unified_mcp_server.py`) consolidates all functionality from:
- `mcp_server.py` (deprecated)
- `enhanced_mcp_server.py` (deprecated)
- `mcp_server_refactored.py` (legacy)
- `enhanced_mcp_server_refactored.py` (legacy)

## Key Improvements

### 1. **Consolidated Interface**
- Single server exposing all features
- No need to run multiple MCP servers
- Unified configuration and management

### 2. **Advanced Features**
- Hybrid search with dense+sparse vectors
- BGE reranking for improved accuracy
- Multi-provider embedding support
- Smart model selection
- Two-tier caching system
- Batch processing
- Cost estimation and optimization
- Analytics and monitoring

### 3. **Service Layer Integration**
- Uses completed service layer architecture
- Direct SDK integration (no MCP proxying)
- Better error handling and resilience
- Connection pooling and optimization

## Migration Steps

### 1. Update Configuration

Replace your existing MCP configuration:

**Old Configuration (Multiple Servers):**
```json
{
  "mcpServers": {
    "qdrant": {
      "command": "uvx",
      "args": ["mcp-server-qdrant"],
      "env": { ... }
    },
    "firecrawl": {
      "command": "npx",
      "args": ["-y", "firecrawl-mcp"],
      "env": { ... }
    },
    "ai-docs": {
      "command": "uv",
      "args": ["run", "python", "src/enhanced_mcp_server.py"],
      "env": { ... }
    }
  }
}
```

**New Configuration (Unified Server):**
```json
{
  "mcpServers": {
    "ai-docs-vector-db-unified": {
      "command": "uv",
      "args": ["run", "python", "src/unified_mcp_server.py"],
      "cwd": "/path/to/ai-docs-vector-db-hybrid-scraper",
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key",
        "FIRECRAWL_API_KEY": "your-firecrawl-api-key",
        "QDRANT_URL": "http://localhost:6333",
        "REDIS_URL": "redis://localhost:6379"
      }
    }
  }
}
```

### 2. Update Tool Names

Some tools have been renamed or consolidated:

| Old Tool | New Tool | Changes |
|----------|----------|---------|
| `search` | `search_documents` | Added advanced options |
| `scrape_url` | `add_document` | Enhanced with chunking strategies |
| `create_collection` | Automatic | Collections created on demand |
| `list_collections` | `list_collections` | Added statistics |
| N/A | `search_similar` | New similarity search |
| N/A | `generate_embeddings` | Direct embedding access |
| N/A | `add_documents_batch` | Batch processing |
| N/A | `create_project` | Project management |
| N/A | `get_analytics` | Analytics and monitoring |

### 3. Update Tool Parameters

#### Search Documents
**Old:**
```python
search(query="test", collection="docs", limit=10)
```

**New:**
```python
search_documents({
    "query": "test",
    "collection": "docs",
    "limit": 10,
    "strategy": "hybrid",  # New: dense, sparse, or hybrid
    "enable_reranking": true,  # New: BGE reranking
    "include_metadata": true  # New: metadata control
})
```

#### Add Document
**Old:**
```python
scrape_url(url="https://example.com", collection="docs")
```

**New:**
```python
add_document({
    "url": "https://example.com",
    "collection": "docs",
    "chunking_strategy": "enhanced",  # New: basic, enhanced, or ast
    "chunk_size": 1600,  # New: customizable
    "chunk_overlap": 200  # New: overlap control
})
```

### 4. Environment Variables

Ensure all required environment variables are set:

```bash
# Required
export OPENAI_API_KEY="sk-..."
export QDRANT_URL="http://localhost:6333"

# Optional but recommended
export FIRECRAWL_API_KEY="fc-..."
export REDIS_URL="redis://localhost:6379"

# Optional
export CACHE_TTL="3600"
export MAX_CONCURRENT_CRAWLS="10"
```

### 5. Service Dependencies

Ensure all services are running:

```bash
# Start Qdrant
docker-compose up -d

# Start Redis (optional but recommended)
docker run -d -p 6379:6379 redis:alpine

# Verify services
curl http://localhost:6333/health
redis-cli ping
```

## New Features to Explore

### 1. **Project Management**
Create projects to group related documents:
```python
create_project({
    "name": "API Documentation",
    "description": "All API docs",
    "quality_tier": "premium",  # economy, balanced, or premium
    "urls": ["https://api.example.com/docs"]
})
```

### 2. **Batch Processing**
Process multiple documents efficiently:
```python
add_documents_batch({
    "urls": [
        "https://docs.example.com/guide1",
        "https://docs.example.com/guide2",
        "https://docs.example.com/guide3"
    ],
    "collection": "guides",
    "parallel_limit": 5
})
```

### 3. **Analytics and Monitoring**
Get insights into your data:
```python
get_analytics({
    "include_performance": true,
    "include_costs": true
})
```

### 4. **Cost Estimation**
Estimate costs before processing:
```python
estimate_costs({
    "text_count": 1000,
    "average_length": 1500,
    "include_storage": true
})
```

### 5. **Smart Search Strategies**
Choose the optimal search approach:
- **Dense**: Traditional vector similarity
- **Sparse**: Keyword matching with SPLADE
- **Hybrid**: Combined approach (recommended)

### 6. **Similarity Search**
Find conceptually similar content:
```python
search_similar({
    "content": "Your reference text here",
    "collection": "docs",
    "threshold": 0.8
})
```

## Troubleshooting

### Common Issues

1. **"Service not initialized" errors**
   - Ensure all required services are running
   - Check environment variables
   - Verify network connectivity

2. **"Collection not found" errors**
   - Collections are now created automatically
   - Check collection name spelling
   - Use `list_collections()` to see available collections

3. **Performance issues**
   - Enable Redis for caching
   - Use batch processing for multiple documents
   - Optimize chunk sizes based on your content

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
uv run python src/unified_mcp_server.py
```

## Best Practices

1. **Use Projects** for grouping related documentation
2. **Enable Caching** with Redis for better performance
3. **Choose Quality Tiers** based on your needs:
   - Economy: Fast, lower accuracy
   - Balanced: Good trade-off (default)
   - Premium: Highest accuracy, slower
4. **Batch Process** when adding multiple documents
5. **Monitor Costs** with analytics tools
6. **Use Hybrid Search** for best results

## Deprecation Timeline

- **Immediate**: Stop using `mcp_server.py` and `enhanced_mcp_server.py`
- **Next Release**: Remove legacy refactored servers
- **Future**: All functionality through unified server only

## Support

For issues or questions:
1. Check the [troubleshooting guide](../TROUBLESHOOTING.md)
2. Review [MCP documentation](./README.md)
3. Open an issue on GitHub