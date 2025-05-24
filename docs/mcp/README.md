# MCP Server Documentation

This directory contains all documentation related to the Model Context Protocol (MCP) server implementation for the AI Documentation Vector DB project.

## 🎉 New: Unified MCP Server

We've consolidated all MCP functionality into a single, powerful unified server! See the [Migration Guide](./06_UNIFIED_MIGRATION_GUIDE.md) to upgrade.

## Documentation Structure

### 1. [MCP Server Guide](01_GUIDE.md)
Comprehensive guide to understanding and using the MCP servers, including:
- Overview of MCP server capabilities
- Tool descriptions and usage examples
- Integration with Claude Desktop
- Performance considerations

### 2. [MCP Server Setup](02_SETUP.md)
Step-by-step setup instructions for:
- Environment configuration
- Service dependencies (Qdrant, Redis)
- API key management
- Running the MCP servers
- Claude Desktop configuration

### 3. [MCP Server Architecture](03_ARCHITECTURE.md)
Technical architecture documentation covering:
- Service layer design patterns
- Tool composition and workflow patterns
- Error handling and resilience
- Security considerations
- Performance optimizations

### 4. [MCP Server Enhancement Plan](04_ENHANCEMENT_PLAN.md)
Future enhancement roadmap including:
- Planned features and improvements
- Performance optimization strategies
- Integration enhancements
- V2 feature planning

### 5. [Unified MCP Server Implementation](05_UNIFIED_IMPLEMENTATION.md)
Details about the unified MCP server approach:
- Consolidation strategy
- FastMCP 2.0 integration
- Service layer utilization
- Tool organization and patterns

### 6. [Unified MCP Server Migration Guide](06_UNIFIED_MIGRATION_GUIDE.md) 🆕
Step-by-step migration to the unified server:
- Migration instructions
- New features and improvements
- Tool name mappings
- Troubleshooting guide

## Current Implementation

The **Unified MCP Server** (`src/unified_mcp_server.py`) consolidates all functionality:

### Core Features
- **FastMCP 2.0**: Modern, high-performance MCP implementation
- **Service Layer**: Clean architecture with direct SDK integration
- **No MCP Proxying**: Direct API access for better performance

### Advanced Capabilities
- **Hybrid Search**: Dense + sparse vectors with BGE reranking
- **Smart Chunking**: Basic, Enhanced, and AST-based strategies
- **Multi-Provider Embeddings**: OpenAI, FastEmbed with failover
- **Project Management**: Group and manage related documents
- **Batch Processing**: Efficient multi-document handling
- **Two-Tier Caching**: L1 local + L2 Redis for performance
- **Analytics & Monitoring**: Comprehensive metrics and insights
- **Cost Optimization**: Estimation and smart model selection

## Quick Start

### For New Users

1. Install dependencies:
   ```bash
   uv sync
   uv run crawl4ai-setup
   ```

2. Start services:
   ```bash
   ./scripts/start-services.sh
   ```

3. Configure Claude Desktop:
   ```json
   {
     "mcpServers": {
       "ai-docs-vector-db-unified": {
         "command": "uv",
         "args": ["run", "python", "src/unified_mcp_server.py"],
         "cwd": "/path/to/project",
         "env": {
           "OPENAI_API_KEY": "your-key",
           "QDRANT_URL": "http://localhost:6333"
         }
       }
     }
   }
   ```

4. Start using the tools in Claude!

### For Existing Users

Follow the [Migration Guide](./06_UNIFIED_MIGRATION_GUIDE.md) to upgrade from legacy servers.

## Available Tools

### Search & Retrieval
- `search_documents` - Advanced hybrid search with reranking
- `search_similar` - Pure vector similarity search

### Document Management
- `add_document` - Add single document with smart chunking
- `add_documents_batch` - Efficient batch processing

### Embedding Management
- `generate_embeddings` - Direct embedding generation
- `list_embedding_providers` - Available providers info

### Project Management
- `create_project` - Create documentation project
- `list_projects` - List all projects
- `search_project` - Search within project

### Collection Management
- `list_collections` - List vector collections
- `delete_collection` - Remove collection
- `optimize_collection` - Performance optimization

### Analytics & Monitoring
- `get_analytics` - Comprehensive analytics
- `get_system_health` - Health check
- `estimate_costs` - Cost prediction

### Cache Management
- `clear_cache` - Clear cache entries
- `get_cache_stats` - Cache metrics

### Utilities
- `validate_configuration` - Configuration validation

## Architecture Overview

```
┌─────────────────────────────────────┐
│       MCP Tools (FastMCP 2.0)       │ ← Thin interface layer
├─────────────────────────────────────┤
│    UnifiedServiceManager            │ ← Orchestration
├─────────────────────────────────────┤
│     Service Layer (Managers)        │
│  - EmbeddingManager                 │ ← Business logic
│  - CrawlManager                     │
│  - QdrantService                    │
│  - CacheManager                     │
├─────────────────────────────────────┤
│        Direct SDK Access            │
│  - OpenAI SDK                       │ ← No MCP proxying
│  - Qdrant Client                    │
│  - Firecrawl SDK                    │
└─────────────────────────────────────┘
```

## Key Principles

1. **Service Layer Pattern**: MCP tools as thin wrappers over services
2. **Direct SDK Integration**: No external MCP server dependencies
3. **Performance First**: Caching, pooling, batch processing
4. **Cost Optimization**: Smart model selection, usage tracking
5. **Error Resilience**: Graceful degradation, circuit breakers
6. **Security**: Input validation, rate limiting, key management

## Performance & Cost

- **Search Latency**: < 100ms (95th percentile)
- **Embedding Cost**: ~$0.02 per 1M tokens (text-embedding-3-small)
- **Cache Hit Rate**: > 80% with proper configuration
- **Batch Efficiency**: 50% cost reduction for bulk operations
- **Storage**: 83-99% reduction with vector quantization

## Contributing

When working on MCP server improvements:

1. Add new tools to `unified_mcp_server.py`
2. Update tests in `test_unified_mcp_server.py`
3. Follow the service layer pattern
4. Use direct SDK integration (no MCP proxying)
5. Document new features thoroughly
6. Ensure ≥90% test coverage

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for general guidelines.

## Deprecation Notice

The following servers are deprecated:
- ❌ `mcp_server.py` - Use unified server
- ❌ `enhanced_mcp_server.py` - Use unified server
- ⚠️ `mcp_server_refactored.py` - Legacy, migrate soon
- ⚠️ `enhanced_mcp_server_refactored.py` - Legacy, migrate soon

## Support

- Check [Troubleshooting Guide](../../docs/TROUBLESHOOTING.md)
- Review [Migration Guide](./06_UNIFIED_MIGRATION_GUIDE.md)
- Open GitHub issues for bugs or features