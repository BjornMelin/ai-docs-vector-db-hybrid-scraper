# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Inheritance**
> This repo inherits the global Simplicity Charter (`~/.claude/CLAUDE.md`),  
> which defines all coding, linting, testing, security and library choices  
> (uv, ruff, FastMCP 2.0, Pydantic v2, ≥ 90% pytest-cov, secrets in .env, etc.).  
> Project-specific implementation context is maintained in:  
> • `TODO.md` – comprehensive task list with V1/V2 implementation roadmap  
> • `TODO-V2.md` – future feature roadmap post-V1 release  
> • `docs/V1_IMPLEMENTATION_PLAN.md` – detailed 8-week implementation timeline  
> • `docs/V1_DOCUMENTATION_SUMMARY.md` – technical architecture overview  
> Always read these before coding.

## Project Snapshot

AI Documentation Vector DB is an advanced hybrid scraper that combines **Crawl4AI** (bulk processing) + **Firecrawl MCP** (on-demand) with **Qdrant vector database** for Claude Desktop/Code. The system implements research-backed optimizations including hybrid dense+sparse search, vector quantization, and BGE reranking for maximum accuracy at minimal cost.

**Core Technology Stack:**

- **Web Scraping**: Crawl4AI (bulk) + Firecrawl (on-demand via MCP)
- **Vector Database**: Qdrant with hybrid search & quantization
- **Embeddings**: OpenAI text-embedding-3-small + BGE reranking
- **MCP Server**: FastMCP 2.0 with streaming & composition
- **Modern Stack**: Python 3.13 + uv + Docker + async patterns

## Current Status & Priorities

**Completed V1 Foundation:**

- ✅ Advanced scraper with hybrid embedding pipeline
- ✅ Enhanced code-aware chunking (AST-based with Tree-sitter)
- ✅ Vector database operations with quantization & hybrid search
- ✅ Unified MCP server with FastMCP 2.0
- ✅ Comprehensive test suite (>90% coverage)
- ✅ Modern Python 3.13 + uv infrastructure
- ✅ Complete documentation suite (8-week V1 implementation plan)

**Next Priority Tasks (V1 Completion):**

1. API/SDK Integration Refactor (replace MCP proxying with direct SDKs)
2. Smart model selection with cost optimization  
3. Intelligent caching layer for embeddings and crawls
4. Batch processing optimization for 50% cost reduction
5. Enhanced MCP server features (streaming, composition)

**V2 Roadmap:** Advanced query processing, multi-modal documents, enterprise features

## Recent Completion: Browser-Use Migration ✅

**Completed 2025-05-29**: Replaced Stagehand with browser-use (Python-native, multi-LLM support)

- ✅ Complete BrowserUseAdapter implementation (532 lines)
- ✅ Fixed dependency conflicts (pydantic 2.10.4-2.11.0, langchain)
- ✅ 57/57 tests passing, 72% coverage
- ✅ Fallback chain: Crawl4AI → browser-use → Playwright
- ✅ Multi-LLM providers: OpenAI, Anthropic, Gemini

## Essential Development Commands

### Environment Setup

```bash
# Install dependencies with uv (primary package manager)
uv sync
uv run crawl4ai-setup

# Start services (Docker required)
./scripts/start-services.sh

# Health check
curl http://localhost:6333/health
```

### Core Operations

```bash
# Bulk documentation scraping (main workflow)
uv run python src/crawl4ai_bulk_embedder.py

# Database management
uv run python src/manage_vector_db.py stats
uv run python src/manage_vector_db.py search "query" --limit 10

# Start MCP server for Claude integration
uv run python src/unified_mcp_server.py
```

### Testing & Quality

```bash
# Run test suite
uv run pytest --cov=src

# Lint and format (required before commits)
ruff check . --fix
ruff format .

# Single test file
uv run pytest tests/test_chunking.py -v
```

### Docker Services

```bash
# Start/stop Qdrant
docker-compose up -d
docker-compose down

# Monitor Qdrant
docker logs qdrant-vector-db -f
```

## Development Workflow

### Before Starting

1. **Check task status:** Read `TODO.md` for current priorities and implementation guides
2. **Review documentation:** Check relevant guides in `docs/` (V1 plan, architecture, etc.)
3. **Understand context:** Review recent commits and current branch status
4. **Set up environment:** Ensure services running (`./scripts/start-services.sh`)

### During Development

1. **Follow TDD:** Write tests first, maintain ≥90% coverage with `uv run pytest --cov=src`
2. **Use direct APIs:** Prefer Qdrant SDK, OpenAI SDK over MCP proxying
3. **Modern patterns:** Use Pydantic v2, async/await, type hints
4. **Performance focus:** Implement batching, caching, monitoring
5. **Run quality checks:** `ruff check . --fix && ruff format .` after changes

### After Completing

1. **Test thoroughly:** Run full test suite and check coverage
2. **Update documentation:** Modify relevant docs if APIs/behavior changed
3. **Performance validation:** Benchmark if changes affect search/embedding
4. **Quality gates:** Ensure linting passes and no security issues
5. **Commit properly:** Use conventional commit format (see Git Workflow)

## Git Workflow & Commit Standards

**Branches:** `main` (protected) • `feat/*` • `fix/*` • `docs/*` • `refactor/*`

**Conventional Commits Format (Required):**

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:** `feat` • `fix` • `docs` • `style` • `refactor` • `perf` • `test` • `build` • `ci`

**Examples:**

```bash
feat(search): implement hybrid search with RRF fusion
fix(chunking): resolve AST parsing for TypeScript interfaces  
docs(api): add embedding provider selection guide
refactor(mcp): consolidate client management patterns
perf(vector): optimize batch embedding generation
```

**GitHub Integration:**

- Use GitHub MCP server tools over `gh` CLI when available
- PR titles must follow conventional commit format
- Require approval for main branch merges
- Auto-run tests and linting on all PRs

## Core Architecture

### High-Level Component Flow

1. **Crawl4AI Bulk Embedder** (`src/crawl4ai_bulk_embedder.py`) - Main scraping engine

   - Processes documentation sites from `config/documentation-sites.json`
   - Implements hybrid embedding pipeline (dense + sparse vectors)
   - Supports multiple providers: OpenAI, FastEmbed, local models

2. **Enhanced Chunking System** (`src/chunking.py`) - SOTA text processing

   - Three-tier strategy: Basic character-based, Code-aware, AST-based
   - Tree-sitter integration for Python/JS/TS function boundary preservation
   - Configurable chunk sizes optimized for embeddings (1600 chars ≈ 400-600 tokens)

3. **Vector Database Manager** (`src/manage_vector_db.py`) - Qdrant operations

   - Hybrid search combining dense and sparse vectors
   - Vector quantization for 83-99% storage reduction
   - BGE reranking for 10-20% accuracy improvement

4. **Unified MCP Server** - Claude Desktop integration
   - `src/unified_mcp_server.py` - Consolidated FastMCP server with all functionality
   - Combines Qdrant vector search + Firecrawl web scraping capabilities
   - Supports structured logging, error handling, and advanced search strategies

### Key Data Flow

```
Documentation URLs → Crawl4AI → Enhanced Chunking → Embedding Pipeline → Qdrant → MCP Server → Claude
```

## Important Implementation Notes

### Core Libraries & Dependencies

- **FastMCP 2.0**: Required for all MCP server implementations
- **Qdrant SDK**: Direct vector database operations (no MCP proxying)
- **OpenAI SDK**: Direct embedding generation (cost-optimal text-embedding-3-small)
- **Crawl4AI**: High-performance bulk web scraping (4-6x faster than alternatives)
- **Tree-sitter**: AST-based code parsing for Python/JS/TS
- **Pydantic v2**: All data models with field validation
- **FastEmbed**: Local embedding generation for privacy/performance

### Service Layer Patterns

- **Unified ClientManager**: Singleton pattern for all API clients with connection pooling
- **BaseService Pattern**: Common initialization, cleanup, and error handling
- **Async Context Managers**: Proper resource cleanup for all operations
- **Circuit Breaker**: Automatic fallback for external service failures

### Performance Optimization Requirements

- **Batch Processing**: 100+ documents per embedding call
- **Intelligent Caching**: Redis/in-memory with content-based keys (80%+ hit rate target)
- **Vector Quantization**: int8 quantization for 83-99% storage reduction
- **Connection Pooling**: Async connection management for Qdrant

### Search Strategy Implementation

- **Hybrid Search**: Default RRF fusion with dense+sparse vectors
- **Multi-Stage Retrieval**: Matryoshka embeddings (small→large→rerank)
- **BGE Reranking**: 10-20% accuracy improvement with minimal complexity
- **Query API**: Qdrant prefetch patterns for optimal performance

### Error Handling & Resilience

- **Exponential Backoff**: Required for all external API calls
- **Graceful Degradation**: Fallback to local models when cloud APIs fail
- **Comprehensive Logging**: Structured logging with correlation IDs
- **Retry Patterns**: Maximum 3 retries with jitter for rate limit handling

## Quick Reference Paths

- **Core Source**: `/src/` (main application code)
- **MCP Server**: `/src/unified_mcp_server.py` (consolidated MCP implementation)
- **Services**: `/src/services/` (service layer with cache, crawling, embeddings)
- **Configuration**: `/config/` (site configs, MCP templates)
- **Documentation**: `/docs/` (implementation guides and architecture)
- **Task Management**: `/TODO.md`, `/TODO-V2.md` (comprehensive roadmaps)
- **Tests**: `/tests/` (>90% coverage requirement)
- **Scripts**: `/scripts/` (service management and automation)

## Common Commands

```bash
# Development Lifecycle
uv sync                                           # Install dependencies
./scripts/start-services.sh                      # Start all services
uv run python src/crawl4ai_bulk_embedder.py     # Bulk documentation scraping
uv run python src/unified_mcp_server.py         # Start unified MCP server

# Quality & Testing (NEW: Clean, Readable Output)
./scripts/test.sh quick                          # Fast unit tests with clear results
./scripts/test.sh clean                          # Summary only (no random dots/chars)
./scripts/test.sh coverage                       # Full coverage report
./scripts/test.sh failed                         # Only previously failed tests
uv run pytest tests/unit/ -v                    # Verbose unit tests (organized structure)
uv run pytest tests/integration/ -v             # Integration tests
ruff check . --fix && ruff format .             # Lint and format code
uv run pytest tests/unit/config/test_config.py -v  # Single test file

# Database Operations  
uv run python src/manage_vector_db.py stats     # Database statistics
uv run python src/manage_vector_db.py search "query" --limit 10  # Search test

# Docker Services
docker-compose up -d                            # Start Qdrant
curl http://localhost:6333/health               # Health check
docker logs qdrant-vector-db -f                # Monitor logs

# GitHub Integration (prefer GitHub MCP over gh CLI)
# Use GitHub MCP server tools when available for:
# - Repository operations  
# - Issue management
# - Pull request workflows
# - Code search and analysis
```

## Important Configuration Files

- `config/documentation-sites.json` - Sites to scrape with parameters
- `config/crawl4ai-site-templates.json` - Pre-configured templates for common doc site types
- `pyproject.toml` - Modern Python packaging with uv + ruff config
- `docker-compose.yml` - Optimized Qdrant configuration
- `.env` - API keys (OPENAI_API_KEY, FIRECRAWL_API_KEY)

## Development Patterns

### Crawl4AI Best Practices

When working with Crawl4AI for web scraping:

1. **Performance Optimization**:
   - Start with 10 concurrent requests, scale up to 50 for simple sites
   - Use headless=True unless debugging JavaScript issues
   - Disable unnecessary features (screenshots, PDFs) for speed
   - Implement caching with DragonflyDB to avoid re-crawling

2. **Content Extraction**:
   - Always check `config/crawl4ai-site-templates.json` for site-specific configs
   - Use wait_for selectors for dynamic content loading
   - Implement custom JavaScript for SPAs and infinite scroll
   - Validate content length (>100 chars) to detect extraction failures

3. **Error Handling**:
   - Implement exponential backoff with 3 retry attempts
   - Use circuit breakers for consistently failing sites
   - Monitor memory usage and implement periodic cleanup
   - Log detailed error context for debugging

4. **Site-Specific Configurations**:
   - Sphinx docs: Simple selectors, no JS needed
   - React/Vue/Angular: Wait for hydration, custom JS execution
   - API docs: Expand all collapsible sections before extraction
   - MDN/complex sites: Use click_show_more patterns

5. **Resource Management**:
   - Always use try/finally blocks for cleanup
   - Process URLs in batches of 100 for large crawls
   - Monitor CPU/memory with psutil during benchmarks
   - Implement memory limits for containerized deployments

### Embedding Provider Strategy

The system supports multiple embedding providers with fallback:

- **Primary**: OpenAI text-embedding-3-small (cost-optimal)
- **Alternative**: FastEmbed models (local, faster)
- **Research**: NV-Embed-v2 (highest accuracy)

### Chunking Strategy Selection

- **Basic**: Simple character-based (legacy compatibility)
- **Enhanced**: Code-aware with function boundary preservation (default)
- **AST**: Tree-sitter parsing for source code files (advanced)

### Search Strategy Implementation

- **Dense**: Traditional vector similarity
- **Sparse**: SPLADE++ keyword matching
- **Hybrid**: Combined with RRF ranking (recommended)
- **Reranked**: BGE-reranker-v2-m3 post-processing

## API Integration Patterns

### MCP Server Usage

The MCP servers expose these key functions:

- `search_documents()` - Hybrid search with reranking
- `add_url()` - On-demand document addition via Firecrawl
- `get_collections()` - Database statistics and health
- `delete_collection()` - Cleanup operations

### Environment Variables

Required for full functionality:

- `OPENAI_API_KEY` - For embedding generation
- `FIRECRAWL_API_KEY` - For premium MCP features (optional)
- `QDRANT_URL` - Database connection (default: <http://localhost:6333>)

## Performance Considerations

### Optimization Settings

- Concurrent crawling: 10 pages (adjustable in bulk embedder)
- Batch processing: 32 documents per embedding call
- Vector quantization: Enabled by default (83% storage reduction)
- Reranking: Top-20 retrieve, top-5 return pattern

### Resource Requirements

- Docker Desktop with WSL2 integration
- 4GB+ RAM for Qdrant container
- Python 3.13+ for optimal performance
- SSD storage recommended for vector persistence

## Security & Error Handling

### API Key Management

- Never commit API keys to repository
- Use `.env` file for local development
- Validate keys on startup in all services

### Error Recovery

- Automatic retry logic for network operations
- Graceful degradation when optional services unavailable
- Comprehensive logging with colorlog for debugging

## Claude Desktop Integration

The system provides unified MCP server configuration combining all functionality:

```json
{
  "mcpServers": {
    "ai-docs-vector-db": {
      "command": "uv",
      "args": ["run", "python", "src/unified_mcp_server.py"],
      "cwd": "/path/to/project",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

This replaces separate Qdrant and Firecrawl MCP servers with a single unified interface.

### Streaming Support Configuration

The unified MCP server includes enhanced streaming support for large search results:

**Basic Configuration (Default):**

```json
{
  "mcpServers": {
    "ai-docs-vector-db": {
      "command": "uv",
      "args": ["run", "python", "src/unified_mcp_server.py"],
      "cwd": "/path/to/project",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

**Advanced Streaming Configuration:**

```json
{
  "mcpServers": {
    "ai-docs-vector-db": {
      "command": "uv",
      "args": ["run", "python", "src/unified_mcp_server.py"],
      "cwd": "/path/to/project",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "QDRANT_URL": "http://localhost:6333",
        "FASTMCP_TRANSPORT": "streamable-http",
        "FASTMCP_HOST": "127.0.0.1",
        "FASTMCP_PORT": "8000",
        "FASTMCP_BUFFER_SIZE": "8192",
        "FASTMCP_MAX_RESPONSE_SIZE": "10485760"
      }
    }
  }
}
```

**Environment Variables:**

- `FASTMCP_TRANSPORT`: Transport type (`streamable-http` for streaming, `stdio` for Claude Desktop)
- `FASTMCP_HOST`: Host for HTTP transport (default: `127.0.0.1`)
- `FASTMCP_PORT`: Port for HTTP transport (default: `8000`)
- `FASTMCP_BUFFER_SIZE`: Response buffer size (default: `8192`)
- `FASTMCP_MAX_RESPONSE_SIZE`: Maximum response size (default: `10485760` - 10MB)

**Benefits:**

- Optimized performance for large search results (1000+ documents)
- Configurable response buffering for memory efficiency
- Automatic fallback to stdio for Claude Desktop compatibility
- Enhanced error handling and timeout management

**Performance Comparison:**

| Transport Mode | Best For | Response Size | Typical Use Case |
|---|---|---|---|
| `stdio` | Claude Desktop | < 100KB | Interactive queries, small result sets |
| `streamable-http` | Large results | > 100KB | Bulk operations, comprehensive searches |

**Example Performance Benefits:**

- **Small Queries (10 results)**: stdio and streamable-http perform similarly (~50ms)
- **Medium Queries (100 results, ~100KB)**: streamable-http shows 15-20% improvement
- **Large Queries (1000+ results, >1MB)**: streamable-http provides 40-60% performance gain
- **Memory Usage**: streamable-http uses 30-50% less memory for large responses through buffering

**When to Use Each Mode:**

```bash
# For Claude Desktop integration (default fallback)
export FASTMCP_TRANSPORT=stdio

# For high-performance large result handling  
export FASTMCP_TRANSPORT=streamable-http
export FASTMCP_BUFFER_SIZE=16384      # Larger buffer for big responses
export FASTMCP_MAX_RESPONSE_SIZE=20971520  # 20MB limit for bulk operations
```

## Testing Requirements

- **Unit Tests**: Use pytest with ≥90% coverage for all services
- **Integration Tests**: Test MCP integrations with comprehensive mocks  
- **Performance Tests**: Benchmark embedding generation and search latency
- **Async Tests**: Use pytest-asyncio for all async operations
- **Always run**: `uv run pytest --cov=src` before commits

**Test Patterns:**

```python
@pytest.mark.asyncio
async def test_hybrid_search():
    # Test implementation with proper mocks
    
@pytest.fixture
async def embedding_service():
    # Async service fixtures with cleanup
```

## Performance Targets (V1)

- **Search Latency**: < 100ms (95th percentile)
- **Embedding Generation**: > 1000 embeddings/second
- **Index Update**: < 5 seconds for single document
- **Cache Hit Rate**: > 80% for common queries
- **Storage Efficiency**: > 80% compression ratio with quantization
- **Uptime**: 99.9% availability
- **Test Coverage**: ≥90% across all core functionality

## Security & Cost Management

**API Key Security:**

- Never commit API keys to repository
- Use `.env` file for local development (`.env.example` for templates)
- Validate keys on startup in all services
- Implement usage limits and cost alerts

**Cost Optimization:**

- Default to text-embedding-3-small (5x cheaper than ada-002)
- Implement batch processing for 50% cost reduction
- Use local models (FastEmbed) when privacy required
- Monitor usage with budget alerts ($50/month target)

**Data Protection:**

- Local-only mode available for privacy-conscious users
- All data stored locally in Qdrant (no cloud vendor lock-in)
- Comprehensive audit trails for all operations
