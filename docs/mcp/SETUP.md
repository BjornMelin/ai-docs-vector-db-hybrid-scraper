# MCP Server Setup Guide

**Status**: Current  
**Last Updated**: 2025-05-26

## Overview

This guide covers setting up the unified MCP server for AI Documentation Vector DB with Claude Desktop/Code integration.

## Prerequisites

- Python 3.13+ with `uv` package manager
- Docker Desktop (for Qdrant)
- Claude Desktop or Claude Code
- API Keys: OpenAI (required), Firecrawl (optional)

## Quick Setup

### 1. Install Dependencies

```bash
# Install uv if not already installed
pip install uv

# Install project dependencies
uv sync

# Set up Crawl4AI
uv run crawl4ai-setup
```

### 2. Configure Environment

Create `.env` file:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
FIRECRAWL_API_KEY=fc-...
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
```

### 3. Start Services

```bash
# Start Qdrant and Redis
./scripts/start-services.sh

# Verify services
curl http://localhost:6333/health
```

### 4. Test MCP Server

```bash
# Run the unified MCP server
uv run python src/unified_mcp_server.py

# The server will output available tools on startup
```

## Claude Desktop Configuration

### Basic Configuration

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

### Advanced Configuration

For production environments with all features:

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

## Available MCP Tools

The unified server provides 25+ tools:

### Search Tools

- `search_documents` - Hybrid vector search with reranking
- `search_by_collection` - Search within specific collections
- `search_similar` - Find similar documents

### Document Management

- `add_url` - Add single URL to index
- `add_urls` - Bulk URL addition
- `update_document` - Update existing documents
- `delete_document` - Remove documents

### Collection Management

- `list_collections` - Show all collections
- `create_collection` - Create new collection
- `delete_collection` - Remove collection
- `get_collection_stats` - Collection metrics

### Project Management

- `create_project` - Initialize new project
- `list_projects` - Show all projects
- `update_project` - Modify project settings
- `delete_project` - Remove project

### Analytics

- `get_usage_stats` - API usage metrics
- `get_performance_metrics` - Search performance
- `get_cache_stats` - Cache hit rates

## Testing Your Setup

### 1. Verify Server Startup

After configuring Claude Desktop, restart it and check:

1. Open Claude Desktop
2. Start a new conversation
3. Type: "Can you list my vector collections?"
4. Claude should use the `list_collections` tool

### 2. Test Search Functionality

```plaintext
You: "Search for documentation about authentication"
Claude: [Uses search_documents tool with your query]
```

### 3. Test Document Addition

```plaintext
You: "Add https://docs.example.com to my documentation index"
Claude: [Uses add_url tool to crawl and index the page]
```

## Configuration Options

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
| `MAX_SEARCH_RESULTS` | No | 10 | Default search limit |

### Service Configuration

The unified configuration system (`src/config/models.py`) provides:

- Nested configuration objects
- Environment variable overrides
- Runtime validation
- Type safety with Pydantic v2

## Troubleshooting

### Common Issues

#### 1. "MCP server not found"

- Ensure absolute path in `cwd`
- Verify `uv` is in PATH
- Check file permissions

#### 2. "Connection refused"

- Ensure Qdrant is running: `docker ps`
- Check QDRANT_URL is correct
- Verify port 6333 is not blocked

#### 3. "API key invalid"

- Verify OPENAI_API_KEY is set correctly
- Check for extra spaces or quotes
- Ensure key has proper permissions

### Debug Mode

Enable verbose logging:

```json
{
  "env": {
    "LOG_LEVEL": "DEBUG",
    "PYTHONUNBUFFERED": "1"
  }
}
```

### View Logs

**macOS/Linux**:

```bash
tail -f ~/Library/Logs/Claude/mcp-server-ai-docs-vector-db.log
```

**Windows**:

```powershell
Get-Content "$env:APPDATA\Claude\Logs\mcp-server-ai-docs-vector-db.log" -Tail 50 -Wait
```

## Performance Optimization

### Connection Pooling

The server uses connection pooling by default:

- Qdrant: 10 connections
- Redis: 20 connections
- HTTP: 100 connections

### Batch Processing

Enable batch operations for better performance:

```json
{
  "env": {
    "ENABLE_BATCH_PROCESSING": "true",
    "BATCH_SIZE": "32"
  }
}
```

### Resource Limits

Set appropriate limits:

```json
{
  "env": {
    "MAX_MEMORY_MB": "2048",
    "REQUEST_TIMEOUT": "30"
  }
}
```

## Next Steps

- Read the [Migration Guide](./MIGRATION_GUIDE.md) if upgrading
- Check [System Overview](../architecture/SYSTEM_OVERVIEW.md) for architecture
- See [Advanced Search](../features/ADVANCED_SEARCH_IMPLEMENTATION.md) for search features

## Related Documentation

- [MCP Overview](./README.md)
- [Migration Guide](./MIGRATION_GUIDE.md)
- [Troubleshooting](../operations/TROUBLESHOOTING.md)
- [Performance Guide](../operations/PERFORMANCE_GUIDE.md)
