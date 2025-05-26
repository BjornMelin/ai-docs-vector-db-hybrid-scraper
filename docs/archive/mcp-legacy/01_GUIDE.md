# MCP Server Guide

## Overview

This project provides two Model Context Protocol (MCP) servers built with FastMCP 2.0:

1. **Basic MCP Server** (`mcp_server.py`) - Core functionality for scraping and search
2. **Enhanced MCP Server** (`enhanced_mcp_server.py`) - Advanced features with project management

Both servers integrate with existing Crawl4AI functionality and can leverage Firecrawl and Qdrant MCP servers for enhanced capabilities.

## Installation

1. Install dependencies:

   ```bash
   uv pip install -e .
   ```

2. Set up environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export QDRANT_URL="http://localhost:6333"  # Optional, defaults to localhost
export FIRECRAWL_API_KEY="your-firecrawl-key"  # Optional, for Firecrawl integration
```

## Running the MCP Server

### Option 1: Using the command line script

```bash
uv run mcp-server
```

### Option 2: Using FastMCP CLI

```bash
fastmcp run src/mcp_server.py
```

### Option 3: Development mode with inspector

```bash
fastmcp dev src/mcp_server.py
```

### Option 4: HTTP mode for testing

```bash
fastmcp run src/mcp_server.py --transport streamable-http --port 8000
```

## Available Tools

### Basic MCP Server Tools

#### Scraping Tools

- `scrape_url` - Scrape and index documentation from any URL using Crawl4AI
- `scrape_with_firecrawl` - Use Firecrawl API for high-quality extraction

#### Search Tools

- `search` - Semantic search across indexed documentation

#### Collection Management

- `list_collections` - List all vector database collections
- `create_collection` - Create a new collection with custom settings
- `delete_collection` - Delete a collection
- `get_collection_info` - Get detailed collection information

#### Utilities

- `clear_cache` - Clear the Crawl4AI scraping cache

### Enhanced MCP Server Tools

#### Project Management

- `create_project` - Create a new documentation project
- `list_projects` - List all documentation projects
- `update_project` - Update project configuration
- `export_project` - Export project data

#### Intelligent Scraping

- `plan_scraping` - Create an intelligent scraping plan
- `execute_scraping_plan` - Execute a scraping plan with progress tracking

#### Advanced Search

- `smart_search` - Advanced search with reranking and filtering

#### Documentation Management

- `index_documentation` - Index content with optimal chunking

## Claude Desktop Integration

1. Copy the configuration to Claude Desktop config directory:

   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2. Add the server configuration:

   ```json
   {
     "mcpServers": {
       "ai-docs-vector-db": {
         "command": "uv",
         "args": ["run", "mcp-server"],
         "cwd": "/path/to/ai-docs-vector-db-hybrid-scraper",
         "env": {
           "OPENAI_API_KEY": "your-openai-api-key",
           "QDRANT_URL": "http://localhost:6333"
         }
       }
     }
   }
   ```

3. Restart Claude Desktop

## Example Usage

### Basic Scraping and Search

```python
# Scrape documentation
await scrape_url(
    url="https://docs.example.com",
    max_depth=3,
    chunk_size=1600
)

# Search indexed content
results = await search(
    query="installation guide",
    collection="documentation",
    limit=5
)
```

### Project-Based Workflow (Enhanced Server)

```python
# Create a project
await create_project(
    name="my_docs",
    description="My documentation project",
    source_urls=["https://docs.example.com"]
)

# Plan scraping
plan = await plan_scraping(
    project_name="my_docs",
    urls=["https://docs.example.com"],
    auto_discover=True
)

# Execute plan
await execute_scraping_plan(plan)

# Search within project
results = await smart_search(
    query="API reference",
    project_name="my_docs"
)
```

## Architecture

### Basic MCP Server

- Direct integration with existing Crawl4AI functionality
- Simple tool-based interface
- Stateless operation
- Suitable for ad-hoc documentation tasks

### Enhanced MCP Server

- Project-based organization
- Intelligent scraping strategies
- Advanced search with reranking
- Designed for managing multiple documentation sources

## Testing

Run the test suite:

```bash
uv run pytest tests/test_mcp_server.py -v
```

Run with coverage:

```bash
uv run pytest tests/test_mcp_server.py --cov=src --cov-report=html
```

## Integration with Other MCP Servers

Both servers can integrate with:

1. **Firecrawl MCP Server** - For advanced web scraping
2. **Qdrant MCP Server** - For vector database operations

To enable integration, ensure the respective MCP servers are running and accessible.

## Performance Considerations

- The basic server is lightweight and suitable for most use cases
- The enhanced server adds overhead but provides better organization for large projects
- Both servers support concurrent operations
- Chunking strategies are optimized for documentation content

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**

   - Ensure `OPENAI_API_KEY` is set in environment
   - Check API key validity

2. **Qdrant Connection Error**

   - Verify Qdrant is running at the specified URL
   - Check Docker: `docker ps | grep qdrant`

3. **Firecrawl Integration Issues**
   - Ensure `FIRECRAWL_API_KEY` is set if using Firecrawl
   - Check API quota and rate limits

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
uv run mcp-server
```

## Future Enhancements

- [ ] Direct integration with Firecrawl MCP server client
- [ ] Direct integration with Qdrant MCP server client
- [ ] Support for additional embedding models
- [ ] Real-time documentation monitoring
- [ ] Automated documentation updates
- [ ] Multi-language support for code-aware chunking
