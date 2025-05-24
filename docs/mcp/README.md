# MCP Server Documentation

This directory contains all documentation related to the Model Context Protocol (MCP) server implementations for the AI Documentation Vector DB project.

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

## Current Implementation Status

The project currently has two main MCP server implementations:

1. **mcp_server_refactored.py**: Basic search and indexing functionality
2. **enhanced_mcp_server_refactored.py**: Extended project management capabilities

Both servers properly utilize the service layer architecture for:
- Direct SDK integration (no MCP proxying)
- Proper abstraction and testability
- Efficient resource management
- Comprehensive error handling

## Quick Start

1. Install dependencies:
   ```bash
   uv sync
   uv run crawl4ai-setup
   ```

2. Start services:
   ```bash
   ./scripts/start-services.sh
   ```

3. Run MCP server:
   ```bash
   # Basic server
   uv run python src/mcp_server_refactored.py
   
   # Enhanced server
   uv run python src/enhanced_mcp_server_refactored.py
   ```

4. Configure Claude Desktop (see [Setup Guide](02_SETUP.md) for details)

## Key Features

- 🔍 **Hybrid Search**: Dense + sparse vectors with RRF fusion
- 🌐 **Web Scraping**: Crawl4AI bulk + Firecrawl on-demand
- 📊 **Vector Management**: Qdrant with quantization
- 🧮 **Smart Embeddings**: Auto-selection with cost optimization
- 🔄 **Caching**: Two-tier caching for performance
- 🛡️ **Security**: Input validation and rate limiting
- 📈 **Monitoring**: Comprehensive metrics and logging

## Architecture Principles

1. **Service Layer Pattern**: All business logic in services, MCP as thin wrapper
2. **Direct SDK Usage**: No MCP proxying, direct API integration
3. **Resource Efficiency**: Connection pooling, batch processing
4. **Error Resilience**: Circuit breakers, exponential backoff
5. **Cost Optimization**: Smart model selection, caching
6. **Security First**: Input validation, rate limiting

## Contributing

When working on MCP server improvements:
1. Follow the service layer pattern
2. Use direct SDK integration
3. Implement proper error handling
4. Add comprehensive tests
5. Update relevant documentation

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for general contribution guidelines.