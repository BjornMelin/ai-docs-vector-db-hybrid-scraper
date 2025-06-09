# Quick Start Guide

> **Status**: Active  
> **Last Updated**: 2025-01-09  
> **Purpose**: 5-minute setup guide for new users  
> **Audience**: Anyone wanting to start using the system immediately

Get up and running with AI Documentation Vector DB in 5 minutes! This guide will have you searching documents and scraping websites quickly.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.13+** - Download from [python.org](https://python.org)
- **Docker Desktop** - For the Qdrant vector database
- **OpenAI API key** - For embeddings (get from [OpenAI](https://platform.openai.com/api-keys))
- **Git** - For cloning the repository

## Step 1: Clone and Install

```bash
# Clone the repository
git clone https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper.git
cd ai-docs-vector-db-hybrid-scraper

# Install dependencies with uv
pip install uv
uv sync

# Set up Crawl4AI for web scraping
uv run crawl4ai-setup

# Copy environment template
cp .env.example .env
```

## Step 2: Configure API Keys

Edit the `.env` file and add your API keys:

```bash
# Required for embeddings
OPENAI_API_KEY=sk-your-openai-key-here

# Optional for premium features
FIRECRAWL_API_KEY=fc-your-firecrawl-key-here
```

## Step 3: Start Services

```bash
# Start Qdrant vector database
./scripts/start-services.sh

# Verify it's running (should return "ok")
curl localhost:6333/health
```

## Step 4: Start the MCP Server

```bash
# Start the unified MCP server
uv run python src/unified_mcp_server.py
```

The server will start on the default port and show available MCP tools.

## Step 5: Try Your First Search

With the server running, you can now use the MCP tools to:

### Search Documents

```bash
# Search for content across indexed documents
mcp search --query "your search terms"
```

### Scrape a Website

```bash
# Extract content from a webpage
mcp scrape --url "https://example.com"
```

### Add Documents

```bash
# Index new documents for searching
mcp add-documents --path "/path/to/documents"
```

## Quick Usage Examples

### Example 1: Search Technical Documentation

```bash
mcp search --query "how to configure embeddings" --limit 5
```

### Example 2: Scrape and Index a Website

```bash
# Scrape content and automatically index it
mcp scrape --url "https://docs.example.com" --auto-index
```

### Example 3: Enhanced Search with HyDE

```bash
# Use AI-enhanced search for better results
mcp advanced-search --query "vector database optimization" --use-hyde
```

## Configuration Quick Reference

| Setting | Purpose | Default |
|---------|---------|---------|
| `OPENAI_API_KEY` | Embedding generation | Required |
| `QDRANT_HOST` | Vector database location | `localhost` |
| `QDRANT_PORT` | Vector database port | `6333` |
| `EMBEDDING_MODEL` | Model for embeddings | `text-embedding-3-small` |
| `CHUNK_SIZE` | Document chunk size | `1000` |

## Troubleshooting

### Server Won't Start

- **Check Python version**: `python --version` (needs 3.13+)
- **Verify dependencies**: `uv sync`
- **Check ports**: Ensure port 6333 is available for Qdrant

### Can't Connect to Qdrant

- **Start services**: `./scripts/start-services.sh`
- **Check Docker**: `docker ps` should show qdrant container
- **Test connection**: `curl localhost:6333/health`

### Search Returns No Results

- **Check if documents are indexed**: Use `mcp list-collections`
- **Verify embeddings**: Check that `OPENAI_API_KEY` is set correctly
- **Try broader search terms**: Start with general queries

## Next Steps

Once you're up and running:

1. **[Search & Retrieval](./search-and-retrieval.md)** - Learn advanced search techniques
2. **[Web Scraping](./web-scraping.md)** - Master the 5-tier scraping system  
3. **[Examples & Recipes](./examples-and-recipes.md)** - Real-world use cases
4. **[Troubleshooting](./troubleshooting.md)** - Solutions for common issues

## Need Help?

- **User Issues**: Check [troubleshooting guide](./troubleshooting.md)
- **Examples**: See [examples and recipes](./examples-and-recipes.md)
- **Developer Integration**: Visit [../developers/](../developers/)
- **Deployment**: See [../operators/](../operators/)

---

*🎉 Congratulations! You now have a powerful AI-enhanced document search and web scraping system running locally.*
