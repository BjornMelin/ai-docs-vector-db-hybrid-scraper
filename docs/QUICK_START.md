# Quick Start Guide

Get up and running with AI Documentation Vector DB in 5 minutes!

## Prerequisites

- Python 3.13+
- Docker Desktop (for Qdrant)
- OpenAI API key (for embeddings)
- Git

## 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper.git
cd ai-docs-vector-db-hybrid-scraper

# Install dependencies with uv
pip install uv
uv sync

# Set up Crawl4AI
uv run crawl4ai-setup

# Copy environment template
cp .env.example .env
```

## 2. Configure API Keys

Edit `.env` and add your API keys:

```bash
OPENAI_API_KEY=sk-...
FIRECRAWL_API_KEY=fc-...  # Optional, for premium features
```

## 3. Start Services

```bash
# Start Qdrant vector database
./scripts/start-services.sh

# Verify it's running
curl http://localhost:6333/health
```

## 4. Run Your First Scrape

```bash
# Scrape documentation (uses sites from config/documentation-sites.json)
uv run python src/crawl4ai_bulk_embedder.py

# Check what was indexed
uv run python src/manage_vector_db.py stats
```

## 5. Search Your Documentation

```bash
# Command-line search
uv run python src/manage_vector_db.py search "how to implement authentication" --limit 5

# Start MCP server for Claude Desktop
uv run python src/unified_mcp_server.py
```

## 6. Integrate with Claude Desktop

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "ai-docs-vector-db": {
      "command": "uv",
      "args": ["run", "python", "src/unified_mcp_server.py"],
      "cwd": "/path/to/ai-docs-vector-db-hybrid-scraper",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

## Common Commands

```bash
# Scraping
uv run python src/crawl4ai_bulk_embedder.py              # Bulk scrape all sites
uv run python src/crawl4ai_bulk_embedder.py --url URL    # Scrape specific URL

# Database Management
uv run python src/manage_vector_db.py stats              # Show statistics
uv run python src/manage_vector_db.py search "query"     # Search documents
uv run python src/manage_vector_db.py list-collections   # List all collections
uv run python src/manage_vector_db.py delete COLLECTION   # Delete collection

# Development
uv run pytest --cov=src                                   # Run tests
ruff check . --fix && ruff format .                      # Lint and format

# Services
docker-compose up -d                                      # Start Qdrant
docker-compose down                                       # Stop Qdrant
docker logs qdrant-vector-db -f                          # View logs
```

## Configuration

### Adding Documentation Sites

Edit `config/documentation-sites.json`:

```json
{
  "sites": [
    {
      "name": "Your Docs",
      "url": "https://docs.example.com",
      "collection_name": "your_docs",
      "crawl_params": {
        "max_pages": 100,
        "exclude_patterns": ["*/api/*", "*/changelog/*"]
      }
    }
  ]
}
```

### Advanced Configuration

See [Unified Configuration Guide](./architecture/UNIFIED_CONFIGURATION.md) for:

- Environment variables
- Service-specific settings
- Performance tuning
- Security options

## Next Steps

- 📖 Read the [System Overview](./architecture/SYSTEM_OVERVIEW.md)
- 🔍 Implement [Advanced Search](./features/ADVANCED_SEARCH_IMPLEMENTATION.md)
- 🚀 Optimize with the [Performance Guide](./operations/PERFORMANCE_GUIDE.md)
- 🐛 Check [Troubleshooting](./operations/TROUBLESHOOTING.md) if you hit issues

## Getting Help

- Check the [documentation index](./README.md)
- Open an [issue on GitHub](https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper/issues)
- Review [common issues](./operations/TROUBLESHOOTING.md)

---

Happy searching! 🚀
