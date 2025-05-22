# 🤖 MCP Server Setup Guide - SOTA 2025

> **Complete guide for configuring Qdrant and Firecrawl MCP servers with Claude Desktop/Code**

## 📋 Table of Contents

- [Prerequisites](#-prerequisites)
- [Qdrant MCP Server Setup](#-qdrant-mcp-server-setup)
- [Firecrawl MCP Server Setup](#-firecrawl-mcp-server-setup)
- [Claude Desktop Configuration](#️-claude-desktop-configuration)
- [Testing MCP Integration](#-testing-mcp-integration)
- [Advanced Configuration](#️-advanced-configuration)
- [Troubleshooting](#-troubleshooting)

## 🔧 Prerequisites

### System Requirements

- **Claude Desktop** or **Claude Code** installed
- **Python 3.11+** with `uv` package manager
- **Node.js 18+** with `npm` or `pnpm`
- **Docker Desktop** running (for Qdrant)
- **API Keys**: OpenAI, Firecrawl (optional)

### Environment Setup

```bash
# Install modern package managers
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://get.pnpm.io/install.sh | sh

# Verify installations
uv --version
pnpm --version
node --version
```

## 🔍 Qdrant MCP Server Setup

The Qdrant MCP server enables seamless vector search and database operations directly in Claude.

### 1. Install Qdrant MCP Server

#### Option A: Using uvx (Recommended - 2025)

```bash
# Install globally with uvx (fastest, most reliable)
uvx install mcp-server-qdrant

# Verify installation
uvx mcp-server-qdrant --help
```

#### Option B: Using pip/uv

```bash
# Install with uv (modern Python package manager)
uv tool install mcp-server-qdrant

# Alternative: pip installation
pip install mcp-server-qdrant
```

#### Option C: Development Installation

```bash
# Clone and install from source for latest features
git clone https://github.com/qdrant/mcp-server-qdrant.git
cd mcp-server-qdrant
uv install
uv run mcp-server-qdrant --help
```

### 2. Start Qdrant Database

Ensure your SOTA 2025 Qdrant instance is running:

```bash
# Start optimized Qdrant with our docker-compose.yml
cd /path/to/ai-docs-vector-db-hybrid-scraper
docker-compose up -d

# Verify Qdrant is healthy
curl http://localhost:6333/health
# Expected: {"status":"ok"}

# Check collections (should show 'documents' after scraping)
curl http://localhost:6333/collections
```

### 3. Environment Variables

Create `.env` file for MCP server configuration:

```bash
# Core Qdrant connection
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # Leave empty for local instance

# Collection configuration (matches our SOTA 2025 setup)
COLLECTION_NAME=documents
EMBEDDING_MODEL=text-embedding-3-small
VECTOR_SIZE=1536

# OpenAI configuration for embeddings
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# Advanced SOTA 2025 settings
ENABLE_HYBRID_SEARCH=true
QUANTIZATION_TYPE=int8
DEFAULT_SEARCH_LIMIT=10
```

### 4. Test Qdrant MCP Server

```bash
# Test server directly
uvx mcp-server-qdrant

# Test with specific collection
COLLECTION_NAME=documents uvx mcp-server-qdrant

# Debug mode for troubleshooting
DEBUG=1 uvx mcp-server-qdrant
```

## 🔥 Firecrawl MCP Server Setup

Firecrawl MCP enables real-time web scraping and content extraction in Claude.

### 1. Install Firecrawl MCP Server

#### Option A: Using npx (Recommended)

```bash
# Install and test with npx (no local installation needed)
npx -y firecrawl-mcp --help

# Verify Firecrawl API access
npx -y firecrawl-mcp test
```

#### Option B: Global Installation

```bash
# Install globally with npm/pnpm
npm install -g firecrawl-mcp
# or
pnpm add -g firecrawl-mcp

# Verify installation
firecrawl-mcp --version
```

### 2. Firecrawl API Setup

#### Get API Key

1. Visit [Firecrawl.dev](https://firecrawl.dev)
2. Sign up for an account
3. Navigate to API Keys section
4. Generate new API key
5. Copy key for configuration

#### Test API Access

```bash
# Test API key validity
export FIRECRAWL_API_KEY="your_firecrawl_api_key_here"
curl -X GET "https://api.firecrawl.dev/v0/scrape" \
  -H "Authorization: Bearer $FIRECRAWL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://docs.qdrant.tech"}'
```

### 3. Environment Configuration

Add to your `.env` file:

```bash
# Firecrawl configuration
FIRECRAWL_API_KEY=your_firecrawl_api_key_here
FIRECRAWL_BASE_URL=https://api.firecrawl.dev

# Default scraping options (SOTA 2025 optimized)
DEFAULT_FORMAT=markdown
INCLUDE_TAGS=main,article,section,div[class*="content"]
EXCLUDE_TAGS=nav,footer,aside,script,style
WAIT_FOR=1000
TIMEOUT=30000

# Integration with local Qdrant
AUTO_EMBED_TO_QDRANT=true
QDRANT_COLLECTION=documents
```

## 🖥️ Claude Desktop Configuration

### Configuration File Locations

| Platform    | Configuration Path                                                |
| ----------- | ----------------------------------------------------------------- |
| **Windows** | `%APPDATA%\Claude\claude_desktop_config.json`                     |
| **macOS**   | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| **Linux**   | `~/.config/claude-desktop/config.json`                            |

### Complete SOTA 2025 Configuration

Create or update your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "qdrant": {
      "command": "uvx",
      "args": ["mcp-server-qdrant"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "documents",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "VECTOR_SIZE": "1536",
        "OPENAI_API_KEY": "your_openai_api_key_here",
        "ENABLE_HYBRID_SEARCH": "true",
        "QUANTIZATION_TYPE": "int8",
        "DEFAULT_SEARCH_LIMIT": "10"
      }
    },
    "firecrawl": {
      "command": "npx",
      "args": ["-y", "firecrawl-mcp"],
      "env": {
        "FIRECRAWL_API_KEY": "your_firecrawl_api_key_here",
        "DEFAULT_FORMAT": "markdown",
        "INCLUDE_TAGS": "main,article,section,div[class*=\"content\"]",
        "EXCLUDE_TAGS": "nav,footer,aside,script,style",
        "WAIT_FOR": "1000",
        "TIMEOUT": "30000",
        "AUTO_EMBED_TO_QDRANT": "true",
        "QDRANT_COLLECTION": "documents"
      }
    }
  },
  "globalShortcut": "Ctrl+Shift+C"
}
```

### Alternative Configurations

#### Minimal Configuration (Essential only)

```json
{
  "mcpServers": {
    "qdrant": {
      "command": "uvx",
      "args": ["mcp-server-qdrant"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "documents",
        "OPENAI_API_KEY": "your_openai_api_key_here"
      }
    },
    "firecrawl": {
      "command": "npx",
      "args": ["-y", "firecrawl-mcp"],
      "env": {
        "FIRECRAWL_API_KEY": "your_firecrawl_api_key_here"
      }
    }
  }
}
```

#### Development Configuration (Local paths)

```json
{
  "mcpServers": {
    "qdrant-dev": {
      "command": "python",
      "args": ["-m", "mcp_server_qdrant"],
      "cwd": "/path/to/mcp-server-qdrant",
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "documents",
        "DEBUG": "1"
      }
    }
  }
}
```

## 🧪 Testing MCP Integration

### 1. Restart Claude Desktop

After configuration changes:

1. **Close Claude Desktop completely**
2. **Wait 5 seconds**
3. **Restart Claude Desktop**
4. **Look for MCP connection indicators**

### 2. Verify Server Connections

In Claude Desktop, test the servers:

```bash
# Test Qdrant MCP server
"Show me the collections in my vector database"

# Expected response: Information about your 'documents' collection

# Test Firecrawl MCP server
"Scrape this documentation page: https://docs.qdrant.tech/concepts/collections/"

# Expected response: Markdown content of the scraped page
```

### 3. Test Hybrid Workflow

```bash
# Test integrated workflow
"Search my documentation for 'vector similarity' and if you don't find good results, scrape https://docs.qdrant.tech/concepts/search/ and add it to my knowledge base"

# This should:
# 1. Search existing documents via Qdrant MCP
# 2. If needed, scrape new content via Firecrawl MCP
# 3. Automatically add to vector database
# 4. Return comprehensive results
```

### 4. Performance Testing

```bash
# Test Qdrant performance
curl -X POST "http://localhost:6333/collections/documents/points/search" \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3],
    "limit": 5,
    "with_payload": true
  }'

# Test Firecrawl performance
time npx -y firecrawl-mcp scrape --url "https://docs.qdrant.tech"
```

## ⚙️ Advanced Configuration

### Environment-Specific Settings

#### Production Configuration

```json
{
  "mcpServers": {
    "qdrant-prod": {
      "command": "uvx",
      "args": ["mcp-server-qdrant"],
      "env": {
        "QDRANT_URL": "https://your-qdrant-cloud.qdrant.tech:6333",
        "QDRANT_API_KEY": "your_production_api_key",
        "COLLECTION_NAME": "production_documents",
        "EMBEDDING_MODEL": "text-embedding-3-large",
        "ENABLE_HYBRID_SEARCH": "true",
        "RATE_LIMIT_RPM": "60"
      }
    }
  }
}
```

#### Multiple Collections Setup

```json
{
  "mcpServers": {
    "qdrant-docs": {
      "command": "uvx",
      "args": ["mcp-server-qdrant"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "documents",
        "OPENAI_API_KEY": "your_key_here"
      }
    },
    "qdrant-code": {
      "command": "uvx",
      "args": ["mcp-server-qdrant"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "code_snippets",
        "OPENAI_API_KEY": "your_key_here"
      }
    }
  }
}
```

### Custom Embedding Models

#### Using FastEmbed (Local inference)

```json
{
  "mcpServers": {
    "qdrant-fastembed": {
      "command": "uvx",
      "args": ["mcp-server-qdrant"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "documents",
        "EMBEDDING_PROVIDER": "fastembed",
        "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
        "VECTOR_SIZE": "384"
      }
    }
  }
}
```

#### Using Multiple Embedding Providers

```json
{
  "mcpServers": {
    "qdrant-openai": {
      "command": "uvx",
      "args": ["mcp-server-qdrant"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "openai_docs",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "OPENAI_API_KEY": "your_openai_key"
      }
    },
    "qdrant-local": {
      "command": "uvx",
      "args": ["mcp-server-qdrant"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "local_docs",
        "EMBEDDING_PROVIDER": "fastembed",
        "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5"
      }
    }
  }
}
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### Issue: "MCP server not found"

```bash
# Solution 1: Verify installation
uvx list | grep qdrant
npx -y firecrawl-mcp --version

# Solution 2: Reinstall servers
uvx uninstall mcp-server-qdrant
uvx install mcp-server-qdrant

# Solution 3: Use absolute paths
which uvx  # Use full path in Claude config
```

#### Issue: "Connection refused to Qdrant"

```bash
# Check Qdrant status
docker ps | grep qdrant
curl http://localhost:6333/health

# Restart Qdrant
docker-compose restart qdrant

# Check firewall/networking
netstat -tulpn | grep 6333
```

#### Issue: "Invalid Firecrawl API key"

```bash
# Test API key
curl -H "Authorization: Bearer $FIRECRAWL_API_KEY" \
  "https://api.firecrawl.dev/v0/account"

# Check environment variables in Claude config
echo $FIRECRAWL_API_KEY
```

#### Issue: "Embedding model mismatch"

```bash
# Check collection info
curl http://localhost:6333/collections/documents

# Recreate collection with correct dimensions
curl -X DELETE http://localhost:6333/collections/documents
python src/crawl4ai_bulk_embedder.py  # Will recreate
```

#### Issue: "Slow MCP responses"

```bash
# Enable debug mode
DEBUG=1 uvx mcp-server-qdrant

# Optimize Qdrant settings in docker-compose.yml
# Increase memory allocation
# Enable quantization
```

### Debug Mode Configuration

For troubleshooting, enable debug mode:

```json
{
  "mcpServers": {
    "qdrant-debug": {
      "command": "uvx",
      "args": ["mcp-server-qdrant"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "documents",
        "OPENAI_API_KEY": "your_key_here",
        "DEBUG": "1",
        "LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

### Logs and Monitoring

#### Check MCP Server Logs

```bash
# Qdrant MCP logs
uvx mcp-server-qdrant 2>&1 | tee qdrant-mcp.log

# Firecrawl MCP logs
npx -y firecrawl-mcp 2>&1 | tee firecrawl-mcp.log

# Claude Desktop logs (Windows)
type "%APPDATA%\Claude\logs\claude_desktop.log"

# Claude Desktop logs (macOS)
tail -f ~/Library/Logs/Claude/claude_desktop.log
```

#### Monitor Qdrant Performance

```bash
# Real-time stats
watch -n 1 'curl -s http://localhost:6333/collections/documents | jq'

# Memory usage
docker stats qdrant-sota-2025

# Database size
du -sh ~/.qdrant_data/
```

### Health Check Script

Create a health check script for your MCP setup:

```bash
#!/bin/bash
# health-check-mcp.sh

echo "🔍 MCP Server Health Check"
echo "=========================="

# Check Qdrant
echo "Checking Qdrant..."
if curl -s http://localhost:6333/health | grep -q "ok"; then
    echo "✅ Qdrant: Healthy"
else
    echo "❌ Qdrant: Unhealthy"
fi

# Check MCP servers
echo "Checking MCP servers..."
if uvx mcp-server-qdrant --help > /dev/null 2>&1; then
    echo "✅ Qdrant MCP: Installed"
else
    echo "❌ Qdrant MCP: Not found"
fi

if npx -y firecrawl-mcp --help > /dev/null 2>&1; then
    echo "✅ Firecrawl MCP: Available"
else
    echo "❌ Firecrawl MCP: Not found"
fi

# Check API keys
if [ -n "$OPENAI_API_KEY" ]; then
    echo "✅ OpenAI API Key: Set"
else
    echo "❌ OpenAI API Key: Missing"
fi

if [ -n "$FIRECRAWL_API_KEY" ]; then
    echo "✅ Firecrawl API Key: Set"
else
    echo "⚠️  Firecrawl API Key: Missing (optional)"
fi

echo "=========================="
echo "MCP Health Check Complete"
```

```bash
chmod +x health-check-mcp.sh
./health-check-mcp.sh
```

## 📚 Usage Examples

### Basic Search Operations

```bash
# Search documentation
"Search my documentation for 'vector quantization'"

# Semantic search with filters
"Find information about HNSW indexing in vector databases"

# Multi-vector search
"Compare different embedding models mentioned in my docs"
```

### Content Addition Workflows

```bash
# Scrape and add single page
"Scrape https://docs.qdrant.tech/concepts/points/ and add it to my knowledge base"

# Bulk content addition
"Scrape the entire FastAPI documentation and add it to my vector database"

# Selective scraping
"Scrape only the API reference sections from https://docs.example.com"
```

### Advanced Hybrid Operations

```bash
# Search-then-scrape workflow
"Search for 'API rate limiting' in my docs, and if you don't find enough information, scrape the relevant pages from the FastAPI documentation"

# Cross-reference and update
"Compare my current documentation on vector search with the latest Qdrant docs and update my knowledge base with any new information"

# Intelligent content curation
"Find all mentions of 'embedding models' in my knowledge base and create a comprehensive summary, filling any gaps by scraping recent research papers"
```

---

🎉 **Your SOTA 2025 MCP server setup is now complete!**

The hybrid Crawl4AI + Firecrawl + Qdrant architecture provides the most cost-effective and powerful documentation processing system available.
