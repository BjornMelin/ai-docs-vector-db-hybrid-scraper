# 🚀 AI Documentation Vector Database Hybrid Scraper

[![GitHub Stars](https://img.shields.io/github/stars/BjornMelin/ai-docs-vector-db-hybrid-scraper?style=social)](https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

> **Ultra-fast, cost-effective hybrid documentation scraping system for AI-powered development workflows**

## 🌟 Overview

This project implements an optimal hybrid architecture combining:
- **[Crawl4AI](https://github.com/unclecode/crawl4ai)** for lightning-fast bulk documentation scraping (4-6x faster than alternatives)
- **[Firecrawl MCP](https://github.com/mendableai/firecrawl-mcp-server)** for seamless on-demand scraping in Claude Desktop/Code
- **[Qdrant](https://qdrant.tech/)** vector database with persistent local storage
- **OpenAI Embeddings** for high-quality semantic search

## 🎯 Why This Hybrid Approach?

| Component | Use Case | Benefits |
|-----------|----------|----------|
| **Crawl4AI** | Bulk documentation scraping | 🚀 4-6x faster • 💰 Free/open-source • ⚡ Async concurrent processing |
| **Firecrawl MCP** | On-demand page additions | 🔗 Claude Desktop integration • 🌐 Complex JS handling • 🎯 Structured extraction |
| **Qdrant** | Vector storage | 🏠 Local persistence • 🔍 Semantic search • 📊 Efficient indexing |

## ✨ Key Features

- **⚡ Ultra-Fast Scraping**: Crawl4AI processes 100+ documentation pages in minutes
- **💰 Cost-Effective**: Zero API costs for bulk operations vs paid alternatives
- **🧠 AI-Optimized**: Purpose-built for LLM/RAG workflows with clean markdown output  
- **🔄 Hybrid Workflow**: Bulk population + on-demand additions
- **🏠 Local-First**: All data stored persistently in your WSL environment
- **🤖 Claude Integration**: Seamless access via MCP servers in Claude Desktop/Code

## 🚀 Quick Start

### Prerequisites
- Windows 11 with WSL2
- Docker Desktop with WSL2 integration
- Node.js 18+ and Python 3.8+
- OpenAI API key
- Firecrawl API key (for MCP only)

### 1. Clone and Setup
```bash
git clone https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper.git
cd ai-docs-vector-db-hybrid-scraper
chmod +x setup.sh
./setup.sh
```

### 2. Start Services
```bash
./scripts/start-services.sh
```

### 3. Run Initial Documentation Scraping
```bash
export OPENAI_API_KEY="your_openai_api_key"
python src/crawl4ai_bulk_embedder.py
```

### 4. Configure Claude Desktop
Add to `%APPDATA%\Claude\claude_desktop_config.json`:
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

## 📁 Project Structure

```
ai-docs-vector-db-hybrid-scraper/
├── 📄 README.md                    # This file
├── 🐳 docker-compose.yml           # Qdrant service configuration
├── ⚙️ setup.sh                    # One-command setup script
├── 📋 requirements.txt             # Python dependencies
├── 📊 config/
│   ├── documentation-sites.json   # Sites to scrape configuration
│   └── claude-mcp-config.json     # MCP server configuration template
├── 🐍 src/
│   ├── crawl4ai_bulk_embedder.py  # Main bulk scraping engine
│   ├── crawl_single_site.py       # Single site scraper
│   ├── manage_vector_db.py        # Database management utilities
│   └── performance_test.py        # Speed benchmark tools
├── 📜 scripts/
│   ├── start-services.sh          # Service startup script
│   ├── update-documentation.sh    # Automated update script
│   └── cleanup.sh                 # Maintenance utilities
├── 📖 docs/
│   ├── INSTALLATION.md            # Detailed setup guide
│   ├── USAGE.md                   # Usage examples
│   └── TROUBLESHOOTING.md         # Common issues & solutions
└── 🧪 examples/
    ├── basic-search.py            # Search examples
    └── claude-integration.md      # Claude usage examples
```

## 🔧 Configuration

### Documentation Sites
Edit `config/documentation-sites.json` to customize which sites to scrape:
```json
{
  "sites": [
    {
      "name": "Qdrant Documentation",
      "url": "https://docs.qdrant.tech/",
      "max_pages": 100,
      "priority": "high"
    }
  ]
}
```

### Performance Tuning
Adjust concurrent crawling in `src/crawl4ai_bulk_embedder.py`:
```python
MAX_CONCURRENT_CRAWLS = 10  # Adjust based on your system
CHUNK_SIZE = 2000          # Characters per embedding chunk
```

## 📊 Performance Benchmarks

Based on real-world testing:

| Metric | Crawl4AI | Firecrawl |
|--------|----------|-----------|
| **Speed** | 4-6x faster | Baseline |
| **Cost** | $0 (open-source) | API costs |
| **Concurrency** | 10+ simultaneous | Limited |
| **Customization** | Full control | API constraints |

## 🎯 Usage Examples

### Bulk Documentation Scraping
```python
from src.crawl4ai_bulk_embedder import AdvancedDocumentationScraper

scraper = AdvancedDocumentationScraper()
await scraper.bulk_scrape_documentation_sites(DOCUMENTATION_SITES)
```

### Search Your Knowledge Base
```python
from src.manage_vector_db import VectorDBManager

manager = VectorDBManager()
manager.search_similar("vector database operations", limit=5)
```

### Claude Desktop Integration
```
# In Claude Desktop/Code:
"Search my documentation for information about async web crawling"
"Add this new documentation page to my knowledge base: [URL]"
```

## 🛠️ Management Commands

```bash
# Check database statistics
python src/manage_vector_db.py stats

# Search documentation
python src/manage_vector_db.py search "your query here"

# Add a new documentation site
python src/crawl_single_site.py "https://docs.example.com/" 50

# Update existing documentation
./scripts/update-documentation.sh

# Performance test
python src/performance_test.py
```

## 🐳 Docker Support

The project includes Docker configuration for easy deployment:
```bash
# Start all services with Docker Compose
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs qdrant
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'feat: add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Crawl4AI](https://github.com/unclecode/crawl4ai) - Ultra-fast web crawling engine
- [Firecrawl](https://github.com/mendableai/firecrawl) - AI-powered web scraping
- [Qdrant](https://github.com/qdrant/qdrant) - High-performance vector database
- [Claude MCP](https://modelcontextprotocol.io/) - Model Context Protocol integration

## 🔗 Related Projects

- [Crawl4AI Official Documentation](https://docs.crawl4ai.com/)
- [Firecrawl MCP Server](https://docs.firecrawl.dev/mcp)
- [Qdrant MCP Server](https://github.com/qdrant/mcp-server-qdrant)

---

⭐ **Star this repository if it helps your AI development workflow!**

Built with ❤️ for the AI developer community
