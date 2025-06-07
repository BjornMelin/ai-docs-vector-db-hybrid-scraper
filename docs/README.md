# AI Documentation Vector DB - Documentation Index

Welcome to the AI Documentation Vector DB documentation. This guide helps you navigate our comprehensive documentation suite.

## 📚 Documentation Structure

### 🚀 Getting Started

- [**QUICK_START.md**](./QUICK_START.md) - Get up and running in 5 minutes
- [**Project README**](../README.md) - Project overview and features

### 📖 User Guides

Comprehensive guides consolidating all implementation, configuration, and troubleshooting:

- [**Browser Automation Guide**](./user-guides/browser-automation.md) - Complete 5-tier browser automation system
- [**Crawl4AI Guide**](./user-guides/crawl4ai.md) - Complete Crawl4AI configuration, implementation, and troubleshooting

### 🏗️ Architecture

Core architectural patterns and design decisions:

- [**Unified Configuration**](./architecture/UNIFIED_CONFIGURATION.md) - Centralized config with Pydantic v2
- [**Client Management**](./architecture/CENTRALIZED_CLIENT_MANAGEMENT.md) - Singleton pattern for API clients
- [**System Overview**](./architecture/SYSTEM_OVERVIEW.md) - High-level architecture and data flow

### ✨ Features

Detailed implementation guides for key features:

- [**Advanced Search**](./features/ADVANCED_SEARCH_IMPLEMENTATION.md) - Hybrid search with BGE reranking
- [**Embedding Models**](./features/EMBEDDING_MODEL_INTEGRATION.md) - Multi-model support and selection
- [**Enhanced Chunking**](./features/ENHANCED_CHUNKING_GUIDE.md) - Three-tier chunking strategies
- [**Reranking**](./features/RERANKING_GUIDE.md) - BGE-reranker implementation
- [**Vector DB Best Practices**](./features/VECTOR_DB_BEST_PRACTICES.md) - Production Qdrant patterns

### 🔧 Operations

Production deployment and maintenance:

- [**Performance Guide**](./operations/PERFORMANCE_GUIDE.md) - Optimization strategies and tuning
- [**Troubleshooting**](./operations/TROUBLESHOOTING.md) - Common issues and solutions
- [**Monitoring**](./operations/MONITORING.md) - Metrics and observability

### 💻 Development

Planning and development resources:

- [**V1 Implementation Plan**](./development/V1_IMPLEMENTATION_PLAN.md) - 8-week roadmap with examples
- [**Testing Strategy**](./development/TESTING_QUALITY_ENHANCEMENTS.md) - 95%+ coverage targets
- [**Architecture Improvements**](./development/ARCHITECTURE_IMPROVEMENTS.md) - Future enhancements

### 🔌 MCP Integration

Model Context Protocol server documentation:

- [**MCP Overview**](./mcp/README.md) - Unified MCP server guide
- [**Setup Guide**](./mcp/SETUP.md) - Configuration and deployment
- [**Migration Guide**](./mcp/MIGRATION_GUIDE.md) - Upgrading to unified server

### 🔬 Research

Deep dives and technical research:

- [**Chunking Research**](./research/chunking/CHUNKING_RESEARCH.md) - AST-based chunking analysis

## 🎯 Quick Links by Use Case

### "I want to..."

- **Set up the project** → [QUICK_START.md](./QUICK_START.md)
- **Understand the architecture** → [System Overview](./architecture/SYSTEM_OVERVIEW.md)
- **Implement search** → [Advanced Search](./features/ADVANCED_SEARCH_IMPLEMENTATION.md)
- **Optimize performance** → [Performance Guide](./operations/PERFORMANCE_GUIDE.md)
- **Debug an issue** → [Troubleshooting](./operations/TROUBLESHOOTING.md)
- **Contribute code** → [V1 Implementation Plan](./development/V1_IMPLEMENTATION_PLAN.md)

## 📊 Documentation Status

| Category | Docs | Status |
|----------|------|--------|
| User Guides | 2 | ✅ Current |
| Architecture | 3 | ✅ Current |
| Features | 5 | ✅ Current |
| Operations | 3 | ✅ Current |
| Development | 3 | ✅ Active |
| MCP | 3 | ✅ Current |
| Research | 1 | ✅ Current |

## 🗂️ Archive

Historical documentation is preserved in the `archive/` directory:

- `consolidated/` - Documentation consolidated into User Guides (Dec 2024)
- `sprint-2025-05/` - Completed sprint documentation
- `mcp-legacy/` - Pre-unified MCP server docs

## 📝 Documentation Standards

All documentation follows these conventions:

- **Status**: Current/Planning/Deprecated at top
- **Last Updated**: Date of last significant update
- **Related Docs**: Cross-references to related guides
- **Code Examples**: Working examples with current architecture

## 🔄 Recent Updates

- **2025-05-26**: Major reorganization following PR #32 architectural changes
- **2025-05-25**: Unified configuration and service layer implementation
- **2025-05-24**: Advanced search and reranking features documented

---

For questions or improvements, please open an issue on [GitHub](https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper).
