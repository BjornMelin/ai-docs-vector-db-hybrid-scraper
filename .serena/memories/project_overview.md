# AI Docs Vector DB Hybrid Scraper - Project Overview

## Purpose
Enterprise-grade AI RAG system with hybrid documentation scraping combining Crawl4AI (bulk) + Firecrawl MCP (on-demand) with Qdrant vector database for Claude Desktop/Code integration.

## Tech Stack
- **Python**: 3.11-3.13 with uv package management
- **Web Framework**: FastAPI with Starlette and Uvicorn
- **Data Validation**: Pydantic v2+ and Pydantic Settings
- **Vector Database**: Qdrant for vector storage and similarity search
- **Cache**: DragonflyDB for caching layer
- **Scraping**: Crawl4AI for bulk operations + Firecrawl MCP for on-demand
- **Embeddings**: Multi-provider support (OpenAI, FastEmbed)
- **CLI**: Click with Rich for enhanced user interface
- **Testing**: pytest with coverage, hypothesis for property-based testing
- **Linting**: ruff for formatting and linting
- **Type Checking**: mypy

## Architecture Modes
- **Simple Mode**: 25K lines for basic usage
- **Enterprise Mode**: 70K lines with full monitoring, A/B testing, predictive scaling

## Key Features
- Hybrid Vector Search with dense + sparse vectors
- Query Enhancement with HyDE (Hypothetical Document Embeddings)
- 5-Tier Browser Automation (HTTP → Playwright)
- Circuit Breaker Patterns with adaptive thresholds
- Multi-Level Caching with 86% hit rate
- Zero-Maintenance infrastructure with self-healing

## Performance Achievements
- 887.9% throughput increase
- 50.9% latency reduction (P95)
- 83% memory usage reduction
- 94% configuration reduction (18 → 1 file)