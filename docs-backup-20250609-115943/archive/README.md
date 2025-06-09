# Documentation Archive

> **Purpose**: Historical documentation preserved for reference  
> **Status**: Archived - These documents are no longer actively maintained  
> **Last Updated**: 2025-06-06

## Overview

This archive contains historical documentation that has been superseded by newer guides or completed its purpose. Documents are organized by their original context and time period.

## Archive Structure

### 📁 research-v1/
**Period**: 2025-06  
**Purpose**: V1 research and analysis documents

Contains comprehensive research conducted to validate architectural decisions for V1:
- **comprehensive-analysis-scoring.md** - Expert analysis with 8.9/10 implementation score
- **external-research-validation.md** - Third-party validation of technology choices
- **performance-analysis-scoring.md** - Detailed performance benchmarks and comparisons
- **web-scraping-architecture-analysis.md** - Architecture assessment and recommendations
- **implementation-task-summary.md** - Task breakdown for V1 implementation
- **README.md** - Original research overview

**Superseded by**: Current feature guides and architecture documentation

### 📁 refactor-v1/
**Period**: 2025-05 to 2025-06  
**Purpose**: V1 refactor planning and implementation guides

Contains the original V1 refactor plans (now completed):
- **01_QDRANT_QUERY_API_MIGRATION.md** - Query API migration plan
- **02_PAYLOAD_INDEXING.md** - Payload indexing implementation
- **03_HNSW_OPTIMIZATION.md** - HNSW parameter tuning guide
- **04_HYDE_IMPLEMENTATION.md** - HyDE implementation plan
- **06_DRAGONFLYDB_CACHE.md** - Cache layer implementation
- **08_COLLECTION_ALIASES.md** - Zero-downtime deployment patterns
- **20_WEEK_BY_WEEK_PLAN.md** - Original 8-week implementation timeline
- **README.md** - Refactor overview

**Superseded by**: 
- [V1 Implementation Plan](../development/V1_IMPLEMENTATION_PLAN.md) (completed)
- [Feature guides](../features/) with integrated implementations
- [Architecture documentation](../architecture/) with current patterns

### 📁 consolidated/
**Period**: December 2024  
**Purpose**: Pre-V1 documentation consolidation

Contains documentation from before the V1 refactor:
- Browser automation guides (now in [user-guides](../user-guides/))
- Crawl4AI configuration (now in [user-guides](../user-guides/))
- Firecrawl evaluation (replaced by Crawl4AI)

### 📁 sprint-2025-05/
**Period**: May 2025  
**Purpose**: Sprint-specific documentation

Contains documentation from the Critical Architecture Cleanup & Unification sprint (Issues #17-28, PR #32):
- Sprint summaries and architectural decisions
- API/SDK integration refactor documentation
- V1 documentation summaries

### 📁 mcp-legacy/
**Period**: Pre-May 2025  
**Purpose**: Legacy MCP server documentation

Contains documentation for the pre-unified MCP server implementation:
- Original MCP guide (pre-unified architecture)
- Legacy architecture patterns
- Enhancement plans (now implemented in unified server)

## Why Archive?

Documents are archived when they:
1. **Complete their purpose** - Research validated, plans executed
2. **Become outdated** - Superseded by new implementations
3. **Consolidate elsewhere** - Content integrated into active guides
4. **Historical value** - Useful for understanding project evolution

## Using Archived Documents

### ⚠️ Important Notes

1. **Not Current** - These documents reflect past states and decisions
2. **For Reference Only** - Do not use for implementation guidance
3. **See Current Docs** - Always refer to active documentation for current practices
4. **Historical Context** - Useful for understanding how the project evolved

### Finding Current Information

If you're looking for current information on topics in these archives:

| Archived Topic | Current Location |
|----------------|------------------|
| Query API Migration | [API Reference](../api/API_REFERENCE.md) |
| Payload Indexing | [Payload Indexing Performance](../PAYLOAD_INDEXING_PERFORMANCE.md) |
| HNSW Optimization | [Vector DB Best Practices](../features/VECTOR_DB_BEST_PRACTICES.md) |
| HyDE Implementation | [HyDE Query Enhancement](../features/HYDE_QUERY_ENHANCEMENT.md) |
| Browser Automation | [Browser Automation Guide](../user-guides/browser-automation.md) |
| Crawl4AI Setup | [Crawl4AI Guide](../user-guides/crawl4ai.md) |
| Cache Implementation | [Performance Guide](../operations/PERFORMANCE_GUIDE.md) |
| Collection Aliases | [Collection Aliases](../COLLECTION_ALIASES.md) |
| MCP Legacy Servers | [Unified MCP Server](../mcp/README.md) |
| Sprint Documentation | [V1 Implementation Plan](../development/V1_IMPLEMENTATION_PLAN.md) |

## Archive Policy

Documents are archived following these principles:

1. **Preserve History** - Maintain record of project evolution
2. **Reduce Confusion** - Keep only current docs in main directories
3. **Enable Learning** - Allow understanding of past decisions
4. **Save Space** - Archive rather than delete valuable content

---

**Need current documentation?** Return to the [Documentation Hub](../README.md)
