# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Documentation Enhancements**:
  - Added comprehensive Google-style docstrings to major service classes
  - Enhanced `EmbeddingManager` with detailed method documentation
  - Added docstrings to `Crawl4AIProvider` Memory-Adaptive Dispatcher methods
  - Updated README.md with current features and MCP Server integration section
  - Modernized architecture documentation to reflect current AI-powered features
  - Updated MkDocs site description for better clarity

### Changed
- Comprehensive dependency documentation suite:
  - `DEPENDENCY_UPGRADE_GUIDE.md` - Detailed guide for all dependency changes
  - `DEPENDENCY_CHANGES_SUMMARY.md` - Quick reference for updates
  - `MIGRATION_CHECKLIST.md` - Step-by-step migration checklist
  - `dependencies/index.md` - Central hub for dependency documentation
- New resilience and performance dependencies:
  - `tenacity` (9.1.0) - Advanced retry patterns with exponential backoff
  - `slowapi` (0.1.9) - Rate limiting for FastAPI endpoints
  - `purgatory-circuitbreaker` (0.7.2) - Distributed circuit breakers
  - `aiocache` (0.12.0) - Async-first caching with Redis backend
- AI/ML enhancement dependencies:
  - `FlagEmbedding` (1.3.5) - 2-3x faster embeddings than OpenAI
  - `pydantic-ai` (0.3.6) - AI-enhanced validation
  - `scikit-learn` (1.5.1) - DBSCAN clustering for documents
- Testing and development tools:
  - `pytest-benchmark` (5.1.0) - Performance benchmarking
  - `respx` (0.22.0) - Modern HTTP mocking for httpx
  - `schemathesis` (3.21.0) - Property-based API testing
  - `taskipy` (1.14.0) - Task runner integration
  - `watchdog` (6.0.0) - File system monitoring
  - `dependency-injector` (4.48.1) - Dependency injection framework
  - `pydeps` (3.0.1) - Dependency visualization

### Changed
- **FastAPI**: Removed `[standard]` extra, now requires separate `httpx` installation
- **NumPy**: Now supports 2.x versions (`numpy>=1.26.0,<3.0.0`) for better performance
- **psutil**: Downgraded from 7.0.0 to 6.0.0 for taskipy compatibility
- Updated dependencies via Dependabot:
  - `faker`: 36.1.0 → 37.4.0 (#159)
  - `mutmut`: 2.5.1 → 3.3.0 (#158)
  - `pyarrow`: 18.1.0 → 20.0.0 (#160)
  - `cachetools`: 5.3.0 → 6.1.0 (#161)
  - `starlette`: 0.41.0 → 0.47.0 (#162)
  - `lxml`: 5.3.0 → 6.0.0 (#164)
  - `prometheus-client`: 0.21.1 → 0.22.1 (#144)
- CI/CD dependency updates:
  - `actions/setup-python`: v4 → v5 (#157)
  - `actions/cache`: v3 → v4 (#156)
  - `actions/github-script`: v6 → v7 (#154)

### Fixed
- HTTP client conflicts between FastAPI and httpx
- Compatibility issues with Python 3.13
- Test mocking migration from aioresponses to respx

### Security
- All dependencies scanned and updated to latest secure versions
- Added `defusedxml` (0.7.1) for secure XML parsing
- No known vulnerabilities as of July 2025

## [0.1.0] - 2025-06-30

### Added
- Initial release with Portfolio ULTRATHINK transformation
- Dual-mode architecture (Simple/Enterprise)
- 5-tier browser automation system
- Hybrid vector search with BGE reranking
- FastMCP 2.0 server with 25+ tools
- Comprehensive monitoring and observability stack

### Performance
- 887.9% throughput increase
- 50.9% latency reduction (P95)
- 83% memory usage reduction
- 95% circular dependency elimination

### Architecture
- 94% configuration file reduction (18 → 1 file)
- 87.7% ClientManager complexity reduction
- Zero high-severity security vulnerabilities
- Modern Pydantic Settings 2.0 configuration

[Unreleased]: https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper/releases/tag/v0.1.0