# TEST_CLEANUP

This file lists outdated or failing unit tests in tests/unit/ that require cleanup
and rewriting to align with current src/ implementations.

## ✅ COMPLETED: Config Models and Enums

**Completed work:**
- ✅ Created 16 comprehensive test files for `src/config/models.py` (206 tests, 94% coverage)
- ✅ Created 1 test file for `src/config/enums.py` (45 tests, 100% coverage)
- ✅ All Pydantic v2 configuration models fully tested with validation scenarios

## ✅ COMPLETED: Config Core Modules

**Completed work:**
- ✅ Removed outdated config test files (test_config.py, test_rate_limiting_config.py, test_unified_config.py)
- ✅ Created 5 comprehensive test files for config core modules:
  - `test_loader.py` - ConfigLoader class with 22 test methods covering environment merging and documentation site loading
  - `test_schema.py` - ConfigSchemaGenerator class tests (simplified due to Pydantic v2 API changes)  
  - `test_validator.py` - ConfigValidator class with 35 test methods covering API key validation and service connections
  - `test_migrator.py` - ConfigMigrator class with 35 test methods covering configuration migration between versions
  - `test_cli.py` - CLI commands with 38 test methods covering argument parsing and command execution
- ✅ Fixed implementation bugs discovered during testing (URL comparison, field name mismatches, validation logic)
- ✅ All config core tests passing with comprehensive validation scenarios

## ✅ COMPLETED: MCP Models

**Completed work:**
- ✅ Created 2 test files for MCP models (49 tests, 100% coverage)
- ✅ test_requests.py covers all MCP request models
- ✅ test_responses.py covers SearchResult and CrawlResult models

## ✅ COMPLETED: Core Modules

**Completed work:**
- ✅ Created 4 test files for core modules:
  - `test_constants.py` - 25 test methods covering all constant definitions including timeouts, limits, HNSW configs, cache settings  
  - `test_decorators.py` - 47 test methods covering retry_async, circuit_breaker, handle_mcp_errors, validate_input, RateLimiter
  - `test_errors.py` - 30 test methods covering error hierarchy, utility functions, and Pydantic integration
  - `test_utils.py` - 20 test methods covering async-to-sync conversion utilities
- ✅ Total: 112 test methods, 1,637 lines of code
- ✅ All core tests passing with comprehensive validation scenarios

## ✅ COMPLETED: Infrastructure Module

**Completed work:**
- ✅ Created comprehensive test file for `src/infrastructure/client_manager.py`:
  - `test_client_manager.py` - 52 test methods covering all classes and functionality:
    * ClientState enum and ClientHealth dataclass tests
    * ClientManagerConfig with Pydantic validation tests
    * CircuitBreaker implementation with failure/recovery scenarios
    * ClientManager singleton pattern and factory method tests
    * Client creation, health checks, and lifecycle management
    * Async context manager and concurrency tests
    * Integration tests with full lifecycle validation
- ✅ 82% test coverage for infrastructure module
- ✅ All infrastructure tests passing with comprehensive scenarios

## ✅ COMPLETED: Unified MCP Server

**Completed work:**
- ✅ Created comprehensive test file for `src/unified_mcp_server.py`:
  - `test_unified_mcp_server.py` - 35 test methods covering all functionality:
    * Streaming configuration validation with various scenarios
    * Configuration validation with API keys and service connections
    * Lifespan context manager with initialization/cleanup
    * Environment variable handling for different transport types
    * Server configuration and module structure validation
    * Error handling and import management
    * Integration points with client manager and tool registration
- ✅ 85% test coverage for unified MCP server
- ✅ All unified server tests passing with proper mocking for external dependencies

## MCP Tools and Registry (Remaining)

**Test files to remove:**
- tests/unit/mcp/test_utilities_tools.py
- tests/unit/mcp/test_search_tools.py
- tests/unit/mcp/test_collections_tools.py
- tests/unit/mcp/test_unified_server.py
- tests/unit/mcp/test_streaming_edge_cases.py
- tests/unit/mcp/test_documents_tools.py
- tests/unit/mcp/test_tool_modules.py
- tests/unit/mcp/test_cache_tools.py
- tests/unit/mcp/test_tool_registry.py
- tests/unit/mcp/test_embeddings_tools.py
- tests/unit/mcp/test_streaming_mocks.py
- tests/unit/mcp/test_validate_configuration.py

**Findings:**
- Legacy tests fail with `ModuleNotFoundError: No module named 'mcp.server'`
- Tests rely on deprecated fastmcp APIs that have been refactored

**Actionable TODOs:**

- [ ] Remove outdated MCP test files (all files listed above)
- [ ] Create new unit tests for `src/mcp/tool_registry.py` to verify:
  - Registry initialization, tool registration, lookup, and error handling
- [ ] Create new unit tests for each module in `src/mcp/tools/` to verify:
  - Public helper function imports and basic execution with mock inputs for utilities, collections, search, embeddings, documents, and cache
  - Proper error handling for invalid parameters or missing dependencies

## Services group

**Test files:**

- tests/unit/services/test_embedding_providers.py
- tests/unit/services/test_qdrant_client.py
- tests/unit/services/test_rate_limiter.py
- tests/unit/services/test_base.py
- tests/unit/services/test_qdrant_collections.py
- tests/unit/services/test_hyde_config.py
- tests/unit/services/test_payload_indexing.py
- tests/unit/services/test_browser_playwright_adapter.py
- tests/unit/services/test_qdrant_indexing.py
- tests/unit/services/test_cache_embedding.py
- tests/unit/services/test_browser_action_schemas.py
- tests/unit/services/test_deployment_ab_testing.py
- tests/unit/services/test_hyde_engine_comprehensive.py
- tests/unit/services/test_hnsw_coverage.py
- tests/unit/services/test_openai_coverage.py
- tests/unit/services/test_hyde_generator.py
- tests/unit/services/test_crawl_providers.py
- tests/unit/services/test_hyde_cache_comprehensive.py
- tests/unit/services/test_cache_manager.py
- tests/unit/services/test_hyde_generator_comprehensive.py
- tests/unit/services/test_qdrant_alias_manager.py
- tests/unit/services/test_search_models.py
- tests/unit/services/test_hnsw_optimizer.py
- tests/unit/services/test_embedding_coverage.py
- tests/unit/services/test_logging_config.py
- tests/unit/services/test_cache_local.py
- tests/unit/services/test_hnsw_optimizer_simple.py
- tests/unit/services/test_deployment_blue_green.py
- tests/unit/services/test_fastembed_coverage.py
- tests/unit/services/test_qdrant_search.py
- tests/unit/services/test_cache_search.py
- tests/unit/services/test_base_coverage.py
- tests/unit/services/test_dragonfly_cache.py
- tests/unit/services/test_qdrant_documents.py
- tests/unit/services/test_vector_db_client.py
- tests/unit/services/test_browser_coverage.py
- tests/unit/services/test_browser_crawl4ai_adapter.py
- tests/unit/services/test_crawl_manager_comprehensive.py
- tests/unit/services/test_embedding_final_coverage.py
- tests/unit/services/test_qdrant_service_facade.py
- tests/unit/services/test_browser_use_adapter.py

**Findings:**

- Tests in this group fail with permission errors due to asyncio event loop creation in container environments.
- Many tests assume legacy combined service implementation and cover outdated workflows, making them incompatible with current modular `src/services` structure.
- Coverage for `src/services` cannot be evaluated because tests error out before execution.

**Actionable TODOs:**

- [ ] Remove the `tests/unit/services/` directory and all contained test files.
- [ ] Create new unit tests for each module in `src/services`, including:
  - core modules (`base.py`, `errors.py`, `logging_config.py`)
  - utilities (`rate_limiter.py`, `hnsw_optimizer.py`, `search_models.py`)
  - crawling providers and manager (`crawling/base.py`, `crawling/manager.py`, `crawling/crawl4ai_provider.py`, `crawling/firecrawl_provider.py`)
  - vector_db components (`vector_db/client.py`, `vector_db/collections.py`, `vector_db/documents.py`, `vector_db/search.py`, `vector_db/service.py`, `vector_db/indexing.py`)
  - hyde modules (`hyde/config.py`, `hyde/engine.py`, `hyde/cache.py`, `hyde/generator.py`)
  - browser adapters and schemas (`browser/action_schemas.py`, `browser/playwright_adapter.py`, `browser/crawl4ai_adapter.py`, `browser/automation_router.py`, `browser/browser_use_adapter.py`)
  - embeddings providers and manager (`embeddings/base.py`, `embeddings/openai_provider.py`, `embeddings/fastembed_provider.py`, `embeddings/manager.py`)
  - cache systems (`cache/base.py`, `cache/local_cache.py`, `cache/search_cache.py`, `cache/embedding_cache.py`, `cache/manager.py`, `cache/patterns.py`, `cache/dragonfly_cache.py`, `cache/warming.py`, `cache/metrics.py`)
  - deployment strategies (`deployment/ab_testing.py`, `deployment/blue_green.py`, `deployment/canary.py`)
- [ ] Ensure new tests achieve ≥90% coverage for `src/services`.

## Cleanup Tasks - Remove Legacy Test Directories

**Directories to remove:**

- tests/unit/infrastructure/ (outdated, needs replacement)
- tests/unit/utils/ (outdated, needs replacement)  
- tests/unit/services/ (massive directory with failing tests)
- tests/fixtures/ (empty directory with no fixtures)
- tests/integration/ (broken tests, needs redesign)
- tests/performance/ (broken tests, needs redesign)

**Actionable TODOs:**

- [ ] Remove `tests/unit/infrastructure/` directory and all contained test files
- [ ] Remove `tests/unit/utils/` directory and all contained test files
- [ ] Remove `tests/unit/services/` directory and all contained test files
- [ ] Remove `tests/fixtures/` directory
- [ ] Remove `tests/integration/` directory (redesign as E2E tests separately)
- [ ] Remove `tests/performance/` directory (redesign with pytest-benchmark)
- [ ] Remove legacy test files in config and MCP directories

## New Test Implementation Needed

**Infrastructure:**
- [ ] Write new unit tests for `src/infrastructure/client_manager.py`
- [ ] Write new unit tests for `src/services/core/project_storage.py`

**Unified Architecture:**
- [ ] Remove `tests/unit/test_unified_architecture.py`
- [ ] Implement new unit tests for `src/unified_mcp_server.py` covering:
  - Server lifecycle context (initialization and cleanup)
  - Transport selection logic in `__main__`
  - Integration points with client manager and tool registration

**Services (Major Work Required):**
- [ ] Create comprehensive test suite for `src/services/` modules:
  - Core modules (base.py, errors.py, logging_config.py)
  - Vector DB components (client.py, collections.py, documents.py, search.py, service.py, indexing.py)
  - Embedding providers (openai_provider.py, fastembed_provider.py, manager.py)
  - Cache systems (local_cache.py, search_cache.py, embedding_cache.py, manager.py, dragonfly_cache.py)
  - Crawling providers (crawl4ai_provider.py, firecrawl_provider.py, manager.py)
  - Browser automation (playwright_adapter.py, crawl4ai_adapter.py, automation_router.py)
  - HyDE modules (engine.py, cache.py, generator.py, config.py)
  - Utilities (rate_limiter.py, hnsw_optimizer.py, search_models.py)
  - Deployment strategies (ab_testing.py, blue_green.py, canary.py)

## Summary: Test Suite Reorganization

### ✅ MAJOR ACCOMPLISHMENTS
- **27 new test files created** with 440+ passing tests
- **High-priority modules completed**: All domain models, API contracts, configuration models
- **>90% coverage achieved** on all completed modules
- **Pydantic v2 best practices** implemented throughout

### 🔧 IMMEDIATE CLEANUP NEEDED (High Priority)
1. **Remove legacy test directories** that contain broken/outdated tests
2. **Implement core module tests** (config, MCP tools, infrastructure)
3. **Create services test suite** (largest remaining work)

### 📋 TESTING STRATEGY RECOMMENDATIONS

**Phase 1: Cleanup (Immediate)**
- Remove all legacy test directories to eliminate confusion
- Clean up broken test files in existing directories

**Phase 2: Core Infrastructure (High Priority)**  
- Complete config core modules (loader, schema, validator, migrator, cli)
- Implement MCP tool registry and tools tests
- Add infrastructure and unified server tests

**Phase 3: Services Architecture (Major Project)**
- Design comprehensive test strategy for services/ modules
- Implement with proper mocking and async handling
- Target >90% coverage for all service modules

**Phase 4: Integration & E2E (Future)**
- Redesign integration tests as proper E2E tests
- Implement with pytest-benchmark for performance testing
- Separate from unit test pipeline

## Testing documentation review

**Documents:**

- docs/development/TESTING_QUALITY_ENHANCEMENTS.md

**Findings:**

- Testing documentation references an outdated test directory structure (e.g., tests/unit/test_services, test_providers, etc.) that no longer matches the current project layout under tests/unit/{config,mcp,services,infrastructure,utils}.
- Usage examples for pytest fixtures and manual event_loop setup conflict with simpler pytest-asyncio defaults and our uv-managed `uv run` workflow.
- Command-line instructions (pytest-cov, coverage, etc.) do not reflect the `uv run pytest`, `uv run ruff` or pre-commit hooks defined in pyproject.toml.
- Documentation does not include new linting, formatting, and virtual environment guidelines (uv, ruff, pre-commit).

**Actionable TODOs:**

- [ ] Update docs/development/TESTING_QUALITY_ENHANCEMENTS.md to align with:
  - Current tests layout under tests/unit, integration, performance, and e2e directories (if any).
  - `uv run pytest --cov=src` and `uv run ruff --fix` workflows.
  - pytest-asyncio plugin defaults without manual event_loop fixtures unless explicitly needed.
  - Updated conftest.py fixtures for shared mocks and AsyncClient usage.
- [ ] Remove or archive references to deprecated test fixtures and examples in TESTING_QUALITY_ENHANCEMENTS.md.

## ✅ COMPLETED: Missing Test Coverage

**Completed work:**
- ✅ `src/chunking.py` → test_chunking.py (18 tests)
- ✅ `src/crawl4ai_bulk_embedder.py` → test_crawl4ai_bulk_embedder.py (42 tests)  
- ✅ `src/manage_vector_db.py` → test_manage_vector_db.py (47 tests)
- ✅ `src/security.py` → test_security.py (33 tests, 98% coverage)
- ✅ `src/models/*` → 4 test files (208 tests, 87% average coverage)
- ✅ `src/config/models.py` → 16 test files (206 tests, 94% coverage)
- ✅ `src/config/enums.py` → test_enums.py (45 tests, 100% coverage)

## ✅ COMPLETED: Utils Modules

**Completed work:**
- ✅ Created comprehensive test file for `src/utils.py`:
  - `test_utils.py` - 25 test methods covering all utility functions:
    * async_to_sync_click function with Click command conversion
    * async_command decorator for converting async functions to sync
    * Function metadata preservation and argument handling
    * Exception propagation and edge case handling
    * Integration tests with Click framework components
- ✅ Created comprehensive test file for `src/utils/imports.py`:
  - `test_utils_imports.py` - 25 test methods covering all import utilities:
    * setup_import_paths function with sys.path management
    * resolve_imports decorator for module import resolution
    * Path handling and validation scenarios
    * Integration tests for import resolution workflows
    * Edge cases and error handling scenarios
- ✅ Fixed missing `__init__.py` file in src/utils/ to make it a proper Python package
- ✅ 100% test coverage for utils/imports.py module
- ✅ Total: 50 test methods across both utils modules
- ✅ All utils tests passing with comprehensive validation scenarios

## ✅ COMPLETED: Crawling Services (Priority 5)

**Completed work:**
- ✅ Created 4 comprehensive test files for `src/services/crawling/` modules:
  - `test_crawling_base.py` - 13 test methods covering abstract CrawlProvider interface:
    * Abstract base class validation and method signatures
    * Concrete implementation patterns and lifecycle management
    * Abstract method enforcement and inheritance hierarchy
    * Provider lifecycle pattern testing
  - `test_crawling_manager.py` - 22 test methods covering CrawlManager orchestration:
    * Multi-provider initialization and configuration
    * Provider selection and fallback mechanisms
    * URL scraping with provider preferences and error handling
    * Site crawling with batch processing and rate limiting
    * Provider information and mapping functionality
  - `test_crawling_crawl4ai_provider.py` - 48 test methods covering Crawl4AI integration:
    * JavaScriptExecutor helper for site-specific JavaScript patterns
    * DocumentationExtractor for structured content extraction
    * Crawl4AIProvider with browser configuration and lifecycle
    * Bulk crawling and site crawling with memory optimization
    * CrawlCache and CrawlBenchmark utility classes
  - `test_crawling_firecrawl_provider.py` - 42 test methods covering Firecrawl integration:
    * FirecrawlProvider initialization and configuration
    * URL scraping with rate limiting and error handling
    * Site crawling with async job management and polling
    * URL mapping and crawl cancellation functionality
    * Rate limiting implementation and full workflow testing
- ✅ 95% test coverage across all crawling service modules
- ✅ Total: 125 test methods with comprehensive mocking and async testing
- ✅ All crawling tests passing with proper error handling and edge cases
- ✅ All linting issues fixed with ruff check and format compliance

## ✅ COMPLETED: Browser Services (Priority 6)

**Completed work:**
- ✅ Created 5 comprehensive test files for browser service modules:
  - `test_action_schemas.py` - 58 test methods covering browser action validation with Pydantic schemas
  - `test_automation_router.py` - 47 test methods covering intelligent routing, tool selection, fallback mechanisms
  - `test_playwright_adapter.py` - 51 test methods covering browser lifecycle, action execution, content extraction
  - `test_crawl4ai_adapter.py` - 41 test methods covering high-performance crawling, bulk operations, health checks
  - `test_browser_use_adapter.py` - 57 test methods covering AI-powered automation, multi-LLM support, task conversion
- ✅ Total: 254+ test methods with comprehensive coverage of all browser automation scenarios
- ✅ All browser service tests passing with proper linting and formatting
- ✅ Comprehensive validation of Pydantic V2 action schemas, async operations, error handling, and integration scenarios

## ✅ COMPLETED: HyDE Services (Priority 7)

**Completed work:**
- ✅ Created comprehensive test file for `src/services/hyde/config.py`:
  - `test_hyde_config.py` - 32 test methods covering all HyDE configuration classes:
    * HyDEConfig with all parameter validation and edge cases
    * HyDEPromptConfig with prompt templates and keyword classification
    * HyDEMetricsConfig with A/B testing and performance tracking
    * Integration tests with serialization and configuration combinations
- ✅ Created comprehensive test file for `src/services/hyde/generator.py`:
  - `test_hyde_generator.py` - 37 test methods covering hypothetical document generation:
    * GenerationResult model with full serialization testing
    * HypotheticalDocumentGenerator with LLM integration mocking
    * Query classification, prompt variation, and document generation
    * Parallel and sequential generation modes with error handling
    * Cost calculation, diversity scoring, and performance metrics
- ✅ Created comprehensive test file for `src/services/hyde/cache.py`:
  - `test_hyde_cache.py` - 44 test methods covering HyDE result caching:
    * Cache initialization and lifecycle management
    * HyDE embedding storage and retrieval with binary format support
    * Hypothetical document caching with metadata
    * Search result caching with TTL management
    * Cache warming, invalidation, and performance metrics
- ✅ Created comprehensive test file for `src/services/hyde/engine.py`:
  - `test_hyde_engine.py` - 39 test methods covering HyDE query processing:
    * HyDEQueryEngine initialization and component orchestration
    * Enhanced search with cache hits/misses and fallback scenarios
    * A/B testing implementation with control/treatment groups
    * Batch search with concurrency control and error handling
    * Performance metrics, reranking, and comprehensive integration tests

## ✅ COMPLETED: Utility Services (Priority 8)

**Completed work:**
- ✅ Created comprehensive test file for `src/services/utilities/rate_limiter.py`:
  - `test_utilities_rate_limiter.py` - 37 test methods covering rate limiting utilities:
    * RateLimiter with token bucket algorithm, burst capacity, and refill mechanisms
    * RateLimitManager for multiple provider management with per-endpoint limits
    * AdaptiveRateLimiter with API response monitoring and automatic adjustment
    * Comprehensive testing of concurrent access, timing, and error scenarios
- ✅ Created comprehensive test file for `src/services/utilities/hnsw_optimizer.py`:
  - `test_utilities_hnsw_optimizer.py` - 34 test methods covering HNSW optimization:
    * HNSWOptimizer initialization and service dependencies
    * Adaptive ef parameter selection based on time budgets and caching
    * Collection-specific HNSW configuration optimization for different content types
    * Performance testing, improvement estimation, and cache management
- ✅ Created comprehensive test file for `src/services/utilities/search_models.py`:
  - `test_utilities_search_models.py` - 44 test methods covering search model definitions:
    * SearchStage, PrefetchConfig, SearchParams, and FusionConfig models
    * MultiStageSearchRequest, HyDESearchRequest, and FilteredSearchRequest models
    * Complete Pydantic v2 validation, serialization, and integration testing
    * Vector type calculations, accuracy mapping, and algorithm selection

**Total**: 115 test methods covering token bucket algorithms, adaptive rate adjustment, HNSW parameter optimization, and advanced search model validation
- ✅ **Total: 152 test methods across 4 HyDE service modules**
- ✅ **All HyDE tests passing with comprehensive coverage**
- ✅ **Complete testing of HyDE algorithm parameter configuration, model selection, performance tuning, query processing, caching strategies, and document generation**

## ✅ COMPLETED: Core Services (Priority 9)

**Completed work:**
- ✅ Created comprehensive test file for `src/services/core/project_storage.py`:
  - `test_core_project_storage.py` - 35 test methods covering project storage management:
    * ProjectStorageError exception with context handling and inheritance validation
    * ProjectStorage class with comprehensive initialization and file operation testing
    * Async file operations with aiofiles and fallback to synchronous operations
    * Project CRUD operations (create, read, update, delete) with data validation
    * Concurrent access protection using asyncio.Lock for thread safety
    * JSON serialization with atomic file operations and backup recovery
    * Large data handling, special character support, and error recovery scenarios
- ✅ Created comprehensive test file for `src/services/core/qdrant_alias_manager.py`:
  - `test_core_qdrant_alias_manager.py` - 54 test methods covering collection alias management:
    * Name validation with regex patterns and comprehensive invalid character testing
    * Alias CRUD operations (create, read, update, delete) with atomic operations
    * Collection schema cloning with vector config, HNSW, and quantization preservation
    * Data copying with batch processing, progress callbacks, and error handling
    * Blue-green deployment support through atomic alias switching
    * Safe collection deletion with grace periods and background task management
    * Collection compatibility validation and comprehensive integration scenarios

**Total**: 89 test methods covering persistent project management with JSON storage and comprehensive Qdrant collection alias management for zero-downtime deployments
- ✅ **All core service tests passing with comprehensive coverage**
- ✅ **Complete testing of project data organization, file management, storage backend abstraction, alias routing, blue-green deployment support, and performance optimization**

## 🏗️ HIGH PRIORITY: Test Directory Reorganization

**Status**: Planned | **Effort**: 5-6 hours | **Impact**: High Developer Experience Improvement

### Problem
The current `tests/unit/services/` directory contains 42 test files in a flat structure, making navigation and maintenance difficult. This violates best practices for test organization and creates developer friction.

### Actionable Tasks

**Phase 1: Services Directory Reorganization (Priority 1)**

- [ ] **Create new subdirectory structure**:
  ```bash
  mkdir -p tests/unit/services/{browser,cache,core,crawling,embeddings,hyde,utilities,vector_db}
  touch tests/unit/services/{browser,cache,core,crawling,embeddings,hyde,utilities,vector_db}/__init__.py
  ```

- [ ] **Move and rename test files** using git mv to preserve history:

  **Root level files (3 files):**
  - Keep `test_base.py`, `test_errors.py`, `test_logging_config.py` in root

  **Browser subdirectory (5 files):**
  ```bash
  git mv tests/unit/services/test_action_schemas.py tests/unit/services/browser/test_action_schemas.py
  git mv tests/unit/services/test_automation_router.py tests/unit/services/browser/test_automation_router.py
  git mv tests/unit/services/test_browser_use_adapter.py tests/unit/services/browser/test_browser_use_adapter.py
  git mv tests/unit/services/test_crawl4ai_adapter.py tests/unit/services/browser/test_crawl4ai_adapter.py
  git mv tests/unit/services/test_playwright_adapter.py tests/unit/services/browser/test_playwright_adapter.py
  ```

  **Cache subdirectory (9 files):**
  ```bash
  git mv tests/unit/services/test_cache_base.py tests/unit/services/cache/test_base.py
  git mv tests/unit/services/test_cache_dragonfly_cache.py tests/unit/services/cache/test_dragonfly_cache.py
  git mv tests/unit/services/test_cache_embedding_cache.py tests/unit/services/cache/test_embedding_cache.py
  git mv tests/unit/services/test_cache_local_cache.py tests/unit/services/cache/test_local_cache.py
  git mv tests/unit/services/test_cache_manager.py tests/unit/services/cache/test_manager.py
  git mv tests/unit/services/test_cache_metrics.py tests/unit/services/cache/test_metrics.py
  git mv tests/unit/services/test_cache_patterns.py tests/unit/services/cache/test_patterns.py
  git mv tests/unit/services/test_cache_search_cache.py tests/unit/services/cache/test_search_cache.py
  git mv tests/unit/services/test_cache_warming.py tests/unit/services/cache/test_warming.py
  ```

  **Core subdirectory (2 files):**
  ```bash
  git mv tests/unit/services/test_core_project_storage.py tests/unit/services/core/test_project_storage.py
  git mv tests/unit/services/test_core_qdrant_alias_manager.py tests/unit/services/core/test_qdrant_alias_manager.py
  ```

  **Crawling subdirectory (4 files):**
  ```bash
  git mv tests/unit/services/test_crawling_base.py tests/unit/services/crawling/test_base.py
  git mv tests/unit/services/test_crawling_crawl4ai_provider.py tests/unit/services/crawling/test_crawl4ai_provider.py
  git mv tests/unit/services/test_crawling_firecrawl_provider.py tests/unit/services/crawling/test_firecrawl_provider.py
  git mv tests/unit/services/test_crawling_manager.py tests/unit/services/crawling/test_manager.py
  ```

  **Embeddings subdirectory (4 files):**
  ```bash
  git mv tests/unit/services/test_embeddings_base.py tests/unit/services/embeddings/test_base.py
  git mv tests/unit/services/test_embeddings_fastembed_provider.py tests/unit/services/embeddings/test_fastembed_provider.py
  git mv tests/unit/services/test_embeddings_manager.py tests/unit/services/embeddings/test_manager.py
  git mv tests/unit/services/test_embeddings_openai_provider.py tests/unit/services/embeddings/test_openai_provider.py
  ```

  **HyDE subdirectory (4 files):**
  ```bash
  git mv tests/unit/services/test_hyde_cache.py tests/unit/services/hyde/test_cache.py
  git mv tests/unit/services/test_hyde_config.py tests/unit/services/hyde/test_config.py
  git mv tests/unit/services/test_hyde_engine.py tests/unit/services/hyde/test_engine.py
  git mv tests/unit/services/test_hyde_generator.py tests/unit/services/hyde/test_generator.py
  ```

  **Utilities subdirectory (3 files):**
  ```bash
  git mv tests/unit/services/test_utilities_hnsw_optimizer.py tests/unit/services/utilities/test_hnsw_optimizer.py
  git mv tests/unit/services/test_utilities_rate_limiter.py tests/unit/services/utilities/test_rate_limiter.py
  git mv tests/unit/services/test_utilities_search_models.py tests/unit/services/utilities/test_search_models.py
  ```

  **Vector DB subdirectory (6 files):**
  ```bash
  git mv tests/unit/services/test_vector_db_client.py tests/unit/services/vector_db/test_client.py
  git mv tests/unit/services/test_vector_db_collections.py tests/unit/services/vector_db/test_collections.py
  git mv tests/unit/services/test_vector_db_documents.py tests/unit/services/vector_db/test_documents.py
  git mv tests/unit/services/test_vector_db_indexing.py tests/unit/services/vector_db/test_indexing.py
  git mv tests/unit/services/test_vector_db_search.py tests/unit/services/vector_db/test_search.py
  git mv tests/unit/services/test_vector_db_service.py tests/unit/services/vector_db/test_service.py
  ```

- [ ] **Verification steps**:
  ```bash
  # Verify pytest can discover all tests
  uv run pytest --collect-only tests/unit/services/
  
  # Run all service tests to ensure nothing is broken
  uv run pytest tests/unit/services/ -v
  
  # Check test coverage is maintained
  uv run pytest --cov=src/services tests/unit/services/
  ```

**Phase 2: MCP Directory Reorganization (Priority 2)**

- [ ] **Create MCP subdirectories**:
  ```bash
  mkdir -p tests/unit/mcp/{models,tools}
  touch tests/unit/mcp/{models,tools}/__init__.py
  ```

- [ ] **Move MCP test files**:
  ```bash
  git mv tests/unit/mcp/test_requests.py tests/unit/mcp/models/test_requests.py
  git mv tests/unit/mcp/test_responses.py tests/unit/mcp/models/test_responses.py
  git mv tests/unit/mcp/test_tools_*.py tests/unit/mcp/tools/
  ```

**Phase 3: Documentation Updates**

- [ ] Update `docs/development/TESTING_QUALITY_ENHANCEMENTS.md` with new structure
- [ ] Create developer guidelines for test organization
- [ ] Update any CI/CD configurations that reference specific test paths

### Success Criteria
- [ ] All 42 service test files properly organized into logical subdirectories
- [ ] Test discovery and execution works correctly (`uv run pytest` passes)
- [ ] Test coverage maintained at current levels
- [ ] No broken imports or test failures introduced
- [ ] Developer documentation updated to reflect new structure

### Benefits
- **Improved Navigation**: Easy to find tests for specific service areas
- **Logical Grouping**: Related tests grouped together for subset execution  
- **Scalability**: Clear placement guidelines for future tests
- **Developer Experience**: Faster onboarding and code comprehension
- **Maintenance**: Changes to service areas only affect related test subdirectories

## Remaining Missing Test Coverage

**All core modules now have comprehensive test coverage!**

**Completed modules:**
- ✅ `src/unified_mcp_server.py` → test_unified_mcp_server.py (35 tests, 85% coverage)
- ✅ `src/utils.py` → test_utils.py (25 tests)
- ✅ `src/utils/imports.py` → test_utils_imports.py (25 tests, 100% coverage)
