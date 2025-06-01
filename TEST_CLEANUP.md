# TEST_CLEANUP

This file lists outdated or failing unit tests in tests/unit/ that require cleanup
and rewriting to align with current src/ implementations.

## Config group

**Test files:**

- tests/unit/config/test_config.py
- tests/unit/config/test_rate_limiting_config.py
- tests/unit/config/test_unified_config.py

**Findings:**

- Overall coverage for `src/config` is ~45%, far below the 80–90% target.
- `test_unified_config.py::TestConfigLoader::test_load_config_with_priority` fails due to Pydantic's stricter `extra_forbidden` behavior, indicating test assumptions no longer match the loader logic.
- Core modules (`cli.py`, `loader.py`, `migrator.py`, `schema.py`, `validator.py`) have evolved with new features and parameter structures that existing tests do not cover.
- No tests for critical behaviors such as config priority resolution, legacy migration paths, schema validation errors, or CLI command execution.

**Actionable TODOs:**

- [ ] Remove the `tests/unit/config/` directory and all contained test files.
- [ ] Implement unit tests for `src/config/loader.py` to verify:
  - Loading of default and environment-specific settings.
  - Priority resolution and merging of multiple config sources.
- [ ] Implement unit tests for `src/config/schema.py` to verify:
  - Schema validation for valid and invalid config structures.
  - Correct error messages and missing field handling.
- [ ] Implement unit tests for `src/config/validator.py` to verify:
  - Custom validation rules and exception raising on invalid input.
- [ ] Implement unit tests for `src/config/migrator.py` to verify:
  - Migration of legacy configurations through defined migrator functions.
- [ ] Implement unit tests for `src/config/cli.py` to verify:
  - CLI argument parsing and execution of config commands (e.g., `validate`, `migrate`).

## MCP group

**Test files:**

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

- Multiple tests in the MCP group fail with `ModuleNotFoundError: No module named 'mcp.server'`, indicating import paths and fastmcp Context usage are outdated.
- Existing tests rely on deprecated server-based APIs from `fastmcp`, which have been refactored or removed.
- Coverage for `src/mcp` cannot be evaluated because import errors prevent any execution of the code under test.

**Actionable TODOs:**

- [ ] Remove the `tests/unit/mcp/` directory and all its contents.
- [ ] Create new unit tests for `src/mcp/tool_registry.py` to verify:
  - Registry initialization, tool registration, lookup, and error handling.
- [ ] Create new unit tests for each module in `src/mcp/tools/` to verify:
  - Public helper function imports and basic execution with mock inputs for utilities, collections, search, embeddings, documents, and cache.
  - Proper error handling for invalid parameters or missing dependencies.

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

## Infrastructure group

**Test files:**

- tests/unit/infrastructure/test_client_manager.py
- tests/unit/infrastructure/test_project_storage.py

**Findings:**

- Tests error out with permission issues when creating asyncio event loops and refer to outdated circuit breaker design.
- `test_project_storage.py` references `src/infrastructure/project_storage`, but the functionality has been relocated to `src/services/core/project_storage.py`.

**Actionable TODOs:**

- [ ] Remove the `tests/unit/infrastructure/` directory and all contained test files.
- [ ] Write new unit tests for `src/infrastructure/client_manager.py` and for `src/services/core/project_storage.py`.

## Utils group

**Test files:**

- tests/unit/utils/test_security.py
- tests/unit/utils/test_smart_model_selection.py
- tests/unit/utils/test_chunking.py
- tests/unit/utils/test_error_handling.py

**Findings:**

- Tests assume outdated utility implementations; many reference modules or functions that have moved or been refactored.
- Coverage for `src/utils` and related utility logic cannot be assessed due to import errors and incompatibilities.

**Actionable TODOs:**

- [ ] Remove the `tests/unit/utils/` directory and all contained test files.
- [ ] Create new unit tests for current utility modules, aligning with the existing code in `src/utils`.

## Unified architecture group

**Test files:**

- tests/unit/test_unified_architecture.py

**Findings:**

- Tests now fail on asyncio event loop creation and reference legacy module paths (`src.infrastructure`, `src.mcp`).
- Unified server and client manager logic has been updated; existing tests do not match the current `src/unified_mcp_server.py` and related modules.

**Actionable TODOs:**

- [ ] Remove `tests/unit/test_unified_architecture.py`.
- [ ] Implement new unit tests for `src/unified_mcp_server.py` covering:
  - Server lifecycle context (initialization and cleanup).
  - Transport selection logic in `__main__`.
  - Integration points with client manager and tool registration.

## Fixtures group

**Test files:**

- tests/fixtures/**

**Findings:**

- The `tests/fixtures/` directory contains only an empty `__init__.py` with a docstring and no actual fixture definitions.
- No shared fixtures are provided; tests that require fixtures have no centralized setup.

**Actionable TODOs:**

- [ ] Remove the `tests/fixtures/` directory.
- [ ] Consolidate any needed shared fixtures into `tests/conftest.py` or per‑group fixture modules.

## Integration test group

**Test files:**

- tests/integration/**/*.py

**Findings:**

- `uv run pytest tests/integration` currently panics in the uv runner due to signal/socket issues; running `pytest tests/integration` shows import failures (e.g., missing `fakeredis`) and network dependencies.
- Tests rely on real external services (Redis, Qdrant, web network, vector DB) without mocks or test doubles.
- Many tests reference outdated module paths and legacy APIs that no longer exist.
- The directory mixes unit‑style service tests, higher‑level integration tests, and end‑to‑end scripts without clear separation.

**Actionable TODOs:**

- [ ] Remove the entire `tests/integration/` directory.
- [ ] Reclassify true end‑to‑end scripts into a dedicated `tests/e2e/` suite, to be run in a separate pipeline stage with real services.
- [ ] Implement new lightweight integration tests for core modules under `src/`, using fixtures or mocks for external dependencies.
- [ ] Update CI/workflow documentation to invoke integration/E2E tests separately from unit tests.

## Performance test group

**Test files:**

- tests/performance/**/*.py

**Findings:**

- Performance tests fail with `PermissionError: Operation not permitted` when creating asyncio event loops under container security policy.
- Tests use manual loop creation and external services, leading to fragility and runtime errors.
- The structure mixes simple benchmarks and functional tests without a standardized benchmarking framework.

**Actionable TODOs:**

- [ ] Remove the `tests/performance/` directory and all contained test files.
- [ ] If performance benchmarks are required, migrate to a dedicated benchmarking suite (e.g., `pytest-benchmark`) isolated from the main test pipeline.
- [ ] Refactor any necessary performance tests to avoid manual event loop creation and to use standardized fixtures/plugins.

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

## Missing test coverage

The following source modules and packages lack any corresponding unit tests under tests/unit or other test suites:

- **Root-level modules:** `src/chunking.py`, `src/crawl4ai_bulk_embedder.py`, `src/manage_vector_db.py`, `src/security.py`.
- **Models package:** `src/models/` (all modules under models have no test coverage).
- **Infrastructure core:** `src/infrastructure/` (client_manager tests to be added; verify any other submodules).
- **Config models and enums:** `src/config/models.py`, `src/config/enums.py` (no direct model tests yet).

**Actionable TODOs:**

- [ ] Create new unit tests covering root-level utility scripts and modules (chunking, scrape embedders, security).
- [ ] Add model validation tests for `src/models/*` and `src/config/models.py`, `src/config/enums.py`.
- [ ] Verify and implement tests for any other uncovered modules after test suite stabilization.
