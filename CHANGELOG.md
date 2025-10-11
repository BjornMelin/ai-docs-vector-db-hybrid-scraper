# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added focused FastAPI security middleware tests and published release notes summarising the configuration/file-watch hardening work.
- Documented osquery prerequisites for configuration file watching to help operators enable the feature safely.
- Captured ADR 0009 documenting the tiered Playwright anti-bot stack and exported the new browser metrics section in `docs/observability/query_processing_metrics.md` with challenge counters.
- Published the consolidated Evaluation Harness Playbook (`docs/testing/evaluation-harness.md`) covering OpenTelemetry-aligned RAG metrics,
  dashboard guardrails, and operational workflows.
- Extended the RAG evaluation harness to emit structured OpenTelemetry spans and
  aggregated JSON metrics alongside golden-set similarity scores for CI
  regression analysis.
- Introduced a LangGraph `StateGraph` pipeline that chains retrieval, grading, and generation for RAG queries.
- Replaced the bespoke FastEmbed provider with a LangChain FastEmbed wrapper and refreshed unit coverage for embeddings.
- Shadow parity coverage for vector grouping fallback and score normalisation across
  `tests/unit/services/vector_db/test_service.py` and the new RAG retriever
  suites to lock in Phase A behaviour.
- `tests/data/rag/golden_set.jsonl` and `scripts/eval/rag_golden_eval.py` to
  provide a reproducible RAG regression harness leveraged by CI and manual
  comparison runs.
- Introduced `.github/dependabot.yml` to automate weekly updates for GitHub Actions and Python dependencies.
- Documented CI branch-protection guidance and pinned action examples across developer and security guides.
- Captured the comprehensive unit-test refactor roadmap in `planning/unit_test_refactor_plan.md` covering fixture cleanup,
  deterministic rewrites, and sprint sequencing.
- Added a `quality-unit` make target that runs Ruff, Pylint, Pyright, and the unit test suite in one command for local gating.
- Published the consolidated MCP test strategy doc (`docs/testing/mcp-unit-tests.md`) capturing the coverage map, decision
  record, and technical debt register for the new suites.

### Refactor

- Removed dual-mode architecture. Single unified app with feature flags.
- **[Core]:** Unified dependency lifecycle under the dependency-injector
  container, deleting the legacy `ClientManager` pattern and introducing shared
  bootstrap helpers for FastAPI, MCP, CLI utilities, and evaluation scripts.
- **[Config]:** Deprecated `src/core`, relocating defaults into Pydantic settings
  models so `Settings` remains the canonical source of configuration values.
- **[API]:** Removed the duplicated `/api/routers/simple` module and unified all endpoints to use the canonical `SearchRequest` and `SearchRecord` data models.
- **[Models]:** Consolidated all search-related request and response models into canonical `SearchRequest` and `SearchRecord` contracts, removing significant code duplication.
- **[Core]:** Removed custom rate-limiting implementation in favor of the
  `slowapi` library, centralizing all rate-limiting logic at the middleware
  layer.
- **[Architecture]:** Standardized all custom exceptions to inherit from the
  canonical error hierarchy defined in `src/services/errors.py`, unifying error
  handling across the application.
- **[Observability]:** Replaced the legacy `TelemetryRepository` with the unified, OpenTelemetry-based `AIOperationTracker` for all in-memory application telemetry.
- **[Observability]:** Removed the legacy `RealTimePerformanceMonitor` and consolidated its functionality. Operation timing is now handled by the OpenTelemetry-based `PerformanceMonitor`, and system resource metrics are exposed via the `system_health` MCP tool.
- **[Observability]:** Completed the unification of monitoring systems by deleting the legacy `src/services/monitoring` package and updating all application entry points to use the modern `initialize_observability` lifecycle function.
- **[Observability]:** Eliminated the bespoke Prometheus `MetricsRegistry` and middleware. All `/metrics` exposure now flows through `prometheus-fastapi-instrumentator` while application telemetry is captured via OpenTelemetry tracing utilities.
- **[Observability]:** Replaced database query monitors with OpenTelemetry spans and the `db.query.duration` histogram, and relocated the shared `HealthCheckManager` under `services/observability` for unified health wiring.
- **[Architecture]:** Unified circuit breaker implementation by refactoring
  `TimeoutMiddleware` to use the shared `CircuitBreakerManager`, ensuring a
  global state for all breakers.
- **[Architecture]:** Unified all service access patterns by consolidating logic
  from `src/services/managers` into the central `ClientManager`, establishing it
  as the single service locator for the application.
- **Architectural Unification:** Consolidated the `monitoring` and `observability` packages into a single, OpenTelemetry-based `observability` service. This centralizes all metrics, tracing, and health check logic.
- **Dead Code Elimination:** Removed several unused and legacy modules, including `services/middleware/rate_limiting.py`, `services/processing/algorithms.py`, and the top-level `security.py`, significantly reducing codebase size.
- **API Modernization:** Updated all calls to the `@circuit_breaker` decorator to use the modern `purgatory` API, removing deprecated arguments and silencing warnings.
- **Test Suite Overhaul:** Deleted test suites corresponding to removed modules and modernized remaining tests to align with the refactored implementation, ensuring full test coverage and correctness.
- **[Health]:** Centralized all system health checking logic into the `HealthCheckManager`, and refactored the `system_health` MCP tool to use this central service.
- **[Cache]:** Consolidated all caching logic to use the central `CacheManager`. Removed the redundant `PerformanceCache` and refactored `HyDECache` to be a client of the main cache service.
- Centralized tier selection inside `AutomationRouter`, reducing MCP crawling
  tools to thin facades over the unified routing service.

### Changed

- CLI commands, wizard utilities, and tests now emit technical, emoji-free
  messaging aligned with the canonical `SearchRecord` workflow, ensuring
  consistent help output across CLI entry points.
- MCP tools, fixtures, and tests now consume the unified service payload models
  (content intelligence, embeddings, retrieval, analytics, collection management,
  lightweight scrape) with optional dependency guards; CLI batch/setup helpers
  and docs were refreshed, and the full unit suite passes with the new entrypoints.
- Replaced the ModeAware service factory and ServiceRegistry layers with a single
  ClientManager-driven dependency graph, rewiring FastAPI dependencies, CLI
  utilities, and tests to rely on the new lifecycle helpers.
- CLI and MCP tooling now load settings via `load_settings_from_file`, supporting JSON/YAML config import/export, refreshing CLI configuration tests and fixtures.
- Configuration documentation (developer, user, operator guides) updated to reflect the unified settings loader and CLI helpers.
- CI lint workflow now runs `python scripts/guards/check_settings_usage.py` to prevent legacy `get_config`/`set_config` usage from regressing.
- Lightweight scraper escalates 4xx/5xx responses without raising, mode helpers resolve from injected settings, Qdrant default collection alias was removed, typed config fixtures were introduced for tests, and docs/env guidance now reference `primary_collection` and `AI_DOCS_CONFIG_PATH`.
- Clarified the analytics and RAG service package facades with typed `__all__`
  exports, refreshed coverage, and updated API docs to describe the supported
  entry points.
- Configuration reload and file-watching endpoints now validate override paths, surface operator errors as HTTP 400, and wait until the osquery-backed provider reports readiness before returning success.
- Rebuilt the configuration stack around a single `Settings` provider, removed legacy helpers/aliases, deleted the hot-reload subsystem, and refactored services/tests to consume nested credentials directly.
- Monitoring initialization and middleware derive health checks from the unified settings provider, gracefully degrade when optional dependencies (flagsmith, purgatory, respx, asgi-lifespan) are absent, and pytest fixtures adopt importorskip-based guards for optional extras.
- Simplified the security configuration model to the fields actually enforced by the middleware, updated templates/docs, and aligned the middleware implementation with the lean schema.
- Removed the fail-closed `search_service`/`cache_service` placeholders; mode configs now advertise only supported embedding/vector services and the health endpoint reflects the leaner set.
- Dependency re-exports for FastAPI helpers now live under `src/services/fastapi/dependencies/__init__.py`, trimming redundant wrapper modules.
- Collection deletion tooling now requires a native `delete_collection` implementation and removes legacy `drop_collection` shims; CI fails fast on unexpected skips to prevent dead suites from lingering.
- Chunking defaults route `ChunkingStrategy.BASIC` through the enhanced semantic pipeline, and circuit breaker decorators/configuration have dropped all deprecated parameters.
- Replaced the bespoke FastAPI `DependencyContainer` and module-level singletons with a new `ServiceRegistry` (`src/services/registry.py`), wiring FastAPI lifespan, CLI tooling, and dependency helpers to the shared registry.
- MCP embeddings tools now normalise provider metadata directly from `EmbeddingManager`, cache snapshots with fallbacks, and expose the reusable `provider_metadata` helper for other tooling.
- `/health` now centralises on `HealthCheckManager`, removes auxiliary health endpoints, rewires FastAPI dependencies and CLI status reporting to the shared builder, and documents the consolidated response contract.
- `pyproject.toml` dev extras now include `asgi-lifespan`; docs refreshed to describe the registry architecture and note the dev-extras requirement.
- LangGraph GraphRunner now emits structured metrics/errors, enforces optional
  run timeouts, and surfaces discovery/tool telemetry across MCP entry points;
  MCP orchestrator and tests were refreshed to exercise the new pipeline.
- Dependency pin for `tiktoken` raised to the 0.12.x series to pick up the latest
  model encodings, Python 3.14/free-threaded wheels, and file I/O improvements.
- Metrics registry now exposes stage-level latency, answer, error, and token
  counters for LangGraph pipelines; `docs/observability/query_processing_metrics.md`
  and ADR 0007 describe the new instrumentation and SemVer impact.
- `scripts/eval/rag_golden_eval.py` now initialises an in-memory metrics registry,
  exposes a JSON metrics export, and ships table-driven tests for deterministic
  harness behaviour.
- Search orchestrator delegates RAG execution to the LangGraph pipeline, reusing contextual compression and vector service retrievers.
- Migrated VectorStoreService, MCP ingestion, and database manager paths onto LangChain's QdrantVectorStore to share the normalisation/grouping pipeline.
- SearchOrchestrator now owns expansion and personalisation helpers directly; MCP tools were updated to invoke the unified service without legacy stages.
- `VectorServiceRetriever` now pre-chunks documents with
  `RecursiveCharacterTextSplitter`, using `tiktoken` when available, before the
  LangChain compression pipeline to keep token reductions deterministic while
  retaining metrics wiring.
- Canonicalized query-processing payloads around the shared `SearchRecord` model and
  updated MCP response adapters to subclass the service-layer Pydantic types,
  eliminating duplicate DTO maintenance.
- VectorStoreService now emits canonical `SearchRecord` objects directly, deleting bespoke
  `SearchResult`/`SearchResultItem` models, unifying API and MCP responses, and refreshing
  the CLI/test suites around the single retrieval contract.
- Centralized VectorMatch to `SearchRecord` conversion through the new
  `SearchRecord.from_vector_match` factory and added contract-focused unit tests to guard
  the normalized payload mapping.
- Refined MCP tool registrars to route all success and error paths through the
  shared response converter, returning consistent warnings metadata and
  serializing enums via the canonical service models.
- Simplified query processing by replacing the `QueryProcessingPipeline`
  wrapper with `SearchRequest.from_input`, updating the RAG evaluation harness
  and orchestrator tests to use the shared entrypoint.
- Extended browser monitoring with the `*_browser_challenges_total` counter
  and propagated tier/runtime challenge labels through `BrowserAutomationMonitor`.
- Hardened LangGraph observability by switching the compression retriever to
  `ainvoke`, registering OpenTelemetry callbacks, and recording metrics via the
  shared registry with dedicated unit coverage.
- Hardened converter helpers and tests to use `model_dump(mode="json")`
  fallbacks, normalizing mocked inputs while preserving the JSON contract
  snapshot tracked in `tests/unit/mcp_tools/tools/test_response_converter_helpers.py`.
- Rebuilt the Playwright automation stack as a tiered pipeline (baseline
  stealth + Rebrowser undetected tier + ScrapeOps-compatible proxies +
  CapMonster integration), surfaced challenge outcomes in metadata, and
  refreshed unit coverage to validate tier escalation and captcha flows.
- Added focused unit coverage for `SearchRecord` normalization and documentation
  of the new ownership boundaries in
  `docs/developers/queries/response-contract.md`.
- Consolidated CI into a lean `core-ci.yml` pipeline (lint → tests with coverage → build → dependency audit) and introduced on-demand security and regression workflows while deleting `fast-feedback.yml`, `status-dashboard.yml`, and other scheduled automation.
- Simplified documentation checks to rely on `scripts/dev.py validate --check-docs --strict` plus strict MkDocs builds with pinned docs extras.
- Documented manual triggers for the security and regression workflows in `CONTRIBUTING.md` so contributors can opt into deeper validation without slowing default CI.
- Standardized workflow environment setup on the shared `.github/actions/setup-environment` composite and ensured all referenced actions remain pinned to immutable SHAs.
- Retired the SARIF upload path in favor of `pip-audit`, `safety`, and `bandit` reports stored as artifacts for manual review when the security workflow runs.
- Refined `README.md` with a table of contents, environment variable reference, expanded MCP integration steps, and SEO-aligned positioning for AI engineers.
- Updated repository description and topics to emphasize the retrieval-augmented documentation ingestion stack and surface the project in GitHub search.
- Rebuilt the dual-mode architecture unit suite with parametrized helpers, FeatureFlag stubs, and service factory coverage to remove brittle duplication and rely on pytest + Pydantic behaviors instead of bespoke assertions.
- Updated `src/architecture/modes.py` to drop the legacy `AI_DOCS_DEPLOYMENT__TIER` fallback and to expose typed accessors for feature and resource lookup used by the new tests.
- Replaced the property-based configuration suite with deterministic parametrized coverage that asserts full `model_dump` payloads and strict validator behavior for the unified settings models.
- Streamlined the observability configuration unit tests with fixtures and parametrization to focus on the supported Pydantic surface.
- Rebuilt `tests/unit/services/functional/test_embeddings.py` around deterministic async stubs covering provider info,
  smart recommendations, usage reports, and batch orchestration while removing brittle assertions.
- Replaced the agent orchestration test matrix with deterministic suites covering the base agent, dynamic tool discovery, and
  agentic orchestrator fallback flows using pytest fixtures instead of Hypothesis strategies.
- Removed the remaining Hypothesis utilities from `tests/utils/ai_testing_utilities.py` to consolidate deterministic embedding
  helpers that back the infrastructure smoke checks.
- Collapsed the monolithic `tests/conftest.py` into plugin registrations and a focused configuration fixture module, removing
  bespoke Hypothesis scaffolding and sys.path shims.
- Replaced the sprawling chunking regression tests with a compact `test_chunker_behavior` module that verifies semantic window
  splitting and AST fallbacks deterministically.
- Rebuilt the vector database CLI test suite around async-aware stubs and `CliRunner`, ensuring coverage targets only the
  maintained `VectorDBManager` surface.
- Replaced the batch CLI help-text smoke tests with deterministic coverage of completion helpers, dry-run previews, and
  destructive-operation confirmations using asyncio-backed stubs.
- Simplified `pytest.ini` to enforce warnings-as-errors, strict markers, and a seeded pytest-randomly configuration that aligns
  with the new deterministic fixtures.
- Rebuilt `tests/unit/mcp_tools` and `tests/unit/mcp_services` around async-aware stubs, shared MCP decorators, and focused
  validation coverage while deleting the legacy duplicated suites.
- Centralized the MCP config and tool-module builders in `tests/unit/conftest.py`, parameterized streaming guard checks, and
  added coverage for empty hybrid-search results plus orchestrator partial-initialization guard rails.
- Pinned the regression workflow `uv` dependency to `0.2.37` to stabilize CI setup.
- Centralized RAG generator management through the shared ClientManager cache for MCP tooling and FastAPI dependencies, refreshing unit coverage to assert reuse and disabled-mode handling.

### Removed

- Deleted the unused `src/services/enterprise` package and example security integration module; optional services now fail closed until replacements arrive.
- Deleted query_processing clustering, expansion, ranking, and utils modules plus their tests in favour of the final LangChain-backed stack.
- Deleted support for the deprecated `AI_DOCS_DEPLOYMENT__TIER` environment variable in favor of `AI_DOCS_MODE` as the sole mode selector.
- Removed the flaky async configuration validation unit suite that only exercised mock sleep-based helpers.
- Dropped the Hypothesis-based embedding property suite and the associated strategies/fixtures from `tests/conftest.py` to
  eliminate duplicated utilities and randomness.
- Removed the `QueryProcessingPipeline` wrapper; callers should invoke
  `SearchOrchestrator` directly and normalise inputs with
  `SearchRequest.from_input` (**SemVer: MAJOR**).

### Security

- Applied SHA pinning across composite actions and documentation snippets, aligning with GitHub’s secure-use guidance to mitigate supply-chain risk.

[Unreleased]: https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper/compare/main...HEAD
