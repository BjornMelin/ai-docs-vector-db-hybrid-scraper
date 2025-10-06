---
id: remaining_tasks.backlog
last_reviewed: 2025-07-02
status: draft
---

# Canonical Backlog

## Quality & Testing

### QA-01 – Stabilize Test Execution & Coverage

- **Legacy IDs:** 1, 24
- **Summary:** pytest currently fails because required plugins are not installed; coverage targets are unmet.
- **Acceptance Criteria:**
  - `uv sync --dev` installs pytest extras (`pytest-asyncio`, `pytest-cov`, `pytest-xdist`, `pytest-timeout`, `pytest-env`).
  - `uv run pytest -q` succeeds across `tests/` without missing-plugin errors.
  - Coverage ≥38 % overall and ≥90 % for V1-critical modules; mutation smoke via `mutmut` documented.
  - Test execution guide updated under `docs/testing/`.
- **Dependencies:** —
- **Owner:** QA Lead (TBD)
- **Evidence:** `pytest tests/unit/ai/test_embedding_properties.py -q` → missing plugin error (2025‑07‑02); `tests/test_infrastructure.py:21`.

### QA-02 – Modernize CI for Python 3.13

- **Legacy IDs:** 23
- **Summary:** Core CI still pins Python 3.12; needs 3.13 matrix + coverage artifacts.
- **Acceptance Criteria:**
  - `.github/workflows/core-ci.yml` runs on a matrix including `3.13` (with caches).
  - CI fails on test/coverage regressions and uploads coverage reports.
  - Release workflow uses pinned `uv python pin 3.13`.
- **Dependencies:** QA-01
- **Owner:** QA Lead (TBD)
- **Evidence:** `.github/workflows/core-ci.yml:85` (python-version `'3.12'` only).

### QA-03 – Enforce Static Typing in CI

- **Legacy IDs:** 26
- **Summary:** mypy config exists but no CI job executes it.
- **Acceptance Criteria:**
  - Add CI job running `uv run mypy` with baseline cache and HTML report.
  - Type violations fail the pipeline; exclusions documented.
  - `docs/developers/testing-best-practices.md` updated with typing expectations.
- **Dependencies:** QA-01
- **Owner:** QA Lead (TBD)
- **Evidence:** `pyproject.toml:748` (mypy config) vs `.github/workflows` (no mypy step).

## Configuration & Platform

### INF-01 – Harden Unified Configuration & Secrets

- **Legacy IDs:** 2, 11, 12
- **Summary:** Consolidated config lacks secret handling and environment alias validation.
- **Acceptance Criteria:**
  - Introduce `SecretStr` (or equivalent) for sensitive fields; secrets never exposed in logs.
  - Provide env alias matrix + profile templates (personal, production, testing) with unit coverage for config loading edge cases.
  - Document configuration migration + rollback in `docs/operators/configuration.md`.
- **Dependencies:** QA-01
- **Owner:** Platform Engineer (TBD)
- **Evidence:** `rg "Secret" src/config/models.py` → none; config alias handling lacks coverage in `tests/unit/infrastructure/test_pydantic_settings_patterns.py`.

### INF-02 – Centralize API Error Handling

- **Legacy IDs:** 3
- **Summary:** No global exception handlers; routers raise raw `HTTPException`.
- **Acceptance Criteria:**
  - Implement exception handler layer with consistent error schema (code, message, trace id).
  - Hook handlers in `app_factory` and `mcp_services`; add unit tests for 4xx/5xx sanitization.
  - Structured logging (with correlation IDs) emitted for unexpected failures.
- **Dependencies:** INF-01
- **Owner:** Platform Engineer (TBD)
- **Evidence:** `src/api/app_factory.py:224-280` (no `add_exception_handler` usage).

### ARC-01 – Finish Service Layer Flattening

- **Legacy IDs:** 4
- **Summary:** Manager classes remain and contain TODOs; functional refactor incomplete.
- **Acceptance Criteria:**
  - Replace class-based `services/managers/*` modules with function-first providers (dependency-injected).
  - Update `ModeAwareServiceFactory` to register lightweight callables; remove unused lifecycle shims.
  - Integration tests confirm service init/cleanup works without manager wrappers.
- **Dependencies:** INF-01
- **Owner:** Platform Architect (TBD)
- **Evidence:** `src/services/managers/crawling_manager.py:27-101` (class + TODOs).

### INF-03 – Wire Circuit Breaker Manager

- **Legacy IDs:** 5
- **Summary:** `CircuitBreakerManager` exists but is never consumed by dependencies.
- **Acceptance Criteria:**
  - Inject `CircuitBreakerManager` via `src/services/dependencies.py` for external clients (Qdrant, browser automation, Firecrawl).
  - Add resilience tests simulating open/half-open transitions with metrics exposed.
  - Document breaker tuning in `docs/operators/monitoring.md`.
- **Dependencies:** INF-01
- **Owner:** Platform Engineer (TBD)
- **Evidence:** `src/services/circuit_breaker/circuit_breaker_manager.py:23-251` vs `src/services/dependencies.py` (no references).

## Observability & Operations

### OPS-01 – Activate Observability Stack

- **Legacy IDs:** 20, 32
- **Summary:** Prometheus middleware is defined but never attached; OTEL instrumentation absent.
- **Acceptance Criteria:**
  - Register `PrometheusMiddleware` and OTEL exporters in FastAPI startup (enterprise & simple modes) and MCP services.
  - `/metrics` and `/health` respond with live data; Prometheus scrape documented.
  - CI smoke test hits `/metrics` and asserts key series.
- **Dependencies:** INF-03
- **Owner:** Observability Engineer (TBD)
- **Evidence:** `src/api/app_factory.py:320-380` (middleware never registered).

### OPS-02 – Baseline Performance Benchmarks

- **Legacy IDs:** 28
- **Summary:** No automated benchmarks verify query latency, ingestion throughput, or crawl speeds.
- **Acceptance Criteria:**
  - Build reproducible benchmark harness (Jupyter or `scripts/perf_bench.py`) with seeded datasets.
  - Capture baseline metrics (p50/p90 latency, throughput, resource usage) and publish to Grafana.
  - Document re-run instructions in `docs/testing/performance.md`.
- **Dependencies:** OPS-01
- **Owner:** Performance Engineer (TBD)
- **Evidence:** `tests/performance/` contains only TODO placeholders; Grafana dashboards absent.

### OPS-03 – Production Release Gate

- **Legacy IDs:** 49
- **Summary:** No release gating automation; tags can be pushed without passing QA/observability checks.
- **Acceptance Criteria:**
  - Implement release workflow requiring green CI, minimum coverage, and manual approval.
  - Gate ensures docs + release notes updated.
  - Document release checklist in `docs/operators/release-notes.md`.
- **Dependencies:** OPS-01, QA-02
- **Owner:** Operations Lead (TBD)
- **Evidence:** `.github/workflows/release.yml` lacks gating steps.

## Retrieval & Data

### RAG-01 – Runnable RAG Pipeline

- **Legacy IDs:** 6, 14
- **Summary:** Graph-based RAG pipeline exists behind feature flags; needs final wiring.
- **Acceptance Criteria:**
  - Integrate RAG pipeline into production API with toggles and metrics.
  - Unit/integration tests cover retrieval, grading, generation stages.
  - Document pipeline in `docs/developers/architecture.md` + `users/search-and-retrieval.md`.
- **Dependencies:** OPS-01
- **Owner:** Retrieval Engineer (TBD)
- **Evidence:** `src/services/rag/pipeline.py` flagged with TODOs; API endpoints disabled.

### RAG-02 – Multi-Collection & Caching Strategy

- **Legacy IDs:** 7
- **Summary:** Multi-collection fan-out removed; need a defined approach or docs for multi-tenancy.
- **Acceptance Criteria:**
  - Either reintroduce multi-collection via configuration or publish guidance for per-collection isolation.
  - Implement caching policy for hybrid queries.
  - Update `docs/developers/service_adapters.md` accordingly.
- **Dependencies:** RAG-01
- **Owner:** Retrieval Engineer (TBD)
- **Evidence:** `docs/query_processing_response_contract.md` notes multi-collection removed; no replacement guidance.

### RAG-03 – Multi-Language Ingestion

- **Legacy IDs:** 9
- **Summary:** Ingestion pipeline lacks multilingual support.
- **Acceptance Criteria:**
  - Add language detection and routing for non-English docs.
  - Ensure embedding models selected per language; update chunking.
  - Document workflow in `docs/users/web-scraping.md`.
- **Dependencies:** OPS-02
- **Owner:** Data Engineer (TBD)
- **Evidence:** `src/services/ingestion/` lacks language handling; backlog references multilingual goals.

## Security & Access

### SEC-01 – Enterprise SSO Integration

- **Legacy IDs:** 8
- **Summary:** Enterprise mode requires SSO support (e.g., Okta, Azure AD).
- **Acceptance Criteria:**
  - Implement OAuth/OIDC integration with tenant configuration.
  - Add automated tests for login/logout flows.
  - Document SSO setup in `docs/operators/security.md`.
- **Dependencies:** INF-01
- **Owner:** Security Engineer (TBD)
- **Evidence:** Enterprise flag toggles nonexistent SSO endpoint.

### SEC-02 – Actionable Security Scans

- **Legacy IDs:** 10
- **Summary:** Security scans run but results not integrated into CI nor actionable.
- **Acceptance Criteria:**
  - Integrate dependency and container scans into CI with fail-on-high.
  - Publish remediation workflow.
  - Update security docs with scanning procedures.
- **Dependencies:** QA-02
- **Owner:** Security Engineer (TBD)
- **Evidence:** `.github/workflows/security.yml` missing; CLI instructions stale.

## Browser & Crawling

### BRW-01 – Upgrade to browser-use 0.3.2

- **Legacy IDs:** 44
- **Summary:** Adapter warns "browser-use not available"; dependency missing.
- **Acceptance Criteria:**
  - Add `browser-use>=0.3.2,<0.4.0` to `pyproject` (optionally browser extra).
  - Update adapter + tier selection to use new API.
  - Unit tests cover failure modes.
- **Dependencies:** QA-01
- **Owner:** Automation Engineer (TBD)
- **Evidence:** `pyproject.toml:36-80` (no browser-use); `src/services/browser/browser_use_adapter.py:22-59`.

### BRW-02 – FastAPI Endpoints for Browser Orchestration

- **Legacy IDs:** 45
- **Summary:** No FastAPI endpoints expose browser automation orchestration.
- **Acceptance Criteria:**
  - Implement REST endpoints to manage crawl jobs (start/stop/status).
  - Add RBAC and rate limiting for endpoints.
  - Document usage in `docs/users/web-scraping.md`.
- **Dependencies:** BRW-01
- **Owner:** Automation Engineer (TBD)
- **Evidence:** `src/api/routers/browser.py` missing; backlog references planned endpoints.

### BRW-03 – Resilience & QA for Browser Stack

- **Legacy IDs:** 46
- **Summary:** Browser stack lacks resilience tests and chaos scenarios.
- **Acceptance Criteria:**
  - Implement chaos tests for playwright/browser-use tiers.
  - Add retry/backoff strategies.
  - Document resilience playbook.
- **Dependencies:** BRW-01
- **Owner:** Automation Engineer (TBD)
- **Evidence:** Tests only cover happy path; no chaos coverage.

### BRW-04 – Browser Metrics Instrumentation

- **Legacy IDs:** 47
- **Summary:** No metrics for browser tier selection/challenges.
- **Acceptance Criteria:**
  - Emit metrics (`*_browser_requests_total`, challenge outcomes).
  - Wire metrics into observability stack.
  - Update docs.
- **Dependencies:** OPS-01
- **Owner:** Automation Engineer (TBD)
- **Evidence:** `docs/observability/query_processing_metrics.md` notes placeholders.

### BRW-05 – QA Playbook for Browser Stack

- **Legacy IDs:** 48
- **Summary:** QA lacks guidance for browser tiers.
- **Acceptance Criteria:**
  - Create QA playbook with test matrix, fixture setup.
  - Update docs.
- **Dependencies:** BRW-01
- **Owner:** QA + Automation (TBD)
- **Evidence:** Gap noted in backlog.

## Analytics & User Experience

### ANA-01 – Search Analytics Dashboard

- **Legacy IDs:** 17, 18
- **Summary:** No dashboards for search queries.
- **Acceptance Criteria:**
  - Build analytics dashboard for query volume, success rate, top queries.
  - Integrate with observability stack.
  - Document usage.
- **Dependencies:** OPS-01
- **Owner:** Analytics Engineer (TBD)
- **Evidence:** Observability docs lack analytics.

### ANA-02 – Embedding Visualization Tools

- **Legacy IDs:** 19
- **Summary:** No visualization for embeddings.
- **Acceptance Criteria:**
  - Provide tools (UMAP/TSNE) for embedding inspection.
  - Integrate into developer docs.
- **Dependencies:** ANA-01
- **Owner:** Analytics Engineer (TBD)
- **Evidence:** Developer docs mention future visualization.

### ANA-03 – Advanced Insights & Notifications

- **Legacy IDs:** 50
- **Summary:** Analytics features planned but unbuilt.
- **Acceptance Criteria:**
  - Implement insight generation and notifications.
  - Document.
- **Dependencies:** ANA-01
- **Owner:** Analytics Engineer (TBD)
- **Evidence:** Backlog references.

### UX-01 – Natural Language Query Interface

- **Legacy IDs:** 21
- **Summary:** Need NL query interface.
- **Acceptance Criteria:**
  - Develop NL interface.
  - Add tests & docs.
- **Dependencies:** RAG-01
- **Owner:** UX Engineer (TBD)
- **Evidence:** Product roadmap.

## Documentation

### DOC-01 – Documentation Alignment

- **Legacy IDs:** 52
- **Summary:** Docs need updates to align with backlog.
- **Acceptance Criteria:**
  - Update roadmap, operator guides, release notes.
  - Add docs for new features.
- **Dependencies:** All domain tasks.
- **Owner:** Docs Team (TBD)
- **Evidence:** Remaining work backlog.

## Decision Log

- Legacy ID 13 merged into QA-02.
- Legacy IDs 15,16,27,28,30,31 had no corresponding tasks; monitoring ongoing.

## Risk Register

| Risk ID | Description                                                                                               | Impact      | Likelihood | Mitigation / Linked Tasks              | Owner             | Status | Evidence                                                                  |
| ------- | --------------------------------------------------------------------------------------------------------- | ----------- | ---------- | -------------------------------------- | ----------------- | ------ | ------------------------------------------------------------------------- |
| R1      | Automated tests cannot run due to missing pytest extras, preventing regression detection.                 | High        | High       | QA-01 (install extras, restore pytest) | QA Lead           | Open   | `pytest … -q` failure (missing plugins)                                   |
| R2      | Observability middleware never attached; production incidents would lack metrics/traces.                  | High        | Medium     | OPS-01 (wire Prometheus/OTel)          | Observability Eng | Open   | `src/api/app_factory.py:320-380`                                          |
| R3      | Browser automation disabled by missing `browser-use` dependency; tiered crawling relies on fallback only. | High        | Medium     | BRW-01..BRW-03                         | Automation Eng    | Open   | `src/services/browser/browser_use_adapter.py:22-59`                       |
| R4      | Circuit breaker logic unused; upstream outages propagate directly to clients.                             | Medium-High | Medium     | INF-03                                 | Platform Eng      | Open   | `src/services/circuit_breaker/circuit_breaker_manager.py` vs dependencies |
| R5      | Unified config stores secrets as plain strings; no secret management or rotation story.                   | High        | Medium     | INF-01                                 | Platform Eng      | Open   | `rg "SecretStr" src/config/models.py` → none                            |

## Migration Notes – GitHub Issues & Projects

Import the canonical backlog into GitHub Issues/Projects while preserving legacy traceability and new ownership fields. Use IDs (e.g., `QA-01`) as canonical references and include legacy IDs in issue bodies.

### Issue Template

```
## Summary
<short task statement>

## Legacy References
- Legacy IDs: <comma-separated>
- Evidence: <paths / commands>

## Acceptance Criteria
- ...
- ...

## Dependencies
- Blocks: <ID list or n/a>
- Blocked by: <ID list or n/a>

## Owner
<role or assignee placeholder>

## Notes
<any context, risks, or follow-ups>
```

### Labels & Metadata

- `Category::Quality`, `Category::Platform`, `Category::Ops`, `Category::Retrieval`, `Category::Security`, `Category::Browser`, `Category::Analytics`, `Category::Docs`
- `Status::Pending`, `Status::InProgress`, `Status::Done`
- `Priority::P0/P1/P2` (set during triage)
- `LegacyID::###` (optional custom label if automated parsing desired)

### Dependency Management

- Use GitHub issue linking (`blocks` / `blocked by`) reflecting backlog dependencies.
- Add automation (Projects v2) to auto-update item status when linked issues close.

### Import Steps

1. Export `backlog.md` to CSV (columns: Title, Body, Labels). Include summary + template fields per task.
2. Use GitHub Issue Importer (beta) or `gh issue import` to bulk create issues.
3. Create a Project board with views by Category and Status; auto-add new issues via filter `label:Category::*`.
4. Populate dependencies manually or via `gh issue edit --add-project` and `--add-assignee`.
5. Sync `decision_log.md` as project note for auditors; reference it in issue descriptions.

### Automation Hooks

- Extend existing workflows to post status back to the Project when issues close.
- Consider adding CODEOWNERS mappings based on Owner placeholders (e.g., `QA Lead` → `@org/qa-team`).
- Schedule weekly report aggregating Project status against `risk_register.md`.

### Legacy References

Keep `decision_log.md` in repo to translate future discoveries; link to it from each issue (`See backlog decision log for provenance`). This satisfies traceability expectations during audits.
