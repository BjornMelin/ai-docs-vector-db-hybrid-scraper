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
- **Summary:** Consolidated config lacks secret handling, env alias validation, and auto-detect tests.
- **Acceptance Criteria:**
  - Introduce `SecretStr` (or equivalent) for sensitive fields; secrets never exposed in logs.
  - Provide env alias matrix + profile templates (personal, production, testing) and tests covering auto-detection (`src/services/dependencies.py`).
  - Document configuration migration + rollback in `docs/operators/configuration.md`.
- **Dependencies:** QA-01
- **Owner:** Platform Engineer (TBD)
- **Evidence:** `rg "Secret" src/config/models.py` → none; `src/services/dependencies.py:55-132` auto-detection lacks tests.

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
- **Evidence:** `src/services/monitoring/middleware.py:24-138` (unused), `src/api/app_factory.py:320-380` (no instrumentation calls).

### OPS-02 – Establish Performance Benchmarks

- **Legacy IDs:** 48
- **Summary:** Portfolio claims 887.9% throughput but no automated benchmarks enforce this.
- **Acceptance Criteria:**
  - Add `pytest-benchmark` suite with saved baselines; job runs on scheduled and PR workflows.
  - Regression >10% triggers failure comment.
  - Document how to record new baselines.
- **Dependencies:** QA-01
- **Owner:** Performance Engineer (TBD)
- **Evidence:** `.github/workflows/core-ci.yml` lacks benchmark stage; `tests/benchmarks/` modules unused in CI.

### OPS-03 – Production Readiness Gate

- **Legacy IDs:** 49
- **Summary:** No consolidated go-live checklist or gating automation.
- **Acceptance Criteria:**
  - Create release checklist covering security, tests, docs, deployment rehearsals.
  - Add automation (GitHub workflow / script) verifying prerequisites before tagging releases.
  - Capture sign-off artefacts in `docs/operators/release-checklist.md`.
- **Dependencies:** QA-02, OPS-01, SEC-02
- **Owner:** Technical Program Manager (TBD)
- **Evidence:** `planning/master_report.md` immediate next steps expect such gating.

## Retrieval & Data

### RAG-01 – Deliver Runnable RAG Pipeline

- **Legacy IDs:** 7
- **Summary:** RAG generator exists but no API or integration uses it.
- **Acceptance Criteria:**
  - Compose search + RAG generator into a service and expose `POST /api/v1/rag/query`.
  - Handle API-key absence gracefully; add integration tests with mocked LLM.
  - Document end-to-end flow in `docs/users/search-and-retrieval.md`.
- **Dependencies:** INF-03, QA-01
- **Owner:** Search Lead (TBD)
- **Evidence:** `src/services/rag/generator.py:40-120`; `rg "rag" src/api/routers` → none.

### RAG-02 – Multi-Collection & Hybrid Caching

- **Legacy IDs:** 14, 19
- **Summary:** Vector service lacks runtime controls for multi-collection and Redis 8 vector caching.
- **Acceptance Criteria:**
  - Implement collection management API + DBSF hybrid search.
  - Integrate Redis vector sets for semantic cache with eviction policies.
  - Provide migration scripts and tests verifying cross-collection search quality.
- **Dependencies:** RAG-01
- **Owner:** Search Lead (TBD)
- **Evidence:** `src/services/vector_db/service.py:32-170` (single collection facade).

### RAG-03 – Extend Language Support

- **Legacy IDs:** 17
- **Summary:** Only Python/JS/TS parsers exist; Go/Rust/Java support missing.
- **Acceptance Criteria:**
  - Add tree-sitter parsers for Go, Rust, Java; update chunking pipeline.
  - Expand embeddings pipeline with language-aware tokenization.
  - Tests ingesting sample docs for each new language.
- **Dependencies:** RAG-01
- **Owner:** Search Lead (TBD)
- **Evidence:** `src/chunking.py:17-39` imports only python/javascript/typescript.

## Security & Access

### SEC-01 – Implement Enterprise SSO

- **Legacy IDs:** 18
- **Summary:** No OAuth/OIDC flows or RBAC exist in the codebase.
- **Acceptance Criteria:**
  - Add OIDC/OAuth2 provider integration with RBAC & session management.
  - Provide admin APIs for identity lifecycle; audit logging for auth events.
  - Tests covering login/logout and role enforcement.
- **Dependencies:** INF-02
- **Owner:** Security Engineer (TBD)
- **Evidence:** `rg "OAuth" src` (only benchmark mention).

### SEC-02 – Actionable Security Scanning

- **Legacy IDs:** 25
- **Summary:** Bandit/pip-audit run but results aren’t surfaced or enforced consistently.
- **Acceptance Criteria:**
  - Ensure CI fails on critical Bandit/Safety findings and uploads SARIF to code scanning.
  - Document triage workflow + escalation in `docs/security/security-testing-framework.md`.
  - Automate ticket creation for unresolved findings.
- **Dependencies:** QA-02
- **Owner:** Security Engineer (TBD)
- **Evidence:** `.github/workflows/core-ci.yml:214-233` (runs bandit without SARIF/policy).

## Browser & Crawling

### BRW-01 – Upgrade to browser-use 0.3.2

- **Legacy IDs:** 44
- **Summary:** Adapter warns "browser-use not available"; dependency missing.
- **Acceptance Criteria:**
  - Add `browser-use>=0.3.2,<0.4.0` to `pyproject` (optionally browser extra).
  - Update install docs and setup wizard to cover playwright dependencies.
  - Adapter initialization succeeds in smoke tests.
- **Dependencies:** QA-01
- **Owner:** Automation Engineer (TBD)
- **Evidence:** `pyproject.toml:36-80` (no browser-use); `src/services/browser/browser_use_adapter.py:22-59`.

### BRW-02 – Expose Browser Automation APIs

- **Legacy IDs:** 45
- **Summary:** No FastAPI endpoints expose browser automation orchestration.
- **Acceptance Criteria:**
  - Add routers for browser tasks (queue job, status, results).
  - Tie into orchestration logic with auth controls.
  - Provide integration tests mocking browser responses.
- **Dependencies:** BRW-01
- **Owner:** Automation Engineer (TBD)
- **Evidence:** `rg "browser" src/api/routers` → none.

### BRW-03 – Browser Session Resilience

- **Legacy IDs:** 46
- **Summary:** TODOs note missing rate limiter/circuit breaker integration.
- **Acceptance Criteria:**
  - Persist browser sessions in Redis with TTL and cleanup tasks.
  - Wire circuit breaker for provider failures; add chaos tests.
  - Document recovery & retry strategy.
- **Dependencies:** BRW-02, INF-03
- **Owner:** Automation Engineer (TBD)
- **Evidence:** `src/services/managers/crawling_manager.py:59-96` (TODO comments).

### BRW-04 – Browser Observability & QA

- **Legacy IDs:** 47
- **Summary:** No telemetry dashboards or dedicated test suite for browser automation.
- **Acceptance Criteria:**
  - Emit structured logs/traces for browser tiers; integrate with OPS-01 metrics.
  - Create regression tests (function + visual) covering tiers 0–4.
  - Document dashboard setup in `docs/operators/monitoring.md`.
- **Dependencies:** BRW-02, OPS-01
- **Owner:** Automation Engineer (TBD)
- **Evidence:** Lack of browser instrumentation in current codebase.

### BRW-05 – Crawl4AI Advanced Features Validation

- **Legacy IDs:** 29
- **Summary:** Memory-adaptive dispatcher logic exists but requires validation & toggles.
- **Acceptance Criteria:**
  - Provide config flags and run-time metrics for the adaptive dispatcher.
  - Create load tests comparing dispatcher vs baseline; document gains.
  - Update docs with tuning guidance.
- **Dependencies:** QA-01
- **Owner:** Automation Engineer (TBD)
- **Evidence:** `src/services/crawling/c4a_presets.py` (memory-adaptive dispatcher preset and toggles).

## Analytics & UX

### ANA-01 – Search Analytics Dashboard

- **Legacy IDs:** 8
- **Summary:** Analytics modules exist but no API or UI exposes them.
- **Acceptance Criteria:**
  - Expose analytics endpoints (FastAPI + MCP) returning aggregated search metrics.
  - Provide minimal UI (dashboard or CLI) and automated data quality checks.
  - Document usage and KPIs.
- **Dependencies:** RAG-01
- **Owner:** Product Analytics (TBD)
- **Evidence:** `src/services/analytics/search_dashboard.py` unused; routers lack analytics.

### ANA-02 – Embedding Visualization Delivery

- **Legacy IDs:** 9
- **Summary:** Visualization engine exists but not hooked into workflows.
- **Acceptance Criteria:**
  - Provide CLI/API to generate embedding visualizations with persisted artifacts.
  - Tests validating output schema and caching.
  - Docs showcasing usage for portfolio.
- **Dependencies:** ANA-01
- **Owner:** Product Analytics (TBD)
- **Evidence:** `src/services/analytics/vector_visualization.py:31-280` unused externally.

### ANA-03 – Advanced Analytics & Data Management

- **Legacy IDs:** 50
- **Summary:** HDBSCAN/advanced analytics require optional deps not packaged.
- **Acceptance Criteria:**
  - Add `hdbscan` (and related libs) to optional extras; ensure dynamic import guard.
  - Expose analytics APIs for clustering/topic modeling/federated search.
  - Provide integration tests and performance benchmarks.
- **Dependencies:** ANA-01, RAG-02
- **Owner:** Product Analytics (TBD)
- **Evidence:** `src/services/query_processing/clustering.py:509-520` (requires hdbscan) with no dependency in `pyproject`.

### UX-01 – Natural Language Query Interface

- **Legacy IDs:** 10
- **Summary:** No endpoint for conversational querying.
- **Acceptance Criteria:**
  - Implement NL query endpoint leveraging RAG + analytics signals.
  - Add latency and satisfaction metrics.
  - Document usage with examples.
- **Dependencies:** RAG-01, ANA-01
- **Owner:** Product Experience (TBD)
- **Evidence:** Routers lack NL endpoints.

## Documentation

### DOC-01 – Update Documentation & Release Notes

- **Legacy IDs:** 6
- **Summary:** Docs/roadmap and release materials predate new backlog.
- **Acceptance Criteria:**
  - Update roadmap, operators, security, and testing docs to reflect new plan.
  - Provide V1 release notes covering completed scopes and risks.
  - Ensure doc build passes (`mkdocs`/`sphinx`).
- **Dependencies:** QA-02, OPS-03
- **Owner:** Technical Writer (TBD)
- **Evidence:** `docs/roadmap.md` still references phased plan superseded by backlog.

---

id: remaining_tasks.decision_log
last_reviewed: 2025-07-02
status: draft

---

| Legacy ID                              | Disposition | Notes                                                      | Evidence                                                 | New Mapping |
| -------------------------------------- | ----------- | ---------------------------------------------------------- | -------------------------------------------------------- | ----------- |
| 1                                      | Merged      | Tests still fail (missing plugins); merged into QA-01.     | `pytest … -q` failure; `tests/test_infrastructure.py:21` | QA-01       |
| 2                                      | Retained    | Config hardening incomplete; becomes INF-01.               | `rg "Secret" src/config/models.py` → none              | INF-01      |
| 3                                      | Retained    | No global handlers; becomes INF-02.                        | `src/api/app_factory.py:224-280`                         | INF-02      |
| 4                                      | Retained    | Manager classes still present; becomes ARC-01.             | `src/services/managers/crawling_manager.py:27-101`       | ARC-01      |
| 5                                      | Retained    | Circuit breaker unused; becomes INF-03.                    | `src/services/dependencies.py` lacks references          | INF-03      |
| 6                                      | Retained    | Docs not updated; becomes DOC-01.                          | `docs/roadmap.md` (legacy roadmap)                       | DOC-01      |
| 7                                      | Retained    | RAG generator unused; becomes RAG-01.                      | `src/services/rag/generator.py`                          | RAG-01      |
| 8                                      | Retained    | Analytics dashboard unused; becomes ANA-01.                | `src/services/analytics/search_dashboard.py`             | ANA-01      |
| 9                                      | Retained    | Embedding viz not exposed; becomes ANA-02.                 | `src/services/analytics/vector_visualization.py`         | ANA-02      |
| 10                                     | Retained    | NL query absent; becomes UX-01.                            | Routers lack NL endpoints                                | UX-01       |
| 11                                     | Merged      | Auto-detect enhancements rolled into INF-01.               | `src/services/dependencies.py:55-132`                    | INF-01      |
| 12                                     | Merged      | Profile/CLI validation merged into INF-01.                 | `src/cli/commands/setup.py` integration needed           | INF-01      |
| 13                                     | Completed   | Interactive wizard implemented.                            | `src/cli/commands/setup.py`                              | —           |
| 14                                     | Retained    | Multi-collection work pending; becomes RAG-02.             | `src/services/vector_db/service.py`                      | RAG-02      |
| 17                                     | Retained    | Multi-language work outstanding; becomes RAG-03.           | `src/chunking.py:17-39`                                  | RAG-03      |
| 18                                     | Retained    | No SSO implementation; becomes SEC-01.                     | `rg "OAuth" src`                                         | SEC-01      |
| 19                                     | Merged      | Redis vector caching bundled with RAG-02.                  | Service lacks caching toggles                            | RAG-02      |
| 20                                     | Retained    | Observability middleware unused; becomes OPS-01.           | `src/services/monitoring/middleware.py`, `app_factory`   | OPS-01      |
| 21                                     | Completed   | Python 3.13 setup scripted.                                | `setup.sh:44-120`                                        | —           |
| 22                                     | Completed   | Import restructuring done (tests run once deps installed). | `pyproject.toml` + reorganized modules                   | —           |
| 23                                     | Retained    | CI still on 3.12; becomes QA-02.                           | `.github/workflows/core-ci.yml:85`                       | QA-02       |
| 24                                     | Merged      | Coverage goals merged into QA-01.                          | Planning statements; tests missing coverage gating       | QA-01       |
| 25                                     | Retained    | Security scans need enforcement; becomes SEC-02.           | `.github/workflows/core-ci.yml:214-233`                  | SEC-02      |
| 26                                     | Retained    | mypy not enforced; becomes QA-03.                          | `.github/workflows` lacking mypy                         | QA-03       |
| 29                                     | Retained    | Crawl4AI enhancements pending; becomes BRW-05.             | `src/services/crawling/c4a_provider.py`                 | BRW-05      |
| 32                                     | Superseded  | Duplicate of observability work; rolled into OPS-01.       | Planning note (cancelled)                                | OPS-01      |
| 44                                     | Retained    | browser-use dependency absent; becomes BRW-01.             | `pyproject.toml:36-80`                                   | BRW-01      |
| 45                                     | Retained    | FastAPI endpoints missing; becomes BRW-02.                 | `rg "browser" src/api/routers` → none                    | BRW-02      |
| 46                                     | Retained    | Session management TODOs; becomes BRW-03.                  | `src/services/managers/crawling_manager.py:63-96`        | BRW-03      |
| 47                                     | Retained    | Browser observability lacking; becomes BRW-04.             | No telemetry in browser modules                          | BRW-04      |
| 48                                     | Split       | Performance suite migrated to OPS-02.                      | Missing bench job                                        | OPS-02      |
| 49                                     | Retained    | Production gate incomplete; becomes OPS-03.                | Planning docs                                            | OPS-03      |
| 50                                     | Retained    | Advanced analytics packaged as ANA-03.                     | `src/services/query_processing/clustering.py:509`        | ANA-03      |
| Original backlog IDs 15,16,27,28,30,31 | Not present | No corresponding `.taskmaster` entries surfaced.           | —                                                        | —           |

---

id: remaining_tasks.risk_register
last_reviewed: 2025-07-02
status: draft

---

| Risk ID | Description                                                                                               | Impact      | Likelihood | Mitigation / Linked Tasks              | Owner             | Status | Evidence                                                                  |
| ------- | --------------------------------------------------------------------------------------------------------- | ----------- | ---------- | -------------------------------------- | ----------------- | ------ | ------------------------------------------------------------------------- |
| R1      | Automated tests cannot run due to missing pytest extras, preventing regression detection.                 | High        | High       | QA-01 (install extras, restore pytest) | QA Lead           | Open   | `pytest … -q` failure (missing plugins)                                   |
| R2      | Observability middleware never attached; production incidents would lack metrics/traces.                  | High        | Medium     | OPS-01 (wire Prometheus/OTel)          | Observability Eng | Open   | `src/api/app_factory.py:320-380`                                          |
| R3      | Browser automation disabled by missing `browser-use` dependency; tiered crawling relies on fallback only. | High        | Medium     | BRW-01..BRW-03                         | Automation Eng    | Open   | `src/services/browser/browser_use_adapter.py:22-59`                       |
| R4      | Circuit breaker logic unused; upstream outages propagate directly to clients.                             | Medium-High | Medium     | INF-03                                 | Platform Eng      | Open   | `src/services/circuit_breaker/circuit_breaker_manager.py` vs dependencies |
| R5      | Unified config stores secrets as plain strings; no secret management or rotation story.                   | High        | Medium     | INF-01                                 | Platform Eng      | Open   | `rg "SecretStr" src/config/models.py` → none                            |

---

id: remaining_tasks.migration_notes
last_reviewed: 2025-07-02
status: draft

---

## Migration Notes – GitHub Issues & Projects

## Overview

Import the canonical backlog into GitHub Issues/Projects while preserving legacy traceability and new ownership fields. Use IDs (e.g., `QA-01`) as canonical references and include legacy IDs in issue bodies.

## Issue Template

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

## Labels & Metadata

- `Category::Quality`, `Category::Platform`, `Category::Ops`, `Category::Retrieval`, `Category::Security`, `Category::Browser`, `Category::Analytics`, `Category::Docs`
- `Status::Pending`, `Status::InProgress`, `Status::Done`
- `Priority::P0/P1/P2` (set during triage)
- `LegacyID::###` (optional custom label if automated parsing desired)

## Dependency Management

- Use GitHub issue linking (`blocks` / `blocked by`) reflecting backlog dependencies.
- Add automation (Projects v2) to auto-update item status when linked issues close.

## Import Steps

1. Export `backlog.md` to CSV (columns: Title, Body, Labels). Include summary + template fields per task.
2. Use GitHub Issue Importer (beta) or `gh issue import` to bulk create issues.
3. Create a Project board with views by Category and Status; auto-add new issues via filter `label:Category::*`.
4. Populate dependencies manually or via `gh issue edit --add-project` and `--add-assignee`.
5. Sync `decision_log.md` as project note for auditors; reference it in issue descriptions.

## Automation Hooks

- Extend existing workflows to post status back to the Project when issues close.
- Consider adding CODEOWNERS mappings based on Owner placeholders (e.g., `QA Lead` → `@org/qa-team`).
- Schedule weekly report aggregating Project status against `risk_register.md`.

## Legacy References

Keep `decision_log.md` in repo to translate future discoveries; link to it from each issue (`See backlog decision log for provenance`). This satisfies traceability expectations during audits.
