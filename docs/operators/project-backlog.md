---
id: remaining_tasks.summary
last_reviewed: 2025-07-02
status: draft
---

# Backlog Snapshot (Condensed)

This page keeps a lightweight view of the shared backlog. Each entry links to the audience that
owns the work and the most relevant evidence. For the archival research deck see commit
`14c4acb^`.

## Quality & Testing

| ID    | Goal                                 | Owner (placeholder) | Evidence                             |
| ----- | ------------------------------------ | ------------------- | ------------------------------------ |
| QA-01 | Restore green pytest run with extras | QA Lead             | `uv run pytest -q` (fails: plugins)  |
| QA-02 | Add Python 3.13 to CI matrix         | QA Lead             | `.github/workflows/ci.yml`          |
| QA-03 | Enforce mypy in CI                   | QA Lead             | `pyproject.toml` (no CI job)        |

## Platform & Configuration

| ID    | Goal                                          | Owner              | Evidence                             |
| ----- | --------------------------------------------- | ------------------ | ------------------------------------ |
| INF-01 | Secrets + env alias hardening                | Platform Engineer  | `src/config/models.py`               |
| INF-02 | Centralised FastAPI error handling           | Platform Engineer  | `src/api/app_factory.py`             |
| INF-03 | Circuit breaker injected for outbound calls  | Platform Engineer  | `src/services/service_resolver.py`   |

## Observability & Operations

| ID    | Goal                                 | Owner             | Evidence                              |
| ----- | ------------------------------------ | ----------------- | ------------------------------------- |
| OPS-01 | Turn on Prometheus/OTel middleware | Observability Eng | Middleware defined but not mounted    |
| OPS-02 | Deterministic RAG evaluation harness | Observability Eng | `scripts/eval/rag_golden_eval.py`     |
| OPS-03 | Release gate with QA + docs checks | Operations Lead   | `.github/workflows/release.yml`       |

## Retrieval & Data

| ID    | Goal                                   | Owner              | Evidence                                     |
| ----- | -------------------------------------- | ------------------ | -------------------------------------------- |
| RAG-01 | Wire LangGraph RAG pipeline           | Retrieval Engineer | `src/services/rag/pipeline.py` feature flag  |
| RAG-02 | Multi-collection / caching guidance   | Retrieval Engineer | Removal noted in `query_processing_response` |
| RAG-03 | Multilingual ingestion path           | Data Engineer      | No locale handling in ingestion services     |

## Security & Browser

| ID    | Goal                                 | Owner              | Evidence                                         |
| ----- | ------------------------------------ | ------------------ | ------------------------------------------------ |
| SEC-01 | Enterprise SSO                       | Security Engineer  | Enterprise mode toggle lacks SSO plumbing       |
| SEC-02 | Actionable dependency/container scans | Security Engineer | No CI step for Snyk/Trivy results               |
| BRW-01 | Upgrade to `browser-use>=0.3.2`      | Automation Engineer| `pyproject.toml` only lists pydantic constraint |
| BRW-02 | REST endpoints for browser orchestration | Automation Engineer | Missing router                                 |
| BRW-03 | Resilience & metrics for browser tiers | Automation Engineer | Tests + metrics absent                          |

## Analytics & Docs

| ID     | Goal                             | Owner              | Evidence                                 |
| ------ | -------------------------------- | ------------------ | ---------------------------------------- |
| ANA-01 | Search analytics dashboard       | Analytics Engineer | No dashboard defined                     |
| ANA-02 | Embedding visualisation tooling  | Analytics Engineer | No tooling in repo                       |
| UX-01  | Natural-language query interface | UX Engineer        | Product backlog item only                |
| DOC-01 | Keep docs aligned with backlog   | Docs Team          | Mixed quality & multiple stale entries   |

---

### Using This Backlog

- Track the latest status in GitHub Issues/Projects; mirror the ID (e.g., `QA-01`) in issue titles.
- Update the “Owner” column once work is assigned; until then it reflects the expected discipline.
- When a row is completed, PRs should mention the ID so this table stays honest.
