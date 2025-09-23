---
id: remaining_tasks.decision_log
last_reviewed: 2025-07-02
status: draft
---

| Legacy ID | Disposition | Notes | Evidence | New Mapping |
| --- | --- | --- | --- | --- |
| 1 | Merged | Tests still fail (missing plugins); merged into QA-01. | `pytest … -q` failure; `tests/test_infrastructure.py:21` | QA-01 |
| 2 | Retained | Config hardening incomplete; becomes INF-01. | `rg "Secret" src/config/settings.py` → none | INF-01 |
| 3 | Retained | No global handlers; becomes INF-02. | `src/api/app_factory.py:224-280` | INF-02 |
| 4 | Retained | Manager classes still present; becomes ARC-01. | `src/services/managers/crawling_manager.py:27-101` | ARC-01 |
| 5 | Retained | Circuit breaker unused; becomes INF-03. | `src/services/dependencies.py` lacks references | INF-03 |
| 6 | Retained | Docs not updated; becomes DOC-01. | `docs/roadmap.md` (legacy roadmap) | DOC-01 |
| 7 | Retained | RAG generator unused; becomes RAG-01. | `src/services/rag/generator.py` | RAG-01 |
| 8 | Retained | Analytics dashboard unused; becomes ANA-01. | `src/services/analytics/search_dashboard.py` | ANA-01 |
| 9 | Retained | Embedding viz not exposed; becomes ANA-02. | `src/services/analytics/vector_visualization.py` | ANA-02 |
| 10 | Retained | NL query absent; becomes UX-01. | Routers lack NL endpoints | UX-01 |
| 11 | Merged | Auto-detect enhancements rolled into INF-01. | `src/services/dependencies.py:55-132` | INF-01 |
| 12 | Merged | Profile/CLI validation merged into INF-01. | `src/cli/commands/setup.py` integration needed | INF-01 |
| 13 | Completed | Interactive wizard implemented. | `src/cli/commands/setup.py` | — |
| 14 | Retained | Multi-collection work pending; becomes RAG-02. | `src/services/vector_db/service.py` | RAG-02 |
| 17 | Retained | Multi-language work outstanding; becomes RAG-03. | `src/chunking.py:17-39` | RAG-03 |
| 18 | Retained | No SSO implementation; becomes SEC-01. | `rg "OAuth" src` | SEC-01 |
| 19 | Merged | Redis vector caching bundled with RAG-02. | Service lacks caching toggles | RAG-02 |
| 20 | Retained | Observability middleware unused; becomes OPS-01. | `src/services/monitoring/middleware.py`, `app_factory` | OPS-01 |
| 21 | Completed | Python 3.13 setup scripted. | `setup.sh:44-120` | — |
| 22 | Completed | Import restructuring done (tests run once deps installed). | `pyproject.toml` + reorganized modules | — |
| 23 | Retained | CI still on 3.12; becomes QA-02. | `.github/workflows/core-ci.yml:85` | QA-02 |
| 24 | Merged | Coverage goals merged into QA-01. | Planning statements; tests missing coverage gating | QA-01 |
| 25 | Retained | Security scans need enforcement; becomes SEC-02. | `.github/workflows/core-ci.yml:214-233` | SEC-02 |
| 26 | Retained | mypy not enforced; becomes QA-03. | `.github/workflows` lacking mypy | QA-03 |
| 29 | Retained | Crawl4AI enhancements pending; becomes BRW-05. | `src/services/crawling/crawl4ai_provider.py` | BRW-05 |
| 32 | Superseded | Duplicate of observability work; rolled into OPS-01. | Planning note (cancelled) | OPS-01 |
| 44 | Retained | browser-use dependency absent; becomes BRW-01. | `pyproject.toml:36-80` | BRW-01 |
| 45 | Retained | FastAPI endpoints missing; becomes BRW-02. | `rg "browser" src/api/routers` → none | BRW-02 |
| 46 | Retained | Session management TODOs; becomes BRW-03. | `src/services/managers/crawling_manager.py:63-96` | BRW-03 |
| 47 | Retained | Browser observability lacking; becomes BRW-04. | No telemetry in browser modules | BRW-04 |
| 48 | Split | Performance suite migrated to OPS-02. | Missing bench job | OPS-02 |
| 49 | Retained | Production gate incomplete; becomes OPS-03. | Planning docs | OPS-03 |
| 50 | Retained | Advanced analytics packaged as ANA-03. | `src/services/query_processing/clustering.py:509` | ANA-03 |
| Original backlog IDs 15,16,27,28,30,31 | Not present | No corresponding `.taskmaster` entries surfaced. | — | — |
