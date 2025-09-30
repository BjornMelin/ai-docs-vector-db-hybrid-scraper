# MCP Tool and Service Unit Test Strategy

## Overview
This document captures the consolidated unit-test approach for the maintained MCP tools and services. The suites target the public contracts that remain supported after the refactor, emphasizing deterministic async coverage, strict validation, and lean fixtures.

## Feature Coverage Map
| Feature Surface | Key Behaviour | Unit Tests |
| --- | --- | --- |
| Search tool suite | `search_documents_core` caching, sparse-index safeguards, and Qdrant `search_similar` hand-off | `tests/unit/mcp_tools/test_search_utils.py`, `tests/unit/mcp_tools/test_search_tools.py` |
| Tool registry | Aggregated tool registration and optional dependency handling | `tests/unit/mcp_tools/test_tool_registry.py` |
| Request validation | Pydantic model bounds and defaults for core request types | `tests/unit/mcp_tools/test_requests.py` |
| Document service | Registers document, collection, project, crawling, and content intelligence tools | `tests/unit/mcp_services/test_services.py::test_document_service_registers_modules` |
| Analytics service | Tool wiring and observability helper exposure | `tests/unit/mcp_services/test_services.py::test_analytics_service_registers_modules`, `tests/unit/mcp_services/test_services.py::test_analytics_service_observability_tools` |
| Search service | Search tool wiring | `tests/unit/mcp_services/test_services.py::test_search_service_registers_modules` |
| System service | System tool wiring and guard rails | `tests/unit/mcp_services/test_services.py::test_system_service_registers_modules`, `tests/unit/mcp_services/test_services.py::test_register_methods_raise_when_uninitialized` |
| Orchestrator service | Initialization, agentic orchestration wiring, orchestration tool surface | `tests/unit/mcp_services/test_services.py::test_orchestrator_service_*` |
| Unified MCP server config | Streaming config guard rails and configuration validation | `tests/unit/mcp_services/test_unified_mcp_server.py` |

## Decision Record DR-001: Consolidate MCP Suites
### Option Evaluation
| Option | Solution Leverage (35%) | Application Value (30%) | Maintenance & Cognitive Load (25%) | Architectural Adaptability (10%) | Weighted Total |
| --- | --- | --- | --- | --- | --- |
| A. Retain legacy suites with minor fixes | 3 | 4 | 2 | 3 | 3.05 |
| B. Replace with focused async-aware unit tests (selected) | 9 | 8 | 8 | 7 | 8.25 |
| C. Rely solely on higher-level integration tests | 4 | 5 | 6 | 5 | 4.90 |

### Selected Approach
Option B provides the highest leverage by relying on `pytest.mark.asyncio` to exercise async orchestration boundaries, `AsyncMock` to isolate service fan-out, and Pydantic validation to guard request and response models. The resulting suites cover the maintained FastMCP surfaces without reintroducing deprecated flows while keeping fixtures narrow and shareable.

## Technical Debt Register
| Area | Issue | Severity | Impact | Maintenance Cost | Fix Effort | Dependency Risk | Notes / Decision Links |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MCP Tools | Document tool flows depend on complex external managers that remain untested in unit scope. | Medium | Failure scenarios in `documents.register_tools` are only smoke-tested; deeper integration covered elsewhere. | Medium | High | Reference external services (crawl, embeddings). | See DR-001. |
| MCP Tools | Agentic RAG optional module skipped when dependencies missing. | Low | Logging-only coverage; downstream behaviour validated in integration tests. | Low | Low | Optional dependency import. | See DR-001. |
| MCP Services | Orchestrator domain service initialization mocked to avoid heavy dependency graphs. | Medium | Async task fan-out not simulated; validated via integration. | Medium | Medium | Relies on FastMCP + service stack. | See DR-001. |
| MCP Services | Analytics observability tools return static payloads until richer fixtures exist. | Low | Unit tests assert contract only; behaviour validated via observability smoke tests. | Medium | Medium | Depends on observability adapters. | See DR-001. |

## References
- Pytest async support (`pytest.mark.asyncio`) — *pytest-asyncio* 0.23.7 documentation. Accessed 2024-12-05. <https://pytest-asyncio.readthedocs.io/en/v0.23.7/reference.html>
- `AsyncMock` for coroutine boundaries — Python 3.12.4 documentation. Accessed 2024-12-05. <https://docs.python.org/3.12/library/unittest.mock.html#unittest.mock.AsyncMock>
- Pydantic validation error handling — Pydantic 2.11 documentation. Accessed 2024-12-05. <https://docs.pydantic.dev/2.11/concepts/models/#error-handling>
