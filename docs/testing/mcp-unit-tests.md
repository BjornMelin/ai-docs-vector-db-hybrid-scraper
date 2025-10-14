# MCP Tool and Service Unit Test Strategy

## Overview

This document captures the consolidated unit-test approach for the maintained MCP tools and services. The suites target the public contracts that remain supported after the refactor, emphasizing deterministic async coverage, strict validation, and lean fixtures.

## Feature Coverage Map

| Feature Surface           | Key Behaviour                                                                              | Unit Tests                                     |
| ------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------- |
| Search tool suite         | FastMCP search entrypoints (`search_documents`, `search_similar`) calling vector services   | `tests/unit/mcp_tools/test_search_tools.py`     |
| Tool registry             | Aggregated tool registration and optional dependency handling                              | `tests/unit/mcp_tools/test_tool_registry.py`    |
| Request validation        | Pydantic model bounds and defaults for core request types                                  | `tests/unit/mcp_tools/test_requests.py`         |
| Unified MCP server config | Streaming config guard rails and configuration validation                                   | `tests/unit/test_unified_mcp_server.py`         |
| MCP server lifespan       | Container wiring, dependency resolution, and monitoring task orchestration                 | `tests/integration/mcp/test_server_lifespan.py` |

## Decision Record DR-001: Consolidate MCP Suites

### Option Evaluation

| Option                                                    | Solution Leverage (35%) | Application Value (30%) | Maintenance & Cognitive Load (25%) | Architectural Adaptability (10%) | Weighted Total |
| --------------------------------------------------------- | ----------------------- | ----------------------- | ---------------------------------- | -------------------------------- | -------------- |
| A. Retain legacy suites with minor fixes                  | 3                       | 4                       | 2                                  | 3                                | 3.05           |
| B. Replace with focused async-aware unit tests (selected) | 9                       | 8                       | 8                                  | 7                                | 8.25           |
| C. Rely solely on higher-level integration tests          | 4                       | 5                       | 6                                  | 5                                | 4.90           |

### Selected Approach

Option B provides the highest leverage by relying on `pytest.mark.asyncio` to exercise async orchestration boundaries, `AsyncMock` to isolate service fan-out, and Pydantic validation to guard request and response models.
The resulting suites cover the maintained FastMCP surfaces without reintroducing deprecated flows while keeping fixtures narrow and shareable.

## Decision Record DR-002: Share Fixture Builders and Parameterize Streaming Guards

### Option Evaluation

| Option                                                         | Solution Leverage (35%) | Application Value (30%) | Maintenance & Cognitive Load (25%) | Architectural Adaptability (10%) | Weighted Total |
| -------------------------------------------------------------- | ----------------------- | ----------------------- | ---------------------------------- | -------------------------------- | -------------- |
| A. Keep per-module helpers with duplicated logic               | 4                       | 5                       | 3                                  | 4                                | 4.05           |
| B. Promote shared builders with parametrized checks (selected) | 9                       | 8                       | 8                                  | 8                                | 8.35           |
| C. Introduce bespoke helper classes per suite                  | 6                       | 6                       | 4                                  | 5                                | 5.40           |

### Selected Approach

Option B centralizes the configuration and tool-module builders in `tests/unit/conftest.py`, which reduces duplication while keeping the helpers inline with standard pytest fixture discovery. Parametrizing the streaming configuration tests via `pytest.mark.parametrize`
trims boilerplate and keeps guard-rail expectations explicit in a data-driven table, aligning with pytest's recommended patterns for combinatorial coverage.

## Technical Debt Register

| Area         | Issue                                                                                       | Severity | Impact                                                                                                       | Maintenance Cost | Fix Effort | Dependency Risk                                  | Notes / Decision Links |
| ------------ | ------------------------------------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------ | ---------------- | ---------- | ------------------------------------------------ | ---------------------- |
| MCP Tools    | Document tool flows depend on complex external managers that remain untested in unit scope. | Medium   | Failure scenarios in `documents.register_tools` are only smoke-tested; deeper integration covered elsewhere. | Medium           | High       | Reference external services (crawl, embeddings). | See DR-001.            |
| MCP Tools    | Agentic RAG optional module skipped when dependencies missing.                              | Low      | Logging-only coverage; downstream behaviour validated in integration tests.                                  | Low              | Low        | Optional dependency import.                      | See DR-001.            |

## References

- Pytest async support (`pytest.mark.asyncio`) — _pytest-asyncio_ 0.23.7 documentation. Accessed 2024-12-05. <https://pytest-asyncio.readthedocs.io/en/v0.23.7/reference.html>
- Pytest parametrization (`pytest.mark.parametrize`) — _pytest_ 8.3.x documentation. Accessed 2024-12-05. <https://docs.pytest.org/en/8.3.x/how-to/parametrize.html>
- `AsyncMock` for coroutine boundaries — Python 3.12.4 documentation. Accessed 2024-12-05. <https://docs.python.org/3.12/library/unittest.mock.html#unittest.mock.AsyncMock>
- Pydantic validation error handling — Pydantic 2.11 documentation. Accessed 2024-12-05. <https://docs.pydantic.dev/2.11/concepts/models/#error-handling>
