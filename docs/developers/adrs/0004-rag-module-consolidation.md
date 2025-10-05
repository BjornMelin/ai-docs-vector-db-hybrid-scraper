# ADR 0004: Consolidate RAG Compression Utilities Under `src/services/rag`

**Date:** 2025-02-21  
**Status:** Accepted  
**Drivers:** Avoid cross-package coupling between query processing and shared RAG stack, enable LangChain compression reuse by future services, reduce duplicated test scaffolding, simplify observability wiring  
**Deciders:** AI Docs Platform Team

## Context

- Legacy compression helpers lived under `src/services/query_processing/rag`, yet the primary consumers are the shared `VectorServiceRetriever` (`src/services/rag/retriever.py`), the query processing orchestrator, and CI/evaluation scripts.
- This layout creates an inverted dependency: the core RAG package must import `src.services.query_processing.rag`, producing module tangles during initialization and complicating future reuse by other entrypoints (CLI, MCP, batch summarisation).
- Compression metrics and CI gate require exposing compression utilities directly from the RAG package so telemetry and harnesses remain aligned without extra adapter layers.

## Decision Framework

| Criteria           | Weight   | Consolidate compression helpers into the shared RAG module | Keep packages separate with cross-imports |
| ------------------ | -------- | ---------------------------------------------------------- | ----------------------------------------- |
| Leverage           | 35%      | 10.0                                                       | 6.0                                       |
| Value              | 30%      | 9.5                                                        | 6.5                                       |
| Maintenance        | 25%      | 9.7                                                        | 5.5                                       |
| Adaptability       | 10%      | 9.0                                                        | 6.5                                       |
| **Weighted Total** | **100%** | **9.68**                                                   | **6.25**                                  |

**Rationale:** Consolidation scores higher because it eliminates cross-package
imports, exposes compression as a first-class RAG concern, and lets multiple
services consume the same telemetry and configuration without bespoke shims.
The alternative keeps awkward dependencies and increases maintenance drag.

## Decision

- Rely on LangChain document compressors from `src/services/rag/retriever.py`, centralising configuration via `RAGConfig`.
- Update all imports (services, scripts, tests, MCP tooling) to reference the shared retriever implementation.
- Introduce targeted regression tests under `tests/unit/services/rag/` to cover the LangChain pipeline while retaining existing telemetry scenarios.
- Delete the now-empty `src/services/query_processing/rag/` package to enforce the new boundary.

## Consequences

- Query processing orchestrator and pipeline now depend on the shared RAG package exclusively, aligning module ownership with runtime responsibilities.
- CI/evaluation scripts no longer reference query processing internals, simplifying future standalone RAG tooling.
- Maintenance burden decreases: compression config changes surface in one location, and new capabilities (e.g., multilingual splitters) can be added without touching query processing internals.
- Short-term work includes adjusting imports, regenerating docs, and migrating tests; no runtime behaviour change is expected beyond module paths.

## Follow-Up Actions

1. Implement the relocation and ensure `__all__` exports match previous surfaces.
2. Update documentation (API refs, developer guides) to reference the new module path.
3. Confirm Sphinx builds succeed without references to the deleted package.
4. Extend telemetry wiring to emit compression stats via the unified RAG module.
