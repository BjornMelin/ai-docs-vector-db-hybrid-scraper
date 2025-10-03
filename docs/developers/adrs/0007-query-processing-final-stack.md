# ADR 0007: Final Query Processing Stack Consolidation

**Date:** 2025-10-03  
**Status:** Accepted  
**Drivers:** Legacy modules duplicating functionality now available in LangChain primitives, maintenance overhead, risk of split DTO evolution  
**Deciders:** AI Docs Platform Team

## Context

Replace bespoke query-processing services with LangChain-backed primitives.
Legacy modules (`clustering.py`, `federated.py`, and the `utils/` package)
implemented MapReduce-style helpers, score clustering, and hand-rolled caching
that pre-date the new `SearchOrchestrator`. After migrating to LangChain's
`QdrantVectorStore` and embedding wrappers, those modules duplicate
functionality already delivered by `VectorStoreService` and the orchestrator's
in-module query expansion and personalisation helpers. Keeping them would force
continued maintenance, risk split DTO evolution, and violate the final-only
policy.

## Decision Framework

Original evaluation used four weighted criteria to decide on full removal vs. partial retention. Scores (1–5) reflect leverage, platform value, maintenance reduction, and adaptability.

| Criteria           | Weight   | Remove legacy helpers | Retain behind flags |
| ------------------ | -------- | --------------------- | ------------------- |
| Leverage           | 0.25     | 4                     | 2                   |
| Value              | 0.25     | 4                     | 3                   |
| Maintenance        | 0.25     | 5                     | 2                   |
| Adaptability       | 0.25     | 4                     | 3                   |
| **Weighted Total** | **1.00** | **4.25**              | **2.50**            |

Decision: Removal path selected based on higher composite score (4.25 vs 2.50) and elimination of duplicated code paths.

## Decision

Delete the remaining query-processing helper modules and update downstream code
(MCP tools, database manager, enterprise flows, and tests) to rely solely on the
unified orchestrator and vector service. Alternatives retained lower leverage
because they preserved duplicated pipelines or required parallel support for
legacy DTOs.

Key changes:

- Remove `src/services/query_processing/clustering.py` and the `utils/` package.
- Remove `src/services/query_processing/pipeline.py`, directing callers to
  instantiate `SearchOrchestrator` and normalise inputs via
  `SearchRequest.from_input`.
- Drop the associated unit suites now superseded by orchestrator coverage.
- Ensure MCP tools, ingestion scripts, and the database manager call the
  LangChain-powered vector service directly without legacy helpers.
- Update documentation (CHANGELOG, this ADR) to record the final-only stack.

## Consequences

- Less bespoke code to maintain; QueryProcessing becomes a thin wrapper over the
  orchestrator and LangChain services.
- Tests now exercise only the supported path, avoiding accidental reintroduction
  of deprecated utilities.
- Any future enhancements route through LangChain primitives or orchestrator
  helpers, keeping the library-first principle enforced.

## Alternatives Considered

1. **Retain clustering/utils behind feature flags (Score 2.6/5).**

   - Pros: minimal immediate code churn.
   - Cons: Perpetuates dual paths, violates final-only policy, and keeps sklearn
     dependency surface just for clustering.

2. **Rewrite utilities on top of LangChain callbacks (Score 3.1/5).**
   - Pros: Potentially richer analytics.
   - Cons: Adds complexity without clear application value; overlaps with planned
     LangGraph telemetry.
