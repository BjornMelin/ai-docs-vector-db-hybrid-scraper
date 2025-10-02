# Query Processing Pipeline – Continuation Report (2025-10-01)

## 1. Session Overview
- **Focus:** Resume phases 4–6 of query processing modernization (post Phase 3 completion). Consolidate DTOs/orchestrator, finalize deterministic compression and telemetry, update docs/ADRs, and enforce quality gates.
- **Constraints:** Library-first mandate, delete superseded code immediately, maintain Google-style docstrings, strict quality gates (`ruff`, `pylint`, `pyright`, `pytest`), deterministic tests (`pytest-randomly` seed recorded), acceptance checklist ≥9.5 decision score.
- **Outstanding Risks:** Schema drift between services and MCP tooling, loss of multi-collection capability without documentation, metrics cardinality blow-up, compression impacting recall, missing ADR updates.

## 2. Research & Tool Usage Log
| # | Tool Call | Purpose | Key Findings / Next Actions |
|---|-----------|---------|------------------------------|
| 1 | `exa.get_code_context_exa {"query":"LangChain Qdrant retriever integration Python 2025"}` | Mandatory code-context search | Gathered LangChain `QdrantVectorStore` usage patterns, hybrid search examples, retrieval-as-retriever integration. Reinforces library-first adoption. |
| 2 | `context7.resolve-library-id` → `/websites/python_langchain`; `context7.get-library-docs` topic `vectorstores qdrant` | Official LangChain docs | Verified usage of named vectors, hybrid retrieval, `FastEmbedSparse`, retriever transformation. |
| 3 | `context7.resolve-library-id` → `/qdrant/qdrant`; `context7.get-library-docs` topic `query point groups` | Qdrant QueryPointGroups API | Confirmed grouped search schema, protobuf definitions, need for capability detection. |
| 4 | `firecrawl.firecrawl_deep_research` (query: “LangChain and Qdrant integration capabilities…”, `maxDepth`:3, `timeLimit`:300, `maxUrls`:30) | Deep research attempt | **Failed (HTTP 502)**—documented failure; fallback to `firecrawl_search` + `firecrawl_scrape`. |
| 5 | `firecrawl_search` + `firecrawl_scrape` | Current hybrid/compression articles, Qdrant docs | Retrieved LangChain Qdrant integration docs, hybrid search articles, contextual compression how-to. |
| 6 | `zen.analyze` (2-step session) | Map architecture, identify duplication | Highlighted legacy DTOs, orchestrator dependencies, missing normalization metadata, telemetry gaps. |
| 7 | `zen.thinkdeep` (2-step session) | Explore approaches, confirm phased strategy | Validated phased plan (A: foundation; B: compression+telemetry; C: docs/cleanup) and risks. |
| 8 | `zen.planner` (6-step session, continuation `972547c9-b237-4c25-aed0-f78f34fb5d8d`) | Produce executable roadmap | Final plan captured below; use continuation for future adjustments. |

> **Fallback Note:** `firecrawl_deep_research` unavailable (repeated 502). If deep research is required later, consider reattempting or switching to Exa deep researcher.

## 3. Decision Framework Summary
For the phased execution approach (Foundation → Compression+Telemetry → Docs/Cleanup):

| Criterion | Weight | Score (0–5) | Weighted |
|-----------|--------|-------------|----------|
| Solution Leverage | 0.35 | 4.8 | 1.68 |
| Application Value | 0.30 | 4.6 | 1.38 |
| Maintenance Load | 0.25 | 4.7 | 1.18 |
| Adaptability | 0.10 | 4.5 | 0.45 |
| **Total** | | | **4.69 / 5 → 9.38 / 10** |

Rationale: sequential phases minimize risk, keep codebase library-first, and align deliverables with acceptance checklist. Alternative (big-bang) rejected due to high maintenance burden.

## 4. Structured Plan (see also `update_plan` state)
```
Phase A – DTO & Orchestrator Foundation (Status: TODO)
  A1. Define canonical SearchRequest/SearchRecord DTOs (raw_score + normalized_score + grouping metadata).
  A2. Refactor VectorStoreService to emit both raw and normalized scores; add grouped search fallback if QueryPointGroups unsupported.
  A3. Simplify SearchOrchestrator/QueryProcessingPipeline around VectorStoreService, removing federated artifacts. Ensure compatibility with MCP tooling.
  A4. Update / add unit + integration tests (grouping applied & fallback). Confirm acceptance checklist items (no legacy fields).

Phase B – Compression & Telemetry (Status: TODO)
  B1. Wire LangChain contextual compression (EmbeddingsRedundantFilter + EmbeddingsFilter) into the retriever path; enforce deterministic similarity filtering and metadata tagging.
  B2. Instrument Prometheus counters/histograms (grouping status, compression applied/fallback, latency, reduction ratio) with bounded labels.
  B3. Implement CI gate (`scripts/ci/check_rag_compression.py`) + regression harness (token reduction vs recall). Add tests asserting /metrics contents.
  B4. Document graceful fallbacks (metric absence, embedding errors) and ensure config toggles default on/off states.

Phase C – Documentation, ADRs, Quality Gates (Status: TODO)
  C1. Update docs (`docs/users/search-and-retrieval.md`, `docs/users/examples-and-recipes.md`, API explorer) to reflect grouped search, normalization, compression, and deprecation of federated features.
  C2. Author ADRs for (i) federated deprecation, (ii) deterministic compression design & evaluation. Update Changelog/decision tables.
  C3. Remove legacy tests/config flags; run full quality gates (`ruff format/check`, `pylint`, `pyright`, `pytest -q` with seed capture). Record tool versions.
  C4. Final acceptance review: checklist completion, decision-framework scores logged, SemVer notes prepared if release-bound.
```

## 5. Immediate Execution Notes
- `update_plan` reflects Phases A–C as TODO (only one step will be marked `in_progress` during execution).
- Ensure capability detection is implemented before deleting client-side dedup logic.
- When modifying schema, maintain backward-compatible field names; add new fields instead of renaming until docs updated.
- Prepare regression dataset for compression evaluation (baseline vs compressed context recall).

## 6. Tool Invocation Instructions
Re-run the following commands in subsequent sessions to reload critical context:
- `exa.get_code_context_exa {"query":"LangChain Qdrant retriever integration Python 2025"}`
- `context7.resolve-library-id` + `context7.get-library-docs` for `/websites/python_langchain` (topic `vectorstores qdrant`).
- `context7.resolve-library-id` + `context7.get-library-docs` for `/qdrant/qdrant` (topic `query point groups`).
- `firecrawl_search {"query":"LangChain Qdrant QueryPointGroups contextual compression hybrid search best practices","limit":6,...}` (supplemental articles).
- `firecrawl_scrape {"url":"https://qdrant.tech/documentation/concepts/search/#grouping-api",...}` for precise API syntax.
- `zen.analyze`, `zen.thinkdeep`, `zen.planner`, `zen.codereview`, `zen.precommit` per workflow recipe in plan.

## 7. Research References & Citations
1. LangChain Qdrant integration: https://python.langchain.com/docs/integrations/vectorstores/qdrant/
2. Qdrant Query API & grouping: https://qdrant.tech/documentation/concepts/search/#grouping-api
3. LangChain contextual compression how-to: https://python.langchain.com/docs/how_to/contextual_compression/
4. Qdrant hybrid search article: https://qdrant.tech/articles/hybrid-search/
5. Hybrid RAG tutorial (LangGraph + Qdrant miniCOIL): https://datacouch.io/blog/hybrid-rag-with-langgraph-qdrant-advanced-tutorial/

## 8. Quality & Acceptance Reminders
- All quality gates defined in `pyproject.toml` are mandatory before concluding phases.
- Deterministic tests must record the `pytest-randomly` seed.
- Use canonical DTOs for MCP tooling to avoid schema drift; ensure `src/mcp_tools/models/query_processing.py` mirrors final contract.
- Document decision-framework scores alongside ADR updates.

## 9. Next Steps for Future Sessions
1. Start Phase A task A1 (mark plan step `in_progress`).
2. Run targeted tests to identify legacy field usage prior to schema change.
3. Prepare compression regression dataset + baseline metrics (needed in Phase B).
4. Schedule doc/ADR update placeholders to ensure no drift (Phase C).

*Use planner continuation `972547c9-b237-4c25-aed0-f78f34fb5d8d` for future roadmap adjustments.*
