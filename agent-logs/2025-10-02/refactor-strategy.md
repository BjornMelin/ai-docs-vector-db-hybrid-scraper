# Refactor Strategy – Retrieval & RAG Stack (2025-10-02)

## Summary
- Legacy query_processing services (clustering, expansion, ranking, utils, federated) duplicate LangChain features and add maintenance burden.
- Vector store layer wraps Qdrant client with custom adapters; grouping/normalisation logic is the only differentiator.
- RAG pipeline (`RAGGenerator`) reimplements LangChain RetrievalQA orchestration; telemetry custom.
- MCP tooling still performs manual SearchRecord conversions despite canonical DTO.
- Docs/tests reference removed modules; parity coverage missing for grouping fallback, normalisation and RAG outputs.
- Observability relies on bespoke Prometheus counters; opportunity to adopt LangChain callbacks / OpenTelemetry.

## Key Options (Decision Framework)
| Option | Leverage (35%) | Value (30%) | Maintenance (25%) | Adaptability (10%) | Weighted |
| --- | --- | --- | --- | --- | --- |
| A. Aggressive library migration | 4.6 | 4.1 | 4.0 | 4.7 | **4.31 / 5 (86.2/100)** |
| B. Incremental cleanup | 3.6 | 3.8 | 3.3 | 3.7 | 3.58 / 5 (71.9/100) |
| B (model contra) | 7 | 8 | 8 | 9 | **7.75 / 10 (reweighted)** |

- Consensus favours **Option A** (library-first) with phased rollout and strong interface boundaries. Option B advocates incremental pilots before full migration. Adopt hybrid: structured pilot + hard sunset of bespoke modules.

## Proposed Migration Blueprint
1. **Phase 0 – Foundations**
   - Introduce internal interfaces (EmbeddingService, VectorStoreService, RetrievalPipeline, RagOrchestrator).
   - Establish contract tests + golden RAG eval dataset (RAGAS / LangChain Eval).
   - Pin dependency versions; add Renovate or scheduled upgrade window.

2. **Phase 1 – Embedding Modernisation**
   - Swap custom embedding providers for FastEmbed via LangChain embeddings API.
   - Validate cost/perf, seed caches, update observability.

3. **Phase 2 – Vector Layer Migration**
   - Dual-write VectorStoreService to LangChain `QdrantVectorStore` (with grouping metadata helper).
   - Backfill/consistency jobs; canary read path; remove adapter_base once parity confirmed.

4. **Phase 3 – Retrieval & Ranking**
   - Replace clustering/ranking/expansion modules with LangChain retrievers + rerankers (e.g., ContextualCompressionRetriever, CohereRerank, LLMChainExtractor).
   - Flatten `query_processing` to orchestrator + LangChain wrappers.

5. **Phase 4 – RAG Orchestration**
   - Adopt LangChain Runnable pipeline or LangGraph for deterministic RAG flow.
   - Integrate LangChain callbacks / LangSmith for tracing.

6. **Phase 5 – Tooling & Surface Alignment**
   - Update MCP/API to reuse orchestrator outputs directly; delete residual helpers & docs.
   - Refresh docs/API references; remove deprecated sections.

7. **Phase 6 – Observability & Quality Gates**
   - Standardise metrics via OpenTelemetry/LangChain integration; deprecate bespoke counters.
   - Build parity/shadow tests (grouping fallback, normalisation, RAG) and integrate into CI gates.

## Identified Duplicates & Deletions
- `src/services/query_processing/clustering.py`, `expansion.py`, `ranking.py`, `utils/` – replace with LangChain components or retire.
- `src/services/vector_db/adapter.py`, `adapter_base.py`, large portions of `service.py` – superseded by LangChain `QdrantVectorStore`.
- `src/services/rag/generator.py` – replace with LangChain RetrievalQA or LangGraph.
- `src/mcp_tools/tools` helpers (already partially removed) – finish removal, align with canonical SearchRecord.
- Docs: `docs/api/*query_processing*`, `docs/query_processing_response_contract.md` – update.
- Tests: large set of deleted suites; introduce new parity tests in unit/integration directories.

## Risks & Mitigations
- **Framework churn / lock-in** – wrap libraries behind thin adapters; pin versions; add contract tests.
- **Performance regression** – maintain direct Qdrant SDK path for hot queries until benchmarking complete; use canary + shadow evaluation.
- **Feature parity gaps** – run targeted comparisons for grouping, normalisation, personalization; ensure library features support required metadata.
- **Team ramp-up** – schedule training on LangChain/LangGraph best practices; document adapter patterns.

## Immediate Next Actions
- Approve hybrid migration strategy (Option A with phased rollout).
- Kick off Phase 0 tasks (interfaces, contract tests, eval dataset, dependency pinning).
- Prepare detailed implementation plan (Zen planner) aligned with roadmap phases.
