# ADR 0005: Library-First Retrieval and RAG Architecture

**Date:** 2025-10-02  
**Status:** Accepted  
**Context:** The legacy query processing stack retained numerous bespoke
components (federated routing, clustering, ranking, synonym expansion) that
replicate capabilities now available in mature libraries (LangChain,
FastEmbed, Qdrant). Maintenance burden and feature delivery velocity have both
suffered as a result.

## Decision

- Adopt a library-first posture across embeddings, vector storage, retrieval
  orchestration, and RAG generation.  The core implementation will use:
  - `fastembed` for embedding generation (CPU optimised)
  - `langchain_qdrant.QdrantVectorStore` for vector persistence and search
  - LangChain retrievers/rerankers for query expansion, grouping, contextual
    compression, and personalisation
  - LangChain Runnable graphs (or LangGraph) for RAG orchestration and agent
    workflows
- Retain only thin adapter layers that enforce the canonical
  `SearchRecord`/`SearchResponse` contract and funnel telemetry through shared
  interfaces.
- Remove legacy modules that duplicate library behaviour (clustering,
  expansion, ranking, federated) once parity validation is complete.

## Consequences

- Significant reduction in bespoke code, freeing teams to focus on product
  features and evaluation pipelines.
- Dependency management becomes critical.  Versions are pinned in
  `pyproject.toml` and documented in the compatibility matrix.  Renovate
  handles grouped upgrade proposals.
- Performance regressions must be monitored via regression harnesses (see ADR
  0006) and canary deployments.  Hot paths may still call the Qdrant SDK
  directly if required.

## Status

- Phase A extension tasks capture the initial groundwork (tests, docs,
  evaluation harness, pinned versions).
- Phase B will migrate embeddings/vector stores and replace the RAG
  orchestrator.

## References

- Phase roadmap in `agent-logs/2025-10-02/refactor-strategy.md`
- Compatibility matrix in `docs/developers/compatibility-matrix.md`
- Consensus summary (Option A vs Option B) stored in plan log
