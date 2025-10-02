# ADR 0003: Deterministic Contextual Compression with Quality Gates

**Date:** 2025-02-20  
**Status:** Accepted  
**Drivers:** Rising RAG token costs, inconsistent LLM-dependent compression, lack of measurable recall safeguards, missing metrics for context trimming  
**Deciders:** AI Docs Platform Team

## Context

- Prior RAG flows passed full retrieval context to LangChain without trimming, driving token usage and latency beyond budget. Existing experimental compression relied on LLM calls, making results non-deterministic and untestable.
- No automated gate validated compression quality; regressions in recall or token savings could ship unnoticed.
- Observability lacked counters/histograms for compression effectiveness, hampering alerting and tuning.

## Decision

- Replace bespoke compression logic with LangChain's `EmbeddingsRedundantFilter` and `EmbeddingsFilter` inside a `DocumentCompressorPipeline`, driven by FastEmbed embeddings for deterministic behaviour.
- Integrate the LangChain pipeline into `VectorServiceRetriever` so RAG context is trimmed before generation, retaining the existing telemetry hooks and feature flags for controlled rollout.
- Instrument Prometheus metrics (`compression_ratio`, `compression_tokens_total`, `compression_documents_total`) and expose compression stats in RAG telemetry via the retriever's `compression_stats` output.
- Update CLI tooling (`scripts/eval/rag_compression_eval.py`) and the CI gate (`scripts/ci/check_rag_compression.py`) to exercise the LangChain pipeline against curated datasets, enforcing minimum token reduction and recall proxies.

## Consequences

- Compression output is deterministic and testable through the LangChain pipeline; unit tests cover filter integration, retriever wiring, and telemetry.
- RAG responses include `contextual_compression` feature flags and metrics, enabling product analytics and SLA monitoring.
- CI can block regressions when token reduction or recall fall below configurable thresholds. Operational dashboards can alert on compression anomalies.
- Pipeline configuration is expressed via the shared `RAGConfig`, ensuring future tuning uses LangChain-maintained primitives rather than bespoke code.

## Status Notes

- Future enhancements may add language-specific sentence splitters and richer evaluation datasets. Alert thresholds should be reviewed after production telemetry stabilizes.
- A follow-up task will integrate compression metrics into central observability dashboards and document recommended alert rules in the operations playbook.
- 2025-10-03 update: pipeline now prepends a `RecursiveCharacterTextSplitter` (tiktoken-aware when available) so compression operates on deterministic sub-chunks before embedding filters, preserving library-first alignment.
