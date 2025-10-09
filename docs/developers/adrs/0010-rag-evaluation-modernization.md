# ADR 0010: RAG Golden Harness Modernization

**Date:** 2025-10-05  
**Status:** Accepted  
**Drivers:** Need richer regression signals for the LangGraph pipeline, desire to align evaluation outputs with new telemetry metrics, commitment to library-first tooling  
**Deciders:** AI Docs Platform Team

## Context

ADR 0006 introduced the original golden harness to secure the migration toward a
LangChain-oriented retrieval stack. The harness relied on a single
`difflib.SequenceMatcher` score which failed to identify regressions in
retrieval, grounding, or observability. Phase B3 requires confidence that the
LangGraph pipeline delivers the intended metrics catalogue
(`docs/testing/evaluation-harness.md`) and that the dataset structure can
express the final "final-only" implementation.

## Decision

1. **Hybrid Scoring Architecture**

   - Retain deterministic baselines (string similarity plus
     precision/recall/reciprocal-rank computed from `SearchRecord` metadata).
   - Layer optional RAGAS evaluators (context precision/recall, faithfulness,
     answer relevancy) behind a pluggable `RagasEvaluator` that only activates
     when API-backed LLM/embedding providers are configured. This honours the
     consensus score of 4.56/5 in the decision log while keeping the harness
     runnable in offline CI environments.

2. **Telemetry Integration via OpenTelemetry**

   - Capture stage latency, answer counters, and compression statistics through
     OpenTelemetry spans recorded during the evaluation run. The structured
     JSON report now includes these aggregates directly, eliminating the need
     for a dedicated Prometheus snapshot or custom registry wiring.

3. **Dataset Revision and JSON Reporting**
   - Extend `tests/data/rag/golden_set.jsonl` with `expected_contexts` and
     collection metadata to support retrieval hit analysis and context-aware
     metrics.
   - Update `scripts/eval/rag_golden_eval.py` to emit structured JSON (per-sample
     results, aggregates, telemetry) suitable for downstream CI comparisons.

## Consequences

- Evaluations now surface which layer regressed (retrieval, grounding,
  generation, or telemetry) rather than a single opaque similarity score.
- Teams can opt-in to semantic metrics without blocking local development;
  deterministic baselines remain available for quick checks.
- The harness remains library-first by delegating semantic scoring to RAGAS and
  relies on OpenTelemetry instrumentation rather than bespoke Prometheus
  plumbing, reducing maintenance overhead.

## References

- ADR 0006 – Evaluation Harness and Dependency Governance
- `scripts/eval/rag_golden_eval.py`
- `tests/data/rag/golden_set.jsonl`
- `tests/unit/scripts/test_rag_golden_eval.py`
- Decision log entry 2025-10-05 (RAG evaluation modernization, score 4.56/5)

### 2025-10-05 Update

We finalised the modernization by fixing the RAGAS ground-truth mapping, expanding the golden dataset to cover curated cases, seeding a reproducible `golden_eval`
Qdrant collection, and wiring cost-aware CLI flags. A metrics allowlist and deterministic budget artefact now keep CI outputs stable. Future contributors must validate dataset
changes with `scripts/eval/dataset_validator.py` and refresh the seed corpus when new cases are added.
