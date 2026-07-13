# ADR 0006: Evaluation Harness and Dependency Governance

**Date:** 2025-10-02  
**Status:** Accepted  
**Drivers:** Need for behavioral parity validation during library migration, lack of automated regression checks, loosely specified dependency ranges  
**Deciders:** AI Docs Platform Team

## Context

To safely migrate toward the library-first architecture (ADR 0005) we must
guarantee behavioural parity and control upstream churn. The previous stack
lacked automated regression checks and had loosely specified dependency ranges.

## Decision

1. **Golden Evaluation Harness**
   - Maintain a lightweight golden dataset (`tests/data/rag/golden_set.jsonl`).
   - Provide a standard evaluation script (`scripts/eval/rag_golden_eval.py`)
     that executes the `SearchOrchestrator` with RAG enabled using
     `SearchRequest.from_input` for normalisation and emits similarity scores.
     The function remains framework neutral so it will continue to work after
     the LangChain migration.
   - Integrate richer metrics (RAGAS or LangChain evaluators) once the
     LangChain pipeline is in place.

2. **Dependency Governance**
   - Declare retrieval stack dependency ranges in `pyproject.toml` and resolve
     exact installations in `uv.lock`.
   - Introduce Renovate configuration (`.github/renovate.json`) to batch and
     schedule upgrades of the retrieval stack packages with appropriate review
     windows.
   - Require contract tests, golden harness results, and performance benchmarks
     in every dependency upgrade PR.

## Consequences

- Regression detection now has a canonical entry point that can run locally and
  in CI.
- Dependency changes are auditable and deliberate, minimising disruption from
  upstream API changes.
- Additional maintenance overhead: the lockfile and evaluation script require
  deliberate updates when supported dependency ranges change.

## References

- `tests/data/rag/golden_set.jsonl`
- `scripts/eval/rag_golden_eval.py`
- `pyproject.toml`
- `uv.lock`
- `.github/renovate.json`
