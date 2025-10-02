# ADR 0006: Evaluation Harness and Dependency Governance

**Date:** 2025-10-02  
**Status:** Accepted  
**Context:** To safely migrate toward the library-first architecture (ADR 0005)
we must guarantee behavioural parity and control upstream churn.  The previous
stack lacked automated regression checks and had loosely specified dependency
ranges.

## Decision

1. **Golden Evaluation Harness**
   - Maintain a lightweight golden dataset (`tests/data/rag/golden_set.jsonl`).
   - Provide a standard evaluation script (`scripts/eval/rag_golden_eval.py`)
     that executes the QueryProcessingPipeline with RAG enabled and emits
     similarity scores.  The function is intentionally framework neutral so it
     will continue to work after the LangChain migration.
   - Integrate richer metrics (RAGAS or LangChain evaluators) once the
     LangChain pipeline is in place.

2. **Dependency Governance**
   - Pin retrieval stack dependencies using narrow ranges in
     `pyproject.toml` and document them in
     `docs/developers/compatibility-matrix.md`.
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
- Additional maintenance overhead: the compatibility matrix must be kept up to
  date and the evaluation script requires periodic enhancement.

## Status

- Dataset, evaluation script, compatibility matrix, and Renovate rules have
  been added in Phase A.
- CI integration and metric enhancements are tracked in Phase B.

## References

- `tests/data/rag/golden_set.jsonl`
- `scripts/eval/rag_golden_eval.py`
- `docs/developers/compatibility-matrix.md`
- `.github/renovate.json`
