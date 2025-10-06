# Performance Test Suite

This suite provides lightweight, deterministic performance smoke tests that
exercise the LangGraph-based retrieval stack. The focus is to detect latency or
throughput regressions introduced by code changes without requiring external
infrastructure or long-running load jobs.

## Included Scenarios

- **LangGraph pipeline latency** (`test_rag_pipeline_performance.py`)
  - Runs the LangGraph RAG pipeline repeatedly with stubbed dependencies and
    asserts that the average wall-clock latency stays below the micro-benchmark
    budget.
- **Search orchestrator concurrency** (`test_search_orchestrator_performance.py`)
  - Executes the `SearchOrchestrator` against a stub pipeline under high
    concurrency and verifies both response correctness and throughput.
- **Metrics recording throughput** (`test_metrics_registry_performance.py`)
  - Performs hundreds of metric updates against the `MetricsRegistry` and checks
    that operations remain fast while the resulting Prometheus samples reflect
    the expected counts.

Each test is labeled with `pytest.mark.performance` so the suite can be run
on-demand:

```bash
uv run pytest -m performance tests/performance -q
```

These tests act as guardrails for the final implementation—they do not replace
full-scale load testing, but they provide quick feedback that the LangGraph RAG
stack and observability plumbing remain efficient.
