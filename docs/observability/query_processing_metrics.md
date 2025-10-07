# Query Processing Observability Metrics

The vector search and RAG pipelines emit the following Prometheus series after
Phase 5. All metrics share the namespace configured via `MetricsConfig.namespace`
(default `ml_app`). For detailed RAG instrumentation, consult
[Evaluation Harness Playbook](../testing/evaluation-harness.md).

## Grouped Search Metrics

| Metric                       | Type      | Labels                                                     | Description                                       |
| ---------------------------- | --------- | ---------------------------------------------------------- | ------------------------------------------------- |
| `*_grouping_requests_total`  | Counter   | `collection`, `status` (`applied`, `fallback`, `disabled`) | Count of grouped query attempts.                  |
| `*_grouping_latency_seconds` | Histogram | `collection`                                               | Latency distribution for grouped query responses. |

**Alert recommendations**

- Fire when the `status="fallback"` ratio exceeds 5% over 5 minutes.
- Track P95 latency per collection; page when above 0.25s.
- Escalate if `status="empty"` in the RAG funnel exceeds 10% for two
  consecutive observation windows.

## Compression Metrics

| Metric                          | Type      | Labels                                             | Description                                       |
| ------------------------------- | --------- | -------------------------------------------------- | ------------------------------------------------- |
| `*_compression_ratio`           | Histogram | `collection`                                       | Ratio of retained tokens (after / before).        |
| `*_compression_tokens_total`    | Counter   | `collection`, `kind` (`before`, `after`)           | Aggregated token counts for compressed documents. |
| `*_compression_documents_total` | Counter   | `collection`, `status` (`compressed`, `unchanged`) | Number of documents processed by the compressor.  |

**Alert recommendations**

- Fire when the rolling reduction (1 - ratio) dips below 0.3.
- Warn when recall proxy (CI gate) fails; see `scripts/ci/check_rag_compression.py`.

## RAG Outcome Metrics (Summary)

| Metric                          | Type      | Labels                                                  | Description                                           |
| ------------------------------- | --------- | ------------------------------------------------------- | ----------------------------------------------------- |
| `*_rag_stage_latency_seconds`   | Histogram | `collection`, `stage` (`retrieve`, `grade`, `generate`) | Per-stage LangGraph latency for triaging regressions. |
| `*_rag_answers_total`           | Counter   | `collection`, `status` (`generated`, `empty`)           | Tracks answer success versus empty fallbacks.         |
| `*_rag_errors_total`            | Counter   | `collection`, `stage`, `error_type`                     | Error taxonomy for pipeline stages.                   |
| `*_rag_generation_tokens_total` | Counter   | `model`, `token_type` (`prompt`, `completion`, `total`) | Token consumption used for cost governance.           |

## LangGraph Agent Metrics

| Metric                                 | Type      | Labels                                           | Description                                                  |
| -------------------------------------- | --------- | ------------------------------------------------ | ------------------------------------------------------------ |
| `*_agentic_graph_runs_total`           | Counter   | `mode`, `status` (`success`, `error`, `timeout`) | Number of LangGraph executions by entry mode.                |
| `*_agentic_graph_latency_ms`           | Histogram | `mode`                                           | End-to-end runtime per orchestration.                        |
| `*_agentic_retrieval_attempts_total`   | Counter   | `collection`, `outcome` (`success`, `fallback`)  | Retrieval attempts and fallbacks per collection.             |
| `*_agentic_retrieval_latency_ms`       | Histogram | `collection`                                     | Retrieval stage latency after tool selection.                |
| `*_agentic_tool_errors_total`          | Counter   | `tool`, `error_code`                             | Aggregated tool failures mapped from `ToolExecutionError`.   |
| `*_agentic_parallel_slots_in_use`      | Gauge     | `mode`                                           | Current parallel tool execution slots consumed.              |
| `*_agentic_checkpoint_persist_latency` | Histogram | `backend`                                        | Checkpoint flush latency when persistent storage is enabled. |

Operational tips:

- Alert when `status="timeout"` exceeds 3% of `*_agentic_graph_runs_total` over 15 minutes.
- Track the ratio of `outcome="fallback"` to proactive retries; increases usually point to stale tool metadata.
- When using persistent savers, keep `*_agentic_checkpoint_persist_latency` P95 under 150 ms to avoid back-pressure.

## Integration Notes

- Metrics are registered in `src/services/monitoring/metrics.py` and emitted by
  `VectorStoreService` (grouping), `VectorServiceRetriever` (compression), and
  `LangGraphRAGPipeline` (stage latency, answers, token metrics).
- Dashboards should group by `collection` to spot regressions in specific
  datasets.
- The CI gate script consumes fixtures in
  `tests/data/compression_gate_samples.json` and enforces KPI thresholds during
  pipeline runs.

## Browser Automation Metrics

| Metric                            | Type      | Labels                                | Description                                               |
| --------------------------------- | --------- | ------------------------------------- | --------------------------------------------------------- |
| `*_browser_requests_total`        | Counter   | `tier`, `status` (`success`, `error`) | Total browser automation attempts per tier and outcome.   |
| `*_browser_response_time_seconds` | Histogram | `tier`                                | Response time distribution for browser executions.        |
| `*_browser_challenges_total`      | Counter   | `tier`, `runtime`, `outcome`          | Bot-detection challenges observed (`detected`, `solved`). |
| `*_browser_tier_health_status`    | Gauge     | `tier`                                | Current health signal (1 healthy, 0 unhealthy) per tier.  |

**Operational guidance**

- Alert when `outcome="detected"` exceeds 10% of `*_browser_requests_total`
  for a tier over a 15-minute window.
- Track `runtime` to confirm undetected tiers are only invoked on hardened
  domains; sudden spikes may signal routing or proxy regressions.
- Combine `*_browser_challenges_total` with `*_browser_response_time_seconds` to
  monitor captcha solve latency.
