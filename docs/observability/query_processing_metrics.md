# Query Processing Observability Metrics

The vector search and RAG pipelines emit the following Prometheus series after
Phase 5. All metrics share the namespace configured via `MetricsConfig.namespace`
(default `ml_app`). For detailed RAG instrumentation, consult
[`rag_metrics_catalog.md`](rag_metrics_catalog.md).

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

## Integration Notes

- Metrics are registered in `src/services/monitoring/metrics.py` and emitted by
  `VectorStoreService` (grouping), `VectorServiceRetriever` (compression), and
  `LangGraphRAGPipeline` (stage latency, answers, token metrics).
- Dashboards should group by `collection` to spot regressions in specific
  datasets.
- The CI gate script consumes fixtures in
  `tests/data/compression_gate_samples.json` and enforces KPI thresholds during
  pipeline runs.
