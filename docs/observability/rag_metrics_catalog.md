# RAG Telemetry Catalog

This catalog enumerates the metrics emitted by the LangGraph-based retrieval augmented generation (RAG) pipeline. All series share the namespace configured via `MetricsConfig.namespace` (defaults to `ml_app`).
Instruments follow OpenTelemetry semantic conventions for GenAI workloads where applicable ([OpenTelemetry AI Agent guidance, 2025](https://opentelemetry.io/blog/2025/ai-agent-observability/)).

## Stage Latency Metrics

| Metric suffix                     | Instrument | Labels                                                  | Description                                                                                                                                          |
| --------------------------------- | ---------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_rag_stage_latency_seconds`      | Histogram  | `collection`, `stage` (`retrieve`, `grade`, `generate`) | Wall-clock latency for each graph stage. Buckets target sub-second resolution to detect regressions before user-facing latency budgets are breached. |
| `_rag_generation_latency_seconds` | Histogram  | `collection`, `model`                                   | End-to-end generation latency from prompt assembly to response emission.                                                                             |

**Operational guidance**

- Alert when the 95th percentile of `stage="generate"` exceeds 3s for two consecutive intervals (aligned with Google Cloud RAG evaluation guardrails [[Selbie & Pakeman, 2024](https://cloud.google.com/blog/products/ai-machine-learning/optimizing-rag-retrieval)]).
- Track `stage="retrieve"` alongside vector search histograms to spot regressions introduced by contextual compression.

## Answer Quality & Outcomes

| Metric suffix                | Instrument | Labels                                        | Description                                                                                                                    |
| ---------------------------- | ---------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `_rag_answers_total`         | Counter    | `collection`, `status` (`generated`, `empty`) | Counts successful generations versus empty fallbacks. Use for SLO error budgets.                                               |
| `_rag_errors_total`          | Counter    | `collection`, `stage`, `error_type`           | Classifies pipeline exceptions by stage and surfaced error type.                                                               |
| `_rag_generation_confidence` | Histogram  | `collection`                                  | Confidence heuristic emitted by the generator; derived from graded document scores when the generator omits an explicit value. |

**Alert recommendations**

- Fire a warning if the rolling ratio of `status="empty"` exceeds 10% in any collection.
- Record `error_type` samples in incident postmortems; align categories with evaluation tooling (e.g., DeepEval hallucination/faithfulness metrics [[Patronus AI, 2025](https://www.patronus.ai/llm-testing/rag-evaluation-metrics)]).

## Token & Compression Telemetry

| Metric suffix                  | Instrument | Labels                                                  | Description                                                                                                   |
| ------------------------------ | ---------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `_rag_generation_tokens_total` | Counter    | `model`, `token_type` (`prompt`, `completion`, `total`) | Aggregates token consumption reported by LangChain callbacks. Enables cost tracking and guardrail automation. |
| `_compression_token_ratio`     | Histogram  | `collection`                                            | Ratio of tokens retained after contextual compression (values in `[0,1]`).                                    |
| `_compression_tokens_total`    | Counter    | `collection`, `kind` (`before`, `after`)                | Raw token counts before and after compression.                                                                |
| `_compression_documents_total` | Counter    | `collection`, `status` (`compressed`, `unchanged`)      | Number of documents processed by the compressor, partitioned by whether content was trimmed.                  |

**Usage notes**

- Combine `token_type="prompt"` with `stage_latency` dashboards to prove compression ROI (cf. LangGraph instrumentation patterns [[Last9, 2025](https://last9.io/blog/langchain-and-langgraph-instrumentation-guide/)]).
- Feed these metrics into the golden evaluation harness to assert compression recall thresholds before deployment (see `tests/unit/services/rag/test_langgraph_pipeline.py`).

## Tracing Integration

- The pipeline registers `RagTracingCallback`, which emits OpenTelemetry spans named `rag.pipeline`, `rag.retrieve`, `rag.grade`, `rag.generate`, and `rag.llm`. Spans carry `gen_ai.*` attributes compliant with LangSmith/GenAI semantic conventions [[LangChain LangSmith docs, 2025](https://docs.smith.langchain.com/observability/how_to_guides/trace_with_opentelemetry)].
- To export traces, configure the OpenTelemetry SDK or Vertex AI evaluation reporters; spans fan out to any OTLP-compatible collector.

## Dashboard Checklist

1. **Latency heatmap** – plot `_rag_stage_latency_seconds` (P50/P95) per collection and compare with `_vector_search_duration_seconds`.
2. **Answer funnel** – stacked area of `_rag_answers_total` by status with overlay of `_rag_errors_total` to pinpoint failing stages.
3. **Cost guardrail** – combine `_rag_generation_tokens_total` with confidence histogram to quantify “expensive but low-confidence” responses.
4. **Compression ROI** – visualise `before` vs. `after` token counters alongside stage latency to confirm net savings.

Link dashboards back to the metrics catalog for future audits; update this document whenever new RAG instrumentation surfaces are added.
