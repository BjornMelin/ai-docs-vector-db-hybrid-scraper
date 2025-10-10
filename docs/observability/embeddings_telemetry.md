# OpenAI Embedding Telemetry

This guide explains how to monitor synchronous and batch embedding jobs after
consolidating on OpenTelemetry and `prometheus-fastapi-instrumentator`.

## Instrumentation Overview

- `record_ai_operation` now annotates spans with the GenAI semantic
  conventions (`gen_ai.operation.name`, `gen_ai.request.model`,
  `gen_ai.usage.prompt_tokens`, `gen_ai.usage.total_tokens`, `gen_ai.cost.usd`).
- Metrics emitted by the global tracker:
  - `ai.operation.duration` – histogram (seconds) partitioned by
    `operation`/`model`/`provider`/`success`.
  - `ai.operation.tokens` – counter that increments with the exact token counts
    returned by OpenAI responses (fallback to `tiktoken` when usage metadata is
    missing).
  - `ai.operation.cost` – counter tracking USD cost based on the recorded token
    totals and the provider price table.
- Batch submissions surface as `operation="embedding_batch"` with additional
  span attributes:
  - `gen_ai.request.batch_size`
  - `gen_ai.request.custom_ids_provided`
  - `gen_ai.usage.prompt_tokens` (estimated from the batch payload when OpenAI
    does not return usage metadata)

## OTLP Collector Template

Configure the application with OTLP export variables:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://otel-collector:4317"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer ${OBSERVABILITY_API_TOKEN}"
```

A minimal collector configuration that forwards metrics to Prometheus and
traces to your backend:

```yaml
receivers:
  otlp:
    protocols:
      grpc:
      http:

processors:
  batch: {}

exporters:
  prometheus:
    endpoint: "0.0.0.0:9464"
  otlphttp:
    endpoint: "https://observability.example.com/v1/traces"
    headers:
      Authorization: "Bearer ${OBSERVABILITY_API_TOKEN}"

service:
  pipelines:
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus]
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlphttp]
```

> **Tip:** The default FastAPI process already exposes `/metrics`. The
> Prometheus exporter in the collector avoids double-scraping and keeps the OTLP
> exporter isolated from application restarts.

## Prometheus & Grafana Recipes

```promql
# Token burn per model over the last hour
sum by (model) (increase(ai_operation_tokens{operation="embedding"}[1h]))

# Cost trend (USD) per embedding mode
sum by (operation) (increase(ai_operation_cost[6h]))

# 95th percentile duration per operation
histogram_quantile(
  0.95,
  sum by (operation, le)(
    rate(ai_operation_duration_bucket{provider="openai"}[10m])
  )
)

# Batch health (success ratio)
sum(rate(ai_operation_tokens{operation="embedding_batch", success="True"}[5m]))
/ sum(rate(ai_operation_tokens{operation="embedding_batch"}[5m]))
```

Suggested Grafana panels:

1. **Token burn (stacked area)** grouped by `model`.
2. **Cost per operation** (bar) using `increase(ai_operation_cost[1h])`.
3. **Batch success gauge** using the success ratio above.
4. **Latency heatmap** sourced from `ai_operation_duration_bucket`.
5. **Error table** filtered on `success="False"` to highlight rate-limit or quota
   issues.

## Alerting

| Alert               | Expression                                                                                                  | Rationale                                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Batch failure spike | `sum(rate(ai_operation_tokens{operation="embedding_batch", success="False"}[10m])) > 0`                     | Batch pipelines should not silently fail; rely on token counter because the OTEL span still records even on failure. |
| Cost anomaly        | `increase(ai_operation_cost[1h]) > $COST_BUDGET_THRESHOLD`                                                  | Triggers when hourly spend crosses an agreed threshold.                                                              |
| Latency degradation | `histogram_quantile(0.95, sum(rate(ai_operation_duration_bucket{operation="embedding"}[5m])) by (le)) > 10` | Warns when embedding latency rises above 10s.                                                                        |

## Additional Resources

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [docs/operators/monitoring.md](../operators/monitoring.md) for platform-wide dashboards.
- [Elastic OpenAI monitoring guide](https://www.elastic.co/search-labs/blog/monitor-openai-api-gpt-models-opentelemetry-elastic) for an end-to-end example.
