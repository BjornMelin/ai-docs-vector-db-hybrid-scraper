# Evaluation Harness Playbook

This playbook integrates the operational, observability, and domain guidance required to run and maintain the regression evaluation harness.

## 1. Golden Dataset and Corpus

- **Dataset location:** `tests/data/rag/golden_set.jsonl`.
- **Collection name:** `golden_eval` in Qdrant. Each dataset row maps to a deterministic document stored in the corpus.
- **Ground truth expectations:** `expected_answer` values feed RAGAS ground truths and `expected_contexts` capture oracle passages for retrieval metrics.

### 1.1 Seeding Qdrant

Recreate the deterministic corpus before running CI gating or local evaluations:

```bash
uv run python scripts/eval/seed_qdrant.py --recreate   --host localhost --port 6333   --collection golden_eval   --model BAAI/bge-small-en-v1.5   --corpus data/golden_corpus.json
```

The corpus file mirrors every entry in the golden dataset, enabling reproducible embedding generation.

### 1.2 Dataset Validation

- `scripts/eval/dataset_validator.py` enforces the schema for the golden dataset.
- The regression harness (`scripts/eval/rag_golden_eval.py`) accepts `--metrics-allowlist` and falls back to configuration defaults when omitted.

## 2. Evaluation Budgets and Cost Controls

Budget thresholds for the deterministic CI lane live in `config/eval_budgets.yml`:

- minimum `similarity_avg`
- minimum `precision_at_k` and `recall_at_k`
- maximum average latency (`max_latency_ms`)

Semantic evaluation is optional and governed by:

- default cap of 25 samples (`--ragas-max-samples` to override)
- cost guardrails in `config/eval_costs.yml`, including provider token budgets
- CLI warnings reminding operators to set rate limits whenever `--enable-ragas` is enabled

Run the semantic lane in CI only when secrets are provided; it publishes JSON artefacts but does not block merges by default.

## 3. Observability Surfaces

Evaluation telemetry focuses on a curated Prometheus allowlist aligned with OpenTelemetry GenAI guidance (OpenTelemetry, 2025).

### 3.1 Stage Latency Metrics

| Series                             | Instrument | Labels                                                  | Purpose                                                                        |
| ---------------------------------- | ---------- | ------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `*_rag_stage_latency_seconds`      | Histogram  | `collection`, `stage` (`retrieve`, `grade`, `generate`) | Detect stage-level latency regressions; monitor P95/P99 slices per collection. |
| `*_rag_generation_latency_seconds` | Histogram  | `collection`, `model`                                   | Track end-to-end generation latency including prompt assembly.                 |

Alert when `stage="generate"` exceeds 3s at P95 for two consecutive intervals (Selbie & Pakeman, 2024).

### 3.2 Answer Quality and Outcomes

| Series                        | Instrument | Labels                                        | Purpose                                                                      |
| ----------------------------- | ---------- | --------------------------------------------- | ---------------------------------------------------------------------------- |
| `*_rag_answers_total`         | Counter    | `collection`, `status` (`generated`, `empty`) | Monitor success rates and empty responses for SLO tracking.                  |
| `*_rag_errors_total`          | Counter    | `collection`, `stage`, `error_type`           | Classify pipeline exceptions for CI gating and incident review.              |
| `*_rag_generation_confidence` | Histogram  | `collection`                                  | Records generator confidence heuristics derived from graded document scores. |

Escalate when the rolling ratio of `status="empty"` exceeds 10% in any collection (Patronus AI, 2025).

### 3.3 Token and Compression Telemetry

| Series                          | Instrument | Labels                                                  | Purpose                                                                   |
| ------------------------------- | ---------- | ------------------------------------------------------- | ------------------------------------------------------------------------- |
| `*_rag_generation_tokens_total` | Counter    | `model`, `token_type` (`prompt`, `completion`, `total`) | Quantify token spend and enforce cost guardrails.                         |
| `*_compression_token_ratio`     | Histogram  | `collection`                                            | Measure retained tokens after contextual compression (values in `[0,1]`). |
| `*_compression_tokens_total`    | Counter    | `collection`, `kind` (`before`, `after`)                | Track absolute tokens before/after compression.                           |
| `*_compression_documents_total` | Counter    | `collection`, `status` (`compressed`, `unchanged`)      | Count documents affected by compression to prove ROI.                     |

### 3.4 Tracing

The LangGraph pipeline emits OpenTelemetry spans (`rag.pipeline`, `rag.retrieve`, `rag.grade`, `rag.generate`, `rag.llm`) annotated with `gen_ai.*` attributes. Configure OTLP exporters to forward traces to the observability stack.

### 3.5 Metrics Allowlist

Only metrics listed in `config/metrics_allowlist.json` are serialised by the harness. Update the allowlist when new instrumentation lands and keep the list concise to avoid noisy diffs.

### 3.6 Dashboard Checklist

1. Latency heatmap comparing stage latency histograms with vector search timings.
2. Answer funnel combining `_rag_answers_total` and `_rag_errors_total` to visualise success vs failure rates.
3. Cost guardrail combining `*_rag_generation_tokens_total` with confidence histograms to highlight expensive, low-confidence runs.
4. Compression ROI panel contrasting `before` vs `after` token counters alongside stage latency trends.

## 4. Query Behaviour Reference

- Grouping deduplicates results sharing the same `doc_id` unless the request overrides `group_by` with another payload key.
- The synonym dictionary maps `install` to both `setup` and `configure`, enabling retrievers to match alternate installation verbs.

## 5. Operational Checklist

1. Seed Qdrant and validate datasets.
2. Run the deterministic evaluation lane with budget thresholds.
3. Optionally enable semantic scoring with cost guardrails.
4. Review telemetry snapshots against the metrics allowlist.
5. Export traces and metrics to dashboards, updating thresholds as necessary.
6. Run regression tests (`pytest -q tests/unit/scripts/test_rag_golden_eval.py tests/integration`) to confirm threshold enforcement and failure-path coverage stay green.
