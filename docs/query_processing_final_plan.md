# Query Processing Pipeline Finalization Plan

**Zen Planner Continuation ID:** `72a61b20-0050-4b24-81d1-e1a6cb901e31`

## Overview

Goal: deliver a polished, library-first retrieval pipeline using Qdrant server-side grouping and embeddings-based contextual compression, eliminating bespoke dedup logic while ensuring schema integrity, observability, and comprehensive tests.

Key Stakeholders: Retrieval/infra engineers, ingestion team, QA, DevOps (for Qdrant upgrades), product analytics.

Success Criteria:

- All search results unique per document with predictable grouping semantics (default `doc_id`).
- RAG contexts compressed deterministically without LLM dependencies, reducing token usage while preserving recall.
- Legacy dedup code paths removed or gated; fallbacks tested.
- Telemetry dashboards and alerting cover grouping, compression, schema health, and recall metrics.
- Test suites updated; no references to legacy `_total_*` alias fields remain.

## 2) NON-NEGOTIABLES

- **FINAL-ONLY:** Remove all legacy/back-compat/deprecated code and tests. Keep only final implementations.
- **DELETE SUPERSCEDED CODE IMMEDIATELY** as new code lands.
- **LIBRARY-FIRST:** Use maintained libraries when they cover ≥80% at ≤30% custom work.
- **KISS/DRY/YAGNI:** Simplicity, minimal indirection, no over-engineering.
- **WRAPPERS:** Avoid redundant wrappers. Add only with proven value; include a one-line rationale in code.
- **LOGGING:** Minimal and local. Useful for debugging only. No ops bloat.
- **SEMVER:** If publishing, treat breaking changes as **MAJOR** and record in release notes.

---

## 3) DECISION FRAMEWORK

Score 0–5 and show weighted total with a 2–4 line rationale. Tie-breakers: lower maintenance → higher leverage.

- **Leverage 35%** (library-first, reuse)
- **Value 30%** (feature completeness, user impact)
- **Maintenance 25%** (simplicity, cognitive load)
- **Adaptability 10%** (modularity when justified)

Weighted total ≥ 9.5 required; document rationale for each major decision.

---

## 4) EXECUTION PRINCIPLES

- Replace bespoke utilities with library features where equivalent.
- Keep APIs cohesive and small; collapse unnecessary indirections.
- Make exception paths explicit and testable.
- Document deviations with a short decision entry.
- Prefer pure functions and immutability when sensible.

---

## 5) TOOL USE GUIDELINES

- Use the fewest tools necessary; never fabricate outputs; wait for tool completion.
- Prefer official documentation; escalate from quick checks → broader crawl → deep research.
- When results conflict, trust current primary sources.

### Workflow recipes

**A) Quick library docs + examples**

1. `exa.get_code_context_exa` for every programming/code query (mandatory when “exa-code” or code lookup requested).
2. `exa.web_search_exa` for supplemental comparisons.
3. If additional detail needed: `firecrawl.scrape` or `firecrawl.extract` on official docs/tutorials.

**B) Deep research**

- Default: `firecrawl.firecrawl_deep_research` for multi-site synthesis.
- Exceptionally hard tasks: use Exa Research (`deep_researcher_start`/`check`) and wait for completion.

**C) Code refactor + assurance**

1. `zen.thinkdeep` for options/library fit.
2. `zen.analyze` for mapping specs/configs.
3. Implement and delete superseded code immediately.
4. `zen.codereview` on new code.
5. `zen.challenge` for adversarial checks.
6. `zen.consensus` for disputed choices.
7. `zen.planner` for rollout and follow-ups.

**D) MCP hygiene**

- If MCP tools used, record MCP spec/date and tool versions in outputs.

---

## 6) QUALITY GATES (single source of truth: `pyproject.toml`)

All must pass locally and in CI; fail fast on breach.

- Format/Lint: `ruff format && ruff check --fix` clean.
- Pylint: 0 errors, 0 warnings (or documented minimal disables).
- Type check: `mypy --strict` **or** `pyright` zero errors.
- Tests: `pytest -q` green; control randomness/time via fixtures.
- Determinism: enable `pytest-randomly`; record seed; freeze/patch clocks as needed.
- Warnings: treat unexpected warnings as errors (`filterwarnings`).
- Repo search gate: fail if `deprecated|legacy|compat|shim|TODO` remain after refactor.
- Config unification: keep pytest/ruff/pylint/type-checker settings centralized in `pyproject.toml`.

---

## 7) ACCEPTANCE CHECKLIST

- [ ] Only final implementations remain; legacy code/tests deleted.
- [ ] Phase feature checklist satisfied, tests cover each item.
- [ ] Google-style docstrings on all public APIs (generators use **Yields:**).
- [ ] Ruff, Pylint, and type checker clean.
- [ ] Deterministic tests pass; seed recorded; warnings policy enforced.
- [ ] Decisions recorded with scores/rationales.
- [ ] SemVer notes updated when publishing major changes.

---

## 8) DELIVERABLES

- Unified diff/patch (including deletions).
- Feature checklist ↔ tests coverage map.
- Decision table with scores and rationales.
- Test, lint, type-check reports.
- Tool versions and MCP spec/date (if applicable).

---

## Decision Framework Matrix (Scores 1-10)

| Criterion         | Weight | Score | Weighted  |
| ----------------- | ------ | ----- | --------- |
| Library Leverage  | 0.35   | 9.5   | 3.325     |
| Application Value | 0.30   | 9.6   | 2.880     |
| Maintenance Load  | 0.25   | 9.6   | 2.400     |
| Adaptability      | 0.10   | 9.5   | 0.950     |
| **Total**         | 1.00   |       | **9.555** |

Provide a 2–4 line rationale for each major decision entry (see Section 5 template).

## Phase Roadmap (Dependency Flow)

```
Phase1 --> Phase2 --> Phase3 --> Phase4
               |            \
               v             v
            Phase5 ------> Phase6
```

1. **Phase 1:** Schema Audit & Migration Infrastructure
2. **Phase 2:** Qdrant QueryPointGroups Integration & Capability Detection
3. **Phase 3:** Federated/Orchestrator Dedup Simplification & Over-Fetch Controls
4. **Phase 4:** Embeddings-Based Contextual Compression for RAG
5. **Phase 5:** Telemetry, Metrics, and Quality Gates
6. **Phase 6:** Documentation, Tests, and Decision Validation

---

## Phase 1 – Schema Audit & Migration Infrastructure

**Objectives**

- Ensure every point ingested into Qdrant contains canonical payload fields: `doc_id` (string keyword), `chunk_id` (int), `content_hash` (string), `source`, `tenant`, `created_at`.
- Enforce schema via startup validation and migration scripts; create payload indexes for `doc_id`, `tenant`, and frequently filtered fields.

**Primary Tasks**

1. **Payload Audit Script** (`scripts/vector/payload_audit.py`)
   - Libraries: `qdrant-client` (`qdrant_client.AsyncQdrantClient`), `pandas` (optional for reporting).
   - Actions: scan representative collections, report missing/invalid fields, sample duplicates by `content_hash`.
2. **Backfill & Mutation Utility**
   - Libraries: `qdrant-client`, `hashlib`, `uuid`.
   - Actions: populate `doc_id` (from existing metadata or synthetic UUID), compute normalized `content_hash` (`hashlib.blake2b`), upsert updates in batches.
3. **Ingestion Path Updates**
   - Files: `src/services/vector_db/service.py` (document ingestion helpers), any ETL scripts.
   - Ensure upstream ingestion always sets required fields before calling adapter.
4. **Schema Validator**
   - Add module `src/services/vector_db/schema_validator.py` with `validate_payload_schema(collection)` and `ensure_payload_indexes(collection)`.
   - Invoke during service startup (`VectorStoreService.initialize`).
5. **Migration Checks in CI**
   - Add pytest `tests/integration/vector_db/test_schema_validation.py` mocking Qdrant to verify validation logic.

**Success Metrics**

- Audit reports 100% coverage of mandatory fields post-backfill.
- Payload indexes confirmed via `client.get_collection()` metadata.
- Validation failures break CI with actionable reports.

**Risks & Mitigations**

- Legacy data lacking metadata → provide fallback mapping or quarantine set.
- Index creation cost → run during maintenance window; batch operations; monitor progress.

**References**

- Qdrant payload schema/index docs: https://qdrant.tech/documentation/concepts/payload/
- `qdrant-client` examples: https://qdrant.tech/documentation/interfaces/python/#payload-indexing

---

## Phase 2 – Qdrant QueryPointGroups Integration & Capability Detection

**Status:** Completed. Adapter, service, and orchestrator now invoke server-side grouping with automatic capability probes, telemetry, and fallbacks in place.

**Objectives**

- Use server-side grouping (`QueryPointGroups`) to fetch distinct documents, with runtime capability checks and graceful fallback.
- Provide configuration knobs for `group_by`, `group_size`, and `groups_limit_multiplier`.

**Primary Tasks**

1. **Adapter Enhancements** (`src/services/vector_db/adapter.py`)
   - `query_groups` now wraps `AsyncQdrantClient.query_points_groups`, caches capability results, and annotates payloads with `_grouping` metadata on success.
2. **VectorStoreService Glue** (`src/services/vector_db/service.py`)
   - `_query_with_optional_grouping` always attempts grouped retrieval, enforces payload indexes once per collection, and records latency/fallback metrics via `MetricsRegistry`.
3. **Configuration** (`src/config/models.py`)
   - `QdrantConfig` exposes `enable_grouping`, `group_by_field`, `group_size`, and `groups_limit_multiplier`; defaults exercise grouping whenever the backend supports it.
4. **Orchestrator Integration** (`src/services/query_processing/orchestrator.py`)
   - Downstream results include canonical `collection`, `collection_confidence`, and `collection_priority` fields consumed by MCP tooling and analytics.
5. **Telemetry Hooks**
   - Prometheus counters/histograms track grouped query attempts, fallbacks, and latency (`grouping_requests_total`, `grouping_latency_seconds`).

**Impacted Tests**

- `tests/unit/services/query_processing/test_federated_service_merging.py`
- `tests/unit/services/query_processing/test_pipeline.py`
- `tests/unit/mcp_tools/tools/test_query_processing.py`

**Success Metrics**

- Grouped search telemetry shows high success ratio with fallbacks surfaced in alerts.
- Duplicate rate post-processing < 2% for single-collection queries (verified via merged-result tests).

**Risks & Mitigations**

- Qdrant version mismatch → capability probe, config fallback, document version pinning.
- Over-fetch latency → monitor P95; adjust multiplier; consider score thresholds.

**References**

- Qdrant Query Groups API: https://qdrant.tech/documentation/concepts/search/#query-groups
- Python client usage: https://github.com/qdrant/qdrant-client

---

## Phase 3 – Federated/Orchestrator Dedup Simplification & Over-Fetch Controls

**Status:** Completed. Client-side dedup now honours grouping metadata, normalized scores, and configurably over-fetches with minimal duplication.

**Objectives**

- Remove redundant client-side dedup loops when grouping is applied upstream.
- Introduce consistent over-fetch and normalized scoring across collections.

**Primary Tasks**

1. `_should_skip_dedup` inspects `search_metadata.grouping_applied` to avoid redundant dedup loops when server-side grouping succeeds.
2. `_prepare_hits` records raw score stats while `_normalize_scores_in_place` applies deterministic min-max or z-score normalization controlled by config.
3. `_calculate_collection_limit` honours `overfetch_multiplier` with sane ceilings.
4. Telemetry captures grouped usage via `grouping_requests_total`; MCP metadata now exposes `collection` for downstream analytics.

**Tests**

- `tests/unit/services/query_processing/test_federated_service_core.py`
- `tests/unit/services/query_processing/test_federated_service_merging.py`
- Integration tests: multi-collection grouping & normalization.

**Success Metrics**

- Post-merge duplicates < 1% (see `tests/unit/services/query_processing/test_federated_service_merging.py`).
- Federated latency within target SLO after grouping.

**Risks & Mitigations**

- Collections missing doc_id → schema enforcement (Phase 1), logging of offending collections.

---

## Phase 4 – Embeddings-Based Contextual Compression for RAG

**Status:** Completed. LangChain-based embeddings filters now drive RAG context trimming with quality gates and telemetry.

**Objectives**

- Apply deterministic embeddings-based compression using LangChain primitives; no LLM dependency.

**Primary Tasks**

1. **Compression Pipeline** (LangChain `EmbeddingsRedundantFilter` + `EmbeddingsFilter`)
   - Implements deterministic similarity-based filtering using FastEmbed embeddings and configurable thresholds.
   - Emits `_compression` metadata per document and exposes aggregate statistics (tokens before/after, reduction ratios).
2. **Retriever Integration** (`src/services/rag/retriever.py`, `src/services/query_processing/orchestrator.py`)
   - LangChain retriever wraps `VectorStoreService` results, applies compression when enabled, and forwards stats for telemetry.
   - Orchestrator propagates `contextual_compression` feature flags and embeds compression metrics in RAG responses.
3. **Metrics & Evaluation**
   - Prometheus counters/histograms capture tokens before/after and compression ratios per collection.
   - Evaluation CLI (`scripts/eval/rag_compression_eval.py`) and CI gate script (`scripts/ci/check_rag_compression.py`) enforce reduction and recall thresholds using labelled datasets.

**Tests**

- `tests/unit/services/rag/test_retriever.py`
- `tests/integration/services/query_processing/test_rag_pipeline.py`

**Success Metrics**

- Average context tokens reduced ≥ 40% with < 2% recall loss (validated via gate dataset and unit tests).
- Compression latency < 100 ms per query on baseline hardware (tracked via instrumentation).

**Risks & Mitigations**

- Over-aggressive compression harming answers → tune parameters, maintain fallback mode, track answer quality metrics.

**References**

- LangChain contextual compression (embeddings filter): https://js.langchain.com/docs/how_to/contextual_compression/
- MMR background: https://huggingface.co/docs/datasets/v3.0.0/en/process#maximal-marginal-relevance

---

## Phase 5 – Telemetry, Metrics, and Quality Gates

**Status:** Completed. Prometheus exposes grouped-search and compression metrics, and CI includes deterministic compression guardrails.

**Objectives**

- Provide observability to validate grouping/compression efficacy and detect regressions.

**Primary Tasks**

1. **Metrics Instrumentation**
   - Added `grouping_requests_total`, `grouping_latency_seconds`, `compression_ratio`, `compression_tokens_total`, and `compression_documents_total` with bounded labels (`collection`, `status`, `kind`).
   - Vector adapter and retriever report metric events whenever grouping or compression executes.
2. **Quality Gates**
   - CI entry point `scripts/ci/check_rag_compression.py` enforces minimum token reduction and recall on curated fixtures.
   - Metrics wiring ensures dashboards/alerts can consume the exported series.
3. **Alerts (Operational Guidelines)**
   - Recommended alert thresholds: grouping fallback >5%, grouped latency P95 > 250ms, compression recall < 0.8, compression reduction < 0.3. Configuration templates pending in observability repo.

**Tests**

- `tests/unit/services/monitoring/test_metrics.py` validates metric registration API.
- `tests/unit/services/query_processing/test_rag_compression.py` covers deterministic compression paths and retriever integration.

**Success Metrics**

- Dashboards live; alert policies tested.

**Risks & Mitigations**

- Metric cardinality inflation → limit labels to essential dimensions.

---

## Phase 6 – Documentation, Tests, and Decision Validation

**Objectives**

- Finalize documentation, decision framework scoring, and comprehensive test coverage.

**Primary Tasks**

1. **Documentation**
   - Update `docs/query_processing/architecture.md`, `docs/runbooks/query_processing.md`, and `CHANGELOG.md`.
   - Add decision matrix report to `docs/reports/query_processing_decision_matrix.md`.
2. **Test Overhaul**
   - Remove legacy `_total_*` references across unit/integration tests.
   - Expand tests for multi-tenant grouping, compression toggles, fallback behavior.
3. **Decision Framework Validation**
   - Run scoring script; archive results alongside metrics.

**Success Metrics**

- All tests pass; docs reviewed/approved; decision score ≥ 9.5 archived.

**Risks & Mitigations**

- Missed legacy references → repo search for keywords enforced via quality gate.

---

## Key References & Research

1. Qdrant QueryPointGroups: https://qdrant.tech/documentation/concepts/search/#query-groups
2. Qdrant Payload Schema/Indexes: https://qdrant.tech/documentation/concepts/payload/
3. Qdrant Python Client: https://github.com/qdrant/qdrant-client
4. LangChain Contextual Compression: https://js.langchain.com/docs/how_to/contextual_compression/
5. MMR / embeddings selection: https://huggingface.co/docs/datasets/v3.0.0/en/process#maximal-marginal-relevance

---

## Next Steps Checklist (Initial Actions)

1. Run payload audit and design migration plan (Phase 1 Task 1).
2. Implement adapter capability probe + grouped query method (Phase 2 Task 1).
3. Coordinate with DevOps to confirm Qdrant version compatibility and schedule index creation.
