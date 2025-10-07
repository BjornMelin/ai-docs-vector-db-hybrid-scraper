# Query Processing Response Contract

The simplified query processing pipeline emits a single canonical response type:
`src.services.query_processing.models.SearchResponse`. Each response contains:

- `records`: `list[src.contracts.retrieval.SearchRecord]` with canonical fields:
  - `id`, `content`, optional `title`/`url` metadata
  - `collection`: source collection identifier
  - `raw_score`: un-normalized similarity score
  - `normalized_score`: per-request normalized score when enabled
  - `group_id`, `group_rank`, `grouping_applied`: grouping metadata derived from
    Qdrant `QueryPointGroups`
  - Arbitrary provider metadata preserved in `metadata`
- `total_results`: integer count of returned records
- `query`: processed query text (including expansion when enabled)
- `expanded_query`: optional expanded query string
- `processing_time_ms`: observed latency in milliseconds
- `features_used`: list of feature flags applied (e.g., `query_expansion`,
  `score_normalization`)
- `grouping_applied`: boolean indicating whether server-side grouping succeeded
- Optional RAG fields (`generated_answer`, `answer_confidence`, `answer_sources`)

`SearchResponse` and `SearchRecord` are the only supported DTOs. Legacy
`QueryProcessingResponse`/`QueryProcessingRequest` models, compatibility
wrappers, and response converters have been removed. Multi-collection fan-out
has been deprecated; clients must select a single collection per request and
opt into multi-tenant behaviour via dedicated orchestrator helpers. See
`docs/developers/compatibility-matrix.md` for the supported client and library
versions.

MCP tooling consumes the same canonical DTOs directly. Tests targeting the old
response converter pipeline have been deleted; new tests assert the simplified
pipeline behaviour in `tests/unit/services/query_processing/test_pipeline.py`
and `tests/unit/services/query_processing/test_orchestrator.py`. Regression
coverage for grouping fallback and score normalisation now lives in
`tests/unit/services/vector_db/test_service.py`.
