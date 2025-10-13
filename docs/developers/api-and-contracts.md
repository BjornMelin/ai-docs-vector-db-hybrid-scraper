# API & Contracts

This guide summarizes the canonical REST surface exposed under `/api/v1` and
the response contracts produced by the retrieval pipeline. Both supported
profiles (`simple` and `enterprise`) mount the same versioned routers; the
profile only influences middleware, rate limits, and which background services
are initialised.

## 1. REST Endpoints (`/api/v1/*`)

All endpoints are defined in `src/api/routers/v1/` and backed by container
managed services from `src/services/dependencies.py`.

### POST `/api/v1/search`

```json
{
  "query": "vector databases",
  "collection": "documents",
  "limit": 10
}
```

Response (`SearchResponse`):

```json
{
  "query": "vector databases",
  "records": [
    {
      "id": "doc_123",
      "content": "Qdrant is a vector database...",
      "score": 0.89,
      "raw_score": 0.93,
      "normalized_score": 0.91,
      "collection": "documents",
      "metadata": {"title": "Introduction to Qdrant"}
    }
  ],
  "total_results": 1,
  "processing_time_ms": 12.5,
  "features_used": ["hybrid_search", "rerank"]
}
```

### GET `/api/v1/search`

Accepts the same query parameters as the POST variant (`query`, `collection`,
`limit`). This route is useful for manual smoke tests.

### Document management

- `POST /api/v1/documents` – Adds a document using
  `VectorStoreService.add_document`.
- `GET /api/v1/documents/{id}` – Fetches a document (404 if missing).
- `DELETE /api/v1/documents/{id}` – Removes a document and returns a success
  payload.
- `GET /api/v1/documents` – Lists documents with pagination (`limit`, `offset`).
- `GET /api/v1/collections` – Lists available collections.

#### Canonical ingestion payload

The ingestion surface (MCP tools, CLI pipelines, and bulk embedders) now emits
`TextDocument` payloads constructed from LangChain `Document` chunks via
`src.services.vector_db.document_builder`. Each chunk guarantees the same
metadata keys so downstream services can rely on a predictable schema:

- `source`, `uri_or_path`, `doc_id`, and `tenant` – provenance identifiers
- `title`, `content_type`, `lang` – presentation metadata
- `chunk_index`, `chunk_id`, `chunk_hash`, `total_chunks` – chunk bookkeeping
- `created_at`, `updated_at` – ISO timestamps captured during ingestion
- Content Intelligence enrichments when available (`content_type`,
  `content_confidence`, `quality_*`, `ci_*` fields)

Legacy chunk dictionaries and ad-hoc metadata fields are no longer produced nor
accepted by caches. Cached `AddDocumentResponse` objects are serialised in-place
and hydrated directly from JSON when read back.

### Health

`GET /health` exposes readiness information collected by `HealthCheckManager`.
Example payload:

```json
{
  "status": "healthy",
  "mode": "simple",
  "services": {
    "qdrant": {
      "status": "healthy",
      "message": "Qdrant service is operational",
      "metadata": {"collection_count": 3}
    },
    "redis": {
      "status": "healthy",
      "message": "Redis server is responding",
      "metadata": {"connected_clients": 12}
    }
  },
  "healthy_count": 2,
  "total_count": 2,
  "timestamp": 1728501123.123
}
```

## 2. Search Response Contract

`src.contracts.retrieval.SearchResponse` is the canonical DTO returned by both
FastAPI routes and MCP tooling.

Fields:

- `records`: list of `SearchRecord` items providing:
  - `id`, `content`, optional `title` / `url`
  - `collection`
  - `raw_score` (unnormalised), `normalized_score`
  - `group_id`, `group_rank`, `grouping_applied`
  - `metadata` (provider-specific annotations)
- `total_results`: number of returned records
- `query`: processed query text
- `expanded_query`: optional expanded variant
- `processing_time_ms`: observed latency
- `features_used`: applied features (`query_expansion`, `score_normalization`,
  etc.)
- `grouping_applied`: boolean flag
- Optional RAG fields: `generated_answer`, `answer_confidence`,
  `answer_sources`

Legacy DTOs (`QueryProcessingResponse`, multi-collection fan-out) have been
removed. Clients must supply a single collection per request and use
orchestrator helpers for multi-tenant behaviour.

MCP tooling consumes the same DTOs. Contract coverage lives in
`tests/unit/services/query_processing/test_orchestrator.py`,
`tests/unit/services/vector_db/test_service.py`, and
`tests/unit/models/test_search_request.py`.

## 3. Enterprise Extensions

Enterprise deployments mount additional routers for LangGraph workflows and MCP
tooling, but continue to rely on the same `/api/v1` contract. See
`docs/developers/architecture-and-orchestration.md` for a deep dive into
extended surfaces.
