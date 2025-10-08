# API & Contracts

This guide documents the simple profile REST endpoints and the canonical
response contract used by the retrieval pipeline.

## 1. REST Endpoints (Simple Profile)

When `AI_DOCS__MODE=simple`, routes under `src/api/routers/simple/` are mounted.

### POST /search

```json
{
  "query": "vector databases",
  "collection": "documents",
  "limit": 10
}
```

Response (`SimpleSearchResponse`):

```json
{
  "query": "vector databases",
  "results": [
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
  "total_count": 1,
  "processing_time_ms": 12.5
}
```

### GET /search

Accepts the same parameters as the POST variant (`query`, `collection`, `limit`).
Returns a `SimpleSearchResponse` and serves as a convenient manual test endpoint.

### Document Endpoints

- `POST /documents` – Adds a document using `VectorStoreService.add_document`.
- `GET /documents/{id}` – Fetches a document (404 if missing).
- `DELETE /documents/{id}` – Removes a document and returns a success payload.
- `GET /documents` – Lists documents with pagination (`limit`, `offset`).
- `GET /collections` – Lists available collections.

### GET /health

Centralised readiness endpoint powered by `HealthCheckManager`. Example response:

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

## 2. Query Processing Response Contract

`src.services.query_processing.models.SearchResponse` is the canonical DTO.

Fields:

- `records`: `list[src.contracts.retrieval.SearchRecord]` with:
  - `id`, `content`, optional `title`/`url`
  - `collection`
  - `raw_score` (unnormalised), `normalized_score`
  - `group_id`, `group_rank`, `grouping_applied`
  - `metadata` (provider-specific annotations, replaces legacy vector payloads)
- `total_results`: number of returned records
- `query`: processed query text
- `expanded_query`: optional expanded variant
- `processing_time_ms`: observed latency
- `features_used`: applied features (`query_expansion`, `score_normalization`, etc.)
- `grouping_applied`: boolean flag
- Optional RAG fields: `generated_answer`, `answer_confidence`, `answer_sources`

Legacy DTOs (`QueryProcessingResponse`, multi-collection fan-out) have been
removed. Clients must supply a single collection per request and use orchestrator
helpers for multi-tenant behaviour.

MCP tooling consumes the same DTOs. Tests covering the simplified pipeline live
in `tests/unit/services/query_processing/test_pipeline.py`,
`tests/unit/services/query_processing/test_orchestrator.py`, and
`tests/unit/services/vector_db/test_service.py` (for grouping fallback).

## 3. Enterprise Surface

Enterprise mode mounts additional routers (LangGraph orchestration, MCP tooling).
See `docs/developers/architecture-and-orchestration.md` for endpoints delivered
through LangChain/LangGraph and FastMCP.

OpenAPI documentation for the active profile is always available at `/docs` and
`/openapi.json`.
