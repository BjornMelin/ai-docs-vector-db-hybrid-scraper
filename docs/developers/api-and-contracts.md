# API & Contracts

This guide summarizes the canonical REST surface exposed under `/api/v1` and
the response contracts produced by the retrieval pipeline. Both supported
profiles (`simple` and `enterprise`) mount the same versioned routers; the
profile only influences middleware, rate limits, and which background services
are initialised.

## 1. REST Endpoints (`/api/v1/*`)

All endpoints are defined in `src/api/routers/v1/` and backed by container
managed services from `src/services/service_resolver.py`.

### POST `/api/v1/search`

```json
{
  "query": "vector databases",
  "collection": "documents",
  "limit": 10
}
```

Retrieval mode is an application startup setting. Set
`AI_DOCS_EMBEDDING__RETRIEVAL_MODE` to `dense`, `sparse`, or `hybrid`; hybrid
combines FastEmbed dense vectors with configured sparse vectors.

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
LangChain `Document` instances via
`src.services.vector_db.document_builder`. Document building provides the
ingestion metadata below; the persistence boundary adds storage-owned fields:

- `source`, `uri_or_path`, `doc_id`, and `tenant` – provenance identifiers
- `title`, `content_type`, `lang` – presentation metadata
- `chunk_index`, `total_chunks` – chunk bookkeeping assigned during document
  building
- `content_hash` – change detection assigned only at the vector persistence
  boundary
- `created_at`, `updated_at` – ISO timestamps captured during ingestion
- Content Intelligence enrichments when available (`content_type`,
  `content_confidence`, `quality_*`, `ci_*` fields)

Legacy chunk dictionaries and ad-hoc metadata fields are no longer produced nor
accepted by caches. Cached `AddDocumentResponse` objects are serialised in-place
and hydrated directly from JSON when read back.

Chunk generation is centralised in
`src/services/document_chunking.chunk_to_documents`, which inspects crawler
metadata to select LangChain splitters (Markdown headers, semantic HTML,
code-aware recursive character splitting, JSON segmentation, token-aware, or
plain-text splitters). `ChunkingConfig` exposes chunk size/overlap, token-aware
limits, JSON window sizes, and HTML normalisation flags; MCP and CLI requests map
one-to-one to those fields.

`VectorStoreService` persists the resulting payloads through LangChain's
`QdrantVectorStore` using its native `page_content` plus nested `metadata`
payload. Point IDs are stable UUID5 values derived from tenant, document, and
chunk position; the ID returned by create is the ID used by get and delete.
Complete document re-ingestion removes obsolete trailing chunks only after the
new chunk set is stored successfully. Collection vector shape, distance, and
sparse-vector policy come from the configured embedding and retrieval stack;
the CLI does not expose incompatible per-collection overrides.
HTTP request collection fields default to `settings.qdrant.collection_name`
when omitted. FastEmbed dense and sparse embeddings are initialised once
and reused across ingestion surfaces so hybrid scoring is available when the
application starts with `EmbeddingConfig.retrieval_mode` set to `hybrid`.

#### Required collection rebuild

This contract is a forward-only hard cut. Collections written by earlier
versions use a different flat payload and content-derived point IDs; the new
runtime does not read or migrate them. Before rollout, stop ingestion writers,
close application traffic for a maintenance window, deploy the new runtime,
run `uv run manage-db clear <collection>` for every existing collection, and
rerun the authoritative ingestion jobs before reopening traffic. Skipping the
clear step leaves legacy points that the new read path cannot decode.

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

### MCP Tooling Response Models

MCP server APIs now expose a single, final surface under
`src/mcp_tools/models/responses.py`. Only the active DTOs remain:

- `AnalyticsResponse`, `SystemHealthResponse`
- `CacheClearResponse`, `CacheStatsResponse`
- `CollectionInfo`, `CollectionOperationResponse`, `ReindexCollectionResponse`
- `AddDocumentResponse`, `DocumentBatchResponse`
- `EmbeddingGenerationResponse`, `EmbeddingProviderInfo`
- `OperationStatus`, `ProjectInfo`, `GenericDictResponse`
- `ContentIntelligenceResult`

Import `ContentType` and other content-intelligence enums directly from
`src/services/content_intelligence/models.py`; no compatibility re-export is
provided by the MCP layer.

## 3. Enterprise Extensions

Enterprise deployments mount additional routers for LangGraph workflows and MCP
tooling, but continue to rely on the same `/api/v1` contract. See
`docs/developers/architecture-and-orchestration.md` for a deep dive into
extended surfaces.
