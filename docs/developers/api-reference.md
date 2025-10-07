# API Reference (Simple Profile)

The simple profile exposes a minimal REST interface for testing the retrieval
stack. Routes live under `src/api/routers/simple/` and are mounted when
`AI_DOCS__MODE=simple`.

## Search Endpoints

### POST /search

Executes a vector search using `SimpleSearchRequest`.

Request body:
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
      "score": 0.89,
      "normalized_score": 0.91,
      "collection": "documents",
      "payload": {"title": "Introduction to Qdrant"}
    }
  ],
  "total_count": 1,
  "processing_time_ms": 12.5
}
```

### GET /search

Query parameters mirror the POST body and return the same response model. Useful
for quick manual checks.

### GET /search/health

Returns cached collection stats and indicates whether the simple search service
is ready.

```json
{
  "status": "healthy",
  "service_type": "simple",
  "stats": {
    "collections": ["documents"],
    "default_collection": "documents",
    "default_collection_stats": {"vectors_count": 1527}
  }
}
```

## Document Endpoints

### POST /documents

Indexes a single document by calling `VectorStoreService.add_document`.

```json
{
  "content": "Full document text",
  "metadata": {"source": "docs"},
  "collection_name": "documents"
}
```

Response:
```json
{
  "id": "doc_abc123",
  "status": "success",
  "message": "Document added successfully"
}
```

### GET /documents/{id}

Fetches a stored document from the collection. Returns `404` if not found.

### DELETE /documents/{id}

Removes a document. Success payload:
```json
{
  "status": "success",
  "message": "Document deleted successfully"
}
```

### GET /documents

Lists documents with basic pagination.

```json
{
  "documents": [
    {"id": "doc_abc123", "payload": {...}}
  ],
  "count": 1,
  "limit": 10,
  "next_offset": null
}
```

### GET /collections

Returns the collections known to the vector store service.

```json
{
  "collections": ["documents", "knowledge-base"]
}
```

## Enterprise Profile

When `AI_DOCS__MODE=enterprise`, additional routers are mounted to expose
orchestrated retrieval endpoints and administrative tools. Refer to the LangGraph
and FastMCP documentation for those APIs:

- `docs/developers/agentic-orchestration.md` – agentic workflows and payloads.
- `docs/developers/mcp-integration.md` – MCP tool interfaces and error models.
- `docs/developers/queries/response-contract.md` – response schema (see the
  relocated contract document).

All routes surface OpenAPI documentation at `/docs` and `/openapi.json`.
