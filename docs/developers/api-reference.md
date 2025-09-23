# API Reference (Essential Endpoints Only)

## Base Configuration

Base URL: https://api.example.com/v1

Headers:
- Content-Type: application/json
- Authorization: Bearer {api_key}
- Accept: application/json

## Search Endpoints

### POST /search

Search documents using simple query syntax.

Request:
{
  "query": "machine learning",
  "collection": "research-papers",
  "limit": 10,
  "offset": 0
}

Response:
{
  "results": [
    {
      "id": "doc_12345",
      "title": "Introduction to Machine Learning",
      "score": 0.95,
      "metadata": {
        "author": "John Smith",
        "year": 2023
      }
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}

### POST /search/advanced

Search documents using advanced filtering and sorting.

Request:
{
  "query": "neural networks",
  "filters": {
    "year": {"gte": 2020},
    "author": "Jane Doe"
  },
  "sort": [{"field": "year", "order": "desc"}],
  "collection": "academic",
  "limit": 5
}

Response:
{
  "results": [
    {
      "id": "doc_67890",
      "title": "Deep Neural Networks in 2023",
      "score": 0.87,
      "metadata": {
        "author": "Jane Doe",
        "year": 2023,
        "journal": "AI Review"
      }
    }
  ],
  "total": 1,
  "limit": 5,
  "offset": 0
}

## Document Management

### POST /documents

Create a new document in a collection.

Request:
{
  "title": "Document Title",
  "content": "Full text content of the document",
  "metadata": {
    "author": "Alice Johnson",
    "tags": ["research", "analysis"]
  },
  "collection": "documents"
}

Response:
{
  "id": "doc_abc123",
  "title": "Document Title",
  "created_at": "2023-01-15T10:30:00Z"
}

### GET /documents/{id}

Retrieve a document by its ID.

Response:
{
  "id": "doc_abc123",
  "title": "Document Title",
  "content": "Full text content of the document",
  "metadata": {
    "author": "Alice Johnson",
    "tags": ["research", "analysis"],
    "created_at": "2023-01-15T10:30:00Z"
  },
  "collection": "documents"
}

### DELETE /documents/{id}

Delete a document by its ID.

Response:
{
  "deleted": true,
  "id": "doc_abc123"
}

## Collection Management

### GET /collections

List all available collections.

Response:
{
  "collections": [
    {
      "name": "research-papers",
      "document_count": 1250
    },
    {
      "name": "academic",
      "document_count": 842
    }
  ]
}

### POST /collections

Create a new collection.

Request:
{
  "name": "new-collection",
  "description": "Collection for new documents"
}

Response:
{
  "name": "new-collection",
  "description": "Collection for new documents",
  "document_count": 0,
  "created_at": "2023-01-15T10:30:00Z"
}

### DELETE /collections/{name}

Delete a collection by name.

Response:
{
  "deleted": true,
  "name": "new-collection"
}

## Essential Schemas

### SearchRequest
{
  "query": "string",
  "collection": "string",
  "limit": "integer (optional)",
  "offset": "integer (optional)"
}

### AdvancedSearchRequest
{
  "query": "string",
  "filters": "object (optional)",
  "sort": "array (optional)",
  "collection": "string",
  "limit": "integer (optional)"
}

### Document
{
  "id": "string",
  "title": "string",
  "content": "string",
  "metadata": "object",
  "collection": "string"
}

### Collection
{
  "name": "string",
  "description": "string",
  "document_count": "integer"
}