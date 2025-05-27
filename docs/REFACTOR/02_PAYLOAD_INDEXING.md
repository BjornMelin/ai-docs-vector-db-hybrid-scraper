# Payload Indexing Implementation Guide

**GitHub Issue**: [#56](https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper/issues/56)

## Overview

Payload indexing in Qdrant enables 10-100x faster filtered searches by creating indexes on metadata fields. This is crucial for our documentation search system where users often filter by language, framework, version, or documentation type.

## Current State

Our current implementation stores metadata but doesn't index it:

```python
# Current: Unindexed payload storage
point = PointStruct(
    id=point_id,
    vector={"dense": dense_vector, "sparse": sparse_vector},
    payload={
        "content": chunk["content"],
        "metadata": chunk["metadata"],
        "url": chunk["metadata"].get("url", ""),
        "title": chunk["metadata"].get("title", ""),
        # These fields are stored but not indexed!
        "language": chunk["metadata"].get("language", ""),
        "framework": chunk["metadata"].get("framework", ""),
        "version": chunk["metadata"].get("version", ""),
    }
)
```

## Implementation Plan

### 1. Identify High-Value Fields

Based on our documentation sites configuration, these fields need indexing:

```python
INDEXED_FIELDS = {
    # Keyword indexes (exact match)
    "language": "keyword",      # "python", "typescript", "rust"
    "framework": "keyword",     # "fastapi", "nextjs", "react"
    "doc_type": "keyword",      # "api", "guide", "tutorial", "reference"
    "version": "keyword",       # "3.0", "14.2", "latest"
    
    # Text indexes (full-text search)
    "title": "text",           # Document titles
    "section": "text",         # Section headers
    
    # Numeric indexes (range queries)
    "last_updated": "integer",  # Unix timestamp
    "relevance_score": "float", # Pre-computed relevance
}
```

### 2. Create Indexes on Existing Collections

```python
async def create_payload_indexes(self, collection_name: str):
    """Create indexes on payload fields for faster filtering."""
    
    # Keyword indexes for exact matching
    for field in ["language", "framework", "doc_type", "version"]:
        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True
        )
    
    # Text indexes for full-text search
    for field in ["title", "section"]:
        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=PayloadSchemaType.TEXT,
            wait=True
        )
    
    # Numeric indexes for range queries
    await self.client.create_payload_index(
        collection_name=collection_name,
        field_name="last_updated",
        field_schema=PayloadSchemaType.INTEGER,
        wait=True
    )
    
    logger.info(f"Created payload indexes for {collection_name}")
```

### 3. Enhanced Search with Filters

```python
async def search_with_filters(
    self,
    query_vector: list[float],
    filters: dict[str, Any],
    limit: int = 10
) -> list[SearchResult]:
    """Search with indexed payload filters."""
    
    # Build filter conditions
    filter_conditions = []
    
    if language := filters.get("language"):
        filter_conditions.append(
            FieldCondition(
                key="language",
                match=MatchValue(value=language)
            )
        )
    
    if framework := filters.get("framework"):
        filter_conditions.append(
            FieldCondition(
                key="framework",
                match=MatchValue(value=framework)
            )
        )
    
    if version := filters.get("version"):
        filter_conditions.append(
            FieldCondition(
                key="version",
                match=MatchValue(value=version)
            )
        )
    
    # Date range filter
    if after_date := filters.get("updated_after"):
        filter_conditions.append(
            FieldCondition(
                key="last_updated",
                range=Range(gte=after_date)
            )
        )
    
    # Combine filters
    query_filter = Filter(
        must=filter_conditions
    ) if filter_conditions else None
    
    # Execute search with Query API
    results = await self.client.query_points(
        collection_name=self.collection_name,
        query=query_vector,
        filter=query_filter,
        limit=limit,
        with_payload=True
    )
    
    return results
```

### 4. Metadata Extraction Enhancement

Update our chunking system to extract structured metadata:

```python
def extract_metadata(self, content: str, url: str) -> dict[str, Any]:
    """Extract structured metadata for indexing."""
    
    metadata = {
        "url": url,
        "timestamp": int(time.time()),
    }
    
    # Extract from URL patterns
    url_parts = urlparse(url)
    domain = url_parts.netloc
    
    # Framework detection
    if "fastapi" in domain:
        metadata["framework"] = "fastapi"
        metadata["language"] = "python"
    elif "nextjs" in domain:
        metadata["framework"] = "nextjs"
        metadata["language"] = "typescript"
    elif "react" in domain:
        metadata["framework"] = "react"
        metadata["language"] = "javascript"
    
    # Version extraction from URL or content
    version_match = re.search(r'/v?(\d+\.?\d*\.?\d*)/', url)
    if version_match:
        metadata["version"] = version_match.group(1)
    else:
        metadata["version"] = "latest"
    
    # Document type classification
    if any(term in url.lower() for term in ["api", "reference"]):
        metadata["doc_type"] = "api"
    elif any(term in url.lower() for term in ["guide", "tutorial"]):
        metadata["doc_type"] = "guide"
    elif "blog" in url.lower():
        metadata["doc_type"] = "blog"
    else:
        metadata["doc_type"] = "reference"
    
    # Extract title from content
    title_match = re.search(r'<h1[^>]*>([^<]+)</h1>', content)
    if title_match:
        metadata["title"] = title_match.group(1).strip()
    
    return metadata
```

### 5. Index Management Tools

```python
async def list_indexes(self, collection_name: str) -> list[str]:
    """List all payload indexes in a collection."""
    collection_info = await self.client.get_collection(collection_name)
    return [
        field for field, schema in collection_info.payload_schema.items()
        if schema.indexed
    ]

async def drop_index(self, collection_name: str, field_name: str):
    """Drop a payload index."""
    await self.client.delete_payload_index(
        collection_name=collection_name,
        field_name=field_name,
        wait=True
    )

async def reindex_collection(self, collection_name: str):
    """Reindex all payload fields (useful after bulk updates)."""
    # Drop existing indexes
    existing_indexes = await self.list_indexes(collection_name)
    for index in existing_indexes:
        await self.drop_index(collection_name, index)
    
    # Recreate indexes
    await self.create_payload_indexes(collection_name)
```

## Performance Benchmarks

Expected improvements with payload indexing:

| Query Type | Without Index | With Index | Improvement |
|------------|---------------|------------|-------------|
| language="python" | 850ms | 12ms | 70x |
| framework="fastapi" AND version="0.100" | 1200ms | 18ms | 66x |
| updated_after=timestamp | 950ms | 25ms | 38x |
| Complex multi-filter | 1500ms | 35ms | 42x |

## Integration with Query API

Payload indexing works seamlessly with the Query API:

```python
# Multi-stage retrieval with filtered prefetch
results = await self.client.query_points(
    collection_name=collection_name,
    query=query_vector,
    filter=Filter(must=[
        FieldCondition(key="language", match=MatchValue(value="python"))
    ]),
    prefetch=[
        Prefetch(
            query=query_vector,
            using="dense",
            filter=Filter(must=[
                FieldCondition(key="framework", match=MatchValue(value="fastapi"))
            ]),
            limit=50
        )
    ],
    limit=10
)
```

## Migration Strategy

1. **Phase 1**: Add indexes to existing collections without downtime
2. **Phase 2**: Update insertion code to include structured metadata
3. **Phase 3**: Backfill metadata for existing documents
4. **Phase 4**: Enable filtered search in MCP server

## Testing

```python
@pytest.mark.asyncio
async def test_payload_indexing_performance():
    """Test that payload indexing improves query performance."""
    # Create collection without indexes
    await create_test_collection(indexed=False)
    
    # Benchmark unindexed search
    start = time.time()
    results = await search_with_filter({"language": "python"})
    unindexed_time = time.time() - start
    
    # Add indexes
    await create_payload_indexes()
    
    # Benchmark indexed search
    start = time.time()
    results = await search_with_filter({"language": "python"})
    indexed_time = time.time() - start
    
    # Assert significant improvement
    assert indexed_time < unindexed_time / 10  # At least 10x faster
```

## Best Practices

1. **Index only high-cardinality fields** - Don't index boolean fields
2. **Use keyword for exact matches** - Languages, frameworks, versions
3. **Use text for partial matches** - Titles, descriptions
4. **Monitor index size** - Indexes use memory, balance with performance
5. **Batch index creation** - Create all indexes before inserting data
6. **Regular maintenance** - Reindex after major updates

## Common Pitfalls

1. **Over-indexing** - Too many indexes slow down insertions
2. **Wrong field types** - Using text index for exact matches wastes resources
3. **Missing null handling** - Always provide default values
4. **Ignoring compound filters** - Test multi-field filter performance
5. **No monitoring** - Track index usage and performance

## Next Steps

After implementing payload indexing:

1. Update MCP server to expose filter parameters
2. Add filter UI to documentation search
3. Implement faceted search using indexed fields
4. Create analytics on most-used filters
5. Optimize index configuration based on usage patterns
