# MCP Server Architecture & Tool Specifications

## Overview

Our unified MCP server provides comprehensive documentation management capabilities through a carefully designed set of tools. This architecture leverages FastMCP 2.0's advanced features including streaming, resources, tool composition, and context injection.

## Core Design Principles

1. **Direct API/SDK Integration** - Use Qdrant Python SDK and OpenAI SDK directly (no MCP proxying)
2. **Smart Model Selection** - Automatically choose optimal embedding models based on task
3. **Advanced Search Capabilities** - Hybrid search with RRF/DBSF fusion and reranking
4. **Resource-Based Architecture** - Expose data and configurations as MCP resources
5. **Streaming Support** - Handle large result sets efficiently
6. **Tool Composition** - Combine multiple operations into intelligent workflows

## MCP Server Tool Specifications

### 1. Documentation Scraping Tools

#### `scrape_documentation`

```python
@mcp.tool()
async def scrape_documentation(
    url: str,
    max_depth: int = 3,
    chunk_size: int = 1600,
    chunk_strategy: str = "enhanced",  # basic, enhanced, ast
    include_code: bool = True,
    follow_patterns: list[str] = None,
    exclude_patterns: list[str] = None,
    ctx: Context
) -> dict:
    """
    Scrape and index documentation from a URL with intelligent chunking.
    
    Features:
    - Auto-detects documentation framework (Sphinx, MkDocs, etc.)
    - Uses AST-based chunking for code blocks
    - Preserves document structure and hierarchy
    - Generates embeddings with optimal model selection
    """
```

#### `scrape_github_repo`

```python
@mcp.tool()
async def scrape_github_repo(
    repo_url: str,
    include_docs: bool = True,
    include_code: bool = True,
    include_issues: bool = False,
    branch: str = "main",
    ctx: Context
) -> dict:
    """
    Scrape documentation and code from a GitHub repository.
    
    Features:
    - Extracts README, docs/, wiki content
    - Parses code with language-aware chunking
    - Optional issue/discussion extraction
    - Respects .gitignore patterns
    """
```

### 2. Advanced Search Tools

#### `hybrid_search`

```python
@mcp.tool()
async def hybrid_search(
    query: str,
    collection: str = "documentation",
    search_type: str = "hybrid",  # dense, sparse, hybrid
    fusion_method: str = "rrf",  # rrf, dbsf
    prefetch_limit: int = 100,
    final_limit: int = 10,
    rerank: bool = True,
    filters: dict = None,
    ctx: Context
) -> dict:
    """
    Perform advanced hybrid search with multi-stage retrieval.
    
    Implements Qdrant Query API with:
    - Dense + sparse vector search
    - RRF or DBSF fusion
    - Multi-stage prefetch and reranking
    - Metadata filtering
    - Score boosting based on recency/relevance
    """
```

#### `semantic_search`

```python
@mcp.tool()
async def semantic_search(
    query: str,
    collection: str = "documentation",
    model: str = "auto",  # auto, small, large, multilingual
    limit: int = 10,
    min_score: float = 0.7,
    include_context: bool = True,
    ctx: Context
) -> dict:
    """
    Pure semantic search with smart model selection.
    
    Model selection:
    - auto: Chooses based on query complexity
    - small: Fast, cost-effective (text-embedding-3-small)
    - large: High accuracy (text-embedding-3-large)
    - multilingual: BGE-m3 for cross-lingual search
    """
```

#### `multi_query_search`

```python
@mcp.tool()
async def multi_query_search(
    queries: list[str],
    strategy: str = "ensemble",  # ensemble, chain, comparative
    aggregate_method: str = "weighted",  # weighted, union, intersection
    ctx: Context
) -> dict:
    """
    Execute multiple search queries with intelligent aggregation.
    
    Strategies:
    - ensemble: Combines results from all queries
    - chain: Uses results from one query to inform the next
    - comparative: Finds documents matching all queries
    """
```

### 3. Vector Database Management Tools

#### `create_collection`

```python
@mcp.tool()
async def create_collection(
    name: str,
    vector_config: dict = None,
    optimized_for: str = "balanced",  # balanced, accuracy, speed, storage
    enable_compression: bool = True,
    ctx: Context
) -> dict:
    """
    Create optimized vector collection with smart defaults.
    
    Optimization profiles:
    - balanced: int8 quantization, HNSW index
    - accuracy: Full precision, optimized HNSW
    - speed: Aggressive quantization, fast indexing
    - storage: Maximum compression, binary quantization
    """
```

#### `index_documents`

```python
@mcp.tool()
async def index_documents(
    documents: list[dict],
    collection: str,
    embedding_strategy: str = "auto",
    batch_size: int = 100,
    generate_sparse: bool = True,
    extract_keywords: bool = True,
    ctx: Context
) -> dict:
    """
    Batch index documents with optimal embedding generation.
    
    Features:
    - Automatic batching for efficiency
    - Dual embedding generation (dense + sparse)
    - Keyword extraction for hybrid search
    - Progress reporting via ctx.report_progress()
    """
```

#### `update_document`

```python
@mcp.tool()
async def update_document(
    doc_id: str,
    collection: str,
    content: str = None,
    metadata: dict = None,
    regenerate_embeddings: bool = True,
    ctx: Context
) -> dict:
    """
    Update existing document with incremental reindexing.
    
    Features:
    - Partial updates (content or metadata only)
    - Smart embedding regeneration
    - Version tracking
    - Maintains document history
    """
```

### 4. Embedding Management Tools

#### `generate_embeddings`

```python
@mcp.tool()
async def generate_embeddings(
    texts: list[str],
    model: str = "auto",
    output_format: str = "standard",  # standard, compressed, matryoshka
    dimensions: int = None,
    ctx: Context
) -> dict:
    """
    Generate embeddings with flexible model selection.
    
    Models:
    - OpenAI text-embedding-3-small/large
    - BGE-base/large/reranker
    - NV-Embed-v2 (if available)
    - Local models via FastEmbed
    
    Output formats:
    - standard: Full precision embeddings
    - compressed: Int8 quantization
    - matryoshka: Truncatable embeddings
    """
```

#### `compare_embedding_models`

```python
@mcp.tool()
async def compare_embedding_models(
    test_queries: list[str],
    test_documents: list[str],
    models: list[str] = ["all"],
    metrics: list[str] = ["accuracy", "speed", "cost"],
    ctx: Context
) -> dict:
    """
    Compare embedding models for specific use case.
    
    Returns:
    - Accuracy scores (semantic similarity)
    - Inference speed benchmarks
    - Cost analysis per 1M tokens
    - Recommendations based on requirements
    """
```

### 5. Reranking Tools

#### `rerank_results`

```python
@mcp.tool()
async def rerank_results(
    query: str,
    documents: list[dict],
    model: str = "bge-reranker-v2-m3",
    top_k: int = 10,
    include_scores: bool = True,
    ctx: Context
) -> dict:
    """
    Rerank search results with cross-encoder models.
    
    Models:
    - BGE-reranker-v2-m3 (recommended)
    - ColBERT (for multi-vector reranking)
    - Custom fine-tuned rerankers
    """
```

### 6. Analytics and Monitoring Tools

#### `get_search_analytics`

```python
@mcp.tool()
async def get_search_analytics(
    collection: str = None,
    time_range: str = "7d",
    include_queries: bool = True,
    include_performance: bool = True,
    ctx: Context
) -> dict:
    """
    Retrieve search analytics and performance metrics.
    
    Metrics:
    - Query frequency and patterns
    - Search latency percentiles
    - Cache hit rates
    - Model usage distribution
    - Cost analysis by operation
    """
```

#### `optimize_collection`

```python
@mcp.tool()
async def optimize_collection(
    collection: str,
    optimization_goal: str = "balanced",
    dry_run: bool = True,
    ctx: Context
) -> dict:
    """
    Analyze and optimize collection configuration.
    
    Optimizations:
    - Index parameter tuning
    - Quantization recommendations
    - Shard rebalancing
    - Unused vector cleanup
    """
```

### 7. Composed Tools (Advanced Workflows)

#### `smart_index_documentation`

```python
@mcp.tool()
async def smart_index_documentation(
    source: str,  # URL, GitHub repo, or file path
    project_name: str,
    auto_organize: bool = True,
    enable_monitoring: bool = True,
    ctx: Context
) -> dict:
    """
    Intelligent documentation indexing with full pipeline.
    
    Workflow:
    1. Detect source type and optimal scraping strategy
    2. Create optimized collection if needed
    3. Scrape with intelligent chunking
    4. Generate hybrid embeddings
    5. Set up monitoring and analytics
    6. Provide search recommendations
    """
```

#### `migrate_collection`

```python
@mcp.tool()
async def migrate_collection(
    source_collection: str,
    target_collection: str,
    upgrade_embeddings: bool = True,
    new_model: str = None,
    ctx: Context
) -> dict:
    """
    Migrate collection with optional embedding upgrades.
    
    Features:
    - Streaming migration for large collections
    - Optional embedding regeneration
    - Progress tracking with resume capability
    - Zero-downtime migration option
    """
```

## Resources Architecture

### Configuration Resources

```python
@mcp.resource("config://embedding-models")
async def get_embedding_models() -> dict:
    """Available embedding models with capabilities and costs."""
    return {
        "models": {
            "text-embedding-3-small": {
                "dimensions": 768,
                "cost_per_1m_tokens": 0.02,
                "speed": "fast",
                "accuracy": "good"
            },
            "text-embedding-3-large": {
                "dimensions": 3072,
                "cost_per_1m_tokens": 0.13,
                "speed": "medium",
                "accuracy": "excellent"
            },
            # ... more models
        }
    }

@mcp.resource("config://search-strategies")
async def get_search_strategies() -> dict:
    """Available search strategies and their configurations."""

@mcp.resource("config://chunking-strategies")
async def get_chunking_strategies() -> dict:
    """Documentation chunking strategies with examples."""
```

### Dynamic Resources

```python
@mcp.resource("stats://collections/{collection_name}")
async def get_collection_stats(collection_name: str) -> dict:
    """Real-time statistics for a specific collection."""

@mcp.resource("analytics://search-performance")
async def get_search_performance() -> dict:
    """Current search performance metrics."""

@mcp.resource("costs://usage-summary")
async def get_usage_costs() -> dict:
    """Cost breakdown by operation and model."""
```

## Advanced Features Implementation

### 1. Streaming Support

```python
@mcp.tool()
async def stream_search_results(
    query: str,
    collection: str,
    page_size: int = 20,
    ctx: Context
) -> AsyncIterator[dict]:
    """
    Stream search results for large result sets.
    
    Yields results in pages with progress updates.
    """
    total_results = await get_result_count(query, collection)
    await ctx.report_progress(0, total_results)
    
    async for page in paginated_search(query, collection, page_size):
        yield {
            "results": page.results,
            "page": page.number,
            "has_more": page.has_next
        }
        await ctx.report_progress(page.number * page_size, total_results)
```

### 2. Context-Aware Operations

All tools receive a `Context` object enabling:

- Progress reporting: `await ctx.report_progress(current, total)`
- Logging: `await ctx.info("Processing...")`
- Resource access: `await ctx.read_resource("config://settings")`
- LLM sampling: `await ctx.sample("Summarize: " + content)`

### 3. Error Handling

```python
from fastmcp.exceptions import ToolError

@mcp.tool()
async def robust_operation(params: dict, ctx: Context) -> dict:
    try:
        # Operation logic
        pass
    except ValidationError as e:
        raise ToolError(f"Invalid parameters: {e}")
    except ExternalAPIError as e:
        # Retry logic
        await ctx.warning(f"Retrying due to: {e}")
        # ...
```

## Security and Performance

### Security Measures

- Input validation on all tool parameters
- Rate limiting for expensive operations
- API key management via environment variables
- Sanitization of file paths and URLs

### Performance Optimizations

- Connection pooling for Qdrant client
- Embedding caching with TTL
- Batch processing for bulk operations
- Async/await throughout for non-blocking I/O
- Smart prefetch limits based on query complexity

## Next Steps

1. Implement core tools with FastMCP decorators
2. Set up comprehensive testing suite
3. Create usage examples and tutorials
4. Deploy with monitoring and analytics
5. Iterate based on user feedback

This architecture provides a solid foundation for a powerful documentation vector database with advanced search capabilities while maintaining flexibility for future enhancements.
