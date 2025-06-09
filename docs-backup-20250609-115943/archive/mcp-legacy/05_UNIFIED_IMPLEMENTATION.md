# Unified MCP Server Implementation Plan

> **Status**: Deprecated  
> **Last Updated**: 2025-06-09  
> **Purpose**: 05_Unified_Implementation archived documentation  
> **Audience**: Historical reference

## Executive Summary

Based on deep research of Crawl4AI, Firecrawl, Qdrant, and FastMCP documentation, this plan outlines how to create a unified MCP server that exposes our advanced features while integrating external MCP servers as clients.

## Key Findings from Documentation Research

### 1. Crawl4AI Advanced Features

- **AsyncWebCrawler** with JS execution, custom extraction, smart chunking
- **Deep crawling** with link analysis and multi-page strategies
- **LLM-optimized extraction** with markdown conversion and content cleaning
- **Session management** for stateful crawling
- **Advanced proxy support** with authentication
- **PDF and screenshot capture** capabilities
- **SSL certificate handling** for compliance
- **Multiple extraction strategies**: LLMExtractionStrategy, RegexExtractionStrategy, JsonCssExtractionStrategy, CosineStrategy
- **Chunking strategies**: RegexChunking, SlidingWindowChunking, OverlappingWindowChunking
- **Robots.txt compliance** with caching

### 2. Firecrawl MCP Server Capabilities

- **10 powerful tools**: scrape, map, crawl, search, extract, deep_research, generate_llmstxt, batch_scrape, check_batch_status
- **Batch operations** for processing multiple URLs with built-in rate limiting
- **Advanced extraction** with LLM-based structured data extraction
- **Deep research** tool that combines crawling, search, and AI analysis
- **LLMs.txt generation** for standardized AI interaction guidelines
- **Configurable retry logic** with exponential backoff
- **Credit usage monitoring** for cloud API usage
- **Support for both cloud and self-hosted** deployments

### 3. Qdrant MCP Server Features

- **Semantic memory layer** with store/find operations
- **Multi-collection support** for organizing knowledge
- **Metadata filtering** for precise retrieval
- **Vector similarity search** with configurable parameters
- **Hybrid queries** with Reciprocal Rank Fusion (RRF) and Distribution-Based Score Fusion (DBSF)
- **Multi-stage queries** for coarse-to-fine retrieval
- **Score boosting** with custom formulas based on payload values
- **Grouping capabilities** to avoid redundancy
- **Query API** with prefetch support for complex search strategies

### 4. FastMCP 2.0 Client Integration

- **External MCP client support** via `fastmcp.Client`
- **Tool wrapping** to expose external MCP tools as native tools
- **Async operation** for efficient multi-server coordination
- **Proxy server capabilities** via `FastMCP.as_proxy()`
- **Transport bridging** (e.g., SSE to Stdio)
- **Configuration-based proxies** supporting MCPConfig format
- **Multi-server composition** with automatic prefixing

### 5. mcp-crawl4ai-rag Features

- **Smart URL detection** for sitemaps, llms-full.txt, and regular pages
- **Recursive crawling** with parallel processing
- **Content chunking** by headers and size
- **Vector search with source filtering**
- **Supabase integration** for vector storage
- **Vision**: Multiple embedding models, advanced RAG strategies, Context 7-inspired chunking

## Proposed Unified Architecture

### Core MCP Server Structure

```python
# src/unified_mcp_server.py
from fastmcp import FastMCP, Context
from fastmcp.client import Client
from typing import Dict, List, Any, Optional
from enum import Enum
import asyncio

class UnifiedMCPServer:
    def __init__(self):
        self.mcp = FastMCP("ai-docs-unified-mcp")
        self.external_clients: Dict[str, Client] = {}
        self.setup_external_clients()
        self.register_tools()
    
    async def setup_external_clients(self):
        """Initialize connections to external MCP servers"""
        # Firecrawl MCP client
        self.external_clients['firecrawl'] = Client(
            "firecrawl-mcp",
            transport="stdio",
            command=["npx", "firecrawl-mcp"]
        )
        
        # Qdrant MCP client
        self.external_clients['qdrant'] = Client(
            "qdrant-mcp",
            transport="stdio", 
            command=["npx", "@qdrant/mcp-server", "--apiKey", "YOUR_KEY"]
        )
        
        # Initialize all clients
        for client in self.external_clients.values():
            await client.initialize()
```

### Tool Categories to Implement

#### 1. Advanced Embedding Tools

```python
@mcp.tool()
async def embed_with_provider(
    text: str,
    provider: str = "hybrid",  # openai, fastembed, hybrid
    model: Optional[str] = None,
    strategy: str = "semantic"  # semantic, late_chunking, contextual
) -> Dict[str, Any]:
    """Embed text using SOTA models with advanced strategies"""
    # Expose our NV-Embed-v2, BGE models, SPLADE++
    # Implement late chunking and contextual retrieval
```

#### 2. Hybrid Search Tools

```python
@mcp.tool()
async def hybrid_search(
    query: str,
    collection: str,
    use_reranking: bool = True,
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    alpha: float = 0.7,  # Weight between dense/sparse
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Perform hybrid search with automatic reranking"""
    # Combine dense and sparse embeddings
    # Apply BGE-reranker-v2-m3 for improved accuracy
```

#### 3. Advanced Crawling Tools

```python
@mcp.tool()
async def deep_crawl(
    url: str,
    max_depth: int = 3,
    strategy: str = "smart",  # smart, bfs, dfs
    extraction_schema: Optional[Dict] = None,
    use_session: bool = True
) -> Dict[str, Any]:
    """Deep crawl with Crawl4AI's advanced features"""
    # Use AsyncWebCrawler with JS execution
    # Apply smart link following
    # Extract structured data
```

#### 4. External MCP Integration Tools

```python
@mcp.tool()
async def firecrawl_deep_research(
    query: str,
    max_depth: int = 3,
    max_urls: int = 20
) -> Dict[str, Any]:
    """Proxy to Firecrawl's deep_research tool"""
    firecrawl = self.external_clients['firecrawl']
    return await firecrawl.call_tool(
        "firecrawl_deep_research",
        {"query": query, "maxDepth": max_depth, "maxUrls": max_urls}
    )

@mcp.tool()
async def qdrant_semantic_store(
    content: str,
    metadata: Dict[str, Any],
    collection: str = "knowledge"
) -> Dict[str, Any]:
    """Store in Qdrant with semantic embeddings"""
    qdrant = self.external_clients['qdrant']
    return await qdrant.call_tool(
        "qdrant-store",
        {"content": content, "metadata": metadata, "collection": collection}
    )
```

#### 5. Unified Workflow Tools

```python
@mcp.tool()
async def intelligent_document_pipeline(
    urls: List[str],
    extraction_prompt: str,
    embedding_strategy: str = "hybrid",
    store_in_qdrant: bool = True
) -> Dict[str, Any]:
    """Complete pipeline: crawl -> extract -> embed -> store"""
    results = []
    
    for url in urls:
        # 1. Use Firecrawl for advanced extraction
        extracted = await self.firecrawl_extract(url, extraction_prompt)
        
        # 2. Process with our SOTA chunking
        chunks = await self.smart_chunk(extracted['content'])
        
        # 3. Generate hybrid embeddings
        embeddings = await self.embed_with_provider(
            chunks, provider="hybrid", strategy="contextual"
        )
        
        # 4. Store in Qdrant if requested
        if store_in_qdrant:
            await self.qdrant_semantic_store(
                content=extracted['content'],
                metadata={"url": url, "chunks": len(chunks)},
                collection="documents"
            )
        
        results.append({
            "url": url,
            "chunks": len(chunks),
            "stored": store_in_qdrant
        })
    
    return {"processed": len(urls), "results": results}
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

1. Create `unified_mcp_server.py` with FastMCP 2.0 structure
2. Implement external MCP client connections
3. Set up basic tool registration framework
4. Create configuration system for API keys and settings

### Phase 2: Advanced Embedding Tools (Week 1-2)

1. Expose all embedding providers (OpenAI, FastEmbed, Hybrid)
2. Implement SOTA models (NV-Embed-v2, BGE, SPLADE++)
3. Add late chunking and contextual retrieval strategies
4. Create embedding comparison tools

### Phase 3: Hybrid Search & Reranking (Week 2)

1. Implement hybrid search combining dense/sparse vectors
2. Add BGE-reranker-v2-m3 integration
3. Create search evaluation tools
4. Add metadata filtering and faceted search

### Phase 4: External MCP Integration (Week 3)

1. Wrap Firecrawl MCP tools (especially deep_research)
2. Integrate Qdrant MCP for semantic storage
3. Create unified interfaces for common operations
4. Add error handling and retry logic

### Phase 5: Unified Workflows (Week 3-4)

1. Build intelligent document pipeline
2. Create research assistant workflow
3. Add knowledge graph building tools
4. Implement conversation memory management

### Phase 6: Testing & Documentation (Week 4)

1. Comprehensive test suite for all tools
2. Integration tests with external MCP servers
3. Performance benchmarks
4. Complete API documentation

## Configuration Example

```python
# config/unified_mcp_config.json
{
    "embedding": {
        "default_provider": "hybrid",
        "models": {
            "dense": "nvidia/NV-Embed-v2",
            "sparse": "prithvida/Splade_PP_en_v1"
        },
        "strategies": ["semantic", "late_chunking", "contextual"]
    },
    "search": {
        "default_reranker": "BAAI/bge-reranker-v2-m3",
        "hybrid_alpha": 0.7,
        "default_limit": 10
    },
    "external_mcp": {
        "firecrawl": {
            "enabled": true,
            "api_key_env": "FIRECRAWL_API_KEY"
        },
        "qdrant": {
            "enabled": true,
            "url": "http://localhost:6333",
            "api_key_env": "QDRANT_API_KEY"
        }
    },
    "workflows": {
        "document_pipeline": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "auto_store": true
        }
    }
}
```

## Tool Inventory

### Native Tools (Our Implementation)

1. **embed_with_provider** - SOTA embedding with multiple strategies (OpenAI, FastEmbed, Hybrid)
2. **hybrid_search** - Combined dense/sparse search with BGE-reranker-v2-m3
3. **smart_chunk** - Advanced chunking with late chunking, contextual retrieval
4. **deep_crawl** - Crawl4AI-based deep web crawling with JS execution
5. **extract_structured** - Multi-strategy extraction (LLM, Regex, CSS, Cosine)
6. **compare_embeddings** - Evaluate different embedding strategies
7. **build_knowledge_graph** - Create entity relationships from content
8. **capture_web_snapshot** - PDF and screenshot capture with metadata
9. **check_ssl_certificate** - Verify and export SSL certificates
10. **manage_crawl_session** - Handle stateful crawling sessions
11. **apply_chunking_strategy** - Apply various chunking strategies (regex, sliding window, overlapping)
12. **calculate_hybrid_scores** - Implement RRF and DBSF fusion algorithms

### Proxied External Tools

1. **firecrawl_scrape** - Advanced single-page scraping with JS rendering
2. **firecrawl_batch_scrape** - Batch scraping with rate limiting
3. **firecrawl_check_batch_status** - Monitor batch operation progress
4. **firecrawl_map** - Discover URLs from sitemap/links
5. **firecrawl_crawl** - Async multi-page crawling with depth control
6. **firecrawl_search** - Web search with optional content scraping
7. **firecrawl_extract** - Structured data extraction with schema
8. **firecrawl_deep_research** - AI-powered research with time limits
9. **firecrawl_generate_llmstxt** - Generate LLMs.txt files
10. **qdrant_store** - Semantic memory storage with collections
11. **qdrant_find** - Similarity search with metadata filtering
12. **qdrant_hybrid_query** - Execute hybrid queries with prefetch
13. **qdrant_multi_stage_search** - Coarse-to-fine retrieval
14. **qdrant_score_boost** - Apply custom scoring formulas
15. **qdrant_group_results** - Group results by field

### Unified Workflow Tools

1. **intelligent_document_pipeline** - End-to-end crawl → extract → embed → store
2. **research_assistant** - Multi-source research with deep analysis
3. **knowledge_sync** - Sync between Qdrant collections and other storage
4. **conversation_memory** - Manage conversation context with semantic search
5. **contextual_rag_pipeline** - Advanced RAG with contextual retrieval
6. **multi_modal_extraction** - Extract from text, images, PDFs
7. **incremental_knowledge_update** - Update existing knowledge base
8. **source_aware_search** - Search with automatic source filtering
9. **adaptive_crawl_strategy** - Choose crawl strategy based on content type
10. **embedding_optimization** - Optimize embeddings for specific domains

## Advanced Implementation Examples

### Hybrid Query with Qdrant Integration

```python
@mcp.tool()
async def advanced_hybrid_search(
    query: str,
    collection: str,
    fusion_method: str = "rrf",  # rrf or dbsf
    prefetch_configs: Optional[List[Dict]] = None
) -> List[Dict[str, Any]]:
    """Execute advanced hybrid search with Qdrant's Query API"""
    # Default prefetch for dense and sparse vectors
    if not prefetch_configs:
        prefetch_configs = [
            {
                "query": await self.embed_with_provider(
                    query, provider="fastembed", model="BAAI/bge-base-en-v1.5"
                ),
                "using": "dense",
                "limit": 20
            },
            {
                "query": await self.embed_with_provider(
                    query, provider="fastembed", model="prithvida/Splade_PP_en_v1"
                ),
                "using": "sparse", 
                "limit": 20
            }
        ]
    
    # Use Qdrant MCP for hybrid query
    return await self.external_clients['qdrant'].call_tool(
        "qdrant_hybrid_query",
        {
            "collection": collection,
            "prefetch": prefetch_configs,
            "fusion": fusion_method,
            "limit": 10
        }
    )
```

### Multi-Stage Search with Reranking

```python
@mcp.tool()
async def multi_stage_reranked_search(
    query: str,
    collection: str,
    use_matryoshka: bool = True
) -> List[Dict[str, Any]]:
    """Multi-stage search: coarse → fine → rerank"""
    stages = []
    
    if use_matryoshka:
        # Stage 1: Small MRL vector for candidates
        stages.append({
            "query": await self.embed_with_provider(
                query, provider="fastembed", 
                model="nvidia/NV-Embed-v2", 
                dimensions=256  # Smaller dimension
            ),
            "using": "mrl_256",
            "limit": 1000
        })
    
    # Stage 2: Full vector refinement
    stages.append({
        "query": await self.embed_with_provider(
            query, provider="fastembed", 
            model="nvidia/NV-Embed-v2"  # Full 768 dimensions
        ),
        "using": "full",
        "limit": 100
    })
    
    # Execute multi-stage query
    results = await self.external_clients['qdrant'].call_tool(
        "qdrant_multi_stage_search",
        {"collection": collection, "stages": stages}
    )
    
    # Stage 3: Rerank with BGE-reranker
    if results:
        reranked = await self.rerank_results(query, results)
        return reranked[:10]
    
    return results
```

### Crawl4AI Advanced Extraction

```python
@mcp.tool()
async def advanced_web_extraction(
    url: str,
    extraction_strategy: str = "auto",
    capture_evidence: bool = True
) -> Dict[str, Any]:
    """Advanced extraction with multiple strategies and evidence capture"""
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
    from crawl4ai.extraction_strategy import (
        LLMExtractionStrategy, JsonCssExtractionStrategy,
        RegexExtractionStrategy, CosineStrategy
    )
    
    async with AsyncWebCrawler() as crawler:
        # Auto-detect best strategy
        if extraction_strategy == "auto":
            # First, get the page to analyze structure
            initial_result = await crawler.arun(
                url=url,
                config=CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    scan_full_page=True
                )
            )
            
            # Analyze page structure to choose strategy
            if self._has_structured_data(initial_result.html):
                strategy = JsonCssExtractionStrategy(
                    schema=self._generate_schema(initial_result.html)
                )
            elif self._has_semantic_content(initial_result.markdown):
                strategy = LLMExtractionStrategy(
                    llm_config=LLMConfig(provider="ollama/llama2"),
                    instruction="Extract key information"
                )
            else:
                strategy = RegexExtractionStrategy(
                    pattern=RegexExtractionStrategy.All
                )
        
        # Execute extraction with evidence capture
        config = CrawlerRunConfig(
            extraction_strategy=strategy,
            pdf=capture_evidence,
            screenshot=capture_evidence,
            fetch_ssl_certificate=True
        )
        
        result = await crawler.arun(url=url, config=config)
        
        return {
            "extracted_content": result.extracted_content,
            "evidence": {
                "pdf": result.pdf if capture_evidence else None,
                "screenshot": result.screenshot if capture_evidence else None,
                "ssl_cert": result.ssl_certificate.to_dict() if result.ssl_certificate else None
            },
            "metadata": {
                "strategy_used": type(strategy).__name__,
                "tokens_used": len(result.markdown.split()) * 0.75
            }
        }
```

### Context-Aware Chunking

```python
@mcp.tool()
async def context_aware_chunking(
    content: str,
    chunk_strategy: str = "contextual",
    target_chunk_size: int = 1000
) -> List[Dict[str, Any]]:
    """Apply Context 7-inspired chunking for better retrieval"""
    from crawl4ai.chunking_strategy import (
        OverlappingWindowChunking, RegexChunking
    )
    
    if chunk_strategy == "contextual":
        # Context 7 approach: Focus on examples and semantic boundaries
        chunks = []
        
        # Split by headers first
        header_splitter = RegexChunking(
            patterns=[r'\n#{1,6}\s+', r'\n\n## ']
        )
        sections = header_splitter.chunk(content)
        
        for section in sections:
            # Extract examples and code blocks
            examples = self._extract_examples(section)
            
            # Create contextual chunks with surrounding context
            for i, example in enumerate(examples):
                chunk = {
                    "content": example["content"],
                    "context": example.get("surrounding_text", ""),
                    "type": example["type"],  # code, example, explanation
                    "metadata": {
                        "section_title": self._extract_section_title(section),
                        "has_code": bool(example.get("code")),
                        "position": i
                    }
                }
                chunks.append(chunk)
        
        return chunks
    
    else:
        # Fallback to overlapping chunks
        chunker = OverlappingWindowChunking(
            window_size=target_chunk_size,
            overlap=int(target_chunk_size * 0.2)
        )
        return [{"content": chunk} for chunk in chunker.chunk(content)]
```

## Benefits of This Approach

1. **Best of All Worlds**: Combines our SOTA embeddings with external specialized services
2. **Flexibility**: Users can choose providers and strategies per use case
3. **Performance**: Leverages optimized external services while maintaining local control
4. **Extensibility**: Easy to add new external MCP servers or tools
5. **Unified Interface**: Single MCP server exposes all capabilities
6. **Advanced Capabilities**: Hybrid search, multi-stage retrieval, contextual chunking
7. **Evidence Trail**: Capture PDFs, screenshots, and SSL certificates for compliance
8. **Smart Defaults**: Auto-detection of best strategies based on content

## Next Steps

1. Review and approve this implementation plan
2. Begin Phase 1 implementation with core infrastructure
3. Set up development environment with all external dependencies
4. Create initial test suite structure
5. Start documentation framework
6. Implement the most critical tools first (hybrid search, advanced embeddings)
7. Set up integration tests with external MCP servers

This unified approach will create a powerful MCP server that exposes our advanced features while leveraging the best external tools available, providing a comprehensive solution for AI-powered document processing and retrieval.
