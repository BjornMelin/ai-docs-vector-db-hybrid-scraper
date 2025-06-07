# Crawl4AI Integration Guide

**GitHub Issue**: [#58](https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper/issues/58)

## Overview

Crawl4AI is a high-performance, free web scraping library that will replace our current Firecrawl dependency. It offers 4-6x performance improvement, $0 cost, and advanced features like JavaScript execution and intelligent content extraction.

## Current State vs Target State

### Current: Firecrawl Integration

```python
# Current implementation in crawl4ai_bulk_embedder.py
class CrawlService:
    def __init__(self):
        self.firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
        self.app = FirecrawlApp(api_key=self.firecrawl_api_key)
    
    async def crawl_url(self, url: str) -> dict:
        result = self.app.scrape_url(url, params={
            'formats': ['markdown'],
            'onlyMainContent': True
        })
        return result
```

### Target: Crawl4AI Integration

```python
# New implementation with Crawl4AI
from crawl4ai import AsyncWebCrawler, CrawlerConfig, CrawlResult
from crawl4ai.extraction_strategy import LLMExtractionStrategy

class EnhancedCrawlService:
    def __init__(self):
        self.config = CrawlerConfig(
            browser_type="chromium",
            headless=True,
            verbose=False,
            user_agent="Mozilla/5.0 (compatible; AIDocs/1.0)",
            page_timeout=30000,
            fetch_timeout=10000,
        )
    
    async def crawl_url(self, url: str) -> CrawlResult:
        async with AsyncWebCrawler(config=self.config) as crawler:
            result = await crawler.arun(
                url=url,
                word_count_threshold=10,
                extraction_strategy=LLMExtractionStrategy(
                    provider="ollama/llama2",
                    instruction="Extract technical documentation content"
                ),
                chunking_strategy="semantic",
                css_selector="main, article, .content, .documentation",
                excluded_tags=['nav', 'footer', 'header', 'aside'],
                wait_for_selector=".content",
                js_code="window.scrollTo(0, document.body.scrollHeight);",
                cache_mode="enabled",
            )
            return result
```

## Implementation Plan

### 1. Core Integration

```python
# src/services/crawling/crawl4ai_provider.py
from typing import Any, Optional
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerConfig, CrawlResult
from crawl4ai.extraction_strategy import (
    JsonCssExtractionStrategy,
    LLMExtractionStrategy,
    SemanticChunkingStrategy
)

from ..base import BaseService
from ..errors import CrawlingError
from ..rate_limiter import RateLimiter

class Crawl4AIProvider(BaseService):
    """High-performance web crawling with Crawl4AI."""
    
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=config.get("rate_limit", 60)
        )
        
        # Configure crawler
        self.crawler_config = CrawlerConfig(
            browser_type=config.get("browser", "chromium"),
            headless=config.get("headless", True),
            verbose=config.get("verbose", False),
            page_timeout=config.get("page_timeout", 30000),
            fetch_timeout=config.get("fetch_timeout", 10000),
            user_agent=config.get("user_agent", "AIDocs/1.0"),
            headers=config.get("headers", {}),
            cookies=config.get("cookies", []),
            proxy=config.get("proxy"),
        )
        
        # Concurrent crawling settings
        self.max_concurrent = config.get("max_concurrent", 10)
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
    
    async def crawl_single(
        self,
        url: str,
        extraction_type: str = "markdown",
        wait_for: Optional[str] = None,
        js_code: Optional[str] = None,
    ) -> dict[str, Any]:
        """Crawl a single URL with specified extraction."""
        
        async with self.semaphore:
            await self.rate_limiter.acquire()
            
            try:
                async with AsyncWebCrawler(config=self.crawler_config) as crawler:
                    # Configure extraction strategy
                    if extraction_type == "structured":
                        strategy = JsonCssExtractionStrategy(
                            schema={
                                "title": "h1, h2",
                                "content": "main, article, .content",
                                "code_blocks": "pre code",
                                "metadata": {
                                    "author": ".author",
                                    "date": ".date, time",
                                    "tags": ".tag, .category",
                                }
                            }
                        )
                    elif extraction_type == "llm":
                        strategy = LLMExtractionStrategy(
                            provider="ollama/llama2",
                            instruction="Extract technical documentation with code examples"
                        )
                    else:
                        strategy = None  # Default markdown extraction
                    
                    # Crawl with options
                    result = await crawler.arun(
                        url=url,
                        extraction_strategy=strategy,
                        wait_for=wait_for,
                        js_code=js_code,
                        word_count_threshold=10,
                        css_selector="main, article, .content, .documentation",
                        excluded_tags=['nav', 'footer', 'header', 'aside', 'script'],
                        cache_mode="enabled",
                    )
                    
                    if result.success:
                        return {
                            "success": True,
                            "url": url,
                            "title": result.metadata.get("title", ""),
                            "content": result.markdown,
                            "html": result.html,
                            "metadata": result.metadata,
                            "links": result.links,
                            "media": result.media,
                        }
                    else:
                        raise CrawlingError(f"Failed to crawl {url}: {result.error}")
                        
            except Exception as e:
                self.logger.error(f"Crawl4AI error for {url}: {e}")
                raise CrawlingError(str(e))
    
    async def crawl_bulk(
        self,
        urls: list[str],
        extraction_type: str = "markdown"
    ) -> list[dict[str, Any]]:
        """Crawl multiple URLs concurrently."""
        
        tasks = [
            self.crawl_single(url, extraction_type)
            for url in urls
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful = []
        failed = []
        
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                failed.append({
                    "url": url,
                    "error": str(result)
                })
            else:
                successful.append(result)
        
        if failed:
            self.logger.warning(f"Failed to crawl {len(failed)} URLs")
        
        return successful
```

### 2. JavaScript Execution Support

```python
class JavaScriptExecutor:
    """Handle complex JavaScript execution for dynamic content."""
    
    def __init__(self):
        self.common_patterns = {
            "spa_navigation": """
                // Wait for SPA navigation
                await new Promise(resolve => {
                    const observer = new MutationObserver(() => {
                        if (document.querySelector('.content-loaded')) {
                            observer.disconnect();
                            resolve();
                        }
                    });
                    observer.observe(document.body, {childList: true, subtree: true});
                    setTimeout(resolve, 5000); // Timeout fallback
                });
            """,
            
            "infinite_scroll": """
                // Load all content via infinite scroll
                let lastHeight = 0;
                while (true) {
                    window.scrollTo(0, document.body.scrollHeight);
                    await new Promise(r => setTimeout(r, 1000));
                    let newHeight = document.body.scrollHeight;
                    if (newHeight === lastHeight) break;
                    lastHeight = newHeight;
                }
            """,
            
            "click_show_more": """
                // Click all "show more" buttons
                const buttons = document.querySelectorAll('[class*="show-more"], [class*="load-more"]');
                for (const button of buttons) {
                    button.click();
                    await new Promise(r => setTimeout(r, 500));
                }
            """,
        }
    
    def get_js_for_site(self, url: str) -> Optional[str]:
        """Get custom JavaScript for specific documentation sites."""
        
        domain = urlparse(url).netloc
        
        # Site-specific JavaScript
        site_js = {
            "docs.python.org": self.common_patterns["spa_navigation"],
            "reactjs.org": self.common_patterns["spa_navigation"],
            "developer.mozilla.org": self.common_patterns["click_show_more"],
            "stackoverflow.com": self.common_patterns["infinite_scroll"],
        }
        
        return site_js.get(domain)
```

### 3. Content Extraction Optimization

```python
class DocumentationExtractor:
    """Optimized extraction for technical documentation."""
    
    def __init__(self):
        self.selectors = {
            # Common documentation selectors
            "content": [
                "main",
                "article",
                ".content",
                ".documentation",
                "#main-content",
                ".markdown-body",
                ".doc-content",
            ],
            
            # Code blocks
            "code": [
                "pre code",
                ".highlight",
                ".code-block",
                ".language-*",
            ],
            
            # Navigation (to extract structure)
            "nav": [
                ".sidebar",
                ".toc",
                "nav",
                ".navigation",
            ],
            
            # Metadata
            "metadata": {
                "title": ["h1", ".title", "title"],
                "description": ["meta[name='description']", ".description"],
                "author": [".author", "meta[name='author']"],
                "version": [".version", ".release"],
                "last_updated": ["time", ".last-updated", ".modified"],
            }
        }
    
    def create_extraction_schema(self, doc_type: str) -> dict:
        """Create extraction schema based on documentation type."""
        
        schemas = {
            "api_reference": {
                "endpoints": "section.endpoint",
                "parameters": ".parameter",
                "responses": ".response",
                "examples": ".example",
            },
            
            "tutorial": {
                "steps": ".step, .tutorial-step",
                "code_examples": "pre code",
                "prerequisites": ".prerequisites",
                "objectives": ".objectives",
            },
            
            "guide": {
                "sections": "h2, h3",
                "content": "p, ul, ol",
                "callouts": ".note, .warning, .tip",
                "related": ".related-links",
            }
        }
        
        base_schema = {
            "title": self.selectors["metadata"]["title"],
            "content": self.selectors["content"],
            "code_blocks": self.selectors["code"],
        }
        
        return {**base_schema, **schemas.get(doc_type, {})}
```

### 4. Caching Layer Integration

```python
class CrawlCache:
    """Intelligent caching for crawled content."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.ttl = 86400  # 24 hours default
    
    async def get_or_crawl(
        self,
        url: str,
        crawler: Crawl4AIProvider,
        force_refresh: bool = False
    ) -> dict[str, Any]:
        """Get from cache or crawl if needed."""
        
        # Generate cache key
        cache_key = f"crawl:{hashlib.md5(url.encode()).hexdigest()}"
        
        # Check cache unless forced refresh
        if not force_refresh:
            cached = await self.cache.get(cache_key)
            if cached:
                self.logger.info(f"Cache hit for {url}")
                return cached
        
        # Crawl and cache
        result = await crawler.crawl_single(url)
        
        # Determine TTL based on content
        ttl = self.calculate_ttl(result)
        
        await self.cache.set(cache_key, result, ttl=ttl)
        
        return result
    
    def calculate_ttl(self, result: dict) -> int:
        """Dynamic TTL based on content characteristics."""
        
        # API docs change less frequently
        if "api" in result["url"].lower():
            return 604800  # 7 days
        
        # Blog posts are static
        if "blog" in result["url"].lower():
            return 2592000  # 30 days
        
        # Tutorials moderate change frequency
        return 259200  # 3 days
```

### 5. Migration from Firecrawl

```python
class CrawlServiceMigrator:
    """Migrate from Firecrawl to Crawl4AI."""
    
    def __init__(self):
        self.firecrawl_provider = FirecrawlProvider()
        self.crawl4ai_provider = Crawl4AIProvider()
    
    async def migrate_with_fallback(
        self,
        url: str,
        prefer_crawl4ai: bool = True
    ) -> dict[str, Any]:
        """Crawl with fallback during migration."""
        
        primary = self.crawl4ai_provider if prefer_crawl4ai else self.firecrawl_provider
        fallback = self.firecrawl_provider if prefer_crawl4ai else self.crawl4ai_provider
        
        try:
            # Try primary provider
            result = await primary.crawl_single(url)
            self.logger.info(f"Successfully crawled with {primary.__class__.__name__}")
            return result
            
        except Exception as e:
            self.logger.warning(f"Primary provider failed: {e}, trying fallback")
            
            try:
                # Try fallback provider
                result = await fallback.crawl_single(url)
                self.logger.info(f"Successfully crawled with fallback")
                return result
                
            except Exception as fallback_error:
                self.logger.error(f"Both providers failed for {url}")
                raise CrawlingError(f"All providers failed: {e}, {fallback_error}")
```

### 6. Performance Benchmarking

```python
class CrawlBenchmark:
    """Benchmark Crawl4AI vs Firecrawl."""
    
    async def run_comparison(self, urls: list[str]) -> dict:
        """Compare performance of both crawlers."""
        
        results = {
            "crawl4ai": {"times": [], "success": 0, "failed": 0},
            "firecrawl": {"times": [], "success": 0, "failed": 0}
        }
        
        # Test Crawl4AI
        for url in urls:
            start = time.time()
            try:
                await self.crawl4ai.crawl_single(url)
                results["crawl4ai"]["success"] += 1
                results["crawl4ai"]["times"].append(time.time() - start)
            except:
                results["crawl4ai"]["failed"] += 1
        
        # Test Firecrawl
        for url in urls:
            start = time.time()
            try:
                await self.firecrawl.crawl_single(url)
                results["firecrawl"]["success"] += 1
                results["firecrawl"]["times"].append(time.time() - start)
            except:
                results["firecrawl"]["failed"] += 1
        
        # Calculate statistics
        for crawler in results:
            times = results[crawler]["times"]
            if times:
                results[crawler]["avg_time"] = np.mean(times)
                results[crawler]["p95_time"] = np.percentile(times, 95)
        
        return results
```

## Integration with Existing System

### 1. Update Bulk Embedder

```python
# src/crawl4ai_bulk_embedder.py
class EnhancedBulkEmbedder:
    def __init__(self):
        # Initialize Crawl4AI instead of Firecrawl
        self.crawler = Crawl4AIProvider(config={
            "max_concurrent": 10,
            "rate_limit": 60,
            "browser": "chromium",
            "headless": True,
        })
        
        # Keep existing embedding and storage logic
        self.embedding_manager = EmbeddingManager()
        self.vector_store = QdrantService()
```

### 2. Update MCP Server

```python
# src/unified_mcp_server.py
@server.tool()
async def crawl_url(url: str, options: dict = {}) -> CrawlResult:
    """Crawl URL using high-performance Crawl4AI."""
    
    crawler = server.get_service("crawler")  # Now Crawl4AI
    
    result = await crawler.crawl_single(
        url=url,
        extraction_type=options.get("extraction_type", "markdown"),
        wait_for=options.get("wait_for"),
        js_code=options.get("js_code"),
    )
    
    return result
```

## Performance Expectations

| Metric | Firecrawl | Crawl4AI | Improvement |
|--------|-----------|----------|-------------|
| Avg crawl time | 2.5s | 0.4s | 6.25x |
| Concurrent requests | 5 | 50 | 10x |
| Cost per 1K pages | $15 | $0 | ∞ |
| JavaScript support | Limited | Full | - |
| Local deployment | No | Yes | - |

## Testing

```python
@pytest.mark.asyncio
async def test_crawl4ai_performance():
    """Test Crawl4AI performance vs Firecrawl."""
    
    urls = [
        "https://docs.python.org/3/library/asyncio.html",
        "https://fastapi.tiangolo.com/tutorial/",
        "https://react.dev/learn",
    ]
    
    # Benchmark both
    crawl4ai_times = []
    firecrawl_times = []
    
    for url in urls:
        # Crawl4AI
        start = time.time()
        await crawl4ai_provider.crawl_single(url)
        crawl4ai_times.append(time.time() - start)
        
        # Firecrawl
        start = time.time()
        await firecrawl_provider.crawl_single(url)
        firecrawl_times.append(time.time() - start)
    
    # Assert Crawl4AI is faster
    assert np.mean(crawl4ai_times) < np.mean(firecrawl_times) / 4
```

## Migration Checklist

- [ ] Install Crawl4AI dependencies
- [ ] Implement Crawl4AIProvider class
- [ ] Add JavaScript execution patterns
- [ ] Create extraction schemas
- [ ] Set up caching integration
- [ ] Implement fallback mechanism
- [ ] Update bulk embedder
- [ ] Update MCP server
- [ ] Run performance benchmarks
- [ ] Update documentation
- [ ] Remove Firecrawl dependency
