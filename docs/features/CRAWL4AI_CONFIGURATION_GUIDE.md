# Crawl4AI Configuration Guide

This guide provides comprehensive configuration examples and best practices for using Crawl4AI as the primary web scraper in the AI Documentation Vector DB system.

## Overview

Crawl4AI is a high-performance, free web scraping library that provides:

- 4-6x faster performance compared to alternatives
- Advanced JavaScript execution capabilities
- Intelligent content extraction
- Zero cost (open source)
- Local deployment options

## Basic Configuration

### Minimal Setup

```python
from src.services.crawling.crawl4ai_provider import Crawl4AIProvider

# Basic provider with defaults
provider = Crawl4AIProvider()
await provider.initialize()

# Scrape a single URL
result = await provider.scrape_url("https://docs.example.com")
await provider.cleanup()
```

### Standard Configuration

```python
# Recommended configuration for documentation scraping
provider = Crawl4AIProvider(config={
    "max_concurrent": 10,      # Concurrent requests
    "rate_limit": 60,          # Requests per minute
    "browser": "chromium",     # Browser engine
    "headless": True,          # Run headless
    "viewport_width": 1920,    # Wide viewport for docs
    "viewport_height": 1080,
    "page_timeout": 30000,     # 30 second timeout
    "user_agent": "Mozilla/5.0 (compatible; AIDocs/1.0)"
})
```

## Advanced Configuration Examples

### High-Performance Bulk Scraping

```python
# Configuration for maximum throughput
high_perf_config = {
    "max_concurrent": 50,      # Maximum concurrent crawls
    "rate_limit": 300,         # 5 requests per second
    "browser": "chromium",
    "headless": True,
    "page_timeout": 20000,     # Shorter timeout for speed
    "viewport_width": 1280,    # Smaller viewport for performance
    "viewport_height": 720,
}

provider = Crawl4AIProvider(config=high_perf_config)
await provider.initialize()

# Bulk crawl multiple URLs
urls = ["https://docs.example.com/page1", "https://docs.example.com/page2"]
results = await provider.crawl_bulk(urls)
```

### JavaScript-Heavy Sites (SPAs)

```python
# Configuration for React, Angular, Vue documentation
spa_config = {
    "max_concurrent": 5,       # Lower concurrency for complex pages
    "browser": "chromium",
    "headless": False,         # Set to False for debugging
    "page_timeout": 60000,     # Longer timeout for SPAs
}

provider = Crawl4AIProvider(config=spa_config)
await provider.initialize()

# Scrape with custom JavaScript execution
result = await provider.scrape_url(
    url="https://angular.io/docs",
    wait_for=".doc-content",   # Wait for content to load
    js_code="""
        // Custom JS to handle Angular routing
        await new Promise(r => setTimeout(r, 2000));
        window.scrollTo(0, document.body.scrollHeight);
    """
)
```

### Structured Data Extraction

```python
# Extract structured data from API documentation
result = await provider.scrape_url(
    url="https://api.example.com/docs",
    extraction_type="structured",
    wait_for=".api-endpoint"
)

# Access structured data
if result["success"] and result.get("structured_data"):
    endpoints = result["structured_data"].get("endpoints", [])
    for endpoint in endpoints:
        print(f"Endpoint: {endpoint}")
```

## Site-Specific Configurations

### Python Documentation

```python
# Optimized for docs.python.org
python_docs_config = {
    "url": "https://docs.python.org/3/library/",
    "wait_for": ".toctree-wrapper",
    "extraction_type": "markdown",
    "js_code": None  # Auto-detected for python.org
}
```

### FastAPI Documentation

```python
# FastAPI with sidebar navigation
fastapi_config = {
    "url": "https://fastapi.tiangolo.com/",
    "wait_for": "article",
    "js_code": """
        // Expand all collapsible sections
        document.querySelectorAll('.md-nav__toggle').forEach(toggle => {
            if (!toggle.checked) toggle.click();
        });
    """
}
```

### React Documentation

```python
# React docs with client-side routing
react_config = {
    "url": "https://react.dev/learn",
    "wait_for": "main",
    "js_code": """
        // Wait for React hydration
        await new Promise(resolve => {
            if (document.querySelector('[data-hydrated="true"]')) {
                resolve();
            } else {
                const observer = new MutationObserver((mutations, obs) => {
                    if (document.querySelector('[data-hydrated="true"]')) {
                        obs.disconnect();
                        resolve();
                    }
                });
                observer.observe(document.body, {
                    attributes: true,
                    childList: true,
                    subtree: true
                });
            }
        });
    """
}
```

### MDN Web Docs

```python
# MDN with lazy-loaded content
mdn_config = {
    "url": "https://developer.mozilla.org/en-US/docs/Web",
    "wait_for": ".main-page-content",
    "js_code": """
        // Click all "show more" buttons
        const buttons = document.querySelectorAll('[aria-label*="Show"]');
        for (const button of buttons) {
            button.click();
            await new Promise(r => setTimeout(r, 100));
        }
    """
}
```

## Performance Optimization

### Caching Configuration

```python
from src.services.cache.manager import CacheManager
from src.services.crawling.crawl4ai_provider import CrawlCache

# Setup caching layer
cache_manager = CacheManager(config)
crawl_cache = CrawlCache(cache_manager)

# Use cached crawling
result = await crawl_cache.get_or_crawl(
    url="https://docs.example.com",
    crawler=provider,
    force_refresh=False  # Use cache if available
)
```

### Memory Optimization

```python
# Configuration for memory-constrained environments
memory_optimized_config = {
    "max_concurrent": 3,       # Low concurrency
    "browser": "chromium",
    "headless": True,
    "viewport_width": 1024,    # Smaller viewport
    "viewport_height": 768,
    "page_timeout": 15000,     # Shorter timeout
}

# Use for large-scale crawls
provider = Crawl4AIProvider(config=memory_optimized_config)
```

### Rate Limiting

```python
from src.services.rate_limiter import RateLimiter

# Custom rate limiter for API-heavy sites
rate_limiter = RateLimiter(
    max_calls=30,      # 30 requests
    time_window=60     # per minute
)

provider = Crawl4AIProvider(
    config={"max_concurrent": 5},
    rate_limiter=rate_limiter
)
```

## Error Handling and Retry Logic

```python
async def crawl_with_retry(url: str, max_retries: int = 3):
    """Crawl with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            result = await provider.scrape_url(url)
            if result["success"]:
                return result
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                await asyncio.sleep(wait_time)
            else:
                raise
    
    return {"success": False, "error": "Max retries exceeded"}
```

## Bulk Crawling Patterns

### Parallel Crawling with Progress

```python
from rich.progress import Progress
import asyncio

async def crawl_documentation_site(base_url: str, max_pages: int = 100):
    """Crawl entire documentation site with progress tracking."""
    
    provider = Crawl4AIProvider(config={
        "max_concurrent": 10,
        "rate_limit": 60
    })
    await provider.initialize()
    
    # Get site map or initial URLs
    initial_result = await provider.scrape_url(base_url)
    urls_to_crawl = extract_documentation_urls(initial_result)
    
    results = []
    with Progress() as progress:
        task = progress.add_task("Crawling docs...", total=len(urls_to_crawl))
        
        # Process in batches
        batch_size = 10
        for i in range(0, len(urls_to_crawl), batch_size):
            batch = urls_to_crawl[i:i + batch_size]
            batch_results = await provider.crawl_bulk(batch)
            results.extend(batch_results)
            progress.advance(task, len(batch))
    
    await provider.cleanup()
    return results
```

### Incremental Crawling

```python
async def incremental_crawl(base_url: str, last_crawl_date: datetime):
    """Only crawl new or updated pages since last crawl."""
    
    provider = Crawl4AIProvider()
    await provider.initialize()
    
    # Get sitemap or changelog
    sitemap_result = await provider.scrape_url(f"{base_url}/sitemap.xml")
    
    # Filter URLs by last modified date
    new_urls = filter_urls_by_date(sitemap_result, last_crawl_date)
    
    if new_urls:
        results = await provider.crawl_bulk(new_urls)
        return results
    
    return []
```

## Integration with Embedding Pipeline

```python
from src.services.embeddings.manager import EmbeddingManager
from src.chunking import EnhancedChunker

async def crawl_and_embed(url: str):
    """Complete pipeline: crawl -> chunk -> embed."""
    
    # 1. Crawl content
    provider = Crawl4AIProvider()
    await provider.initialize()
    
    result = await provider.scrape_url(url)
    
    if result["success"]:
        # 2. Chunk content
        chunker = EnhancedChunker()
        chunks = chunker.chunk_text(
            result["content"],
            chunk_size=1600,
            chunk_overlap=200
        )
        
        # 3. Generate embeddings
        embedding_manager = EmbeddingManager(config)
        embeddings = await embedding_manager.generate_embeddings(chunks)
        
        # 4. Store in vector database
        await store_in_qdrant(url, chunks, embeddings)
    
    await provider.cleanup()
```

## Monitoring and Metrics

```python
import time
from prometheus_client import Counter, Histogram

# Define metrics
crawl_counter = Counter('crawl4ai_requests_total', 'Total crawl requests')
crawl_duration = Histogram('crawl4ai_duration_seconds', 'Crawl duration')
crawl_errors = Counter('crawl4ai_errors_total', 'Total crawl errors')

async def monitored_crawl(url: str):
    """Crawl with Prometheus metrics."""
    start_time = time.time()
    
    try:
        crawl_counter.inc()
        result = await provider.scrape_url(url)
        
        if result["success"]:
            crawl_duration.observe(time.time() - start_time)
            return result
        else:
            crawl_errors.inc()
            return result
            
    except Exception as e:
        crawl_errors.inc()
        raise
```

## Best Practices

### 1. **Resource Management**

```python
# Always use context managers or explicit cleanup
async with AsyncWebCrawler(config=browser_config) as crawler:
    result = await crawler.arun(url)
# Resources automatically cleaned up
```

### 2. **Error Recovery**

- Implement retry logic with exponential backoff
- Use circuit breakers for failing sites
- Log detailed error context for debugging

### 3. **Performance Tuning**

- Start with conservative concurrency (10)
- Monitor memory and CPU usage
- Adjust based on target site characteristics

### 4. **Content Validation**

```python
def validate_crawl_result(result: dict) -> bool:
    """Validate crawled content quality."""
    if not result.get("success"):
        return False
    
    content = result.get("content", "")
    
    # Check minimum content length
    if len(content) < 100:
        return False
    
    # Check for error pages
    error_indicators = ["404", "not found", "error", "forbidden"]
    if any(indicator in content.lower() for indicator in error_indicators):
        return False
    
    return True
```

### 5. **Site-Specific Optimization**

- Profile target sites before bulk crawling
- Adjust timeouts based on page load times
- Use custom JavaScript for complex interactions

## Migration from Firecrawl

```python
# Fallback pattern during migration
async def crawl_with_fallback(url: str):
    """Try Crawl4AI first, fallback to Firecrawl if needed."""
    try:
        # Primary: Crawl4AI
        result = await crawl4ai_provider.scrape_url(url)
        if result["success"]:
            return result
    except Exception as e:
        logger.warning(f"Crawl4AI failed: {e}, trying Firecrawl")
    
    # Fallback: Firecrawl
    if firecrawl_provider:
        return await firecrawl_provider.scrape_url(url)
    
    return {"success": False, "error": "All providers failed"}
```

## Conclusion

Crawl4AI provides superior performance and flexibility for documentation scraping. Key advantages:

- **Performance**: 4-6x faster than alternatives
- **Cost**: Completely free vs $15/1000 pages
- **Features**: Advanced JavaScript execution and content extraction
- **Scalability**: Handles 50+ concurrent requests efficiently

For production deployments, start with the standard configuration and optimize based on monitoring data.
