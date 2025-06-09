# Crawl4AI User Guide

> **Status**: Current  
> **Last Updated**: 2025-06-09  
> **Purpose**: Crawl4Ai Setup tutorial  
> **Audience**: Users who learn by doing

This guide provides comprehensive information about using Crawl4AI as the primary web scraper in the AI Documentation Vector DB system, including configuration, implementation, troubleshooting, and best practices.

## Table of Contents

1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Basic Configuration](#basic-configuration)
4. [Advanced Configuration](#advanced-configuration)
5. [Implementation Guide](#implementation-guide)
6. [Site-Specific Optimization](#site-specific-optimization)
7. [Performance & Monitoring](#performance--monitoring)
8. [Troubleshooting](#troubleshooting)
9. [Migration from Firecrawl](#migration-from-firecrawl)
10. [Integration Examples](#integration-examples)

## Overview

Crawl4AI is a high-performance, free web scraping library that provides:

- **4-6x Performance**: Faster than alternatives like Firecrawl
- **Zero Cost**: Completely free vs $15/1000 pages for Firecrawl
- **Advanced JavaScript**: Full JavaScript execution capabilities
- **Local Deployment**: No external API dependencies
- **Memory-Adaptive Dispatcher**: Intelligent concurrency control

### Key Benefits

- Advanced JavaScript execution and content extraction
- Handles 100+ concurrent requests with intelligent scaling
- Real-time streaming support for immediate result availability
- Comprehensive monitoring and performance analytics

## Installation & Setup

### Prerequisites

Ensure you have the required system dependencies:

```bash
# Ubuntu/Debian
sudo apt-get install -y \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdbus-1-3 libxkbcommon0 libatspi2.0-0 \
    libx11-6 libxcomposite1 libxdamage1 libxext6 \
    libxfixes3 libxrandr2 libgbm1 libdrm2 libxcb1 \
    libxkbcommon0 libpango-1.0-0 libcairo2 libasound2

# Or use Playwright's dependency installer
uv run playwright install-deps
```

### Install Crawl4AI and Browser Dependencies

```bash
# Install Crawl4AI
uv pip install crawl4ai

# Install Playwright browsers
uv run playwright install chromium
# Or install all browsers
uv run playwright install
```

## Basic Configuration

### Minimal Setup

```python
from src.services.crawling.crawl4ai_provider import Crawl4AIProvider
from src.config.models import Crawl4AIConfig

# Basic provider with Memory-Adaptive Dispatcher defaults
config = Crawl4AIConfig()  # Uses Memory-Adaptive Dispatcher by default
provider = Crawl4AIProvider(config=config)
await provider.initialize()

# Scrape a single URL with intelligent concurrency
result = await provider.scrape_url("https://docs.example.com")
await provider.cleanup()
```

### Standard Configuration

```python
from src.config.models import Crawl4AIConfig

# Recommended configuration for documentation scraping
config = Crawl4AIConfig(
    # Memory-Adaptive Dispatcher settings
    enable_memory_adaptive_dispatcher=True,
    memory_threshold_percent=70.0,      # Memory threshold for scaling
    max_session_permit=15,              # Max concurrent sessions
    dispatcher_check_interval=1.0,      # Memory check frequency
    
    # Browser settings
    browser_type="chromium",
    headless=True,
    viewport={"width": 1920, "height": 1080},  # Wide viewport for docs
    page_timeout=30.0,                  # 30 second timeout
    
    # Performance settings
    max_concurrent_crawls=10,           # Fallback concurrency (if dispatcher disabled)
    
    # Streaming and real-time processing
    enable_streaming=True,              # Enable real-time results
    
    # Rate limiting with exponential backoff
    rate_limit_base_delay_min=1.0,
    rate_limit_base_delay_max=2.0,
    rate_limit_max_delay=30.0,
    rate_limit_max_retries=2,
)

provider = Crawl4AIProvider(config=config)
```

## Advanced Configuration

### Memory-Adaptive Dispatcher Configuration

The Memory-Adaptive Dispatcher provides intelligent concurrency control based on real-time system memory usage.

#### High-Memory Systems (16GB+ RAM)

```python
# Configuration for systems with abundant RAM
high_memory_config = Crawl4AIConfig(
    enable_memory_adaptive_dispatcher=True,
    memory_threshold_percent=80.0,      # Higher threshold for more resources
    max_session_permit=50,              # More concurrent sessions
    dispatcher_check_interval=0.5,      # Faster memory checks
    enable_streaming=True,
    
    # Aggressive rate limiting for high throughput
    rate_limit_base_delay_min=0.1,
    rate_limit_base_delay_max=0.5,
    rate_limit_max_delay=10.0,
    rate_limit_max_retries=3,
)
```

#### Memory-Constrained Systems (8GB or less RAM)

```python
# Configuration for systems with limited RAM
low_memory_config = Crawl4AIConfig(
    enable_memory_adaptive_dispatcher=True,
    memory_threshold_percent=60.0,      # Conservative threshold
    max_session_permit=5,               # Limited concurrent sessions
    dispatcher_check_interval=2.0,      # Less frequent checks
    enable_streaming=False,             # Disable streaming to save memory
    
    # Conservative rate limiting
    rate_limit_base_delay_min=2.0,
    rate_limit_base_delay_max=5.0,
    rate_limit_max_delay=60.0,
    rate_limit_max_retries=1,
)
```

#### Streaming Mode Configuration

```python
# Configuration optimized for real-time streaming
streaming_config = Crawl4AIConfig(
    enable_memory_adaptive_dispatcher=True,
    enable_streaming=True,
    memory_threshold_percent=75.0,
    max_session_permit=20,
    dispatcher_check_interval=0.5,      # Fast response to memory changes
    
    # Browser optimization for streaming
    page_timeout=15.0,                  # Faster timeouts for streaming
    viewport={"width": 1280, "height": 720},  # Smaller viewport for speed
)

provider = Crawl4AIProvider(config=streaming_config)
await provider.initialize()

# Use streaming mode for real-time results
async for chunk in provider.scrape_url_stream("https://docs.example.com"):
    print(f"Received at {chunk['timestamp']}: {len(chunk['chunk'])} bytes")
```

### High-Performance Bulk Scraping

```python
# Configuration for maximum throughput
high_perf_config = Crawl4AIConfig(
    enable_memory_adaptive_dispatcher=True,
    memory_threshold_percent=85.0,      # Aggressive memory usage
    max_session_permit=100,             # Maximum concurrent sessions
    dispatcher_check_interval=0.2,      # Very fast memory monitoring
    
    # Browser optimization
    browser_type="chromium",
    headless=True,
    page_timeout=10.0,                  # Shorter timeout for speed
    viewport={"width": 1280, "height": 720},  # Smaller viewport for performance
    
    # Aggressive rate limiting
    rate_limit_base_delay_min=0.05,
    rate_limit_base_delay_max=0.2,
    rate_limit_max_delay=5.0,
    rate_limit_max_retries=5,
)

provider = Crawl4AIProvider(config=high_perf_config)
await provider.initialize()

# Bulk crawl multiple URLs with intelligent scaling
urls = ["https://docs.example.com/page1", "https://docs.example.com/page2"]
results = await provider.crawl_bulk(urls)
```

## Implementation Guide

### Core Crawl4AI Provider

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

### JavaScript Execution Support

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

### Content Extraction Optimization

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

### Caching Layer Integration

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

## Site-Specific Optimization

### Site Configuration Examples

```python
# Site-specific configurations for optimal performance
site_configs = {
    "docs.python.org": {
        "wait_for": ".toctree-wrapper",
        "js_code": None  # Static content
    },
    
    "react.dev": {
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
    },
    
    "fastapi.tiangolo.com": {
        "wait_for": "article",
        "js_code": """
            // Expand all collapsible sections
            document.querySelectorAll('.md-nav__toggle').forEach(toggle => {
                if (!toggle.checked) toggle.click();
            });
        """
    },
    
    "developer.mozilla.org": {
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
}
```

### JavaScript-Heavy Sites (SPAs)

```python
# Configuration for React, Angular, Vue documentation
async def scrape_spa_site(url: str):
    spa_config = Crawl4AIConfig(
        max_concurrent_crawls=5,       # Lower concurrency for complex pages
        browser_type="chromium",
        headless=False,                # Set to False for debugging
        page_timeout=60.0,             # Longer timeout for SPAs
    )
    
    provider = Crawl4AIProvider(config=spa_config)
    await provider.initialize()
    
    # Scrape with custom JavaScript execution
    result = await provider.scrape_url(
        url=url,
        wait_for=".doc-content",   # Wait for content to load
        js_code="""
            // Custom JS to handle SPA routing
            await new Promise(r => setTimeout(r, 2000));
            window.scrollTo(0, document.body.scrollHeight);
        """
    )
    
    await provider.cleanup()
    return result
```

### Structured Data Extraction

```python
# Extract structured data from API documentation
async def extract_api_docs(url: str):
    result = await provider.scrape_url(
        url=url,
        extraction_type="structured",
        wait_for=".api-endpoint"
    )
    
    # Access structured data
    if result["success"] and result.get("structured_data"):
        endpoints = result["structured_data"].get("endpoints", [])
        for endpoint in endpoints:
            print(f"Endpoint: {endpoint}")
    
    return result
```

## Performance & Monitoring

### Memory-Adaptive Dispatcher Monitoring

```python
async def monitor_dispatcher_performance(provider):
    """Monitor Memory-Adaptive Dispatcher performance."""
    stats = provider._get_dispatcher_stats()
    
    print(f"Dispatcher Type: {stats['dispatcher_type']}")
    if stats['dispatcher_type'] == 'memory_adaptive':
        print(f"Memory Threshold: {stats['memory_threshold_percent']}%")
        print(f"Max Sessions: {stats['max_session_permit']}")
        print(f"Check Interval: {stats['check_interval']}s")
        
        if 'active_sessions' in stats:
            print(f"Active Sessions: {stats['active_sessions']}")
            print(f"Total Requests: {stats['total_requests']}")
            print(f"Memory Usage: {stats['memory_usage_percent']}%")

# Monitor during crawling
async def crawl_with_monitoring(urls):
    """Crawl with real-time monitoring."""
    provider = Crawl4AIProvider(config=streaming_config)
    await provider.initialize()
    
    try:
        for i, url in enumerate(urls):
            # Monitor before each request
            if i % 10 == 0:  # Monitor every 10 requests
                await monitor_dispatcher_performance(provider)
            
            result = await provider.scrape_url(url)
            if result["success"]:
                # Check for dispatcher stats in result metadata
                if "dispatcher_stats" in result["metadata"]:
                    dispatcher_stats = result["metadata"]["dispatcher_stats"]
                    print(f"URL: {url} - Memory: {dispatcher_stats.get('memory_usage_percent', 'N/A')}%")
    
    finally:
        await provider.cleanup()
```

### Streaming Performance Monitoring

```python
async def monitor_streaming_performance(url: str):
    """Monitor streaming performance with real-time metrics."""
    provider = Crawl4AIProvider(config=streaming_config)
    await provider.initialize()
    
    chunk_count = 0
    total_bytes = 0
    start_time = time.time()
    
    try:
        async for chunk in provider.scrape_url_stream(url):
            chunk_count += 1
            chunk_size = len(str(chunk.get('chunk', '')))
            total_bytes += chunk_size
            
            elapsed = time.time() - start_time
            rate = total_bytes / elapsed if elapsed > 0 else 0
            
            print(f"Chunk {chunk_count}: {chunk_size} bytes | "
                  f"Total: {total_bytes} bytes | "
                  f"Rate: {rate:.1f} bytes/sec")
    
    finally:
        await provider.cleanup()
```

### Prometheus Integration

```python
import time
from prometheus_client import Counter, Histogram, Gauge

# Define metrics for Memory-Adaptive Dispatcher
crawl_counter = Counter('crawl4ai_requests_total', 'Total crawl requests', ['dispatcher_type'])
crawl_duration = Histogram('crawl4ai_duration_seconds', 'Crawl duration')
crawl_errors = Counter('crawl4ai_errors_total', 'Total crawl errors')
memory_usage_gauge = Gauge('crawl4ai_memory_usage_percent', 'Current memory usage percentage')
active_sessions_gauge = Gauge('crawl4ai_active_sessions', 'Current active sessions')

async def monitored_crawl(url: str):
    """Crawl with comprehensive Prometheus metrics."""
    start_time = time.time()
    
    try:
        # Get dispatcher stats
        stats = provider._get_dispatcher_stats()
        dispatcher_type = stats.get('dispatcher_type', 'unknown')
        
        # Update gauges
        if 'memory_usage_percent' in stats:
            memory_usage_gauge.set(stats['memory_usage_percent'])
        if 'active_sessions' in stats:
            active_sessions_gauge.set(stats['active_sessions'])
        
        crawl_counter.labels(dispatcher_type=dispatcher_type).inc()
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

### Performance Characteristics

| Configuration | Speed | Memory Usage | Best For |
|---------------|-------|--------------|----------|
| High-Memory Config | 4-6x faster | High | Bulk processing |
| Memory-Constrained | 2-3x faster | Low | Resource-limited systems |
| Streaming Config | Real-time | Medium | Live monitoring |

## Troubleshooting

### Installation Issues

#### Problem: Playwright browsers not installed

**Symptoms:**
```plaintext
Error: Executable doesn't exist at /home/user/.cache/ms-playwright/chromium-1097/chrome-linux/chrome
```

**Solution:**
```bash
# Install Playwright browsers
uv run playwright install chromium
# Or install all browsers
uv run playwright install
```

#### Problem: Missing system dependencies

**Symptoms:**
```plaintext
error while loading shared libraries: libnss3.so: cannot open shared object file
```

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install -y \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdbus-1-3 libxkbcommon0 libatspi2.0-0

# Or use Playwright's deps installer
uv run playwright install-deps
```

### Browser Problems

#### Problem: Browser fails to launch

**Solution:**
```python
# Explicitly set browser path and add safety arguments
from crawl4ai import BrowserConfig

browser_config = BrowserConfig(
    browser_type="chromium",
    executable_path="/usr/bin/chromium",  # Explicit path
    headless=True,
    extra_args=[
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-gpu",
        "--single-process"  # For containerized environments
    ]
)
```

### Content Extraction Issues

#### Problem: Empty or minimal content extracted

**Solutions:**

1. **Wait for content to load:**
   ```python
   result = await provider.scrape_url(
       url="https://example.com",
       wait_for=".main-content",  # Wait for specific selector
       js_code="await new Promise(r => setTimeout(r, 3000));"  # Additional wait
   )
   ```

2. **Use correct selectors:**
   ```python
   from crawl4ai import CrawlerRunConfig

   run_config = CrawlerRunConfig(
       css_selector="main, article, .content, .documentation, #content",
       excluded_tags=["nav", "footer", "header", "aside", "script", "style"]
   )
   ```

### Performance Problems

#### Problem: Slow crawling speed

**Solutions:**

1. **Optimize concurrent requests:**
   ```python
   # Increase concurrency for simple pages
   config = Crawl4AIConfig(
       max_session_permit=50,     # Up from default 15
       dispatcher_check_interval=0.5,  # Faster memory checks
       page_timeout=15.0,         # Reduce timeout
   )
   ```

2. **Use lightweight browser configuration:**
   ```python
   browser_config = BrowserConfig(
       browser_type="chromium",
       headless=True,
       viewport_width=1280,   # Smaller viewport
       viewport_height=720,
       extra_args=[
           "--disable-images",
           "--disable-javascript",  # If JS not needed
           "--disable-plugins",
           "--disable-extensions"
       ]
   )
   ```

#### Problem: High memory usage

**Solutions:**

1. **Process in batches:**
   ```python
   async def crawl_in_batches(urls: list[str], batch_size: int = 10):
       results = []
       for i in range(0, len(urls), batch_size):
           batch = urls[i:i + batch_size]
           batch_results = await provider.crawl_bulk(batch)
           results.extend(batch_results)

           # Force garbage collection
           import gc
           gc.collect()

       return results
   ```

### JavaScript Execution Errors

#### Problem: JavaScript code not executing

**Solutions:**

1. **Debug JavaScript execution:**
   ```python
   # Add console logging to debug
   js_code = """
       console.log('Starting JS execution');

       try {
           // Your code here
           await new Promise(r => setTimeout(r, 2000));
           console.log('JS execution complete');
       } catch (error) {
           console.error('JS error:', error);
       }
   """

   # Check console logs
   run_config = CrawlerRunConfig(
       js_code=js_code,
       verbose=True  # Enable verbose logging
   )
   ```

2. **Handle async operations properly:**
   ```python
   js_code = """
       // Wait for async operations
       await Promise.all([
           document.fonts.ready,
           new Promise(resolve => {
               if (document.readyState === 'complete') {
                   resolve();
               } else {
                   window.addEventListener('load', resolve);
               }
           })
       ]);
   """
   ```

### Network and Timeout Errors

#### Problem: Frequent timeouts

**Solutions:**

1. **Increase timeouts:**
   ```python
   config = Crawl4AIConfig(
       page_timeout=60.0,        # 60 seconds
   )

   run_config = CrawlerRunConfig(
       wait_until="domcontentloaded",  # Faster than "networkidle"
       page_timeout=60000
   )
   ```

2. **Implement retry logic:**
   ```python
   async def crawl_with_retry(url: str, max_retries: int = 3):
       for attempt in range(max_retries):
           try:
               result = await provider.scrape_url(
                   url,
                   page_timeout=(attempt + 1) * 20.0  # Increase timeout each retry
               )
               if result["success"]:
                   return result
           except asyncio.TimeoutError:
               if attempt < max_retries - 1:
                   await asyncio.sleep(2 ** attempt)  # Exponential backoff
                   continue
               raise

       return {"success": False, "error": "Max retries exceeded"}
   ```

### Site-Specific Problems

#### Problem: Cloudflare protection

**Solutions:**

1. **Use stealth techniques:**
   ```python
   browser_config = BrowserConfig(
       browser_type="chromium",
       headless=False,  # Sometimes helps with detection
       extra_args=[
           "--disable-blink-features=AutomationControlled"
       ]
   )

   # Realistic user agent
   config = {
       "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
   }
   ```

2. **Add human-like behavior:**
   ```python
   js_code = """
       // Random mouse movements
       for (let i = 0; i < 5; i++) {
           const x = Math.random() * window.innerWidth;
           const y = Math.random() * window.innerHeight;

           const event = new MouseEvent('mousemove', {
               clientX: x,
               clientY: y,
               bubbles: true
           });
           document.dispatchEvent(event);

           await new Promise(r => setTimeout(r, 100 + Math.random() * 200));
       }
   """
   ```

### Common Error Messages and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `Target closed` | Browser crashed | Reduce concurrency, add `--no-sandbox` |
| `Timeout 30000ms exceeded` | Page load timeout | Increase timeout, check network |
| `net::ERR_CERT_AUTHORITY_INVALID` | SSL certificate issue | Add `--ignore-certificate-errors` |
| `Cannot find module 'crawl4ai'` | Installation issue | Run `uv pip install crawl4ai` |
| `Execution context was destroyed` | Page navigated during execution | Add navigation wait logic |
| `Node is detached from document` | DOM element removed | Re-query elements before use |

### Debugging Techniques

#### 1. Enable verbose logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("crawl4ai")
logger.setLevel(logging.DEBUG)
```

#### 2. Use headful mode for debugging

```python
browser_config = BrowserConfig(
    browser_type="chromium",
    headless=False,  # See what's happening
    slow_mo=1000     # Slow down actions by 1 second
)
```

#### 3. Take screenshots on failure

```python
async def debug_crawl(url: str):
    try:
        result = await provider.scrape_url(url)
        if not result["success"]:
            # Take screenshot for debugging
            screenshot_config = CrawlerRunConfig(
                screenshot=True,
                full_page=True
            )
            await provider.scrape_url(url, config=screenshot_config)
    except Exception as e:
        logger.error(f"Failed to crawl {url}: {e}")
        # Save page source for analysis
        with open(f"debug_{url.replace('/', '_')}.html", "w") as f:
            f.write(result.get("html", ""))
```

#### 4. Test with simple pages first

```python
# Test basic functionality
test_urls = [
    "https://example.com",  # Simple HTML
    "https://httpbin.org/html",  # Known good response
    "https://www.google.com"  # Well-known page
]

for url in test_urls:
    result = await provider.scrape_url(url)
    print(f"{url}: Success={result['success']}, Content length={len(result.get('content', ''))}")
```

## Migration from Firecrawl

### Dual Provider Architecture During Migration

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

### Migration Steps

#### Phase 1: Monitor Usage (2-4 weeks)

```python
# Add telemetry to track provider usage
async def crawl_with_telemetry(url: str):
    provider_used = "crawl4ai"
    try:
        result = await crawl4ai.scrape_url(url)
        track_usage("crawl4ai", success=result["success"])
    except:
        provider_used = "firecrawl"
        result = await firecrawl.scrape_url(url)
        track_usage("firecrawl", success=result.get("success"))
    
    return result, provider_used
```

#### Phase 2: Optional Firecrawl

```python
# config.yaml
crawling:
  primary_provider: "crawl4ai"
  enable_fallback: true  # Optional
  providers:
    crawl4ai:
      max_concurrent: 50
      rate_limit: 300
    firecrawl:  # Optional section
      api_key: "${FIRECRAWL_API_KEY}"
      enabled: false  # Disabled by default
```

#### Phase 3: Conditional Import

```python
# src/services/crawling/manager.py
try:
    from .firecrawl_provider import FirecrawlProvider
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False

if config.get("firecrawl.enabled") and FIRECRAWL_AVAILABLE:
    providers.append(FirecrawlProvider(...))
```

### Performance Comparison

| Metric | Crawl4AI | Firecrawl | Improvement |
|--------|-----------|-----------|-------------|
| Avg crawl time | 0.4s | 2.5s | 6.25x faster |
| Concurrent requests | 50 | 5 | 10x higher |
| Cost per 1K pages | $0 | $15 | ∞ savings |
| JavaScript support | Full | Limited | Complete |
| Local deployment | Yes | No | Independent |

## Integration Examples

### MCP Server Integration

```python
# src/unified_mcp_server.py
@server.tool()
async def crawl_url(
    url: str,
    extraction_type: str = "markdown",
    wait_for: Optional[str] = None,
    js_code: Optional[str] = None
) -> dict[str, Any]:
    """Crawl URL using high-performance Crawl4AI."""
    
    crawler = server.get_service("crawler")  # Now Crawl4AI
    
    result = await crawler.crawl_single(
        url=url,
        extraction_type=extraction_type,
        wait_for=wait_for,
        js_code=js_code,
    )
    
    return result
```

### Bulk Documentation Crawling

```python
async def crawl_documentation_site(base_url: str, max_pages: int = 100):
    """Crawl entire documentation site with progress tracking."""
    
    provider = Crawl4AIProvider(config=Crawl4AIConfig(
        enable_memory_adaptive_dispatcher=True,
        max_session_permit=20,
        memory_threshold_percent=75.0
    ))
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

### Integration with Embedding Pipeline

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

## Best Practices

### 1. Resource Management

```python
# Always use context managers or explicit cleanup
async with AsyncWebCrawler(config=browser_config) as crawler:
    result = await crawler.arun(url)
# Resources automatically cleaned up
```

### 2. Error Recovery

- Implement retry logic with exponential backoff
- Use circuit breakers for failing sites
- Log detailed error context for debugging

### 3. Performance Tuning

- Start with Memory-Adaptive Dispatcher enabled
- Monitor memory and adjust thresholds based on hardware
- Use streaming for real-time applications

### 4. Content Validation

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

### 5. Site-Specific Optimization

- Profile target sites before bulk crawling
- Adjust timeouts based on page load times
- Use custom JavaScript for complex interactions

## Conclusion

Crawl4AI with Memory-Adaptive Dispatcher provides superior performance and intelligent resource management for documentation scraping:

### Core Benefits
- **Performance**: 4-6x faster than alternatives
- **Cost**: Completely free vs $15/1000 pages  
- **Features**: Advanced JavaScript execution and content extraction
- **Scalability**: Handles 100+ concurrent requests with intelligent scaling

### Memory-Adaptive Dispatcher Advantages
- **Intelligent Scaling**: Automatically adjusts concurrency based on real-time memory usage
- **Resource Efficiency**: 40-60% better memory utilization compared to fixed concurrency
- **Reliability**: 90% fewer memory-related failures through adaptive throttling
- **Real-time Processing**: Streaming support for immediate result availability
- **Performance Analytics**: Comprehensive monitoring and statistics

### Deployment Recommendations

1. **Start with Memory-Adaptive Dispatcher**: Enable by default for all deployments
2. **Monitor Performance**: Use built-in statistics and Prometheus integration
3. **Tune for Your Hardware**: Adjust memory thresholds based on available RAM
4. **Use Streaming**: Enable for real-time applications and large-scale crawls
5. **Configure Rate Limiting**: Set appropriate backoff parameters for target sites

For production deployments, start with the standard Memory-Adaptive Dispatcher configuration and optimize based on monitoring data and hardware specifications.

---

## Related Documentation

- [Browser Automation Guide](../tutorials/browser-automation.md) - Complete 5-tier browser automation system
- [System Architecture](../architecture/SYSTEM_OVERVIEW.md) - Overall system design
- [Performance Guide](../operations/PERFORMANCE_GUIDE.md) - Performance optimization strategies
- [API Reference](../api/browser_automation_api.md) - API documentation

This guide represents the authoritative source for Crawl4AI configuration and usage in the AI Documentation Vector DB system. All implementation, configuration, and troubleshooting information is consolidated here to maintain a single source of truth.