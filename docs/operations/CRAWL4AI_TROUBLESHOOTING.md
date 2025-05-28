# Crawl4AI Troubleshooting Guide

This guide helps diagnose and resolve common issues when using Crawl4AI for web scraping in the AI Documentation Vector DB system.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Browser and Playwright Problems](#browser-and-playwright-problems)
3. [Content Extraction Issues](#content-extraction-issues)
4. [Performance Problems](#performance-problems)
5. [JavaScript Execution Errors](#javascript-execution-errors)
6. [Memory and Resource Issues](#memory-and-resource-issues)
7. [Network and Timeout Errors](#network-and-timeout-errors)
8. [Site-Specific Problems](#site-specific-problems)
9. [Debugging Techniques](#debugging-techniques)

## Installation Issues

### Problem: Playwright browsers not installed

**Symptoms:**
```
Error: Executable doesn't exist at /home/user/.cache/ms-playwright/chromium-1097/chrome-linux/chrome
```

**Solution:**
```bash
# Install Playwright browsers
uv run playwright install chromium
# Or install all browsers
uv run playwright install
```

### Problem: Missing system dependencies

**Symptoms:**
```
error while loading shared libraries: libnss3.so: cannot open shared object file
```

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install -y \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdbus-1-3 libxkbcommon0 libatspi2.0-0 \
    libx11-6 libxcomposite1 libxdamage1 libxext6 \
    libxfixes3 libxrandr2 libgbm1 libdrm2 libxcb1 \
    libxkbcommon0 libpango-1.0-0 libcairo2 libasound2

# Or use Playwright's deps installer
uv run playwright install-deps
```

## Browser and Playwright Problems

### Problem: Browser fails to launch

**Symptoms:**
```python
Error: Failed to launch chromium because executable doesn't exist
```

**Solution:**
```python
# Explicitly set browser path
from crawl4ai import BrowserConfig

browser_config = BrowserConfig(
    browser_type="chromium",
    executable_path="/usr/bin/chromium",  # Explicit path
    headless=True
)
```

### Problem: Browser crashes on startup

**Symptoms:**
```
Protocol error (Browser.getVersion): Target closed
```

**Solutions:**

1. **Increase browser resources:**
```python
browser_config = BrowserConfig(
    browser_type="chromium",
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

2. **Check system resources:**
```bash
# Check available memory
free -h

# Check disk space
df -h

# Monitor during crawl
htop
```

## Content Extraction Issues

### Problem: Empty or minimal content extracted

**Symptoms:**
- `content` field contains only 1-2 characters
- Missing expected text from page

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

3. **Check if content is in iframes:**
```python
# Custom JS to extract iframe content
js_code = """
    const iframes = document.querySelectorAll('iframe');
    for (const iframe of iframes) {
        try {
            const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
            console.log('Iframe content:', iframeDoc.body.innerText);
        } catch (e) {
            console.log('Cannot access iframe:', e);
        }
    }
"""
```

### Problem: Structured extraction returns empty

**Solutions:**

1. **Verify schema matches page structure:**
```python
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

# Test selectors first
test_schema = {
    "title": "h1",
    "content": "article",
    "test": "*"  # Match everything to debug
}

strategy = JsonCssExtractionStrategy(schema=test_schema)
```

2. **Use browser DevTools to find selectors:**
```python
# Run with headless=False to debug
browser_config = BrowserConfig(
    browser_type="chromium",
    headless=False  # See browser
)
```

## Performance Problems

### Problem: Slow crawling speed

**Solutions:**

1. **Optimize concurrent requests:**
```python
# Increase concurrency for simple pages
config = {
    "max_concurrent": 50,  # Up from default 10
    "rate_limit": 300,     # 5 per second
    "page_timeout": 15000, # Reduce timeout
}
```

2. **Disable unnecessary features:**
```python
run_config = CrawlerRunConfig(
    screenshot=False,           # Don't take screenshots
    pdf=False,                 # Don't generate PDFs
    extract_media=False,       # Skip media extraction
    cache_mode="disabled"      # Disable cache for speed
)
```

3. **Use lightweight browser configuration:**
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

### Problem: High memory usage

**Solutions:**

1. **Limit concurrent crawls:**
```python
config = {
    "max_concurrent": 3,  # Reduce concurrency
}
```

2. **Process in batches:**
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

## JavaScript Execution Errors

### Problem: JavaScript code not executing

**Symptoms:**
- Dynamic content not loading
- SPA pages showing loading state

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

### Problem: Infinite scroll not working

**Solution:**
```python
js_code = """
    let previousHeight = 0;
    let attempts = 0;
    const maxAttempts = 10;
    
    while (attempts < maxAttempts) {
        window.scrollTo(0, document.body.scrollHeight);
        await new Promise(r => setTimeout(r, 2000));
        
        const currentHeight = document.body.scrollHeight;
        if (currentHeight === previousHeight) {
            console.log('No more content to load');
            break;
        }
        
        previousHeight = currentHeight;
        attempts++;
    }
    
    console.log(`Scrolled ${attempts} times, final height: ${previousHeight}`);
"""
```

## Memory and Resource Issues

### Problem: Memory leaks during long crawls

**Solutions:**

1. **Implement proper cleanup:**
```python
async def crawl_with_cleanup(urls: list[str]):
    provider = Crawl4AIProvider(config)
    
    try:
        await provider.initialize()
        
        # Process URLs in chunks
        chunk_size = 100
        for i in range(0, len(urls), chunk_size):
            chunk = urls[i:i + chunk_size]
            await provider.crawl_bulk(chunk)
            
            # Periodic cleanup
            if i % 500 == 0:
                await provider.cleanup()
                await provider.initialize()
    
    finally:
        await provider.cleanup()
```

2. **Monitor memory usage:**
```python
import psutil
import logging

async def monitored_crawl(url: str):
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    result = await provider.scrape_url(url)
    
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = final_memory - initial_memory
    
    if memory_increase > 100:  # MB
        logging.warning(f"High memory usage increase: {memory_increase}MB for {url}")
    
    return result
```

## Network and Timeout Errors

### Problem: Frequent timeouts

**Solutions:**

1. **Increase timeouts:**
```python
config = {
    "page_timeout": 60000,    # 60 seconds
    "fetch_timeout": 30000,   # 30 seconds for initial fetch
}

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
                page_timeout=(attempt + 1) * 20000  # Increase timeout each retry
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

### Problem: SSL/TLS errors

**Solution:**
```python
browser_config = BrowserConfig(
    browser_type="chromium",
    extra_args=[
        "--ignore-certificate-errors",
        "--ignore-certificate-errors-spki-list"
    ]
)
```

## Site-Specific Problems

### Problem: Cloudflare protection

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

### Problem: Login-required content

**Solution:**
```python
# Use cookies from authenticated session
browser_config = BrowserConfig(
    browser_type="chromium",
    headless=True,
    storage_state="auth_state.json"  # Save/load auth state
)

# Or inject cookies
run_config = CrawlerRunConfig(
    js_code="""
        document.cookie = 'session_id=abc123; path=/';
        document.cookie = 'auth_token=xyz789; path=/';
    """
)
```

## Debugging Techniques

### 1. Enable verbose logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("crawl4ai")
logger.setLevel(logging.DEBUG)
```

### 2. Use headful mode for debugging

```python
browser_config = BrowserConfig(
    browser_type="chromium",
    headless=False,  # See what's happening
    slow_mo=1000     # Slow down actions by 1 second
)
```

### 3. Take screenshots on failure

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

### 4. Test with simple pages first

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

### 5. Monitor network traffic

```python
# Log all network requests
js_code = """
    const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
            console.log(`Network: ${entry.name} - ${entry.duration}ms`);
        }
    });
    observer.observe({ entryTypes: ['resource'] });
"""
```

## Common Error Messages and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `Target closed` | Browser crashed | Reduce concurrency, add `--no-sandbox` |
| `Timeout 30000ms exceeded` | Page load timeout | Increase timeout, check network |
| `net::ERR_CERT_AUTHORITY_INVALID` | SSL certificate issue | Add `--ignore-certificate-errors` |
| `Cannot find module 'crawl4ai'` | Installation issue | Run `uv pip install crawl4ai` |
| `Execution context was destroyed` | Page navigated during execution | Add navigation wait logic |
| `Node is detached from document` | DOM element removed | Re-query elements before use |

## Getting Help

If you encounter issues not covered in this guide:

1. Check Crawl4AI GitHub issues: https://github.com/unclecode/crawl4ai/issues
2. Enable debug logging and collect error details
3. Test with a minimal reproducible example
4. Include system information (OS, Python version, Crawl4AI version)

Remember: Most issues can be resolved by:
- Adjusting timeouts
- Using proper wait conditions
- Implementing retry logic
- Monitoring resource usage