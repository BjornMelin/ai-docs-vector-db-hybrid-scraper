# Crawl4AI Provider Usage

The Crawl4AI provider now exposes three asynchronous entry points that map directly to Crawl4AI 0.7.x primitives:

```python
from crawl4ai import CacheMode
from src.services.crawling import crawl_page, crawl_deep_bfs, crawl_best_first
from src.services.crawling.c4a_presets import (
    preset_browser_config,
    base_run_config,
    bfs_run_config,
    best_first_run_config,
)

browser_cfg = preset_browser_config()
run_cfg = base_run_config(cache_mode=CacheMode.BYPASS)

# Single page or batch crawl
result = await crawl_page("https://docs.crawl4ai.com", run_cfg, browser_cfg)

# BFS coverage crawl
bfs_cfg = bfs_run_config(depth=2, base_config=run_cfg)
bfs_results = await crawl_deep_bfs("https://docs.crawl4ai.com", 2, bfs_cfg, browser_cfg)

# Best-first streaming crawl
keywords = ["crawler", "markdown", "deep crawl"]
best_cfg = best_first_run_config(keywords, base_config=run_cfg, stream=True)
async for page in await crawl_best_first("https://docs.crawl4ai.com", keywords, best_cfg, browser_cfg):
    print(page.url)
```

## Sessions and Dynamic Pages

Reuse an `AsyncWebCrawler` when a session or repeated JavaScript execution is required:

```python
from crawl4ai import AsyncWebCrawler

async with AsyncWebCrawler(config=browser_cfg) as crawler:
    session_cfg = run_cfg.clone(session_id="doc_session", js_code="document.querySelector('.load-more')?.click();")
    await crawl_page("https://docs.crawl4ai.com", session_cfg, browser_cfg, crawler=crawler)
    await crawl_page("https://docs.crawl4ai.com/next", session_cfg, browser_cfg, crawler=crawler)
```

### Adaptive Dispatcher

Use the `memory_dispatcher` preset to process a URL list with backpressure safeguards:

```python
from src.services.crawling.c4a_presets import memory_dispatcher

urls = ["https://docs.crawl4ai.com/core/deep-crawling", "https://docs.crawl4ai.com/core/fit-markdown"]
adaptive_dispatcher = memory_dispatcher(max_session_permit=8)
results = await crawl_page(urls, run_cfg, browser_cfg, dispatcher=adaptive_dispatcher)
```

The presets also expose helpers for filter chains, link previews, and markdown tuning so downstream services (FastAPI, MCP) can pick the right configuration without redeclaring Crawl4AI internals.
