# Crawl4AI Provider Rebuild Mapping

Legacy logic has been replaced with thin wrappers around Crawl4AI 0.7.x primitives. The table maps the new surfaces to the canonical documentation/examples that back them.

| Implementation surface                                      | Crawl4AI feature                                                                 | Reference snippet                                                                                   |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `src/services/crawling.c4a_provider.crawl_page`             | `AsyncWebCrawler.arun()` / `arun_many()` for single and batched crawls           | https://raw.githubusercontent.com/unclecode/crawl4ai/main/docs/md_v2/api/async-webcrawler.md        |
| `c4a_provider.crawl_deep_bfs`                               | `BFSDeepCrawlStrategy` with optional `FilterChain`                               | https://docs.crawl4ai.com/core/deep-crawling/                                                       |
| `c4a_provider.crawl_best_first`                             | `BestFirstCrawlingStrategy` + `KeywordRelevanceScorer` streaming support         | https://docs.crawl4ai.com/core/deep-crawling/                                                       |
| `c4a_presets.base_run_config`                               | Default `CrawlerRunConfig` with `DefaultMarkdownGenerator` Fit Markdown pipeline | https://raw.githubusercontent.com/unclecode/crawl4ai/main/docs/md_v2/core/fit-markdown.md           |
| `c4a_presets.build_filter_chain`                            | URL/domain/content filters via `FilterChain` helpers                             | https://docs.crawl4ai.com/core/deep-crawling/                                                       |
| `c4a_presets.best_first_run_config` / `bfs_run_config`      | Strategy presets for best-first and BFS runs                                     | https://docs.crawl4ai.com/core/deep-crawling/                                                       |
| `c4a_presets.memory_dispatcher`                             | `MemoryAdaptiveDispatcher` + `RateLimiter` + `CrawlerMonitor`                    | https://raw.githubusercontent.com/unclecode/crawl4ai/main/docs/md_v2/advanced/multi-url-crawling.md |
| `c4a_presets.link_preview_config`                           | `LinkPreviewConfig` with `score_links=True`                                      | https://raw.githubusercontent.com/unclecode/crawl4ai/main/docs/md_v2/core/link-media.md             |
| `c4a_provider` session hooks                                | Session reuse via `session_id`, `js_code`, `wait_for` in `CrawlerRunConfig`      | https://docs.crawl4ai.com/core/page-interaction/                                                    |
| Adapter bulk path (`crawl_page` with sequence + dispatcher) | `AsyncWebCrawler.arun_many()` streaming & batching                               | https://raw.githubusercontent.com/unclecode/crawl4ai/main/docs/md_v2/api/arun_many.md               |
