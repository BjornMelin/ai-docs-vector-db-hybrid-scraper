# Crawl4AI Rebuild Feature ↔ Test Coverage

| Feature                                    | Tests                                                                                                             |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| Single-page crawl via `crawl_page`         | `tests/unit/services/crawling/test_c4a_provider.py::test_crawl_page_single_url_uses_arun`                         |
| Batch crawl via `crawl_page` + `arun_many` | `test_crawl_page_multiple_urls_calls_arun_many`, `test_crawl_page_with_dispatcher_uses_run_urls`                  |
| BFS deep crawling                          | `test_crawl_deep_bfs_applies_strategy_when_missing`                                                               |
| Best-first keyword streaming               | `test_crawl_best_first_returns_list_when_not_streaming`, `test_crawl_best_first_streaming_returns_async_iterator` |
| Session reuse with shared crawler          | `test_crawl_page_supports_existing_session`                                                                       |
| Markdown presets (Fit Markdown & BM25)     | `tests/unit/services/crawling/test_c4a_presets.py::test_markdown_generator_*`                                     |
| Filter chain composition                   | `tests/unit/services/crawling/test_c4a_presets.py::test_build_filter_chain_combines_filters`                      |
| Adaptive dispatcher preset                 | `tests/unit/services/crawling/test_c4a_presets.py::test_memory_dispatcher_configuration`                          |
| Adapter payload & fit markdown propagation | `tests/unit/services/browser/test_crawl4ai_adapter.py::test_scrape_success`                                       |
| Adapter health & metrics                   | `tests/unit/services/browser/test_crawl4ai_adapter.py::test_health_check`, `test_get_performance_metrics`         |
