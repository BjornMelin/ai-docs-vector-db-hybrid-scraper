# Local Type Stub Overlay

This directory contains minimal stub definitions for third-party packages that do
not currently ship type information.  Stubs should only model the surface used by
this project and must stay intentionally small so upstream changes are easy to
track.

Maintenance rules:

- Prefer contributing fixes to upstream ``types-`` packages when feasible.
- Keep every stub accompanied by unit or integration coverage that exercises the
  annotated API via the nightly ``verify-types`` automation (`make verify-types`).
- When removing runtime dependencies, delete the matching stubs in the same
  change.

Current modules:

- ``crawl4ai`` – provides ``AsyncWebCrawler`` and ``CacheMode`` plus the
  ``CrawlResult`` payload shape required by the browser adapter.
- ``src.services.crawling`` – exposes the ``crawl_page`` coroutine signature
  relied upon by adapters without requiring the optional orchestration code to
  be installed locally.
- ``src.services.crawling.c4a_presets`` – captures the minimal preset helpers
  (`BrowserOptions`, `base_run_config`, `memory_dispatcher`) used when wiring
  Crawl4AI.
