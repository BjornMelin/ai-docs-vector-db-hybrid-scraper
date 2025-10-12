---
title: Browser Orchestration
audience: developers
status: active
owner: platform-engineering
last_reviewed: 2025-10-12
---

## Browser Orchestration

The browser subsystem delivers five-tier adaptive scraping. This guide
summarises the architecture, the provider capabilities, and the integration
patterns you need when extending or operating the stack.

## Goals

- Replace monolithic adapters with thin wrappers around maintained SDKs.
- Eliminate backwards-compatibility shims—only the new API surface is supported.
- Keep configuration library-first (`Settings.browser.*`).
- Provide a consistent result schema and error taxonomy regardless of tier.

## Module Layout

```
src/services/browser/
├── __init__.py
├── config.py            # Pydantic settings (BrowserAutomationConfig + per-provider)
├── errors.py            # BrowserProviderError, BrowserRouterError
├── models.py            # BrowserResult, ScrapeRequest, ProviderKind
├── providers/           # Provider implementations and base protocol
├── router.py            # BrowserRouter orchestrating tier selection
└── telemetry.py         # Lightweight metrics recorder
```

## Provider Contract

All providers implement `BrowserProvider`:

- `initialize()` – Acquire runtime resources (Playwright browser, Firecrawl
  client, etc.).
- `close()` – Release resources.
- `run(request)` – Wrapper that enforces deadline measurement and error
  normalization.
- `_failure()` – Helper for constructing `BrowserResult.failure` payloads.

Providers raise `BrowserProviderError` for recoverable failures. The router
converts these into `BrowserRouterError` with an `attempted_providers` trail.

## Tier Overview

| Tier | Provider              | Highlights                                               | When to use                                |
| ---- | --------------------- | -------------------------------------------------------- | ------------------------------------------ |
| T0   | `LightweightProvider` | `httpx` + `trafilatura`, no JS                           | Static documentation, raw files            |
| T1   | `Crawl4AIProvider`    | Crawl4AI `AsyncWebCrawler`, Fit Markdown, wait selectors | Dynamic pages without heavy anti-bot       |
| T2   | `PlaywrightProvider`  | Playwright with maintained stealth plugins               | Rich JS sites, scripted interactions       |
| T3   | `BrowserUseProvider`  | `browser-use` agent + LLM orchestration                  | Agentic flows, bespoke instructions        |
| T4   | `FirecrawlProvider`   | Firecrawl v2 API                                         | Site-wide crawl or anti-bot hardened sites |

Every provider returns a normalized `BrowserResult` containing
`success`, `url`, `title`, `content`, `html`, `metadata`, optional `links`
and `assets`, and an `elapsed_ms` measurement.

## Router Behaviour

`BrowserRouter` accepts a `ScrapeRequest` and:

1. Computes the provider order (auto or forced) via `RouterSettings` heuristics.
2. Enforces per-provider rate limits (`aiolimiter.AsyncLimiter`).
3. Captures deadline budget (`Settings.browser.router.per_attempt_cap_ms`).
4. Records per-tier metrics using `MetricsRecorder`.
5. Returns the first successful `BrowserResult` or raises `BrowserRouterError`.

## Configuration

`Settings.browser` exposes a single `BrowserAutomationConfig` object containing
all provider settings. Example usage:

```python
from src.config.loader import get_settings

settings = get_settings()
router_settings = settings.browser.router
playwright_settings = settings.browser.playwright
```

No other modules define browser configuration—`src/services/browser/config.py`
is the single source of truth.

## Testing Strategy

- Provider unit tests use stubs/mocks of SDK clients (e.g., Crawl4AI, Firecrawl)
  to validate normalization and error handling.
- Router unit tests exercise tier selection, rate limiting, and deadline expiry.
- Integration tests spin up SDKs only when optional dependencies are available;
  fall back to mocked responses otherwise.

## Observability

`telemetry.py` exposes a `MetricsRecorder` for in-memory counters. Each provider
updates the recorder with `record_success`, `record_failure`, and
`record_rate_limited`. Structured logging stays local to the provider modules.

Expose higher-level metrics via:

```python
router = BrowserRouter(...)
await router.initialize()
metrics = router.get_metrics_snapshot()
```

## Public API Surface

- `BrowserRouter.scrape(request: ScrapeRequest) -> BrowserResult`
- `BrowserRouter.initialize()` / `BrowserRouter.cleanup()`
- `UnifiedBrowserManager` provides a facade for callers that require
  initialization management and crawl helpers (`scrape_url`, `crawl_site`).

No legacy modules (`action_schemas`, `anti_detection`, legacy adapters) remain.
Upstream call sites must use the providers and router described here.
