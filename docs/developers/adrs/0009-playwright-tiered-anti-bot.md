# ADR 0009: Tiered Playwright Anti-Bot Hardening

**Date:** 2025-10-05  
**Status:** Accepted  
**Drivers:** Cloudflare/DataDome failures on priority targets, need for library-first undetected Playwright integration, requirement to capture bot challenges across tiers  
**Deciders:** AI Docs Platform Team

## Context

Earlier we introduced a lean Playwright stack (ADR 0008) built around
`tf_playwright_stealth`. That simplification eliminated bespoke wrappers but left
hard targets (Cloudflare, Akamai, DataDome, PerimeterX) with failure rates above
35%. Recent research and field reports recommend combining patched,
anti-detect Playwright builds with managed residential proxies and hosted captcha
solvers to reach production-success rates above 90%
[[ZenRows 2025](https://www.zenrows.com/blog/playwright-cloudflare-bypass),
[ScrapeOps 2025](https://scrapeops.io/playwright-web-scraping-playbook/nodejs-playwright-bypass-datadome/),
Firecrawl deep research Oct 2025].

We must honour the FINAL-ONLY policy (no legacy fallbacks) while keeping the
library-first mandate. The new stack therefore needs tiered execution, library
replacements rather than custom stealth scripts, and observability of challenge
outcomes to drive future tuning.

## Decision Framework

| Criteria           | Weight  | Adopt tiered undetected stack | Retain baseline-only stack |
| ------------------ | ------- | ----------------------------- | -------------------------- |
| Leverage           | 0.35    | 4.9                           | 3.6                        |
| Value              | 0.30    | 4.8                           | 3.8                        |
| Maintenance        | 0.25    | 4.4                           | 4.7                        |
| Adaptability       | 0.10    | 4.7                           | 3.9                        |
| **Weighted Total** | **1.0** | **4.73**                      | **4.02**                   |

**Decision:** Adopt the tiered, library-first stack. Integrate Rebrowser’s
patched Playwright runtime, ScrapeOps-compatible proxy injection, and a
CapMonster-backed solver workflow, while retaining the baseline stealth tier as
Tier 0 to satisfy KISS/DRY requirements.

## Decision

1. Extend `PlaywrightConfig` with structured tier definitions (`PlaywrightTierConfig`),
   proxy settings (`PlaywrightProxySettings`), and captcha settings
   (`PlaywrightCaptchaSettings`). Inject defaults so existing configs remain valid.
2. Refactor `PlaywrightAdapter` to:
   - Lazily manage separate Playwright runtimes for baseline and undetected tiers.
   - Apply proxy credentials per tier and surface runtime/tier metadata.
   - Drive captcha resolution via the CapMonster HTTP API when configured,
     marking challenge outcomes (`detected`, `solved`, `none`).
   - Return structured results containing tier/runtime metadata so upstream routing
     and monitoring can act on escalation outcomes.
3. Enhance `MetricsRegistry`/`BrowserAutomationMonitor` with the
   `*_browser_challenges_total` counter and propagate challenge outcome labels to
   Prometheus.
4. Update unit suites covering the adapter, automation router, and metrics to
   assert tier metadata and challenge accounting. Expose tiered config through
   `src/config/__init__.py` so application code can reference the models.

## Consequences

- **Positive:** Higher success rates on hardened sites (residential/mobile proxies
  plus undetected fingerprints) without bespoke stealth code. Automatic
  escalation to CapMonster reduces manual retries and operator time.
- **Positive:** Metrics now expose challenge frequency and runtime usage, feeding
  downstream dashboards and alerting.
- **Neutral:** More third-party dependencies (Rebrowser, proxy credentials,
  captcha API) require key management and version pinning.
- **Negative:** Runtime management is more complex—each tier stands up its own
  Playwright runtime; upgrades must track both upstream Playwright and Rebrowser
  release cadences.

## Alternatives Considered

1. **Baseline-only stealth with aggressive header de-duplication (Score 4.02/5).**

   - Pros: Minimal dependencies, zero new operational footprint.
   - Cons: Still fails on modern WAFs; requires custom heuristics to keep pace
     with bot-detection updates.

2. **Full managed scraping API (ZenRows/ScrapeOps Smart Proxy) (Score 4.18/5).**

   - Pros: Offloads proxy, fingerprint, and captcha handling entirely.
   - Cons: Higher variable cost, opaque internals, reduced flexibility for
     on-platform instrumentation; conflicts with library-first policy when
     built-in Playwright stack suffices for ≥80% of use cases.

3. **Local browser pool with rotating user data dirs (Score 3.71/5).**
   - Pros: Keeps stack self-hosted, avoids captcha service costs.
   - Cons: High maintenance; still needs proxy diversity and advanced fingerprint
     patching—effectively recreating Rebrowser with more toil.
