---
id: remaining_tasks.risk_register
last_reviewed: 2025-07-02
status: draft
---

| Risk ID | Description | Impact | Likelihood | Mitigation / Linked Tasks | Owner | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | Automated tests cannot run due to missing pytest extras, preventing regression detection. | High | High | QA-01 (install extras, restore pytest) | QA Lead | Open | `pytest … -q` failure (missing plugins) |
| R2 | Observability middleware never attached; production incidents would lack metrics/traces. | High | Medium | OPS-01 (wire Prometheus/OTel) | Observability Eng | Open | `src/api/app_factory.py:320-380` |
| R3 | Browser automation disabled by missing `browser-use` dependency; tiered crawling relies on fallback only. | High | Medium | BRW-01..BRW-03 | Automation Eng | Open | `src/services/browser/browser_use_adapter.py:22-59` |
| R4 | Circuit breaker logic unused; upstream outages propagate directly to clients. | Medium-High | Medium | INF-03 | Platform Eng | Open | `src/services/circuit_breaker/modern.py` vs dependencies |
| R5 | Unified config stores secrets as plain strings; no secret management or rotation story. | High | Medium | INF-01 | Platform Eng | Open | `rg "SecretStr" src/config/settings.py` → none |
