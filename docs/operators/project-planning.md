---
id: remaining_tasks.plan
last_reviewed: 2025-07-02
status: draft
---

# Delivery Plan (Condensed)

This summary highlights the highest-impact themes, the order we intend to tackle them, and the top
risks. Detailed acceptance criteria live in the table below; create GitHub issues referencing the
ID column when you schedule work.

## Phase Outline

1. **Stabilise the Toolchain** – QA-01 · QA-02 · QA-03
2. **Harden Platform Foundations** – INF-01 · INF-02 · INF-03 · OPS-01
3. **Restore Browser & Retrieval Features** – BRW-01/02/03 · RAG-01/02
4. **Security & Analytics Enhancements** – SEC-01/02 · ANA-01/02 · UX-01

Each phase assumes the prior one is functionally complete (tests green, docs updated, release notes
prepared).

## Task Roll-up

| ID    | Summary                               | Acceptance Criteria (abridged)                                   | Dependencies |
| ----- | ------------------------------------- | --------------------------------------------------------------- | ------------ |
| QA-01 | Pytest extras + green run             | Install extras, restore pass, update testing docs               | —            |
| QA-02 | CI Python 3.13                        | Matrix includes 3.13, coverage uploaded, release flow updated   | QA-01        |
| QA-03 | CI mypy gate                          | Add job, fail on errors, document workflow                      | QA-01        |
| INF-01| Secrets + env alias hygiene           | Use `SecretStr`, provide env templates, document rollback       | QA-01        |
| INF-02| Unified API error handling            | Global handlers, structured logs, tests for 4xx/5xx             | INF-01       |
| INF-03| Circuit breaker wiring                | Inject manager, tests for open/half-open, tuning guidance       | INF-01       |
| OPS-01| Prometheus + OTEL enabled             | Middleware mounted, scrape documented, CI smoke test            | INF-03       |
| OPS-02| Baseline performance benchmarks       | Deterministic harness + dashboard                               | OPS-01       |
| OPS-03| Release gate automation               | Release workflow fails without QA/docs approvals                | OPS-01, QA-02|
| RAG-01| Activate LangGraph RAG pipeline       | Production toggle, tests, docs                                  | OPS-01       |
| RAG-02| Multi-collection guidance             | Strategy published + caching policy implemented                 | RAG-01       |
| BRW-01| Upgrade browser-use dependency        | Add dependency, update adapter tests                            | QA-01        |
| BRW-02| Browser orchestration endpoints       | REST endpoints + RBAC + docs                                    | BRW-01       |
| BRW-03| Browser resilience & metrics          | Chaos tests, retry/backoff, metrics emitted                     | BRW-01       |
| SEC-01| Enterprise SSO                        | OAuth/OIDC integration + tests + docs                           | INF-01       |
| SEC-02| Actionable security scans             | CI scans with fail-on-high + remediation workflow               | QA-02        |
| ANA-01| Search analytics dashboard            | Dashboard shipped, documented                                   | OPS-01       |
| ANA-02| Embedding visualisation tooling       | Visual tool shipped + docs                                      | ANA-01       |
| UX-01 | Natural language query interface      | UX flow implemented + docs + telemetry                          | RAG-01       |
| DOC-01| Documentation sync                    | Major docs refreshed post-changes                               | All          |

## Risks (Top 3)

| Risk | Impact | Mitigation                                           |
| ---- | ------ | ----------------------------------------------------- |
| R1   | Tests stay red, blocking releases      | Prioritise QA-01, add CI guard that fails quickly               |
| R2   | No observability in production         | Complete OPS-01 before feature work, add CI smoke for `/metrics`|
| R3   | Browser stack remains disabled         | Upgrade dependency (BRW-01) before building orchestration       |

Update the table when priorities shift. Keep the long-form decision log in version control for
auditors, but run projects from this shorter view.
