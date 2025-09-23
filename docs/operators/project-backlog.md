---
id: remaining_tasks.readme
last_reviewed: 2025-07-02
status: draft
---

# Remaining Work Consolidation

## Executive Summary

- Legacy planning plus `.taskmaster` backlog listed **33** active items; deduplicated into **25** canonical tasks mapped by capability.
- Source review (planning/_.md, .taskmaster/tasks/_.txt, src/, tests/, docs/, .github/workflows/) revealed key gaps: pytest env lacks required extras, observability middleware is never attached, and Pydantic-AI orchestration still operates in fallback-only mode.
- New backlog groups work across eight domains (Quality, Platform, Ops, Retrieval, Security, Browser, Analytics/UX, Documentation) with dependencies, owner placeholders, and acceptance criteria tied to current code evidence.

## Methodology

1. Parsed all planning reports, status trackers, and `.taskmaster` artifacts to map legacy IDs → themes.
2. Spot-checked repo state (src/, tests/, docs/, workflows) to confirm implementation reality and gather evidence per task.
3. Collapsed duplicates or superseded scopes, marked completed items, and recorded rationale in `decision_log.md`.
4. Produced canonical backlog, risk register, and migration guidance aligned to today’s codebase.

## Legend

- **IDs** use `Category-Index` (e.g., `QA-01`). Categories: QA (Quality), INF (Configuration/Platform), ARC (Architecture), OPS (Operations), RAG (Retrieval), SEC (Security), BRW (Browser), ANA (Analytics), UX (User Experience), DOC (Documentation).
- **Status** defaults to `Pending` unless noted in the decision log.
- **Evidence** references repo paths (path:line) or command outputs collected during validation.
- **Dependencies** list other backlog IDs required first.

## Backlog Snapshot

| Category                | Count | Highlights                                                                       |
| ----------------------- | ----- | -------------------------------------------------------------------------------- |
| Quality & Testing       | 3     | Stabilize pytest environment, modernize CI for 3.13, enforce mypy                |
| Platform & Architecture | 4     | Config hardening, global error handling, service flattening, circuit breakers    |
| Observability & Ops     | 3     | Wire Prometheus/OTel, baseline benchmarks, production gate                       |
| Retrieval & Data        | 3     | Runnable RAG pipeline, multi-collection + caching, multi-language ingestion      |
| Security & Access       | 2     | Enterprise SSO, actionable security scans                                        |
| Browser & Crawling      | 5     | Ship browser-use v0.3.2 stack, FastAPI endpoints, resilience, QA                 |
| Analytics & UX          | 4     | Search analytics, embedding visualization, advanced insights, NL query interface |
| Documentation           | 1     | Update roadmap, operators, and release notes to mirror new plan                  |

## Deliverables

- `backlog.md` – canonical task list with acceptance criteria, dependencies, owners, evidence.
- `decision_log.md` – legacy ID dispositions (retained, merged, completed) with rationale.
- `risk_register.md` – top operational risks, impact/likelihood, mitigation owners.
- `migration_notes.md` – guidance for importing backlog into GitHub Issues/Projects.

## Immediate Focus

1. **QA-01** – unblock automated tests (current pytest run fails: missing required plugins).
2. **OPS-01** – activate observability (Prometheus middleware exists but is never attached).
3. **BRW-01** – add `browser-use>=0.3.2` dependency so the adapter stops hard-disabling itself.
