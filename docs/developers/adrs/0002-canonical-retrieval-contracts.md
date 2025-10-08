# ADR 0002: Canonical Retrieval Contracts and Grouping Metadata Normalization

**Date:** 2025-02-20  
**Status:** Accepted  
**Drivers:** Schema drift between services and MCP tooling, duplicated DTO maintenance overhead, mismatched grouping metadata downstream, missing observability for grouped queries  
**Deciders:** AI Docs Platform Team

## Context

- `SearchRecord` and companion response objects were duplicated across three locations (`src/services/query_processing/models.py`, `src/mcp_tools/models/responses.py`, `src/services/vector_db/types.py`). Divergent fields (`collection` vs `_collection`, legacy `_total_*`)
- created brittle conversions and inconsistent payloads.
- MCP tooling expected `_collection` metadata while orchestrator clients consumed `collection`. Downstream analytics lacked confidence labels and canonical grouping flags, complicating dedup removal and SLAs.
- Server-side grouping required reliable propagation of capability probes and payload metadata so federated merging and MCP outputs stayed aligned.

## Decision

- Introduce `src/contracts/retrieval.py::SearchRecord` as the single authoritative DTO for retrieval responses. Service and MCP layers re-export this contract instead of redefining schemas.
- Normalize grouped result metadata to `collection`, `collection_confidence`, and `collection_priority`, discarding `_collection*` variants. Vector service annotates payloads, federated merging copies fields, and MCP converters preserve them.
- Instrument grouped search outcomes via Prometheus (`grouping_requests_total`, `grouping_latency_seconds`) inside `VectorStoreService` so fallbacks and success ratios are observable.

## Consequences

- Schema drift is eliminated: updating `SearchRecord` automatically propagates to service pipelines, MCP tooling, and CLI helpers. Tests reference the same contract snapshot.
- Grouped-result consumers (federated merges, orchestrator, MCP tools, analytics) now read consistent metadata, enabling removal of bespoke dedup loops and accurate telemetry.
- Telemetry surfaces grouping adoption and latency; alerting and dashboards can react when fallbacks or runtime issues occur.
- Rollout required updating import paths, tests, and payload converters. The vector service now emits `SearchRecord`
  instances directly so downstream layers no longer maintain bespoke adapters.

## Status Notes

- Grouping metrics still need to be wired into operational dashboards, but all CLI and MCP helpers now consume the shared
  `SearchRecord` contract without bespoke `SearchResult` shims.
- Additional contracts (e.g., ingestion and analytics payloads) may migrate into `src/contracts/` over time to ensure consistent reuse.
