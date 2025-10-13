---
title: AI Docs Vector DB Hybrid Scraper
audience: general
status: active
owner: documentation
last_reviewed: 2025-03-13
---

## AI Docs Vector DB Hybrid Scraper

The platform ingests documentation from the web, enriches it with browser
automation, stores hybrid embeddings in Qdrant, and exposes retrieval workflows
through a unified API surface. This page summarises the core capabilities and
points to the detailed guides for users, developers, and operators.

## Platform Highlights

- **Hybrid search pipeline** – FastEmbed dense + sparse embeddings,
  LangChain-managed chunking, HyDE augmentation, and reranking
- **Five-tier browser automation** – Selects the lightest tool that can load and
  extract a page
- **Unified configuration** – Single Pydantic settings module controls both simple
  and enterprise deployments
- **Observability and security** – Built-in metrics, logging, authentication, and
  rate limiting

## Architecture Snapshot

```mermaid
graph LR
    Source[Documentation sources] --> Browser
    Browser[Browser automation] --> Extract[Extractors]
    Extract --> Chunker[Chunking + metadata]
    Chunker --> Embeddings[Embedding providers]
    Embeddings --> Qdrant[(Qdrant)]
    Qdrant --> Rerank[Reranking]
    Rerank --> API[Unified API]
```

- **Ingestion** – A tiered browser manager fetches pages, extractors clean them,
  and `chunk_to_documents` routes each payload through the LangChain splitter
  matrix (Markdown/HTML/code/JSON/token-aware paths) before canonical metadata is
  attached.
- **Processing** – Dense and sparse embedding providers run in parallel; HyDE
  expansion and caching keep retrieval responsive.
- **Retrieval** – LangChain's `QdrantVectorStore` persists FastEmbed dense +
  sparse embeddings in Qdrant; the API orchestrates
  filtering, reranking, and response formatting (REST, MCP, browser integrations).

## Choose Your Guide

=== "Users"

    - [Quick Start](users/quick-start.md)
    - [Configuration Management](users/configuration-management.md)
    - [Search & Retrieval](users/search-and-retrieval.md)
    - [Troubleshooting](users/troubleshooting.md)

=== "Developers"

    - [Developer Hub](developers/index.md)
    - [Setup & Configuration](developers/setup-and-configuration.md)
    - [Architecture & Orchestration](developers/architecture-and-orchestration.md)
    - [Browser Orchestration](developers/browser-orchestration.md)
    - [Agentic Orchestration](developers/agentic-orchestration.md)
    - [Cache & Performance](developers/cache-and-performance.md)
    - [GPU Acceleration](developers/gpu-acceleration.md)
    - [Platform Operations](developers/platform-operations.md)
    - [API & Contracts](developers/api-and-contracts.md)
    - [Contributing & Testing](developers/contributing-and-testing.md)

=== "Operators"

    - [Operator Hub](operators/index.md)
    - [Deployment](operators/deployment.md)
    - [Operations Runbook](operators/operations.md)
    - [Monitoring](operators/monitoring.md)

## Additional Resources

- [Query Processing Metrics](observability/query_processing_metrics.md)
- [Evaluation Harness Playbook](testing/evaluation-harness.md)
- [Security Checklist](security/security-checklist.md)
- [Technical Debt Log](TECH_DEBT.md)

Questions or improvements? Open an issue or contribute a pull request so that
these docs stay up to date.
