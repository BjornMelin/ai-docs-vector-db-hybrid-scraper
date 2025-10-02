# Retrieval Stack Compatibility Matrix

The following table tracks the pinned library versions validated for the
library-first retrieval architecture. Update this document whenever dependencies
are bumped through Renovate or manual maintenance windows.

| Component              | Package                               | Version Range        | Resolved (uv.lock) | Notes |
|------------------------|----------------------------------------|----------------------|--------------------|-------|
| Embedding generation   | `fastembed`                            | `>=0.7.3,<0.7.4`     | `0.7.3`            | CPU-optimised embeddings used by default service. |
| Vector store client    | `qdrant-client[fastembed]`             | `>=1.15.1,<1.15.2`   | `1.15.1`           | Provides gRPC/REST client plus FastEmbed integration. |
| Orchestration framework| `langchain`                            | `>=0.3.12,<0.4.0`    | `0.3.27`           | Wrap library usage behind internal adapters to avoid lock-in. |
| LangChain core         | `langchain-core`                       | `>=0.3.76`           | `0.3.76`           | Implicit dependency of `langchain`. |
| LangChain community    | `langchain-community`                  | `>=0.3.12,<0.4.0`    | `0.3.30`           | Houses vector store integrations (e.g., Qdrant). |
| LangChain OpenAI       | `langchain-openai`                     | `>=0.3.33,<1.0.0`    | `0.3.33`           | Required for OpenAI-backed evaluators during regression runs. |

## Maintenance Guidelines

- Use Renovate (see `.github/renovate.json`) to propose updates outside the
  pinned ranges. Each PR must include:
  - Contract test run results
  - RAG regression harness output (`scripts/eval/rag_golden_eval.py`)
  - Performance benchmark summary (latency, recall, cost)
- Update this matrix whenever the ranges change and capture the decision in the
  ADR log.
- If a version bump introduces breaking changes, add migration guidance to the
  developer runbook before merging.
