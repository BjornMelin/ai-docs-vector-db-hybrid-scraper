# Embedding Manager Modular Breakdown

This refactor splits the legacy `EmbeddingManager` into composable services within
`src/services/embeddings/manager/`:

- `providers.py` – provider registry and lifecycle (initialize/cleanup, tier mapping,
  reranker management) for OpenAI/FastEmbed.
- `selection.py` – text analysis, smart recommendation, and model scoring logic.
- `usage.py` – usage accounting and budget guardrails (UsageStats, reports).
- `pipeline.py` – orchestration of embedding generation, caching, metrics, and
  result assembly using collaborators.
- `__init__.py` – thin facade exposing `EmbeddingManager`, `QualityTier`, and
  shared dataclasses while wiring the above components together.

Additional shared dataclasses and enums (`TextAnalysis`, `EmbeddingMetricsContext`)
will live alongside their owning modules to keep ownership clear.
