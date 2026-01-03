# Embeddings Test Suite Guidelines

This document describes fixtures, fakes, and invariants for the embeddings service tests.

## Fixtures (tests/unit/services/embeddings/conftest.py)

- `ai_operation_calls`: Captures telemetry via patched `record_ai_operation` across import sites.
- Deterministic tokenizer: Simple encoder to make token counts stable in unit tests that assert usage.
- Time control (recommended): Use `time-machine` with a function-scoped fixture for freeze + shift. This keeps latency math and backoff deterministic.
- HTTP (DI-first): Prefer `httpx.MockTransport` injected via factory fixtures. Use `pytest-httpx` only for black-box clients when DI is impossible.

Scopes: Default to function scope. Use module scope for expensive, immutable resources only. Avoid autouse except for truly global concerns (e.g., global seed/time freeze in a dedicated profile).

## Fakes & Stubs

- Provider stubs: `_TestProvider` and `_StubProvider` return deterministic vectors and expose model/dimensions. Failures are injected via `raise_error` attributes.
- Cache: `_FakeCache` mimics both `embedding_cache` attribute and direct `get_embedding`/`set_embedding` API variants.
- Selection: `_FakeSelectionEngine` returns precomputed analysis + recommendation to keep tests fast and local.
- OpenAI client stub: `_StubAsyncClient` provides `embeddings`, `files`, and `batches` namespaces patched per-test.

## Invariants

- Token estimate ≈ chars / `chars_per_token` (default 4).
- Batch-to-call mapping = ceil(n / batch_size); embeddings preserve input order.
- Cost (USD) = `tokens * cost_per_token` (0 for local FastEmbed).
- Cache keys = (`text`, `provider`, `model`, `dimensions`); single-text hits short-circuit with `cache_hit=True`.
- Budget policy: warn ≥ `budget_warning_threshold`; block when projected spend exceeds budget.
- Sparse payload schema: objects expose `indices` and `values` arrays; missing fields raise `EmbeddingServiceError`.

## Async & Time

- Pytest config now lives under `[tool.pytest.ini_options]` in `pyproject.toml`; `asyncio_mode = auto` reduces boilerplate—add explicit `@pytest.mark.asyncio` where helpful.
- Time freeze: prefer `time-machine` for async compatibility and performance; keep function-scoped to avoid bleed between tests.

## Coverage & Profiles

- Unit-fast (PR): run embeddings unit scope with `--maxfail=1` and `--cov=src/services/embeddings`.
- Full/nightly: include slow/integration/performance (optional). Combine coverage across jobs; upload HTML.

## HTTP Best Practices

- DI-first: pass transports/clients via fixtures. Avoid hand-written network stubs beyond adapter boundaries.
- Recording (VCR): opt-in only; scrub secrets; set expiries; avoid in default PR jobs.

## Parallel Safety

- Use `tmp_path`/`tmp_path_factory` for filesystem; avoid global singletons. Cache tests verify idempotent set/get and no races under `asyncio.gather`.
