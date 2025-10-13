# Dependency Modernization Execution Plan (Final Rewrite)

## Snapshot

- **Scope:** Perform a single, atomic refactor that replaces every legacy integration with the final implementation. No feature flags, no incremental rollout, and no backward compatibility.
- **Targets:** Direct OpenAI SDK usage, ragas ≥0.3 evaluator, NumPy 2.3.x, lxml 6.x, aiohttp 3.13.x, xformers 0.0.32.post2, Torch GPU extras, end-to-end validation harnesses, refreshed test suites, and updated documentation.
- **Outcome:** Repository contains only the new code path; all deprecated modules, wrappers, and tests are deleted in the same change set.

## Mandates

1. **Direct integrations only:** Import official SDKs or libraries directly where needed; avoid bespoke adapter layers unless they provide critical shared logic.
2. **No legacy fallback:** Remove all feature flags, compatibility shims, deprecated modules, and unused configuration immediately.
3. **Deterministic validation:** Establish CPU/GPU harnesses and regression datasets to guarantee reproducibility after dependency upgrades.
4. **Single merge:** Execute the entire rewrite in one branch/PR; final commit must leave the repository in a fully linted, typed, and tested state.
5. **Documentation parity:** Update operator and developer guides concurrently so they only describe the new implementation.

## Workstream Topology

```
WS1 (Direct OpenAI) ─┐
                     ├─► WS4 (Repo Cleanup & Tests) ─► WS5 (Docs & Dependencies)
WS2 (Ragas v0.3) ────┘
WS3 (Validation Harnesses) ──────────────────────────▲
```

## Workstream Detail

### WS1 – Direct OpenAI SDK Rewrite

- Remove `src/infrastructure/clients/openai_client.py`, DI providers, and any legacy wrappers.
- Refactor all services (embeddings, HYDE, observability, CLI) to call `openai.AsyncOpenAI` / `openai.OpenAI` directly via the modern Responses and Embeddings APIs. Reference: [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses).
- Replace `chat.completions.create` calls with `client.responses.create` or `client.responses.stream`; ensure tool calling, JSON mode, and streaming responses are handled inline.
- Apply retries/backoff using built-in SDK options or shared helpers; scrub prompts before logging by hashing or redacting sensitive content.
- Emit OpenTelemetry spans/metrics at the point of use (latency, token counts, error codes, cost estimates) without introducing a new adapter layer.
- Acceptance: zero references to deprecated Chat Completions endpoints, telemetry confirmed in development logs, updated unit tests cover streaming, tool calls, moderation errors, and embeddings.

### WS2 – Ragas Pipeline Modernization

- Delete the legacy evaluator (`scripts/eval/rag_golden_eval.py`) and rebuild the pipeline around ragas ≥0.3 using `EvaluationDataset`, `SingleTurnSample`, and the new metric interfaces. Reference: [Ragas v0.1 → v0.2 Migration Guide](https://docs.ragas.io/en/stable/howtos/migrations/migrate_from_v01_to_v02/).
- Create a deterministic regression dataset with fixed seeds, temperature=0 prompting, and stored expected metric ranges (faithfulness, answer relevancy, context precision/recall).
- Implement a simple CLI entry point that runs the new evaluator only; remove all `--enable-ragas` flags and LangChain wrappers.
- Regenerate unit/integration tests to assert metric stability and error handling for missing contexts or evaluator misconfiguration.
- Acceptance: only the new evaluator exists in the repository, regression suite passes with ragas 0.3.x, and all tests reference the modern API.

### WS3 – CPU/GPU Validation Harnesses

- CPU harness: verify imports and ABI for NumPy/SciPy/scikit-learn, run PCA and t-SNE on sample embeddings, confirm BLAS provider, and check clustering behaviour under NumPy 2.3.x. Reference: [NumPy 2.3.0 Release Notes](https://numpy.org/devdocs/release/2.3.0-notes.html).
- GPU harness: probe CUDA/ROCm versions, run torch/xformers/flash-attn inference smoke tests, and capture device memory statistics. Reference: [xformers 0.0.32.post2 on PyPI](https://pypi.org/project/xformers/).
- Wire both harnesses into a single CI workflow; the job must fail on environmental incompatibilities and publish an artifact summarising results.
- Acceptance: harness reports stored in CI artifacts, failure blocks merge, documented command for local execution.

### WS4 – Repository Cleanup & Test Regeneration

- Delete all unused modules: OpenAI wrappers, outdated GPU helpers, compatibility flags, and fixtures tied to deprecated semantics.
- Refactor services and utilities to reflect the new direct integrations (e.g., update health checks to avoid deprecated `models.list` calls).
- Regenerate unit, integration, and end-to-end tests across embeddings, HYDE, observability, ragas, and validation harnesses.
- Enforce lint rules that forbid direct inclusion of removed modules by adding checks to lint/test scripts.
- Acceptance: repository search confirms deletion of deprecated files, coverage unchanged or improved, and all tests green under the new architecture.

### WS5 – Documentation & Dependency Alignment

- Update `pyproject.toml` with final versions: `openai>=2.0.0`, `ragas>=0.3.0`, `numpy>=2.3.0,<3`, `lxml>=6.0.0,<7`, `aiohttp>=3.13.0,<4`, `xformers>=0.0.32.post2`. References: [lxml 6.0.0 Changelog](https://lxml.de/changes-6.0.0.html), [aiohttp Changelog](https://docs.aiohttp.org/en/stable/changes.html).
- Regenerate `uv.lock`, run `uv sync --all-extras`, and execute the quality gates listed below.
- Rewrite documentation (developer guide, operator runbooks, plan) to describe only the new workflows, validation harness usage, and dependency baselines. Include compatibility notes (glibc ≥2.28 for NumPy/lxml wheels, CUDA/ROCm requirements for GPU stack).
- Acceptance: all tooling commands succeed, documentation merged, and CHANGELOG entry summarises the aggressive rewrite.

## Quality Gates (Blocking)

1. `uv run ruff format .`
2. `uv run ruff check . --fix`
3. `uv run pylint --fail-under=9.5 $(fd -e py src tests scripts)`
4. `uv run pyright`
5. `uv run python -m pytest -q`
6. CPU/GPU validation harness job (fails on incompatibility)

## Dependency Summary

| Library | Final Version Policy | Primary Reference |
| --- | --- | --- |
| openai | ≥2.0.0 | [GitHub: openai/openai-python](https://github.com/openai/openai-python) |
| ragas | ≥0.3.0 | [Ragas Migration Guide](https://docs.ragas.io/en/stable/howtos/migrations/migrate_from_v01_to_v02/) |
| numpy | ≥2.3.0,<3 | [NumPy 2.3.0 Release Notes](https://numpy.org/devdocs/release/2.3.0-notes.html) |
| lxml | ≥6.0.0,<7 | [lxml 6.0.0 Changelog](https://lxml.de/changes-6.0.0.html) |
| aiohttp | ≥3.13.0,<4 | [aiohttp Changelog](https://docs.aiohttp.org/en/stable/changes.html) |
| xformers | ≥0.0.32.post2 | [xformers on PyPI](https://pypi.org/project/xformers/) |

## Execution Checklist (Single PR)

- [ ] Rewrite OpenAI integrations (WS1).
- [ ] Implement ragas evaluator and regression suite (WS2).
- [ ] Add CPU/GPU validation harness and CI workflow (WS3).
- [ ] Purge deprecated modules and regenerate tests (WS4).
- [ ] Update dependencies, lockfile, docs, and run quality gates (WS5).
- [ ] Confirm documentation, CHANGELOG, and plan reflect only the new architecture.

Completion of this checklist delivers the final, simplified implementation with all legacy code removed and dependencies aligned to their latest stable releases.
