# Unit Test Refactor Plan

## Scope Overview
- **Global fixtures (`tests/conftest.py`)** – Centralized utilities currently mix deterministic fixtures, property-based Hypothesis strategies, and compatibility shims for optional dependencies. This file is over 700 lines with duplicated helpers that already exist under `tests/utils/` and includes backwards-compatibility sys.path injections that should be eliminated for a single final surface. 【F:tests/conftest.py†L1-L760】
- **AI embeddings suite (`tests/unit/ai/`)** – Property-based tests generate random embeddings and text without exercising production code paths, duplicating utilities from `tests/utils/ai_testing_utilities.py` while slowing execution. 【F:tests/unit/ai/test_embedding_properties.py†L1-L372】
- **Functional embedding tests (`tests/unit/services/functional/test_embeddings.py`)** – Covers the functional API but mixes redundant assertions, inlined mocks, and incomplete coverage (missing `get_provider_info`, `get_smart_recommendation`, and `get_usage_report`). Also contains unreachable code and lax docstrings. 【F:tests/unit/services/functional/test_embeddings.py†L1-L240】
- **Agent orchestration suites (`tests/unit/services/agents/`)** – Heavy Hypothesis usage with slow strategies and broad mocks. Needs deterministic parametrized coverage targeting public agent APIs. 【F:tests/unit/services/agents/test_tool_discovery_core.py†L1-L76】
- **Infrastructure guardrails (`tests/test_infrastructure.py`)** – Still checks for deprecated Hypothesis strategies exported from `tests.conftest` instead of the consolidated utility modules. 【F:tests/test_infrastructure.py†L222-L255】

## Technical Debt Themes
1. **Legacy property-based scaffolding** – Hypothesis strategies across AI and agent suites inflate runtime with little additional coverage. Replace with table-driven pytest cases using deterministic fixtures from `tests/utils`. 【F:tests/unit/ai/test_embedding_properties.py†L1-L372】【F:tests/unit/services/agents/test_tool_discovery_core.py†L24-L60】
2. **Duplicated utilities and back-compat shims** – `tests/conftest.py` re-implements GPU fallbacks, embedding generators, and document factories already available in dedicated helper modules. These should be deleted in favor of a single utility import path. 【F:tests/conftest.py†L560-L756】
3. **Coverage gaps for functional embedding helpers** – Key coroutine wrappers (`get_provider_info`, `get_smart_recommendation`, `get_usage_report`) lack regression tests, risking silent regressions when refactoring the functional API surface. 【F:src/services/functional/embeddings.py†L200-L360】
4. **Inconsistent organization** – Tests targeting the same subsystems live across `tests/unit/ai/`, `tests/unit/services/functional/`, and top-level infrastructure sanity suites. Plan consolidates by feature (functional embeddings under `tests/unit/services/functional/`, agent orchestration under `tests/unit/services/agents/`, etc.).
5. **Quality gate drift** – Several modules fail Ruff, Pylint, or Pyright when run individually due to unused imports, unreachable code, or missing async markers. The refactor will adopt per-directory gates during migration and enforce `pytest --strict-markers --maxfail=1` once suites are stabilized.

## Refactor Workstreams
1. **Fixture consolidation (Sprint 1)**
   - [x] Delete Hypothesis strategies and bespoke fixtures from `tests/conftest.py`.
   - [x] Move reusable deterministic helpers into `tests/utils/` and expose lightweight fixtures per domain.
   - [x] Remove sys.path shims; rely on packaging metadata.
2. **Embedding coverage cleanup (Sprint 1)**
   - [x] Drop `tests/unit/ai/test_embedding_properties.py` in favor of deterministic coverage in `tests/unit/services/functional/test_embeddings.py`.
   - [x] Extend functional tests to cover smart recommendation, provider info, and usage report flows using async stubs.
   - [x] Update infrastructure sanity checks to validate the new deterministic helper access pattern.
3. **Agent orchestration rewrite (Sprint 2)**
   - [x] Replace Hypothesis-driven dataclass instantiations with parametrized fixtures using canonical tool descriptors.
   - [x] Introduce focused tests for `DynamicToolDiscovery` behavior (registration, capability filtering, orchestration flows) using `pytest.raises` and `pytest.mark.parametrize`.
4. **Vector and processing suites (Sprint 3)**
   - [x] Review `tests/unit/services/vector_db/` and `tests/unit/processing/` for duplicated mocks; leverage maintained client fixtures and Pydantic model factories.
   - [x] Remove obsolete tests targeting legacy manager classes replaced by functional modules.
5. **Framework and CLI tests (Sprint 4)**
   - [x] Normalize CLI tests around `pytester` or `CliRunner` with golden-file comparisons where helpful.
   - [x] Delete back-compat CLI path assertions once the new entry points are finalized.
6. **Global gates (Sprint 4)**
   - [x] Configure `pytest.ini` or `pyproject.toml` with strict markers, `filterwarnings = error`, and `pytest-randomly` seeds once suites stabilized.
   - [x] Add `make quality-unit` command alias to run Ruff, Pylint, Pyright, and targeted pytest on unit scopes.

## Immediate Deletions & Additions (Sprint 1)
- Remove `TestEmbeddingPropertiesFixed` and related Hypothesis fixtures; replace with deterministic async coverage hitting `src/services/functional/embeddings` directly.
- Update `tests/test_infrastructure.py` to reference `EmbeddingTestUtils.generate_test_embeddings` for smoke validation.
- Introduce targeted tests for `get_provider_info`, `get_smart_recommendation`, and `get_usage_report` covering success and failure paths.

## Future Cleanup Targets
- `tests/unit/services/agents/test_dynamic_tool_discovery.py`: convert scenario-based Hypothesis tests into table-driven assertions and delete legacy `uuid4` factories.
- `tests/unit/services/agents/test_agentic_orchestrator.py`: collapse overlapping orchestrator state machine tests into deterministic state transition checks.
- `tests/unit/framework/`: audit for redundant FastAPI app instantiation mocks; prefer `TestClient` fixtures scoped per module.
- `tests/unit/utils/`: deduplicate `MinimalEmbeddingTestUtils` and `EmbeddingTestUtils` into a single helper module with deterministic RNG seeding.

## Quality Gate Strategy
- During each workstream run `uv run ruff format {paths} && uv run ruff check {paths} --fix`, `uv run pylint {module_dirs}`, `uv run pyright {paths}`, and `uv run pytest {paths} -q --strict-markers --maxfail=1`.
- Record pytest-randomly seeds in test output and enforce deterministic randomness via `random.Random` or `numpy.random.default_rng` with explicit seeds.

## Deliverables Snapshot
- Progressive removal of Hypothesis across unit suites.
- Consolidated deterministic fixtures available via `tests/utils` modules.
- Updated changelog entries documenting each major suite rewrite and deletion of legacy tests.
- Quality gates documented per sprint with recorded seeds and tool versions.
