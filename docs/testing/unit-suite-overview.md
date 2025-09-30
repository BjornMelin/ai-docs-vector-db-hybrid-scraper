# Unit Test Suite Overview

This document captures the consolidated state of the unit-test suites after the refactor that flattened legacy directories, replaced bespoke helpers, and focused on the supported production surfaces.

## Coverage Map

| Feature Surface | Primary Tests |
| --- | --- |
| Dual-mode architecture (simple vs. enterprise) | `tests/unit/test_architecture_modes.py` |
| Unified configuration defaults (embedding, performance, security) | `tests/unit/test_config_settings.py` |
| Core constants consumed by SDK tooling | `tests/unit/test_core_constants.py` |
| Client manager provider wiring and context manager | `tests/unit/test_infrastructure_client_manager.py` |
| Agentic collection optimisation metadata | `tests/unit/test_ai_agentic_configs.py` |
| Vector search security models & batch orchestration | `tests/unit/test_models_vector.py`, `tests/unit/test_processing_batch.py` |
| ML security validator behaviours | `tests/unit/test_security_validator.py` |
| GPU fallbacks and service health probes | `tests/unit/test_utils_runtime.py` |

## Decision Scorecard

The refactor used the weighted decision framework (Solution Leverage 35%, Application Value 30%, Maintenance Load 25%, Adaptability 10%) to ensure each change favoured simplicity and determinism.

### Architecture Coverage Strategy

| Option | Solution Leverage | Application Value | Maintenance Load | Adaptability | Weighted Total |
| --- | --- | --- | --- | --- | --- |
| Maintain legacy property-based suite | 3 | 5 | 3 | 4 | 3.70 |
| Replace with deterministic parametrised tests | 9 | 8 | 9 | 8 | 8.60 |

Parameterised pytest cases kept the dual-mode matrix compact while still asserting the hybrid-search toggles. Hypothesis scaffolding previously introduced runtime variance without expanding coverage.

### ClientManager Fixture Strategy

| Option | Solution Leverage | Application Value | Maintenance Load | Adaptability | Weighted Total |
| --- | --- | --- | --- | --- | --- |
| Spin up the real dependency-injector container | 4 | 6 | 3 | 4 | 4.35 |
| Stub providers and assert behaviour through `managed_client` | 8 | 8 | 8 | 7 | 7.90 |

A documented stub provider validates the singleton lifecycle without relying on an external container boot, keeping the suite fast and deterministic.

### Security Validator Scope

| Option | Solution Leverage | Application Value | Maintenance Load | Adaptability | Weighted Total |
| --- | --- | --- | --- | --- | --- |
| Execute `pip-audit`/`trivy` subprocesses inside tests | 2 | 6 | 2 | 3 | 3.30 |
| Focus on skip paths and pattern rejection with monkeypatched tools | 7 | 7 | 8 | 6 | 7.15 |

Monkeypatched tool detection keeps the security validator deterministic without invoking heavyweight scanners during unit runs.

### Processing Assertion Tolerance

| Option | Solution Leverage | Application Value | Maintenance Load | Adaptability | Weighted Total |
| --- | --- | --- | --- | --- | --- |
| Enforce exact floating-point equality for analyser scores | 2 | 5 | 3 | 4 | 3.20 |
| Assert ranges using `pytest.approx` and bounded inequalities | 7 | 8 | 8 | 7 | 7.35 |

`pytest.approx` provides numerically stable comparisons that handle tiny floating-point drift while guaranteeing metrics stay in the required range.【https://docs.pytest.org/en/8.3.x/reference/reference.html#pytest-approx†L1-L9】

### Runtime Health Check Coverage

| Option | Solution Leverage | Application Value | Maintenance Load | Adaptability | Weighted Total |
| --- | --- | --- | --- | --- | --- |
| Skip dragonfly health assertions when the toggle is absent | 3 | 4 | 6 | 5 | 4.40 |
| Provide a lightweight namespace stub with the expected cache flags | 8 | 7 | 8 | 6 | 7.35 |

`types.SimpleNamespace` offers a minimal standard-library stand-in that exposes the attributes required by the health checker without mutating production settings models.【https://docs.python.org/3/library/types.html#types.SimpleNamespace†L1-L24】

## Technical Debt Register

| Item | Severity | Impact | Maintenance Cost | Fix Effort | Dependency Risk | Decision Link |
| --- | --- | --- | --- | --- | --- | --- |
| `ClientManager` still relies on a global dependency-injection container making true unit isolation awkward. Stubbing works but hides wiring regressions. | Medium | Provider initialisation regressions ripple across agentic search and MCP services. | Medium | Medium | Low | [Decision: ClientManager Fixture Strategy](#clientmanager-fixture-strategy) |
| `MLSecurityValidator` shells out to `pip-audit`/`trivy`; tests only cover skip paths. | Low | External scanners can fail noisily in CI, masking genuine security regressions. | Low | Medium | Medium | [Decision: Security Validator Scope](#security-validator-scope) |
| Text analytics heuristics return floating-point scores that vary slightly across Python versions. | Low | Overly strict assertions could introduce flaky behaviour. | Low | Low | Low | [Decision: Processing Assertion Tolerance](#processing-assertion-tolerance) |
| `ServiceHealthChecker.check_dragonfly_connection` hard-references `CacheConfig.enable_dragonfly_cache`, which is absent in the default settings model. | Medium | Health status endpoints raise `AttributeError`, hiding actual cache misconfiguration in production. | Medium | Medium | Low | [Decision: Runtime Health Check Coverage](#runtime-health-check-coverage) |

## Maintenance Notes

- Fixtures have been centralised under `tests/conftest.py` with domain helpers in `tests/_helpers/` to avoid bespoke scaffolding.
- The suite enforces determinism via parametrisation, seeded randomness, and `pytest.approx` for floating-point checks.【https://docs.pytest.org/en/8.3.x/reference/reference.html#pytest-approx†L1-L9】
- Simple namespace stubs remain the lightest approach for modelling missing configuration flags without mutating production models.【https://docs.python.org/3/library/types.html#types.SimpleNamespace†L1-L24】
