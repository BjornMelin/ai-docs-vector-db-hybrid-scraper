---
title: Application Profiles
audience: developers
status: draft
owner: platform-engineering
last_reviewed: 2025-02-14
---

## Application Profiles

The API server now supports _profiles_ that describe the composition of routes,
services, and middleware that should be mounted at startup. Profiles keep the
codebase DRY while allowing us to ship mode-specific experiences like the
original "simple" and "enterprise" modes.

## Profiles

| Profile      | Environment Flag          | Description                                          |
| ------------ | ------------------------- | ---------------------------------------------------- |
| `simple`     | `AI_DOCS_MODE=simple`     | Local-first experience with the minimal route set.   |
| `enterprise` | `AI_DOCS_MODE=enterprise` | Enterprise surface area (routers must be installed). |

_Profiles are resolved through `AppProfile` (`src/api/app_profiles.py`)._

## Runtime Detection

`AppProfile.detect_profile()` inspects `AI_DOCS_MODE` and returns a profile
value. `create_app(profile)` accepts:

- An `AppProfile` instance (`AppProfile.SIMPLE`)
- A legacy `ApplicationMode`
- A case-insensitive string (`"simple"`)
- `None` – auto-detects using `AI_DOCS_MODE`

## Composition Rules

- Routes are installed by profile-specific installer functions. Enterprise
  routes are imported lazily and must exist for the profile to boot.
- Services register through `ModeAwareServiceFactory`. When optional
  implementations are missing, fail-closed placeholders raise predictable
  errors at runtime instead of returning partially-initialized services.
- The `/health` endpoint now publishes `service_status` entries for every
  enabled service in the active profile.

## Local Workflow

```bash
# Run unit tests for the simple profile
PYTEST_BENCHMARK_DISABLE=1 AI_DOCS_MODE=simple \
  uv run python scripts/dev.py test --profile quick

# Smoke-test enterprise profile (fails fast if routers are missing)
PYTEST_BENCHMARK_DISABLE=1 AI_DOCS_MODE=enterprise \
  uv run python scripts/dev.py test --profile quick
```

> `PYTEST_BENCHMARK_DISABLE=1` avoids the `pytest-benchmark` + `xdist` warning
> that otherwise aborts the test session.

## Adding a New Profile

1. Extend `AppProfile` with the new enum value.
2. Implement router and service installers in `app_factory`.
3. Gate optional dependencies with lazy imports and fail-closed placeholders.
4. Add the profile to the CI matrix and document the expected capability delta.
