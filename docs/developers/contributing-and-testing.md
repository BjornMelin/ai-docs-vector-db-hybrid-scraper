# Contributing & Testing

This guide combines contribution expectations with the unified testing workflow.

## 1. Contribution Workflow

1. Fork and clone the repository.
2. Create a feature branch (`git checkout -b feat/xyz`).
3. Run local quality gates (see below).
4. Commit changes using conventional commits.
5. Push and open a pull request with a clear description and linked issues.

### Conventional Commits

Use `type(scope): description`:

- `feat:` – new feature
- `fix:` – bug fix
- `docs:` – documentation update
- `style:` – formatting / style only
- `refactor:` – code refactoring
- `test:` – add or adjust tests
- `chore:` – maintenance tasks

Example: `feat(api): add vector search endpoint`.

### Pull Request Checklist

- [ ] Code follows project style.
- [ ] Tests added/updated.
- [ ] Documentation updated where applicable.
- [ ] Self-reviewed changes.
- [ ] No merge conflicts.

Maintainers review PRs within two business days. Address all comments and ensure
CI passes before merge. Squash-and-merge keeps history clean.

## 2. Local Quality Gates

Run the same commands enforced by CI:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pylint --fail-under=9.5 src scripts
uv run pyright
uv run python scripts/dev.py test --profile ci
uv build
```

### Documentation Verification

```bash
uv sync --frozen --group docs-dev
uv run python scripts/dev.py validate --check-docs --strict
uv run mkdocs build --strict
```

## 3. Testing Strategy

| Category | Location | Marker |
| --- | --- | --- |
| Unit | `tests/unit/` | `@pytest.mark.unit` |
| Service integration | `tests/services/` | `@pytest.mark.service` |
| End-to-end | `tests/e2e/` | `@pytest.mark.e2e` |

Common commands:

```bash
uv run python scripts/dev.py test --profile full            # all tests
uv run python scripts/dev.py test --profile ci             # mirrors CI
uv run pytest tests/test_example.py                        # specific file
uv run pytest -k "search"                                  # pattern match
uv run pytest -m integration                               # marker
uv run pytest -n auto                                      # parallel
```

Pytest markers are declared in `pyproject.toml` to document custom tags (`unit`,
`integration`, `e2e`, `slow`, etc.).

### Async & Mocking Patterns

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None

@patch("src.service.async_external_call", new_callable=AsyncMock)
async def test_async_mock(mock_call):
    mock_call.return_value = "mocked"
    result = await function_that_calls_async_api()
    assert result == "mocked"
```

Use context manager mocks and fixtures to arrange/act/assert cleanly. Negative
path coverage is strongly encouraged for error handling.

### Coverage

CI enforces the project-wide 80% threshold via the `ci` profile. Run it locally
when adjusting critical paths.

## 4. Maintainer Tips

- Label PRs using the maintained GitHub Actions workflows (`labeler.yml`).
- Use draft PRs for feedback when work is in progress.
- Update ADRs when architectural decisions change.

Keep this document up to date when workflows, linting rules, or testing
strategies evolve.
