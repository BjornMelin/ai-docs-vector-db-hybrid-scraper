# Testing Documentation

Modern test practices and infrastructure for the AI Documentation Vector DB system.

## Essential Testing Documents

- [Testing Guide](testing-guide.md) - Modern test practices, patterns, and commands for developers
- [MCP Tool and Service Unit Test Strategy](mcp-unit-tests.md) - Coverage map, decision record, and technical debt register for the MCP suites
- [Evaluation Harness Playbook](evaluation-harness.md) - Golden dataset workflow, telemetry catalog, and operational checklist for regression gating

## Quick Test Commands

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest -m unit                    # Unit tests only
uv run pytest -m integration             # Integration tests only
uv run pytest -m e2e                     # End-to-end tests only

# Run tests in parallel
uv run pytest -n auto                    # Auto-detect CPU cores
uv run pytest -n 4                       # Use 4 workers

# Run with coverage
uv run python scripts/dev.py test --profile ci

# Debug failing tests
uv run pytest tests/specific/test.py::test_function -xvs
```

## Test Categories

- **Unit Tests**: Fast, isolated tests (no I/O)
- **Integration Tests**: Cross-boundary tests with mocked external services
- **E2E Tests**: Full user journey tests
- **Performance Tests**: Benchmarks and load tests

## Development Workflow

1. Write unit tests first (TDD)
2. Run tests locally: `uv run pytest -m unit`
3. Run integration tests: `uv run pytest -m integration`
4. Check coverage: `uv run python scripts/dev.py test --profile ci`
5. Run full suite before PR: `uv run pytest`
