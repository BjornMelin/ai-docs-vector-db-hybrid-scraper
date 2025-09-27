# GitHub Composite Actions

This directory exposes a small set of composite actions that keep workflow
definitions lean while still delegating to first-party tooling for
implementation details.

## `setup-environment`
Sets up Python and [`uv`](https://github.com/astral-sh/uv) with caching so
workflows can immediately run quality gates or tests.

### Inputs
- `python-version` (optional): Python version to install. Defaults to `3.12`.
- `cache-suffix` (optional): Extra entropy for the dependency cache key. Use it
  to bust the cache after dependency changes.
- `install-dev` (optional): When `true` (default) installs development
  dependencies. Set to `false` for runtime-only environments.

### Outputs
- `cache-hit`: Surface whether the dependency cache restored successfully.

### Example
```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/setup-environment
        with:
          python-version: '3.12'
      - run: |
          set -euo pipefail
          uv run ruff check .
```

## `validate-config`
Runs the configuration validation helper (`scripts/ci/validate_config.py`) with
consistent wiring so workflows do not have to duplicate argument handling.

### Inputs
- `config-root` (optional): Directory containing configuration assets. Defaults
  to `config`.
- `templates-dir` (optional): Directory containing environment templates.
  Defaults to `config/templates`.
- `environment` (optional): Preferred environment template. Leave blank to
  validate every template.

### Example
```yaml
jobs:
  config-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/setup-environment
      - uses: ./.github/actions/validate-config
        with:
          environment: development
```

Both actions are exercised by `.github/workflows/test-composite-actions.yml` to
catch regressions whenever the implementation changes.
