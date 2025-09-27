# GitHub Composite Actions

This directory currently exposes a single composite action that we rely on across
multiple CI workflows. The action keeps our workflow files lean while still
using first-party tooling for everything else.

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

The action is exercised by `.github/workflows/test-composite-actions.yml` to
catch regressions whenever the implementation changes.
