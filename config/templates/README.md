# Configuration Templates

Templates are now composed from a shared baseline (`base.json`) and a compact
profile index (`profiles.json`). Each profile records metadata plus the
`Settings` overrides that differ from the baseline defaults defined in
`src.config.loader.Settings`. The CLI wizard consumes this structure to display
profiles, preview metadata, and emit validated configurations.

## Layout

- `base.json` – canonical defaults shared by every profile. The file mirrors the
  lightweight "minimal" configuration: FastEmbed embeddings, Crawl4AI routing,
  Prometheus metrics on `/metrics`, and observability disabled by default.
- `profiles.json` – mapping of profile names to metadata (`description`,
  `use_case`, `features`) and an `overrides` object. Overrides may be shallow or
  deeply nested; anything omitted inherits from `base.json` and ultimately from
  the `Settings` model defaults.
- `browser-routing-rules.json` – curated router presets for specialised crawling
  tiers. Copy sections into the `browser.router` block inside a profile override
  or environment-derived configuration.

## Available profiles

Profiles ship in `profiles.json` with the following intents:

- `minimal` – baseline configuration that mirrors `base.json`.
- `development` – local debugging with Crawl4AI, headful Playwright, and relaxed
  router budgets.
- `production` – OpenAI embeddings, Firecrawl crawling, Dragonfly cache, and
  OTLP tracing enabled.
- `personal-use` – single-machine defaults with FastEmbed only and conservative
  concurrency limits.
- `local-only` – offline-friendly routing constrained to localhost targets.
- `testing` – deterministic throttles and metrics disabled for CI pipelines.

## Usage

Generate a configuration for a specific profile using the wizard:

```bash
uv run python -m src.cli.wizard template use --profile development --output config.json
uv run python -m src.cli.wizard validate --config config.json
```

Alternatively, merge overrides manually:

```bash
python - <<'PY'
import json
from pathlib import Path

base = json.loads(Path("config/templates/base.json").read_text())
profiles = json.loads(Path("config/templates/profiles.json").read_text())
profile = profiles["production"]

merged = base | {}
stack = [(merged, profile["overrides"])]
while stack:
    target, source = stack.pop()
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            stack.append((target[key], value))
        else:
            target[key] = value

Path("config.json").write_text(json.dumps(merged, indent=2))
PY
```

Secrets and runtime-specific values should continue to flow through environment
variables:

```bash
export AI_DOCS__OPENAI__API_KEY="sk-..."
export AI_DOCS__FIRECRAWL__API_KEY="fc-..."
```

Add new profiles by updating `profiles.json`. The CLI automatically recomputes
metadata tables and persists custom profiles via the wizard.
