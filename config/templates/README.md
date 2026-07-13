# Configuration Templates

Templates are composed from the packaged baseline
(`src/config/templates/base.json`) and profile index
(`src/config/templates/profiles.json`). Each profile records metadata plus the
`Settings` overrides that differ from the baseline defaults defined in
`src.config.loader.Settings`. The CLI wizard consumes this structure to display
profiles, preview metadata, and emit validated configurations.

## Layout

- `src/config/templates/base.json` – canonical defaults shared by every profile.
  The file mirrors the lightweight "minimal" configuration: FastEmbed
  embeddings, Crawl4AI routing, Prometheus metrics on `/metrics`, and
  observability disabled by default.
- `src/config/templates/profiles.json` – mapping of profile names to metadata
  (`description`, `use_case`, `features`) and an `overrides` object. Overrides
  may be shallow or deeply nested; anything omitted inherits from `base.json`
  and ultimately from the `Settings` model defaults.

## Available profiles

Profiles ship in the packaged `profiles.json` with the following intents:

- `minimal` – baseline configuration that mirrors `base.json`.
- `development` – local debugging with Crawl4AI, headful Playwright, and relaxed
  router budgets.
- `production` – Dragonfly cache, OTLP tracing, and production limits. Select
  external providers through environment variables.
- `personal-use` – single-machine defaults with FastEmbed only and conservative
  concurrency limits.
- `local-only` – offline-friendly routing constrained to localhost targets.
- `testing` – deterministic throttles and metrics disabled for CI pipelines.

## Usage

Generate a configuration for a specific profile using the wizard:

```bash
uv run ai-docs setup --profile development
uv run ai-docs --config config/profiles/development.json config validate
```

Alternatively, inspect the packaged assets with the standard library:

```bash
python - <<'PY'
import json
from importlib.resources import files

assets = files("src.config.templates")
base = json.loads(assets.joinpath("base.json").read_text())
profiles = json.loads(assets.joinpath("profiles.json").read_text())
profile = profiles["production"]
print(json.dumps({"base": base, "production": profile}, indent=2))
PY
```

Secrets and runtime-specific values should continue to flow through environment
variables:

```bash
export AI_DOCS_OPENAI__API_KEY="sk-your_openai_api_key_here"
export AI_DOCS_BROWSER__FIRECRAWL__API_KEY="fc-your_firecrawl_api_key_here"
```

Add new profiles by updating `src/config/templates/profiles.json`. The CLI
automatically recomputes metadata tables and emits validated profile
configurations through the wizard.
