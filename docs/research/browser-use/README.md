---
title: Browser Automation Research
audience: developers
status: active
owner: research
last_reviewed: 2025-03-13
---

# Browser Automation Research

Summary of ongoing investigations into browser automation improvements, especially the migration to
browser-use v0.3.x and its integration with the tiered automation manager.

## Current Objectives

- Evaluate browser-use v0.3.2 capabilities and API changes.
- Measure reliability and performance across scripted browsing scenarios.
- Determine fallbacks and guardrails before promoting v0.3.x to production tiers.

## Key Documents

- `browser-use-v3-solo-dev-master-guide.md` – Detailed migration plan (local evaluation focus)
- `browser-use-v3-solo-dev-quick-start-plan.md` – Abbreviated setup instructions
- `archive/` – Historical research and PRDs for earlier versions

## Next Steps

1. Complete compatibility testing across the five-tier automation pipeline.
2. Document configuration changes needed for enterprise deployments.
3. Update the operator runbooks once the new version is validated.

When promotion criteria are met, move stable guidance into the main operator/developer docs and
reference it from this summary.
