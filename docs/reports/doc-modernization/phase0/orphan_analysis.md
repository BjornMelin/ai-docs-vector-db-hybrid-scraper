# Orphaned Documentation Analysis

This note classifies the 225 orphaned artefacts identified by `doc_inventory.py` and recommends dispositions for the Phase 3 migration.

## Summary by Directory

| Directory Segment | Count | Primary Themes | Proposed Action |
| --- | --- | --- | --- |
| `docs/api/**` | 97 | Auto-generated Sphinx API stubs | Archive under `docs/archive/` after publishing regeneration instructions. |
| `docs/developers/**` | 32 | Historical engineering guides (`architecture-diagrams`, `config-*`, CI reports) | Review against new IA; fold essential content into `docs/developers/` target sections, archive superseded reports. |
| `docs/operators/**` | 15 | Legacy operator guides referencing removed heading anchors | Merge relevant runbook material into the upcoming IA structure, delete outdated duplicates post-consolidation. |
| `planning/done/**` | 15 | Completed research reports | Migrate to `docs/archive/planning/` with provenance README, keep summaries for compliance. |
| `docs/security/**` | 9 | Security playbooks and assessments | Map into security section of new IA; confirm overlap with requirements/ADR coverage before deletion. |
| `docs/research/**` | 7 | Archived browser-use / transformation studies | Archive as read-only appendix linked from portfolio if still valuable. |
| `docs/reports/**` | 5 | Interim validation reports | Decide whether to incorporate findings into updated requirements/ADR narratives, otherwise archive. |
| `docs/build-config/**` | 3 | MkDocs/Sphinx config notes | Inline into developer doc on doc toolchain; archive redundant READMEs. |
| `docs/portfolio/**` | 3 | Marketing-oriented briefs | Confirm target audience post-IA; move to `docs/archive/legacy-messaging/` if not required. |
| `docs/users/**` | 3 | Early user-facing guides | Update/merge into new MkDocs IA; remove stale copies once replacements live. |

## Next Steps

1. Incorporate this classification into the forthcoming `mapping.csv` so that Phase 3 migrations can proceed in auditable batches.
2. For `docs/api/**`, draft regeneration instructions (Sphinx build command + prerequisites) in the archive README to avoid knowledge loss.
3. Attach compliance-sensitive reports (`planning/done/**`, `docs/security/**`) to the retention policy being drafted in Phase 1 to ensure traceable archival decisions.


## Regeneration Instructions for Archived Sphinx API

To regenerate the API documentation when needed:

```bash
uv sync --group docs
uv run sphinx-build -b html docs/build-config docs/api
```

Document the build hash and timestamp in the archive README after regeneration.
