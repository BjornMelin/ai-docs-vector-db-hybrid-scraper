---
title: Documentation Metadata
audience: developers
status: active
owner: documentation
last_reviewed: 2025-06-30
---

## Documentation Metadata Guidelines

Every Markdown page should include YAML front matter so we can track ownership,
audience, and review cadence. Use the fields below unless a page has specific
requirements:

```yaml
---
title: <Short page title>
audience: <users|developers|operators|general>
status: <draft|active|deprecated>
owner: <team-or-role>
last_reviewed: <YYYY-MM-DD>
---
```

## Field Reference

- **title** – Human-friendly page title; MkDocs will reuse it for navigation when set.
- **audience** – Primary readership (`users`, `developers`, `operators`, or
  `general`). Note other audiences in the introduction if needed.
- **status** – Lifecycle indicator (`draft`, `active`, or `deprecated`). Deprecated
  pages should reference their replacements.
- **owner** – Team or role responsible for keeping the content current.
- **last_reviewed** – Date of the most recent content review. Update it when
  shipping meaningful changes or as part of release checklists.

## Tips

- Place the front matter at the top of the file, followed by a blank line and the
  first heading.
- Archived content should set `status: deprecated` and include a banner pointing to
  the canonical replacement.
- Keep review dates fresh to make stale pages easy to identify during audits.
- Templates that require different metadata (e.g., ADRs) may define their own
  schema; make sure their guidance is captured alongside the template.

For reusable skeletons and additional examples, see
`docs/developers/templates/document-metadata.md`.
