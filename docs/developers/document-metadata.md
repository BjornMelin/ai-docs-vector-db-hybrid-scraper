---
title: Documentation Metadata
status: draft
owner: documentation
last_reviewed: 2025-03-13
---

## Documentation Metadata Guidelines

All Markdown files should include YAML front matter so that we can track ownership, intended
audience, and review cadence. Use the fields below unless a document has specific needs.

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

- **title** – Human-friendly page title; used by MkDocs if set.
- **audience** – Primary readership, one of `users`, `developers`, `operators`, or `general` for
  cross-audience material.
- **status** – Lifecycle indicator. Use `draft` while a page is under revision, `active` for
  published content, and `deprecated` when the page is kept for historical reference only.
- **owner** – Team or role responsible for keeping the page current.
- **last_reviewed** – Date of the most recent content review; update when making significant changes.

## Tips

- Place the front matter at the top of the file, followed by a blank line and the first heading.
- Archived content should set `status: deprecated` and include a banner pointing readers to the
  canonical replacement.
- When a page serves multiple audiences, note the primary audience in `audience` and reference other
  audiences in the introduction.
- Update `last_reviewed` as part of routine maintenance or release checklists so stale pages are easy
  to spot.
