---
id: remaining_tasks.migration_notes
last_reviewed: 2025-07-02
status: draft
---

# Migration Notes – GitHub Issues & Projects

## Overview
Import the canonical backlog into GitHub Issues/Projects while preserving legacy traceability and new ownership fields. Use IDs (e.g., `QA-01`) as canonical references and include legacy IDs in issue bodies.

## Issue Template
```
## Summary
<short task statement>

## Legacy References
- Legacy IDs: <comma-separated>
- Evidence: <paths / commands>

## Acceptance Criteria
- ...
- ...

## Dependencies
- Blocks: <ID list or n/a>
- Blocked by: <ID list or n/a>

## Owner
<role or assignee placeholder>

## Notes
<any context, risks, or follow-ups>
```

## Labels & Metadata
- `Category::Quality`, `Category::Platform`, `Category::Ops`, `Category::Retrieval`, `Category::Security`, `Category::Browser`, `Category::Analytics`, `Category::Docs`
- `Status::Pending`, `Status::InProgress`, `Status::Done`
- `Priority::P0/P1/P2` (set during triage)
- `LegacyID::###` (optional custom label if automated parsing desired)

## Dependency Management
- Use GitHub issue linking (`blocks` / `blocked by`) reflecting backlog dependencies.
- Add automation (Projects v2) to auto-update item status when linked issues close.

## Import Steps
1. Export `backlog.md` to CSV (columns: Title, Body, Labels). Include summary + template fields per task.
2. Use GitHub Issue Importer (beta) or `gh issue import` to bulk create issues.
3. Create a Project board with views by Category and Status; auto-add new issues via filter `label:Category::*`.
4. Populate dependencies manually or via `gh issue edit --add-project` and `--add-assignee`.
5. Sync `decision_log.md` as project note for auditors; reference it in issue descriptions.

## Automation Hooks
- Extend existing workflows to post status back to the Project when issues close.
- Consider adding CODEOWNERS mappings based on Owner placeholders (e.g., `QA Lead` → `@org/qa-team`).
- Schedule weekly report aggregating Project status against `risk_register.md`.

## Legacy References
Keep `decision_log.md` in repo to translate future discoveries; link to it from each issue (`See backlog decision log for provenance`). This satisfies traceability expectations during audits.
