# Documentation File Naming Guidelines

> **Status**: Active  
> **Last Updated**: 2025-01-09  
> **Purpose**: Standardize file naming across all documentation

## File Naming Convention

All documentation files will use **kebab-case** (lowercase with hyphens):

```
✅ quick-start.md
✅ system-overview.md
✅ advanced-search-implementation.md
❌ QUICK_START.md
❌ quickStart.md
❌ quick_start.md
```

## Rationale

1. **Web-friendly**: URLs look cleaner without underscores or capitals
2. **Readable**: Hyphens create natural word separation
3. **Standard**: Common convention in modern documentation systems
4. **SEO-friendly**: Search engines treat hyphens as word separators
5. **Cross-platform**: No case sensitivity issues

## Naming Patterns

### Feature Documentation
```
{feature}-{type}.md
```
Examples:
- `vector-search-guide.md`
- `embedding-models-reference.md`
- `browser-automation-tutorial.md`

### Architecture Documentation
```
{component}-architecture.md
```
Examples:
- `system-architecture.md`
- `browser-automation-architecture.md`

### Operations Documentation
```
{task}-guide.md
```
Examples:
- `deployment-guide.md`
- `monitoring-guide.md`
- `troubleshooting-guide.md`

### Special Files
- `README.md` - Always uppercase (GitHub convention)
- `index.md` - For documentation systems that need index files

## Migration from Old Names

| Old Name | New Name |
|----------|----------|
| QUICK_START.md | quick-start.md |
| SYSTEM_OVERVIEW.md | system-overview.md |
| ADVANCED_SEARCH_IMPLEMENTATION.md | advanced-search-implementation.md |
| VECTOR_DB_BEST_PRACTICES.md | vector-db-best-practices.md |

## Directory Names

Directories also use kebab-case:
```
docs/
├── getting-started/
├── how-to-guides/
├── reference/
├── concepts/
└── operations/
```

## Implementation

1. All new files must follow kebab-case
2. Existing files will be renamed in bulk
3. Git redirects will preserve history
4. All internal links will be updated after renaming