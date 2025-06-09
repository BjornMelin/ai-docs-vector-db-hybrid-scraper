# Documentation Restructure Plan

> **Status**: In Progress  
> **Last Updated**: 2025-01-09  
> **Purpose**: Map current structure to new audience-based organization

## New Directory Structure

```
docs/
├── README.md                    # Main navigation hub (keep as-is)
├── getting-started/            # For new users
│   ├── README.md              # Getting started overview
│   ├── quick-start.md         # From QUICK_START.md
│   ├── installation.md        # Extract from quick-start
│   └── first-project.md       # Tutorial content
│
├── tutorials/                  # Learning-oriented (hands-on)
│   ├── README.md
│   ├── browser-automation.md   # From user-guides/
│   ├── crawl4ai-setup.md      # From user-guides/
│   └── build-search-app.md    # New tutorial
│
├── how-to-guides/             # Task-oriented guides
│   ├── README.md
│   ├── implement-search/
│   │   ├── README.md
│   │   ├── basic-search.md
│   │   ├── advanced-search.md    # From features/
│   │   ├── hyde-enhancement.md   # From features/
│   │   └── add-reranking.md      # From features/
│   ├── process-documents/
│   │   ├── README.md
│   │   ├── chunking-guide.md     # From features/
│   │   └── embedding-models.md   # From features/
│   ├── optimize-performance/
│   │   ├── README.md
│   │   ├── vector-db-tuning.md   # From features/
│   │   ├── caching-strategy.md
│   │   └── monitoring.md         # From operations/
│   └── deploy/
│       ├── README.md
│       ├── canary-deployment.md  # From deployment/
│       └── production-guide.md
│
├── reference/                  # Information-oriented
│   ├── README.md
│   ├── api/
│   │   ├── README.md
│   │   ├── rest-api.md         # From API_REFERENCE.md
│   │   └── browser-api.md      # From browser_automation_api.md
│   ├── configuration/
│   │   ├── README.md
│   │   ├── config-schema.md     # From UNIFIED_CONFIGURATION.md
│   │   └── environment-vars.md
│   ├── mcp-tools/              # Keep MCP grouped
│   │   ├── README.md           # From mcp/README.md
│   │   ├── setup.md            # From mcp/SETUP.md
│   │   └── migration.md        # From mcp/MIGRATION_GUIDE.md
│   └── cli/
│       ├── README.md
│       └── commands.md
│
├── concepts/                   # Understanding-oriented
│   ├── README.md
│   ├── architecture/
│   │   ├── README.md
│   │   ├── system-overview.md   # From SYSTEM_OVERVIEW.md
│   │   ├── v1-architecture.md   # From INTEGRATED_V1_ARCHITECTURE.md
│   │   └── component-design.md
│   ├── features/
│   │   ├── README.md
│   │   ├── search-explained.md
│   │   ├── chunking-theory.md   # From chunking/CHUNKING_RESEARCH.md
│   │   └── embedding-concepts.md
│   └── best-practices/
│       ├── README.md
│       └── vector-db-patterns.md
│
├── operations/                 # Ops & maintenance
│   ├── README.md
│   ├── deployment/
│   │   ├── README.md
│   │   └── deployment-options.md
│   ├── monitoring/
│   │   ├── README.md
│   │   ├── metrics-guide.md      # From MONITORING.md
│   │   └── troubleshooting.md    # From TROUBLESHOOTING.md
│   └── maintenance/
│       ├── README.md
│       ├── backup-restore.md
│       └── upgrades.md
│
├── contributing/               # For contributors
│   ├── README.md
│   ├── development-setup.md     # From DEVELOPMENT_WORKFLOW.md
│   ├── testing-guide.md         # From TESTING_DOCUMENTATION.md
│   ├── architecture-guide.md    # From ARCHITECTURE_IMPROVEMENTS.md
│   └── style-guide.md
│
└── archive/                    # Keep as-is
    └── [existing structure]
```

## Migration Mapping

### Phase 1: File Renames (Current Location)

| Current File | New Name |
|--------------|----------|
| QUICK_START.md | quick-start.md |
| SYSTEM_OVERVIEW.md | system-overview.md |
| ADVANCED_SEARCH_IMPLEMENTATION.md | advanced-search.md |
| HYDE_QUERY_ENHANCEMENT.md | hyde-enhancement.md |
| RERANKING_GUIDE.md | add-reranking.md |
| VECTOR_DB_BEST_PRACTICES.md | vector-db-tuning.md |
| ENHANCED_CHUNKING_GUIDE.md | chunking-guide.md |
| EMBEDDING_MODEL_INTEGRATION.md | embedding-models.md |
| BROWSER_AUTOMATION_ARCHITECTURE.md | browser-architecture.md |
| UNIFIED_CONFIGURATION.md | config-schema.md |
| API_REFERENCE.md | rest-api.md |
| browser_automation_api.md | browser-api.md |
| MONITORING.md | metrics-guide.md |
| TROUBLESHOOTING.md | troubleshooting.md |
| CANARY_DEPLOYMENT_GUIDE.md | canary-deployment.md |
| DEVELOPMENT_WORKFLOW.md | development-setup.md |
| TESTING_DOCUMENTATION.md | testing-guide.md |
| ARCHITECTURE_IMPROVEMENTS.md | architecture-guide.md |

### Phase 2: File Moves

| From | To |
|------|-----|
| quick-start.md | getting-started/quick-start.md |
| user-guides/browser-automation.md | tutorials/browser-automation.md |
| user-guides/crawl4ai.md | tutorials/crawl4ai-setup.md |
| features/advanced-search.md | how-to-guides/implement-search/advanced-search.md |
| features/hyde-enhancement.md | how-to-guides/implement-search/hyde-enhancement.md |
| features/add-reranking.md | how-to-guides/implement-search/add-reranking.md |
| features/chunking-guide.md | how-to-guides/process-documents/chunking-guide.md |
| features/embedding-models.md | how-to-guides/process-documents/embedding-models.md |
| features/vector-db-tuning.md | how-to-guides/optimize-performance/vector-db-tuning.md |
| operations/metrics-guide.md | how-to-guides/optimize-performance/monitoring.md |
| deployment/canary-deployment.md | how-to-guides/deploy/canary-deployment.md |
| api/rest-api.md | reference/api/rest-api.md |
| api/browser-api.md | reference/api/browser-api.md |
| architecture/config-schema.md | reference/configuration/config-schema.md |
| mcp/* | reference/mcp-tools/* |
| architecture/system-overview.md | concepts/architecture/system-overview.md |
| architecture/v1-architecture.md | concepts/architecture/v1-architecture.md |
| operations/troubleshooting.md | operations/monitoring/troubleshooting.md |
| development/development-setup.md | contributing/development-setup.md |
| development/testing-guide.md | contributing/testing-guide.md |
| development/architecture-guide.md | contributing/architecture-guide.md |

## Benefits

1. **Clear audience targeting**: Users know where to look based on their needs
2. **Progressive disclosure**: From tutorials → how-to → reference → concepts
3. **Better organization**: Related content grouped together
4. **Easier navigation**: Logical hierarchy
5. **Scalable**: Easy to add new content in appropriate sections

## Implementation Steps

1. Create new directory structure
2. Rename files in place (preserve git history)
3. Move files to new locations
4. Update all internal links
5. Add README.md to each directory
6. Update main docs/README.md navigation