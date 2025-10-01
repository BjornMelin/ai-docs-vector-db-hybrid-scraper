# Technical Debt Inventory (Phase 0)

This document captures the current debt landscape for modules touched during the
vector/RAG/MCP consolidation. Each item is scheduled for remediation within the
phase that replaces the corresponding surface.

## Vector Database Layer
- `src/services/vector_db/model_selector.py` contains numerous TODOs around
  proper typing (`QueryClassification`, `ModelSelectionStrategy`) and logging
  formatting. These modules will be deleted when the new adapter lands
  (Phase 1).
- `src/services/vector_db/query_classifier.py` exposes TODOs for replacing
  ad-hoc types. The new adapter removes the need for these bespoke classifiers.
- Legacy files (`collections.py`, `search.py`, `indexing.py`, `documents.py`,
  `hybrid_search.py`) duplicate features available in LangChain/Qdrant and
  carry significant maintenance overhead. Entire directory will be replaced in
  Phase 1.

## RAG Pipeline
- `src/services/rag/generator.py` still relies on `TODO: Convert f-string to
  logging format` annotations. The LangChain rewrite in Phase 2 will drop the
  existing generator implementation and eliminate these TODOs.
- Model/metrics classes in `src/services/rag/models.py` include fields unused by
  the streamlined pipeline; they will be pruned alongside the new DTOs.

## MCP Tools
- `src/mcp_tools/tools/helpers/tool_registrars.py` has TODO markers for logging
  consistency. Phase 3 consolidation replaces tool registration with a reduced
  surface, removing this helper entirely.
- Redundant tool modules (`search.py`, `search_tools.py`, `documents.py`, etc.)
  introduce divergent schemas and duplicated logic. All will be replaced by thin
  adapters that delegate to the consolidated vector/RAG services.

## Cross-Cutting Items within Scope
- Numerous change logs and research docs reference "legacy" behaviours. These
  will be updated during Phase 4 when documentation is refreshed.
- Contract tests are missing for the vector layer; `tests/contracts/vector_db`
  now houses a placeholder that will be filled once the adapter is implemented.

This inventory will be updated after each phase to ensure no residual debt
remains in the modules we touch.
