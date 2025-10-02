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
- Legacy files (`search.py`, `indexing.py`, `documents.py`, `hybrid_search.py`)
  duplicate features available in LangChain/Qdrant and carry significant
  maintenance overhead. The new `collection_management.py` tool suite and the
  consolidated vector adapter replace these surfaces in Phase 1.

## RAG Pipeline
- ✅ `src/services/rag/generator.py` now delegates to the LangChain retriever and
  no longer carries bespoke logging TODOs after the Phase 2 rewrite.
- ✅ Model/metrics DTOs in `src/services/rag/models.py` have been trimmed to the
  new response schema, removing the legacy `search_results`/follow-up fields.
- ✅ Added `VectorServiceRetriever` abstraction with unit coverage, aligning
  LangChain retrieval with the vector adapter while avoiding test-only hooks in
  production modules.

## MCP Tools
- [x] Search adapters now delegate directly to `VectorStoreService`; legacy helper
  modules such as `_search_utils.py` were removed in Phase 3.
- [x] Document management tools route lifecycle operations through
  `VectorStoreService` CRUD methods, eliminating bespoke caching layers.
- Remaining registration helpers under `src/mcp_tools/tools/helpers` can be
  removed once the reduced service catalog has stabilised (tracked for Phase 4).

## Cross-Cutting Items within Scope
- Numerous change logs and research docs reference "legacy" behaviours. These
  will be updated during Phase 4 when documentation is refreshed.
- Contract tests are missing for the vector layer; `tests/contracts/vector_db`
  now houses a placeholder that will be filled once the adapter is implemented.

This inventory will be updated after each phase to ensure no residual debt
remains in the modules we touch.
