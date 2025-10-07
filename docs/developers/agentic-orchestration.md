# Agentic Orchestration

This guide documents the LangChain/LangGraph orchestration stack that replaced the legacy Pydantic-based agents. Use it to understand how agent workflows are assembled, where to extend them, and how to validate changes.

## Architecture Overview

The orchestration entry point is `GraphRunner` (`src/services/agents/langgraph_runner.py`). Key collaborators:

- `DynamicToolDiscovery` (`src/services/agents/dynamic_tool_discovery.py`) enumerates active tools from MCP servers and classifies them into capability groups.
- `ToolExecutionService` (`src/services/agents/tool_execution_service.py`) executes selected tools over `langchain_mcp_adapters.MultiServerMCPClient`, normalises errors, and enforces timeouts.
- `RetrievalHelper` (`src/services/agents/retrieval.py`) performs dense + sparse retrieval using LangChain vector abstractions.
- `ClientManager` (`src/infrastructure/client_manager.py`) provides shared transports and configuration handles.

`GraphRunner` composes these components into a LangGraph `StateGraph` with the following nodes:

```
entry -> discover -> retrieve -> execute -> respond -> END
```

Conditional edges route failures to retry, fallback, or failure sinks depending on `ToolExecutionError` subclasses.

## LangGraph Configuration

- State storage uses `MemorySaver` by default. Long-running sessions can switch to a persistent saver (SQLite, Redis) by injecting an alternative checkpointer when constructing `GraphRunner`.
- Every invocation accepts a `RunnableConfig`. Callers pass correlation identifiers, telemetry callbacks, and per-request overrides (timeouts, tool filters) through this config rather than bespoke kwargs.
- Prometheus metrics (`agentic_graph_runs_total`, `agentic_graph_latency_ms`, etc.) and OpenTelemetry spans are emitted inside each node. The metrics registry lives in `src/services/monitoring/telemetry_repository.py`.

## Tool Discovery

`DynamicToolDiscovery` resolves active tools by querying MCP endpoints and mapping them to internal `ToolCapability` enums. The discovery stage supplies:

- Capability metadata (primary action, provider, cost hints).
- Tool schemas as JSON definitions gathered from MCP responses.
- Heuristics for filtering tools based on query intent and requested mode (analysis vs search).

Planned enhancements:

- Replace `ToolCapability` enums with LangChain `StructuredTool` instances to eliminate duplicated schemas.
- Persist discovery results in the configured LangGraph checkpointer to reuse capability scores across retries.

## Tool Execution

`ToolExecutionService` wraps MCP calls with:

- Async timeouts per tool invocation.
- Structured exception mapping (`ToolExecutionTimeout`, `ToolExecutionInvalidArgument`, `ToolExecutionFailure`).
- Normalised payload handling so every tool returns dictionaries with consistent keys (`output`, `metadata`).

Streamed tools use async generators; the runner collects batches and appends them to the state under `tool_outputs`.

## Retrieval and RAG Support

The retrieval node delegates to `RetrievalHelper`:

- Dense embeddings via FastEmbed integrations in `src/services/embeddings/fastembed_provider.py`.
- Sparse scoring using `langchain_qdrant.FastEmbedSparse` when configured.
- Adaptive limits: defaults come from `client_manager.config.agentic.retrieval_limit` but can be overridden through the `RunnableConfig` metadata.

Self-healing behaviour relies on rerun policies inside the graph: failed retrieval increments an error counter and triggers alternate tool sets on retry.

## Parallel Coordination

When `client_manager.config.agentic.max_parallel_tools > 1`, the execution node spawns parallel tool tasks up to that limit. Results are merged deterministically by tool name and execution order. Errors from individual tools are recorded without aborting the entire run unless every selected tool fails.

## Testing Strategy

- Unit tests for dependency wiring: `tests/unit/services/fastapi/test_dependencies_core.py` ensures FastAPI dependencies surface a configured `GraphRunner`.
- Tool discovery coverage: `tests/unit/services/browser/test_unified_manager.py` exercises capability hydration for browser automation tools.
- End-to-end graph behaviour is validated through integration targets under `tests/integration/rag/` (invoke orchestrator, assert on aggregated answers and metrics).

## Extension Checklist

1. Add new tool via MCP server → ensure `DynamicToolDiscovery` classifies it, and update capability mapping if necessary.
2. Update `ToolExecutionService` to handle new error types; add tests under `tests/unit/services/agents/`.
3. If a workflow requires additional state fields, extend `AgenticGraphState` and backfill defaults for older checkpoints.
4. Document operational implications in `docs/operators/operations.md` (see Browser automation and Retrieval sections).

This document supersedes `planning/done/P3/A1_langchain_integration_analysis.md`, `planning/done/P3/A2_langchain_integration_analysis_dual.md`, and the agentic sections in the P2 research reports.
