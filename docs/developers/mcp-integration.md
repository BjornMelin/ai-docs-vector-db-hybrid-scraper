# FastMCP Integration Guide

This document consolidates the FastMCP research previously stored under `planning/done/P3/`. It describes the current server implementation and outlines the supported optimisation paths.

## Current Layout

- Unified server entrypoint: `src/unified_mcp_server.py`
- Tool modules registered via `src/mcp_tools/tools/*`
- Dependency layer backed by `ClientManager` and the DI container
- Middleware coverage limited to logging and basic authentication

Registration currently relies on manual `register_tools` functions. This pattern remains necessary for tools with complex setup, but LangChain-aligned decorators can be introduced for simple tools.

## Recommended Patterns

### Tool Registration

Use FastMCP decorators for stateless tools:

```python
from fastmcp import FastMCP

mcp = FastMCP("Search Services")

@mcp.tool()
async def search_documents(request: SearchRequest, ctx: Context) -> SearchResults:
    return await search_documents_core(request, client_manager, ctx)
```

Keep manual modules for tools that require multi-step initialisation, custom dependency graphs, or lifecycle hooks.

### Tool Transformations

Adopt `Tool.from_tool` when only arguments change between variants:

```python
from fastmcp.tools import Tool
from fastmcp.tools.tool_transform import ArgTransform

hybrid_search = Tool.from_tool(
    search_documents,
    name="hybrid_search_docs",
    description="Hybrid semantic + sparse search",
    transform_args={
        "collection": ArgTransform(default="documents", hide=True),
        "domain": ArgTransform(default="documentation", hide=True),
    },
)
```

### Middleware

FastMCP 2.x provides middleware hooks for caching, metrics, and authentication. Integrate them centrally instead of per-tool wrappers.

```python
from fastmcp.middleware import cache, metrics

mcp.use(cache(ttl_seconds=300))
mcp.use(metrics(namespace="agentic_tools"))
```

### Server Composition

Split large servers into focused components and mount them into a unified instance:

```python
search_server = FastMCP("search")
content_server = FastMCP("content")

root_server = FastMCP("ai-docs")
root_server.mount("/search", search_server)
root_server.mount("/content", content_server)
```

This structure simplifies access control and reduces startup coupling.

## Migration Strategy

1. **Foundation**
   - Add middleware for caching and telemetry.
   - Fix outstanding task management issues by using structured task groups instead of bare `asyncio.create_task`.

2. **Selective Decorator Adoption**
   - New tools default to `@mcp.tool()` decorators.
   - Migrate existing stateless utilities incrementally.
   - Document the decision matrix in `docs/developers/service_adapters.md`.

3. **Advanced Features**
   - Enable streaming responses for large result sets.
   - Use FastMCP sessions for connection pooling instead of bespoke managers where viable.
   - Adopt FastMCP in-memory transports for unit tests and keep integration coverage for end-to-end flows.

## Testing

- Unit tests for tool modules live under `tests/unit/mcp_tools/`.
- Integration tests exercise MCP servers via CLI commands in `tests/integration/mcp/`.
- When enabling new middleware, add regression coverage to `tests/unit/mcp_tools/test_server.py` to ensure lifespans shut down cleanly.

## Reference Checklist

- [ ] Every tool exposes accurate JSON schema (verify with `mcp describe` output).
- [ ] Middleware stack includes caching, metrics, and error handling.
- [ ] Server composition documented in `docs/operators/operations.md` for runbooks.
- [ ] Streaming endpoints declare chunk semantics and include tests.

This guide supersedes `planning/done/P3/B1_mcp_framework_optimization_analysis.md`, `planning/done/P3/B2_mcp_framework_optimization_dual.md`, `planning/done/P3/C1_fastmcp_integration_analysis.md`, and `planning/done/P3/C2_fastmcp_integration_analysis_dual.md`.
