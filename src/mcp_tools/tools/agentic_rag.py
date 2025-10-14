"""Agentic RAG MCP tools backed by the LangGraph runner."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import uuid4

from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.contracts.retrieval import SearchRecord
from src.services.agents import (
    GraphAnalysisOutcome,
    GraphRunner,
    GraphSearchOutcome,
)
from src.services.errors import ToolError
from src.services.observability.tracking import get_ai_tracker


logger = logging.getLogger(__name__)

SCHEMA_VERSION = "agentic_rag.v2"
RUN_OPERATION = "agent.graph.run"

_RUNNER_LOCK = asyncio.Lock()
_RUNNER_INSTANCE: GraphRunner | None = None


class OperationSamplePayload(BaseModel):
    """Aggregated statistics for a tracked operation."""

    model_config = ConfigDict(extra="forbid")

    count: int = Field(0, ge=0, description="Total recorded operations")
    success_count: int = Field(0, ge=0, description="Successful operations")
    total_duration_s: float = Field(
        0.0, ge=0.0, description="Cumulative duration in seconds"
    )
    total_tokens: int = Field(0, ge=0, description="Tokens processed")
    total_cost_usd: float = Field(0.0, ge=0.0, description="Cost incurred in USD")


class AgentPerformanceMetricsResponse(BaseModel):
    """Metrics response exposing aggregated operation statistics."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    schema_version: str
    request_id: str
    operations: dict[str, OperationSamplePayload] = Field(
        default_factory=dict, description="Aggregated metrics keyed by operation"
    )
    warnings: list[str] = Field(default_factory=list)


class RunSummary(BaseModel):
    """Aggregated run summary statistics."""

    model_config = ConfigDict(extra="forbid")

    total_runs: int = Field(0, ge=0, description="Total recorded runs")
    successful_runs: int = Field(0, ge=0, description="Successful runs")
    total_duration_s: float = Field(
        0.0, ge=0.0, description="Cumulative duration for all runs"
    )


class AgenticOrchestrationMetricsResponse(BaseModel):
    """Aggregated orchestration metrics for LangGraph runs."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    schema_version: str
    request_id: str
    run_summary: RunSummary
    operations: dict[str, OperationSamplePayload] = Field(
        default_factory=dict, description="Aggregated metrics keyed by operation"
    )
    warnings: list[str] = Field(default_factory=list)


async def _get_runner(override: GraphRunner | None = None) -> GraphRunner:
    """Initialise and cache the LangGraph runner instance."""
    if override is not None:
        return override

    global _RUNNER_INSTANCE  # pylint: disable=global-statement
    if _RUNNER_INSTANCE is not None:
        return _RUNNER_INSTANCE

    async with _RUNNER_LOCK:
        if _RUNNER_INSTANCE is None:
            _, _, _, runner = GraphRunner.build_components()
            _RUNNER_INSTANCE = runner
    return _RUNNER_INSTANCE


class AgenticSearchRequest(BaseModel):
    """Request payload for agentic search."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="User query to process")
    collection: str = Field("documentation", description="Target collection")
    max_results: int | None = Field(
        None, ge=1, le=50, description="Maximum documents to return"
    )
    session_id: str | None = Field(None, description="Session identifier")
    user_context: dict[str, Any] | None = Field(
        None, description="Optional user context"
    )
    filters: dict[str, Any] | None = Field(None, description="Metadata filters")


class AgenticAnalysisRequest(BaseModel):
    """Request payload for agentic analysis."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., description="Analysis prompt")
    data: list[dict[str, Any]] = Field(
        default_factory=list, description="Context documents"
    )
    session_id: str | None = Field(None, description="Session identifier")
    user_context: dict[str, Any] | None = Field(
        None, description="Optional user context"
    )


class AgenticSearchResponse(BaseModel):
    """Response payload for agentic search."""

    model_config = ConfigDict(extra="forbid")

    success: bool = Field(
        ..., description="Whether the workflow completed successfully"
    )
    session_id: str = Field(..., description="Session identifier")
    results: list[SearchRecord] = Field(
        default_factory=list, description="Search results"
    )
    answer: str | None = Field(None, description="Generated answer")
    confidence: float | None = Field(None, description="Confidence heuristic")
    tools_used: list[str] = Field(default_factory=list, description="Tools executed")
    reasoning: list[str] = Field(
        default_factory=list, description="Internal trace lines"
    )
    total_latency_ms: float = Field(
        ..., description="Total workflow latency in milliseconds"
    )
    metrics: dict[str, Any] = Field(default_factory=dict, description="Stage metrics")
    errors: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured errors collected during execution",
    )
    schema_version: str = Field(SCHEMA_VERSION, description="Schema version marker")


class AgenticAnalysisResponse(BaseModel):
    """Response payload for agentic analysis."""

    model_config = ConfigDict(extra="forbid")

    success: bool = Field(
        ..., description="Whether the workflow completed successfully"
    )
    analysis_id: str = Field(..., description="Analysis identifier")
    summary: str = Field(..., description="Workflow summary")
    insights: dict[str, Any] = Field(
        default_factory=dict, description="Structured insights"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Follow-up recommendations"
    )
    confidence: float | None = Field(None, description="Confidence heuristic")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Stage metrics")
    errors: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured errors collected during execution",
    )
    schema_version: str = Field(SCHEMA_VERSION, description="Schema version marker")


def _build_search_response(
    *, success: bool, session_id: str, payload: Mapping[str, Any]
) -> AgenticSearchResponse:
    """Build an AgenticSearchResponse from raw payload data."""
    results: Sequence[Any] | None = payload.get("results")
    normalised_results: list[SearchRecord] = []
    if results:
        for item in results:
            try:
                normalised_results.append(SearchRecord.from_payload(item))
            except (TypeError, ValidationError) as exc:
                logger.debug(
                    "Unexpected search result type %s: %s",
                    type(item).__name__,
                    exc,
                )
                fallback = {
                    "id": str(uuid4()),
                    "content": str(item),
                    "score": 0.0,
                }
                normalised_results.append(SearchRecord.model_validate(fallback))

    tools_used: Sequence[str] | None = payload.get("tools_used")
    reasoning: Sequence[str] | None = payload.get("reasoning")
    metrics: Mapping[str, Any] | None = payload.get("metrics")
    normalised_errors = _normalise_errors(payload.get("errors"))

    return AgenticSearchResponse(
        success=success,
        session_id=session_id,
        results=normalised_results,
        answer=payload.get("answer"),
        confidence=payload.get("confidence"),
        tools_used=list(tools_used or []),
        reasoning=list(reasoning or []),
        total_latency_ms=float(payload.get("latency_ms", 0.0)),
        metrics=dict(metrics or {}),
        errors=normalised_errors,
        schema_version=SCHEMA_VERSION,
    )


def _build_analysis_response(
    *, success: bool, analysis_id: str, payload: Mapping[str, Any]
) -> AgenticAnalysisResponse:
    """Build an AgenticAnalysisResponse from raw payload data."""
    return AgenticAnalysisResponse(
        success=success,
        analysis_id=analysis_id,
        summary=(payload.get("summary") or ""),
        insights=dict(payload.get("insights") or {}),
        recommendations=list(payload.get("recommendations") or []),
        confidence=payload.get("confidence"),
        metrics=dict(payload.get("metrics") or {}),
        errors=_normalise_errors(payload.get("errors")),
        schema_version=SCHEMA_VERSION,
    )


def _normalise_errors(values: Sequence[Any] | None) -> list[dict[str, Any]]:
    """Coerce error payloads into a stable list of dictionaries."""
    items: Sequence[Any] = values or []
    normalised: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, Mapping):
            normalised.append(dict(item))
        else:
            normalised.append({"message": str(item)})
    return normalised


def _safe_latency(metrics: Mapping[str, Any]) -> float:
    """Extract a safe latency value from the metrics dictionary."""
    value = metrics.get("latency_ms", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.debug("Unable to parse latency value: %r", value)
        return 0.0


async def _run_search(
    request: AgenticSearchRequest, graph_runner: GraphRunner | None = None
) -> AgenticSearchResponse:
    """Execute an agentic search and return the response."""
    call_id = str(uuid4())
    logger.info(
        "agentic_search started call_id=%s session_id=%s collection=%s",
        call_id,
        request.session_id,
        request.collection,
    )
    runner = await _get_runner(graph_runner)
    try:
        outcome: GraphSearchOutcome = await runner.run_search(
            query=request.query,
            collection=request.collection,
            session_id=request.session_id,
            top_k=request.max_results,
            filters=request.filters,
            user_context=request.user_context,
        )
    except ToolError as exc:  # Surface known tool failures cleanly.
        logger.warning("agentic_search tool failure call_id=%s error=%s", call_id, exc)
        session_id = request.session_id or call_id
        return _build_search_response(
            success=False,
            session_id=session_id,
            payload={
                "results": [],
                "answer": None,
                "confidence": None,
                "tools_used": [],
                "reasoning": [],
                "metrics": {},
                "errors": [
                    {
                        "source": "tool_execution",
                        "code": getattr(exc, "error_code", "TOOL_ERROR"),
                        "message": str(exc),
                    }
                ],
                "latency_ms": 0.0,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive guard.
        logger.exception(
            "agentic_search unexpected error call_id=%s", call_id, exc_info=exc
        )
        session_id = request.session_id or call_id
        return _build_search_response(
            success=False,
            session_id=session_id,
            payload={
                "results": [],
                "answer": None,
                "confidence": None,
                "tools_used": [],
                "reasoning": [],
                "metrics": {},
                "errors": [
                    {
                        "source": "graph",
                        "code": "INTERNAL_ERROR",
                        "message": "Unexpected failure",
                    }
                ],
                "latency_ms": 0.0,
            },
        )

    metrics = dict(getattr(outcome, "metrics", {}) or {})
    errors = list(getattr(outcome, "errors", []) or [])
    results = list(getattr(outcome, "results", []) or [])
    tools_used = list(getattr(outcome, "tools_used", []) or [])
    reasoning = list(getattr(outcome, "reasoning", []) or [])
    session_id = getattr(outcome, "session_id", None) or request.session_id or call_id

    response = _build_search_response(
        success=bool(getattr(outcome, "success", False)),
        session_id=session_id,
        payload={
            "results": results,
            "answer": getattr(outcome, "answer", None),
            "confidence": getattr(outcome, "confidence", None),
            "tools_used": tools_used,
            "reasoning": reasoning,
            "metrics": metrics,
            "errors": errors,
            "latency_ms": _safe_latency(metrics),
        },
    )
    logger.info(
        "agentic_search completed call_id=%s session_id=%s success=%s results=%d",
        call_id,
        session_id,
        response.success,
        len(response.results),
    )
    return response


async def _run_analysis(
    request: AgenticAnalysisRequest, graph_runner: GraphRunner | None = None
) -> AgenticAnalysisResponse:
    """Execute an agentic analysis and return the response."""
    call_id = str(uuid4())
    logger.info(
        "agentic_analysis started call_id=%s session_id=%s",
        call_id,
        request.session_id,
    )
    runner = await _get_runner(graph_runner)
    try:
        outcome: GraphAnalysisOutcome = await runner.run_analysis(
            query=request.query,
            session_id=request.session_id,
            context_documents=request.data,
            user_context=request.user_context,
        )
    except ToolError as exc:
        logger.warning(
            "agentic_analysis tool failure call_id=%s error=%s", call_id, exc
        )
        analysis_id = request.session_id or call_id
        return _build_analysis_response(
            success=False,
            analysis_id=analysis_id,
            payload={
                "summary": None,
                "insights": {},
                "recommendations": [],
                "confidence": None,
                "metrics": {},
                "errors": [
                    {
                        "source": "tool_execution",
                        "code": getattr(exc, "error_code", "TOOL_ERROR"),
                        "message": str(exc),
                    }
                ],
            },
        )
    except Exception as exc:  # pragma: no cover - defensive guard.
        logger.exception(
            "agentic_analysis unexpected error call_id=%s", call_id, exc_info=exc
        )
        analysis_id = request.session_id or call_id
        return _build_analysis_response(
            success=False,
            analysis_id=analysis_id,
            payload={
                "summary": None,
                "insights": {},
                "recommendations": [],
                "confidence": None,
                "metrics": {},
                "errors": [
                    {
                        "source": "graph",
                        "code": "INTERNAL_ERROR",
                        "message": "Unexpected failure",
                    }
                ],
            },
        )

    metrics = dict(getattr(outcome, "metrics", {}) or {})
    errors = list(getattr(outcome, "errors", []) or [])
    analysis_id = getattr(outcome, "analysis_id", None) or request.session_id or call_id

    response = _build_analysis_response(
        success=bool(getattr(outcome, "success", False)),
        analysis_id=analysis_id,
        payload={
            "summary": getattr(outcome, "summary", ""),
            "insights": getattr(outcome, "insights", {}),
            "recommendations": getattr(outcome, "recommendations", []),
            "confidence": getattr(outcome, "confidence", None),
            "metrics": metrics,
            "errors": errors,
        },
    )
    logger.info(
        "agentic_analysis completed call_id=%s analysis_id=%s success=%s",
        call_id,
        analysis_id,
        response.success,
    )
    return response


def _export_metrics_snapshot() -> tuple[dict[str, Any], list[str]]:
    """Return tracker snapshot with warnings for any export issues."""
    try:
        snapshot = get_ai_tracker().snapshot()
        return snapshot, []
    except Exception as exc:  # pragma: no cover - defensive guard.
        logger.exception("Failed to export telemetry snapshot", exc_info=exc)
        return {}, [f"export_error:{exc}"]


def _parse_operation_samples(
    data: Any, warnings: list[str]
) -> dict[str, OperationSamplePayload]:
    """Validate and coerce tracker snapshots into payload objects."""
    if not isinstance(data, Mapping):
        if data is not None:
            warnings.append("operations: expected mapping")
        return {}

    parsed: dict[str, OperationSamplePayload] = {}
    for name, entry in data.items():
        if not isinstance(entry, Mapping):
            warnings.append(f"operations[{name}]: expected mapping")
            continue
        try:
            parsed[str(name)] = OperationSamplePayload(**dict(entry))
        except ValidationError as exc:
            warnings.append(f"operations[{name}]: validation error {exc.errors()}")
    return parsed


def register_tools(
    mcp: FastMCP,
    graph_runner: GraphRunner | None = None,
) -> None:
    """Register the agentic RAG tools with the FastMCP server."""

    @mcp.tool(
        name="agentic_search",
        description="Execute the LangGraph-backed agentic search workflow",
        tags={"rag", "search"},
    )
    async def agentic_search(request: AgenticSearchRequest) -> AgenticSearchResponse:
        """Execute the LangGraph-based search workflow."""
        return await _run_search(request, graph_runner)

    @mcp.tool(
        name="agentic_analysis",
        description="Run LangGraph agentic analysis over provided context",
        tags={"analysis", "rag"},
    )
    async def agentic_analysis(
        request: AgenticAnalysisRequest,
    ) -> AgenticAnalysisResponse:
        """Execute the LangGraph-based analysis workflow."""
        return await _run_analysis(request, graph_runner)

    @mcp.tool(
        name="agent_performance_metrics",
        description="Inspect aggregated in-memory agent telemetry",
        tags={"telemetry", "metrics"},
    )
    async def get_agent_performance_metrics() -> AgentPerformanceMetricsResponse:
        """Return aggregated tracker statistics for agent operations."""
        snapshot, warnings = _export_metrics_snapshot()
        operations = _parse_operation_samples(snapshot, warnings)
        return AgentPerformanceMetricsResponse(
            success=not warnings,
            schema_version=SCHEMA_VERSION,
            request_id=str(uuid4()),
            operations=operations,
            warnings=warnings,
        )

    @mcp.tool(
        name="reset_agent_learning",
        description="Reset in-memory agent telemetry counters",
        tags={"telemetry", "maintenance"},
    )
    async def reset_agent_learning(confirm: bool = False) -> dict[str, Any]:
        """Reset the in-memory telemetry snapshot."""
        if not confirm:
            raise ToolError(
                "Confirmation required; call with confirm=True to reset telemetry",
                error_code="CONFIRMATION_REQUIRED",
                context={"request_id": str(uuid4())},
            )

        get_ai_tracker().reset()
        return {
            "success": True,
            "schema_version": SCHEMA_VERSION,
            "request_id": str(uuid4()),
            "message": "Agent telemetry reset",
        }

    @mcp.tool(
        name="optimize_agent_configuration",
        description="Placeholder for future optimisation workflow",
        tags={"configuration"},
    )
    async def optimize_agent_configuration(
        optimization_target: str = "balanced",
        constraints: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Raise an informative error for unavailable optimisation features."""
        raise ToolError(
            "Agent configuration optimisation is not implemented",
            error_code="NOT_IMPLEMENTED",
            context={
                "request_id": str(uuid4()),
                "optimization_target": optimization_target,
                "constraints": constraints or {},
            },
        )

    @mcp.tool(
        name="agentic_orchestration_metrics",
        description="Summarise LangGraph run performance from the tracker",
        tags={"telemetry", "metrics"},
    )
    async def get_agentic_orchestration_metrics() -> (
        AgenticOrchestrationMetricsResponse
    ):
        """Return aggregated LangGraph orchestration metrics."""
        snapshot, warnings = _export_metrics_snapshot()
        operations = _parse_operation_samples(snapshot, warnings)
        run_stats = operations.get(RUN_OPERATION)
        summary = RunSummary(
            total_runs=run_stats.count if run_stats else 0,
            successful_runs=run_stats.success_count if run_stats else 0,
            total_duration_s=run_stats.total_duration_s if run_stats else 0.0,
        )

        return AgenticOrchestrationMetricsResponse(
            success=not warnings,
            schema_version=SCHEMA_VERSION,
            request_id=str(uuid4()),
            run_summary=summary,
            operations=operations,
            warnings=warnings,
        )

    logger.info("Agentic RAG MCP tools registered using LangGraph runner")
