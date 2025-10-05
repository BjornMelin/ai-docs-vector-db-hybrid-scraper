"""Agentic RAG MCP tools backed by the LangGraph runner."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import uuid4
from weakref import WeakKeyDictionary

from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.infrastructure.client_manager import ClientManager
from src.services.agents import (
    DynamicToolDiscovery,
    GraphAnalysisOutcome,
    GraphRunner,
    GraphSearchOutcome,
    RetrievalHelper,
    ToolExecutionService,
)
from src.services.errors import ToolError
from src.services.monitoring.telemetry_repository import get_telemetry_repository


logger = logging.getLogger(__name__)

SCHEMA_VERSION = "agentic_rag.v2"
RUN_COUNT_METRIC = "agentic_graph_runs_total"
LATENCY_METRIC = "agentic_graph_latency_ms"

_runner_cache: WeakKeyDictionary[ClientManager, GraphRunner] = WeakKeyDictionary()
_runner_locks: WeakKeyDictionary[ClientManager, asyncio.Lock] = WeakKeyDictionary()


class CounterSamplePayload(BaseModel):
    """Telemetry counter sample."""

    model_config = ConfigDict(extra="forbid")

    tags: dict[str, str] = Field(default_factory=dict, description="Sample tags")
    value: float = Field(0.0, description="Observed counter value")


class HistogramSamplePayload(BaseModel):
    """Telemetry histogram sample."""

    model_config = ConfigDict(extra="forbid")

    tags: dict[str, str] = Field(default_factory=dict, description="Sample tags")
    count: int = Field(0, ge=0, description="Number of observations")
    sum: float = Field(0.0, description="Sum of observations")
    values: list[float] = Field(default_factory=list, description="Raw observations")


class AgentPerformanceMetricsResponse(BaseModel):
    """Metrics response capturing counters and histograms."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    schema_version: str
    request_id: str
    counters: dict[str, list[CounterSamplePayload]] = Field(default_factory=dict)
    histograms: dict[str, list[HistogramSamplePayload]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class RunSummary(BaseModel):
    """Aggregated run summary statistics."""

    model_config = ConfigDict(extra="forbid")

    total: int = Field(0, ge=0, description="Total recorded runs")
    successful: int = Field(0, ge=0, description="Successful runs")
    samples: list[CounterSamplePayload] = Field(default_factory=list)


class AgenticOrchestrationMetricsResponse(BaseModel):
    """Aggregated orchestration metrics for LangGraph runs."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    schema_version: str
    request_id: str
    run_summary: RunSummary
    latency_samples: list[HistogramSamplePayload] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


async def _get_runner(client_manager: ClientManager) -> GraphRunner:
    """Initialise and cache the LangGraph runner for the provided client manager."""

    runner = _runner_cache.get(client_manager)
    if runner is not None:
        return runner

    lock = _runner_locks.get(client_manager)
    if lock is None:
        lock = asyncio.Lock()
        _runner_locks[client_manager] = lock

    async with lock:
        runner = _runner_cache.get(client_manager)
        if runner is not None:
            return runner

        discovery = DynamicToolDiscovery(client_manager)
        tool_service = ToolExecutionService(client_manager)
        retrieval_helper = RetrievalHelper(client_manager)
        runner = GraphRunner(
            client_manager=client_manager,
            discovery=discovery,
            tool_service=tool_service,
            retrieval_helper=retrieval_helper,
            run_timeout_seconds=30.0,
        )
        _runner_cache[client_manager] = runner
        return runner


class AgenticSearchRequest(BaseModel):
    """Request payload for agentic search."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="User query to process")
    collection: str = Field("documentation", description="Target collection")
    max_results: int = Field(8, ge=1, le=50, description="Maximum documents to return")
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
    results: list[dict[str, Any]] = Field(
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
    results: Sequence[Any] | None = payload.get("results")
    normalised_results: list[dict[str, Any]] = []
    if results:
        for item in results:
            if isinstance(item, Mapping):
                normalised_results.append(dict(item))
                continue
            try:
                normalised_results.append(dict(item))
            except (TypeError, ValueError):
                logger.debug(
                    "Unexpected search result type %s; storing as string",
                    type(item).__name__,
                )
                normalised_results.append({"value": str(item)})

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
    items: Sequence[Any] = values or []
    normalised: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, Mapping):
            normalised.append(dict(item))
        else:
            normalised.append({"message": str(item)})
    return normalised


def _safe_latency(metrics: Mapping[str, Any]) -> float:
    value = metrics.get("latency_ms", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.debug("Unable to parse latency value: %r", value)
        return 0.0


async def _run_search(
    request: AgenticSearchRequest, client_manager: ClientManager
) -> AgenticSearchResponse:
    call_id = str(uuid4())
    logger.info(
        "agentic_search started call_id=%s session_id=%s collection=%s",
        call_id,
        request.session_id,
        request.collection,
    )
    runner = await _get_runner(client_manager)
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
    request: AgenticAnalysisRequest, client_manager: ClientManager
) -> AgenticAnalysisResponse:
    call_id = str(uuid4())
    logger.info(
        "agentic_analysis started call_id=%s session_id=%s",
        call_id,
        request.session_id,
    )
    runner = await _get_runner(client_manager)
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
    """Return telemetry snapshot with warnings for any export issues."""

    try:
        snapshot = get_telemetry_repository().export_snapshot()
        return snapshot, []
    except Exception as exc:  # pragma: no cover - defensive guard.
        logger.exception("Failed to export telemetry snapshot", exc_info=exc)
        return {"counters": {}, "histograms": {}}, [f"export_error:{exc}"]


def _coerce_counter_samples(
    data: Any, warnings: list[str]
) -> dict[str, list[CounterSamplePayload]]:
    if not isinstance(data, Mapping):
        if data is not None:
            warnings.append("counters: expected mapping")
        return {}

    parsed: dict[str, list[CounterSamplePayload]] = {}
    for name, entries in data.items():
        if not isinstance(entries, Sequence):
            warnings.append(f"counters[{name}]: expected sequence")
            continue
        samples: list[CounterSamplePayload] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                type_name = type(entry).__name__
                warnings.append(
                    f"counters[{name}][{index}]: expected mapping, got {type_name}"
                )
                continue
            try:
                samples.append(CounterSamplePayload(**dict(entry)))
            except ValidationError as exc:
                warnings.append(
                    f"counters[{name}][{index}]: validation error {exc.errors()}"
                )
        if samples:
            parsed[str(name)] = samples
    return parsed


def _coerce_histogram_samples(
    data: Any, warnings: list[str]
) -> dict[str, list[HistogramSamplePayload]]:
    if not isinstance(data, Mapping):
        if data is not None:
            warnings.append("histograms: expected mapping")
        return {}

    parsed: dict[str, list[HistogramSamplePayload]] = {}
    for name, entries in data.items():
        if not isinstance(entries, Sequence):
            warnings.append(f"histograms[{name}]: expected sequence")
            continue
        samples: list[HistogramSamplePayload] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                type_name = type(entry).__name__
                warnings.append(
                    f"histograms[{name}][{index}]: expected mapping, got {type_name}"
                )
                continue
            try:
                samples.append(HistogramSamplePayload(**dict(entry)))
            except ValidationError as exc:
                warnings.append(
                    f"histograms[{name}][{index}]: validation error {exc.errors()}"
                )
        if samples:
            parsed[str(name)] = samples
    return parsed


def register_tools(mcp: FastMCP, client_manager: ClientManager) -> None:
    """Register the agentic RAG tools with the FastMCP server."""

    @mcp.tool(
        name="agentic_search",
        description="Execute the LangGraph-backed agentic search workflow",
        tags={"rag", "search"},
    )
    async def agentic_search(request: AgenticSearchRequest) -> AgenticSearchResponse:
        """Execute the LangGraph-based search workflow."""

        return await _run_search(request, client_manager)

    @mcp.tool(
        name="agentic_analysis",
        description="Run LangGraph agentic analysis over provided context",
        tags={"analysis", "rag"},
    )
    async def agentic_analysis(
        request: AgenticAnalysisRequest,
    ) -> AgenticAnalysisResponse:
        """Execute the LangGraph-based analysis workflow."""

        return await _run_analysis(request, client_manager)

    @mcp.tool(
        name="agent_performance_metrics",
        description="Inspect raw telemetry counters and histograms",
        tags={"telemetry", "metrics"},
    )
    async def get_agent_performance_metrics() -> AgentPerformanceMetricsResponse:
        """Return raw telemetry counters and histograms for agent runs."""

        snapshot, warnings = _export_metrics_snapshot()
        counters = _coerce_counter_samples(snapshot.get("counters"), warnings)
        histograms = _coerce_histogram_samples(snapshot.get("histograms"), warnings)
        return AgentPerformanceMetricsResponse(
            success=not warnings,
            schema_version=SCHEMA_VERSION,
            request_id=str(uuid4()),
            counters=counters,
            histograms=histograms,
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

        get_telemetry_repository().reset()
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
        description="Summarise LangGraph run counters and latency histograms",
        tags={"telemetry", "metrics"},
    )
    async def get_agentic_orchestration_metrics() -> (
        AgenticOrchestrationMetricsResponse
    ):
        """Return aggregated LangGraph orchestration metrics."""

        snapshot, warnings = _export_metrics_snapshot()
        counters = _coerce_counter_samples(snapshot.get("counters"), warnings)
        histograms = _coerce_histogram_samples(snapshot.get("histograms"), warnings)

        run_samples = counters.get(RUN_COUNT_METRIC, [])
        total_runs = int(sum(sample.value for sample in run_samples))
        successful_runs = int(
            sum(
                sample.value
                for sample in run_samples
                if sample.tags.get("success") == "true"
            )
        )
        latency_samples = histograms.get(LATENCY_METRIC, [])

        return AgenticOrchestrationMetricsResponse(
            success=not warnings,
            schema_version=SCHEMA_VERSION,
            request_id=str(uuid4()),
            run_summary=RunSummary(
                total=total_runs,
                successful=successful_runs,
                samples=run_samples,
            ),
            latency_samples=latency_samples,
            warnings=warnings,
        )

    logger.info("Agentic RAG MCP tools registered using LangGraph runner")
