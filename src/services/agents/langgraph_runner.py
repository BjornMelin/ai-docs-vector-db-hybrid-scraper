"""LangGraph-powered agentic orchestration runner."""

# pylint: disable=too-many-instance-attributes,too-many-arguments

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypedDict, cast
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from opentelemetry import trace

from src.contracts.retrieval import SearchRecord
from src.infrastructure.client_manager import ClientManager
from src.services.agents.dynamic_tool_discovery import (
    DynamicToolDiscovery,
    ToolCapability,
    ToolCapabilityType,
)
from src.services.agents.retrieval import (
    RetrievalHelper,
    RetrievalQuery,
    RetrievedDocument,
)
from src.services.agents.tool_execution_service import (
    ToolExecutionError,
    ToolExecutionFailure,
    ToolExecutionInvalidArgument,
    ToolExecutionResult,
    ToolExecutionService,
    ToolExecutionTimeout,
)
from src.services.observability.tracking import record_ai_operation


logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

_OP_AGENT_RUN = "agent.graph.run"
_OP_AGENT_DISCOVERY_ERROR = "agent.graph.discovery.error"
_OP_AGENT_RETRIEVAL = "agent.graph.retrieval"


def _serialise_document(document: Any) -> dict[str, Any]:
    """Return a JSON-serialisable representation of a retrieval document."""

    identifier = getattr(document, "id", None)
    score = float(getattr(document, "score", 0.0))
    metadata = getattr(document, "metadata", None)
    if metadata is None and hasattr(document, "dict"):
        metadata = document.dict().get("metadata")  # type: ignore[call-arg]
    return {"id": identifier, "score": score, "metadata": metadata}


class AgenticGraphState(TypedDict, total=False):
    """State container exchanged between LangGraph nodes."""

    mode: Any
    query: Any
    collection: Any
    filters: Any
    session_id: Any
    user_context: Any
    intent: Any
    retrieval_limit: Any
    discovered_tools: Any
    selected_tools: Any
    retrieved_documents: Any
    tool_outputs: Any
    agent_notes: Any
    errors: Any
    answer: Any
    confidence: Any
    metrics: Any
    success: Any
    start_time: Any


class AgentErrorCode(str, Enum):
    """Normalized error codes surfaced by the LangGraph runner."""

    RUN_TIMEOUT = "RUN_TIMEOUT"
    DISCOVERY_ERROR = "DISCOVERY_ERROR"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    TOOL_INVALID_ARGUMENT = "TOOL_INVALID_ARGUMENT"
    TOOL_FAILURE = "TOOL_FAILURE"
    TOOL_UNEXPECTED = "TOOL_UNEXPECTED"


def _error_entry(source: str, code: AgentErrorCode, **extra: Any) -> dict[str, Any]:
    """Construct a structured error payload with optional metadata."""

    entry: dict[str, Any] = {"source": source, "code": code.value}
    for key, value in extra.items():
        if value is not None:
            entry[key] = value
    return entry


def _map_tool_error_code(exc: ToolExecutionError) -> AgentErrorCode:
    """Translate ToolExecutionError subclasses into canonical codes."""

    if isinstance(exc, ToolExecutionTimeout):
        return AgentErrorCode.TOOL_TIMEOUT
    if isinstance(exc, ToolExecutionInvalidArgument):
        return AgentErrorCode.TOOL_INVALID_ARGUMENT
    if isinstance(exc, ToolExecutionFailure):
        return AgentErrorCode.TOOL_FAILURE
    return AgentErrorCode.TOOL_UNEXPECTED


@dataclass(slots=True)
class GraphSearchOutcome:  # pylint: disable=too-many-instance-attributes
    """Structured result produced by ``run_search``."""

    success: bool
    session_id: str
    answer: str | None
    confidence: float | None
    results: list[SearchRecord]
    tools_used: list[str]
    reasoning: list[str]
    metrics: dict[str, Any]
    errors: list[dict[str, Any]]


@dataclass(slots=True)
class GraphAnalysisOutcome:  # pylint: disable=too-many-instance-attributes
    """Structured result produced by ``run_analysis``."""

    success: bool
    analysis_id: str
    summary: str
    insights: dict[str, Any]
    recommendations: list[str]
    confidence: float | None
    metrics: dict[str, Any]
    errors: list[dict[str, Any]]


class GraphRunner:  # pylint: disable=too-many-instance-attributes
    """Execute agentic workflows via a LangGraph state machine."""

    def __init__(
        self,
        client_manager: ClientManager,
        discovery: DynamicToolDiscovery,
        tool_service: ToolExecutionService,
        *,
        retrieval_helper: RetrievalHelper | None = None,
        max_parallel_tools: int = 3,
        run_timeout_seconds: float | None = None,
        retrieval_limit: int = 8,
    ) -> None:
        self._client_manager = client_manager
        self._discovery = discovery
        self._tool_service = tool_service
        self._retrieval_helper = retrieval_helper or RetrievalHelper(client_manager)
        self._max_parallel_tools = max(1, max_parallel_tools)
        self._run_timeout_seconds = run_timeout_seconds
        self._default_retrieval_limit = max(1, retrieval_limit)
        self._graph = self._build_graph()

    @classmethod
    def build_components(
        cls, client_manager: ClientManager
    ) -> tuple[
        DynamicToolDiscovery,
        ToolExecutionService,
        RetrievalHelper,
        GraphRunner,
    ]:
        """Construct discovery, execution service, retrieval helper, and runner."""

        discovery = DynamicToolDiscovery(client_manager)
        tool_service = ToolExecutionService(client_manager)
        retrieval_helper = RetrievalHelper(client_manager)
        agentic_cfg = getattr(client_manager.config, "agentic", None)
        max_parallel = getattr(agentic_cfg, "max_parallel_tools", 3)
        run_timeout = getattr(agentic_cfg, "run_timeout_seconds", 30.0)
        retrieval_limit = getattr(agentic_cfg, "retrieval_limit", 8)
        runner = cls(
            client_manager=client_manager,
            discovery=discovery,
            tool_service=tool_service,
            retrieval_helper=retrieval_helper,
            max_parallel_tools=max_parallel,
            run_timeout_seconds=run_timeout,
            retrieval_limit=retrieval_limit,
        )
        return discovery, tool_service, retrieval_helper, runner

    async def run_search(
        self,
        *,
        query: str,
        collection: str,
        session_id: str | None = None,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
        user_context: Mapping[str, Any] | None = None,
    ) -> GraphSearchOutcome:
        """Execute the search workflow and return a structured outcome.

        Args:
            query: Natural-language search query.
            collection: Vector collection to query.
            session_id: Optional session identifier for checkpointing.
            top_k: Maximum number of documents to retrieve.
            filters: Optional metadata filters applied to the vector search.
            user_context: Optional per-request context forwarded to tools.

        Returns:
            ``GraphSearchOutcome`` containing retrieval results and operational
            telemetry.
        """

        session_identifier = session_id or str(uuid4())
        effective_top_k = (
            self._default_retrieval_limit if top_k is None else max(1, top_k)
        )
        state: AgenticGraphState = {
            "mode": "search",
            "query": query,
            "collection": collection,
            "filters": filters or {},
            "session_id": session_identifier,
            "user_context": user_context or {},
            "retrieval_limit": effective_top_k,
            "agent_notes": [],
            "errors": [],
            "start_time": time.perf_counter(),
        }
        config: RunnableConfig = {"configurable": {"thread_id": session_identifier}}
        final_state = await self._invoke_with_timeout(state, config)
        return self._to_search_outcome(final_state)

    async def run_analysis(
        self,
        *,
        query: str,
        session_id: str | None = None,
        context_documents: Sequence[Mapping[str, Any]] | None = None,
        user_context: Mapping[str, Any] | None = None,
    ) -> GraphAnalysisOutcome:
        """Execute the analysis workflow and return a structured outcome.

        Args:
            query: High-level analysis prompt.
            session_id: Optional session identifier for checkpointing.
            context_documents: Documents injected directly into the state prior
                to synthesis.
            user_context: Optional per-request preferences forwarded to tools.

        Returns:
            ``GraphAnalysisOutcome`` describing the aggregated analysis result.
        """

        session_identifier = session_id or str(uuid4())
        state: AgenticGraphState = {
            "mode": "analysis",
            "query": query,
            "collection": None,
            "filters": {},
            "session_id": session_identifier,
            "user_context": user_context or {},
            "retrieval_limit": 0,
            "agent_notes": [],
            "errors": [],
            "retrieved_documents": [
                _serialise_document(
                    RetrievedDocument(
                        id=str(item.get("id", idx)),
                        score=float(item.get("score", 0.0)),
                        metadata=item.get("metadata")
                        if "metadata" in item
                        else item.get("payload"),
                        raw=None,
                    )
                )
                for idx, item in enumerate(context_documents or [])
            ],
            "start_time": time.perf_counter(),
        }
        config: RunnableConfig = {"configurable": {"thread_id": session_identifier}}
        final_state = await self._invoke_with_timeout(state, config)
        return self._to_analysis_outcome(final_state)

    async def _invoke_with_timeout(
        self, state: AgenticGraphState, config: RunnableConfig
    ) -> AgenticGraphState:
        session_identifier = state.get("session_id", "unknown")
        attrs = {
            "agent.session_id": session_identifier,
            "agent.mode": state.get("mode", "search"),
        }
        with tracer.start_as_current_span("agent.run", attributes=attrs):
            try:
                if self._run_timeout_seconds is not None:
                    raw_state = await asyncio.wait_for(
                        self._graph.ainvoke(state, config=config),
                        timeout=self._run_timeout_seconds,
                    )
                else:
                    raw_state = await self._graph.ainvoke(state, config=config)
                return cast(AgenticGraphState, raw_state)
            except TimeoutError:
                now = time.perf_counter()
                start_timestamp = float(state.get("start_time", now))
                duration_ms = (now - start_timestamp) * 1000.0
                record_ai_operation(
                    operation_type=_OP_AGENT_RUN,
                    provider=str(state.get("mode", "search")),
                    model="timeout",
                    duration_s=duration_ms / 1000.0,
                    success=False,
                )
                errors = list(state.get("errors", []))
                errors.append(
                    _error_entry(
                        "graph",
                        AgentErrorCode.RUN_TIMEOUT,
                        message="Workflow timed out",
                    )
                )
                metrics = {
                    "latency_ms": duration_ms,
                    "tool_count": len(state.get("tool_outputs", []) or []),
                    "error_count": len(errors),
                }
                timeout_state = cast(
                    AgenticGraphState,
                    {**state, "errors": errors, "metrics": metrics, "success": False},
                )
                return timeout_state

    def _build_graph(self):
        graph = StateGraph(AgenticGraphState)
        graph.add_node("intent", self._intent_node)
        graph.add_node("tool_discovery", self._tool_discovery_node)
        graph.add_node("retrieval", self._retrieval_node)
        graph.add_node("tool_execution", self._tool_execution_node)
        graph.add_node("synthesis", self._synthesis_node)
        graph.add_node("metrics", self._metrics_node)
        graph.set_entry_point("intent")
        graph.add_edge("intent", "tool_discovery")
        graph.add_edge("tool_discovery", "retrieval")
        graph.add_edge("retrieval", "tool_execution")
        graph.add_edge("tool_execution", "synthesis")
        graph.add_edge("synthesis", "metrics")
        graph.add_edge("metrics", END)
        memory = (
            MemorySaver()
        )  # NOTE: replace with persistent checkpoint store when available
        return graph.compile(checkpointer=memory)

    async def _intent_node(self, state: AgenticGraphState) -> AgenticGraphState:
        mode = state.get("mode", "search")
        notes = list(state.get("agent_notes", []))
        with tracer.start_as_current_span(
            "agent.intent", attributes={"agent.mode": mode}
        ):
            notes.append(f"intent_selected:{mode}")
        return {"intent": mode, "agent_notes": notes}

    async def _tool_discovery_node(self, state: AgenticGraphState) -> AgenticGraphState:
        notes = list(state.get("agent_notes", []))
        try:
            await self._discovery.refresh()
            capabilities = list(self._discovery.get_capabilities())
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Tool discovery failed: %s", exc, exc_info=True)
            errors = list(state.get("errors", []))
            errors.append(
                _error_entry(
                    "discovery", AgentErrorCode.DISCOVERY_ERROR, message=str(exc)
                )
            )
            record_ai_operation(
                operation_type=_OP_AGENT_DISCOVERY_ERROR,
                provider=str(state.get("mode", "search")),
                model="refresh",
                duration_s=0.0,
                success=False,
            )
            with tracer.start_as_current_span(
                "agent.tool_discovery",
                attributes={"agent.discovered": 0, "agent.selected": 0},
            ):
                notes.append("discovered:0")
                notes.append("selected:none")
            return cast(
                AgenticGraphState,
                {
                    "discovered_tools": [],
                    "selected_tools": [],
                    "agent_notes": notes,
                    "errors": errors,
                },
            )

        selected = self._select_tools(capabilities, state)
        with tracer.start_as_current_span(
            "agent.tool_discovery",
            attributes={
                "agent.discovered": len(capabilities),
                "agent.selected": len(selected),
            },
        ):
            notes.append(f"discovered:{len(capabilities)}")
            notes.append(f"selected:{','.join(cap.name for cap in selected) or 'none'}")
        return cast(
            AgenticGraphState,
            {
                "discovered_tools": capabilities,
                "selected_tools": selected,
                "agent_notes": notes,
            },
        )

    async def _retrieval_node(self, state: AgenticGraphState) -> AgenticGraphState:
        collection_value = state.get("collection")
        if state.get("mode") != "search" or not collection_value:
            return cast(AgenticGraphState, {})
        start_time = time.perf_counter()
        query = RetrievalQuery(
            collection=str(collection_value),
            text=str(state.get("query", "")),
            top_k=state.get("retrieval_limit", 5),
            filters=state.get("filters"),
        )
        attributes = {
            "agent.collection": query.collection,
            "agent.top_k": query.top_k,
        }
        try:
            with tracer.start_as_current_span("agent.retrieval", attributes=attributes):
                fetched = await self._retrieval_helper.fetch(query)
                documents = [_serialise_document(doc) for doc in fetched]
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                record_ai_operation(
                    operation_type=_OP_AGENT_RETRIEVAL,
                    provider=query.collection,
                    model=str(state.get("mode", "search")),
                    duration_s=duration_ms / 1000.0,
                )
        except Exception:  # pragma: no cover - defensive guard
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            record_ai_operation(
                operation_type=_OP_AGENT_RETRIEVAL,
                provider=query.collection,
                model=str(state.get("mode", "search")),
                duration_s=duration_ms / 1000.0,
                success=False,
            )
            logger.exception("Retrieval failed for collection %s", query.collection)
            errors = list(state.get("errors", []))
            errors.append(
                _error_entry(
                    "retrieval",
                    AgentErrorCode.RETRIEVAL_ERROR,
                    collection=query.collection,
                )
            )
            return cast(
                AgenticGraphState, {"errors": errors, "retrieved_documents": []}
            )

        notes = list(state.get("agent_notes", []))
        notes.append(f"retrieved:{len(documents)}")
        return cast(
            AgenticGraphState,
            {
                "retrieved_documents": documents,
                "agent_notes": notes,
            },
        )

    async def _tool_execution_node(self, state: AgenticGraphState) -> AgenticGraphState:
        selected = state.get("selected_tools", [])
        if not selected:
            return cast(AgenticGraphState, {"tool_outputs": []})

        semaphore = asyncio.Semaphore(self._max_parallel_tools)
        run_deadline: float | None = None
        if self._run_timeout_seconds is not None:
            run_deadline = float(state.get("start_time", time.perf_counter()))
            run_deadline += self._run_timeout_seconds

        async def invoke_tool(
            capability: ToolCapability,
        ) -> tuple[str, ToolCapability, Any]:
            try:
                async with semaphore:
                    read_timeout_ms: int | None = None
                    if run_deadline is not None:
                        remaining = run_deadline - time.perf_counter()
                        if remaining <= 0:
                            raise ToolExecutionTimeout(
                                (
                                    "Run deadline exceeded before executing tool "
                                    f"'{capability.name}'"
                                ),
                                server_name=capability.server,
                            )
                        read_timeout_ms = max(1, int(remaining * 1000))
                    result = await self._execute_tool(
                        capability,
                        state,
                        read_timeout_ms=read_timeout_ms,
                    )
                return "ok", capability, result
            except ToolExecutionError as exc:
                return "tool", capability, exc
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception(
                    "Unexpected failure executing tool %s on server %s",
                    capability.name,
                    capability.server,
                )
                return "unexpected", capability, exc

        outputs: list[dict[str, Any]] = []
        errors = list(state.get("errors", []))
        with tracer.start_as_current_span(
            "agent.tool_execution",
            attributes={"agent.requested_tools": len(selected)},
        ):
            task_results: list[asyncio.Task[tuple[str, ToolCapability, Any]]] = []
            async with asyncio.TaskGroup() as group:
                for capability in selected:
                    task_results.append(group.create_task(invoke_tool(capability)))

        for task in task_results:
            outcome, capability, payload = task.result()
            if outcome == "ok":
                outputs.append(cast(dict[str, Any], payload))
                continue
            if outcome == "tool":
                error_code = _map_tool_error_code(cast(ToolExecutionError, payload))
                errors.append(
                    _error_entry(
                        "tool_execution",
                        error_code,
                        tool=capability.name,
                        server=capability.server,
                        message=str(payload),
                    )
                )
                continue
            errors.append(
                _error_entry(
                    "tool_execution",
                    AgentErrorCode.TOOL_UNEXPECTED,
                    tool=capability.name,
                    server=capability.server,
                    message=str(payload),
                )
            )

        return cast(
            AgenticGraphState,
            {
                "tool_outputs": outputs,
                "errors": errors,
            },
        )

    async def _synthesis_node(self, state: AgenticGraphState) -> AgenticGraphState:
        documents = state.get("retrieved_documents", [])
        tool_outputs = state.get("tool_outputs", [])
        notes = list(state.get("agent_notes", []))
        notes.append(f"tool_outputs:{len(tool_outputs)}")
        answer_parts: list[str] = []
        with tracer.start_as_current_span(
            "agent.synthesis",
            attributes={
                "agent.tool_outputs": len(tool_outputs),
                "agent.documents": len(documents),
            },
        ):
            if tool_outputs:
                tool_names = ", ".join(output["tool_name"] for output in tool_outputs)
                answer_parts.append(f"Tools executed: {tool_names}.")
            if documents:
                answer_parts.append(f"Retrieved {len(documents)} supporting documents.")
        if not answer_parts:
            answer_parts.append("No tools executed; returning baseline response.")
        answer = " ".join(answer_parts)
        confidence = 0.6 + 0.1 * min(len(tool_outputs), 3)
        confidence = min(confidence, 0.95)
        return {
            "answer": answer,
            "confidence": confidence,
            "agent_notes": notes,
        }

    async def _metrics_node(self, state: AgenticGraphState) -> AgenticGraphState:
        duration_ms = (
            time.perf_counter() - state.get("start_time", time.perf_counter())
        ) * 1000.0
        tool_outputs = state.get("tool_outputs", [])
        errors = state.get("errors", [])
        success = not errors
        metrics = {
            "latency_ms": duration_ms,
            "tool_count": len(tool_outputs),
            "error_count": len(errors),
        }
        with tracer.start_as_current_span(
            "agent.metrics",
            attributes={
                "agent.mode": state.get("mode", "search"),
                "agent.success": success,
                "agent.tool_count": len(tool_outputs),
                "agent.error_count": len(errors),
            },
        ):
            record_ai_operation(
                operation_type=_OP_AGENT_RUN,
                provider=str(state.get("mode", "search")),
                model="run",
                duration_s=duration_ms / 1000.0,
                success=success,
            )
        return {"metrics": metrics, "success": success}

    def _select_tools(
        self, capabilities: Sequence[ToolCapability], state: AgenticGraphState
    ) -> list[ToolCapability]:
        mode = state.get("mode", "search")
        priority_types: tuple[ToolCapabilityType, ...]
        if mode == "analysis":
            priority_types = (
                ToolCapabilityType.ANALYSIS,
                ToolCapabilityType.SYNTHESIS,
                ToolCapabilityType.GENERATION,
            )
        else:
            priority_types = (
                ToolCapabilityType.SEARCH,
                ToolCapabilityType.RETRIEVAL,
                ToolCapabilityType.SYNTHESIS,
            )
        ranked = sorted(
            capabilities,
            key=lambda cap: (
                0 if cap.capability_type in priority_types else 1,
                cap.name,
            ),
        )
        return list(ranked[: self._max_parallel_tools])

    async def _execute_tool(
        self,
        capability: ToolCapability,
        state: AgenticGraphState,
        *,
        read_timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        arguments = self._build_tool_arguments(capability, state)
        result: ToolExecutionResult = await self._tool_service.execute_tool(
            capability.name,
            arguments=arguments,
            server_name=capability.server,
            read_timeout_ms=read_timeout_ms,
        )
        return {
            "tool_name": capability.name,
            "server_name": capability.server,
            "duration_ms": result.duration_ms,
            "structured_content": result.result.structuredContent,
            "content": [item.model_dump() for item in result.result.content],
            "meta": result.result.meta,
        }

    def _build_tool_arguments(
        self, capability: ToolCapability, state: AgenticGraphState
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {"query": state.get("query")}
        documents = state.get("retrieved_documents", [])
        if documents:
            arguments["documents"] = [
                doc.get("metadata") for doc in documents if doc.get("metadata")
            ]
        user_context = state.get("user_context")
        if isinstance(user_context, Mapping):
            arguments["user_context"] = dict(user_context)
        return arguments

    def _to_search_outcome(self, state: AgenticGraphState) -> GraphSearchOutcome:
        raw_documents = state.get("retrieved_documents", [])
        results = SearchRecord.parse_list(raw_documents)
        tools_used = [output["tool_name"] for output in state.get("tool_outputs", [])]
        return GraphSearchOutcome(
            success=state.get("success", True),
            session_id=state.get("session_id", str(uuid4())),
            answer=state.get("answer"),
            confidence=state.get("confidence"),
            results=results,
            tools_used=tools_used,
            reasoning=list(state.get("agent_notes", [])),
            metrics=state.get("metrics", {}),
            errors=list(state.get("errors", [])),
        )

    def _to_analysis_outcome(self, state: AgenticGraphState) -> GraphAnalysisOutcome:
        insights = {
            "tool_outputs": state.get("tool_outputs", []),
            "documents_considered": [
                {
                    "id": doc.get("id"),
                    "score": doc.get("score"),
                    "metadata": doc.get("metadata"),
                }
                for doc in state.get("retrieved_documents", [])
            ],
        }
        recommendations = []
        if state.get("tool_outputs"):
            recommendations.append("Review tool outputs for further action")
        summary = (
            state.get("answer", "No analysis available.") or "No analysis available."
        )
        return GraphAnalysisOutcome(
            success=state.get("success", True),
            analysis_id=state.get("session_id", str(uuid4())),
            summary=summary,
            insights=insights,
            recommendations=recommendations,
            confidence=state.get("confidence"),
            metrics=state.get("metrics", {}),
            errors=list(state.get("errors", [])),
        )


__all__ = [
    "AgentErrorCode",
    "AgenticGraphState",
    "GraphAnalysisOutcome",
    "GraphRunner",
    "GraphSearchOutcome",
]
