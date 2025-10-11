"""Unit tests for helper utilities in the LangGraph runner."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.services.agents.langgraph_runner import (
    AgentErrorCode,
    _error_entry,
    _map_tool_error_code,
    _serialise_document,
)
from src.services.agents.tool_execution_service import (
    ToolExecutionError,
    ToolExecutionFailure,
    ToolExecutionInvalidArgument,
    ToolExecutionTimeout,
)
from tests.unit.conftest import require_optional_dependency  # type: ignore[import]


require_optional_dependency("langgraph")
require_optional_dependency("langchain_core")


@dataclass
class DummyDocument:
    """Minimal document stub mimicking search record behaviour."""

    id: str
    score: float
    metadata: dict[str, str]

    def dict(self) -> dict[str, object]:  # noqa: D401 - compatibility shim
        return {"metadata": self.metadata}


def test_serialise_document_uses_metadata_dict() -> None:
    doc = DummyDocument(id="doc-1", score=0.75, metadata={"source": "tests"})

    payload = _serialise_document(doc)

    assert payload == {"id": "doc-1", "score": 0.75, "metadata": {"source": "tests"}}


@pytest.mark.parametrize(
    "exception,expected",
    [
        (ToolExecutionTimeout("timeout"), AgentErrorCode.TOOL_TIMEOUT),
        (ToolExecutionInvalidArgument("bad"), AgentErrorCode.TOOL_INVALID_ARGUMENT),
        (ToolExecutionFailure("fail"), AgentErrorCode.TOOL_FAILURE),
        (ToolExecutionError("boom"), AgentErrorCode.TOOL_UNEXPECTED),
    ],
)
def test_map_tool_error_code(
    exception: ToolExecutionError, expected: AgentErrorCode
) -> None:
    assert _map_tool_error_code(exception) is expected


def test_error_entry_includes_extra_metadata() -> None:
    entry = _error_entry("retrieval", AgentErrorCode.RETRIEVAL_ERROR, detail="timeout")

    assert entry == {
        "source": "retrieval",
        "code": "RETRIEVAL_ERROR",
        "detail": "timeout",
    }
