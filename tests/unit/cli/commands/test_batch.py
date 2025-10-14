"""Focused tests for the batch CLI command surface."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import click
import pytest
from rich.panel import Panel

from src.cli.commands import batch as batch_module


def _always_true(*_args: Any, **_kwargs: Any) -> bool:
    """Return ``True`` regardless of inputs."""

    return True


def _always_false(*_args: Any, **_kwargs: Any) -> bool:
    """Return ``False`` regardless of inputs."""

    return False


@pytest.fixture
def rich_cli_stub() -> SimpleNamespace:
    """Provide a RichCLI-like object that records console output."""

    class _Console:
        def __init__(self, sink: list[Any]):
            self._sink = sink

        def print(self, value: Any) -> None:
            self._sink.append(value)

    printed: list[Any] = []
    return SimpleNamespace(console=_Console(printed), printed=printed)


@pytest.fixture
def cli_context(rich_cli_stub: SimpleNamespace) -> click.Context:
    """Create a click context that exposes the Rich CLI helper."""

    return click.Context(batch_module.batch, obj={"rich_cli": rich_cli_stub})


def test_complete_collection_name_returns_filtered_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completion helper should expose matching collections from the manager."""

    class _VectorDBStub:
        async def list_collections(self) -> list[str]:
            return ["alpha", "beta", "docs"]

        async def cleanup(self) -> None:  # pragma: no cover - nothing to clean
            return None

    monkeypatch.setattr(
        batch_module,
        "_init_vector_manager",
        _VectorDBStub,
    )
    ctx = click.Context(batch_module.batch, obj={"config": object()})
    param = cast(click.Parameter, None)

    completions = batch_module.complete_collection_name(ctx, param, "a")

    assert [item.value for item in completions] == ["alpha"]


def test_complete_collection_name_handles_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completion helper must return an empty list when the lookup fails."""

    class _VectorDBStub:
        async def list_collections(self) -> list[str]:
            raise RuntimeError("boom")

        async def cleanup(self) -> None:  # pragma: no cover - nothing to clean
            return None

    monkeypatch.setattr(batch_module, "_init_vector_manager", lambda: _VectorDBStub())
    ctx = click.Context(batch_module.batch, obj={"config": object()})
    param = cast(click.Parameter, None)

    completions = batch_module.complete_collection_name(ctx, param, "a")

    assert completions == []


def test_operation_queue_execute_runs_all_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`OperationQueue.execute` should invoke queued operations and report success."""

    queue = batch_module.OperationQueue()
    executed: list[str] = []
    queue.add(
        batch_module.BatchOperation("first", "", lambda: executed.append("first"))
    )
    queue.add(
        batch_module.BatchOperation("second", "", lambda: executed.append("second"))
    )

    assert queue.execute(confirm=False) is True
    assert executed == ["first", "second"]


def test_operation_queue_execute_stops_on_error() -> None:
    """A failing operation should short-circuit execution and return ``False``."""

    queue = batch_module.OperationQueue()

    def _fail() -> None:
        raise ValueError("unexpected failure")

    queue.add(batch_module.BatchOperation("fail", "", _fail))

    assert queue.execute(confirm=False) is False


def test_operation_queue_clear_removes_operations() -> None:
    """`clear` should empty the operation list."""

    queue = batch_module.OperationQueue()
    queue.add(batch_module.BatchOperation("noop", "", lambda: None))

    queue.clear()

    assert not queue.operations


def test_show_indexing_preview_emits_panel(rich_cli_stub: SimpleNamespace) -> None:
    """The dry-run preview should render a Rich panel to the console."""

    batch_module._show_indexing_preview(["a", "b", "c"], "collection", 2, rich_cli_stub)

    assert any(isinstance(item, Panel) for item in rich_cli_stub.printed)


def test_index_documents_dry_run_invokes_preview(
    monkeypatch: pytest.MonkeyPatch,
    cli_context: click.Context,
    rich_cli_stub: SimpleNamespace,
) -> None:
    """Dry-run invocations must call the preview helper instead of performing work."""

    captured: dict[str, Any] = {}

    def _capture(
        documents: list[str],
        collection_name: str,
        batch_size: int,
        rich_cli: SimpleNamespace,
    ) -> None:
        captured.update(
            {
                "documents": documents,
                "collection": collection_name,
                "batch_size": batch_size,
                "rich_cli": rich_cli,
            }
        )

    monkeypatch.setattr(batch_module, "_show_indexing_preview", _capture)

    index_callback = batch_module.index_documents.callback
    assert index_callback is not None

    with cli_context:
        index_callback(
            "target",
            ("doc1", "doc2"),
            batch_size=5,
            _parallel=1,
            dry_run=True,
        )

    assert captured == {
        "documents": ["doc1", "doc2"],
        "collection": "target",
        "batch_size": 5,
        "rich_cli": rich_cli_stub,
    }


def test_create_collections_aborts_without_confirmation(
    monkeypatch: pytest.MonkeyPatch, cli_context: click.Context
) -> None:
    """If the operator declines, no queue should be instantiated."""

    monkeypatch.setattr(batch_module, "Confirm", SimpleNamespace(ask=_always_false))

    def _queue_factory() -> None:
        raise AssertionError("Queue should not be constructed when confirmation fails")

    monkeypatch.setattr(batch_module, "OperationQueue", _queue_factory)

    create_callback = batch_module.create_collections.callback
    assert create_callback is not None

    with cli_context:
        create_callback(
            ("alpha",),
            dimension=256,
            distance="cosine",
            _force=False,
        )


def test_create_collections_enqueues_operations(
    monkeypatch: pytest.MonkeyPatch, cli_context: click.Context
) -> None:
    """Confirmed requests should enqueue batch operations and execute them."""

    monkeypatch.setattr(batch_module, "Confirm", SimpleNamespace(ask=_always_true))
    db_manager = SimpleNamespace(calls=[])

    class _VectorDBStub:
        async def create_collection(self, name: str, dimension: int) -> bool:
            db_manager.calls.append((name, dimension))
            return True

        async def cleanup(self) -> None:
            return None

    monkeypatch.setattr(batch_module, "_init_vector_manager", lambda: _VectorDBStub())

    queue_instances: list[_QueueStub] = []

    class _QueueStub:
        def __init__(self) -> None:
            self.operations: list[batch_module.BatchOperation] = []
            self.confirm_flag: bool | None = None
            queue_instances.append(self)

        def add(self, operation: batch_module.BatchOperation) -> None:
            self.operations.append(operation)

        def execute(self, confirm: bool = True) -> bool:
            self.confirm_flag = confirm
            for operation in self.operations:
                assert callable(operation.function)
                operation_fn = cast(Callable[[], Any], operation.function)
                operation_fn()
            return True

    monkeypatch.setattr(batch_module, "OperationQueue", _QueueStub)

    def _run(coro: Any) -> Any:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(batch_module.asyncio, "run", _run)

    create_callback = batch_module.create_collections.callback
    assert create_callback is not None

    with cli_context:
        create_callback(
            ("alpha", "beta"),
            dimension=128,
            distance="cosine",
            _force=False,
        )

    queue = queue_instances[0]
    assert [operation.name for operation in queue.operations] == [
        "Create alpha",
        "Create beta",
    ]
    assert queue.confirm_flag is False
    assert db_manager.calls == [("alpha", 128), ("beta", 128)]


def test_delete_collections_aborts_without_double_confirmation(
    monkeypatch: pytest.MonkeyPatch, cli_context: click.Context
) -> None:
    """Declining the destructive prompts should avoid queue creation."""

    responses = iter([False])

    def _ask(_prompt: str) -> bool:
        return next(responses)

    monkeypatch.setattr(batch_module, "Confirm", SimpleNamespace(ask=_ask))

    def _queue_factory() -> None:
        raise AssertionError("Queue should not be constructed when confirmation fails")

    monkeypatch.setattr(batch_module, "OperationQueue", _queue_factory)

    delete_callback = batch_module.delete_collections.callback
    assert delete_callback is not None

    with cli_context:
        delete_callback(
            ("alpha", "beta"),
            yes=False,
        )


def test_delete_collections_enqueues_deletions(
    monkeypatch: pytest.MonkeyPatch, cli_context: click.Context
) -> None:
    """The delete command should enqueue operations when confirmation is bypassed."""

    db_manager = SimpleNamespace(deleted=[])

    class _VectorDBStub:
        async def delete_collection(self, name: str) -> bool:
            db_manager.deleted.append(name)
            return True

        async def cleanup(self) -> None:
            return None

    monkeypatch.setattr(batch_module, "_init_vector_manager", _VectorDBStub())

    queue_instances: list[_QueueStub] = []

    class _QueueStub:
        def __init__(self) -> None:
            self.operations: list[batch_module.BatchOperation] = []
            queue_instances.append(self)

        def add(self, operation: batch_module.BatchOperation) -> None:
            self.operations.append(operation)

        def execute(self, confirm: bool = True) -> bool:
            for operation in self.operations:
                assert callable(operation.function)
                operation_fn = cast(Callable[[], Any], operation.function)
                operation_fn()
            return True

    monkeypatch.setattr(batch_module, "OperationQueue", _QueueStub)
    monkeypatch.setattr(batch_module, "Confirm", SimpleNamespace(ask=_always_true))

    def _run(coro: Any) -> Any:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(batch_module.asyncio, "run", _run)

    delete_callback = batch_module.delete_collections.callback
    assert delete_callback is not None

    with cli_context:
        delete_callback(
            ("alpha", "beta"),
            yes=True,
        )

    queue = queue_instances[0]
    assert [operation.name for operation in queue.operations] == [
        "Delete alpha",
        "Delete beta",
    ]
    assert db_manager.deleted == ["alpha", "beta"]
