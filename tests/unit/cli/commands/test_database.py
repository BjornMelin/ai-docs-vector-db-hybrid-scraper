"""Unit tests for the database CLI command group."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import Any

import click
import pytest
from click.testing import CliRunner, Result
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from src.cli.commands import database as database_module
from src.manage_vector_db import CollectionCreationError, CollectionDeletionError


class VectorManagerStub:
    """In-memory stand-in for ``VectorDBManager`` used in CLI tests."""

    def __init__(
        self,
        *,
        collections: dict[str, dict[str, Any]] | None = None,
        fail: Iterable[str] | None = None,
        create_success: bool = True,
        delete_success: bool = True,
    ) -> None:
        """Initialize the stub with configurable behaviors."""
        self._collections = collections or {}
        self._fail = set(fail or ())
        self._outcomes = {"create": create_success, "delete": delete_success}
        self.counters = {"list": 0, "cleanup": 0}
        self.records: dict[str, list[Any]] = {
            "info": [],
            "create": [],
            "delete": [],
        }

    async def list_collections(self) -> list[str]:
        """Return configured collection identifiers or raise when instructed."""
        self.counters["list"] += 1
        if "list" in self._fail:
            msg = "listing collections failed"
            raise ValueError(msg)
        return list(self._collections)

    async def get_collection_info(self, name: str) -> SimpleNamespace | None:
        """Return stored collection metadata or propagate configured failures."""
        self.records["info"].append(name)
        if "info" in self._fail:
            msg = "fetching collection info failed"
            raise ValueError(msg)
        payload = self._collections.get(name)
        return None if payload is None else SimpleNamespace(**payload)

    async def create_collection(
        self, collection_name: str, *, vector_size: int
    ) -> bool:
        """Record create attempts and control the return value."""
        self.records["create"].append((collection_name, vector_size))
        if "create_raise" in self._fail:
            msg = "creation raised"
            raise ValueError(msg)
        if "create_false" in self._fail:
            return False
        return bool(self._outcomes["create"])

    async def delete_collection(self, collection_name: str) -> bool:
        """Record delete attempts and control the return value."""
        self.records["delete"].append(collection_name)
        if "delete_raise" in self._fail:
            msg = "deletion raised"
            raise ValueError(msg)
        if "delete_false" in self._fail:
            return False
        return bool(self._outcomes["delete"])

    async def cleanup(self) -> None:
        """Track cleanup usage to ensure resources are reclaimed."""
        self.counters["cleanup"] += 1


@pytest.fixture
def rich_cli_stub() -> SimpleNamespace:
    """Provide a minimal Rich CLI stub that captures printed messages."""

    class _Console:
        """Lightweight console stub that records printed values."""

        def __init__(self, sink: list[Any]):
            self._sink = sink

        def print(self, value: Any) -> None:
            """Capture a Rich console print call."""
            self._sink.append(value)

    class _RichCLI(SimpleNamespace):
        """Rich CLI replacement capturing printed messages and errors."""

        printed: list[Any]
        errors: list[tuple[str, str | None]]
        console: _Console

        def __init__(self) -> None:
            printed: list[Any] = []
            errors: list[tuple[str, str | None]] = []
            console = _Console(printed)
            super().__init__(console=console, printed=printed, errors=errors)

        def show_error(self, message: str, details: str | None = None) -> None:
            """Record structured error output for later inspection."""
            self.errors.append((message, details))

    return _RichCLI()


@pytest.fixture
def config_stub() -> SimpleNamespace:
    """Provide the configuration object required by the CLI context."""
    return SimpleNamespace(qdrant=SimpleNamespace(host="localhost", port=6333))


@pytest.fixture
def cli_obj(
    rich_cli_stub: SimpleNamespace, config_stub: SimpleNamespace
) -> dict[str, Any]:
    """Build the Click context object consumed by database commands."""
    return {"rich_cli": rich_cli_stub, "config": config_stub}


@pytest.fixture
def collection_parameter() -> click.Parameter:
    """Provide a concrete Click parameter instance for completion tests."""
    return click.Option(["--collection-name"])


def _patch_manager(
    monkeypatch: pytest.MonkeyPatch, factory: Callable[[], VectorManagerStub]
) -> None:
    """Patch the database module to use the supplied fake manager factory."""

    def _factory() -> VectorManagerStub:
        return factory()

    monkeypatch.setattr(database_module, "VectorDBManager", _factory)


def _invoke(
    cli_runner: CliRunner,
    args: list[str],
    *,
    obj: dict[str, Any],
) -> Result:
    """Invoke the database CLI with the provided arguments."""
    return cli_runner.invoke(database_module.database, args, obj=obj)


def test_complete_collection_name_filters_matches(
    monkeypatch: pytest.MonkeyPatch,
    collection_parameter: click.Parameter,
) -> None:
    """Completion should return matching collection identifiers."""
    stub = VectorManagerStub(
        collections={
            "alpha": {"name": "alpha", "vector_count": 0, "vector_size": 1},
            "beta": {"name": "beta", "vector_count": 0, "vector_size": 1},
        }
    )
    _patch_manager(monkeypatch, lambda: stub)
    ctx = click.Context(database_module.database, obj={"config": object()})
    items = database_module.complete_collection_name(ctx, collection_parameter, "a")

    assert [item.value for item in items] == ["alpha"]
    assert stub.counters["cleanup"] == 1


def test_complete_collection_name_returns_empty_without_config(
    collection_parameter: click.Parameter,
) -> None:
    """Completion helper should short-circuit when no config is present."""
    ctx = click.Context(database_module.database, obj={})
    items = database_module.complete_collection_name(ctx, collection_parameter, "test")

    assert items == []


def test_complete_collection_name_handles_manager_failure(
    monkeypatch: pytest.MonkeyPatch,
    collection_parameter: click.Parameter,
) -> None:
    """Errors during completion should yield an empty suggestion list."""
    stub = VectorManagerStub(fail={"list"})
    _patch_manager(monkeypatch, lambda: stub)
    ctx = click.Context(database_module.database, obj={"config": object()})
    items = database_module.complete_collection_name(ctx, collection_parameter, "a")

    assert items == []
    assert stub.counters["list"] == 1
    assert stub.counters["cleanup"] == 0  # Failure occurs before cleanup


def test_list_collections_renders_table(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    rich_cli_stub: SimpleNamespace,
    cli_obj: dict[str, Any],
) -> None:
    """The list command should display a table of collections."""
    stub = VectorManagerStub(
        collections={
            "alpha": {"name": "alpha", "vector_count": 10, "vector_size": 1536},
        }
    )
    _patch_manager(monkeypatch, lambda: stub)

    result = _invoke(cli_runner, ["list"], obj=cli_obj)

    assert result.exit_code == 0
    table = next(item for item in rich_cli_stub.printed if isinstance(item, Table))
    assert table.row_count == 1
    assert stub.counters["cleanup"] == 1


def test_list_collections_json_output(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    rich_cli_stub: SimpleNamespace,
    cli_obj: dict[str, Any],
) -> None:
    """JSON output should render inside a Rich panel containing syntax highlighting."""
    stub = VectorManagerStub(
        collections={
            "docs": {"name": "docs", "vector_count": 5, "vector_size": 512},
        }
    )
    _patch_manager(monkeypatch, lambda: stub)

    result = _invoke(cli_runner, ["list", "--format", "json"], obj=cli_obj)

    assert result.exit_code == 0
    panel = next(item for item in rich_cli_stub.printed if isinstance(item, Panel))
    assert isinstance(panel.renderable, Syntax)
    assert '"docs"' in panel.renderable.code


def test_list_collections_failure_reports_error(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    rich_cli_stub: SimpleNamespace,
    cli_obj: dict[str, Any],
) -> None:
    """Manager failures should emit an error and abort the command."""
    stub = VectorManagerStub(fail={"list"})
    _patch_manager(monkeypatch, lambda: stub)

    result = _invoke(cli_runner, ["list"], obj=cli_obj)

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert rich_cli_stub.errors == [
        ("Failed to list collections", "listing collections failed")
    ]


def test_display_collections_table_handles_empty_dataset(
    rich_cli_stub: SimpleNamespace,
) -> None:
    """Empty collection lists should yield a helpful notice."""
    database_module._display_collections_table([], rich_cli_stub)

    assert rich_cli_stub.printed == ["[yellow]No collections found.[/yellow]"]


def test_display_collections_table_prints_rows(rich_cli_stub: SimpleNamespace) -> None:
    """Non-empty collection lists should render a Rich table."""
    data = [
        {
            "name": "alpha",
            "vectors_count": 42,
            "config": {"params": {"vectors": {"size": 1536}}},
            "status": "green",
            "created_at": "yesterday",
        }
    ]

    database_module._display_collections_table(data, rich_cli_stub)

    table = rich_cli_stub.printed[0]
    assert isinstance(table, Table)
    assert table.row_count == 1


def test_display_collections_json_renders_panel(rich_cli_stub: SimpleNamespace) -> None:
    """JSON renderer should produce a panel with syntax highlighting."""
    database_module._display_collections_json([], rich_cli_stub)

    panel = rich_cli_stub.printed[0]
    assert isinstance(panel, Panel)
    assert isinstance(panel.renderable, Syntax)


@pytest.mark.parametrize(
    "helper, args, expected_exception",
    [
        (
            database_module._abort_collection_creation_failed,
            (),
            CollectionCreationError,
        ),
        (
            database_module._abort_collection_deletion_failed,
            (),
            CollectionDeletionError,
        ),
    ],
)
def test_abort_helpers_raise_expected_errors(
    helper: Callable[..., None],
    args: tuple[Any, ...],
    expected_exception: type[Exception],
) -> None:
    """Abort helpers must raise precise exception types."""
    with pytest.raises(expected_exception):
        helper(*args)


def test_abort_collection_exists_reports_error(rich_cli_stub: SimpleNamespace) -> None:
    """Existing collections should be reported with actionable guidance."""
    with pytest.raises(click.exceptions.Abort):
        database_module._abort_collection_exists("alpha", rich_cli_stub)

    assert rich_cli_stub.errors == [
        ("Collection 'alpha' already exists", "Use --force to overwrite")
    ]


def test_abort_collection_not_found_reports_error(
    rich_cli_stub: SimpleNamespace,
) -> None:
    """Missing collections should trigger a descriptive abort."""
    with pytest.raises(click.exceptions.Abort):
        database_module._abort_collection_not_found("alpha", rich_cli_stub)

    assert rich_cli_stub.errors == [("Collection 'alpha' not found", None)]


def test_create_collection_succeeds(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    rich_cli_stub: SimpleNamespace,
    cli_obj: dict[str, Any],
) -> None:
    """Creating a new collection should delegate to the vector manager."""
    stub = VectorManagerStub(collections={})
    _patch_manager(monkeypatch, lambda: stub)

    result = _invoke(
        cli_runner,
        ["create", "alpha", "--dimension", "1024"],
        obj=cli_obj,
    )

    assert result.exit_code == 0
    assert stub.records["create"] == [("alpha", 1024)]
    panel = rich_cli_stub.printed[-1]
    assert isinstance(panel, Panel)
    assert isinstance(panel.renderable, Text)
    assert "Dimension: 1024" in panel.renderable.plain


def test_create_collection_invalid_dimension(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    rich_cli_stub: SimpleNamespace,
    cli_obj: dict[str, Any],
) -> None:
    """Invalid dimension values should raise errors and surface diagnostics."""
    stub = VectorManagerStub(collections={})
    _patch_manager(monkeypatch, lambda: stub)

    invalid_dimensions = ["-1", "0", "notanint"]
    for dim in invalid_dimensions:
        result = _invoke(
            cli_runner,
            ["create", "alpha", "--dimension", dim],
            obj=cli_obj,
        )
        assert result.exit_code != 0
        error_found = any(
            "dimension" in str(err).lower() or "invalid" in str(err).lower()
            for err, _ in getattr(rich_cli_stub, "errors", [])
        )
        assert error_found or "invalid" in result.output.lower()


def test_create_collection_aborts_when_exists(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    rich_cli_stub: SimpleNamespace,
    cli_obj: dict[str, Any],
) -> None:
    """Attempting to create an existing collection should abort without side effects."""
    stub = VectorManagerStub(
        collections={"alpha": {"name": "alpha", "vector_count": 1, "vector_size": 1}}
    )
    _patch_manager(monkeypatch, lambda: stub)

    result = _invoke(cli_runner, ["create", "alpha"], obj=cli_obj)

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert not stub.records["create"]


def test_create_collection_force_deletes_existing(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    rich_cli_stub: SimpleNamespace,
    cli_obj: dict[str, Any],
) -> None:
    """The --force flag should delete existing collections before recreation."""
    stub = VectorManagerStub(
        collections={"alpha": {"name": "alpha", "vector_count": 1, "vector_size": 1}}
    )
    _patch_manager(monkeypatch, lambda: stub)
    monkeypatch.setattr(database_module.Confirm, "ask", lambda *_: True)

    result = _invoke(cli_runner, ["create", "alpha", "--force"], obj=cli_obj)

    assert result.exit_code == 0
    assert stub.records["delete"] == ["alpha"]
    assert stub.records["create"] == [("alpha", 1536)]


def test_create_collection_force_cancelled_by_user(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    rich_cli_stub: SimpleNamespace,
    cli_obj: dict[str, Any],
) -> None:
    """Declining the force confirmation should cancel creation gracefully."""
    stub = VectorManagerStub(
        collections={"alpha": {"name": "alpha", "vector_count": 1, "vector_size": 1}}
    )
    _patch_manager(monkeypatch, lambda: stub)
    monkeypatch.setattr(database_module.Confirm, "ask", lambda *_: False)

    result = _invoke(cli_runner, ["create", "alpha", "--force"], obj=cli_obj)

    assert result.exit_code == 0
    assert not stub.records["delete"]
    assert rich_cli_stub.printed[-1] == "[yellow]Creation cancelled.[/yellow]"


def test_create_collection_failure_raises_custom_error(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    cli_obj: dict[str, Any],
) -> None:
    """Returning ``False`` from the manager should raise ``CollectionCreationError``."""
    stub = VectorManagerStub(collections={}, fail={"create_false"})
    _patch_manager(monkeypatch, lambda: stub)

    result = _invoke(cli_runner, ["create", "alpha"], obj=cli_obj)

    assert result.exit_code == 1
    assert isinstance(result.exception, CollectionCreationError)


def test_create_collection_runtime_error_aborts(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    rich_cli_stub: SimpleNamespace,
    cli_obj: dict[str, Any],
) -> None:
    """Manager exceptions should surface as Click aborts with diagnostics."""
    stub = VectorManagerStub(collections={}, fail={"create_raise"})
    _patch_manager(monkeypatch, lambda: stub)

    result = _invoke(cli_runner, ["create", "alpha"], obj=cli_obj)

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert rich_cli_stub.errors == [("Failed to create collection", "creation raised")]


def test_delete_collection_succeeds(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    rich_cli_stub: SimpleNamespace,
    cli_obj: dict[str, Any],
) -> None:
    """Deleting a collection should report success when the manager confirms."""
    stub = VectorManagerStub(
        collections={"alpha": {"name": "alpha", "vector_count": 0, "vector_size": 1}}
    )
    _patch_manager(monkeypatch, lambda: stub)

    result = _invoke(cli_runner, ["delete", "alpha", "--yes"], obj=cli_obj)

    assert result.exit_code == 0
    assert stub.records["delete"] == ["alpha"]
    assert rich_cli_stub.printed[-1] == "Collection 'alpha' deleted successfully."


def test_delete_collection_prompt_decline(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    rich_cli_stub: SimpleNamespace,
    cli_obj: dict[str, Any],
) -> None:
    """Declining the deletion prompt should cancel the operation."""
    stub = VectorManagerStub(collections={})
    _patch_manager(monkeypatch, lambda: stub)
    monkeypatch.setattr(database_module.Confirm, "ask", lambda *_: False)

    result = _invoke(cli_runner, ["delete", "alpha"], obj=cli_obj)

    assert result.exit_code == 0
    assert not stub.records["delete"]
    assert rich_cli_stub.printed[-1] == "[yellow]Deletion cancelled.[/yellow]"


def test_delete_collection_failure_raises_custom_error(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    cli_obj: dict[str, Any],
) -> None:
    """Returning ``False`` from the manager should raise ``CollectionDeletionError``."""
    stub = VectorManagerStub(collections={}, fail={"delete_false"})
    _patch_manager(monkeypatch, lambda: stub)

    result = _invoke(cli_runner, ["delete", "alpha", "--yes"], obj=cli_obj)

    assert result.exit_code == 1
    assert isinstance(result.exception, CollectionDeletionError)


def test_delete_collection_runtime_error_aborts(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    rich_cli_stub: SimpleNamespace,
    cli_obj: dict[str, Any],
) -> None:
    """Manager exceptions during deletion should trigger Click aborts."""
    stub = VectorManagerStub(collections={}, fail={"delete_raise"})
    _patch_manager(monkeypatch, lambda: stub)

    result = _invoke(cli_runner, ["delete", "alpha", "--yes"], obj=cli_obj)

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert rich_cli_stub.errors == [("Failed to delete collection", "deletion raised")]


def test_collection_info_succeeds(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    rich_cli_stub: SimpleNamespace,
    cli_obj: dict[str, Any],
) -> None:
    """The info command should display details for existing collections."""
    stub = VectorManagerStub(
        collections={"alpha": {"name": "alpha", "vector_count": 10, "vector_size": 256}}
    )
    _patch_manager(monkeypatch, lambda: stub)

    result = _invoke(cli_runner, ["info", "alpha"], obj=cli_obj)

    assert result.exit_code == 0
    table = next(item for item in rich_cli_stub.printed if isinstance(item, Table))
    assert table.row_count == 5


def test_collection_info_not_found_aborts(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    rich_cli_stub: SimpleNamespace,
    cli_obj: dict[str, Any],
) -> None:
    """Missing collections should trigger a Click abort with diagnostics."""
    stub = VectorManagerStub(collections={})
    _patch_manager(monkeypatch, lambda: stub)

    result = _invoke(cli_runner, ["info", "ghost"], obj=cli_obj)

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert ("Collection 'ghost' not found", None) in rich_cli_stub.errors


def test_collection_info_runtime_error_aborts(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    rich_cli_stub: SimpleNamespace,
    cli_obj: dict[str, Any],
) -> None:
    """Errors when fetching info should surface through the Rich CLI helper."""
    stub = VectorManagerStub(collections={"alpha": {}}, fail={"info"})
    _patch_manager(monkeypatch, lambda: stub)

    result = _invoke(cli_runner, ["info", "alpha"], obj=cli_obj)

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert rich_cli_stub.errors == [
        ("Failed to get collection info", "fetching collection info failed")
    ]


def test_search_collection_warns_about_unimplemented_path(
    cli_runner: CliRunner,
    rich_cli_stub: SimpleNamespace,
    cli_obj: dict[str, Any],
) -> None:
    """The search command should communicate its unimplemented state."""
    result = _invoke(cli_runner, ["search", "alpha", "query"], obj=cli_obj)

    assert result.exit_code == 0
    assert "Vector search via CLI is not yet implemented" in rich_cli_stub.printed[0]


def test_database_stats_reports_totals(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    rich_cli_stub: SimpleNamespace,
    cli_obj: dict[str, Any],
) -> None:
    """Database statistics should aggregate counts and display per-collection data."""
    stub = VectorManagerStub(
        collections={
            "alpha": {"name": "alpha", "vector_count": 3, "vector_size": 100},
            "beta": {"name": "beta", "vector_count": 7, "vector_size": 200},
        }
    )
    _patch_manager(monkeypatch, lambda: stub)

    result = _invoke(cli_runner, ["stats"], obj=cli_obj)

    assert result.exit_code == 0
    tables = [item for item in rich_cli_stub.printed if isinstance(item, Table)]
    assert len(tables) == 2  # Summary table and breakdown table
    summary = tables[0]
    assert summary.row_count == 3
    assert stub.counters["cleanup"] == 1


def test_database_stats_without_collections(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    rich_cli_stub: SimpleNamespace,
    cli_obj: dict[str, Any],
) -> None:
    """When no collections exist only the summary table should be displayed."""
    stub = VectorManagerStub(collections={})
    _patch_manager(monkeypatch, lambda: stub)

    result = _invoke(cli_runner, ["stats"], obj=cli_obj)

    assert result.exit_code == 0
    tables = [item for item in rich_cli_stub.printed if isinstance(item, Table)]
    assert len(tables) == 1


def test_database_stats_runtime_error_aborts(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    rich_cli_stub: SimpleNamespace,
    cli_obj: dict[str, Any],
) -> None:
    """Failures during stats collection should abort with an error message."""
    stub = VectorManagerStub(fail={"list"})
    _patch_manager(monkeypatch, lambda: stub)

    result = _invoke(cli_runner, ["stats"], obj=cli_obj)

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert rich_cli_stub.errors == [
        ("Failed to get database stats", "listing collections failed")
    ]
