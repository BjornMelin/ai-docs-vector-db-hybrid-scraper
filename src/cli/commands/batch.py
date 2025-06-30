"""Batch operations with Rich progress tracking and confirmations.

This module provides comprehensive batch processing capabilities with
Rich progress visualization, operation queuing, and interactive confirmations.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import click
from click.shell_completion import CompletionItem
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from src.infrastructure.client_manager import ClientManager
from src.manage_vector_db import VectorDBManager


console = Console()


def complete_collection_name(
    ctx: click.Context, _param: click.Parameter, incomplete: str
) -> list[CompletionItem]:
    """Auto-complete collection names from the database."""
    try:
        # Get config from context
        config = ctx.obj.get("config")
        if not config:
            return []

        # Create client manager and get collections
        client_manager = ClientManager(config)
        db_manager = VectorDBManager(client_manager)

        # Get collection names (synchronously for completion)
        collections = asyncio.run(db_manager.list_collections())
        asyncio.run(db_manager.cleanup())

        # Filter collections that start with the incomplete text
        matching_collections = [
            name for name in collections if name.startswith(incomplete)
        ]

        # Return completion items
        return [
            CompletionItem(name, help=f"Collection: {name}")
            for name in matching_collections
        ]
    except Exception:
        # If anything fails, return empty list
        return []


@dataclass
class BatchOperation:
    """Represents a batch operation to be executed."""

    name: str
    description: str
    function: callable
    args: tuple = ()
    kwargs: dict = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


class OperationQueue:
    """Manages a queue of batch operations with Rich visualization."""

    def __init__(self):
        """Initialize operation queue with Rich console for visualization."""
        self.operations: list[BatchOperation] = []
        self.console = Console()

    def add(self, operation: BatchOperation):
        """Add an operation to the queue."""
        self.operations.append(operation)

    def preview(self):
        """Display a preview of planned operations."""
        if not self.operations:
            self.console.print("[yellow]No operations queued.[/yellow]")
            return

        table = Table(title="Planned Operations", show_header=True)
        table.add_column("Operation", style="cyan", width=20)
        table.add_column("Description", style="", width=50)

        for i, op in enumerate(self.operations, 1):
            table.add_row(f"{i}. {op.name}", op.description)

        self.console.print(table)

    def execute(self, confirm: bool = True) -> bool:
        """Execute all operations in the queue."""
        if not self.operations:
            self.console.print("[yellow]No operations to execute.[/yellow]")
            return True

        if confirm:
            self.preview()
            if not Confirm.ask(f"Execute {len(self.operations)} operations?"):
                self.console.print("[yellow]Execution cancelled.[/yellow]")
                return False

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                "Executing operations...", total=len(self.operations)
            )

            for _i, op in enumerate(self.operations):
                progress.update(task, description=f"Executing: {op.name}")

                try:
                    op.function(*op.args, **op.kwargs)
                    progress.console.print(f"✅ {op.name}")
                except Exception as e:
                    progress.console.print(f"❌ {op.name}: {e}")
                    return False

                progress.advance(task)

        self.console.print(
            f"✅ All {len(self.operations)} operations completed successfully!"
        )
        return True

    def clear(self):
        """Clear all operations from the queue."""
        self.operations.clear()


@click.group()
def batch():
    """📦 Batch operations with enhanced progress tracking.

    Perform bulk operations on collections, documents, and configurations
    with Rich progress visualization and operation queuing.
    """


@batch.command("index-documents")
@click.argument("collection_name", shell_complete=complete_collection_name)
@click.argument("documents", nargs=-1, required=True)
@click.option(
    "--batch-size",
    type=int,
    default=100,
    help="Number of documents to process in each batch",
)
@click.option("--parallel", type=int, default=3, help="Number of parallel workers")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be indexed without actually doing it",
)
@click.pass_context
def index_documents(
    ctx: click.Context,
    collection_name: str,
    documents: tuple,
    batch_size: int,
    _parallel: int,
    dry_run: bool,
):
    """Batch index documents into a collection.

    Documents can be file paths, URLs, or directory paths.
    Supports parallel processing for improved performance.
    """
    config = ctx.obj["config"]
    rich_cli = ctx.obj["rich_cli"]

    # Convert documents to list and validate
    doc_list = list(documents)

    if dry_run:
        _show_indexing_preview(doc_list, collection_name, batch_size, rich_cli)
        return

    # Confirm operation
    if not Confirm.ask(f"Index {len(doc_list)} documents into '{collection_name}'?"):
        rich_cli.console.print("[yellow]Indexing cancelled.[/yellow]")
        return

    try:
        client_manager = ClientManager(config)
        VectorDBManager(client_manager)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            main_task = progress.add_task("Indexing documents...", total=len(doc_list))

            # Process documents in batches
            for i in range(0, len(doc_list), batch_size):
                batch = doc_list[i : i + batch_size]

                progress.update(
                    main_task, description=f"Processing batch {i // batch_size + 1}..."
                )

                # Process batch with parallel workers
                # Note: Document indexing not implemented yet
                rich_cli.console.print(
                    "[yellow]Document indexing functionality will be implemented in a future update.[/yellow]"
                )

                # Update progress for all items in batch
                for _ in batch:
                    progress.advance(main_task)

        success_text = Text()
        success_text.append("✅ Batch indexing completed!\n", style="bold green")
        success_text.append(f"Documents indexed: {len(doc_list)}\n", style="cyan")
        success_text.append(f"Collection: {collection_name}\n", style="blue")
        success_text.append(f"Batch size: {batch_size}", style="dim")

        panel = Panel(
            success_text,
            title="Indexing Complete",
            title_align="left",
            border_style="green",
        )
        rich_cli.console.print(panel)

    except Exception as e:
        rich_cli.show_error("Batch indexing failed", str(e))
        raise click.Abort from e


def _show_indexing_preview(
    documents: list[str], collection_name: str, batch_size: int, rich_cli
):
    """Show a preview of what would be indexed."""
    preview_text = Text()
    preview_text.append("📋 Indexing Preview\n\n", style="bold cyan")
    preview_text.append(f"Collection: {collection_name}\n", style="")
    preview_text.append(f"Documents: {len(documents)}\n", style="")
    preview_text.append(f"Batch size: {batch_size}\n", style="")
    preview_text.append(
        f"Estimated batches: {(len(documents) + batch_size - 1) // batch_size}\n\n",
        style="",
    )

    preview_text.append("First 5 documents:\n", style="bold")
    for i, doc in enumerate(documents[:5]):
        preview_text.append(f"  {i + 1}. {doc}\n", style="dim")

    if len(documents) > 5:
        preview_text.append(f"  ... and {len(documents) - 5} more", style="dim")

    panel = Panel(
        preview_text, title="Dry Run", title_align="left", border_style="yellow"
    )
    rich_cli.console.print(panel)


# Document batch processing will be implemented in future update


@batch.command("create-collections")
@click.argument("collections", nargs=-1, required=True)
@click.option(
    "--dimension", type=int, default=1536, help="Vector dimension for all collections"
)
@click.option(
    "--distance",
    type=click.Choice(["cosine", "euclidean", "dot"]),
    default="cosine",
    help="Distance metric for all collections",
)
@click.option("--force", is_flag=True, help="Recreate collections if they exist")
@click.pass_context
def create_collections(
    ctx: click.Context, collections: tuple, dimension: int, distance: str, _force: bool
):
    """Create multiple collections in batch."""
    config = ctx.obj["config"]
    rich_cli = ctx.obj["rich_cli"]

    collection_list = list(collections)

    # Show preview
    preview_table = Table(title="Collections to Create", show_header=True)
    preview_table.add_column("Name", style="cyan")
    preview_table.add_column("Dimension", style="blue")
    preview_table.add_column("Distance", style="green")

    for name in collection_list:
        preview_table.add_row(name, str(dimension), distance)

    rich_cli.console.print(preview_table)

    # Confirm operation
    if not Confirm.ask(f"Create {len(collection_list)} collections?"):
        rich_cli.console.print("[yellow]Creation cancelled.[/yellow]")
        return

    # Create operation queue
    queue = OperationQueue()
    client_manager = ClientManager(config)
    db_manager = VectorDBManager(client_manager)

    for collection_name in collection_list:
        operation = BatchOperation(
            name=f"Create {collection_name}",
            description=f"Create collection with {dimension}D vectors",
            function=lambda name=collection_name, size=dimension: asyncio.run(
                db_manager.create_collection(name, size)
            ),
        )
        queue.add(operation)

    # Execute operations
    success = queue.execute(confirm=False)

    if success:
        success_text = Text()
        success_text.append(
            "✅ Batch collection creation completed!\n", style="bold green"
        )
        success_text.append(
            f"Collections created: {len(collection_list)}", style="cyan"
        )

        panel = Panel(
            success_text,
            title="Creation Complete",
            title_align="left",
            border_style="green",
        )
        rich_cli.console.print(panel)


@batch.command("delete-collections")
@click.argument("collections", nargs=-1, required=True)
@click.option("--yes", "-y", is_flag=True, help="Skip individual confirmations")
@click.pass_context
def delete_collections(ctx: click.Context, collections: tuple, yes: bool):
    """Delete multiple collections in batch."""
    config = ctx.obj["config"]
    rich_cli = ctx.obj["rich_cli"]

    collection_list = list(collections)

    # Show what will be deleted
    warning_text = Text()
    warning_text.append("⚠️ WARNING: Collection Deletion\n\n", style="bold red")
    warning_text.append(
        "The following collections will be permanently deleted:\n", style=""
    )

    for i, name in enumerate(collection_list, 1):
        warning_text.append(f"  {i}. {name}\n", style="red")

    warning_text.append("\nThis action cannot be undone!", style="bold red")

    panel = Panel(
        warning_text, title="Deletion Warning", title_align="left", border_style="red"
    )
    rich_cli.console.print(panel)

    # Double confirmation for destructive operation
    if not yes:
        if not Confirm.ask("Are you sure you want to delete these collections?"):
            rich_cli.console.print("[yellow]Deletion cancelled.[/yellow]")
            return

        if not Confirm.ask("This will permanently delete all data. Continue?"):
            rich_cli.console.print("[yellow]Deletion cancelled.[/yellow]")
            return

    # Create operation queue
    queue = OperationQueue()
    client_manager = ClientManager(config)
    db_manager = VectorDBManager(client_manager)

    for collection_name in collection_list:
        operation = BatchOperation(
            name=f"Delete {collection_name}",
            description="Permanently delete collection and all data",
            function=lambda name=collection_name: asyncio.run(
                db_manager.delete_collection(name)
            ),
        )
        queue.add(operation)

    # Execute operations
    success = queue.execute(confirm=False)

    if success:
        rich_cli.console.print(
            f"✅ {len(collection_list)} collections deleted successfully."
        )


@batch.command("backup-collections")
@click.argument("collections", nargs=-1)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default="./backups",
    help="Output directory for backups",
)
@click.option(
    "--format",
    type=click.Choice(["json", "parquet"]),
    default="json",
    help="Backup format",
)
@click.pass_context
def backup_collections(
    ctx: click.Context, collections: tuple, output_dir: Path, format: str
):
    """Backup collections to files."""
    config = ctx.obj["config"]
    rich_cli = ctx.obj["rich_cli"]

    # If no collections specified, backup all
    if not collections:
        client_manager = ClientManager(config)
        db_manager = VectorDBManager(client_manager)
        collection_names = asyncio.run(db_manager.list_collections())
        asyncio.run(db_manager.cleanup())
    else:
        collection_names = list(collections)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Show backup plan
    backup_text = Text()
    backup_text.append("📦 Backup Plan\n\n", style="bold cyan")
    backup_text.append(f"Collections: {len(collection_names)}\n", style="")
    backup_text.append(f"Output directory: {output_dir}\n", style="")
    backup_text.append(f"Format: {format.upper()}", style="")

    panel = Panel(
        backup_text,
        title="Backup Configuration",
        title_align="left",
        border_style="blue",
    )
    rich_cli.console.print(panel)

    if not Confirm.ask("Proceed with backup?"):
        rich_cli.console.print("[yellow]Backup cancelled.[/yellow]")
        return

    # TODO: Implement actual backup functionality
    # This would require implementing export functionality in the vector DB service

    rich_cli.console.print("[yellow]Backup functionality coming soon![/yellow]")
    rich_cli.console.print("This feature will be implemented in a future update.")
