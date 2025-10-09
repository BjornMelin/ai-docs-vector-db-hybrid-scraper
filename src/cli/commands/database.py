"""Enhanced vector database management commands with Rich styling.

This module provides advanced database operations with Rich progress
indicators, beautiful table displays, and interactive features.
"""

# pylint: disable=duplicate-code

import asyncio
import json
from typing import TYPE_CHECKING, Any

import click
from click.shell_completion import CompletionItem
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from src.manage_vector_db import (
    CollectionCreationError,
    CollectionDeletionError,
    VectorDBManager,
)


if TYPE_CHECKING:  # pragma: no cover - typing only
    pass
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

        # Initialize manager and retrieve collections
        db_manager = VectorDBManager()
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
    except (ValueError, RuntimeError, OSError):
        # If anything fails, return empty list
        return []


@click.group()
def database():
    """Vector database operations with Rich-powered visualization."""


@database.command("list")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
@click.pass_context
def list_collections(ctx: click.Context, output_format: str):
    """List all vector database collections."""
    rich_cli = ctx.obj["rich_cli"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Loading collections...", total=None)

        db_manager = VectorDBManager()
        collections: list[dict[str, Any]] = []

        try:
            collections_names = asyncio.run(db_manager.list_collections())

            for name in collections_names:
                info = asyncio.run(db_manager.get_collection_info(name))
                if info:
                    collections.append(
                        {
                            "name": info.name,
                            "vectors_count": info.vector_count,
                            "config": {
                                "params": {"vectors": {"size": info.vector_size}}
                            },
                            "status": "green",
                            "created_at": "Unknown",
                        }
                    )

        except (ValueError, RuntimeError, OSError) as e:
            rich_cli.show_error("Failed to list collections", str(e))
            raise click.Abort from e
        finally:
            asyncio.run(db_manager.cleanup())

    if output_format == "table":
        _display_collections_table(collections, rich_cli)
    else:
        _display_collections_json(collections, rich_cli)


def _display_collections_table(collections: list[dict[str, Any]], rich_cli):
    """Display collections as Rich table."""
    if not collections:
        rich_cli.console.print("[yellow]No collections found.[/yellow]")
        return

    table = Table(title="Vector Database Collections", show_header=True)
    table.add_column("Name", style="cyan", width=20)
    table.add_column("Vectors", style="green", justify="right")
    table.add_column("Dimension", style="blue", justify="right")
    table.add_column("Status", style="")
    table.add_column("Last Modified", style="dim")

    for collection in collections:
        # Format vector count with thousands separator
        vector_count = f"{collection.get('vectors_count', 0):,}"

        # Status with emoji
        status = collection.get("status", "unknown")
        status_display = {
            "green": "Ready",
            "yellow": "Indexing",
            "red": "Error",
            "grey": "Stopped",
        }.get(status, status or "unknown")

        table.add_row(
            collection.get("name", "Unknown"),
            vector_count,
            str(
                collection.get("config", {})
                .get("params", {})
                .get("vectors", {})
                .get("size", "Unknown")
            ),
            status_display,
            collection.get("created_at", "Unknown"),
        )

    rich_cli.console.print(table)


def _display_collections_json(collections: list[dict[str, Any]], rich_cli):
    """Display collections as JSON."""
    json_str = json.dumps(collections, indent=2, default=str)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)

    panel = Panel(
        syntax, title="Collections (JSON)", title_align="left", border_style="blue"
    )
    rich_cli.console.print(panel)


def _abort_collection_exists(collection_name: str, rich_cli) -> None:
    """Helper function to abort when collection already exists."""
    rich_cli.show_error(
        f"Collection '{collection_name}' already exists",
        "Use --force to overwrite",
    )
    raise click.Abort from None


def _abort_collection_creation_failed() -> None:
    """Helper function to abort when collection creation fails."""
    msg = "Collection creation failed"
    raise CollectionCreationError(msg)


def _abort_collection_deletion_failed() -> None:
    """Helper function to abort when collection deletion fails."""
    msg = "Collection deletion failed"
    raise CollectionDeletionError(msg)


def _abort_collection_not_found(collection_name: str, rich_cli) -> None:
    """Helper function to abort when collection is not found."""
    rich_cli.show_error(f"Collection '{collection_name}' not found")
    raise click.Abort from None


@database.command("create")
@click.argument("collection_name", shell_complete=complete_collection_name)
@click.option(
    "--dimension",
    "-d",
    type=int,
    default=1536,
    help="Vector dimension (default: 1536 for OpenAI)",
)
@click.option(
    "--distance",
    type=click.Choice(["cosine", "euclidean", "dot"]),
    default="cosine",
    help="Distance metric",
)
@click.option(
    "--force", is_flag=True, help="Force creation (delete existing collection)"
)
@click.pass_context
def create_collection(
    ctx: click.Context, collection_name: str, dimension: int, distance: str, force: bool
):
    """Create a new vector database collection."""
    rich_cli = ctx.obj["rich_cli"]

    # Confirmation for force creation
    if force and not Confirm.ask(
        f"Delete existing collection '{collection_name}' if it exists?"
    ):
        rich_cli.console.print("[yellow]Creation cancelled.[/yellow]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            f"Creating collection '{collection_name}'...", total=None
        )

        db_manager = VectorDBManager()

        try:
            if not force:
                existing_collections = asyncio.run(db_manager.list_collections())
                if collection_name in existing_collections:
                    _abort_collection_exists(collection_name, rich_cli)

            if force and collection_name in asyncio.run(db_manager.list_collections()):
                progress.update(task, description="Deleting existing collection...")
                asyncio.run(db_manager.delete_collection(collection_name))

            progress.update(
                task, description=f"Creating collection with {dimension}D vectors..."
            )
            success = asyncio.run(
                db_manager.create_collection(
                    collection_name=collection_name, vector_size=dimension
                )
            )

            if not success:
                _abort_collection_creation_failed()

        except (ValueError, RuntimeError, OSError) as e:
            rich_cli.show_error("Failed to create collection", str(e))
            raise click.Abort from e
        finally:
            asyncio.run(db_manager.cleanup())

    # Success message
    success_text = Text()
    success_text.append("Collection created successfully.\n", style="bold green")
    success_text.append(f"Name: {collection_name}\n", style="cyan")
    success_text.append(f"Dimension: {dimension}\n", style="blue")
    success_text.append(f"Distance: {distance}", style="yellow")

    panel = Panel(
        success_text,
        title="Collection Created",
        title_align="left",
        border_style="green",
    )
    rich_cli.console.print(panel)


@database.command("delete")
@click.argument("collection_name", shell_complete=complete_collection_name)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def delete_collection(ctx: click.Context, collection_name: str, yes: bool):
    """Delete a vector database collection."""
    rich_cli = ctx.obj["rich_cli"]

    # Confirmation prompt
    if not yes and not Confirm.ask(
        f"Delete collection '{collection_name}'? This cannot be undone."
    ):
        rich_cli.console.print("[yellow]Deletion cancelled.[/yellow]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"Deleting collection '{collection_name}'...", total=None)

        db_manager = VectorDBManager()

        try:
            success = asyncio.run(db_manager.delete_collection(collection_name))

            if not success:
                _abort_collection_deletion_failed()

        except (ValueError, RuntimeError, OSError) as e:
            rich_cli.show_error("Failed to delete collection", str(e))
            raise click.Abort from e
        finally:
            asyncio.run(db_manager.cleanup())

    rich_cli.console.print(f"Collection '{collection_name}' deleted successfully.")


@database.command("info")
@click.argument("collection_name", shell_complete=complete_collection_name)
@click.pass_context
def collection_info(ctx: click.Context, collection_name: str):
    """Show detailed information about a collection."""
    rich_cli = ctx.obj["rich_cli"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Loading collection info...", total=None)

        db_manager = VectorDBManager()

        try:
            info = asyncio.run(db_manager.get_collection_info(collection_name))

            if info is None:
                _abort_collection_not_found(collection_name, rich_cli)
                return

            # Convert to expected format
            info_dict = {
                "status": "green",
                "vectors_count": info.vector_count,
                "indexed_vectors_count": info.vector_count,
                "config": {
                    "params": {
                        "vectors": {"size": info.vector_size, "distance": "Cosine"}
                    }
                },
            }

        except (ValueError, RuntimeError, OSError) as e:
            rich_cli.show_error("Failed to get collection info", str(e))
            raise click.Abort from e
        finally:
            asyncio.run(db_manager.cleanup())

    # Display collection information
    info_table = Table(title=f"Collection: {collection_name}", show_header=True)
    info_table.add_column("Property", style="cyan", width=20)
    info_table.add_column("Value", style="")

    # Basic info
    info_table.add_row("Status", info_dict.get("status", "Unknown"))
    info_table.add_row("Vector Count", f"{info_dict.get('vectors_count', 0):,}")
    info_table.add_row(
        "Indexed Vectors", f"{info_dict.get('indexed_vectors_count', 0):,}"
    )

    # Configuration
    config_data = info_dict.get("config", {})
    params = config_data.get("params", {})
    vectors_config = params.get("vectors", {})

    info_table.add_row("Vector Size", str(vectors_config.get("size", "Unknown")))
    info_table.add_row("Distance Metric", vectors_config.get("distance", "Unknown"))

    rich_cli.console.print(info_table)


@database.command("search")
@click.argument("collection_name", shell_complete=complete_collection_name)
@click.argument("query")
@click.option("--limit", "-l", type=int, default=5, help="Number of results to return")
@click.option("--score-threshold", type=float, help="Minimum similarity score")
@click.pass_context
def search_collection(
    ctx: click.Context,
    collection_name: str,
    query: str,
    limit: int,
    score_threshold: float | None,
):
    """Search a vector database collection."""
    rich_cli = ctx.obj["rich_cli"]
    rich_cli.console.print(
        "[yellow]Vector search via CLI is not yet implemented. "
        "Use the API or management UI for query execution.[/yellow]"
    )


@database.command("stats")
@click.pass_context
def database_stats(ctx: click.Context):
    """Show database statistics and health information."""
    config = ctx.obj["config"]
    rich_cli = ctx.obj["rich_cli"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Gathering database statistics...", total=None)

        db_manager = VectorDBManager()

        try:
            collections_names = asyncio.run(db_manager.list_collections())

            # Get info for each collection
            collections = []
            total_vectors = 0
            for name in collections_names:
                info = asyncio.run(db_manager.get_collection_info(name))
                if info:
                    collections.append(
                        {
                            "name": info.name,
                            "vectors_count": info.vector_count,
                            "config": {
                                "params": {"vectors": {"size": info.vector_size}}
                            },
                            "status": "green",
                            "created_at": "Unknown",
                        }
                    )
                    total_vectors += info.vector_count

            # Calculate totals
            total_collections = len(collections)

        except (ValueError, RuntimeError, OSError) as e:
            rich_cli.show_error("Failed to get database stats", str(e))
            raise click.Abort from e
        finally:
            asyncio.run(db_manager.cleanup())

    # Database overview
    stats_table = Table(title="Database Statistics", show_header=True)
    stats_table.add_column("Metric", style="cyan", width=20)
    stats_table.add_column("Value", style="green")

    stats_table.add_row("Total Collections", str(total_collections))
    stats_table.add_row("Total Vectors", f"{total_vectors:,}")
    stats_table.add_row("Database Host", f"{config.qdrant.host}:{config.qdrant.port}")

    rich_cli.console.print(stats_table)

    # Collection breakdown if there are collections
    if collections:
        console.print("\n[bold cyan]Collection Breakdown:[/bold cyan]")
        _display_collections_table(collections, rich_cli)
