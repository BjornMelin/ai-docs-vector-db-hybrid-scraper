"""Utilities for auditing and normalizing Qdrant payload metadata."""

from __future__ import annotations

import asyncio
from collections import Counter

import typer
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.conversions.common_types import PointId

from src.config import get_settings
from src.services.vector_db.payload_schema import (
    PayloadValidationError,
    ensure_canonical_payload,
)


app = typer.Typer(add_completion=False, help="Audit Qdrant payload schema compliance.")


async def _collect_records(
    client: AsyncQdrantClient,
    collection: str,
    limit: int,
    batch_size: int,
) -> list[models.Record]:
    """Retrieve records from a collection respecting the requested limit."""

    gathered: list[models.Record] = []
    next_offset: PointId | None = None
    remaining = limit if limit > 0 else None

    while remaining is None or remaining > 0:
        request_limit = batch_size if remaining is None else min(batch_size, remaining)
        points, next_offset = await client.scroll(
            collection_name=collection,
            with_payload=True,
            with_vectors=False,
            limit=request_limit,
            offset=next_offset,
        )
        if not points:
            break
        gathered.extend(points)
        if remaining is not None:
            remaining -= len(points)
        if not next_offset:
            break
    return gathered


async def _audit_collection(
    collection: str,
    limit: int,
    apply: bool,
    batch_size: int,
) -> None:
    """Audit payload metadata and optionally persist canonical fields."""

    config = get_settings()
    client = AsyncQdrantClient(url=config.qdrant.url, api_key=config.qdrant.api_key)
    try:
        records = await _collect_records(client, collection, limit, batch_size)
        total = len(records)
        if total == 0:
            typer.echo("No points found for audit.")
            return

        stats = Counter(
            {
                "missing_content": 0,
                "invalid_payloads": 0,
                "mismatched_ids": 0,
                "updated": 0,
            }
        )
        hash_counts: Counter[str] = Counter()

        for record in records:
            payload = dict(record.payload or {})
            content = payload.get("content")
            if not isinstance(content, str) or not content.strip():
                stats["missing_content"] += 1
                continue
            try:
                canonical = ensure_canonical_payload(
                    payload,
                    content=content,
                    id_hint=str(record.id),
                )
            except PayloadValidationError:
                stats["invalid_payloads"] += 1
                continue

            hash_counts[canonical.payload["content_hash"]] += 1

            if canonical.point_id != str(record.id):
                stats["mismatched_ids"] += 1

            if canonical.payload != payload:
                stats["updated"] += 1
                if apply:
                    await client.set_payload(
                        collection_name=collection,
                        payload=canonical.payload,
                        points=[record.id],
                    )
    finally:
        await client.close()

    duplicate_hashes = sum(count for count in hash_counts.values() if count > 1)

    typer.echo(f"Audited records: {total}")
    typer.echo(f"Missing content: {stats['missing_content']}")
    typer.echo(f"Invalid payloads: {stats['invalid_payloads']}")
    typer.echo(f"Mismatched point identifiers: {stats['mismatched_ids']}")
    typer.echo(f"Payloads requiring normalization: {stats['updated']}")
    typer.echo(f"Duplicate content hashes: {duplicate_hashes}")

    if apply:
        typer.echo("Updated payloads written back to Qdrant.")


@app.command()
def audit(
    collection: str = typer.Option(..., help="Target collection name."),
    limit: int = typer.Option(
        -1,
        help="Maximum number of records to inspect (negative scans all records).",
    ),
    apply: bool = typer.Option(
        False,
        help="Persist canonical payload fields back to Qdrant.",
    ),
    batch_size: int = typer.Option(256, help="Batch size for scroll operations."),
) -> None:
    """Audit Qdrant payload metadata for canonical field coverage."""

    record_limit = limit if limit > 0 else -1
    asyncio.run(_audit_collection(collection, record_limit, apply, batch_size))


if __name__ == "__main__":  # pragma: no cover
    app()
