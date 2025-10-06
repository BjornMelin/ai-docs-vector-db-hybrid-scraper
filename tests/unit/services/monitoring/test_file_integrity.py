"""Tests for file integrity providers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.services.monitoring.file_integrity import (
    FileChangeAction,
    FileChangeEvent,
    OsqueryFileIntegrityProvider,
    StubFileIntegrityProvider,
)


@pytest.mark.asyncio
async def test_stub_provider_emits_events() -> None:
    """Stub provider should deliver events to subscribers."""

    provider = StubFileIntegrityProvider()
    received: list[FileChangeEvent] = []

    def _collector(event: FileChangeEvent) -> None:
        received.append(event)

    provider.subscribe(_collector)
    await provider.start()
    assert await provider.wait_until_ready(timeout=0.1) is True

    event = FileChangeEvent(path=Path("/tmp/test.txt"), action=FileChangeAction.CREATED)
    await provider.emit(event)

    assert received == [event]

    await provider.stop()
    assert provider.is_ready() is False


def test_osquery_parse_event(tmp_path: Path) -> None:
    """Osquery provider should parse file_events rows."""

    provider = OsqueryFileIntegrityProvider(
        results_log=tmp_path / "osqueryd.results.log"
    )

    payload = json.dumps(
        {
            "name": "file_events",
            "columns": {
                "action": "UPDATED",
                "target_path": "/etc/example.conf",
                "size": "128",
                "uid": "1000",
                "gid": "1000",
                "mode": "644",
                "time": "1700000000",
                "md5": "abc123",
                "sha256": "deadbeef",
            },
        }
    )

    event = provider._parse_osquery_event(payload)
    assert event is not None
    assert event.path == Path("/etc/example.conf")
    assert event.action is FileChangeAction.UPDATED
    assert event.size == 128
    assert event.uid == 1000
    assert event.hashes["md5"] == "abc123"
    assert event.hashes["sha256"] == "deadbeef"


def test_osquery_filter_globs(tmp_path: Path) -> None:
    """Include/exclude filters should gate emitted events."""

    provider = OsqueryFileIntegrityProvider(
        results_log=tmp_path / "osqueryd.results.log",
        include_globs=["/etc/*"],
        exclude_globs=["/etc/secret/*"],
    )

    assert provider._filter_event(Path("/etc/app/config.yml")) is True
    assert provider._filter_event(Path("/etc/secret/keys")) is False
    assert provider._filter_event(Path("/var/log/syslog")) is False
