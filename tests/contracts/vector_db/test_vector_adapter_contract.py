"""Contract tests for vector adapters.

Phase 0 introduces a placeholder so the suite is ready once the consolidated
adapter lands. Tests are currently skipped until the adapter implementation is
available.
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
@pytest.mark.skip(reason="Vector adapter consolidation pending")
async def test_vector_adapter_contract_placeholder() -> None:
    """Placeholder until the unified adapter is implemented."""

    assert True
