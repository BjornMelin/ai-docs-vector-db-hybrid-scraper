"""Unit tests for browser runtime service."""

from __future__ import annotations

import pytest

from src.services.browser.models import ProviderKind
from src.services.browser.runtime import execute_with_retry


@pytest.mark.asyncio
async def test_execute_with_retry_eventually_succeeds() -> None:
    """Test that execute_with_retry retries a flaky operation."""
    attempts: list[int] = []

    async def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            # simulate transient failure
            raise TimeoutError("transient")
        return "ok"

    result = await execute_with_retry(
        provider=ProviderKind.LIGHTWEIGHT,
        operation="unit",
        func=flaky,
        attempts=3,
        min_wait=0.01,
        max_wait=0.02,
    )
    assert result == "ok"
    assert len(attempts) == 3
