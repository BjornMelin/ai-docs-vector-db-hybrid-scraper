"""HTTP mocking fixtures built on top of respx."""

from __future__ import annotations

import httpx
import pytest
import respx


@pytest.fixture
def respx_mock():
    """Provide a respx router preloaded with common service mocks."""
    with respx.mock(assert_all_mocked=False) as mock:
        mock.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "embedding": [0.1, 0.2, 0.3],
                            "index": 0,
                        }
                    ]
                },
            )
        )
        mock.get("https://test.example.com").mock(
            return_value=httpx.Response(200, text="Test content"),
        )
        yield mock
