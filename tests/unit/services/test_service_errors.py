"""Tests for service error utilities and sanitisation."""

from __future__ import annotations

import re

import pytest
from pydantic import BaseModel, ValidationError as PydanticValidationError

from src.services.errors import (
    ValidationError,
    safe_response,
)


class Payload(BaseModel):
    value: int


def test_validation_error_from_pydantic_single_field() -> None:
    with pytest.raises(PydanticValidationError) as exc_info:
        Payload.model_validate({"value": "abc"})

    validation_error = ValidationError.from_pydantic(exc_info.value)

    snapshot = validation_error.to_dict()
    assert snapshot["error_code"] == "validation_error"
    assert "value" in snapshot["context"]["field"]


def test_safe_response_sanitises_error_messages() -> None:
    response = safe_response(
        False,
        error="Failed at /tmp/secret with api_key=abcd",
        error_type="network",
    )

    assert response["success"] is False
    assert response["error_type"] == "network"
    assert "/****/" in response["error"]
    assert not re.search("api_key", response["error"], re.IGNORECASE)
