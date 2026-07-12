"""Load the configuration templates distributed with the package."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


_TEMPLATE_PACKAGE = "src.config.templates"


def _load_json_object(filename: str) -> dict[str, Any]:
    """Return a packaged JSON asset as an object."""
    resource = files(_TEMPLATE_PACKAGE).joinpath(filename)
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Packaged template asset {filename} must contain a JSON object."
        raise TypeError(msg)
    return payload


def load_builtin_template_assets() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the canonical base template and built-in profile index."""
    return _load_json_object("base.json"), _load_json_object("profiles.json")


__all__ = ["load_builtin_template_assets"]
