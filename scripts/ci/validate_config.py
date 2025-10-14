"""Utilities for validating configuration assets in CI pipelines."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_REQUIRED_TEMPLATE_KEYS = (
    "environment",
    "cache",
    "qdrant",
    "performance",
)


@dataclass
class ValidationSummary:
    """Summary information for a validation category."""

    checked: int
    errors: list[str]


def _iter_files(root: Path, patterns: Sequence[str]) -> Iterable[Path]:
    """Yield files under *root* that match any of the glob *patterns*."""

    for pattern in patterns:
        yield from root.rglob(pattern)


def _validate_files(
    root: Path,
    patterns: Sequence[str],
    loader: Callable[[Path], object | None],
    exception_types: tuple[type[BaseException], ...],
    error_prefix: str,
) -> ValidationSummary:
    """Validate files matched by *patterns* using *loader*.

    Args:
        root: Base directory to search.
        patterns: Glob patterns to evaluate beneath *root*.
        loader: Callable that inspects the matched file and raises on error.
        exception_types: Exceptions that the loader is expected to raise when
            validation fails.
        error_prefix: Human-friendly prefix for surfaced error messages.

    Returns:
        A :class:`ValidationSummary` recording file counts and parse errors.
    """

    errors: list[str] = []
    checked = 0

    for file_path in _iter_files(root, patterns):
        checked += 1
        try:
            loader(file_path)
        except exception_types as exc:
            errors.append(f"{error_prefix} {file_path}: {exc}")

    return ValidationSummary(checked=checked, errors=errors)


def validate_json_files(root: Path) -> ValidationSummary:
    """Validate that JSON files under *root* parse correctly."""

    return _validate_files(
        root,
        ("*.json",),
        lambda path: json.loads(path.read_text(encoding="utf-8")),
        (json.JSONDecodeError,),
        "Invalid JSON in",
    )


def validate_yaml_files(root: Path) -> ValidationSummary:
    """Validate that YAML files under *root* parse correctly."""

    return _validate_files(
        root,
        ("*.yml", "*.yaml"),
        lambda path: yaml.safe_load(path.read_text(encoding="utf-8")),
        (yaml.YAMLError,),
        "Invalid YAML in",
    )


def validate_templates(
    templates_dir: Path,
    required_keys: Sequence[str],
    environment: str | None,
) -> ValidationSummary:
    """Validate configuration templates for required keys."""

    errors: list[str] = []
    checked = 0

    if not templates_dir.exists():
        return ValidationSummary(
            checked=checked,
            errors=[f"Templates directory not found: {templates_dir}"],
        )

    base_path = templates_dir / "base.json"
    profiles_path = templates_dir / "profiles.json"

    try:
        base_template = json.loads(base_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"Base template missing: {base_path}")
        return ValidationSummary(checked=checked, errors=errors)
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid JSON in base template {base_path}: {exc}")
        return ValidationSummary(checked=checked, errors=errors)

    try:
        profiles_index = json.loads(profiles_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"Profile index missing: {profiles_path}")
        return ValidationSummary(checked=checked, errors=errors)
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid JSON in profile index {profiles_path}: {exc}")
        return ValidationSummary(checked=checked, errors=errors)

    if not isinstance(base_template, dict):
        errors.append(f"Base template {base_path} must contain a JSON object")
        return ValidationSummary(checked=checked, errors=errors)
    if not isinstance(profiles_index, dict):
        errors.append(f"Profile index {profiles_path} must contain a JSON object")
        return ValidationSummary(checked=checked, errors=errors)

    names = sorted(profiles_index)
    if environment:
        if environment not in profiles_index:
            errors.append(
                f"Environment template '{environment}' is missing from profiles.json"
            )
            return ValidationSummary(checked=checked, errors=errors)
        names = [environment]

    def merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        merged = deepcopy(base)
        stack: list[tuple[dict[str, Any], dict[str, Any]]] = [(merged, overrides)]
        while stack:
            target, source = stack.pop()
            for key, value in source.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    stack.append((target[key], value))
                else:
                    target[key] = value
        return merged

    for name in names:
        record = profiles_index[name]
        if not isinstance(record, dict):
            errors.append(f"Profile '{name}' in {profiles_path} must be a JSON object")
            continue

        overrides = record.get("overrides", {})
        if not isinstance(overrides, dict):
            errors.append(f"Profile '{name}' overrides must be a JSON object")
            continue

        template_data = merge(base_template, overrides)
        checked += 1

        missing_keys = [key for key in required_keys if key not in template_data]
        if missing_keys:
            errors.append(f"Template '{name}' is missing required keys: {missing_keys}")

        if environment == name:
            actual_env = template_data.get("environment")
            if actual_env != environment:
                errors.append(
                    f"Template '{name}' env '{actual_env}' != '{environment}'."
                )

    return ValidationSummary(checked=checked, errors=errors)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Return parsed command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("config"),
        help="Root directory containing configuration assets.",
    )
    parser.add_argument(
        "--templates-dir",
        type=Path,
        default=Path("config/templates"),
        help="Directory containing environment templates.",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default=None,
        help="Limit validation to a specific environment template.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute validation and return the appropriate exit code."""

    args = parse_args(argv)

    config_root = args.config_root
    if not config_root.exists():
        print(f"Configuration root not found: {config_root}", file=sys.stderr)
        return 1

    summaries = [
        validate_json_files(config_root),
        validate_yaml_files(config_root),
        validate_templates(
            args.templates_dir, DEFAULT_REQUIRED_TEMPLATE_KEYS, args.environment
        ),
    ]

    total_checked = sum(summary.checked for summary in summaries)
    errors = [error for summary in summaries for error in summary.errors]

    for summary, label in zip(
        summaries,
        ("JSON", "YAML", "Templates"),
        strict=True,
    ):
        print(f"{label} files validated: {summary.checked}")

    if errors:
        print("\nConfiguration validation failed:", file=sys.stderr)
        for error in errors:
            print(f" - {error}", file=sys.stderr)
        return 1

    print(f"\nConfiguration validation passed for {total_checked} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
