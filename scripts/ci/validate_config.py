"""Utilities for validating configuration assets in CI pipelines."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

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


def validate_json_files(root: Path) -> ValidationSummary:
    """Validate that JSON files under *root* parse correctly."""

    errors: list[str] = []
    checked = 0

    for file_path in _iter_files(root, ("*.json",)):
        checked += 1
        try:
            json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid JSON in {file_path}: {exc}")

    return ValidationSummary(checked=checked, errors=errors)


def validate_yaml_files(root: Path) -> ValidationSummary:
    """Validate that YAML files under *root* parse correctly."""

    errors: list[str] = []
    checked = 0

    for file_path in _iter_files(root, ("*.yml", "*.yaml")):
        checked += 1
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                yaml.safe_load(handle)
        except yaml.YAMLError as exc:  # type: ignore[attr-defined]
            errors.append(f"Invalid YAML in {file_path}: {exc}")

    return ValidationSummary(checked=checked, errors=errors)


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

    candidates = sorted(templates_dir.glob("*.json"))
    if environment:
        requested_template = templates_dir / f"{environment}.json"
        if requested_template.exists():
            candidates = [requested_template]
        else:
            errors.append(
                f"Environment template '{requested_template.name}' is missing."
            )
            return ValidationSummary(checked=checked, errors=errors)

    for template_path in candidates:
        checked += 1
        try:
            template_data = json.loads(template_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid JSON in template {template_path}: {exc}")
            continue

        missing_keys = [key for key in required_keys if key not in template_data]
        if missing_keys:
            errors.append(
                f"Template {template_path} is missing required keys: {missing_keys}"
            )

        if environment and template_path.stem == environment:
            actual_env = template_data.get("environment")
            if actual_env != environment:
                errors.append(
                    f"Template {template_path} env '{actual_env}' != '{environment}'."
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
