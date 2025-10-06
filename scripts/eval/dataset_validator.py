"""Validate the golden dataset for structural correctness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_KEYS = {
    "query",
    "expected_answer",
    "expected_contexts",
    "references",
    "metadata",
}
REQUIRED_METADATA_KEYS = {"collection"}


class DatasetValidationError(Exception):
    """Raised when the dataset fails structural validation."""


def load_dataset_records(path: Path) -> list[dict[str, Any]]:
    """Return all JSON objects contained in the file."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                msg = f"Malformed JSON at {path}:{index}: {exc.msg}"
                raise DatasetValidationError(msg) from exc
            rows.append(payload)
    return rows


def validate_dataset(path: Path) -> None:
    """Validate that the dataset satisfies structural constraints."""

    rows = load_dataset_records(path)
    if not rows:
        raise DatasetValidationError("Dataset is empty")

    for index, payload in enumerate(rows, start=1):
        missing = REQUIRED_KEYS - payload.keys()
        if missing:
            raise DatasetValidationError(f"Row {index} missing keys: {sorted(missing)}")
        if not isinstance(payload["expected_contexts"], list) or not all(
            isinstance(item, str) for item in payload["expected_contexts"]
        ):
            raise DatasetValidationError(
                f"Row {index} expected_contexts must be a list of strings"
            )
        if not isinstance(payload["references"], list) or not all(
            isinstance(item, str) for item in payload["references"]
        ):
            raise DatasetValidationError(
                f"Row {index} references must be a list of strings"
            )
        metadata = payload["metadata"]
        if not isinstance(metadata, dict):
            raise DatasetValidationError(f"Row {index} metadata must be an object")
        missing_meta = REQUIRED_METADATA_KEYS - metadata.keys()
        if missing_meta:
            raise DatasetValidationError(
                f"Row {index} metadata missing keys: {sorted(missing_meta)}"
            )


def main() -> None:
    """CLI entrypoint that validates the provided dataset file."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to the dataset JSONL file")
    args = parser.parse_args()

    validate_dataset(args.dataset)
    print("Dataset validation succeeded")


if __name__ == "__main__":
    main()
