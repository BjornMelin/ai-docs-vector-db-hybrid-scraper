"""Helpers for merging and diffing configuration template payloads."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True)
class MergeFrame:
    """Frame describing a pending merge operation."""

    destination: MutableMapping[str, Any]
    source: Mapping[str, Any]


@dataclass(frozen=True)
class DiffFrame:
    """Frame describing a pending diff evaluation."""

    path: tuple[str, ...]
    base_value: Any
    candidate: Any


def apply_overrides(
    target: MutableMapping[str, Any], overrides: Mapping[str, Any]
) -> None:
    """Apply ``overrides`` onto ``target`` in place."""

    stack: list[MergeFrame] = [MergeFrame(destination=target, source=overrides)]
    while stack:
        frame = stack.pop()
        for key, value in frame.source.items():
            destination_value = frame.destination.get(key)
            if isinstance(value, dict) and isinstance(destination_value, dict):
                stack.append(
                    MergeFrame(
                        destination=destination_value,
                        source=value,
                    )
                )
            else:
                frame.destination[key] = value


def merge_overrides(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a deep copy of ``base`` with ``overrides`` applied."""

    merged = cast(dict[str, Any], deepcopy(base))
    apply_overrides(merged, overrides)
    return merged


def calculate_diff(base: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
    """Return overrides required to transform ``base`` into ``data``."""

    diff: dict[str, Any] = {}
    stack: list[DiffFrame] = [DiffFrame(path=(), base_value=base, candidate=data)]
    while stack:
        frame = stack.pop()
        if isinstance(frame.candidate, dict):
            base_branch = (
                cast(dict[str, Any], frame.base_value)
                if isinstance(frame.base_value, dict)
                else {}
            )
            for key, value in frame.candidate.items():
                stack.append(
                    DiffFrame(
                        path=(*frame.path, key),
                        base_value=base_branch.get(key),
                        candidate=value,
                    )
                )
            continue

        if frame.path and frame.candidate != frame.base_value:
            cursor = diff
            for key in frame.path[:-1]:
                cursor = cast(dict[str, Any], cursor.setdefault(key, {}))
            cursor[frame.path[-1]] = frame.candidate

    return diff


__all__ = [
    "apply_overrides",
    "calculate_diff",
    "merge_overrides",
]
