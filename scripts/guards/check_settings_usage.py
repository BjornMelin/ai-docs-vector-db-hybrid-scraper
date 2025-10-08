#!/usr/bin/env python3
"""Fail when legacy configuration constructs leak outside `src/config`."""

from __future__ import annotations

import re
import sys
from pathlib import Path


FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern | None], ...] = (
    ("from src.config import Config", None),
    ("get_config(", re.compile(r"(?<![A-Za-z_])get_config\(")),
    ("set_config(", re.compile(r"(?<![A-Za-z_])set_config\(")),
    ("reset_config(", re.compile(r"(?<![A-Za-z_])reset_config\(")),
)

SKIP_DIRS = {
    ".git",
    ".venv",
    ".uv_cache",
    "__pycache__",
    "node_modules",
    "guards",
}


def scan_file(path: Path) -> list[str]:
    """Return forbidden patterns detected in ``path``."""

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    matches: list[str] = []
    for literal, regex in FORBIDDEN_PATTERNS:
        if regex is None:
            if literal in text:
                matches.append(literal)
            continue
        if regex.search(text):
            matches.append(literal)
    return matches


def main() -> int:
    """Scan repository tree for deprecated configuration helpers."""

    repo_root = Path(__file__).resolve().parents[2]
    violations: list[str] = []

    for absolute_path in repo_root.rglob("*.py"):
        relative = absolute_path.relative_to(repo_root)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if relative.as_posix().startswith("src/config"):
            continue
        matches = scan_file(absolute_path)
        if matches:
            pattern_list = ", ".join(sorted(set(matches)))
            violations.append(f"{relative} -> {pattern_list}")

    if violations:
        sys.stderr.write(
            "Detected deprecated configuration helpers outside src/config:\n"
        )
        sys.stderr.write("\n".join(violations) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
