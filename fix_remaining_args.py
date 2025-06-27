#!/usr/bin/env python3
"""Fix remaining unused argument violations."""

import re
import subprocess
from pathlib import Path


def fix_specific_patterns():
    """Fix specific patterns that weren't caught by the first script."""

    fixes = [
        # CLI setup file
        (
            "src/cli/commands/setup.py",
            [
                (260, "template_data", "_template_data"),
                (450, "output", "_output"),
            ],
        ),
        # Template manager
        (
            "src/cli/wizard/template_manager.py",
            [
                (65, "template_data", "_template_data"),
            ],
        ),
        # Config files
        (
            "src/config/cache_optimization.py",
            [
                (126, "ttl", "_ttl"),
            ],
        ),
        (
            "src/config/config_manager.py",
            [
                (83, "field", "_field"),
                (90, "field_name", "_field_name"),
                (90, "field", "_field"),
                (90, "value_is_complex", "_value_is_complex"),
            ],
        ),
        (
            "src/config/drift_detection.py",
            [
                (413, "source", "_source"),
                (466, "source", "_source"),
                (512, "source", "_source"),
            ],
        ),
        (
            "src/config/lifecycle.py",
            [
                (179, "old_config", "_old_config"),
            ],
        ),
    ]

    for file_path, file_fixes in fixes:
        path = Path(file_path)
        if not path.exists():
            continue

        content = path.read_text()
        lines = content.split("\n")

        # Sort fixes by line number in reverse order
        file_fixes.sort(key=lambda x: x[0], reverse=True)

        for line_num, old_arg, new_arg in file_fixes:
            if line_num <= len(lines):
                line = lines[line_num - 1]  # Convert to 0-based

                # Replace argument name with underscore version
                patterns = [
                    rf"\b{re.escape(old_arg)}\b(?=\s*:)",  # arg: type
                    rf"\b{re.escape(old_arg)}\b(?=\s*,)",  # arg,
                    rf"\b{re.escape(old_arg)}\b(?=\s*=)",  # arg=
                    rf"\b{re.escape(old_arg)}\b(?=\s*\))",  # arg)
                ]

                for pattern in patterns:
                    if re.search(pattern, line):
                        lines[line_num - 1] = re.sub(pattern, new_arg, line)
                        print(f"Fixed {old_arg} -> {new_arg} in {file_path}:{line_num}")
                        break

        path.write_text("\n".join(lines))


def fix_test_files():
    """Fix test files with mock arguments."""

    # Performance report files
    perf_files = [
        "src/config/performance_report.py",
    ]

    for file_path in perf_files:
        path = Path(file_path)
        if not path.exists():
            continue

        content = path.read_text()

        # Fix listener function arguments
        content = re.sub(
            r"def (\w*listener)\(old_cfg, new_cfg\):",
            r"def \1(_old_cfg, _new_cfg):",
            content,
        )
        content = re.sub(
            r"async def (\w*listener)\(old_cfg, new_cfg\):",
            r"async def \1(_old_cfg, _new_cfg):",
            content,
        )

        path.write_text(content)
        print(f"Fixed listener arguments in {file_path}")


def main():
    """Main function."""
    print("Fixing specific argument patterns...")
    fix_specific_patterns()

    print("Fixing test file patterns...")
    fix_test_files()

    print("Checking remaining violations...")
    result = subprocess.run(
        ["uv", "run", "ruff", "check", ".", "--select=ARG001,ARG002"],
        capture_output=True,
        text=True,
    )

    violations = [
        l for l in result.stderr.split("\n") if "ARG001" in l or "ARG002" in l
    ]
    print(f"Remaining violations: {len(violations)}")

    if violations:
        print("First 10 remaining violations:")
        for v in violations[:10]:
            print(f"  {v}")


if __name__ == "__main__":
    main()
