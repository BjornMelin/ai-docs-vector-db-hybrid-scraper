#!/usr/bin/env python3
"""
Fix TRY300 violations by moving statements to else blocks.
"""

import json  # noqa: PLC0415
import subprocess
from pathlib import Path
from typing import Dict, List


def get_violations() -> List[Dict]:
    """Get all TRY300 violations using ruff."""
    result = subprocess.run(
        ["uv", "run", "ruff", "check", ".", "--select=TRY300", "--output-format=json"],
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )

    if result.returncode == 0:
        return []

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return []


def fix_try300_simple_cases():
    """Fix simple TRY300 cases where we can easily move statements to else blocks."""
    violations = get_violations()

    # Group violations by file
    by_file = {}
    for v in violations:
        filepath = v["filename"]
        if filepath not in by_file:
            by_file[filepath] = []
        by_file[filepath].append(v)

    fixed_count = 0

    for filepath, file_violations in by_file.items():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            original_lines = lines[:]

            # Sort violations by line number (descending to avoid index issues)
            file_violations.sort(key=lambda x: x["location"]["row"], reverse=True)

            for violation in file_violations:
                row = violation["location"]["row"] - 1  # Convert to 0-based
                if row >= len(lines):
                    continue

                line = lines[row].rstrip()
                indent = len(line) - len(line.lstrip())

                # Look for simple patterns we can fix
                if "return " in line:
                    # Find the corresponding try block
                    try_line = None
                    for i in range(row - 1, -1, -1):
                        if lines[i].strip().startswith("try:"):
                            try_line = i
                            break

                    if try_line is not None:
                        # Find the except block
                        except_line = None
                        for i in range(try_line + 1, len(lines)):
                            if lines[i].strip().startswith("except"):
                                except_line = i
                                break

                        if except_line is not None:
                            # Insert else block before except
                            else_indent = " " * (
                                len(lines[try_line]) - len(lines[try_line].lstrip())
                            )
                            statement_indent = " " * (indent)

                            else_block = f"{else_indent}else:\n{statement_indent}{line.strip()}\n"

                            # Remove the original return statement
                            lines[row] = ""

                            # Insert else block
                            lines.insert(except_line, else_block)
                            fixed_count += 1

                elif "break" in line or "continue" in line:
                    # Similar logic for break/continue statements
                    # This is more complex and risky, so we'll skip for now
                    pass

            if lines != original_lines:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                print(f"Fixed TRY300 violations in {filepath}")

        except Exception:
            print(f"Error processing {filepath}: {e}")

    return fixed_count


def main():
    """Main entry point."""
    print("🔧 Fixing TRY300 violations...")

    initial_violations = get_violations()
    initial_count = len(initial_violations)
    print(f"Initial TRY300 violations: {initial_count}")

    if initial_count == 0:
        print("No TRY300 violations found!")
        return

    fixed = fix_try300_simple_cases()

    final_violations = get_violations()
    final_count = len(final_violations)

    print(f"Fixed: {fixed}")
    print(f"Remaining TRY300 violations: {final_count}")
    print(f"Reduction: {initial_count - final_count}")


if __name__ == "__main__":
    main()
