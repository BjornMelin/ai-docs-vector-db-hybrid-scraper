#!/usr/bin/env python3
"""Fix all remaining unused argument violations."""

import re
import subprocess
from pathlib import Path


def get_all_violations():
    """Get all remaining ARG violations."""
    result = subprocess.run(
        ["uv", "run", "ruff", "check", ".", "--select=ARG001,ARG002"],
        capture_output=True,
        text=True,
    )

    violations = []
    # Ruff outputs to stderr
    output_lines = result.stderr.split("\n")

    for line in output_lines:
        if ".py:" in line and ("ARG001" in line or "ARG002" in line):
            # Parse the format: file.py:line:col: ARG001 Unused function argument: `arg_name`
            match = re.match(
                r"([^:]+):(\d+):\d+: ARG\d+ Unused \w+ argument: `(\w+)`", line
            )
            if match:
                file_path, line_num, arg_name = match.groups()
                violations.append((file_path, int(line_num), arg_name))

    print(f"Parsed {len(violations)} violations from {len(output_lines)} output lines")
    return violations


def fix_all_violations(violations):
    """Fix all violations by prefixing with underscore."""
    files_to_fix = {}

    # Group by file
    for file_path, line_num, arg_name in violations:
        if file_path not in files_to_fix:
            files_to_fix[file_path] = []
        files_to_fix[file_path].append((line_num, arg_name))

    # Fix each file
    for file_path, file_violations in files_to_fix.items():
        path = Path(file_path)
        if not path.exists():
            continue

        try:
            content = path.read_text()
            lines = content.split("\n")

            # Sort by line number in reverse order
            file_violations.sort(key=lambda x: x[0], reverse=True)

            for line_num, arg_name in file_violations:
                if line_num <= len(lines):
                    line = lines[line_num - 1]  # Convert to 0-based

                    # Skip if already prefixed
                    if f"_{arg_name}" in line:
                        continue

                    # Replace with underscore-prefixed version
                    patterns = [
                        rf"\b{re.escape(arg_name)}\b(?=\s*:)",  # arg: type
                        rf"\b{re.escape(arg_name)}\b(?=\s*,)",  # arg,
                        rf"\b{re.escape(arg_name)}\b(?=\s*=)",  # arg=
                        rf"\b{re.escape(arg_name)}\b(?=\s*\))",  # arg)
                    ]

                    for pattern in patterns:
                        if re.search(pattern, line):
                            lines[line_num - 1] = re.sub(pattern, f"_{arg_name}", line)
                            break

            path.write_text("\n".join(lines))
            print(f"Fixed {len(file_violations)} violations in {file_path}")

        except Exception as e:
            print(f"Error fixing {file_path}: {e}")


def main():
    """Main function."""
    print("Getting all remaining violations...")
    violations = get_all_violations()

    if not violations:
        print("No violations found!")
        return

    print(f"Found {len(violations)} violations to fix")

    print("Fixing all violations...")
    fix_all_violations(violations)

    print("Checking final result...")
    final_violations = get_all_violations()
    print(f"Remaining violations: {len(final_violations)}")

    if final_violations:
        print("Some violations remain - this might be due to interface constraints")
        print("First 5 remaining violations:")
        for i, (file_path, line_num, arg_name) in enumerate(final_violations[:5]):
            print(f"  {i + 1}. {file_path}:{line_num} - {arg_name}")


if __name__ == "__main__":
    main()
