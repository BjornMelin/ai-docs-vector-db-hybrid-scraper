#!/usr/bin/env python3
"""Script to fix unused function arguments across the codebase."""

import re
import subprocess
import sys
from pathlib import Path


def get_unused_arg_violations():
    """Get all unused argument violations from ruff."""
    result = subprocess.run(
        ["uv", "run", "ruff", "check", ".", "--select=ARG001,ARG002"],
        capture_output=True,
        text=True,
    )
    # Ruff outputs violations to stderr
    return result.stderr


def parse_violations(violations_text):
    """Parse ruff violations into structured data."""
    violations = []
    lines = violations_text.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i]
        if ".py:" in line and ("ARG001" in line or "ARG002" in line):
            # Parse the violation line
            match = re.match(r"([^:]+):(\d+):(\d+): (ARG\d+) (.+)", line)
            if match:
                file_path, line_num, col_num, violation_type, description = (
                    match.groups()
                )

                # Look for the argument name in the next lines
                arg_name = None
                for j in range(i + 1, min(i + 5, len(lines))):
                    if "|" in lines[j] and "^" in lines[j]:
                        # Extract argument name from the caret line
                        caret_line = lines[j]
                        if "^" in caret_line:
                            # Find the argument name pattern
                            arg_match = re.search(r"\^+\s*(\w+)", caret_line)
                            if arg_match:
                                arg_name = arg_match.group(1)
                        break

                if arg_name:
                    violations.append(
                        {
                            "file": file_path,
                            "line": int(line_num),
                            "col": int(col_num),
                            "type": violation_type,
                            "arg_name": arg_name,
                            "description": description,
                        }
                    )
        i += 1

    return violations


def fix_file_violations(file_path, violations):
    """Fix violations in a single file."""
    file_path = Path(file_path)
    if not file_path.exists():
        return False

    content = file_path.read_text()
    lines = content.split("\n")

    # Sort violations by line number in reverse order to avoid offset issues
    violations.sort(key=lambda x: x["line"], reverse=True)

    modified = False
    for violation in violations:
        line_idx = violation["line"] - 1  # Convert to 0-based indexing
        if line_idx < len(lines):
            line = lines[line_idx]
            arg_name = violation["arg_name"]

            # Skip if already prefixed with underscore
            if f"_{arg_name}" in line:
                continue

            # Replace the argument name with underscore-prefixed version
            # Handle various patterns: arg_name, arg_name:, arg_name =, arg_name,
            patterns = [
                (rf"\b{re.escape(arg_name)}\b(?=\s*:)", f"_{arg_name}"),
                (rf"\b{re.escape(arg_name)}\b(?=\s*,)", f"_{arg_name}"),
                (rf"\b{re.escape(arg_name)}\b(?=\s*=)", f"_{arg_name}"),
                (rf"\b{re.escape(arg_name)}\b(?=\s*\))", f"_{arg_name}"),
            ]

            for pattern, replacement in patterns:
                if re.search(pattern, line):
                    lines[line_idx] = re.sub(pattern, replacement, line)
                    modified = True
                    break

    if modified:
        file_path.write_text("\n".join(lines))
        return True
    return False


def main():
    """Main function to fix all unused argument violations."""
    print("Getting unused argument violations...")
    violations_text = get_unused_arg_violations()

    if not violations_text:
        print("No violations found!")
        return 0

    print("Parsing violations...")
    violations = parse_violations(violations_text)

    if not violations:
        print("No parseable violations found!")
        return 0

    print(f"Found {len(violations)} violations")

    # Group violations by file
    files_violations = {}
    for violation in violations:
        file_path = violation["file"]
        if file_path not in files_violations:
            files_violations[file_path] = []
        files_violations[file_path].append(violation)

    # Fix each file
    fixed_files = 0
    for file_path, file_violations in files_violations.items():
        print(f"Fixing {file_path} ({len(file_violations)} violations)")
        if fix_file_violations(file_path, file_violations):
            fixed_files += 1

    print(f"Fixed {fixed_files} files")

    # Check remaining violations
    print("Checking remaining violations...")
    remaining_violations = get_unused_arg_violations()
    remaining_count = len(parse_violations(remaining_violations))
    print(f"Remaining violations: {remaining_count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
